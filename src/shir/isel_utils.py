#
# utility functions for instruction selection
#

import torch
import torch.nn as nn
import torch.fx as fx
from . import bit_utils

_RELU_OPS = {
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.leaky_relu.default,
}

def extract_qaddmm_relu(n: fx.Node):
  requant = None
  relu = None

  if (n.op != "call_function" or n.target != torch.ops.shir_intrinsic.requantize or
      len(n.args[0].users) != 1):
    return None

  requant, n = n, n.args[0]
  if (n.op == "call_function" and n.target == torch.ops.aten.relu and
      len(n.args[0].users) == 1):
    relu, n = n, n.args[0]

  if n.op != "call_function" or n.target != torch.ops.shir_intrinsic.int_addmm:
    return None

  return (requant, relu, n)

def extract_qconv_relu(n: fx.Node):
  requant = None
  relu = None
  if (n.op != "call_function" or n.target != torch.ops.shir_intrinsic.requantize_channel or
      len(n.args[0].users) != 1):
    return None

  requant, n = n, n.args[0]
  if (n.op == "call_function" and n.target == torch.ops.aten.relu and
      len(n.args[0].users) == 1):
    relu, n = n, n.args[0]

  if n.op != "call_function" or n.target != torch.ops.shir_intrinsic.qconv:
    return None

  return (requant, relu, n)

def extract_qconv_leaky(n: fx.Node, rshamt: int):
  requant = None
  leaky   = None
  if (n.op != "call_function" or n.target != torch.ops.shir_intrinsic.requantize_channel or
      len(n.args[0].users) != 1):
    return None

  requant, n = n, n.args[0]
  if (n.op == "call_function" and n.target == torch.ops.shir_intrinsic.sra_leaky_relu and
      len(n.args) == 2 and n.args[1] == rshamt and len(n.args[0].users) == 1):
    leaky, n = n, n.args[0]

  if n.op != "call_function" or n.target != torch.ops.shir_intrinsic.qconv:
    return None

  return (requant, leaky, n)

def try_requant_param(scales, zp, qconfigs):
  assert all((qw > 0 and rshamt > 0 for qw, rshamt in qconfigs)), "isel_utils::try_requant_param: qw and rshamt must be restrictly positive"

  # XXX: assumes signed 8 bit quantization
  if zp < -128 or zp > 127:
    return None

  q, w, shamt = bit_utils.qscale_to_fixpoint(scales)

  candidate = None
  for qw, rshamt in qconfigs:
    lsl = rshamt - shamt
    if w + lsl <= qw:
      if lsl >= 0:
        # this one can be encoded without truncating / rounding.
        # assume this is the best choice.
        return (qw, rshamt, [x << lsl for x in q])

      if candidate is None or candidate[2] < lsl:
        # this one can be encoded with less loss than the previous candidate
        candidate = qw, rshamt, lsl

  if candidate is None:
    return None

  return (candidate[0], candidate[1], [x >> -candidate[2] for x in q])

def mk_requant_param(scales, zp, qw=28, rshamt=35):
  if r := try_requant_param(scales, zp, [(qw, rshamt)]):
    return r[2]
  return None

def extract_attr(g: fx.GraphModule, n: fx.Node):
  if n is None:
    return None
  if n.op != "get_attr" or n.args != () or n.kwargs != {}:
    return None
  base = g
  for attr in n.target.split('.'):
    assert hasattr(base, attr), f"Invalid attribute {n.target}"
    base = getattr(base, attr)

  return base

def match_quant_per_tensor(g: fx.GraphModule, n: fx.Node, qmin, qmax, ty):
  if n.op != "call_function":
    return None
  if n.target != torch.ops.quantized_decomposed.quantize_per_tensor.default:
    return None
  if n.args[3] != qmin or n.args[4] != qmax or n.args[5] != ty:
    return None
  return (n.args[1], n.args[2])

def match_dequant_per_tensor(g: fx.GraphModule, n: fx.Node, qmin, qmax, ty):
  if n.op != "call_function":
    return None
  if n.target != torch.ops.quantized_decomposed.dequantize_per_tensor.default:
    return None
  if n.args[3] != qmin or n.args[4] != qmax or n.args[5] != ty:
    return None
  return (n.args[1], n.args[2])

def match_quant_per_channel(g: fx.GraphModule, n: fx.Node, chan, qmin, qmax, ty):
  if n.op != "call_function":
    return None
  if n.target != torch.ops.quantized_decomposed.quantize_per_channel.default:
    return None
  if n.args[3] != chan or n.args[4] != qmin or n.args[5] != qmax or n.args[6] != ty:
    return None
  return (extract_attr(g, n.args[1]), extract_attr(g, n.args[2]))

def match_dequant_per_channel(g: fx.GraphModule, n: fx.Node, chan, qmin, qmax, ty):
  if n.op != "call_function":
    return None
  if n.target != torch.ops.quantized_decomposed.dequantize_per_channel.default:
    return None
  if n.args[3] != chan or n.args[4] != qmin or n.args[5] != qmax or n.args[6] != ty:
    return None
  return (extract_attr(g, n.args[1]), extract_attr(g, n.args[2]))

def match_qlinear(g: fx.GraphModule, n: fx.Node):
  if not match_quant_per_tensor(g, n, -128, 127, torch.int8):
    return None

  n_relu = n.args[0]
  if n_relu.op == "call_function" and n_relu.target in _RELU_OPS:
    n_linear = n_relu.args[0]
  else:
    n_relu, n_linear = None, n_relu

  if n_linear.op != "call_function" or n_linear.target != torch.ops.aten.linear.default:
    return None

  n_dq_img = n_linear.args[0]
  n_dq_wgt = n_linear.args[1]

  if not match_dequant_per_tensor(g, n_dq_img, -128, 127, torch.int8):
    return None
  if not match_dequant_per_tensor(g, n_dq_wgt, -127, 127, torch.int8):
    return None

  return (
      n_dq_img,
      n_dq_wgt,
      n_linear,
      n_relu,
  )

def match_qconv2d(g: fx.GraphModule, n: fx.Node):
  # XXX: the input and output are always quantized per tensor
  if not match_quant_per_tensor(g, n, -128, 127, torch.int8):
    return None

  n_relu = n.args[0]
  if n_relu.op == "call_function" and n_relu.target in _RELU_OPS:
    n_conv = n_relu.args[0]
  else:
    n_relu, n_conv = None, n_relu

  if n_conv.op != "call_function" or n_conv.target != torch.ops.aten.conv2d.default:
    return None

  n_dq_img = n_conv.args[0]
  n_dq_wgt = n_conv.args[1]

  if not match_dequant_per_tensor(g, n_dq_img, -128, 127, torch.int8):
    return None
  if match_dequant_per_tensor(g, n_dq_wgt, -127, 127, torch.int8):
    per_tensor = True
  elif match_dequant_per_channel(g, n_dq_wgt, 0, -127, 127, torch.int8):
    per_tensor = False
  else:
    return None

  return (
      n_dq_img,
      n_dq_wgt,
      n_conv,
      n_relu,
  )

def match_qconv2d_add(g: fx.GraphModule, n: fx.Node):
  # XXX: the input and output are always quantized per tensor
  if not match_quant_per_tensor(g, n, -128, 127, torch.int8):
    return None

  n_relu = n.args[0]
  if n_relu.op == "call_function" and n_relu.target in _RELU_OPS:
    n_add = n_relu.args[0]
  else:
    n_relu, n_add = None, n_relu

  if n_add.op != "call_function" or n_add.target != torch.ops.aten.add.Tensor:
    return None

  # one of the operands to the addition must be a conv2d
  n_conv, n_residual = n_add.args
  if n_conv.op != "call_function" or n_conv.target != torch.ops.aten.conv2d.default:
    n_conv, n_residual = n_residual, n_conv
  if n_conv.op != "call_function" or n_conv.target != torch.ops.aten.conv2d.default:
    return None

  n_dq_img = n_conv.args[0]
  n_dq_wgt = n_conv.args[1]

  if not match_dequant_per_tensor(g, n_dq_img, -128, 127, torch.int8):
    return None
  if match_dequant_per_tensor(g, n_dq_wgt, -127, 127, torch.int8):
    per_tensor = True
  elif match_dequant_per_channel(g, n_dq_wgt, 0, -127, 127, torch.int8):
    per_tensor = False
  else:
    return None

  return (
      n_dq_img,
      n_dq_wgt,
      n_conv,
      n_residual,
      n_add,
      n_relu,
  )

