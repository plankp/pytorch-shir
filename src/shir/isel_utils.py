#
# utility functions for instruction selection
#

import torch
import torch.nn as nn
import torch.fx as fx
from . import bit_utils

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

def mk_requant_param(scales, zp, qw=28, rshamt=35):
  assert qw > 0 and rshamt > 0, "isel_utils::mk_requant_param: qw and rshamt must be strictly positive"

  # XXX: assumes signed 8 bit requantization
  if zp < -128 or zp > 127:
    return None

  q, w, shamt = bit_utils.qscale_to_fixpoint(scales)

  if shamt > rshamt:
    print(f"isel_utils::mk_requant_param: shamt of {shamt} is larger, trying to continue by truncating")
    q = [x >> (shamt - rshamt) for x in q]
    w = max(1, w - (shamt - rshamt))
    shamt = rshamt

  lsl = rshamt - shamt
  if w + lsl > qw:
    return None

  return [x << lsl for x in q]

