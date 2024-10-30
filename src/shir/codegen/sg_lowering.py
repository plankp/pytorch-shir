from shir import types, layout, config, bit_utils
import torch
from typing import Tuple, Optional, Dict
from functools import reduce
from itertools import chain

_supported_ops = {}

def register_lowering(key):
  def _magic(lowering):
    assert key not in _supported_ops, f"Operation {key} is repeatedly registered"
    _supported_ops[key] = lowering
    return lowering   # allows stacking this decorator
  return _magic

def fetch_lowering(key):
  return _supported_ops.get(key)

shin = torch.ops.shir_intrinsic
aten = torch.ops.aten
prims = torch.ops.prims

@register_lowering(shin.requantize_channel.default)
class LowerShirRequantize:
  @staticmethod
  def supports(a, s, z) -> bool:
    return all((bit_utils.is_valid_qscale(x) for x in s))

  @staticmethod
  def lower(a, s, z) -> str:
    try:
      q, w, shamt = bit_utils.qscale_to_fixpoint(s)
      fixpoint_method = w <= 32 and shamt < 32 + w + 1
    except AssertionError:
      fixpoint_method = False

    assert fixpoint_method, "DYNAMIC METHOD IS NOT YET SUPPORTED"
    
    # XXX: assume it's a 8-bit signed value
    shape = a.meta.get("val").shape
    zps = f"sg.SolverGuidedTensor(Seq.fill({shape[1]})({z}), sg.TensorType(Seq({shape[1]}), SignedIntType(8)))"
    scl = f"sg.SolverGuidedTensor(Seq({', '.join((str(d) for d in q))}), sg.TensorType(Seq({shape[1]}), SignedIntType({w + 1})))"
    return f"sg.SolverGuidedRequantDim({len(shape)})({a.name}, {scl}, {zps}, {shamt}, 1)"

@register_lowering(shin.qconv.default)
class LowerQConv:
  @staticmethod
  def supports(a, zp, weight, bias, stride, padding, dilation, groups) -> bool:
    # sanity check: the underlying aten.convolution supports them,
    # but the normal nn.ConvNd doesn't seem to generate these cases.
    # we could handle them, but we disable for now.
    N = len(a.meta.get("val").shape) - 2
    if N != len(stride) or N != len(padding) or N != len(dilation):
      return False

    # groups implementation is broken due to unfortunate zipping + select.
    # disable for now.
    if groups != 1:
      return False

    return True

  @staticmethod
  def lower(a, zp, weight, bias, stride, padding, dilation, groups) -> str:
    rank = a.meta.get("val").ndim
    node = a.name

    if any(padding):
      sseq = ", ".join((f"({d}, {d})" for d in padding))
      node = f"sg.SolverGuidedPad({rank})({node}, Seq({sseq}), {zp})"

    # XXX: assume it's a 8-bit signed value (9 bit signed value because 128)
    if zp != 0:
      zpvf = f"Seq(None, Some(ConstantValue({-zp}, Some(SignedIntType(9)))))"
      node = f"sg.SolverGuidedMap({rank})(AddInt2.asFunction({zpvf}), {node})"

    stride = ", ".join((str(d) for d in stride))
    dilation = ", ".join((str(d) for d in dilation))
    node = f"sg.SolverGuidedConvolution({rank})({node}, {weight.name}, Seq({stride}), Seq({dilation}))"

    if bias is not None:
      node = f"sg.SolverGuidedMapDim({rank})(AddInt.asFunction(), {node}, {bias.name}, 1)"

    return node

@register_lowering(shin.int_max_pool2d.default)
class LowerMaxPool2D:
  @staticmethod
  def supports(a, kernel_size, stride, padding, dilation) -> bool:
    # same situation as aten.convolution
    N = 2
    if N != len(kernel_size) or N != len(stride) or N != len(padding) or N != len(dilation):
      return False

    # TODO: disable the channel-implicit variant for now
    if a.meta.get("val").ndim == 3:
      return False

    return True

  @staticmethod
  def lower(a, kernel_size, stride, padding, dilation) -> str:
    shape = a.meta.get("val").shape
    rank = len(shape)
    node = a.name

    if any(padding):
      sseq = ", ".join((f"({d}, {d})" for d in padding))
      node = f"sg.SolverGuidedPad(4)({node}, Seq({sseq}), 0)"

    kernel_size = ", ".join((str(d) for d in kernel_size))
    stride = ", ".join((str(d) for d in stride))
    dilation = ", ".join((str(d) for d in dilation))
    node = f"sg.SolverGuidedMaxPool(4)({node}, Seq({kernel_size}), Seq({stride}), Seq({dilation}))"

    return node

@register_lowering(shin.int_avg_pool2d.default)
class LowerAvgPool2D:
  @staticmethod
  def supports(a, kernel_size, stride, padding) -> bool:
    # same situation as aten.convolution
    N = 2
    if N != len(kernel_size) or N != len(stride) or N != len(padding):
      return False

    # TODO: disable the channel-implicit variant for now
    if a.meta.get("val").ndim == 3:
      return False

    return True

  @staticmethod
  def lower(a, kernel_size, stride, padding) -> str:
    shape = a.meta.get("val").shape
    rank = len(shape)
    node = a.name

    if any(padding):
      sseq = ", ".join((f"({d}, {d})" for d in padding))
      node = f"sg.SolverGuidedPad(4)({node}, Seq({sseq}), 0)"

    kernel_size = ", ".join((str(d) for d in kernel_size))
    stride = ", ".join((str(d) for d in stride))
    node = f"sg.SolverGuidedAvgPool(4)({node}, Seq({kernel_size}), Seq({stride}), Seq(1, 1))"

    return node

@register_lowering(aten.relu.default)
class LowerRelu:
  @staticmethod
  def supports(a) -> bool:
    return True

  @staticmethod
  def lower(a) -> str:
    return f"sg.SolverGuidedRelu({a.name})"

