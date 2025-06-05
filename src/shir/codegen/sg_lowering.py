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

@register_lowering(shin.host_buffer_hint.default)
class LowerShirHostBufferHint:
  @staticmethod
  def supports(a) -> bool:
    return True

  @staticmethod
  def lower(a) -> str:
    # we could just wrap it under a SolverGuidedBuffer and be done with it,
    # but add extra permutes to be consistent with the input/output behaviour.
    annot_typ = types.get_element_type(a)
    ndim = a.meta.get("val").ndim
    shape = a.meta.get("val").shape

    transpose, (h, w) = layout.pack_host_shape(shape)
    shape = [shape[x] for x in transpose]
    itr = layout.inverse_transpose(transpose)

    node = f"sg.SolverGuidedPermute({a.name}, Seq({', '.join((str(d) for d in transpose))}))"
    node = f"sg.SolverGuidedReshape(sg.SolverGuidedRebalance({node}), Seq({h}, {w}))"
    node = f"sg.SolverGuidedBuffer({node})"
    node = f"sg.SolverGuidedRebalance(sg.SolverGuidedReshape({node}, Seq({', '.join((str(d) for d in shape))})))"
    return f"sg.SolverGuidedPermute({node}, Seq({', '.join((str(d) for d in itr))}))"

@register_lowering(shin.requantize_channel.default)
class LowerShirRequantizeChannel:
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

@register_lowering(shin.requantize.default)
class LowerShirRequantize:
  @staticmethod
  def supports(a, s, z) -> bool:
    return bit_utils.is_valid_qscale(s)

  @staticmethod
  def lower(a, s, z) -> str:
    try:
      q, w, shamt = bit_utils.qscale_to_fixpoint([s])
      q = q[0]
      fixpoint_method = w <= 32 and shamt < 32 + w + 1
    except AssertionError:
      fixpoint_method = False

    assert fixpoint_method, "DYNAMIC METHOD IS NOT YET SUPPORTED"

    # XXX: assume it's a 8-bit signed value
    shape = a.meta.get("val").shape
    zps = f"sg.SolverGuidedTensor(Seq.fill({shape[1]})({z}), sg.TensorType(Seq({shape[1]}), SignedIntType(8)))"
    scl = f"sg.SolverGuidedTensor(Seq.fill({shape[1]})({q}), sg.TensorType(Seq({shape[1]}), SignedIntType({w + 1})))"
    return f"sg.SolverGuidedRequantDim({len(shape)})({a.name}, {scl}, {zps}, {shamt}, 1)"

@register_lowering(aten.view.default)
@register_lowering(aten.reshape.default)
class LowerView:
  @staticmethod
  def supports(a, shape) -> bool:
    return True

  @staticmethod
  def lower(a, shape) -> str:
    newshape = ", ".join((str(d) for d in shape))
    return f"sg.SolverGuidedReshape({a.name}, Seq({newshape}))"

@register_lowering(prims.transpose.default)
class LowerView:
  @staticmethod
  def supports(a, axes) -> bool:
    return True

  @staticmethod
  def lower(a, axes) -> str:
    newaxes = ", ".join((str(d) for d in axes))
    return f"sg.SolverGuidedPermute({a.name}, Seq({newaxes}))"

@register_lowering(shin.int_addmm.default)
class LowerIntAddmm:
  @staticmethod
  def supports(acc, lhs, rhs) -> bool:
    return True

  @staticmethod
  def lower(acc, lhs, rhs) -> str:
    return f"sg.SolverGuidedMapDim(2)(AddInt.asFunction(), sg.SolverGuidedMatmul({lhs.name}, {rhs.name}), {acc.name}, 1)"

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

