"""
Where the lowering of each supported operator actually happens
"""

import torch
from functools import reduce
from itertools import chain
from . import bit_utils, types

"""
Registration magic
"""

_supported_ops = {}

def register_lowering(key):
  def _magic(lowering):
    assert key not in _supported_ops, f"Operation {key} is repeatedly registered"
    _supported_ops[key] = lowering
    return lowering   # allows stacking this decorator
  return _magic

def fetch_lowering(key):
  return _supported_ops.get(key)

"""
Registration of the supported operators
"""

shin = torch.ops.shir_intrinsic
aten = torch.ops.aten
prims = torch.ops.prims

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

    if fixpoint_method:
      requant_kernel = (
        f"algo.torch.SIRequantFixed32.asFunction("
        f"{w}, algo.ConstantInteger({q}, Some(algo.IntType({w}))), {shamt}, {z})"
      )
    else:
      fbits32 = bit_utils.to_signed(bit_utils.f32_to_bits(s), 32)
      requant_kernel = (
        f"algo.torch.SIRequantFloat32.asFunction("
        f"algo.ConstantInteger({fbits32}, Some(algo.IntType(32))), {z})"
      )

    rank = len(a.meta.get("val").shape)
    return f"algo.Map({rank}, {requant_kernel}, {a.name})"

@register_lowering(shin.requantize_channel.default)
class LowerShirRequantizeChannel:
  @staticmethod
  def supports(a, s, z) -> bool:
    return all((bit_utils.is_valid_qscale(x) for x in s))

  @staticmethod
  def lower(a, s, z) -> bool:
    try:
      q, w, shamt = bit_utils.qscale_to_fixpoint(s)
      fixpoint_method = w <= 32 and shamt < 32 + w + 1
    except AssertionError:
      fixpoint_method = False

    fixpoint_method = False
    # synthesize the requant function and the stream of scales
    #
    # the main takeaway is that ConstantSeq has last element first,
    # hence the bare minimum is to reverse the stream.
    if fixpoint_method:
      sseq = reversed(q)
      requant_kernel = f"algo.torch.SIRequantFixed32.asFunction({w}, {shamt}, {z})"

    else:
      w = 32
      sseq = (bit_utils.f32_to_bits(x) for x in reversed(s))
      requant_kernel = f"algo.torch.SIRequantFloat32.asFunction({z})"

    sseq = ", ".join((str(bit_utils.to_signed(x, 32)) for x in sseq))
    return (
      f"algo.torch.TZipChannel({requant_kernel}, {a.name},"
      f" algo.ConstantSeq(Seq({sseq}), Some(algo.IntType({w}))))"
    )

@register_lowering(shin.int_addmm.default)
class LowerShirIntAddmm:
  @staticmethod
  def supports(acc, lhs, rhs) -> bool:
    return True

  @staticmethod
  def lower(acc, lhs, rhs) -> str:
    return f"algo.torch.SIAdd32MM8({acc.name}, {lhs.name}, {rhs.name})"

@register_lowering(aten.convolution.default)
class LowerConvolution:
  @staticmethod
  def supports(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups) -> bool:
    if transposed:
      return False
    if any((p != 0 for p in output_padding)):
      return False

    # disallow bias here. expect a rewrite to move it into a separate add.
    if bias is not None:
      return False

    # some cases that are technically allowed, but we weren't able to
    # reproduce it.
    N = len(input.meta.get("val").shape) - 2
    if N != len(stride) or N != len(padding) or N != len(dilation):
      return False

    return True

  @staticmethod
  def lower(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups) -> str:
    stride = ", ".join((str(d) for d in stride))
    padding = ", ".join((str(d) for d in padding))
    dilation = ", ".join((str(d) for d in dilation))
    return (
      f"algo.torch.TConvolution({types.get_element_type(input).name()},"
      f" {input.name}, {weight.name},"
      f" Seq({stride}), Seq({padding}), Seq({dilation}),"
      f" {groups})"
    )
