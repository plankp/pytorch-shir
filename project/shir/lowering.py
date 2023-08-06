"""
Where the lowering of each supported operator actually happens
"""

import torch
from typing import Tuple, Optional
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

@register_lowering(shin.flatten.default)
class LowerFlatten:
  @staticmethod
  def supports(a, start, end) -> bool:
    return True

  @staticmethod
  def lower(a, start, end) -> str:
    rank = len(a.meta.get("val").shape)
    if rank == 0:
      # flattening a scalar always result in a tensor
      return f"algo.Repeat({a.name}, 1)"

    # otherwise, get rid of the negative indexing and then use
    # algo.torch.TFlatten.
    if start < 0:
      start = rank + start
    if end < 0:
      end = rank + end

    return f"algo.torch.TFlatten({a.name}, {start}, {end})"

@register_lowering(aten.view.default)
class LowerView:
  @staticmethod
  def supports(a, shape) -> bool:
    return True

  @staticmethod
  def lower(a, shape) -> str:
    ashape = a.meta.get("val").shape
    if ashape:
      a = f"algo.JoinAll({a.name})"
    else:
      a = f"algo.Repeat({a.name}, 1)"

    if shape == []:
      # turn T[1, ..., 1] into T
      return f"algo.Convert({a}, {shir_type.get_element_type(a).name()})"

    def iter_shape():
      for s in shape:
        if s == -1:
          # there's only supposed to be one that is -1, which is whatever is
          # left over.
          total = reduce(lambda x, y: x * y, ashape)
          divisor = reduce(lambda x, y: x * y, shape)
          s = total // -divisor
        yield s

    w = ", ".join((str(s) for s in iter_shape()))
    return (f"algo.Join(algo.SplitAll({a}, Seq({w})))")

@register_lowering(prims.broadcast_in_dim.default)
class LowerBroadcastInDim:
  @staticmethod
  def supports(a, shape, broadcast_dims) -> bool:
    return True

  @staticmethod
  def lower(a, shape, broadcast_dims) -> str:
    shape = ", ".join((str(d) for d in shape))
    dims = ", ".join((str(d) for d in broadcast_dims))
    return f"algo.torch.TBroadcast({a.name}, Seq({shape}), Seq({dims}))"

@register_lowering(prims.convert_element_type.default)
class LowerConvEltTy:
  @staticmethod
  def supports(a, dtype) -> bool:
    a = types.get_element_type(a)
    t = types.get_scalar_type(dtype)
    match (a, t):
      case (types.UI(_) | types.SI(_), types.UI(_) | types.SI(_)):
        return True
      case _:
        return False

  @staticmethod
  def lower(a, dtype) -> str:
    atype = types.get_element_type(a)
    dtype = types.get_scalar_type(dtype)
    if atype == dtype:
      return a.name

    ss, sbits = types.unpack_int_type(atype)
    ds, dbits = types.unpack_int_type(dtype)

    if sbits >= dbits or not ss:
      # truncate or zero extend
      inner = f"algo.TruncInteger(core.ParamUse(_0), {dbits})"
      if ds:
        # convert from unsigned to signed
        inner = f"core.Conversion({inner}, {dtype.name()})"
      converter = (
        f"{{ val _0 = core.ParamDef({atype.name()}); algo.AlgoLambda(_0, {inner}) }}"
      )

    else:
      # sign extend (assert is for sanity purposes)
      assert ss and sbits < dbits
      inner = f"algo.Add2(core.ParamUse(_0), algo.ConstantInteger(0, Some(algo.SignedIntType({dbits}))))"
      if not ds:
        # convert from signed to unsigned
        inner = f"core.Conversion({inner}, {dtype.name()})"
      converter = (
        f"{{ val _0 = core.ParamDef({atype.name()}); algo.AlgoLambda(_0, {inner}) }}"
      )

    rank = len(a.meta.get("val").shape)
    return f"algo.Map({rank}, {converter}, {a.name})"

class LowerArithBinaryOperatorTemplate:
  @staticmethod
  def apply_op(ty, pair) -> str:
    assert False, "Subtype needs to provide impl"

  @classmethod
  def supports(cls, lhs, rhs) -> bool:
    return cls._get_tensor_type(lhs, rhs) is not None

  @staticmethod
  def _get_tensor_type(lhs, rhs):
    if not isinstance(lhs, torch.fx.Node):
      lhs, rhs = rhs, lhs
    if not isinstance(lhs, torch.fx.Node):
      return None

    t1 = types.get_element_type(lhs)
    if isinstance(rhs, torch.fx.Node):
      t2 = types.get_element_type(rhs)
    else:
      t2 = rhs

    if t1 == t2:
      return t1

    match t1, t2:
      case (types.UI(_) | types.SI(_), int(_)):
        return t1
      case _:
        return None

  @staticmethod
  def _normalize_repr(value, ty) -> Tuple[str, int]:
    match value:
      case int(_):
        return (f"algo.ConstantInteger({value}, Some({ty.name()}))", 0)
      case _:
        return (value.name, len(value.meta.get("val").shape))

  @classmethod
  def lower(cls, lhs, rhs) -> bool:
    ty = cls._get_tensor_type(lhs, rhs)
    tyname = ty.name()

    (lhs, lrank) = cls._normalize_repr(lhs, ty)
    (rhs, rrank) = cls._normalize_repr(rhs, ty)

    if lrank != rrank:
      rank = lrank
      tensor = lhs
      if lrank == 0:
        pair = f"algo.Tuple({lhs}, core.ParamUse(_0))"
        rank = rrank
        tensor = rhs
      else:
        pair = f"algo.Tuple(core.ParamUse(_0), {rhs})"

      kernel = cls.apply_op(ty, pair)
      return (
        f"algo.Map({rank}, {{"
        f" val _0 = core.ParamDef({tyname});"
        f" algo.AlgoLambda(_0, {kernel}) }}, {lhs})"
      )

    kernel = cls.apply_op(ty, "core.ParamUse(_0)")
    return (
      f"algo.torch.TZipAll({{"
      f" val _0 = core.ParamDef(algo.TupleType({tyname}, {tyname}));"
      f" algo.AlgoLambda(_0, {kernel}) }}, {lhs}, {rhs})"
    )

@register_lowering(prims.add.default)
class LowerAdd(LowerArithBinaryOperatorTemplate):
  @staticmethod
  def apply_op(ty, pair):
    signed, bits = types.unpack_int_type(ty)
    f = f"algo.TruncInteger(algo.Add({pair}), {bits})"
    if signed:
      f = f"core.Conversion({f}, {ty.name()})"
    return f

@register_lowering(prims.sub.default)
class LowerSub(LowerArithBinaryOperatorTemplate):
  @staticmethod
  def apply_op(ty, pair):
    signed, _ = types.unpack_int_type(ty)
    if signed:
      return f"algo.Sub({pair})"
    return f"core.Conversion(algo.Sub({pair}), {ty.name()})"

@register_lowering(prims.mul.default)
class LowerMul(LowerArithBinaryOperatorTemplate):
  @staticmethod
  def apply_op(ty, pair):
    signed, bits = types.unpack_int_type(ty)
    f = f"algo.TruncInteger(algo.Mul({pair}), {bits})"
    if signed:
      f = f"core.Conversion({f}, {ty.name()})"
    return f

@register_lowering(prims.maximum.default)
class LowerMax(LowerArithBinaryOperatorTemplate):
  @staticmethod
  def apply_op(ty, pair):
    return f"algo.Max({pair})"

@register_lowering(prims.minimum.default)
class LowerMin(LowerArithBinaryOperatorTemplate):
  @staticmethod
  def apply_op(ty, pair):
    return f"algo.Min({pair})"

@register_lowering(shin.qadd.default)
class LowerQadd:
  @staticmethod
  def supports(a, sa, b, sb, z) -> bool:
    # just like per channel fixed point requant, we can adjust both scales to
    # the same decimal point, perform the multiplication, add and round, and
    # do zero point adjustment.
    #
    # in theory, we can support any scale. in practice, SHIR does not play
    # nicely with values beyond 32 bits. (float point hack doesn't work here)
    try:
      # both multiplications give 32 + w + 1 bits,
      # addition gives an extra bit, so 32 + w + 1 + 1 bits.
      _, w, shamt = bit_utils.qscale_to_fixpoint([sa, sb])
      if w > 32 or shamt >= 32 + w + 2:
        return False
    except AssertionError:
      return False

    return True

  @staticmethod
  def lower(a, sa, b, sb, z) -> str:
    # round(a * sa + b * sb) + z
    #   = round(2^-k (a * ia + b * ib)) + z
    #   = round(a * ia + b * ib, k) + z

    [ia, ib], w, shamt = bit_utils.qscale_to_fixpoint([sa, sb])
    ia = f"algo.ConstantInteger({ia}, Some(algo.IntType({w})))"
    ib = f"algo.ConstantInteger({ib}, Some(algo.IntType({w})))"

    # shape of a and b are the same
    rank = len(a.meta.get("val").shape)
    if rank == 0:
      return (
        f"algo.torch.SIRequantAdd32({a.name}, {b.name},"
        f" {w}, {ia}, {ib}, {shamt}, {z})"
      )

    return (
      f"algo.torch.TZipAll(algo.torch.SIRequantAdd32.asFunction("
      f"{w}, {ia}, {ib}, {shamt}, {z}), {a.name}, {b.name})"
    )

@register_lowering(aten.relu.default)
class LowerRelu:
  @staticmethod
  def supports(a) -> bool:
    return LowerClamp.supports(a, 0, None)

  @staticmethod
  def lower(a) -> str:
    return LowerClamp.lower(a, 0, None)

@register_lowering(aten.clamp.default)
class LowerClamp:
  @staticmethod
  def supports(a, clmin=None, clmax=None) -> bool:
    # since SHIR uses evalInt, disallow ranges that exceed the s32 range.
    s32 = types.SI(32)
    ty = types.get_element_type(a)
    tmin = max(s32.minval(), ty.minval())
    tmax = max(s32.maxval(), ty.maxval())
    return (
      (clmin is None or tmin <= clmin <= tmax) and
      (clmax is None or tmin <= clmax <= tmax)
    )

  @staticmethod
  def lower(a, clmin=None, clmax=None) -> str:
    ty = types.get_element_type(a)
    rank = len(a.meta.get("val").shape)
    vmin = "None" if clmin is None else f"Some({clmin})"
    vmax = "None" if clmax is None else f"Some({clmax})"
    return (
      f"algo.Map({rank}, algo.torch.IClamp.asFunction("
      f"{ty.name()}, {vmin}, {vmax}), {a.name})"
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

@register_lowering(shin.int_max_pool2d.default)
class LowerMaxPool2D:
  @staticmethod
  def supports(input, kernel_size, stride, padding, dilation) -> bool:
    # same situation as aten.convolution
    N = 2
    if N != len(kernel_size) or N != len(stride) or N != len(padding) or N != len(dilation):
      return False

    return True

  @staticmethod
  def lower(input, kernel_size, stride, padding, dilation) -> str:
    # input is either T[N, i1, i2] or T[N, C, i1, i2]
    rank = len(input.meta.get("val").shape)

    has_channel = "true" if rank != 3 else "false"
    kernel_size = ", ".join((str(d) for d in kernel_size))
    stride = ", ".join((str(d) for d in stride))
    padding = ", ".join((str(d) for d in padding))
    dilation = ", ".join((str(d) for d in dilation))
    return (
      f"algo.torch.TPool.imax({types.get_element_type(input).name()},"
      f" {has_channel}, {input.name}, Seq({kernel_size}),"
      f" Seq({stride}), Seq({padding}), Seq({dilation}))"
    )

@register_lowering(shin.int_mean.default)
class LowerMean:
  @staticmethod
  def supports(a, dims, keepDim) -> bool:
    return True

  @staticmethod
  def lower(a, dims, keepDim) -> str:
    rank = len(a.meta.get("val").shape)

    # normalize the negative reduction dimensions
    # (which PyTorch allows but we don't)
    dims = ", ".join((str(d if d >= 0 else rank + d) for d in dims))
    keepDim = "true" if keepDim else "false"
    return (
      f"algo.torch.TReduce.iavg({types.get_element_type(a).name()},"
      f" {a.name}, {dims}, {keepDim})"
    )

@register_lowering(shin.int_adaptive_avg_pool2d.default)
class LowerAdaptiveAvgPool2D:
  @classmethod
  def supports(cls, input, output_size) -> bool:
    # the "adaptive" aspect of adaptive pooling operators is difficult to
    # implement correctly.
    #
    # fortunately, provided the input is preprocessed correctly, torchvision
    # models tend to not trigger the difficult cases.
    #
    # that means, we only support the case where we can convert it into an
    # equivalent non-adaptive pooling operator.

    N = 2
    if N != len(output_size):
      return False

    ashape = input.meta.get("val").shape
    if cls._find_equivalent_slide(ashape, output_size) is None:
      return False

    return True

  @classmethod
  def lower(cls, input, output_size) -> str:
    # input is either T[N, i1, i2] or T[N, C, i1, i2]
    ashape = input.meta.get("val").shape
    rank = len(ashape)

    rslide_info = cls._find_equivalent_slide(ashape, output_size)
    has_channel = "true" if rank != 3 else "false"
    kernel_size = ", ".join((str(d[0]) for d in reversed(rslide_info)))
    stride = ", ".join((str(d[1]) for d in reversed(rslide_info)))
    return (
      f"algo.torch.TPool.iavg({types.get_element_type(input).name()},"
      f" {has_channel}, {input.name}, Seq({kernel_size}),"
      f" Seq({stride}), Seq(0, 0), Seq(1, 1))"
    )

  @staticmethod
  def _find_equivalent_slide(shape, output_size):
    # use reversed zip to avoid shape's channel dimension being problematic
    rev = []
    for in_size, out_size in zip(reversed(shape), reversed(output_size)):
      # start uses floor division
      # end uses ceiling division
      assert in_size > 0 and out_size > 0
      last_start = 0
      last_end = -(in_size // -out_size)
      window_size = last_end - last_start
      stride = None

      for i in range(1, out_size):
        start = i * in_size // out_size
        end = -((i + 1) * in_size // -out_size)
        if end - start != window_size:
          return None

        next_stride = start - last_start
        if stride is None:
          stride = next_stride
        elif stride != next_stride:
          return None

        last_start, last_end = start, end
      rev.append((window_size, stride or 1))
    return rev
