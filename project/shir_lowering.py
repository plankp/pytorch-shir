import torch
import struct
import shir_type
import shir_intrinsic
from functools import reduce
from itertools import chain

# Some configuration flags used to affect the lowering.
# TODO: see if we can pass this from torch.compile
FLAG_REQUANT_PREFER_FIXPOINT_SCALE = True

_ops = {}

"""
signature SHIRLowering:
  # static methods
  def supports(*args) -> bool
  def lower(*args) -> str
"""

# some namespace aliases
shir = torch.ops.shir_intrinsic
aten = torch.ops.aten
prims = torch.ops.prims

def register_operator(key):
  def _magic(lowering):
    assert key not in _ops, f"Operation {key} is repeatedly registered"
    _ops[key] = lowering
    return lowering   # allows stacking this decorator
  return _magic

def fetch_lowering(target):
  return _ops.get(target)

"""
some generic template for lowering
"""

def lower_pairwise_binop(op: str, xname, xshape, yname, yshape) -> str:
  """
  op has type (T, U) -> V,
  the result will operate on x: T[* :> S] and y: U[* :> S] to give V[S]
  """

  if xshape == yshape:
    if not xshape:
      # both are scalars, call the operator directly
      return f"{op}.call(algo.Tuple({xname}, {yname}))"

    # there's at least one axis
    acc = lambda t: f"algo.Map({op}, algo.Zip({t}))"
    for _ in range(1, len(xshape)):
      w = acc("core.ParamUse(_0)")
      acc = lambda t: f"algo.Map({{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0), {w}) }}, algo.Zip({t}))"

    return acc(f"algo.Tuple({xname}, {yname})")

  # exactly one of x or y must be scalar
  if xshape:
    acc = lambda t: f"{op}.call(algo.Tuple({t}, {yname}))"
    val = xname
    dim = xshape
  else:
    acc = lambda t: f"{op}.call(algo.Tuple({xname}, {t}))"
    val = yname
    dim = yshape

  for _ in dim:
    w = acc("core.ParamUse(_0)")
    acc = lambda t: f"algo.Map({{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0), {w}) }}, {t})"
  return acc(val)

def lower_pairwise_tensor_binop(op: str, x: torch.fx.Node, y: torch.fx.Node) -> str:
  """
  special case when x and y are torch fx Nodes where we can fetch the name and
  shape directly from it.
  """

  return lower_pairwise_binop(op, x.name, x.meta.get("val").shape,
                              y.name, y.meta.get("val").shape)

def lower_reduction(x: torch.fx.Node, dims: list[int], reducer) -> str:
  assert dims != []

  elt_ty = shir_type.get_element_type(x)
  xshape = x.meta.get("val").shape
  x = x.name
  N = len(xshape)
  join_layers = len(dims)
  unpeel_layers = N - join_layers

  if unpeel_layers > 0:
    prefix = [i for i in range(N) if i not in dims]
    if all((i == j for (i, j) in zip(prefix, range(N)))):
      # we don't need to transposed anything since we're already going to
      # reduce the inner parts.
      #
      # like the latter case, we compute the type of the innermost map,
      # but we cannot rely on dims since it might be out of order.
      shape = reduce(lambda x, y: f"algo.SeqType({x}, {y})",
                     reversed(xshape[unpeel_layers:]),
                     elt_ty.name())
    else:
      # we have a gap in the reduction axes: need to transpose
      # using tuple splat here is also safe because the axes will have at
      # least two elements
      axes = [*prefix, *dims]
      x = f"algo.TransposeND({x}, Seq{(*(N - i - 1 for i in reversed(axes)),)})"
      shape = reduce(lambda x, y: f"algo.SeqType({x}, {y})",
                     (xshape[d] for d in reversed(dims)),
                     elt_ty.name())

  # for sanity reasons, we avoid using algo.JoinAll here
  hd = (join_layers - 1) * "algo.Join("
  tl = (join_layers - 1) * ")"
  # compute the flattened size, useful for computing the mean
  inner_size = reduce(lambda x, y: x * y, (xshape[i] for i in dims))
  acc = lambda t: reducer(f"{hd}{t}{tl}", inner_size)

  if unpeel_layers > 0:
    w = acc("core.ParamUse(_0)")
    acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef({shape});"
                     f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")

  for _ in range(unpeel_layers - 1):
    w = acc("core.ParamUse(_0)")
    acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                     f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")

  return acc(x)

def to_signed(v: int, bits: int):
  v &= (1 << bits) - 1
  if v & (1 << (bits - 1)):
    return -((1 << bits) - v)
  return v

def to_float_bits(x: float) -> int:
  return struct.unpack(">l", struct.pack(">f", x))[0]

def is_valid_qscale(x: float) -> bool:
  import math
  return x > 0 and math.isfinite(x)

def unpack_qscale(x: float) -> (int, int):
  bits = to_float_bits(x)
  frac = (1 << 23) | (bits & ((1 << 23) - 1))
  shamt = 127 + 23 - (bits >> 23)   # 23 comes from a (normal) mantissa
  return frac, shamt

def qscale_to_fixpoint(x: list[float]) -> (list[int], int, int):
  # one restriction we impose is for the number of fractional bits
  # (in other words, the rounding shift amount) to be non-negative.

  def gen_shortest_qvalue(x):
    for f in x:
      assert is_valid_qscale(f), "Invalid qscale"
      frac, shamt = unpack_qscale(f)
      assert shamt >= 0, "Invalid shift amount"

      # normalize the representation by aggressively shifting right
      # while making sure the shift amount is still valid.
      while shamt >= 0 and (frac & 1) == 0:
        frac >>= 1
        shamt -= 1
      yield frac, shamt

  qvalues = list(gen_shortest_qvalue(x))
  final_scale = max((x[1] for x in qvalues))

  # with values being as "short" as possible, we now try to expand values that
  # are too short by doing the reverse: shifting left.
  N = len(qvalues)
  max_width = 0
  for i in range(N):
    frac, shamt = qvalues[i]
    # generally speaking, this shift only works because Python uses bigints.
    frac <<= final_scale - shamt
    qvalues[i] = frac
    max_width = max(max_width, frac.bit_length())
  return qvalues, max_width, final_scale

"""
magic that actually does the lowering of each node
"""

@register_operator(shir.requantize.default)
class LowerRequantize:
  @classmethod
  def supports(cls, a, s, z) -> bool:
    if shir_type.get_element_type(a) != shir_type.SI(32):
      return False

    # we can always fallback to the float method,
    # provided that the qscale is "valid"
    return is_valid_qscale(s)

  @classmethod
  def lower(cls, a, s, z) -> str:
    # note that unlike per channel requant, we disallow u32 fractional bits.
    # this is because unlike the per channel case, there is ever only one
    # scale, meaning that w should really be at most 24 (the full mantissa).
    fixpoint_method = FLAG_REQUANT_PREFER_FIXPOINT_SCALE
    if fixpoint_method:
      try:
        q, w, shamt = qscale_to_fixpoint([s])
        q = q[0]
        fixpoint_method = w < 32 and shamt < 32 + w + 1
      except AssertionError:
        fixpoint_method = False

    if fixpoint_method:
      def gen_kernel(t):
        qbits = w + 1
        width = 32 + qbits
        acc = (
          f"algo.Mul(algo.Tuple({t},"
          f" algo.ConstantInteger({q}, Some(algo.SignedIntType({qbits})))))"
        )

        if shamt:
          width -= shamt
          acc = (f"algo.Sub(algo.Tuple(algo.TruncInteger("
                 f"algo.ClipBankersRound({acc}, {shamt}, 0), {width}),"
                 f" algo.ConstantInteger(0)))")

        if z != 0:
          width = max(width, 32) if width != 32 else 33
          acc = f"algo.Add2({acc}, algo.ConstantInteger({z}, Some(algo.SignedIntType(32))))"

        return f"algo.ClipBankersRound({acc}, 0, {width - 8})"
    else:
      def gen_kernel(t):
        return (
          f"algo.ClipBankersRound(algo.Add2("
          f"algo.Sub(algo.Tuple(algo.TruncInteger("
          f"algo.QScale(algo.Tuple({t},"
          f" algo.ConstantInteger({to_signed(to_float_bits(s), 32)},"
          f" Some(algo.IntType(32))))), 57), algo.ConstantInteger(0))),"
          f" algo.ConstantInteger({z}, Some(algo.SignedIntType(32)))),"
          f" 0, 49)"
        )

    ashape = a.meta.get("val").shape
    if not ashape:
      return gen_kernel(a.name)

    w = gen_kernel("core.ParamUse(_0)")
    acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef(algo.SignedIntType(32));"
                     f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")
    for _ in ashape[1:]:
      w = acc("core.ParamUse(_0)")
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")

    return acc(a.name)

@register_operator(shir.requantize_channel.default)
class LowerRequantizeChannel:
  @staticmethod
  def supports(a, s, z) -> bool:
    if shir_type.get_element_type(a) != shir_type.SI(32):
      return False

    # in the end of the day,
    # provided that the scales are normal,
    # and they are always positive by nature,
    #
    # the float point method always works.
    # but the fixpoint method may be better for performance.
    return all((is_valid_qscale(x) for x in s))

  @staticmethod
  def lower(a, s, z) -> str:
    # by requantize-channel's restrictions, a must have dimensions N and C.
    # in essense, we want to do this:
    #
    #   a |> Map (\c ->
    #     Zip (c, s) |> Map (\(d, s) ->
    #       d |> MapND (\d -> Requantization d s z)))
    #
    # Of course, we generally need to JoinAll c before zipping and later
    # SplitAll the result.
    ashape = a.meta.get("val").shape

    # decide if we can actually use the fixpoint method:
    # aside from assertions, we add the following restrictions:
    # *  converted values cannot exceed 32 bits (due to SHIR's evalInt)
    # *  the saturating shift cannot shift everything out
    fixpoint_method = FLAG_REQUANT_PREFER_FIXPOINT_SCALE
    if fixpoint_method:
      try:
        q, w, shamt = qscale_to_fixpoint(s)
        fixpoint_method = w <= 32 and shamt < 32 + w + 1
      except AssertionError:
        fixpoint_method = False

    # both methods require emitting the scales (as some format) as a constant
    # sequence.
    #
    # ConstantSeq is reversed!
    if fixpoint_method:
      # the loaded values are unsigned, but multiplication needs both to be
      # signed, so we use algo.Signed to perform the conversion.
      sseq = ", ".join((str(to_signed(x, 32)) for x in reversed(q)))
      smap = "algo.Signed(core.ParamUse(_0))"

      qbits = w + 1
      width = 32 + qbits
      acc = f"algo.Mul(core.ParamUse(_0))"

      if shamt:
        width -= shamt
        acc = (f"algo.Sub(algo.Tuple(algo.TruncInteger("
               f"algo.ClipBankersRound({acc}, {shamt}, 0), {width}),"
               f" algo.ConstantInteger(0)))")
      if z != 0:
        # adding the zero point
        width = max(width, 32) if width != 32 else 33
        acc = f"algo.Add2({acc}, algo.ConstantInteger({z}, Some(algo.SignedIntType(32))))"

      # clip it to 8 bits
      emit_rescale_op = f"algo.ClipBankersRound({acc}, 0, {width - 8})"
    else:
      w = 32
      sseq = ", ".join((str(to_signed(to_float_bits(x), 32)) for x in reversed(s)))
      smap = "core.ParamUse(_0)"

      emit_rescale_op = (
        f"algo.ClipBankersRound(algo.Add2("
        f"algo.Sub(algo.Tuple(algo.TruncInteger("
        f"algo.QScale(core.ParamUse(_0)), 57),"
        f" algo.ConstantInteger(0))),"
        f" algo.ConstantInteger({z}, Some(algo.SignedIntType(32)))),"
        f" 0, 49)"
      )

    # in addition to whatever processing is needed, we also repeat it as
    # needed (avoids issues when zipping with the input later)
    if len(ashape) > 2:
      width = reduce(lambda x, y: x * y, ashape[2:])
      smap = f"algo.Repeat({smap}, {width})"
    sseq = (f"algo.Map({{ val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), {smap}) }},"
            f" algo.ConstantSeq(Seq({sseq}), Some(IntType({w}))))")

    if len(ashape) == 2:
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {emit_rescale_op}) }},"
                       f" algo.Zip(algo.Tuple({t}, {sseq})))")
    else:
      ty = reduce(lambda x, y: f"algo.SeqType({x}, {y})",
                  reversed(ashape[2:]), "algo.SignedIntType(32)")
      kernel = (f"algo.Map({{ val _0 = core.ParamDef();"
                f" algo.AlgoLambda(Seq(_0), {emit_rescale_op}) }},"
                f" algo.Zip(core.ParamUse(_0)))")
      for i in reversed(ashape[3:]):
        kernel = f"algo.Split({kernel}, {i})"
      joinseq = "core.ParamUse(_0)"
      if len(ashape) > 3:
        joinseq = "algo.JoinAll(core.ParamUse(_0))"
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {kernel}) }},"
                       f" algo.Zip(algo.Tuple(algo.Map({{"
                       f" val _0 = core.ParamDef({ty});"
                       f" algo.AlgoLambda(Seq(_0), {joinseq}) }},"
                       f" {t}), {sseq})))")

    # regardless of a's shape, there is at least the first two dimensions.
    # N is the number of instances, C is the number of channels / the one we
    # are requantizing over.
    w = acc("core.ParamUse(_0)")
    return (f"algo.Map({{ val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), {w}) }}, {a.name})")

@register_operator(shir.qadd.default)
class LowerQadd:
  @staticmethod
  def supports(a, sa, b, sb, z) -> bool:
    # due to limitations of SHIR, we cannot support all scales.
    # unlike the per channel requant case, we really need both scales to be
    # fixed points, so the float point QScale workaround doesn't apply here.
    try:
      # the multiplication gives 32 + w + 1 bits.
      # addition gives 32 + w + 1 + 1 bits.
      _, w, shamt = qscale_to_fixpoint([sa, sb])
      if w > 32 or shamt >= 32 + w + 2:
        return False
    except AssertionError:
      return False

    return True

  @staticmethod
  def lower(a, sa, b, sb, z) -> str:
    # recall the per channel requant case: we have multiple integer
    # multiplicands that share the same shift. we would multiply each channel
    # with the relevant multiplicand and then do a rounding shift (and zero
    # point adjustment of course).
    #
    # in this case, we want to add both tensors after multiplying and only
    # perform the rounding shift after that step. to illustrate:
    #   round(a * sa + b * sb)
    #     = round(a * ia, s) + round(b * ib, s)   <-- (1)
    #     = round(a * ia + b * ib, s)             <-- (2)
    #
    # while (1) is a reasonable approximation, (2) is better since the
    # fractional bits are included in the addition.

    def gen_kernel(lhs, rhs):
      [ia, ib], w, shamt = qscale_to_fixpoint([sa, sb])
      ia = f"algo.Signed(algo.ConstantInteger({ia}, Some(IntType({w}))))"
      ib = f"algo.Signed(algo.ConstantInteger({ib}, Some(IntType({w}))))"
      lhs = f"algo.Mul(algo.Tuple({lhs}, {ia}))"
      rhs = f"algo.Mul(algo.Tuple({rhs}, {ib}))"
      acc = f"algo.Add2({lhs}, {rhs})"

      # because both lhs rhs are 32 + w + 1 bits, add would sneak in one extra
      # bit, then ClipBankersRound will drop off shamt bits.
      bits = 32 + w + 2 - shamt
      acc = (
        f"algo.Sub(algo.Tuple(algo.TruncInteger("
        f"algo.ClipBankersRound({acc}, {shamt}, 0),"
        f" {bits}), algo.ConstantInteger(0)))"
      )

      bits = max(bits, 32) if bits != 32 else 33
      acc = f"algo.Add2({acc}, algo.ConstantInteger({z}, Some(algo.SignedIntType(32))))"

      return f"algo.ClipBankersRound({acc}, 0, {bits - 8})"

    ashape = a.meta.get("val").shape
    if not ashape:
      return gen_kernel(a.name, b.name)

    w = gen_kernel("algo.Select(core.ParamUse(_0), 0)", "algo.Select(core.ParamUse(_0), 1)")
    acc = lambda t: (
      f"algo.Map({{ val _0 = core.ParamDef(algo.TupleType(algo.SignedIntType(32), algo.SignedIntType(32)));"
      f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})"
    )

    for _ in ashape[1:]:
      w = acc("core.ParamUse(_0)")
      acc = lambda t: (
        f"algo.Map({{ val _0 = core.ParamDef();"
        f" algo.AlgoLambda(Seq(_0), {w}) }}, algo.Zip({t}))"
      )

    return acc(f"algo.Zip(algo.Tuple({a.name}, {b.name}))")

@register_operator(shir.flatten.default)
class LowerFlatten:
  @staticmethod
  def supports(a, start, end) -> bool:
    return True

  @staticmethod
  def lower(a, start, end) -> str:
    # a flatten is not much different from view.
    # so we rewrite it into an equivalent view.
    ashape = a.meta.get("val").shape
    if not ashape:
      # flattening a scalar always results in a tensor
      ashape = [1]

    # start and end may be negative, in which case, it is reverse indexed.
    if start < 0:
      start = len(ashape) + start
    if end < 0:
      end = len(ashape) + end

    # axes [start, end] are squished together.
    new_shape = [
      *ashape[:start],
      reduce(lambda x, y: x * y, ashape[start:end + 1]),
      *ashape[end + 1:]
    ]

    # then we just piggy-back on the lowering for aten.view.
    return LowerView.lower(a, new_shape)

@register_operator(aten.view.default)
class LowerView:
  def supports(a, shape) -> bool:
    return True

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

@register_operator(prims.collapse_view.default)
class LowerCollapseView:
  def supports(a, start, end) -> bool:
    # make sure it's non-scalar which shouldn't happen anyway
    if a.meta.get("val").shape:
      return True
    return False

  def lower(a, start, end) -> str:
    joins_needed = end - start - 1
    if joins_needed == 0:
      return a.name

    acc = lambda t: joins_needed * "algo.Join(" + t + joins_needed * ")"
    for _ in range(0, start):
      w = acc("core.ParamUse(_0)")
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")

    return acc(a.name)

@register_operator(prims.split_dim.default)
class LowerSplitDim:
  def supports(a, dim, outer_length) -> bool:
    # make sure it's non-scalar which shouldn't happen anyway
    if a.meta.get("val").shape:
      return True
    return False

  def lower(a, dim, outer_length) -> str:
    # prims.split_dim is defined in terms of outer length whereas
    # algo.Split(...) by default is defined in terms of inner length.
    #
    # note: there is an alternative constructor to define algo.Split in terms
    # as outer length, but it tends to cause issues during type checking!

    shape = a.meta.get("val").shape
    acc = lambda t: f"algo.Split({t}, {shape[dim] // outer_length})"
    for _ in range(dim):
      w = acc("core.ParamUse(_0)")
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")
    return acc(a.name)

@register_operator(prims.squeeze.default)
class LowerSqueeze:
  def supports(a, dims) -> bool:
    return True

  def lower(a, dims) -> str:
    # dims seems to be guaranteed at least an ordered set.
    if dims == []:
      return a.name

    shape = a.meta.get("val").shape
    if len(dims) == len(shape):
      # this is a special case where we turn T[1, ..., 1] into T.
      return (f"algo.Convert(algo.JoinAll({a.name}),"
              f" {shir_type.get_element_type(a).name()})")

    # simplest way to do this is to do the same thing 3D+ inputs:
    # flatten the whole thing to 1D and then use SplitAll
    new_shape = ", ".join((str(s) for (i, s) in enumerate(shape) if i not in dims))
    return (f"algo.Join(algo.SplitAll(algo.JoinAll({a.name}),"
            f" Seq({new_shape})))")

@register_operator(prims.transpose.default)
class LowerTranspose:
  def supports(a, perm_axes) -> bool:
    return True

  def lower(a, perm_axes) -> str:
    if not perm_axes:
      # transposing a scalar is free
      return a.name

    # convert the axes to SHIR axes, which are reversed
    N = len(a.meta.get("val").shape)
    axes = ", ".join((str(N - i - 1) for i in reversed(perm_axes)))
    return f"algo.TransposeND({a.name}, Seq({axes}))"

@register_operator(prims.broadcast_in_dim.default)
class LowerBroadcastInDim:
  def supports(a, shape, broadcast_dims) -> bool:
    return True

  def lower(a, shape, broadcast_dims) -> str:
    ashape = a.meta.get("val").shape
    acc = lambda t: None
    last_idx = len(shape)
    for (idx, old_len) in zip(reversed(broadcast_dims), reversed(ashape)):
      w = acc("core.ParamUse(_0)")

      for d in shape[last_idx - 1:idx:-1]:
        if w is None:
          w = "core.ParamUse(_0)"
        w = f"algo.Repeat({w}, {d})"

      new_len = shape[idx]
      last_idx = idx
      if w is not None:
        w = f"{{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0), {w}) }}"

      if old_len != new_len:
        if w is None:
          acc = lambda t: f"algo.Join(algo.Repeat({t}, {new_len}))"
        else:
          acc = lambda t: (f"algo.Join(algo.Repeat(algo.Map({w}, {t}),"
                           f" {new_len}))")
      elif w is None:
        acc = lambda t: None
      else:
        acc = lambda t: f"algo.Map({w}, {t})"

    w = acc(a.name)
    if w is None:
      w = a.name

    # process the extra broadcast dimensions:
    # e.g. the 5 when broadcasting T[2] into T[5, 2] at dims [1]
    for d in reversed(shape[:last_idx]):
      w = f"algo.Repeat({w}, {d})"

    return w

@register_operator(prims.convert_element_type.default)
class LowerConvEltTy:
  def supports(a, dtype) -> bool:
    a = shir_type.get_element_type(a)
    t = shir_type.get_scalar_type(dtype)
    match (a, t):
      case (shir_type.UI(_) | shir_type.SI(_),
            shir_type.UI(_) | shir_type.SI(_)):
        return True
      case _:
        return False

  def lower(a, dtype) -> str:
    ashape = a.meta.get("val").shape
    atype = shir_type.get_element_type(a)
    dtype = shir_type.get_scalar_type(dtype)
    if atype == dtype:
      return a.name

    match dtype:
      case shir_type.SI(dbits):
        dsigned = True
      case shir_type.UI(dbits):
        dsigned = False

    match atype:
      case shir_type.UI(sbits) | shir_type.SI(sbits) if sbits >= dbits:
        ssigned = False
        extn = lambda t: f"algo.TruncInteger({t}, {dbits})"
      case shir_type.UI(sbits):
        ssigned = False
        extn = lambda t: (f"algo.Add2({t}, algo.ConstantInteger(0,"
                          f" Some(algo.IntType({dbits}))))")
      case shir_type.SI(sbits):
        ssigned = True
        extn = lambda t: (f"algo.Add2({t}, algo.ConstantInteger(0,"
                          f" Some(algo.SignedIntType({dbits}))))")

    acc = extn
    if ssigned != dsigned:
      # always defensively insert a truncate (fragile type inference)
      acc = trunc = lambda t: f"algo.TruncInteger({extn(t)}, {dbits})"
      if dsigned:
        acc = lambda t: f"algo.Sub(algo.Tuple({trunc(t)}, algo.ConstantInteger(0)))"

    for _ in ashape:
      w = acc("core.ParamUse(_0)")
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")

    return acc(a.name)

@register_operator(prims.sum.default)
class LowerSum:
  def supports(inp, dims) -> bool:
    if dims == []:
      # that is technically allowed, but we weren't able to produce it through
      # torch.compile
      print("prims.sum.default dims == []")
      return False

    a = shir_type.get_element_type(inp)
    match a:
      case shir_type.UI(_) | shir_type.SI(_):
        return True
      case _:
        return False

  def lower(inp, dims) -> str:
    def reducer(seq, _):
      ty = shir_type.get_element_type(inp)
      return f"_iredsum({ty.name()}, {seq})"

    return lower_reduction(inp, dims, reducer)

@register_operator(shir.int_mean.default)
class LowerMean:
  @staticmethod
  def supports(a, dims, keepDim) -> bool:
    if shir_type.get_element_type(a) != shir_type.SI(32):
      return False

    return True

  @staticmethod
  def lower(x, dims, keepDim) -> str:
    # TODO: keepDim = True...
    def reducer(seq, elts):
      f, w, s = qscale_to_fixpoint([1.0 / elts])
      clip_bits = max(w + 1 - s, 0)
      return (
        f"algo.Sub(algo.Tuple("
        f"algo.Add2(algo.ConstantInteger(0, Some(algo.SignedIntType(32))),"
        f" algo.ClipBankersRound("
        f"algo.Mul(algo.Tuple(algo.ConstantInteger({f[0]},"
        f" Some(algo.SignedIntType({w + 1}))),"
        f" _iredsum(algo.SignedIntType(32), {seq}))),"
        f" {s}, {clip_bits})), algo.ConstantInteger(0)))"
      )

    # dims may contain negative indices. normalize them first
    xshape = x.meta.get("val").shape
    N = len(xshape)
    dims = [x if x >= 0 else N + x for x in dims]

    reduced = lower_reduction(x, dims, reducer)
    if not keepDim:
      return reduced

    if N == len(dims):
      # promote the full reduction (a scalar) into a tensor
      reduced = f"algo.Repeat({reduced}, 1)"
    else:
      reduced = f"algo.JoinAll({reduced})"

    # every dimension being reduced has now length 1
    w = ", ".join(
      ("1" if i in dims else str(d) for i, d in enumerate(xshape))
    )
    return f"algo.Join(algo.SplitAll({reduced}, Seq({w})))"

def get_same_elt_type(lhs, rhs):
  def extract(value):
    if isinstance(value, torch.fx.Node):
      return shir_type.get_element_type(value)
    return value

  lhs = extract(lhs)
  rhs = extract(rhs)
  match (lhs, rhs):
    case (shir_type.UI(_) | shir_type.SI(_), int(_)):
      return lhs
    case (int(_), shir_type.UI(_) | shir_type.SI(_)):
      return rhs
    case _:
      return lhs if lhs == rhs else None

def lift_to_elt_type(value, ty) -> (str, torch.Size):
  match value:
    case int(_):
      return (f"algo.ConstantInteger({value}, Some({ty.name()}))", torch.Size([]))
    case _:
      return (value.name, value.meta.get("val").shape)

@register_operator(aten.relu.default)
class LowerRelu:
  def supports(a) -> bool:
    return True

  def lower(a) -> str:
    ty = shir_type.get_element_type(a)
    a = lift_to_elt_type(a, ty)
    b = lift_to_elt_type(0, ty)
    return lower_pairwise_binop("algo.Max.asFunction()", *a, *b)

@register_operator(aten.clamp.default)
class LowerClamp:
  @staticmethod
  def supports(a, clmin=None, clmax=None) -> bool:
    # since SHIR uses evalInt, disallow values that go beyond s32 range
    # this is good enough for most purposes
    s32 = shir_type.SI(32)
    ty = shir_type.get_element_type(a)
    tmin = max(s32.minval(), ty.minval())
    tmax = min(s32.maxval(), ty.maxval())
    return (
      (clmin is None or tmin <= clmin <= tmax) and
      (clmax is None or tmin <= clmax <= tmax)
    )

  @staticmethod
  def lower(a, min=None, max=None) -> str:
    tname = shir_type.get_element_type(a).name()
    def emit_cmp_op(value):
      if min is not None:
        value = f"algo.Max2({value}, algo.ConstantInteger({min}, Some({tname})))"
      if max is not None:
        value = f"algo.Min2({value}, algo.ConstantInteger({max}, Some({tname})))"
      return value

    ashape = a.meta.get("val").shape
    if not ashape:
      return emit_cmp_op(a.name)

    acc = emit_cmp_op
    for _ in ashape:
      w = acc("core.ParamUse(_0)")
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")

    return acc(a.name)

@register_operator(prims.maximum.default)
class LowerMaximum:
  def supports(lhs, rhs) -> bool:
    a = shir_type.get_element_type(lhs)
    b = shir_type.get_element_type(rhs)
    if a != b:
      return False
    match a:
      case shir_type.UI(_) | shir_type.SI(_):
        return True
      case _:
        return False

  def lower(lhs, rhs) -> str:
    return lower_pairwise_tensor_binop("algo.Max.asFunction()", lhs, rhs)

@register_operator(prims.add.default)
class LowerAdd:
  def supports(lhs, rhs) -> bool:
    a = get_same_elt_type(lhs, rhs)
    match a:
      case shir_type.UI(_) | shir_type.SI(_):
        return True
      case _:
        return False

  def lower(lhs, rhs) -> str:
    ty = get_same_elt_type(lhs, rhs)
    lhs = lift_to_elt_type(lhs, ty)
    rhs = lift_to_elt_type(rhs, ty)

    f = "algo.Add(core.ParamUse(_0))"
    match ty:
      case shir_type.UI(bits):
        f = f"algo.TruncInteger({f}, {bits})"
      case shir_type.SI(bits):
        f = f"algo.Sub(algo.Tuple(algo.TruncInteger({f}, {bits}), algo.ConstantInteger(0)))"
    f = f"{{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0), {f}) }}"
    return lower_pairwise_binop(f, *lhs, *rhs)

@register_operator(prims.sub.default)
class LowerSub:
  def supports(lhs, rhs) -> bool:
    a = get_same_elt_type(lhs, rhs)
    match a:
      case shir_type.UI(_) | shir_type.SI(_):
        return True
      case _:
        return False

  def lower(lhs, rhs) -> str:
    ty = get_same_elt_type(lhs, rhs)
    lhs = lift_to_elt_type(lhs, ty)
    rhs = lift_to_elt_type(rhs, ty)

    match ty:
      case shir_type.UI(bits):
        f = (f"{{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0),"
             f" algo.TruncInteger(algo.Sub(core.ParamUse(_0)), {bits})) }}")
      case shir_type.SI(bits):
        f = "algo.Sub.asFunction()"
    return lower_pairwise_binop(f, *lhs, *rhs)

@register_operator(prims.mul.default)
class LowerMul:
  def supports(lhs, rhs) -> bool:
    a = get_same_elt_type(lhs, rhs)
    match a:
      case shir_type.UI(_) | shir_type.SI(_):
        return True
      case _:
        return False

  def lower(lhs, rhs) -> str:
    ty = get_same_elt_type(lhs, rhs)
    lhs = lift_to_elt_type(lhs, ty)
    rhs = lift_to_elt_type(rhs, ty)

    f = "algo.Mul(core.ParamUse(_0))"
    match ty:
      case shir_type.UI(bits):
        f = f"algo.TruncInteger({f}, {bits})"
      case shir_type.SI(bits):
        f = f"algo.Sub(algo.Tuple(algo.TruncInteger({f}, {bits}), algo.ConstantInteger(0)))"
    f = f"{{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0), {f}) }}"
    return lower_pairwise_binop(f, *lhs, *rhs)

@register_operator(aten.mm.default)
class LowerMM:
  def supports(lhs, rhs) -> bool:
    a = shir_type.get_element_type(lhs)
    b = shir_type.get_element_type(rhs)
    if a != b:
      return False
    match a:
      case shir_type.UI(_) | shir_type.SI(_):
        return True
      case _:
        return False

  def lower(lhs, rhs) -> str:
    ty = shir_type.get_element_type(lhs)
    f = f"_idotp({ty.name()}, core.ParamUse(_0), core.ParamUse(_1))"

    # lhs: T[i, width], rhs: T[width, j]
    width = lhs.meta.get("val").shape[1]
    return (f"algo.Map({{"
            f" val _2 = algo.SeqType({ty.name()}, {width});"
            f" val _0 = core.ParamDef(_2);"
            f" algo.AlgoLambda(Seq(_0),"
            f" algo.Map({{"
            f" val _1 = core.ParamDef(_2);"
            f" algo.AlgoLambda(Seq(_1), {f})"
            f" }}, algo.Transpose({rhs.name}))) }}, {lhs.name})")

@register_operator(shir.int_addmm.default)
class LowerIntAddmm:
  @staticmethod
  def supports(acc, lhs, rhs) -> bool:
    assert shir_type.get_element_type(lhs) == shir_type.SI(8)
    assert shir_type.get_element_type(rhs) == shir_type.SI(8)
    assert shir_type.get_element_type(acc) == shir_type.SI(32)
    return True

  @staticmethod
  def lower(acc, lhs, rhs) -> str:
    return f"_add32_mm8({acc.name}, {lhs.name}, {rhs.name})"

@register_operator(aten.convolution.default)
class LowerConvolution:
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

  @classmethod
  def lower(cls, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups) -> str:
    elt_ty = shir_type.get_element_type(input)
    ishape = input.meta.get("val").shape
    wshape = weight.meta.get("val").shape

    obj = cls(elt_ty, stride, padding, dilation)
    return obj.emit_many(input.name, ishape, weight, wshape, groups)

  def __init__(self, elt_ty, stride, padding, dilation):
    self.elt_ty = elt_ty
    self.stride = stride
    self.padding = padding
    self.dilation = dilation

  def _zero_pad(self, input, idim):
    acc = lambda t: None
    for (pad, width) in zip(reversed(self.padding), reversed(idim)):
      w1 = acc("core.ParamUse(_0)")
      if w1 is None:
        if not pad:
          continue  # acc is lambda t: None
        inner = lambda t: t
      else:
        inner = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                           f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")

      if pad:
        w2 = pad
        acc = lambda t: f"algo.Pad({inner(t)}, {w2}, {w2}, 0)"
      else:
        acc = inner

    w1 = acc("core.ParamUse(_0)")
    if w1 is None:
      # when no padding is required
      return input

    return (f"algo.Map({{ val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), algo.Map({{"
            f" val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), {w1}) }},"
            f" core.ParamUse(_0))) }}, {input})")

  def emit_single(self, input, idim, kernel, kdim):
    # idim[0]: number of inputs
    # idim[1]: InChan
    # kdim[0]: OutChan
    # kdim[1]: InChan
    N = len(idim) - 2

    # JoinAll needs the type of the value to function correctly.
    # the type is the type of the original inner kernel, which is
    # T[InChan, Out0, ..., OutN]
    inkern_ty = self.elt_ty.name()
    for w in reversed(kdim[1:]):
      inkern_ty = f"algo.SeqType({inkern_ty}, {w})"

    stream = zip(reversed(kdim[2:]), reversed(idim[2:]),
                 reversed(self.stride),
                 reversed(self.padding),
                 reversed(self.dilation))

    acc = lambda t: t
    for (kwidth, iwidth, stride, pad, dilation) in stream:
      # compute the widths due to input padding and kernel dilation
      i = iwidth + 2 * pad
      k = dilation * (kwidth - 1) + 1

      leftovers = (i - k) % stride
      if leftovers:
        w1 = acc(f"algo.Drop(core.ParamUse(_0), 0, {leftovers})")
      else:
        w1 = acc("core.ParamUse(_0)")
      w2 = k
      w3 = f"algo.SlideGeneral({w1}, {w2}, {stride})"
      w5 = f"algo.SeqType(algo.AlgoDataTypeVar(), {i})"

      if dilation > 1:
        # map over the slided input (w3) a (slide + join) to drop the dilated
        # indices from the 2nd dimension (K') dimension.
        w3 = (f"algo.Map({{"
              f" val _3 = core.ParamDef(algo.SeqType(algo.AlgoDataTypeVar(), {k}));"
              f" algo.AlgoLambda(Seq(_3),"
              f" algo.Join(algo.SlideGeneral(core.ParamUse(_3), 1, {dilation}))) }}, {w3})")

      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef({w5});"
                       f" algo.AlgoLambda(Seq(_0), {w3}) }}, {t})")

    frag_sliding = acc("core.ParamUse(_0)")

    # we now have T[InChan, Out1, K1, Out2, K2, ..., OutN, KN]
    # we transpose it to T[InChan, Out1, ..., OutN, K1, ..., KN]
    # that gives normal permutation indices of [1, 3, 5, ..., 0, 2, 4, 6, ...]
    #
    # recall that indices in SHIR are backwards
    transpose_axes = chain(range(2 * N, -2, -2), range(2 * N - 1, -1, -2))
    transpose_axes = (2 * N - x for x in transpose_axes)
    frag_transpose = f"algo.TransposeND({frag_sliding}, Seq{(*transpose_axes,)})"

    acc = lambda t: (f"_idotp({self.elt_ty.name()}, algo.JoinAll({t}),"
                     f" algo.JoinAll(core.ParamUse(_1)))")

    # recall that JoinAll needs type annotation.
    w1 = acc("core.ParamUse(_0)")
    acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef(_2);"
                     f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")

    for _ in range(N - 1):
      w1 = acc("core.ParamUse(_0)")
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")
    frag_inner_map = acc(frag_transpose)

    input = self._zero_pad(input, idim)
    return (f"algo.Map({{ val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), algo.Map({{"
            f" val _2 = {inkern_ty}; val _1 = core.ParamDef(_2);"
            f" algo.AlgoLambda(Seq(_1), {frag_inner_map}) }},"
            f" {kernel})) }}, {input})")

  def emit_many(self, input, idim, kernel, kdim, groups):
    if groups == 1:
      return self.emit_single(input, idim, kernel, kdim)

    # we rewrite the grouped convolution into many smaller parallelizable
    # convolutions and then concatenate (using algo.Join) the result.
    #
    # the partition is done on the channel dimension

    iwindow = idim[1] // groups   # split on input channel
    kwindow = kdim[0] // groups   # split on output channel

    inner_idim = [*idim]
    inner_kdim = [*kdim]
    inner_idim[1] = iwindow
    inner_kdim[0] = kwindow

    # compute the types for the partitioned convolution because some operators
    # need them...

    # the input will be transposed, so 0th and 1st dimensions are swapped!
    input = f"algo.Split(algo.Transpose({input}), {iwindow})"
    input_ty = self.elt_ty.name()
    for d in reversed(inner_idim[2:]):
      input_ty = f"algo.SeqType({input_ty}, {d})"
    input_ty = (f"algo.SeqType(algo.SeqType({input_ty},"
                f" {inner_idim[0]}), {inner_idim[1]})")

    kernel = f"algo.Split({kernel}, {kwindow})"
    kernel_ty = self.elt_ty.name()
    for d in reversed(inner_kdim):
      kernel_ty = f"algo.SeqType({kernel_ty}, {d})"

    conv = self.emit_single(
        "algo.Transpose(algo.Select(core.ParamUse(_4), 0))", inner_idim,
        "algo.Select(core.ParamUse(_4), 1)", inner_kdim)
    conv = (f"algo.Map({{ val _4 = core.ParamDef("
            f"algo.TupleType({input_ty}, {kernel_ty}));"
            f" algo.AlgoLambda(Seq(_4), {conv}) }},"
            f" algo.Zip(algo.Tuple({input}, {kernel})))")

    # at this point, we have T[Groups, N, OutChan, S1, S2, ...].
    # merge the 0th (Groups) dimension into the 2nd (OutChan) dimension.
    return f"algo.Map(algo.Join.asFunction(), algo.Transpose({conv}))"

class LowerMaxPoolND:
  # class-level field N controls the amount of dimensions

  @classmethod
  def supports(cls, input, kernel_size, stride, padding, dilation) -> bool:
    # sanity check, make sure the sizes match:
    N = cls.N
    if N != len(kernel_size) or N != len(stride) or N != len(padding) or N != len(dilation):
      return False

    return True

  @classmethod
  def lower(cls, input, kernel_size, stride, padding, dilation) -> str:
    elt_ty = shir_type.get_element_type(input)
    ishape = input.meta.get("val").shape

    obj = cls(elt_ty, kernel_size, stride, padding, dilation)
    return obj.emit(input.name, ishape)

  def __init__(self, elt_ty, kernel_size, stride, padding, dilation):
    self.elt_ty = elt_ty
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation

  def _min_pad(self, input, idim):
    v = self.elt_ty.minval()
    acc = lambda t: None
    for (pad, width) in zip(reversed(self.padding), reversed(idim)):
      w1 = acc("core.ParamUse(_0)")
      if w1 is None:
        if not pad:
          continue  # acc is lambda t: None
        inner = lambda t: t
      else:
        inner = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                           f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")

      if pad:
        w2 = pad
        acc = lambda t: f"algo.Pad({inner(t)}, {w2}, {w2}, {v})"
      else:
        acc = inner

    w1 = acc("core.ParamUse(_0)")
    if w1 is None:
      # when no padding is required
      return input

    if len(idim) == self.N + 1:
      return (f"algo.Map({{ val _0 = core.ParamDef();"
              f" algo.AlgoLambda(Seq(_0), {w1}) }}, {input})")

    return (f"algo.Map({{ val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), algo.Map({{"
            f" val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), {w1}) }},"
            f" core.ParamUse(_0))) }}, {input})")

  def emit(self, input, idim):
    # JoinAll needs the type of the kernel
    inkern_ty = self.elt_ty.name()
    for w in reversed(self.kernel_size):
      inkern_ty = f"algo.SeqType({inkern_ty}, {w})"

    stream = zip(reversed(self.kernel_size), reversed(idim),
                 reversed(self.stride),
                 reversed(self.padding),
                 reversed(self.dilation))

    acc = lambda t: t
    for (kwidth, iwidth, stride, pad, dilation) in stream:
      # compute the widths due to input padding and kernel dilation
      i = iwidth + 2 * pad
      k = dilation * (kwidth - 1) + 1

      leftovers = (i - k) % stride
      if leftovers:
        w1 = acc(f"algo.Drop(core.ParamUse(_0), 0, {leftovers})")
      else:
        w1 = acc("core.ParamUse(_0)")
      w2 = k
      w3 = f"algo.SlideGeneral({w1}, {w2}, {stride})"
      w5 = f"algo.SeqType(algo.AlgoDataTypeVar(), {i})"

      if dilation > 1:
        # map over the slided input (w3) a (slide + join) to drop the dilated
        # indices from the 2nd dimension (K') dimension.
        w3 = (f"algo.Map({{"
              f" val _3 = core.ParamDef(algo.SeqType(algo.AlgoDataTypeVar(), {k}));"
              f" algo.AlgoLambda(Seq(_3),"
              f" algo.Join(algo.SlideGeneral(core.ParamUse(_3), 1, {dilation}))) }}, {w3})")

      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef({w5});"
                       f" algo.AlgoLambda(Seq(_0), {w3}) }}, {t})")

    # we sneak in a Split here (and a Join later) to simplify the mapping
    # sliding process / undo the last map
    frag_sliding = acc(f"algo.Split(core.ParamUse(_0), {i})")

    # we now have T[Out1, K1, Out2, K2, ..., OutN, KN]
    # we transpose it to T[Out1, Out2, ..., K1, ..., KN]
    # that gives permutation indices of [0, 2, ..., 1, 3, ...]
    #
    # (it happens to be identical to SHIR indices)
    transpose_axes = chain(range(0, 2 * self.N, 2), range(1, 2 * self.N + 1, 2))
    frag_transpose = f"algo.TransposeND(algo.Join({frag_sliding}), Seq{(*transpose_axes,)})"

    # obviously, unlike convolution, max pooling performs a maximum reduction
    # instead of a dot product.
    acc = lambda t: f"_iredmax({self.elt_ty.name()}, algo.JoinAll({t}))"

    # recall that JoinAll needs type annotation.
    w1 = acc("core.ParamUse(_0)")
    acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef({inkern_ty});"
                     f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")

    for _ in range(self.N - 1):
      w1 = acc("core.ParamUse(_0)")
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")
    frag_inner_map = acc(frag_transpose)

    input = self._min_pad(input, idim)

    # account for the minibatch dimension that may be omitted.
    if len(idim) == self.N + 1:
      return (f"algo.Map({{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0),"
              f" {frag_inner_map}) }}, {input})")

    assert len(idim) == self.N + 2
    return (f"algo.Map({{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0),"
            f" algo.Map({{ val _0 = core.ParamDef(); algo.AlgoLambda(Seq(_0),"
            f" {frag_inner_map}) }}, core.ParamUse(_0))) }}, {input})")

@register_operator(shir.int_max_pool2d.default)
class LowerMaxPool2D(LowerMaxPoolND):
  N = 2

class LowerAdaptiveAvgPoolND:
  # class-level field N controls the amount of dimensions

  @classmethod
  def supports(cls, input, output_size) -> bool:
    elt_ty = shir_type.get_element_type(input)
    if elt_ty != shir_type.SI(32):
      # sanity check
      return False

    if cls.N != len(output_size):
      return False
    return True

  @classmethod
  def lower(cls, input, output_size) -> str:
    def gen_indices():
      yield cls.N - 1
      for i in range(cls.N - 1):
        yield i
      yield cls.N

    indices = ", ".join((str(x) for x in gen_indices()))
    ishape = input.meta.get("val").shape

    # we need to account for the fact that it input may have shape
    # T[_, _, D1, ..., DN] or T[_, D1, ..., DN]
    body = cls._helper("core.ParamUse(_0)", -cls.N, [],
                       ishape, output_size, indices)
    # inner_ty = reduce(lambda x, y: f"algo.Seq({x}, {y})",
    #                   reversed(ishape[-N:]), "algo.SignedIntType(32)")

    if len(ishape) == cls.N + 1:
      return (f"algo.Map({{ val _0 = core.ParamDef();"
              f" algo.AlgoLambda(Seq(_0), {body}) }}, {input.name})")

    assert len(ishape) == cls.N + 2
    return (f"algo.Map({{ val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0),"
            f" algo.Map({{ val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), {body}) }},"
            f" core.ParamUse(_0))) }}, {input.name})")

  @classmethod
  def _helper(cls, input, level, dims, ishape, output_size, indices):
    ranges = cls._decompose_range(ishape[level], output_size[level])
    def gen_subparts():
      for (hd, tl, w, stride, reps) in ranges:
        s = input
        if hd != 0 or tl != 0:
          s = f"algo.Drop({s}, {hd}, {tl})"
        s = f"algo.SlideGeneral({s}, {w}, {stride})"

        # recall that level starts from -N
        if level != -1:
          # there is at least one more dimension, s : T[FN, WN, D..., W...].
          # 1.  push WN downwards: T[FN, D..., W..., WN]
          # 2.  map over FN: T[D..., W..., WN]
          # 3.  recursively generate the next level
          dims.append(w)
          recr = cls._helper("core.ParamUse(_0)", level + 1, dims,
                             ishape, output_size, indices)
          dims.pop()
          s = (f"algo.Map({{ val _0 = core.ParamDef();"
               f" algo.AlgoLambda(Seq(_0), {recr}) }},"
               f" algo.TransposeND({s}, Seq({indices})))")

        else:
          # this is the innermost dimension, s : T[FN, WN, W1, W2, ...].
          # JoinAll (if necessary) + reduction
          elts = reduce(lambda x, y: x * y, dims, w)
          ty = reduce(lambda x, y: f"algo.SeqType({x}, {y})",
                      chain(reversed(dims), [w]), "algo.SignedIntType(32)")

          if dims == []:
            flatseq = "core.ParamUse(_0)"
          else:
            flatseq = "algo.JoinAll(core.ParamUse(_0))"

          f, w, s = qscale_to_fixpoint([1.0 / elts])
          # multiply and round gives w + 1 + 32 - s bits
          # but then we are saturating it to 32 bits,
          # so it's that expression - 32, giving the following.
          clip_bits = max(w + 1 - s, 0)
          s = (f"algo.Map({{ val _0 = core.ParamDef({ty});"
               f" algo.AlgoLambda(Seq(_0),"
               f" algo.Sub(algo.Tuple(algo.TruncInteger(algo.Add2(algo.ConstantInteger(0,"
               f" Some(algo.SignedIntType(32))),"
               f" algo.ClipBankersRound(algo.Mul(algo.Tuple(algo.ConstantInteger({f[0]},"
               f" Some(algo.SignedIntType({w + 1}))),"
               f" _iredsum(algo.SignedIntType(32),"
               f" {flatseq}))), {s}, {clip_bits})), 32),"
               f" algo.ConstantInteger(0)))) }}, {s})")
        yield s
    return reduce(lambda x, y: f"algo.Concat({x}, {y})", gen_subparts())

  @staticmethod
  def _decompose_range(idim, output_size):
    """
    Turns it into a bunch of (hd, tl, w, s, reps):
    Repeat(SlideGeneral(Drop(input, hd, tl), w, s), reps)
    """

    assert output_size > 0

    def gen_range():
      # start uses floor division
      # end uses ceiling division
      repetition = 1
      last_start = 0
      last_end = -(idim // -output_size)

      for i in range(1, output_size):
        start = i * idim // output_size
        end = -((i + 1) * idim // -output_size)
        if start == last_start and end == last_end:
          repetition += 1
          continue

        yield repetition, last_start, last_end
        repetition = 1
        last_start = start
        last_end = end
      yield repetition, last_start, last_end

    slides = []
    last_run = None
    last_stride = None
    last_idx = None
    drop_left = None

    def save_slide(reps):
      if last_run is not None:
        s = 1 if last_stride is None else last_stride
        leftover = idim - last_idx - last_run
        slides.append((drop_left, leftover, last_run, s, reps))

    for (rep, start, end) in gen_range():
      run = end - start
      if rep != 1:
        # this has to be it's own group
        save_slide(1)
        slides.append((
            start,
            idim - end,
            run,
            1,
            rep,
        ))
        last_run = None
      elif last_run == run:
        new_stride = start - last_idx
        if last_stride is None:
          last_stride = new_stride
        assert last_stride == new_stride, "Stride mismatch"
        last_idx = start
      else:
        save_slide(1)

        last_run = run
        drop_left = start
        last_idx = start
        last_stride = None

    save_slide(1)
    return slides

@register_operator(shir.int_adaptive_avg_pool2d.default)
class LowerAdaptiveAvgPool2D(LowerAdaptiveAvgPoolND):
  N = 2
