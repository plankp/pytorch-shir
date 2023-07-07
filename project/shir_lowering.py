import torch
import struct
import shir_type
import shir_intrinsic
from functools import reduce
from itertools import chain

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

def lower_reduction(x: torch.fx.Node, dims: list[int],
                    reducer: str, castf=lambda x: x) -> str:
  """
  reducer has type T -> T -> U where U and T are compatible*
  castf converts U -> R
  the result will operate on x: T[S] and reduce on dims only to give R[...]

  this is a handwave-y description of what happens.
  casting happens after the Fold step.
  """

  assert dims != []

  elt_ty = shir_type.get_element_type(x)
  xshape = x.meta.get("val").shape
  x = x.name
  N = len(xshape)
  join_layers = len(dims)
  unpeel_layers = N - join_layers

  if unpeel_layers > 0:
    prefix = [i for i in range(N) if i not in dims]
    if any((i != j for (i, j) in zip(prefix, range(N)))):
      # we have a gap in the reduction axes: need to transpose
      # using tuple splat here is also safe because the axes will have at
      # least two elements
      axes = [*prefix, *dim]
      x = f"algo.TransposeND({x}, Seq{(*(N - i - 1 for i in reversed(axes)),)})"

  # for sanity reasons, we avoid using algo.JoinAll here
  hd = (join_layers - 1) * "algo.Join("
  tl = (join_layers - 1) * ")"
  acc = lambda t: castf(f"algo.Fold({reducer}, {hd}{t}{tl})")

  if unpeel_layers > 0:
    # the inner-most map needs a type annotation :shrug:
    shape = reduce(lambda x, y: f"algo.SeqType({x}, {y})",
                   (xshape[d] for d in reversed(dims)),
                   elt_ty.name())

    w = acc("core.ParamUse(_0)")
    acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef({shape});"
                     f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")

  for _ in range(unpeel_layers - 1):
    w = acc("core.ParamUse(_0)")
    acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                     f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})")

  return acc(x)

"""
magic that actually does the lowering of each node
"""

@register_operator(shir.requantize.default)
class LowerRequantize:
  @classmethod
  def supports(cls, a, s, z) -> bool:
    if shir_type.get_element_type(a) != shir_type.SI(32):
      return False

    try:
      # we just ignore the q > 0 case for simplicity
      _, q = cls.qscale_to_fixpoint(s)
      return q <= 0
    except AssertionError:
      pass

    return False

  @classmethod
  def lower(cls, a, s, z) -> str:
    s, q = cls.qscale_to_fixpoint(s)
    assert q <= 0

    def gen_kernel(t):
      width = 32      # we're quantizing a 32-bit integer

      sbits = s.bit_length() + 1  # +1 to make sure it's a positive signed value
      width += sbits  # we multiply by the scale
      # XXX: without the extra Id node, algo to arch lowering seems to fail...
      acc = (f"algo.Mul(algo.Tuple(algo.Id({t}),"
             f" algo.ConstantInteger({s}, Some(algo.SignedIntType({sbits})))))")

      # as usual, ClipBankersRound (both algo and arch versions) need exact
      # width information. we're already tracking it, so just insert
      # TruncIntegers type checking for sanity.
      if q < 0:
        width += q    # we dropped q bits via rounding
        acc = (f"algo.Sub(algo.Tuple(algo.TruncInteger("
               f"algo.ClipBankersRound({acc}, {-q}, 0), {width}),"
               f" algo.ConstantInteger(0)))")

      if z != 0:
        # algo.Add will sneak in an extra bit if both are 32 bits (unlikely)
        width = max(width, 32) if width != 32 else 33
        acc = f"algo.Add2({acc}, algo.ConstantInteger({z}, Some(algo.SignedIntType(32))))"

      return f"algo.ClipBankersRound({acc}, 0, {width - 8})"

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

  @staticmethod
  def qscale_to_fixpoint(f: float) -> (int, int):
    if f == 0:
      return (0, 0)

    assert f >= 0, "qscale cannot be negative"
    bits = struct.unpack(">l", struct.pack(">f", f))[0]
    exp = bits >> 23
    assert 0 < exp < 0xff, "qscale must be normal"

    man = (bits & 0x7f_ffff) | 0x80_0000
    width = 0
    if (man & 0xffff) == 0:
      man >>= 16
      width += 16
    if (man & 0xff) == 0:
      man >>= 8
      width += 8
    if (man & 0xf) == 0:
      man >>= 4
      width += 4
    if (man & 0x3) == 0:
      man >>= 2
      width += 2
    if (man & 0x1) == 0:
      man >>= 1
      width += 1

    return (man, exp - 127 - 23 + width)


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

    shape = ", ".join((str(s) for s in shape))
    return (f"algo.Join(algo.SplitAll({a}, Seq({shape})))")

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
    ty = shir_type.get_element_type(inp)
    match ty:
      case shir_type.UI(bits):
        f = lambda t: f"algo.TruncInteger({t}, {bits})"
      case shir_type.SI(bits):
        f = lambda t: f"algo.Sub(algo.Tuple(algo.TruncInteger({t}, {bits}), algo.ConstantInteger(0)))"

    # due to how Fold works, we need to perform the Add2 as unsigned!
    op = (f"{{ val _0 = core.ParamDef(); val _1 = core.ParamDef();"
          f" algo.AlgoLambda(Seq(_0, _1),"
          f" algo.Add2(algo.TruncInteger(core.ParamUse(_0), {bits}),"
          f" algo.TruncInteger(core.ParamUse(_1), {bits}))) }}")
    return lower_reduction(inp, dims, op, f)

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

@register_operator(aten.convolution.default)
class LowerConvolution:
  def supports(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups) -> bool:
    if transposed:
      return False
    if any((p != 0 for p in output_padding)):
      return False

    # allow bias if it's also the same type.
    # (aten.convolution seems to do weird coercion business otherwise)
    if bias and shir_type.get_element_type(bias) != shir_type.get_element_type(input):
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
    bias = bias.name if bias else None

    obj = cls(elt_ty, stride, padding, dilation)
    return obj.emit_many(input.name, ishape, weight, wshape, bias, groups)

  def __init__(self, elt_ty, stride, padding, dilation):
    self.elt_ty = elt_ty
    self.stride = stride
    self.padding = padding
    self.dilation = dilation

  def _zero_pad(self, input, idim):
    iter = zip(reversed(self.padding), reversed(idim))
    frag_padding = f"algo.ConstantInteger(0, Some({self.elt_ty.name()}))"

    (pad, width) = next(iter)
    w2 = frag_padding
    if not pad:
      acc = lambda t: None
    else:
      acc = lambda t: (f"{{ val _1 = algo.Repeat({w2}, {pad});"
                       f" algo.Concat(algo.Concat(_1, {t}), _1) }}")

    for (next_pad, next_width) in iter:
      frag_padding = f"algo.Repeat({frag_padding}, {width + 2 * pad})"

      w1 = acc("core.ParamUse(_0)")
      w2 = frag_padding
      pad = next_pad
      width = next_width

      if w1 is None:
        if not pad:
          continue  # acc is lambda t: None
        inner = lambda t: t
      else:
        inner = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                           f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")

      if pad:
        acc = lambda t: (f"{{ val _1 = algo.Repeat({w2}, {pad}); "
                         f" algo.Concat(algo.Concat(_1, {inner(t)}), _1) }}")
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

  def emit_single(self, input, idim, kernel, kdim, bias):
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
      out_width = i - k + 1   # the new dimension after sliding

      w1 = acc("core.ParamUse(_0)")
      w2 = k
      w3 = f"algo.Slide({w1}, {w2})"
      w5 = f"algo.SeqType(algo.AlgoDataTypeVar(), {i})"

      if dilation > 1:
        # now that the slided input (w3) has type T[Out', K', ...], we drop the
        # the dilated indices from the 2nd dimension (K') of the slided input.
        indices = range(k // dilation + 1)
        indices = (f"algo.Drop(core.ParamUse(_3), {dilation * i}, {k - 1 - dilation * i})" for i in indices)
        w4 = reduce(lambda x, y: f"algo.Concat({x}, {y})", indices)
        w3 = (f"algo.Map({{"
              f" val _3 = core.ParamDef(algo.SeqType(algo.AlgoDataTypeVar(), {k}));"
              f" algo.AlgoLambda(Seq(_3), {w4}) }}, {w3})")

      if stride > 1:
        # strides work on the 1st dimension (Out') of the slided input, which is
        # affected by the dilated kernel width.
        indices = range(1 if stride > out_width else out_width // stride)
        indices = (f"algo.Drop(_2, {stride * i}, {out_width - 1 - stride * i})" for i in indices)
        w4 = reduce(lambda x, y: f"algo.Concat({x}, {y})", indices)
        w3 = f"{{ val _2 = {w3}; {w4} }}"

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

    if bias is None:
      acc = lambda t: (f"_idotp({self.elt_ty.name()}, algo.JoinAll({t}),"
                       f" algo.JoinAll(core.ParamUse(_1)))")
    else:
      # we needed to add algo.Select now that the values come from tuples
      acc = lambda t: (f"_idotp({self.elt_ty.name()}, algo.JoinAll({t}),"
                       f" algo.JoinAll(algo.Select(core.ParamUse(_1), 0)),"
                       f" Some(algo.Select(core.ParamUse(_1), 1)))")

    # recall that JoinAll needs type annotation.
    w1 = acc("core.ParamUse(_0)")
    acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef(_2);"
                     f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")

    for _ in range(N - 1):
      w1 = acc("core.ParamUse(_0)")
      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef();"
                       f" algo.AlgoLambda(Seq(_0), {w1}) }}, {t})")
    frag_inner_map = acc(frag_transpose)

    if bias is None:
      annot = "_2"
    else:
      # the type of the output-channel (2nd dimension) map is a tuple
      annot = f"algo.TupleType(_2, {self.elt_ty.name()})"
      kernel = f"algo.Zip(algo.Tuple({kernel}, {bias}))"

    input = self._zero_pad(input, idim)
    return (f"algo.Map({{ val _0 = core.ParamDef();"
            f" algo.AlgoLambda(Seq(_0), algo.Map({{"
            f" val _2 = {inkern_ty}; val _1 = core.ParamDef({annot});"
            f" algo.AlgoLambda(Seq(_1), {frag_inner_map}) }},"
            f" {kernel})) }}, {input})")

  def emit_many(self, input, idim, kernel, kdim, bias, groups):
    if groups == 1:
      return self.emit_single(input, idim, kernel, kdim, bias)

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

    if bias is not None:
      bias = f"algo.Split({bias}, {kwindow})"
      bias_ty = f"algo.SeqType({self.elt_ty.name()}, {kwindow})"

    if bias is None:
      conv = self.emit_single(
          "algo.Transpose(algo.Select(core.ParamUse(_4), 0))", inner_idim,
          "algo.Select(core.ParamUse(_4), 1)", inner_kdim,
          None)
      conv = (f"algo.Map({{ val _4 = core.ParamDef("
              f"algo.TupleType({input_ty}, {kernel_ty}));"
              f" algo.AlgoLambda(Seq(_4), {conv}) }},"
              f" algo.Zip(algo.Tuple({input}, {kernel})))")
    else:
      conv = self.emit_single(
          "algo.Transpose(algo.Select3(core.ParamUse(_4), 0))", inner_idim,
          "algo.Select3(core.ParamUse(_4), 1)", inner_kdim,
          "algo.Select3(core.ParamUse(_4), 2)")
      conv = (f"algo.Map({{ val _4 = core.ParamDef("
              f"algo.TupleType({input_ty}, {kernel_ty}, {bias_ty}));"
              f" algo.AlgoLambda(Seq(_4), {conv}) }},"
              f" algo.Zip3(algo.Tuple3({input}, {kernel}, {bias})))")

    # at this point, we have T[Groups, N, OutChan, S1, S2, ...].
    # merge the 0th (Groups) dimension into the 2nd (OutChan) dimension.
    return f"algo.Map(algo.Join.asFunction(), algo.Transpose({conv}))"

class LowerMaxPoolND:
  def supports(input, kernel_size, stride=[], padding=0, dilation=1,
               ceil_mode=False) -> bool:
    # disallow padding for now: it (usually) does not zero-pad
    match padding:
      case int(w) if w == 0:
        return True
      case list(w) if all(x == 0 for x in padding):
        return True
      case _:
        return False

    if ceil_mode:
      return False

    return True

  def __init__(self, N, elt_ty, kernel_size, stride, dilation):
    self.N = N
    self.elt_ty = elt_ty
    self._kernel_size = kernel_size
    self._stride = stride if stride != [] else kernel_size
    self._dilation = dilation

  # For some reason, max_pool?d loves to keep things as scalar and rely on
  # implicit broadcasting, hence all these helpers to fetch the widths.
  #
  # we also assume we never do illegal accesses

  def _broadcast_get(self, u, axis: int) -> int:
    match u:
      case int(w) | [w]:
        return w
      case w:
        return w[axis]

  def kshape(self, axis: int) -> int:
    return self._broadcast_get(self._kernel_size, axis)

  def stride(self, axis: int) -> int:
    return self._broadcast_get(self._stride, axis)

  def dilation(self, axis: int) -> int:
    return self._broadcast_get(self._dilation, axis)

  def emit(self, input, idim):
    # JoinAll needs the type of the kernel
    inkern_ty = self.elt_ty.name()
    for dim in reversed(range(self.N)):
      inkern_ty = f"algo.SeqType({inkern_ty}, {self.kshape(dim)})"

    acc = lambda t: t
    for (dim, iwidth) in zip(reversed(range(self.N)), reversed(idim)):
      kwidth = self.kshape(dim)
      stride = self.stride(dim)
      dilation = self.dilation(dim)

      # compute the widths due to input padding and kernel dilation
      i = iwidth
      k = dilation * (kwidth - 1) + 1
      out_width = i - k + 1   # new dimension after sliding

      w1 = acc("core.ParamUse(_0)")
      w2 = k
      w3 = f"algo.Slide({w1}, {w2})"
      w5 = f"algo.SeqType(algo.AlgoDataTypeVar(), {i})"

      if dilation > 1:
        # now that the slided input (w3) has type T[Out', K', ...], we drop the
        # the dilated indices from the 2nd dimension (K') of the slided input.
        indices = range(k // dilation + 1)
        indices = (f"algo.Drop(core.ParamUse(_3), {dilation * i}, {k - 1 - dilation * i})" for i in indices)
        w4 = reduce(lambda x, y: f"algo.Concat({x}, {y})", indices)
        w3 = (f"algo.Map({{"
              f" val _3 = core.ParamDef(algo.SeqType(algo.AlgoDataTypeVar(), {k}));"
              f" algo.AlgoLambda(Seq(_3), {w4}) }}, {w3})")

      if stride > 1:
        # strides work on the 1st dimension (Out') of the slided input, which is
        # affected by the dilated kernel width.
        indices = range(1 if stride > out_width else out_width // stride)
        indices = (f"algo.Drop(_2, {stride * i}, {out_width - 1 - stride * i})" for i in indices)
        w4 = reduce(lambda x, y: f"algo.Concat({x}, {y})", indices)
        w3 = f"{{ val _2 = {w3}; {w4} }}"

      acc = lambda t: (f"algo.Map({{ val _0 = core.ParamDef({w5});"
                       f" algo.AlgoLambda(Seq(_0), {w3}) }}, {t})")

    # we sneak in a Split here (and a Join later) to simplify the mapping
    # sliding process
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
  @classmethod
  def lower(cls, input, kernel_size, stride=[], padding=0, dilation=1,
            ceil_mode=False) -> str:
    elt_ty = shir_type.get_element_type(input)
    ishape = input.meta.get("val").shape

    obj = cls(2, elt_ty, kernel_size, stride, dilation)
    return obj.emit(input.name, ishape)
