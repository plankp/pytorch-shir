"""
some generic template for lowering
"""

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

"""
magic that actually does the lowering of each node
"""

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
