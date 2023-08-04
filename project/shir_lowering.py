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

      return (
        f"algo.Sub(algo.Tuple(algo.TruncInteger("
        f"algo.ClipBankersRound({acc}, 0, {bits - 8}),"
        f" 8), algo.ConstantInteger(0)))"
      )

    ashape = a.meta.get("val").shape
    if not ashape:
      return gen_kernel(a.name, b.name)

    w = gen_kernel("algo.Select(core.ParamUse(_0), 0)", "algo.Select(core.ParamUse(_0), 1)")
    acc = lambda t: (
      f"algo.Map({{ val _0 = core.ParamDef(algo.TupleType(algo.SignedIntType(32), algo.SignedIntType(32)));"
      f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})"
    )

    for _ in ashape[1:]:
      w = acc("algo.Zip(core.ParamUse(_0))")
      acc = lambda t: (
        f"algo.Map({{ val _0 = core.ParamDef();"
        f" algo.AlgoLambda(Seq(_0), {w}) }}, {t})"
      )

    return acc(f"algo.Zip(algo.Tuple({a.name}, {b.name}))")

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
