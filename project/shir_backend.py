import torch

# Note: torch._dynamo.optimizations.training.aot_autograd got moved
from torch._dynamo.backends.common import (
  aot_module_simplified,
  fake_tensor_unsupported,  #  seems like it's not needed
)
from torch._decomp import core_aten_decompositions, get_decompositions
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from dataclasses import dataclass
from functorch.compile import make_boxed_func
import rewrite_pattern
import shir_type
import shir_lowering
import monkey_patcher
from functools import reduce

# some namespace aliases
aten = torch.ops.aten
prims = torch.ops.prims

# notes on decompositing core ATen to prims:
# -  not all core ATen ops have a prims decomposition (e.g. aten.mm)
# -  not all prims decompositions are "good" (e.g. aten.relu uses masking)
_decomps = get_decompositions([
  aten._to_copy,
  aten.sym_size,
  aten.add,
  aten.rsub,
  aten.sub,
  aten.mul,
  aten.dot,
  aten.sum,
  aten.where,
  aten.maximum,
  aten.squeeze,
  aten.expand.default,
  aten.permute.default,
  aten.unsqueeze.default,
])

class SHIRGraphModule(torch.nn.Module):
  gm: torch.fx.GraphModule

  def __init__(self, gm: torch.fx.GraphModule):
    super(SHIRGraphModule, self).__init__()
    self.gm = gm
    self.emitted = False

  def __call__(self, *args):
    if not self.emitted:
      print(self.emit())
      self.emitted = True

    with monkey_patcher.Patcher():
      return self.gm(*args)

  def emit(self):
    buffer = \
"""// macros...
def _idotp(ty: IntTypeT, x: Expr, y: Expr, acc: Option[Expr]=None): Expr = {
  val e1 = algo.TruncInteger(algo.Fold(algo.Add2.asFunction(),
    algo.Map({
      val _0 = core.ParamDef()
      algo.AlgoLambda(Seq(_0), algo.TruncInteger(algo.Mul(core.ParamUse(_0)), ty.width))
    }, algo.Zip(algo.Tuple(x, y)))), ty.width)
  val e2 = acc match {
    case None => e1
    case Some(e) => algo.TruncInteger(algo.Add2(e1, algo.TruncInteger(e, ty.width)), ty.width)
  }
  ty match {
    case SignedIntType(_) => algo.Sub(algo.Tuple(e2, algo.ConstantInteger(0)))
    case IntType(_) => e2
    case _ => ???
  }
}

def _iredmax(ty: IntTypeT, x: Expr): Expr = {
  def signconv(e: Expr): Expr = ty match {
    case SignedIntType(_) => algo.Sub(algo.Tuple(e, algo.ConstantInteger(0)))
    case IntType(_) => e
    case _ => ???
  }
  signconv(algo.Fold({
    val _0 = core.ParamDef()
    val _1 = core.ParamDef()
    algo.AlgoLambda(Seq(_0, _1),
      algo.Max2(
        signconv(TruncInteger(ParamUse(_0), ty.width)),
        signconv(TruncInteger(ParamUse(_1), ty.width))))
  }, x))
}

// actual emitted stuff starts here...
"""

    for n in self.gm.graph.nodes:
      # input (placeholder) and output nodes must be 2D in SHIR.
      match n.op:
        case "placeholder":
          typ = shir_type.get_element_type(n)
          shape = n.meta.get("val").shape
          dims = len(shape)

          outer = inner = 1
          if dims == 1:
            inner = shape[0]
          elif dims == 2:
            [outer, inner] = shape
          elif dims > 2:
            outer = shape[0]
            inner = reduce(lambda x, y: x * y, shape[1:])

          v = f"algo.Input(\"{n.target}\", {typ.name()}, {inner}, {outer})"
          if dims != 2:
            v = f"algo.Join({v})"
          if dims < 1:
            v = f"core.Conversion({v}, {typ.name()})"
          if dims > 2:
            # the tuple splat usage here is safe because shape always has at
            # least three elements, so the trailing comma case never happens.
            v = f"algo.Join(algo.SplitAll({v}, Seq{(*shape,)}))"

          buffer += f"val {n.name} = core.TypeChecker.check({v})\n"

        case "output":
          [retv] = n.args
          assert isinstance(retv, torch.fx.Node), "Only single fx node output is allowed"

          shape = retv.meta.get("val").shape
          dims = len(shape)
          v = retv.name

          if dims < 1:
            v = f"algo.Repeat({v}, 1)"
          if dims < 2:
            v = f"algo.Repeat({v}, 1)"
          if dims > 2:
            # avoid using Split(X, shape[0], false) since it causes problems
            inner_sz = reduce(lambda x, y: x * y, shape[1:])
            v = f"algo.Split(algo.JoinAll({v}), {inner_sz})"

          buffer += f"return core.TypeChecker.check({v})"

        case "call_function":
          obj = shir_lowering.fetch_lowering(n.target)
          v = obj.lower(*n.args, **n.kwargs)
          buffer += f"val {n.name} = core.TypeChecker.check({v})\n"

        case _:
          assert False, "Unhandled fx node type when emitting"

    return buffer

class SHIROperatorSupport(OperatorSupport):
  def is_node_supported(self, submodules, n: torch.fx.Node) -> bool:
    if n.op not in CALLABLE_NODE_OPS:
      return False

    try:
      # clearly if shir can't represent this type, then we can't process it.
      # e.g. max_pool2d returns a tuple?
      if not shir_type.has_shir_type(n):
        return False

      obj = shir_lowering.fetch_lowering(n.target)
      return obj.supports(*n.args, **n.kwargs)
    except:
      return False

def apply_shir_ops(gm: torch.fx.GraphModule):
  # the supported operators would be punted off into submodules,
  # so only look for those and compile those
  for n in gm.graph.nodes:
    if n.op == "call_module":
      assert not n.kwargs
      submod = gm.get_submodule(n.target)
      gm.delete_submodule(n.target)
      gm.add_submodule(n.target, SHIRGraphModule(submod))

def compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
  # raw ops -> core aten -> rewrite ->
  #   prims -> late rewrite -> partition -> shir-ify

  def phase_partition(gm, example_inputs):
    rewrite_pattern.late_rewrite(gm)
    FakeTensorProp(gm).propagate(*example_inputs) # for shape info
    # gm.print_readable()
    supported_ops = SHIROperatorSupport()
    partitioner = CapabilityBasedPartitioner(gm, supported_ops,
                                             allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)
    apply_shir_ops(fused_graph)

    return make_boxed_func(fused_graph.forward)

  def phase_rewrite(gm, example_inputs):
    rewrite_pattern.early_rewrite(gm)
    # gm.print_readable()
    rewrite_pattern.rewrite(gm)
    f = aot_module_simplified(gm, example_inputs,
                              decompositions=_decomps,
                              fw_compiler=phase_partition)
    return make_boxed_func(f)

  f = aot_module_simplified(gm, example_inputs,
                            decompositions=core_aten_decompositions(),
                            fw_compiler=phase_rewrite)
  return f
