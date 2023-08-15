"""
The entry point to the whole torch.compile business
"""

import torch
from typing import Tuple, List, Optional
from torch._dynamo.backends.common import aot_module_simplified
from torch._decomp import core_aten_decompositions, get_decompositions
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from functorch.compile import make_boxed_func
from . import lowering, types, rewrites
from functools import reduce
from itertools import count
import os
import shutil

# some debug flags and configuration stuff
_CONF_EMIT_SHIR_CODE = True
_CONF_EMIT_DATA_FILES = True
_CONF_EMIT_OUTPUT_DIR = "./data/generated"
_CONF_TEMPLATE_DIR = "./template"

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

def _tensor_to_matrix_csv(t: torch.Tensor, f):
  # XXX: decoding int64's is a bit annoying, ignore it!
  assert t.dtype in {torch.int8, torch.uint8, torch.int16, torch.int32}

  if t.ndim == 0:
    t = torch.unsqueeze(t, 0)
  if t.ndim == 1:
    t = torch.unsqueeze(t, 0)
  if t.ndim > 2:
    t = torch.flatten(t, 1)

  for row in t:
    d = row.shape[0]
    for i in range(d - 1):
      print(row[i].item(), ",", sep="", end="", file=f)
    print(row[d - 1].item(), file=f)

def _collect_inout_nodes(gm: torch.fx.GraphModule) -> Tuple[List[str], torch.fx.Node]:
  placeholders = []
  output = None
  for n in gm.graph.nodes:
    if n.op == "placeholder":
      tinfo = n.meta.get("val")
      assert tinfo is not None, "Placeholder must be a tensor"
      assert all((isinstance(d, int) for d in tinfo.shape)), "Dynamic shapes are not supported"

      placeholders.append(n.target)
    elif n.op == "output":
      assert len(n.args) == 1, "Only single output node is supported"
      node = n.args[0]
      tinfo = node.meta.get("val")
      assert tinfo is not None, "Output must be a tensor"
      assert all((isinstance(d, int) for d in tinfo.shape)), "Dynamic shapes are not supported"

      if output is None:
        output = node
      assert output == node, "Two output nodes returning different values"
  return (placeholders, output)

def _has_many_uses(node: torch.fx.Node) -> bool:
  user_count = len(node.users)
  if user_count > 1:
    return True

  used = False
  for user in node.users:
    # each user may have multiple occurrences / uses of a single node
    for n in user.all_input_nodes:
      if n != node:
        continue
      if used:
        return True
      used = True

  return False

class SHIRGraphModule(torch.nn.Module):
  _iid_couter = count()   # assigns a number of each instance

  # assumption: graph does not change (it really shouldn't!)
  gm: torch.fx.GraphModule
  _inst_id: int
  _call_id: int
  _compiled: bool
  _inout_nodes: Optional[Tuple[List[str], torch.fx.Node]]

  def __init__(self, gm: torch.fx.GraphModule):
    super().__init__()
    self.gm = gm
    self._compiled = False
    self._inst_id = next(self._iid_couter)
    self._call_id = 0
    self._inout_nodes = None

    self._output_dir = os.path.join(_CONF_EMIT_OUTPUT_DIR, f"module{self._inst_id}")
    # print("\n  FUSED GRAPH\n")
    # self.gm.print_readable()

  def __call__(self, *args):
    self._call_id += 1
    self.compile()

    result = self.gm(*args)

    if _CONF_EMIT_DATA_FILES:
      data_dir = os.path.join(self._output_dir, f"data_{self._call_id}")
      if not os.path.exists(data_dir):
        os.mkdir(data_dir)

      for i, arg in enumerate(args):
        with open(
          os.path.join(data_dir, f"arg{i}.csv"), "w", encoding="utf-8"
        ) as argf:
          _tensor_to_matrix_csv(arg, argf)

      with open(
        os.path.join(data_dir, "result.csv"), "w", encoding="utf-8"
      ) as resf:
        _tensor_to_matrix_csv(result, resf)

    return result

  def compile(self):
    if self._compiled:
      return

    self._compiled = True
    if _CONF_EMIT_SHIR_CODE or _CONF_EMIT_DATA_FILES:
      self._prepare_directory()
    if _CONF_EMIT_SHIR_CODE:
      self._emit_source()

  def _prepare_directory(self):
    # purge the output directory
    if os.path.exists(self._output_dir):
      shutil.rmtree(self._output_dir)

    # copy the template directory into the output directory
    shutil.copytree(_CONF_TEMPLATE_DIR, self._output_dir)

  def _emit_source(self):
    self._inout_nodes = _collect_inout_nodes(self.gm)
    clname = f"Module{self._inst_id}"
    with open(os.path.join(
      self._output_dir, "src", "main", "scala",
      f"{clname}.scala"
    ), "w", encoding="utf-8") as f:
      print("// This file is autogenerated", file=f)
      print("import core._", file=f)
      print("import algo._", file=f)
      print("import java.nio.file.Paths", file=f)
      print(file=f)

      print("object", clname, " extends GeneratedModel {", file=f)
      print(file=f)

      print("  val name: String = \"", clname, "\"", sep="", file=f)
      print(file=f)

      print("  def main(args: Array[String]): Unit = {", file=f)
      print("    Util.drive(this, args)", file=f)
      print("  }", file=f)
      print(file=f)

      print("  def generateIR(): Expr = {", file=f)
      self._emit_body(f)
      print("  }", file=f)
      print(file=f)

      print("  def loadData(folder: String): Predef.Map[String, Seq[Seq[Int]]] = Predef.Map(", file=f)
      for i, arg in enumerate(self._inout_nodes[0]):
        print("    \"", arg, "\" -> Util.readIntCSV(Paths.get(folder, \"arg", i, ".csv\").toFile()),", sep="", file=f)
      print("    \"result\" -> Util.readIntCSV(Paths.get(folder, \"result.csv\").toFile())", file=f)
      print("  )", file=f)
      print(file=f)
      print("}", file=f)

  def _emit_body(self, f):
    lets_needed = 0
    for n in self.gm.graph.nodes:
      # assume every node that has many uses needs to be let-bound,
      # which is definitely the case for tensors (which are SeqType's)
      has_many_uses = _has_many_uses(n)
      if has_many_uses:
        lets_needed += 1

      # furthermore, input (placeholder) and output nodes must be 2D in SHIR.
      match n.op:
        case "placeholder":
          typ = types.get_element_type(n)
          shape = ", ".join((str(d) for d in n.meta.get("val").shape))

          if has_many_uses:
            print(
              "  { val _init = core.TypeChecker.check(algo.torch.TInput(",
              typ.name(), ", \"", n.target, "\", Seq(", shape, ")))\n",
              "    val _param = core.ParamDef(_init.t)\n",
              "    core.Let(_param,\n",
              "  { val ", a.name, " = core.ParamUse(_param)",
              sep="", file=f
            )
          else:
            print(
              "    val ", n.name,
              " = core.TypeChecker.check(algo.torch.TInput(",
              typ.name(), ", \"", n.target, "\", Seq(", shape, ")))",
              sep="", file=f
            )

        case "output":
          # sometimes, due to unfortunate graph slicing, we may end up with
          # multiple outputs, which we cannot handle
          [retv] = n.args
          assert isinstance(retv, torch.fx.Node), "Only single node output is allowed"
          assert not has_many_uses  # not sure what this failing would mean...

          shape = retv.meta.get("val").shape
          dims = len(shape)
          v = retv.name

          if dims < 1:
            v = f"algo.Repeat({v}, 1)"
          if dims < 2:
            v = f"algo.Repeat({v}, 1)"
          if dims > 2:
            v = f"algo.torch.TFlatten({v}, 1, {dims - 1})"
          print("    core.TypeChecker.check(", v, ")", sep="", file=f)

        case "call_function":
          obj = lowering.fetch_lowering(n.target)
          expr = obj.lower(*n.args, **n.kwargs)
          if has_many_uses:
            print(
              "    val _init = core.TypeChecker.check(", expr, ")\n",
              "    val _param = core.ParamDef(_init.t)\n",
              "    core.Let(_param,\n",
              "  { val ", n.name, " = core.ParamUse(_param)",
              sep="", file=f
            )
          else:
            print(
              "    val ", n.name, " = core.TypeChecker.check(", expr, ")",
              sep="", file=f
            )

        case _:
          assert False, "Unhandled fx node type when emitting"

    for _ in range(lets_needed):
      print("  }, _init)", file=f)

class SHIROperatorSupport(OperatorSupport):
  def is_node_supported(self, submodules, n: torch.fx.Node) -> bool:
    if n.op not in CALLABLE_NODE_OPS:
      return False

    try:
      # clearly if shir can't represent this type, then we can't process it.
      if not types.has_shir_type(n):
        return False

      obj = lowering.fetch_lowering(n.target)
      if obj is None:
        return False

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

def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
  # raw ops -> quantized rewrite -> core aten + prims
  #   -> partition -> shir

  def lowering(gm, example_inputs):
    # gm.print_readable()

    supported_ops = SHIROperatorSupport()
    partitioner = CapabilityBasedPartitioner(gm, supported_ops,
                                             allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)
    apply_shir_ops(fused_graph)

    return make_boxed_func(fused_graph.forward)

  def prelowering(gm, example_inputs):
    rewrites.rewrite_late(gm)
    f = aot_module_simplified(gm, example_inputs,
                              decompositions=_decomps,
                              fw_compiler=lowering)
    return make_boxed_func(f)

  rewrites.rewrite_quantized_ops(gm)

  f = aot_module_simplified(gm, example_inputs,
                            decompositions=core_aten_decompositions(),
                            fw_compiler=prelowering)
  return f
