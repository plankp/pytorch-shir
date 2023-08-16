"""
The entry point to the whole torch.compile business
"""

import torch
from typing import List
from torch._dynamo.backends.common import aot_module_simplified
from torch._decomp import core_aten_decompositions, get_decompositions
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from functorch.compile import make_boxed_func
from . import types, lowering, graphs, rewrites

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

      # print("\n  FUSED GRAPH\n")
      # submod.print_readable()
      gm.add_submodule(n.target, graphs.SHIRGraphModule(submod))

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
