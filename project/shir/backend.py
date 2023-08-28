"""
The entry point to the whole torch.compile business
"""

import torch
from torch.fx import Node
from torch.fx.graph_module import GraphModule
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._functorch.aot_autograd import aot_export_module
from torch._decomp import core_aten_decompositions, get_decompositions
from itertools import count
from pathlib import Path
from typing import List, Callable
from . import types, lowering, graphs, rewrites, config

# notes on decompositing core ATen to prims:
# -  not all core ATen ops have a prims decomposition (e.g. aten.mm)
# -  not all prims decompositions are "good" (e.g. aten.relu uses masking)
aten = torch.ops.aten
shir_decomps = get_decompositions([
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
decomps = {**core_aten_decompositions(), **shir_decomps}

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

def may_avoid_output_copy(original_node: Node) -> bool:
  for user in original_node.users:
    if user.op != "call_function":
      # assume definitely needs a copy
      # example: the user is an output node, which clearly, we should copy.
      return False
    if isinstance(user.target, torch._ops.OpOverload):
      # as an approximation, query the schema and see if it is marked mutable,
      # which is good enough for most cases (since they end with
      # dequantize_per_tensor, which sets the schema properly)
      return not user.target._schema.is_mutable

    # just to be safe, say the copy is needed / unavoidable
    return False

  return True

# a global counter used to (hopefully) avoid clashing module names when
# generating the SHIR projects
_instance_counter = count()

def apply_shir_ops(gm: GraphModule):
  # look for call_modules which point to a submodule containing only the
  # supported operators. we swap them out with our custom submodules depending
  # on the active configuration.
  for n in gm.graph.nodes:
    if n.op == "call_module":
      assert not n.kwargs
      submod = gm.get_submodule(n.target)

      idnum = next(_instance_counter)
      project = graphs.SHIRProject(
        f"Module{idnum}",
        Path(config.EMIT_OUTPUT_DIR) / f"module{idnum}"
      )

      # regardless of the active configuration, we always trigger compilation.
      # (we'd like to make sure code type checks and what not)
      project.prepare_directory()
      project.emit_source(submod)
      project.generate_hardware_files()

      shir_graph = None
      if config.PERFORM_SYNTHESIS:
        from . import driver
        project.synthesize()
        shir_graph = graphs.SHIRGraphFpgaModule(submod, project, driver)
      elif config.PERFORM_SIMULATION:
        shir_graph = graphs.SHIRGraphSimModule(submod, project)

      if shir_graph:
        gm.delete_submodule(n.target)
        gm.add_submodule(n.target, shir_graph)

  # now is the tricky part.
  #
  # SHIRGraphFpgaModule preallocates the inputs and outputs and expects the
  # caller (us) to pass data by copying it there, so we traverse the graph
  # again to fix this.
  #
  # since our graph now has side effects, we MUST NOT use eliminate_dead_code
  # to get rid of redundant nodes.
  for n in gm.graph.nodes:
    if n.op == "call_module":
      submod = gm.get_submodule(n.target)
      if not isinstance(submod, graphs.SHIRGraphFpgaModule):
        continue

      pending_erase = []
      args = n.args

      g = gm.graph
      with g.inserting_before(n):
        n1 = g.get_attr(n.target)
        n1_used = False
        for i, arg in enumerate(args):
          # we try to avoid the input copy by handling some common cases.
          if arg.op == "get_attr" and not graphs.has_many_uses(arg):
            # this is a weight / bias with a single use.
            # we can copy the values into the expected location in advance.
            dst = submod.get_in_tensor(i)
            dst.copy_(getattr(gm, arg.target))
            setattr(gm, arg.target, torch.nn.Parameter(dst, False))
            # at this point, the original node still "uses" this arg,
            # so delay the erase to later
            pending_erase.append(arg)

          else:
            n1_used = True
            n2 = g.call_method("get_in_tensor", (n1, i))
            n3 = g.call_method("copy_", (n2, arg))

        if not n1_used:
          g.erase_node(n1)

        n4 = g.call_module(n.target, ())

        if may_avoid_output_copy(n):
          n_result = n4
        else:
          n5 = g.call_method("clone", (n4,))
          n6 = g.call_method("detach", (n5,))
          n_result = n6

      n.replace_all_uses_with(n_result)
      g.erase_node(n)
      for n in reversed(pending_erase):
        g.erase_node(n)

  # for sanity sake, recompile it.
  gm.graph.lint()
  gm.recompile()

def compiler(gm: GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
  # first thing is to perform a set of "early" rewrites, which is just
  # quantization related rewrites at the moment.
  #
  # the reason for this is because these rewrites need access to the weights,
  # and it is much easier to do so at this stage (compared to after
  # aot-autograd).
  mode = FakeTensorMode(allow_non_fake_inputs=True)
  FakeTensorProp(gm, mode).propagate(*example_inputs)
  rewrites.rewrite_quantized_ops(gm)

  # then we hand it off to aot-autograd to perform decompositions.
  #
  # we would end up with a graph that only contains decomposed ops and a
  # signature that contains a mapping to the parameters.
  aot_gm, sig = aot_export_module(
    gm, example_inputs,
    trace_joint=False,
    decompositions=decomps,
  )
  assert len(sig.buffers_to_mutate) == 0, "Mutation is not supported at the moment"

  # which, we use that information to rematerialize the code for fetching the
  # model arguments.
  g = aot_gm.graph
  for n in g.nodes:
    if n.op == "placeholder" and (param := sig.inputs_to_parameters.get(n.name)):
      # we bring the parameter from the original graph to the new graph!
      setattr(aot_gm, param, torch.nn.Parameter(getattr(gm, param), False))
      with g.inserting_before(n):
        n1 = g.get_attr(param)
      n.replace_all_uses_with(n1, propagate_meta=True)
      g.erase_node(n)

  # with this decomposed + getattr's graph, we can carve out the bits that are
  # supported by SHIR, and perform extra rewrites as necessary.
  supported_ops = SHIROperatorSupport()
  partitioner = CapabilityBasedPartitioner(aot_gm, supported_ops, allows_single_node_partition=True)
  partitions = partitioner.propose_partitions()
  fused_graph = partitioner.fuse_partitions(partitions)
  apply_shir_ops(fused_graph)

  return fused_graph.forward
