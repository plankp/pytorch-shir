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
from typing import List, Callable
from . import types, lowering, graphs, rewrites

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

def apply_shir_ops(gm: GraphModule):
  # look for call_modules which point to a submodule containing only the
  # supported operators. swap those out with our own SHIRGraphModule and
  # trigger compilation.
  for n in gm.graph.nodes:
    if n.op == "call_module":
      assert not n.kwargs
      submod = gm.get_submodule(n.target)
      # print("\n  FUSED GRAPH\n")
      # submod.print_readable()

      shir_graph = graphs.SHIRGraphModule(submod)
      shir_graph.compile()  # trigger compilation
      gm.delete_submodule(n.target)
      gm.add_submodule(n.target, shir_graph)

  # but recall that SHIRGraphModule places inputs and outputs in special
  # preallocated buffers, so traverse the graph again to fix this.
  #
  # since our graph now has side effects, we MUST clean up redundant nodes
  # by ourselves. (eliminate_dead_code removes too much)
  for n in gm.graph.nodes:
    if n.op == "call_module":
      pending_erase = []
      submod = gm.get_submodule(n.target)
      args = n.args

      g = gm.graph
      with g.inserting_before(n):
        n1 = g.get_attr(n.target)
        n1_used = False
        for i, arg in enumerate(args):
          # we would like to avoid copying the input.
          # it is difficult, and so, we only look for some common cases.
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
  # we ideally would use something like aot_autograd or aot_module_simplified
  # or the like to decompose into an even smaller instruction set.
  # unfortunately, doing so loses too much information.
  #
  # the big problem is that, in order to play nicely with the FPGA, the inputs
  # and outputs has to located on some page of host RAM. at last, PyTorch does
  # not seem to offer guarantees on memory location, hence often times we need
  # to copy.
  #
  # yet, for a good amount of data like weights and biases, the copy is not
  # needed. likewise, depending on the situation, it is possible to avoid the
  # copy for inputs and outputs.
  #
  # by using aot_autograd and friends, you lose access to these values (which
  # is also the reason why rewrite_quantized_ops had to happen super early.)

  mode = FakeTensorMode(allow_non_fake_inputs=True)
  FakeTensorProp(gm, mode).propagate(*example_inputs)

  rewrites.rewrite_quantized_ops(gm)

  FakeTensorProp(gm, mode).propagate(*example_inputs)

  supported_ops = SHIROperatorSupport()
  partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
  partitions = partitioner.propose_partitions()
  fused_graph = partitioner.fuse_partitions(partitions)
  apply_shir_ops(fused_graph)

  return fused_graph.forward
