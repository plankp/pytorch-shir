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
from . import types, lowering, graphs, rewrites, config, bit_utils

# some aliases
aten = torch.ops.aten
qd = torch.ops.quantized_decomposed

# notes on decompositing core ATen to prims:
# -  not all core ATen ops have a prims decomposition (e.g. aten.mm)
# -  not all prims decompositions are "good" (e.g. aten.relu uses masking)
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

      # at this point, all outer values are passed into the inner graph as
      # arguments / placeholders. weights and biases are known, and can be
      # recovered by looking for get_attr nodes.
      #
      # we look for them and calculate the narrowest possible SHIR type for
      # these values.
      #
      # use array instead of map due to lack of sparsity for quantized models.
      arg_elt_types = [None] * len(n.args)
      for i, arg in enumerate(n.args):
        if config.TRY_NARROW_TYPE and arg.op == "get_attr":
          torch_ty = types.get_element_type(arg)
          real_ty = bit_utils.get_narrow_type(getattr(gm, arg.target))

          # Avoid differing signedness for now.
          # (can only happen if the type was signed but tensor was all positive.)
          if isinstance(torch_ty, types.SI) and isinstance(real_ty, types.UI):
            real_ty = real_ty.to_signed()
          arg_elt_types[i] = real_ty

      # then trigger compilation regardless of active configuration using the
      # bitwidth information
      project.prepare_directory()
      project.emit_source(submod, arg_elt_types)
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
  # both SHIR graphs may be expecting inputs of reduced bitwidth (see above).
  #
  # for SHIRGraphSimModule, this is (presently) not a problem since it always
  # passes values by serializing the tensors into csv's, and the
  # deserialization business is done in layout.py.
  #
  # for SHIRGraphFpgaModule, however, we have two problems:
  # 1.  it preallocates memory for the input and output, so we need to make
  #     the values go to where it expects it to be.
  # 2.  PyTorch does not support variable bitwidth values, so naive copy is
  #     not going to work.
  #
  # note that thankfully, the 2nd case only happens to weights and bias, so
  # we only do the tricky copy once.
  #
  # with this copying, our graph is no longer side effect free, so DO NOT use
  # eliminate_dead_code to get rid of redundant nodes.
  #
  # also rewrite some operations. e.g. qd ops because they are slow...
  for n in gm.graph.nodes:
    if n.op == "call_function" and (
      n.target == qd.quantize_per_tensor.default and
      n.args[3] == -128 and n.args[4] == 127 and n.args[5] == torch.int8
    ):
      g = gm.graph
      with g.inserting_before(n):
        n1 = g.call_function(torch.quantize_per_tensor, (n.args[0], n.args[1], n.args[2]), {"dtype": torch.qint8})
        n2 = g.call_method("int_repr", (n1,))
      n.replace_all_uses_with(n2)
      g.erase_node(n)

    elif n.op == "call_function" and (
      n.target == qd.dequantize_per_tensor.default and
      n.args[3] == -128 and n.args[4] == 127 and n.args[5] == torch.int8
    ):
      g = gm.graph
      with g.inserting_before(n):
        n1 = g.call_function(torch._make_per_tensor_quantized_tensor, (n.args[0], n.args[1], n.args[2]))
        n2 = g.call_method("dequantize", (n1,))
      n.replace_all_uses_with(n2)
      g.erase_node(n)

    elif n.op == "call_module":
      submod = gm.get_submodule(n.target)
      if not isinstance(submod, graphs.SHIRGraphFpgaModule):
        continue

      args = n.args
      g = gm.graph
      with g.inserting_before(n):
        n1 = g.get_attr(n.target)
        n1_used = False
        for i, arg in enumerate(args):
          # we try to avoid the input copy by handling some common cases.
          if config.TRY_COPY_AOT and arg.op == "get_attr":
            # this is a weight / bias. we can copy the values into the shared
            # memory buffers AOT.
            dst = submod.get_in_tensor(i)
            src = getattr(gm, arg.target)
            if dst is not None:
              dst.copy_(src)
            else:
              # reaching here means the weights have reduced to a bitwidth that is
              # not supported by PyTorch. example: int32 bias reduced as s20.
              entry = submod._layout.get_entry(f"arg{i}")
              entry.to_buffer(submod._buffer, src)

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
    if n.op != "placeholder":
      continue

    if (name := sig.inputs_to_parameters.get(n.name)) is not None:
      aot_gm.register_parameter(name, torch.nn.Parameter(getattr(gm, name), False))
    elif (name := sig.inputs_to_buffers.get(n.name)) is not None:
      aot_gm.register_buffer(name, getattr(gm, name))

    if name is not None:
      with g.inserting_before(n):
        n1 = g.get_attr(name)
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
