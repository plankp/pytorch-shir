from typing import List, Callable
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

def match_lstm(n: fx.Node):
  import operator

  # only allow the last timestep
  n_get_last_timestep = n
  if (n.op != "call_function" or n.target != torch.ops.aten.select.int or
      n.args[1] != 1 or n.args[2] != -1):
    return None

  # ignore the full batch slicing if it exists
  n = n.args[0]
  n_slice_all = None
  if (n.op == "call_function" and n.target == torch.ops.aten.slice.Tensor and
      n.args[1] == 0 and n.args[2] == 0 and n.args[3] == 9223372036854775807 and
      len(n.users) == 1):
    n_slice_all, n = n, n.args[0]

  # only allow getting the output, not the hidden state
  n_getitem = n
  if (n.op != "call_function" or n.target != operator.getitem or
      n.args[1] != 0 or
      len(n.users) != 1):
    return None

  # then we have the lstm node
  n_lstm = n = n.args[0]
  if (n.op != "call_function" or n.target != torch.ops.aten.lstm.input or
      len(n.users) != 1):
    return None

  # examine the arguments
  img, hx, params, has_biases, num_layers, _dropout, train, bidi, batchfirst = n_lstm.args

  if train or bidi or num_layers != 1:
    return None
  if len(params) != (4 if has_biases else 2):
    return None

  # initial hx must be all zeros
  # TODO: could be relaxed in some cases
  for t in hx:
    if t.op != "call_function" or t.target != torch.ops.aten.zeros.default:
      return None

  return n_lstm, [n_getitem, n_slice_all]

def fetch_tensor(gm: fx.GraphModule, n: fx.Node):
  if n.op != "get_attr":
    return None

  mod = gm
  for atom in n.target.split('.'):
    mod = getattr(mod, atom)
  return mod

def transform(gm: fx.GraphModule):
  import operator

  counter = 0

  def create_new_param():
    nonlocal counter, gm

    counter += 1
    name = f"_isel_param{counter}"
    assert not hasattr(gm, name)

    gm.register_parameter(name, None)
    return name

  graph = gm.graph
  for n in reversed(graph.nodes):
    if p := match_lstm(n):
      n_lstm, n_rest = p
      n_image = n_lstm.args[0]
      n_zeros = n_lstm.args[1]
      n_params = n_lstm.args[2]

      weight_ih = fetch_tensor(gm, n_params[0])
      weight_hh = fetch_tensor(gm, n_params[1])
      
      if not n_lstm.args[3]:
        qbias = torch.zeros(weight_ih.shape[0], dtype=torch.int16)
      else:
        bias = fetch_tensor(gm, n_params[2]) + fetch_tensor(gm, n_params[3])
        qbias = torch.round(bias * (2**16)).to(torch.int16)

      a_qb = create_new_param()
      setattr(gm, a_qb, nn.Parameter(qbias, False))

      a_qih = create_new_param()
      qweight_ih = torch.round(weight_ih * (2**16)).to(torch.int16)
      setattr(gm, a_qih, nn.Parameter(qweight_ih, False))

      a_qhh = create_new_param()
      qweight_hh = torch.round(weight_hh * (2**16)).to(torch.int16)
      setattr(gm, a_qhh, nn.Parameter(qweight_hh, False))

      with graph.inserting_before(n):
        n1 = graph.call_function(operator.mul, (n_image, 2**16))
        n2 = graph.call_function(torch.round, (n1,))
        n_qimage = graph.call_method("to", (n2, torch.int16))
        n_qih = graph.get_attr(a_qih)
        n_qhh = graph.get_attr(a_qhh)
        n_qb  = graph.get_attr(a_qb)
        n_res = graph.call_function(torch.ops.shir_intrinsic.lstm, (n_qimage, n_qih, n_qhh, n_qb))
        n_dq = graph.call_method("to", (n_res, torch.float))
      n.target = operator.truediv
      n.args = (n_dq, 2**16)

      for q in reversed(n_rest):
        graph.erase_node(q)
      graph.erase_node(n_lstm)
      for p in n_zeros:
        graph.erase_node(p)
      for p in n_params:
        graph.erase_node(p)

def compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
  from shir import backend

  mode = FakeTensorMode(allow_non_fake_inputs=True)
  FakeTensorProp(gm, mode).propagate(*example_inputs)

  transform(gm)
  FakeTensorProp(gm, mode).propagate(*example_inputs)
  gm.print_readable()

  supported_ops = backend.SHIROperatorSupport()
  partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
  partitions = partitioner.propose_partitions()
  fused_graph = partitioner.fuse_partitions(partitions)

  fused_graph.print_readable()
  backend.apply_shir_ops(fused_graph)

  return fused_graph.forward

