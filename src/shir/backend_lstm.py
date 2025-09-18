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

  try:
    # make sure we have everything we need...
    it = iter(params)
    _wih = next(it)
    _whh = next(it)
    if has_biases:
      _bih = next(it)
      _bhh = next(it)

    try:
      # optionally projection before the end:
      # one of these calls must throw StopIteration!
      _proj = next(it)
      next(it)
      return None
    except StopIteration:
      pass
  except:
    return None

  # initial hx must be all zeros
  # TODO: could be relaxed in some cases
  for t in hx:
    if t.op != "call_function" or t.target != torch.ops.aten.zeros.default:
      return None

  return n_lstm, [n_getitem, n_slice_all]

def fetch_tensor(gm: fx.GraphModule, n: fx.Node):
  if n is None or n.op != "get_attr":
    return None

  mod = gm
  for atom in n.target.split('.'):
    mod = getattr(mod, atom)
  return mod

def isel(gm: fx.GraphModule):
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

      it = iter(n_params)
      weight_ih = fetch_tensor(gm, next(it))
      weight_hh = fetch_tensor(gm, next(it))

      if not n_lstm.args[3]:
        qbias = torch.zeros(weight_ih.shape[0], dtype=torch.int16)
      else:
        bih = fetch_tensor(gm, next(it))
        bhh = fetch_tensor(gm, next(it))
        qbias = torch.round((bih + bhh) * (2**16)).to(torch.int16)

      proj = fetch_tensor(gm, next(it, None))

      qweight_ih = torch.round(weight_ih * (2**16)).to(torch.int16)
      qweight_hh = torch.round(weight_hh * (2**16)).to(torch.int16)

      # PyTorch stacks all the biases and weights together, but our template
      # prefers having the Wii, Wif, ... separate
      hidden_units = weight_ih.shape[0] // 4

      a_qbs = []
      for i in range(0, 4):
        a = create_new_param()
        setattr(gm, a, nn.Parameter(qbias[i * hidden_units:(i + 1) * hidden_units], False))
        a_qbs.append(a)

      a_qihs = []
      for i in range(0, 4):
        a = create_new_param()
        setattr(gm, a, nn.Parameter(qweight_ih[i * hidden_units:(i + 1) * hidden_units], False))
        a_qihs.append(a)

      a_qhhs = []
      for i in range(0, 4):
        a = create_new_param()
        setattr(gm, a, nn.Parameter(qweight_hh[i * hidden_units:(i + 1) * hidden_units], False))
        a_qhhs.append(a)

      a_proj = None
      if proj is not None:
        qproj = torch.round(proj * (2**16)).to(torch.int16)
        a_proj = create_new_param()
        setattr(gm, a_proj, nn.Parameter(qproj, False))

      with graph.inserting_before(n):
        n1 = graph.call_function(operator.mul, (n_image, 2**16))
        n2 = graph.call_function(torch.round, (n1,))
        n_qimage = graph.call_method("to", (n2, torch.int16))

        n_qihs = []
        for a in a_qihs:
          n_qihs.append(graph.get_attr(a))

        n_qhhs = []
        for a in a_qhhs:
          n_qhhs.append(graph.get_attr(a))

        n_qbs = []
        for a in a_qbs:
          n_qbs.append(graph.get_attr(a))

        n_proj = None
        if a_proj is not None:
          n_proj = graph.get_attr(a_proj)

        n_res = graph.call_function(torch.ops.shir_intrinsic.lstm.default, (n_qimage, n_qihs, n_qhhs, n_qbs, n_proj))
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

  graph.lint()
  gm.recompile()

def simpl(gm: fx.GraphModule):
  graph = gm.graph
  for n in graph.nodes:
    if n.op != "call_function":
      # assume not interesting, ignore
      continue

    if (n.target == torch.ops.shir_intrinsic.lstm.default and
        fetch_tensor(gm, n.args[1][0]) is not None and
        fetch_tensor(gm, n.args[1][0]).shape[1] == 1):
      # if the input size of the LSTM is small, then we want to pack it to
      # avoid slow reads.
      #
      # for example, if the input size is 1, hidden size if 32, then that means
      # we expect the layout to be [32 x 1], meaning each cacheline only has one
      # entry. very wasteful indeed!
      #
      # a better shape would be [1 x 32], which means a single cacheline contains
      # more than one entry (of course, it depends on the data type, but that it).
      #
      # TODO: use a better heuristic instead of hardcoding the input size to 1
      # FIXME: need to be careful when updating these tensors in-place...
      #  -  there might be more than one user
      #  -  the attribute names might be hierarchial (assume flat)
      ihs = list(n.args[1])
      for w in ihs:
        attr = w.target
        with graph.inserting_before(w):
          t = fetch_tensor(gm, w)
          setattr(gm, attr, nn.Parameter(t.view(1, -1), False))
          p = graph.get_attr(attr)
          r = graph.call_function(torch.ops.aten.view.default, (p, t.shape))
          w.replace_all_uses_with(r, propagate_meta=True)
          graph.erase_node(w)

  graph.lint()
  gm.recompile()

def compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
  from shir import backend

  mode = FakeTensorMode(allow_non_fake_inputs=True)
  FakeTensorProp(gm, mode).propagate(*example_inputs)

  isel(gm)
  FakeTensorProp(gm, mode).propagate(*example_inputs)
  gm.print_readable()

  simpl(gm)
  FakeTensorProp(gm, mode).propagate(*example_inputs)

  supported_ops = backend.SHIROperatorSupport()
  partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
  partitions = partitioner.propose_partitions()
  fused_graph = partitioner.fuse_partitions(partitions)

  fused_graph.print_readable()
  backend.apply_shir_ops(fused_graph)

  return fused_graph.forward

