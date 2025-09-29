from typing import List, Callable, Tuple, Optional
import torch
import torch.nn as nn
import torch.fx as fx
import operator
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

# Configuration values
mvm_frac: Optional[Tuple[int, int]] = None
sparsity: Optional[float] = None
_assume_qinput: bool = False
_assume_qoutput: bool = False

def match_rnn(n: fx.Node):
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

  # then we have the rnn node
  n_rnn = n = n.args[0]
  if n.op != "call_function" or len(n.users) != 1:
    return None
  if n.target not in {torch.ops.aten.rnn_tanh.input,
                      torch.ops.aten.rnn_relu.input}:
    return None

  # examine the arguments
  img, hx, params, has_biases, num_layers, _dropout, train, bidi, batchfirst = n_rnn.args

  if train or bidi or num_layers != 1:
    return None

  if len(params) != (4 if has_biases else 2):
    return None

  # initial hx must be all zeros
  # TODO: could be relaxed in some cases
  if hx.op != "call_function" or hx.target != torch.ops.aten.zeros.default:
    return None

  return n_rnn, [n_getitem, n_slice_all]

def match_lstm(n: fx.Node):
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

        n_res = graph.call_function(torch.ops.shir_intrinsic.lstm.default, (n_qimage, n_qihs, n_qhhs, n_qbs, n_proj, mvm_frac, sparsity))
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

    elif p := match_rnn(n):
      n_rnn, n_rest = p
      n_image = n_rnn.args[0]
      n_zeros = n_rnn.args[1]
      n_params = n_rnn.args[2]

      weight_ih = fetch_tensor(gm, n_params[0])
      weight_hh = fetch_tensor(gm, n_params[1])

      if n_rnn.target == torch.ops.aten.rnn_tanh.input:
        nonlinearity = True
      elif n_rnn.target == torch.ops.aten.rnn_relu.input:
        nonlinearity = False

      if not n_rnn.args[3]:
        qbias = torch.zeros(weight_ih.shape[0], dtype=torch.int16)
      else:
        bih = fetch_tensor(gm, n_params[2])
        bhh = fetch_tensor(gm, n_params[3])
        qbias = torch.round((bih + bhh) * (2**16)).to(torch.int16)

      qweight_ih = torch.round(weight_ih * (2**16)).to(torch.int16)
      qweight_hh = torch.round(weight_hh * (2**16)).to(torch.int16)

      a_qb = create_new_param()
      setattr(gm, a_qb, nn.Parameter(qbias, False))

      a_qhh = create_new_param()
      setattr(gm, a_qhh, nn.Parameter(qweight_hh, False))

      a_qih = create_new_param()
      setattr(gm, a_qih, nn.Parameter(qweight_ih, False))

      with graph.inserting_before(n):
        n1 = graph.call_function(operator.mul, (n_image, 2**16))
        n2 = graph.call_function(torch.round, (n1,))
        n_qimage = graph.call_method("to", (n2, torch.int16))

        n_qih = graph.get_attr(a_qih)
        n_qhh = graph.get_attr(a_qhh)
        n_qb  = graph.get_attr(a_qb)
        n_res = graph.call_function(torch.ops.shir_intrinsic.rnn.default, (n_qimage, n_qih, n_qhh, n_qb, nonlinearity))
        n_dq = graph.call_method("to", (n_res, torch.float))
      n.target = operator.truediv
      n.args = (n_dq, 2**16)

      for q in reversed(n_rest):
        graph.erase_node(q)
      graph.erase_node(n_rnn)
      graph.erase_node(n_zeros)
      for p in n_params:
        graph.erase_node(p)

    elif n.op == "call_function" and n.target == torch.ops.aten.linear.default:
      n_img, n_wgt, n_bias = n.args

      qweight = torch.round(fetch_tensor(gm, n_wgt) * (2**16)).to(torch.int16)
      qbias = None
      if n_bias is not None:
        qbias = torch.round(fetch_tensor(gm, n_bias) * (2**16)).to(torch.int16)

      a_qweight = create_new_param()
      setattr(gm, a_qweight, nn.Parameter(qweight, False))

      a_qbias = None
      if qbias is not None:
        a_qbias = create_new_param()
        setattr(gm, a_qbias, nn.Parameter(qbias, False))

      with graph.inserting_before(n):
        n1 = graph.call_function(operator.mul, (n_img, 2**16))
        n2 = graph.call_function(torch.round, (n1,))
        n_qimage = graph.call_method("to", (n2, torch.int16))
        n_qweight = graph.get_attr(a_qweight)
        n_qbias = None
        if a_qbias is not None:
          n_qbias = graph.get_attr(a_qbias)
        n_res = graph.call_function(torch.ops.shir_intrinsic.mm.default, (n_qimage, n_qweight, n_qbias))
        n_dq = graph.call_method("to", (n_res, torch.float))
      n.target = operator.truediv
      n.args = (n_dq, 2**16)

  graph.lint()
  gm.recompile()

def _remap_qinput(gm: fx.GraphModule):
  # to be safe (since we completely change the placeholder information),
  # construct a brand-new graph
  env = {}
  new_graph = fx.Graph()

  def mapper(n):
    return env[n]

  for n in gm.graph.nodes:
    if n.op == "placeholder":
      dt = getattr(n.meta.get("val", {}), "dtype", None)
      if dt == torch.float:
        u = new_graph.placeholder(n.target)
        a = new_graph.call_method("to", (u, torch.float))
        b = new_graph.call_function(operator.truediv, (a, 2**16))
        env[n] = b
        continue

      if dt is None:
        print("backend_lstm::_remap_qinput: warning: some inputs have no type information")

    u = new_graph.node_copy(n, mapper)
    env[n] = u

  new_graph.lint()
  gm.graph = new_graph

def _remap_qoutput(gm: fx.GraphModule):
  graph = gm.graph
  for n in reversed(graph.nodes):
    if n.op != "output":
      continue
    new_outs = []
    maybe_discard = []
    outputs = n.args[0]
    for out in outputs:
      if (out.op == "call_function" and out.target == operator.truediv and out.args[1] == 2**16 and
          (n_t := out.args[0]).op == "call_method" and n_t.target == "to" and n_t.args[1] == torch.float and
          n_t.args[0].meta.get("val").dtype == torch.int16):
        new_outs.append(n_t.args[0])
        maybe_discard.append([out, n_t])
      else:
        print("backend_lstm::simpl: warning: some outputs are not dequantized")
        new_outs.append(out)

    n.args = (tuple(new_outs),)
    for xs in maybe_discard:
      for x in xs:
        if x.users:
          break
        graph.erase_node(x)
    break

def simpl(gm: fx.GraphModule):
  graph = gm.graph
  for n in graph.nodes:
    if n.op not in CALLABLE_NODE_OPS:
      # assume not interesting, ignore
      continue

    if (n.target == "to" and n.args[1] == torch.int16 and
        (n_r := n.args[0]).op == "call_function" and n_r.target == torch.round and
        (n_m := n_r.args[0]).op == "call_function" and n_m.target == operator.mul and n_m.args[1] == 2**16 and
        (n_d := n_m.args[0]).op == "call_function" and n_d.target == operator.truediv and n_d.args[1] == 2**16 and
        (n_t := n_d.args[0]).op == "call_method" and n_t.target == "to" and n_t.args[1] == torch.float and
        n_t.args[0].meta.get("val").dtype == torch.int16):
      n.replace_all_uses_with(n_t.args[0])
      for q in [n, n_r, n_m, n_d, n_t]:
        if q.users:
          break
        graph.erase_node(q)

    elif (n.target == torch.ops.shir_intrinsic.lstm.default and
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

  gm.graph.lint()
  gm.recompile()

def peephole(gm: fx.GraphModule):
  graph = gm.graph
  for n in graph.nodes:
    if n.op not in CALLABLE_NODE_OPS:
      # assume not interesting, ignore
      continue

    if n.target == torch.ops.shir_intrinsic.mm.default:
      # that means for whatever reason, we did not lower it...
      # so turn it back to using the torch implementation
      n.target = torch.ops.aten.linear.default

def compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
  from shir import backend

  mode = FakeTensorMode(allow_non_fake_inputs=True)
  FakeTensorProp(gm, mode).propagate(*example_inputs)

  isel(gm)
  FakeTensorProp(gm, mode).propagate(*example_inputs)

  if _assume_qinput:
    _remap_qinput(gm)
    example_inputs = [i.to(torch.int16) if i.dtype == torch.float else i for i in example_inputs]
    FakeTensorProp(gm, mode).propagate(*example_inputs)

  if _assume_qoutput:
    _remap_qoutput(gm)

  simpl(gm)
  FakeTensorProp(gm, mode).propagate(*example_inputs)
  gm.print_readable()

  supported_ops = backend.SHIROperatorSupport()
  partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
  partitions = partitioner.propose_partitions()
  fused_graph = partitioner.fuse_partitions(partitions)

  peephole(fused_graph)
  fused_graph.print_readable()
  backend.apply_shir_ops(fused_graph)

  return fused_graph.forward

