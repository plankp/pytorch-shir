from typing import Optional, Tuple
from torch.fx.graph_module import GraphModule
from torch.fx import Node
import torch
import operator
from torch.ao.quantization.pt2e.utils import (
  _get_all_arguments,
)
from torch._dynamo.source import (
  NNModuleSource,
  LocalSource,
  AttrSource,
)
from . import functional

# don't match or emit prims ops at this level!
aten = torch.ops.aten
shin = torch.ops.shir_intrinsic
qd = torch.ops.quantized_decomposed

class QuantOpRewrite:
  def __init__(self, gm: GraphModule):
    self.counter = -1
    self.gm = gm

  def _rewrite_node(self, n: Node) -> bool:
    if self._rewrite_qconv_per_channel(n):
      return True
    if self._rewrite_qconv(n):
      return True
    if self._rewrite_qlinear(n):
      return True
    if self._rewrite_qmaxpool(n):
      return True
    if self._rewrite_qavgpool(n):
      return True
    if self._rewrite_qmean(n):
      return True
    if self._rewrite_flatten(n):
      return True
    if self._rewrite_hardtanh(n):
      return True
    if self._rewrite_hardsigmoid(n):
      return True
    if self._rewrite_hardswish(n):
      return True
    if self._rewrite_add(n):
      return True
    if self._rewrite_mul(n):
      return True

    return False

  def rewrite(self):
    changed = False
    for n in self.gm.graph.nodes:
      changed |= self._rewrite_node(n)

    if changed:
      self.gm.graph.eliminate_dead_code()
      self.gm.graph.lint()
      self.gm.recompile()

  def create_new_param(self) -> str:
    while True:
      self.counter += 1
      name = f"_fixed_qconst{self.counter}"
      if not hasattr(self.gm, name):
        assert name not in self.gm._param_name_to_source
        self.gm.register_parameter(name, None)
        self.gm._param_name_to_source[name] = NNModuleSource(
          AttrSource(LocalSource("self"), name)
        )
        return name

  def extract_tensor(self, n: Optional[Node]) -> Optional[torch.Tensor]:
    if n is None:
      return None
    if n.op != "get_attr" or n.args != () or n.kwargs != {}:
      return None
    return getattr(self.gm, n.target)

  """
  slightly annoying because an older revision has:
    y = qd.quant(x, node1, node2, -128, 127, int8)
  but a new revision has:
    y = qd.quant.default(x, 0.02, -10, -128, 128, int8)

  we handle both cases for now...
  """

  def fetch_quant_per_tensor(self, n: Node, min, max, ty) -> Optional[Tuple[float, int]]:
    if n.op != "call_function":
      return None
    if n.target == qd.quantize_per_tensor.default:
      s = n.args[1]
      z = n.args[2]
    elif n.target == qd.quantize_per_tensor:
      s = self.extract_tensor(n.args[1])
      z = self.extract_tensor(n.args[2])
      if s is None or z is None:
        return None
      s = s.item()
      z = z.item()
    else:
      return None

    if n.args[3] == min and n.args[4] == max and n.args[5] == ty:
      return (s, z)
    return None

  def fetch_dequant_per_tensor(self, n: Node, min, max, ty) -> Optional[Tuple[float, int]]:
    if n.op != "call_function":
      return None
    if n.target == qd.dequantize_per_tensor.default:
      s = n.args[1]
      z = n.args[2]
    elif n.target == qd.dequantize_per_tensor:
      s = self.extract_tensor(n.args[1])
      z = self.extract_tensor(n.args[2])
      if s is None or z is None:
        return None
      s = s.item()
      z = z.item()
    else:
      return None

    if n.args[3] == min and n.args[4] == max and n.args[5] == ty:
      return (s, z)
    return None

  def fetch_quant_per_channel(self, n: Node, chan, min, max, ty) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if n.op != "call_function":
      return None
    if n.target not in {qd.quantize_per_channel, qd.quantize_per_channel.default}:
      return None

    if n.args[3] == chan and n.args[4] == min and n.args[5] == max and n.args[6] == ty:
      return self.extract_tensor(n.args[1]), self.extract_tensor(n.args[2])
    return None

  def fetch_dequant_per_channel(self, n: Node, chan, min, max, ty) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if n.op != "call_function":
      return None
    if n.target not in {qd.dequantize_per_channel, qd.dequantize_per_channel.default}:
      return None

    if n.args[3] == chan and n.args[4] == min and n.args[5] == max and n.args[6] == ty:
      return self.extract_tensor(n.args[1]), self.extract_tensor(n.args[2])
    return None

  """
  b + (sx (X - zx)) @ (sy (Y - zy))
    = b + sx sy ((X - zx) @ (Y - zy))
    = (sx sy) ([b / (sx sy)] + (X - zx) @ (Y - zy))

  in our case, zy is 0, so:
    = (sx sy) ([b / (sx sy)] + (X - zx) @ Y)
    = (sx sy) ([b / (sx sy) - sum(zx Y, axis=1)] + X @ Y)
  """

  def _match_qlinear(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_relu = node_q_output.args[0]
    if node_relu.op != "call_function":
      return None
    if node_relu.target in {aten.relu.default, aten.relu_.default}:
      node_matprod = node_relu.args[0]
    else:
      node_matprod = node_relu
      node_relu = None

    # depending on if there is bias, the linear layer could be addmm(b, x, wt)
    # or mm(x, wt).

    if node_matprod.op != "call_function":
      return None
    if node_matprod.target == aten.mm.default:
      node_bias = None
      node_dq_input = node_matprod.args[0]
      node_dq_wt = node_matprod.args[1]
    elif node_matprod.target == aten.addmm.default:
      node_bias = node_matprod.args[0]
      node_dq_input = node_matprod.args[1]
      node_dq_wt = node_matprod.args[2]
    else:
      return None

    if node_dq_wt.op != "call_function" or node_dq_wt.target != aten.t.default:
      return None

    node_dq_weight = node_dq_wt.args[0]

    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None
    qparam_weight = self.fetch_dequant_per_tensor(node_dq_weight, -127, 127, torch.int8)
    if not qparam_weight:
      return None

    node_q_weight = node_dq_weight.args[0]
    qparam_weight1 = self.fetch_quant_per_tensor(node_q_weight, -127, 127, torch.int8)
    if not qparam_weight1:
      return None

    if qparam_weight != qparam_weight1:
      return None

    return (
      node_relu is not None,
      node_dq_input.args[0],
      node_q_weight.args[0],
      node_bias,
      qparam_input,
      qparam_weight,
      qparam_out,
    )

  def _rewrite_qlinear(self, anchor: Node) -> bool:
    node_map = self._match_qlinear(anchor)
    if node_map is None:
      return False

    [needs_relu, x_node, w_node, b_node,
     (s_x, z_x), (s_w, z_w), (s_out, z_out)] = node_map
    b = self.extract_tensor(b_node)
    w = self.extract_tensor(w_node)
    if w is None:
      return None

    if z_w != 0:
      return None

    k = s_x * s_w
    weight_q = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)

    if b is None:
      bias_q = torch.zero([], dtype=torch.int32)
    else:
      bias_q = torch.round(b / k).int()
    bias_q = bias_q - z_x * torch.sum(weight_q, dim=1, dtype=torch.int32)

    weight_attr = w_node.target
    setattr(self.gm, weight_attr, torch.nn.Parameter(weight_q, False))

    if b is None:
      bias_attr = self.create_new_param()
    else:
      bias_attr = b_node.target
    setattr(self.gm, bias_attr, torch.nn.Parameter(bias_q, False))

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.get_attr(weight_attr)
      n2 = graph.get_attr(bias_attr)
      n3 = graph.call_function(shin.int_addmm, (n2, x_node, n1))
      if needs_relu:
        n3 = graph.call_function(aten.relu, (n3,))
      n4 = graph.call_function(shin.requantize, (n3, k / s_out, z_out))

    anchor.replace_all_uses_with(n4)
    return True

  """
  CONV(sx (X - zx), sw (W - zw), b)
    = sx sw (CONV(X - zx, W - zw, [b / (sx sw)])

  in our case, zw is 0, so:
    = sx sw (CONV(X - zx, W, [b / (sx sw)]))

  Note that when there is padding, we cannot factor out the zx term (like in
  the qlinear case). here we assume that it's never safe to do this.
  """

  def _match_qconv(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_relu = node_q_output.args[0]
    if node_relu.op != "call_function":
      return None
    if node_relu.target in {aten.relu.default, aten.relu_.default}:
      node_conv = node_relu.args[0]
    else:
      node_conv = node_relu
      node_relu = None

    if node_conv.op != "call_function" or node_conv.target != aten.convolution.default:
      return None

    node_dq_input = node_conv.args[0]
    node_dq_weight = node_conv.args[1]

    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None
    qparam_weight = self.fetch_dequant_per_tensor(node_dq_weight, -127, 127, torch.int8)
    if not qparam_weight:
      return None

    node_q_weight = node_dq_weight.args[0]
    qparam_weight1 = self.fetch_quant_per_tensor(node_q_weight, -127, 127, torch.int8)
    if not qparam_weight1:
      return None

    if qparam_weight != qparam_weight1:
      return None

    return (
      node_relu is not None,
      node_conv.args[3:],
      node_dq_input.args[0],
      node_q_weight.args[0],
      node_conv.args[2],
      qparam_input,
      qparam_weight,
      qparam_out,
    )

  def _rewrite_qconv(self, anchor: Node) -> bool:
    node_map = self._match_qconv(anchor)
    if node_map is None:
      return False

    [needs_relu, conv_params, x_node, w_node, b_node,
     (s_x, z_x), (s_w, z_w), (s_out, z_out)] = node_map
    b = self.extract_tensor(b_node)
    w = self.extract_tensor(w_node)
    if w is None:
      return False

    if z_w != 0:
      return False

    k = s_x * s_w
    kernel_q = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)

    if b is not None:
      bias_q = torch.round(b / k).int()

    kernel_attr = w_node.target
    setattr(self.gm, kernel_attr, torch.nn.Parameter(kernel_q, False))

    if b is not None:
      bias_attr = b_node.target
      setattr(self.gm, bias_attr, torch.nn.Parameter(bias_q, False))

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_method("int", (x_node,))
      n2 = graph.call_function(aten.sub, (n1, z_x))
      n3 = graph.get_attr(kernel_attr)
      n4 = graph.call_method("int", (n3,))
      n5 = None if b is None else graph.get_attr(bias_attr)
      n6 = graph.call_function(aten.convolution, (n2, n4, n5, *conv_params))
      if needs_relu:
        n6 = graph.call_function(aten.relu, (n6,))
      n7 = graph.call_function(shin.requantize, (n6, k / s_out, z_out))

    anchor.replace_all_uses_with(n7)
    return True

  """
  Per channel convolution is the almost the same as per tensor convolution.
  The difference is that now each output channel of the kernel has it's own
  scale, implying that s_w is now a tensor. z_w is still zero since it's
  symmetric.

  The input is still quantized per tensor!
  """

  def _match_qconv_per_channel(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_relu = node_q_output.args[0]
    if node_relu.op != "call_function":
      return None
    if node_relu.target in {aten.relu.default, aten.relu_.default}:
      node_conv = node_relu.args[0]
    else:
      node_conv = node_relu
      node_relu = None

    if node_conv.op != "call_function" or node_conv.target != aten.convolution.default:
      return None

    node_dq_input = node_conv.args[0]
    node_dq_weight = node_conv.args[1]

    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None
    qparam_weight = self.fetch_dequant_per_channel(node_dq_weight, 0, -127, 127, torch.int8)
    if not qparam_weight:
      return None

    node_q_weight = node_dq_weight.args[0]
    qparam_weight1 = self.fetch_quant_per_channel(node_q_weight, 0, -127, 127, torch.int8)
    if not qparam_weight1:
      return None

    if qparam_weight != qparam_weight1:
      return None

    return (
      node_relu is not None,
      node_conv.args[3:],
      node_dq_input.args[0],
      node_q_weight.args[0],
      node_conv.args[2],
      qparam_input,
      qparam_weight,
      qparam_out,
    )

  def _rewrite_qconv_per_channel(self, anchor: Node) -> bool:
    node_map = self._match_qconv_per_channel(anchor)
    if node_map is None:
      return False

    [needs_relu, conv_params, x_node, w_node, b_node,
     (s_x, z_x), (s_w, z_w), (s_out, z_out)] = node_map
    b = self.extract_tensor(b_node)
    w = self.extract_tensor(w_node)
    if w is None:
      return False

    # check if z_w is all zeros
    if torch.any(z_w).item():
      return False

    k = s_x * s_w
    kernel_q = qd.quantize_per_channel(w, s_w, z_w, 0, -127, 127, torch.int8)

    if b is not None:
      bias_q = torch.round(b / k).int()

    kernel_attr = w_node.target
    setattr(self.gm, kernel_attr, torch.nn.Parameter(kernel_q, False))

    if b is not None:
      bias_attr = b_node.target
      setattr(self.gm, bias_attr, torch.nn.Parameter(bias_q, False))

    # we keep the scales as a Python list and leave the responsibility of
    # quantizing these values to the lowering step!
    scales = (k / s_out).float().tolist()

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_method("int", (x_node,))
      n2 = graph.call_function(aten.sub, (n1, z_x))
      n3 = graph.get_attr(kernel_attr)
      n4 = graph.call_method("int", (n3,))
      n5 = None if b is None else graph.get_attr(bias_attr)
      n6 = graph.call_function(aten.convolution, (n2, n4, n5, *conv_params))
      if needs_relu:
        n6 = graph.call_function(aten.relu, (n6,))
      n7 = graph.call_function(shin.requantize_channel, (n6, scales, z_out))

    anchor.replace_all_uses_with(n7)
    return True

  """
  MAXPOOL2D(sx (X - zx)) = sx MAXPOOL2D(X - zx)
  """

  def _match_qmaxpool(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_getitem = node_q_output.args[0]
    if (
      node_getitem.op != "call_function" or
      node_getitem.target != operator.getitem or
      node_getitem.args[1] != 0
    ):
      return None

    node_pool = node_getitem.args[0]

    if node_pool.op != "call_function" or node_pool.target != aten.max_pool2d_with_indices.default:
      return None

    node_dq_input = node_pool.args[0]
    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None

    # make sure qinput and qoutput are shared
    if qparam_out != qparam_input:
      return None

    # ceil_mode=True is not supported
    pool_args = _get_all_arguments(
      node_pool.args, node_pool.kwargs, node_pool.target._schema.arguments
    )
    if pool_args[-1]:
      return None

    return (
      pool_args[1:-1],
      node_dq_input.args[0],
    )

  def _rewrite_qmaxpool(self, anchor: Node) -> bool:
    node_map = self._match_qmaxpool(anchor)
    if node_map is None:
      return False

    [pool_args, x] = node_map

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_function(shin.int_max_pool2d, (x, *pool_args))

    anchor.replace_all_uses_with(n1)
    return True

  """
  q(MEAN(dq(X)) = MEAN(X)
  """

  def _match_qmean(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_mean = node_q_output.args[0]
    if node_mean.op != "call_function" or node_mean.target != aten.mean.dim:
      return None

    node_dq_input = node_mean.args[0]
    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None

    # make sure qinput and qoutput are shared
    if qparam_out != qparam_input:
      return None

    # make sure explicit dtype is not supported
    mean_args = _get_all_arguments(
      node_mean.args, node_mean.kwargs, node_mean.target._schema.arguments
    )
    if mean_args[-1]:
      return None

    return (
      mean_args[1:-1],
      node_dq_input.args[0],
    )

  def _rewrite_qmean(self, anchor: Node) -> bool:
    node_map = self._match_qmean(anchor)
    if node_map is None:
      return False

    [mean_args, x] = node_map

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_method("int", (x,))
      n2 = graph.call_function(shin.int_mean, (n1, *mean_args))
      n3 = graph.call_method("to", (n2, torch.int8))

    anchor.replace_all_uses_with(n3)
    return True

  """
  AVGPOOL2D(sx (X - zx)) = sx AVGPOOL2D(X - zx)
  """

  def _match_qavgpool(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_pool = node_q_output.args[0]
    if node_pool.op != "call_function":
      return None

    # we might come across _adaptive_avg_pool2d OR avg_pool2d.
    pool_func = None
    pool_args = None
    if node_pool.target == aten._adaptive_avg_pool2d.default:
      # since we don't have the shape information, assume we can lower it.
      pool_func = shin.int_adaptive_avg_pool2d
      pool_args = (node_pool.args[1],)
    elif node_pool.target == aten.avg_pool2d.default:
      # make sure it is something we can lower
      args = _get_all_arguments(node_pool.args, node_pool.kwargs, node_pool.target._schema.arguments)
      if (
        not args[4]           # ceiling mode is false
        and args[5]           # padded zeros count towards the divisor
        and args[6] is None   # apparently aten.avg_pool2d allows a predefined divisor
      ):
        # avg_pool2d allows empty stride.
        # it defaults to the same thing as kernel_size
        pool_func = shin.int_avg_pool2d
        pool_args = (args[1], args[2] or args[1], args[3])

    if pool_func is None:
      return None

    node_dq_input = node_pool.args[0]
    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None

    # make sure qinput and qoutput are shared
    if qparam_out != qparam_input:
      return None

    return (
      pool_func,
      pool_args,
      node_dq_input.args[0],
    )

  def _rewrite_qavgpool(self, anchor: Node) -> bool:
    node_map = self._match_qavgpool(anchor)
    if node_map is None:
      return False

    [pool_func, pool_args, x] = node_map

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_method("int", (x,))
      n2 = graph.call_function(pool_func, (n1, *pool_args))
      n3 = graph.call_method("to", (n2, torch.int8))

    anchor.replace_all_uses_with(n3)
    return True

  """
  FLATTEN(sx (X - zx)) = sx FLATTEN(X - zx)
  """

  def _match_flatten(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_flatten = node_q_output.args[0]
    if node_flatten.op != "call_function" or node_flatten.target != shin.flatten.default:
      return None

    node_dq_input = node_flatten.args[0]
    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None

    # make sure qinput and qoutput are shared
    if qparam_out != qparam_input:
      return None

    return (
      node_flatten.args[1],
      node_flatten.args[2],
      node_dq_input.args[0],
    )

  def _rewrite_flatten(self, anchor: Node) -> bool:
    node_map = self._match_flatten(anchor)
    if node_map is None:
      return False

    [start, end, x] = node_map

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_function(shin.flatten, (x, start, end))

    anchor.replace_all_uses_with(n1)
    return True

  """
  q(HARDTANH(dq(X), min, max))
    = q(CLAMP(dq(X), min, max))
    = CLAMP(X, q(min), q(max))

  ReLU6 is a special case where the clamping range is 0 to 6
  """

  def _match_hardtanh(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_hardtanh = node_q_output.args[0]
    if node_hardtanh.op != "call_function":
      return None
    if node_hardtanh.target not in {aten.hardtanh_.default, aten.hardtanh.default}:
      return None

    node_dq_input = node_hardtanh.args[0]
    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None

    # make sure qinput and qoutput are shared
    if qparam_out != qparam_input:
      return None

    return (
      node_hardtanh.args[1],
      node_hardtanh.args[2],
      node_dq_input.args[0],
      qparam_input
    )

  def _rewrite_hardtanh(self, anchor: Node) -> bool:
    node_map = self._match_hardtanh(anchor)
    if node_map is None:
      return False

    [fmin, fmax, x_node, (s_x, z_x)] = node_map
    qmin = qd.quantize_per_tensor(torch.Tensor([fmin]), s_x, z_x, -128, 127, torch.int8).item()
    qmax = qd.quantize_per_tensor(torch.Tensor([fmax]), s_x, z_x, -128, 127, torch.int8).item()

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_function(aten.clamp, (x_node, qmin, qmax))

    anchor.replace_all_uses_with(n1)
    return True

  """
  q(hardsigmoid(dq(X)))
    = requant(x - zx + 3/sx, 256/6 sx, -128)

  Notes:
  *  provded the output qparam has scale of 1/256 and zero point of -128
  *  3/sx we approximate it with a integer.
  """

  def _match_hardsigmoid(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_act = node_q_output.args[0]
    if node_act.op != "call_function" or node_act.target != aten.hardsigmoid.default:
      return None

    node_dq_input = node_act.args[0]
    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None

    if qparam_out != (1/256.0, -128):
      return None

    return (
      node_dq_input.args[0],
      qparam_input
    )

  def _rewrite_hardsigmoid(self, anchor: Node) -> bool:
    node_map = self._match_hardsigmoid(anchor)
    if node_map is None:
      return False

    [x_node, (s_x, z_x)] = node_map
    s_out = s_x * 256.0 / 6.0
    z_out = -128

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_method("int", (x_node,))
      n2 = graph.call_function(aten.sub, (n1, z_x - round(3 / s_x)))
      n3 = graph.call_function(shin.requantize, (n2, s_out, z_out))

    anchor.replace_all_uses_with(n3)
    return True

  """
  q(hardswish(dq(X)))
    = q(1/6 sx^2 (x - zx) clamp(x - zx + 3/sx, 0, 6/sx))

  As in hardsigmoid, 3/sx and 6/sx are approximated using integers
  """

  def _match_hardswish(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_act = node_q_output.args[0]
    if node_act.op != "call_function":
      return None
    if node_act.target not in {aten.hardswish.default, aten.hardswish_.default}:
      return None

    node_dq_input = node_act.args[0]
    qparam_input = self.fetch_dequant_per_tensor(node_dq_input, -128, 127, torch.int8)
    if not qparam_input:
      return None

    return (
      node_dq_input.args[0],
      qparam_input,
      qparam_out,
    )

  def _rewrite_hardswish(self, anchor: Node) -> bool:
    node_map = self._match_hardswish(anchor)
    if node_map is None:
      return False

    [x_node, (s_x, z_x), (s_out, z_out)] = node_map
    k = s_x / s_out * s_x / 6

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_method("int", (x_node,))
      n2 = graph.call_function(aten.sub, (n1, z_x - round(3 / s_x)))
      n3 = graph.call_function(aten.clamp, (n2, 0, round(6 / s_x)))
      n4 = graph.call_function(aten.sub, (n1, z_x))
      n5 = graph.call_function(aten.mul, (n3, n4))
      n6 = graph.call_function(shin.requantize, (n5, k, z_out))

    anchor.replace_all_uses_with(n6)
    return True

  """
  IF X and Y have the same qparams:
  q(dq(X) + dq(Y))
    = q(s (X + Y - 2z))   <-- provided 2z does not overflow (unlikely)

  OTHERWISE:
  requantize consists of multiple smaller steps: rescale, round, adjust zeros,
  clamp, truncate. the idea is we want to perform the add operation between
  rescale and round.

  of course, much of this is codegen / bitwidth dependent, so we lower it into
  an intrinsic node. (and see you in shir-lowering)
  """

  def _match_add(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_add = node_q_output.args[0]
    if node_add.op != "call_function":
      return None
    if node_add.target not in {aten.add.Tensor, aten.add_.Tensor}:
      return None

    node_dq_lhs = node_add.args[0]
    node_dq_rhs = node_add.args[1]

    qparam_lhs = self.fetch_dequant_per_tensor(node_dq_lhs, -128, 127, torch.int8)
    if not qparam_lhs:
      return None

    qparam_rhs = self.fetch_dequant_per_tensor(node_dq_rhs, -128, 127, torch.int8)
    if not qparam_rhs:
      return None

    return (
      node_dq_lhs.args[0],
      node_dq_rhs.args[0],
      qparam_lhs,
      qparam_rhs,
      qparam_out,
    )

  def _rewrite_add(self, anchor: Node) -> bool:
    node_map = self._match_add(anchor)
    if node_map is None:
      return False

    [lhs_node, rhs_node, (s_x, z_x), (s_y, z_y), (s_out, z_out)] = node_map

    # case when both inputs share qparams
    if s_x == s_y and z_x == z_y and (
      (-1<<31) <= 2 * z_x < (1<<31)   # sanity check
    ):
      graph = self.gm.graph
      with graph.inserting_before(anchor):
        n1 = graph.call_method("int", (lhs_node,))
        n2 = graph.call_method("int", (rhs_node,))
        n3 = graph.call_function(aten.add, (n1, n2))
        n4 = graph.call_function(aten.sub, (n3, 2 * z_x))
        n5 = graph.call_function(shin.requantize, (n4, s_x / s_out, z_out))

      anchor.replace_all_uses_with(n5)
      return True

    # fallback case
    graph = self.gm.graph
    with graph.inserting_before(anchor):
      # adjust the input zero points before handing it off to qadd
      n1 = graph.call_method("int", (lhs_node,))
      n2 = graph.call_function(aten.sub, (n1, z_x))
      n3 = graph.call_method("int", (rhs_node,))
      n4 = graph.call_function(aten.sub, (n3, z_y))
      n5 = graph.call_function(functional.qadd, (n2, s_x / s_out, n4, s_y / s_out, z_out))

    anchor.replace_all_uses_with(n5)
    return True

  """
  q(dq(X) dq(Y))
    = q(sx sy (X - zx) (Y - zy))
  """

  def _match_mul(self, node_q_output: Node):
    qparam_out = self.fetch_quant_per_tensor(node_q_output, -128, 127, torch.int8)
    if not qparam_out:
      return None

    node_mul = node_q_output.args[0]
    if node_mul.op != "call_function":
      return None
    if node_mul.target not in {aten.mul.Tensor, aten.mul_.Tensor}:
      return None

    node_dq_lhs = node_mul.args[0]
    node_dq_rhs = node_mul.args[1]

    qparam_lhs = self.fetch_dequant_per_tensor(node_dq_lhs, -128, 127, torch.int8)
    if not qparam_lhs:
      return None

    qparam_rhs = self.fetch_dequant_per_tensor(node_dq_rhs, -128, 127, torch.int8)
    if not qparam_rhs:
      return None

    return (
      node_dq_lhs.args[0],
      node_dq_rhs.args[0],
      qparam_lhs,
      qparam_rhs,
      qparam_out,
    )

  def _rewrite_mul(self, anchor: Node) -> bool:
    node_map = self._match_mul(anchor)
    if node_map is None:
      return False

    [lhs_node, rhs_node, (s_x, z_x), (s_y, z_y), (s_out, z_out)] = node_map

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_method("int", (lhs_node,))
      n2 = graph.call_function(aten.sub, (n1, z_x))
      n3 = graph.call_method("int", (rhs_node,))
      n4 = graph.call_function(aten.sub, (n3, z_y))
      n5 = graph.call_function(aten.mul, (n2, n4))
      n6 = graph.call_function(shin.requantize, (n5, s_x * s_y / s_out, z_out))

    anchor.replace_all_uses_with(n6)
    return True

def rewrite_quantized_ops(gm: GraphModule):
  obj = QuantOpRewrite(gm)
  obj.rewrite()

def rewrite_late(gm: GraphModule):
  changed = False
  for n in gm.graph.nodes:
    if n.op != "call_function" or n.target != aten.convolution.default:
      continue
    if n.args[2] is None:
      continue

    kernel_node = n.args[1]
    info = kernel_node.meta.get("val")
    if info is None:
      continue

    # Turn:
    #   n = aten.convolution(i, k, b, ...)
    # into:
    #   p = aten.convolution(i, k, None, ...)
    #   q = torch.reshape(b, [-1, 1, 1, ...])
    #   n = torch.add(p, q)
    changed = True
    broadcast = [-1] + [1] * (len(info.shape) - 2)
    new_args = list(n.args)
    new_args[2] = None
    graph = gm.graph
    with graph.inserting_before(n):
      p = graph.call_function(aten.convolution, tuple(new_args))
      q = graph.call_function(torch.reshape, (n.args[2], broadcast))
    n.target = torch.add
    n.args = (p, q)

  if changed:
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
