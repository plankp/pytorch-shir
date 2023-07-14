from typing import Optional, Tuple
from torch.fx.graph_module import GraphModule
from torch.fx import subgraph_rewriter, Node
import torch
import shir_intrinsic   # make sure these are loaded
import operator
from torch.ao.quantization._pt2e.utils import (
  _get_all_arguments,
)

# don't match or emit prims ops at this level!
aten = torch.ops.aten
shir = torch.ops.shir_intrinsic
qd = torch.ops.quantized_decomposed

class QuantOpRewrite:
  def __init__(self, gm: GraphModule):
    self.counter = -1
    self.gm = gm

  def _rewrite_node(self, n: Node) -> bool:
    if self._rewrite_qconv(n):
      return True
    if self._rewrite_qlinear(n):
      return True
    if self._rewrite_qmaxpool(n):
      return True
    if self._rewrite_qavgpool(n):
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

  def find_free_name(self) -> str:
    while True:
      self.counter += 1
      name = f"_fixed_qconst{self.counter}"
      if not hasattr(self.gm, name):
        assert name not in self.gm._param_name_to_source
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
    else:
      return None

    if n.args[3] == min and n.args[4] == max and n.args[5] == ty:
      return (s, z)
    return None

  def fetch_dequant_per_tensor(self, n: Node, min, max, ty) -> bool:
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
    else:
      return None

    if n.args[3] == min and n.args[4] == max and n.args[5] == ty:
      return (s, z)
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

    [needs_relu, x, w_node, b_node, *rest] = node_map
    tensors = [self.extract_tensor(n) for n in rest]
    if not all((isinstance(t, torch.Tensor) for t in tensors)):
      return None

    [s_x, z_x, s_w, z_w, s_out, z_out] = tensors
    b = self.extract_tensor(b_node)
    w = self.extract_tensor(w_node)
    if w is None:
      return None

    if z_w.int().item() != 0:
      return None

    k = (s_x * s_w).item()
    weight_q = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)

    if b is None:
      bias_q = torch.zero([], dtype=torch.int32)
    else:
      bias_q = torch.round(b / k).int()
    bias_q = bias_q - z_x.int().item() * torch.sum(weight_q, dim=1, dtype=torch.int32)

    weight_attr = w_node.target
    setattr(self.gm, weight_attr, torch.nn.Parameter(weight_q, False))

    if b is None:
      bias_attr = self.find_free_name()
    else:
      bias_attr = b_node.target
    setattr(self.gm, bias_attr, torch.nn.Parameter(bias_q, False))

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.get_attr(weight_attr)
      n2 = graph.get_attr(bias_attr)
      n3 = graph.call_function(shir.int_addmm, (n2, x, n1))
      if needs_relu:
        n3 = graph.call_function(aten.relu, (n3,))
      n4 = graph.call_function(shir.requantize, (n3, k / s_out.item(), z_out.item()))

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
    if not self.is_quant_per_tensor(node_q_output, -128, 127, torch.int8):
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

    if not self.is_dequant_per_tensor(node_dq_input, -128, 127, torch.int8):
      return None
    if not self.is_dequant_per_tensor(node_dq_weight, -127, 127, torch.int8):
      return None

    node_q_weight = node_dq_weight.args[0]
    if not self.is_quant_per_tensor(node_q_weight, -127, 127, torch.int8):
      return None

    if (
      node_dq_weight.args[1] != node_q_weight.args[1] or
      node_dq_weight.args[2] != node_q_weight.args[2]
    ):
      return None

    return (
      node_relu is not None,
      node_conv.args[3:],
      node_dq_input.args[0],
      node_q_weight.args[0],
      node_conv.args[2],
      node_dq_input.args[1],
      node_dq_input.args[2],
      node_q_weight.args[1],
      node_q_weight.args[2],
      node_q_output.args[1],
      node_q_output.args[2],
    )

  def _rewrite_qconv(self, anchor: Node) -> bool:
    node_map = self._match_qconv(anchor)
    if node_map is None:
      return False

    [needs_relu, conv_params, x, w_node, b_node, *rest] = node_map
    tensors = [self.extract_tensor(n) for n in rest]
    if not all((isinstance(t, torch.Tensor) for t in tensors)):
      return False

    [s_x, z_x, s_w, z_w, s_out, z_out] = tensors
    b = self.extract_tensor(b_node)
    w = self.extract_tensor(w_node)
    if w is None:
      return False

    if z_w.int().item() != 0:
      return False

    k = (s_x * s_w).item()
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
      n1 = graph.call_method("int", (x,))
      n2 = graph.call_function(aten.sub, (n1, z_x.int().item()))
      n3 = graph.get_attr(kernel_attr)
      n4 = graph.call_method("int", (n3,))
      n5 = None if b is None else graph.get_attr(bias_attr)
      n6 = graph.call_function(aten.convolution, (n2, n4, n5, *conv_params))
      if needs_relu:
        n6 = graph.call_function(aten.relu, (n6,))
      n7 = graph.call_function(shir.requantize, (n6, k / s_out.item(), z_out.item()))

    anchor.replace_all_uses_with(n7)
    return True

  """
  MAXPOOL2D(sx (X - zx)) = sx MAXPOOL2D(X - zx)
  """

  def _match_qmaxpool(self, node_q_output: Node):
    if not self.is_quant_per_tensor(node_q_output, -128, 127, torch.int8):
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
    if not self.is_dequant_per_tensor(node_dq_input, -128, 127, torch.int8):
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
      node_dq_input.args[1],
      node_dq_input.args[2],
      node_q_output.args[1],
      node_q_output.args[2],
    )

  def _rewrite_qmaxpool(self, anchor: Node) -> bool:
    node_map = self._match_qmaxpool(anchor)
    if node_map is None:
      return False

    [pool_args, x, *rest] = node_map
    tensors = [self.extract_tensor(n) for n in rest]
    if not all((isinstance(t, torch.Tensor) for t in tensors)):
      return False

    [s_x, z_x, s_out, z_out] = tensors
    if s_x.item() != s_out.item() or z_x.item() != z_out.item():
      return False

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_function(shir.int_max_pool2d, (x, *pool_args))

    anchor.replace_all_uses_with(n1)
    return True

  """
  AVGPOOL2D(sx (X - zx)) = sx AVGPOOL2D(X - zx)
  """

  def _match_qavgpool(self, node_q_output: Node):
    if not self.is_quant_per_tensor(node_q_output, -128, 127, torch.int8):
      return None

    node_pool = node_q_output.args[0]
    if node_pool.op != "call_function" or node_pool.target != aten._adaptive_avg_pool2d.default:
      return None

    node_dq_input = node_pool.args[0]
    if not self.is_dequant_per_tensor(node_dq_input, -128, 127, torch.int8):
      return None

    return (
      node_pool.args[1],
      node_dq_input.args[0],
      node_dq_input.args[1],
      node_dq_input.args[2],
      node_q_output.args[1],
      node_q_output.args[2],
    )

  def _rewrite_qavgpool(self, anchor: Node) -> bool:
    node_map = self._match_qavgpool(anchor)
    if node_map is None:
      return False

    [output_size, x, *rest] = node_map
    tensors = [self.extract_tensor(n) for n in rest]
    if not all((isinstance(t, torch.Tensor) for t in tensors)):
      return False

    [s_x, z_x, s_out, z_out] = tensors
    if s_x.item() != s_out.item() or z_x.item() != z_out.item():
      return False

    graph = self.gm.graph
    with graph.inserting_before(anchor):
      n1 = graph.call_method("int", (x,))
      n2 = graph.call_function(shir.int_adaptive_avg_pool2d, (n1, output_size))
      n3 = graph.call_method("to", (n2, torch.int8))

    anchor.replace_all_uses_with(n3)
    return True

def rewrite_quantized_ops(gm: GraphModule):
  obj = QuantOpRewrite(gm)
  obj.rewrite()
