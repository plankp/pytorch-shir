from typing import Optional
from torch.fx.graph_module import GraphModule
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch._dynamo.source import (
  NNModuleSource,
  LocalSource,
  AttrSource,
)
from torch.fx import subgraph_rewriter, Node
import torch
import shir_intrinsic   # make sure these are loaded

# don't match or emit prims ops at this level!
aten = torch.ops.aten
shir = torch.ops.shir_intrinsic
qd = torch.ops.quantized_decomposed

class QuantOpRewrite:

  def __init__(self, gm: GraphModule):
    self.counter = -1
    self.gm = gm

  def rewrite(self):
    self._rewrite_qconv_relu()
    self._rewrite_qconv()
    self._rewrite_qlinear_relu()
    self._rewrite_qlinear()
    self._rewrite_qmaxpool()
    self._rewrite_qavgpool()

  def find_free_name(self) -> str:
    while True:
      self.counter += 1
      name = f"_fixed_qconst{self.counter}"
      if not hasattr(self.gm, name):
        assert name not in self.gm._param_name_to_source
        return name

  def synthesize_tensor(self, tensor: torch.Tensor) -> str:
    # because fx nodes don't like raw tensors appearing in arguments, we
    # follow the workaround used in pytorch/pytorch #43512, which is to make
    # it a module-level attribute

    name = self.find_free_name()
    self.gm.register_buffer(name, tensor)
    self.gm._param_name_to_source[name] = NNModuleSource(
      AttrSource(LocalSource("self"), name)
    )
    return name

  def extract_tensor(self, n: Node) -> Optional[torch.Tensor]:
    if n.op != "get_attr" or n.args != () or n.kwargs != {}:
      return None
    return getattr(self.gm, n.target)

  def find_first_user(self, n: Node) -> Optional[Node]:
    if len(n.users) == 1:
      return list(n.users.keys())[0]

    # TODO: actually find
    return None

  """
  b + (sx (X - zx)) @ (sy (Y - zy))
    = b + sx sy ((X - zx) @ (Y - zy))
    = (sx sy) ([b / (sx sy)] + (X - zx) @ (Y - zy))   <-- bias is quantized
  """

  def _rewrite_qlinear(self):
    def biased_pattern(x, w, b, s_x, z_x, s_w, z_w, s_out, z_out):
      x = qd.dequantize_per_tensor(x, s_x, z_x, -128, 127, torch.int8)
      w = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = qd.dequantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = aten.t.default(w)
      x = aten.addmm.default(b, x, w)
      x = qd.quantize_per_tensor(x, s_out, z_out, -128, 127, torch.int8)
      return x

    pattern = torch.fx.symbolic_trace(biased_pattern).graph
    self._match_and_rewrite(pattern, lambda m: self._template_qlinear(m, relu=False))

    def unbiased_pattern(x, w, s_x, z_x, s_w, z_w, s_out, z_out):
      x = qd.dequantize_per_tensor(x, s_x, z_x, -128, 127, torch.int8)
      w = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = qd.dequantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = aten.t.default(w)
      x = aten.mm.default(x, w)
      x = qd.quantize_per_tensor(x, s_out, z_out, -128, 127, torch.int8)
      return x

    pattern = torch.fx.symbolic_trace(unbiased_pattern).graph
    self._match_and_rewrite(pattern, lambda m: self._template_qlinear(m, relu=False))

  """
  ReLU(b + (sx (X - zx)) @ (sy (Y - zy)))
    = (sx sy) ReLU([b / (sx sy)] + (X - zx) @ (Y - zy))
  """

  def _rewrite_qlinear_relu(self):
    def biased_pattern(x, w, b, s_x, z_x, s_w, z_w, s_out, z_out):
      x = qd.dequantize_per_tensor(x, s_x, z_x, -128, 127, torch.int8)
      w = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = qd.dequantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = aten.t.default(w)
      x = aten.addmm.default(b, x, w)
      x = aten.relu.default(x)
      x = qd.quantize_per_tensor(x, s_out, z_out, -128, 127, torch.int8)
      return x

    pattern = torch.fx.symbolic_trace(biased_pattern).graph
    self._match_and_rewrite(pattern, lambda m: self._template_qlinear(m, relu=True))

    def unbiased_pattern(x, w, s_x, z_x, s_w, z_w, s_out, z_out):
      x = qd.dequantize_per_tensor(x, s_x, z_x, -128, 127, torch.int8)
      w = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = qd.dequantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = aten.t.default(w)
      x = aten.mm.default(x, w)
      x = aten.relu.default(x)
      x = qd.quantize_per_tensor(x, s_out, z_out, -128, 127, torch.int8)
      return x

    pattern = torch.fx.symbolic_trace(unbiased_pattern).graph
    self._match_and_rewrite(pattern, lambda m: self._template_qlinear(m, relu=True))

  """
  CONV(sx (X - zx), sw (W - zw), b)
    = sx sw (CONV(X - zx, W - zw, [b / (sx sw)])
  """

  def _rewrite_qconv(self):
    def pattern(stride, padding, dilation, transposed, output_padding, groups,
                x_q, w, b, s_x, z_x, s_w, z_w, s_out, z_out):
      x = qd.dequantize_per_tensor(x_q, s_x, z_x, -128, 127, torch.int8)
      w_q = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = qd.dequantize_per_tensor(w_q, s_w, z_w, -127, 127, torch.int8)
      p = aten.convolution.default(
          x, w, b,
          stride, padding, dilation, transposed, output_padding, groups)
      p = qd.quantize_per_tensor(p, s_out, z_out, -128, 127, torch.int8)
      return p

    pattern = torch.fx.symbolic_trace(pattern).graph
    self._match_and_rewrite(pattern, lambda m: self._template_qconv(m, relu=False))

  """
  ReLU(CONV(sx (X - zx), sw (W - zw)), b)
    = sx sw ReLU(CONV(X - zx, W - zw, [b / (sx sw)]))
  """

  def _rewrite_qconv_relu(self):
    def pattern(stride, padding, dilation, transposed, output_padding, groups,
                x_q, w, b, s_x, z_x, s_w, z_w, s_out, z_out):
      x = qd.dequantize_per_tensor(x_q, s_x, z_x, -128, 127, torch.int8)
      w_q = qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int8)
      w = qd.dequantize_per_tensor(w_q, s_w, z_w, -127, 127, torch.int8)
      p = aten.convolution.default(
          x, w, b,
          stride, padding, dilation, transposed, output_padding, groups)
      p = aten.relu.default(p)
      p = qd.quantize_per_tensor(p, s_out, z_out, -128, 127, torch.int8)
      return p

    pattern = torch.fx.symbolic_trace(pattern).graph
    self._match_and_rewrite(pattern, lambda m: self._template_qconv(m, relu=True))

  """
  MAXPOOL2D(sx (X - zx)) = sx MAXPOOL2D(X - zx)
  """

  def _rewrite_qmaxpool(self):
    def pattern(kernel_size, stride, padding, dilation, x_q,
                s_x, z_x, s_out, z_out):
      x = qd.dequantize_per_tensor(x_q, s_x, z_x, -128, 127, torch.int8)
      x = aten.max_pool2d_with_indices.default(x, kernel_size, stride, padding, dilation)
      x = x[0]
      x = qd.quantize_per_tensor(x, s_out, z_out, -128, 127, torch.int8)
      return x

    pattern = torch.fx.symbolic_trace(pattern).graph
    self._match_and_rewrite(pattern, lambda m: self._template_qmaxpool(m))

  """
  AVGPOOL2D(sx (X - zx)) = sx AVGPOOL2D(X - zx)
  """

  def _rewrite_qavgpool(self):
    def pattern(output_size, x_q, s_x, z_x, s_out, z_out):
      x = qd.dequantize_per_tensor(x_q, s_x, z_x, -128, 127, torch.int8)
      x = aten._adaptive_avg_pool2d.default(x, output_size)
      x = qd.quantize_per_tensor(x, s_out, z_out, -128, 127, torch.int8)
      return x

    pattern = torch.fx.symbolic_trace(pattern).graph
    self._match_and_rewrite(pattern, lambda m: self._template_qavgpool(m))

  def _match_and_rewrite(self, pattern: torch.fx.Graph, callback):
    graph = self.gm.graph
    matcher = SubgraphMatcher(pattern)
    matches = matcher.match(graph)
    replacement_map = {}
    for m in matches:
      for i, old in enumerate(m.placeholder_nodes):
        if isinstance(old, Node) and old in replacement_map:
          replaced = replacement_map[old]
          m.placeholder_nodes[i] = replaced
          for k, v in m.nodes_map.items():
            if v is old:
              m.nodes_map[k] = replaced

      r = callback(m)
      if r is not None:
        original, replacement = r
        original.replace_all_uses_with(replacement)
        replacement_map[original] = replacement

      for node in m.nodes_map.values():
        if node not in m.placeholder_nodes:
          graph.erase_node(node)

  """
  When rewriting, we want to try to quantize the weighs right now. Once that
  happens, we just wipe-out the original weights.
  """

  def _template_qlinear(self, m, relu=False):
    if len(m.returning_nodes) != 1:
      return None

    [x, *rest] = m.placeholder_nodes
    tensors = [self.extract_tensor(n) for n in rest]
    if not all((isinstance(t, torch.Tensor) for t in tensors)):
      return None

    match tensors:
      case [w, s_x, z_x, s_w, z_w, s_out, z_out]:
        b = None
      case [w, b, s_x, z_x, s_w, z_w, s_out, z_out]:
        pass
      case _:
        assert False, "invalid qlinear-relu match"

    k = (s_x * s_w).item()
    weight_v = torch.nn.Parameter(
      qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int32) - z_w.int(),
      False
    )
    if b is not None:
      bias_v = torch.nn.Parameter(torch.round(b / k).int(), False)

    last_node = m.returning_nodes[0]
    first_user = self.find_first_user(last_node)
    if first_user is None:
      return None

    weight_attr = rest[0].target
    setattr(self.gm, weight_attr, weight_v)
    if b is not None:
      bias_attr = rest[1].target
      setattr(self.gm, bias_attr, bias_v)

    graph = self.gm.graph
    with graph.inserting_before(first_user):
      n1 = graph.call_method("int", (x,))
      n2 = graph.call_function(aten.sub, (n1, z_x.int().item()))
      n3 = graph.get_attr(weight_attr)
      if b is not None:
        n4 = graph.get_attr(bias_attr)
      n5 = graph.call_function(aten.t, (n3,))
      n6 = graph.call_function(aten.mm, (n2, n5))
      if b is not None:
        n6 = graph.call_function(aten.add, (n4, n6))
      if relu:
        n6 = graph.call_function(aten.relu, (n6,))
      n7 = graph.call_function(shir.requantize, (n6, k / s_out.item(), z_out.item()))

    return (last_node, n7)

  def _template_qconv(self, m, relu=False):
    if len(m.returning_nodes) != 1:
      return None

    [stride, padding, dilation, transposed, output_padding, groups,
     x, *rest] = m.placeholder_nodes
    tensors = [self.extract_tensor(n) for n in rest]
    if not all((isinstance(t, torch.Tensor) for t in tensors)):
      return None

    [w, b, s_x, z_x, s_w, z_w, s_out, z_out] = tensors

    k = (s_x * s_w).item()
    kernel_v = torch.nn.Parameter(
      qd.quantize_per_tensor(w, s_w, z_w, -127, 127, torch.int32) - z_w.int(),
      False
    )
    if b is not None:
      bias_v = torch.nn.Parameter(torch.round(b / k).int(), False)

    last_node = m.returning_nodes[0]
    first_user = self.find_first_user(last_node)
    if first_user is None:
      return None

    kernel_attr = rest[0].target
    setattr(self.gm, kernel_attr, kernel_v)
    if b is not None:
      bias_attr = rest[1].target
      setattr(self.gm, bias_attr, bias_v)

    graph = self.gm.graph
    with graph.inserting_before(first_user):
      n1 = graph.call_method("int", (x,))
      n2 = graph.call_function(aten.sub, (n1, z_x.int().item()))
      n3 = graph.get_attr(kernel_attr)
      if b is None:
        n4 = None
      else:
        n4 = graph.get_attr(bias_attr)
      n5 = graph.call_function(aten.convolution, (
        n2, n3, n4,
        stride, padding, dilation, transposed, output_padding, groups))
      if relu:
        n5 = graph.call_function(aten.relu, (n5,))
      n6 = graph.call_function(shir.requantize, (n5, k / s_out.item(), z_out.item()))

    return (last_node, n6)

  def _template_qmaxpool(self, m):
    if len(m.returning_nodes) != 1:
      return None

    [kernel_size, stride, padding, dilation, x, *rest] = m.placeholder_nodes
    tensors = [self.extract_tensor(n) for n in rest]
    if not all((isinstance(t, torch.Tensor) for t in tensors)):
      return None
    [s_x, z_x, s_out, z_out] = tensors

    last_node = m.returning_nodes[0]
    first_user = self.find_first_user(last_node)
    if first_user is None:
      return None

    # note that the shir quantizer would have tried to share the qparams
    # between the input and outputs!
    shared_qparams = s_x.item() == s_out.item() and z_x.item() == z_out.item()

    graph = self.gm.graph
    with graph.inserting_before(first_user):
      if shared_qparams:
        n4 = graph.call_function(shir.int_max_pool2d, (
          x, kernel_size, stride, padding, dilation))
      else:
        n1 = graph.call_method("int", (x,))
        n2 = graph.call_function(aten.sub, (n1, z_x.int().item))
        n3 = graph.call_function(shir.int_max_pool2d, (
          n2, kernel_size, stride, padding, dilation))
        n4 = graph.call_function(shir.requantize, (n3, s_x.item() / s_out.item(), z_out.item()))
    return (last_node, n4)

  def _template_qavgpool(self, m):
    if len(m.returning_nodes) != 1:
      return None

    [output_size, x, *rest] = m.placeholder_nodes
    tensors = [self.extract_tensor(n) for n in rest]
    if not all((isinstance(t, torch.Tensor) for t in tensors)):
      return None
    [s_x, z_x, s_out, z_out] = tensors

    last_node = m.returning_nodes[0]
    first_user = self.find_first_user(last_node)
    if first_user is None:
      return None

    # note that the shir quantizer would have tried to share the qparams
    # between the input and outputs!
    shared_qparams = s_x.item() == s_out.item() and z_x.item() == z_out.item()

    graph = self.gm.graph
    with graph.inserting_before(first_user):
      n1 = graph.call_method("int", (x,))
      if shared_qparams:
        n3 = graph.call_function(shir.int_adaptive_avg_pool2d, (n1, output_size))
        n4 = graph.call_method("to", (n3, torch.int8))
      else:
        n2 = graph.call_function(aten.sub, (n1, z_x.int().item))
        n3 = graph.call_function(shir.int_adaptive_avg_pool2d, (n2, output_size))
        n4 = graph.call_function(shir.requantize, (n3, s_x.item() / s_out.item(), z_out.item()))
    return (last_node, n4)

def rewrite_quantized_ops(gm: GraphModule):
  obj = QuantOpRewrite(gm)
  obj.rewrite()
