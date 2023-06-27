from torch.fx.graph_module import GraphModule
from torch.fx import subgraph_rewriter
import torch

# don't match or emit prims ops at this level!
aten = torch.ops.aten
qd = torch.ops.quantized_decomposed

# we'll register rewrites into this list
# earlier patterns are matched first
_rewrite_patterns = []

def rewrite(gm: GraphModule):
  for (pat, repl) in _rewrite_patterns:
    subgraph_rewriter.replace_pattern(gm, pat, repl)
  gm.graph.lint()

"""
def qd.quantize_per_tensor(
  x: float,
  s: positive float,
  z: T,
  min: T,
  max: T,
  type_of_t
) -> T:
  return clamp(round(x / s + z), min, max).to(type_of_T)

def qd.dequantize_per_tensor(
  x: T,
  s: positive float,
  z: T,
  min: T,
  max: T,
  type_of_t
) -> float:
  return s * (x - z)
"""

"""
ReLU(sx * (x - zx))
  = sx * ReLU(x - zx)
"""

def qrelu_pat(x, s_in, z_in, s_out, z_out):
  x = qd.dequantize_per_tensor.tensor(x, s_in, z_in, 0, 127, torch.uint8)
  x = aten.relu.default(x)
  x = qd.quantize_per_tensor.tensor(x, s_out, z_out, 0, 127, torch.uint8)
  return x

def qrelu_repl(x, s_in, z_in, s_out, z_out):
  x = torch.relu(x - z_in)
  # the following is functionally the same as quantize_per_tensor, except we
  # can't use that since x is uint8 and not float32.
  k = s_in / s_out
  x = torch.clamp(torch.round(k * x + z_out), 0, 127).to(torch.uint8)
  return x

"""
(sx (X - zx)) @ (sy (Y - zy))
  = sx sy ((X - zx) @ (Y - zy))
"""

def qlinear_pat(x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  w = qd.dequantize_per_tensor.tensor(x, scl_x, zero_x, 0, 127, torch.uint8)
  x = qd.dequantize_per_tensor.tensor(y, scl_y, zero_y, -128, 127, torch.int8)
  x = aten.permute.default(x, [1, 0])
  x = aten.mm.default(w, x)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, 0, 127, torch.uint8)
  return x

def qlinear_repl(x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  k = scl_x * scl_y
  w_q = (x - zero_x).int() @ (y - zero_y).int().T
  x = torch.clamp(torch.round(w_q * (k / scl_out) + zero_out), 0, 127).to(torch.uint8)
  return x

"""
b + (sx (X - zx)) @ (sy (Y - zy))
  = b + sx sy ((X - zx) @ (Y - zy))
  = (sx sy) ([b / (sx sy)] + (X - zx) @ (Y - zy))   <-- bias is quantized
"""

def qlinear_bias_pat(b, x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  w = qd.dequantize_per_tensor.tensor(x, scl_x, zero_x, 0, 127, torch.uint8)
  x = qd.dequantize_per_tensor.tensor(y, scl_y, zero_y, -128, 127, torch.int8)
  x = aten.permute.default(x, [1, 0])
  x = aten.addmm.default(b, w, x)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, 0, 127, torch.uint8)
  return x

def qlinear_bias_repl(bias, x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  k = scl_x * scl_y
  bias_q = torch.round(bias / k).int()
  w_q = (x - zero_x).int() @ (y - zero_y).int().T
  w_q = bias_q + w_q
  x = torch.clamp(torch.round(w_q * (k / scl_out) + zero_out), 0, 127).to(torch.uint8)
  return x

"""
ReLU((sx (X - zx)) @ (sy (Y - zy)))
  = sx sy ReLU((X - zx) @ (Y - zy))
"""

def qlinear_relu_pat(x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  w = qd.dequantize_per_tensor.tensor(x, scl_x, zero_x, 0, 127, torch.uint8)
  x = qd.dequantize_per_tensor.tensor(y, scl_y, zero_y, -128, 127, torch.int8)
  x = aten.permute.default(x, [1, 0])
  x = aten.mm.default(w, x)
  x = aten.relu.default(x)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, 0, 127, torch.uint8)
  return x

def qlinear_relu_repl(x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  k = scl_x * scl_y
  w_q = torch.relu((x - zero_x).int() @ (y - zero_y).int().T)
  x = torch.clamp(torch.round(w_q * (k / scl_out) + zero_out), 0, 127).to(torch.uint8)
  return x

"""
ReLU(b + (sx (X - zx)) @ (sy (Y - zy)))
  = sx sy ReLU([b / (sx sy)] + (X - zx) @ (Y - zy))
"""

def qlinear_bias_relu_pat(b, x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  w = qd.dequantize_per_tensor.tensor(x, scl_x, zero_x, 0, 127, torch.uint8)
  x = qd.dequantize_per_tensor.tensor(y, scl_y, zero_y, -128, 127, torch.int8)
  x = aten.permute.default(x, [1, 0])
  x = aten.addmm.default(b, w, x)
  x = aten.relu.default(x)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, 0, 127, torch.uint8)
  return x

def qlinear_bias_relu_repl(bias, x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  k = scl_x * scl_y
  bias_q = torch.round(bias / k).int()
  w_q = (x - zero_x).int() @ (y - zero_y).int().T
  w_q = torch.relu(bias_q + w_q)
  x = torch.clamp(torch.round(w_q * (k / scl_out) + zero_out), 0, 127).to(torch.uint8)
  return x

_rewrite_patterns.extend([
  (qrelu_pat, qrelu_repl),
  (qlinear_pat, qlinear_repl),
  (qlinear_bias_pat, qlinear_bias_repl),
  (qlinear_relu_pat, qlinear_relu_repl),
  (qlinear_bias_relu_pat, qlinear_bias_relu_repl),
])
