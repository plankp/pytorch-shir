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

"""
CONV(sx (X - zx), sw (W - zw))
  = sx sw CONV(X - zx, W - zw)    <-- it's a repeated dot product

Translating it this way allows implementing padding as zero padding.
"""

def qconv_pat(x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
              stride, padding, dilation, transposed, output_padding, groups):
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, 0, 127, torch.uint8)
  w = qd.dequantize_per_tensor.tensor(w_q, scl_w, zero_w, -128, 127, torch.int8)
  p = aten.convolution.default(
      x, w, None,
      stride, padding, dilation, transposed, output_padding, groups)
  p = qd.quantize_per_tensor.tensor(p, scl_out, zero_out, 0, 127, torch.uint8)
  return p

def qconv_repl(x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
               stride, padding, dilation, transposed, output_padding, groups):
  k = scl_x * scl_w
  p_q = aten.convolution.default(
      (x_q - zero_x).int(), (w_q - zero_w).int(), None,
      stride, padding, dilation, transposed, output_padding, groups)

  p = torch.clamp(torch.round(p_q * (k / scl_out) + zero_out), 0, 127).to(torch.uint8)
  return p

"""
CONV(sx (X - zx), sw (W - zw), bias)
  = sx sw (CONV(X - zx, W - zw, bias / (sx sw))

As it turns out, the shape of bias makes it pretty difficult to decompose it
into broadcasted add:
After CONV, we have T[N, OutChan, D1, D2, ...].
We need to broadcast bias to that shape.
The difficult part is we don't know the number of dimesions.
"""

def qconv_bias_pat(bias, x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                   stride, padding, dilation, transposed, output_padding, groups):
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, 0, 127, torch.uint8)
  w = qd.dequantize_per_tensor.tensor(w_q, scl_w, zero_w, -128, 127, torch.int8)
  p = aten.convolution.default(
      x, w, bias,
      stride, padding, dilation, transposed, output_padding, groups)
  p = qd.quantize_per_tensor.tensor(p, scl_out, zero_out, 0, 127, torch.uint8)
  return p

def qconv_bias_repl(bias, x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                    stride, padding, dilation, transposed, output_padding, groups):
  k = scl_x * scl_w
  bias_q = torch.round(bias / k).int()
  p_q = aten.convolution.default(
      (x_q - zero_x).int(), (w_q - zero_w).int(), bias_q,
      stride, padding, dilation, transposed, output_padding, groups)

  p = torch.clamp(torch.round(p_q * (k / scl_out) + zero_out), 0, 127).to(torch.uint8)
  return p

"""
ReLU(CONV(sx (X - zx), sw (W - zw)))
  = sx sw ReLU(CONV(X - zx, W - zw))
"""

def qconv_relu_pat(x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
              stride, padding, dilation, transposed, output_padding, groups):
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, 0, 127, torch.uint8)
  w = qd.dequantize_per_tensor.tensor(w_q, scl_w, zero_w, -128, 127, torch.int8)
  p = aten.convolution.default(
      x, w, None,
      stride, padding, dilation, transposed, output_padding, groups)
  p = aten.relu.default(p)
  p = qd.quantize_per_tensor.tensor(p, scl_out, zero_out, 0, 127, torch.uint8)
  return p

def qconv_relu_repl(x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
               stride, padding, dilation, transposed, output_padding, groups):
  k = scl_x * scl_w
  p_q = aten.convolution.default(
      (x_q - zero_x).int(), (w_q - zero_w).int(), None,
      stride, padding, dilation, transposed, output_padding, groups)

  p_q = torch.relu(p_q)
  p = torch.clamp(torch.round(p_q * (k / scl_out) + zero_out), 0, 127).to(torch.uint8)
  return p

"""
ReLU(CONV(sx (X - zx), sw (W - zw), bias))
  = sx sw ReLU((CONV(X - zx, W - zw, bias / (sx sw)))
"""

def qconv_bias_relu_pat(bias, x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                   stride, padding, dilation, transposed, output_padding, groups):
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, 0, 127, torch.uint8)
  w = qd.dequantize_per_tensor.tensor(w_q, scl_w, zero_w, -128, 127, torch.int8)
  p = aten.convolution.default(
      x, w, bias,
      stride, padding, dilation, transposed, output_padding, groups)
  p = aten.relu.default(p)
  p = qd.quantize_per_tensor.tensor(p, scl_out, zero_out, 0, 127, torch.uint8)
  return p

def qconv_bias_relu_repl(bias, x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                    stride, padding, dilation, transposed, output_padding, groups):
  k = scl_x * scl_w
  bias_q = torch.round(bias / k).int()
  p_q = aten.convolution.default(
      (x_q - zero_x).int(), (w_q - zero_w).int(), bias_q,
      stride, padding, dilation, transposed, output_padding, groups)

  p_q = torch.relu(p_q)
  p = torch.clamp(torch.round(p_q * (k / scl_out) + zero_out), 0, 127).to(torch.uint8)
  return p

_rewrite_patterns.extend([
  (qrelu_pat, qrelu_repl),

  (qlinear_pat, qlinear_repl),
  (qlinear_bias_pat, qlinear_bias_repl),
  (qlinear_relu_pat, qlinear_relu_repl),
  (qlinear_bias_relu_pat, qlinear_bias_relu_repl),

  (qconv_pat, qconv_repl),
  (qconv_bias_pat, qconv_bias_repl),
  (qconv_relu_pat, qconv_relu_repl),
  (qconv_bias_relu_pat, qconv_bias_relu_repl),
])
