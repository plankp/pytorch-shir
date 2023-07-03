from torch.fx.graph_module import GraphModule
from torch.fx import subgraph_rewriter
import torch
import operator

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

def _node_is_getitem(node: torch.fx.Node, value: torch.fx.Node, idx: int) -> bool:
  if node.op != "call_function" or node.target != operator.getitem:
    return False
  if node.kwargs != {}:
    return False    # assume something strange is happening
  match node.args:
    case [v, n] if v is value and n == idx:
      return True
    case _:
      return False

def _undo_aten_decomps(gm: GraphModule):
  subgraph_rewriter.replace_pattern(gm, remat_maxpool2d_pat, remat_maxpool2d_repl)
  gm.graph.lint()

def late_rewrite(gm: GraphModule):
  _undo_aten_decomps(gm)

def early_rewrite(gm: GraphModule):
  _undo_aten_decomps(gm)

"""
Some magic to "undo" the decomposition of max_pool?d
"""

def remat_maxpool2d_pat(x, kernel_size, stride, padding, dilation):
  x = aten.max_pool2d_with_indices.default(x, kernel_size, stride, padding, dilation)
  x = x[0]
  return x

def remat_maxpool2d_repl(x, kernel_size, stride, padding, dilation):
  x = aten.max_pool2d.default(x, kernel_size, stride, padding, dilation)
  return x

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
  x = qd.dequantize_per_tensor.tensor(x, s_in, z_in, -128, 127, torch.int8)
  x = aten.relu.default(x)
  x = qd.quantize_per_tensor.tensor(x, s_out, z_out, -128, 127, torch.int8)
  return x

def qrelu_repl(x, s_in, z_in, s_out, z_out):
  x_q = torch.relu(x - z_in)
  x = qd.quantize_per_tensor.tensor(x_q.float(), s_out / s_in, z_out, -128, 127, dtype=torch.int8)
  return x

"""
(sx (X - zx)) @ (sy (Y - zy))
  = sx sy ((X - zx) @ (Y - zy))
"""

def qlinear_pat(x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  w = qd.dequantize_per_tensor.tensor(x, scl_x, zero_x, -128, 127, torch.int8)
  x = qd.dequantize_per_tensor.tensor(y, scl_y, zero_y, -127, 127, torch.int8)
  x = aten.permute.default(x, [1, 0])
  x = aten.mm.default(w, x)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, -128, 127, torch.int8)
  return x

def qlinear_repl(x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  k = scl_x * scl_y
  w_q = (x.int() - zero_x.int()) @ (y.int() - zero_y.int()).T
  x = qd.quantize_per_tensor.tensor(w_q.float(), scl_out / k, zero_out, -128, 127, dtype=torch.int8)
  return x

"""
b + (sx (X - zx)) @ (sy (Y - zy))
  = b + sx sy ((X - zx) @ (Y - zy))
  = (sx sy) ([b / (sx sy)] + (X - zx) @ (Y - zy))   <-- bias is quantized
"""

def qlinear_bias_pat(b, x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  w = qd.dequantize_per_tensor.tensor(x, scl_x, zero_x, -128, 127, torch.int8)
  x = qd.dequantize_per_tensor.tensor(y, scl_y, zero_y, -127, 127, torch.int8)
  x = aten.permute.default(x, [1, 0])
  x = aten.addmm.default(b, w, x)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, -128, 127, torch.int8)
  return x

def qlinear_bias_repl(bias, x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  k = scl_x * scl_y
  bias_q = torch.round(bias / k).int()
  w_q = (x.int() - zero_x.int()) @ (y.int() - zero_y.int()).T
  w_q = bias_q + w_q
  x = qd.quantize_per_tensor.tensor(w_q.float(), scl_out / k, zero_out, -128, 127, dtype=torch.int8)
  return x

"""
ReLU((sx (X - zx)) @ (sy (Y - zy)))
  = sx sy ReLU((X - zx) @ (Y - zy))
"""

def qlinear_relu_pat(x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  w = qd.dequantize_per_tensor.tensor(x, scl_x, zero_x, -128, 127, torch.int8)
  x = qd.dequantize_per_tensor.tensor(y, scl_y, zero_y, -127, 127, torch.int8)
  x = aten.permute.default(x, [1, 0])
  x = aten.mm.default(w, x)
  x = aten.relu.default(x)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, -128, 127, torch.int8)
  return x

def qlinear_relu_repl(x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  k = scl_x * scl_y
  w_q = torch.relu((x.int() - zero_x.int()) @ (y.int() - zero_y.int()).T)
  x = qd.quantize_per_tensor.tensor(w_q.float(), scl_out / k, zero_out, -128, 127, dtype=torch.int8)
  return x

"""
ReLU(b + (sx (X - zx)) @ (sy (Y - zy)))
  = sx sy ReLU([b / (sx sy)] + (X - zx) @ (Y - zy))
"""

def qlinear_bias_relu_pat(b, x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  w = qd.dequantize_per_tensor.tensor(x, scl_x, zero_x, -128, 127, torch.int8)
  x = qd.dequantize_per_tensor.tensor(y, scl_y, zero_y, -127, 127, torch.int8)
  x = aten.permute.default(x, [1, 0])
  x = aten.addmm.default(b, w, x)
  x = aten.relu.default(x)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, -128, 127, torch.int8)
  return x

def qlinear_bias_relu_repl(bias, x, scl_x, zero_x, y, scl_y, zero_y, scl_out, zero_out):
  k = scl_x * scl_y
  bias_q = torch.round(bias / k).int()
  w_q = (x.int() - zero_x.int()) @ (y.int() - zero_y.int()).T
  w_q = torch.relu(bias_q + w_q)
  x = qd.quantize_per_tensor.tensor(w_q.float(), scl_out / k, zero_out, -128, 127, dtype=torch.int8)
  return x

"""
CONV(sx (X - zx), sw (W - zw))
  = sx sw CONV(X - zx, W - zw)    <-- it's a repeated dot product

Translating it this way allows implementing padding as zero padding.
"""

def qconv_pat(x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
              stride, padding, dilation, transposed, output_padding, groups):
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, -128, 127, torch.int8)
  w = qd.dequantize_per_tensor.tensor(w_q, scl_w, zero_w, -127, 127, torch.int8)
  p = aten.convolution.default(
      x, w, None,
      stride, padding, dilation, transposed, output_padding, groups)
  p = qd.quantize_per_tensor.tensor(p, scl_out, zero_out, -128, 127, torch.int8)
  return p

def qconv_repl(x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
               stride, padding, dilation, transposed, output_padding, groups):
  k = scl_x * scl_w
  p_q = aten.convolution.default(
      (x_q.int() - zero_x.int()), (w_q.int() - zero_w.int()), None,
      stride, padding, dilation, transposed, output_padding, groups)

  p = qd.quantize_per_tensor.tensor(p_q.float(), scl_out / k, zero_out, -128, 127, dtype=torch.int8)
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
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, -128, 127, torch.int8)
  w = qd.dequantize_per_tensor.tensor(w_q, scl_w, zero_w, -127, 127, torch.int8)
  p = aten.convolution.default(
      x, w, bias,
      stride, padding, dilation, transposed, output_padding, groups)
  p = qd.quantize_per_tensor.tensor(p, scl_out, zero_out, -128, 127, torch.int8)
  return p

def qconv_bias_repl(bias, x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                    stride, padding, dilation, transposed, output_padding, groups):
  k = scl_x * scl_w
  bias_q = torch.round(bias / k).int()
  p_q = aten.convolution.default(
      (x_q.int() - zero_x.int()), (w_q.int() - zero_w.int()), bias_q,
      stride, padding, dilation, transposed, output_padding, groups)

  p = qd.quantize_per_tensor.tensor(p_q.float(), scl_out / k, zero_out, -128, 127, dtype=torch.int8)
  return p

"""
ReLU(CONV(sx (X - zx), sw (W - zw)))
  = sx sw ReLU(CONV(X - zx, W - zw))
"""

def qconv_relu_pat(x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                   stride, padding, dilation, transposed, output_padding, groups):
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, -128, 127, torch.int8)
  w = qd.dequantize_per_tensor.tensor(w_q, scl_w, zero_w, -127, 127, torch.int8)
  p = aten.convolution.default(
      x, w, None,
      stride, padding, dilation, transposed, output_padding, groups)
  p = aten.relu.default(p)
  p = qd.quantize_per_tensor.tensor(p, scl_out, zero_out, -128, 127, torch.int8)
  return p

def qconv_relu_repl(x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                    stride, padding, dilation, transposed, output_padding, groups):
  k = scl_x * scl_w
  p_q = aten.convolution.default(
      (x_q.int() - zero_x.int()), (w_q.int() - zero_w.int()), None,
      stride, padding, dilation, transposed, output_padding, groups)

  p_q = torch.relu(p_q)
  p = qd.quantize_per_tensor.tensor(p_q.float(), scl_out / k, zero_out, -128, 127, dtype=torch.int8)
  return p

"""
ReLU(CONV(sx (X - zx), sw (W - zw), bias))
  = sx sw ReLU((CONV(X - zx, W - zw, bias / (sx sw)))
"""

def qconv_bias_relu_pat(bias, x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                        stride, padding, dilation, transposed, output_padding, groups):
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, -128, 127, torch.int8)
  w = qd.dequantize_per_tensor.tensor(w_q, scl_w, zero_w, -127, 127, torch.int8)
  p = aten.convolution.default(
      x, w, bias,
      stride, padding, dilation, transposed, output_padding, groups)
  p = aten.relu.default(p)
  p = qd.quantize_per_tensor.tensor(p, scl_out, zero_out, -128, 127, torch.int8)
  return p

def qconv_bias_relu_repl(bias, x_q, scl_x, zero_x, w_q, scl_w, zero_w, scl_out, zero_out,
                         stride, padding, dilation, transposed, output_padding, groups):
  k = scl_x * scl_w
  bias_q = torch.round(bias / k).int()
  p_q = aten.convolution.default(
      (x_q.int() - zero_x.int()), (w_q.int() - zero_w.int()), bias_q,
      stride, padding, dilation, transposed, output_padding, groups)

  p_q = torch.relu(p_q)
  p = qd.quantize_per_tensor.tensor(p_q.float(), scl_out / k, zero_out, -128, 127, dtype=torch.int8)
  return p

"""
MAXPOOL2D(sx (X - zx)) = sx MAXPOOL2D(X - zx)
"""

def qconv_maxpool2d_pat(x_q, scl_x, zero_x, scl_out, zero_out,
                        kernel_size, stride, padding, dilation):
  x = qd.dequantize_per_tensor.tensor(x_q, scl_x, zero_x, -128, 127, torch.int8)
  x = aten.max_pool2d.default(x, kernel_size, stride, padding, dilation)
  x = qd.quantize_per_tensor.tensor(x, scl_out, zero_out, -128, 127, torch.int8)
  return x

def qconv_maxpool2d_repl(x_q, scl_x, zero_x, scl_out, zero_out,
                         kernel_size, stride, padding, dilation):
  x_q = aten.max_pool2d.default(x_q.int() - zero_x.int(), kernel_size, stride, padding, dilation, False)
  x = qd.quantize_per_tensor.tensor(x_q.float(), scl_out / scl_x, zero_out, -128, 127, dtype=torch.int8)
  return x

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

  (qconv_maxpool2d_pat, qconv_maxpool2d_repl),
])
