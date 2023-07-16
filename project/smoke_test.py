import torch
import shir_backend
import shir_intrinsic

@torch.compile(backend=shir_backend.compiler)
def fn(data, kern, bias):
  return torch.ops.aten.convolution(
      data,
      kern,
      bias,
      [1, 1],
      [1, 1],
      [1, 1],
      False,
      [0, 0],
      1
  )

data = (torch.randn(1, 1, 10, 10) * 100).to(torch.int8)
kern = (torch.randn(2, 1, 3, 3) * 100).to(torch.int8)
bias = (torch.ones(2)).to(torch.int8)
print(fn(data, kern, bias).shape)
