import torch
import torch.nn as nn
import shir

'''
@torch.compile(backend=shir.compiler)
def fn(x, y):
  return torch.ops.aten.convolution(x, y, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

x = (torch.randn(2, 1, 3, 3) * 100).to(torch.int32)
y = (torch.randn(2, 1, 3, 3) * 100).to(torch.int32)
print(fn(x, y))
'''

@torch.compile(backend=shir.compiler)
def fn(x):
  return torch.ops.shir_intrinsic.requantize_channel(x, [0.5, 0.75], 1)

x = (torch.randn(2, 2, 3, 3) * 100).to(torch.int32)
print(fn(x))
