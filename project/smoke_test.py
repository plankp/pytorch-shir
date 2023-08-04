import torch
import torch.nn as nn
import shir

@torch.compile(backend=shir.compiler)
def fn(data):
  return torch.ops.shir_intrinsic.int_mean(data, [-1, -2], True)

data = (torch.randn(5, 3, 10, 10) * 100).to(torch.int32)
print(fn(data).shape)
