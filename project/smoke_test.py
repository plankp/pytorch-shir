import torch
import shir_backend

@torch.compile(backend=shir_backend.compiler)
def fn(x, y):
  return torch.relu(x) + 2
  # return torch.maximum(x, torch.ones([], dtype=torch.int32))

print(fn((torch.randn(2, 3) * 10).int(), (torch.randn(2, 3) * 10).int()))
