import torch
import shir_backend

@torch.compile(backend=shir_backend.compiler)
def fn(x, y):
  return torch.ops.aten.convolution.default(
    x, y,
    torch.zeros(4, dtype=torch.int),
    [1, 1],
    [0, 0],
    [1, 1],
    False,
    [0, 0],
    1
)

print(fn(
    torch.ones(9, 2, 5, 5, dtype=torch.int),
    torch.ones(4, 2, 3, 3, dtype=torch.int)
).shape)
