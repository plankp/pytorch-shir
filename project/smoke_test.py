import torch
import shir_backend
import shir_intrinsic

@torch.compile(backend=shir_backend.compiler)
def fn(data, f):
  return torch.ops.shir_intrinsic.requantize_channel(
      data,
      f,
      -10
  )

data = (torch.randn(5, 3, 10, 10) * 100).to(torch.int32)
kern = [0.12663674354553223, 0.11184310913085938, 0.007626994047313929]
print(fn(data, kern).shape)
