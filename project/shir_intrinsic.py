import torch
from torch.library import Library, impl

shir_intrinsic_lib = Library("shir_intrinsic", "DEF")

shir_intrinsic_lib.define(
  "requantize(Tensor self, float s, int z) -> Tensor"
)

aten = torch.ops.aten
qd = torch.ops.quantized_decomposed

# CompositeExplicitAutograd is needed becase we later define a Meta version.
# We define both instead of using the default to avoid
# CompositeImplicitAutograd from kicking in and decomposing the intrinsic.
#
# (and we define the CEA overloads for testing purposes)

@impl(shir_intrinsic_lib, "requantize", "CompositeExplicitAutograd")
def requantize(self, s, z):
  return qd.quantize_per_tensor(self.float(), 1 / s, z, -128, 127, torch.int8)

@impl(shir_intrinsic_lib, "requantize", "Meta")
def requantize_meta(self, s, z):
  assert self.dtype == torch.int32
  assert isinstance(s, float)
  assert isinstance(z, int)

  return torch.empty_like(self, dtype=torch.int8)
