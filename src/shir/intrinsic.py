import torch
from torch.library import Library, impl
from functools import reduce

shir_intrinsic_lib = Library("shir_intrinsic", "DEF")

aten = torch.ops.aten
qd = torch.ops.quantized_decomposed

# CompositeExplicitAutograd is needed becase we later define a Meta version.
# We define both instead of using the default to avoid
# CompositeImplicitAutograd from kicking in and decomposing the intrinsic.
#
# (and we define the CEA overloads for testing purposes)

shir_intrinsic_lib.define(
  "requantize(Tensor self, float s, int z) -> Tensor"
)

@impl(shir_intrinsic_lib, "requantize", "CompositeExplicitAutograd")
def requantize(self, s, z):
  return qd.quantize_per_tensor(self.float(), 1 / s, z, -128, 127, torch.int8)

@impl(shir_intrinsic_lib, "requantize", "Meta")
def requantize_meta(self, s, z):
  assert self.dtype == torch.int32
  assert isinstance(s, float)
  assert isinstance(z, int)

  return torch.empty_like(self, dtype=torch.int8)

shir_intrinsic_lib.define(
  "requantize_channel(Tensor self, float[] scale, int z) -> Tensor"
)

@impl(shir_intrinsic_lib, "requantize_channel", "CompositeExplicitAutograd")
def requantize_channel(self, s, z):
  c = self.size(1)
  return qd.quantize_per_channel(
    self.float(), 1 / torch.Tensor(s), torch.tensor(z).expand(c),
    1, -128, 127, torch.int8
  )

@impl(shir_intrinsic_lib, "requantize_channel", "Meta")
def requantize_channel_meta(self, s, z):
  assert self.dtype == torch.int32
  assert isinstance(z, int)
  assert self.ndim > 1 and self.size(1) == len(s)

  return torch.empty_like(self, dtype=torch.int8)

shir_intrinsic_lib.define(
  "qadd(Tensor self, float s1, Tensor rhs, float s2, int z) -> Tensor"
)

@impl(shir_intrinsic_lib, "qadd", "CompositeExplicitAutograd")
def qadd(self, s1, rhs, s2, z):
  # do the requantization step ourselves
  return torch.clamp(torch.round(self * s1 + rhs * s2) + z, -128, 127).to(torch.int8)

@impl(shir_intrinsic_lib, "qadd", "Meta")
def qadd_meta(self, s1, rhs, s2, z):
  # disallow implicit broadcasting because prims can't help us here
  assert self.shape == rhs.shape
  assert self.dtype == rhs.dtype == torch.int32
  assert isinstance(s1, float) and isinstance(s2, float)
  assert isinstance(z, int)

  return torch.empty_like(self, dtype=torch.int8)

shir_intrinsic_lib.define(
  "qconv(Tensor self, int zp, Tensor weights, Tensor bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor"
)

@impl(shir_intrinsic_lib, "qconv", "CompositeExplicitAutograd")
def qconv_meta(self, zp, weights, bias, stride, padding, dilation, groups):
  assert bias.dtype == torch.int32
  assert self.dtype == weights.dtype == torch.int8

  # trick is to pad by the zero point!
  # we need to turn padding of [A, B, C] into [A, A, B, B, C, C].
  padded = aten.constant_pad_nd(self.float(), [y for x in padding for y in [x, x]], value=zp)
  return aten.convolution(
    padded, weights.float(), bias.float(),
    stride, [0], dilation, False, [0], groups
  ).int()

shir_intrinsic_lib.define(
  "int_addmm(Tensor self, Tensor lhs, Tensor rhs) -> Tensor"
)

@impl(shir_intrinsic_lib, "int_addmm", "CompositeExplicitAutograd")
def int_addmm(self, lhs, rhs):
  return aten.addmm(self, lhs.int(), rhs.int().T)

@impl(shir_intrinsic_lib, "int_addmm", "Meta")
def int_addmm_meta(self, lhs, rhs):
  # self: i32[j], lhs: i8[i, k], rhs: i8[j, k]
  assert self.dtype == torch.int32
  assert lhs.dtype == rhs.dtype == torch.int8
  assert self.ndim == 1 and lhs.ndim == 2 and rhs.ndim == 2
  assert lhs.shape[1] == rhs.shape[1]
  assert rhs.shape[0] == self.shape[0]

  return torch.empty(lhs.shape[0], rhs.shape[0],
                     dtype=torch.int32, device="meta")

shir_intrinsic_lib.define(
  "int_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation) -> Tensor"
)

@impl(shir_intrinsic_lib, "int_max_pool2d", "CompositeExplicitAutograd")
def int_max_pool2d(self, kern_size, stride, pad, dilation):
  assert self.dtype == torch.int8
  return aten.max_pool2d(self.float(), kern_size, stride, pad, dilation).to(self.dtype)

shir_intrinsic_lib.define(
  "int_avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding) -> Tensor"
)

@impl(shir_intrinsic_lib, "int_avg_pool2d", "CompositeExplicitAutograd")
def int_avg_pool2d(self, kernel_size, stride, padding):
  assert self.dtype == torch.int8
  assert len(kernel_size) == len(stride) == len(padding) == 2
  return torch.round(aten.avg_pool2d(self.float(), kernel_size, stride, padding)).to(self.dtype)

shir_intrinsic_lib.define(
  "int_mean(Tensor self, int[]? dim, bool keepDim) -> Tensor"
)

@impl(shir_intrinsic_lib, "int_mean", "CompositeExplicitAutograd")
def int_mean(self, dim, keepDim):
  # we sneak a round it to get an answer that is "closer" to SHIR
  assert self.dtype == torch.int8
  return torch.round(aten.mean.dim(self.float(), dim, keepDim)).to(self.dtype)

shir_intrinsic_lib.define(
  "host_buffer_hint(Tensor self) -> Tensor"
)

@impl(shir_intrinsic_lib, "host_buffer_hint", "CompositeExplicitAutograd")
def host_buffer_hint(self):
  return self
