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
  "flatten(Tensor self, int start, int end) -> Tensor"
)

def flatten_bridge(tensor, start=1, end=-1):
  return torch.ops.shir_intrinsic.flatten(tensor, start, end)

@impl(shir_intrinsic_lib, "flatten", "CompositeExplicitAutograd")
def flatten(self, start, end):
  # XXX: we DON'T want to use torch.flatten in case someone (we) monkey
  # patches flatten, which would cause an infinite loop.
  #
  # instead, we want to use something that is unlikely patched away,
  # such as reshape.

  ashape = self.shape
  if not ashape:
    # flattening a scalar results in a tensor!
    ashape = [1]

  # negative start end indexes the axes in reverse
  if start < 0:
    start = len(ashape) + start
  if end < 0:
    end = len(ashape) + end

  assert 0 <= start <= end < len(ashape), f"Expect 0 <= start <= end < {len(ashape)}"

  # axes [start, end] are squished together
  new_shape = [
    *ashape[:start],
    reduce(lambda x, y: x * y, ashape[start:end + 1]),
    *ashape[end + 1:]
  ]

  return torch.reshape(self, new_shape)

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

def qadd_broadcast(self, s1, rhs, s2, z):
  """
  Identical to the intrinsics version, except the shapes for self and rhs may
  be different, provided that they obey broadcasting rules.
  """

  lhs, rhs = torch.broadcast_tensors(self, rhs)
  return torch.ops.shir_intrinsic.qadd(self, s1, rhs, s2, z)

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
  assert self.dtype in {torch.int8, torch.int32}
  return aten.max_pool2d(self.float(), kern_size, stride, pad, dilation).to(self.dtype)

shir_intrinsic_lib.define(
  "int_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor"
)

@impl(shir_intrinsic_lib, "int_adaptive_avg_pool2d", "CompositeExplicitAutograd")
def int_adaptive_avg_pool2d(self, output_size):
  assert self.dtype == torch.int32
  return aten._adaptive_avg_pool2d(self.float(), output_size).to(self.dtype)

shir_intrinsic_lib.define(
  "int_mean(Tensor self, int[]? dim, bool keepDim) -> Tensor"
)

@impl(shir_intrinsic_lib, "int_mean", "CompositeExplicitAutograd")
def int_mean(self, dim, keepDim):
  # we sneak a round it to get an answer that is "closer" to SHIR
  assert self.dtype == torch.int32
  return torch.round(aten.mean.dim(self.float(), dim, keepDim)).to(self.dtype)
