import torch
from . import intrinsic

def flatten(tensor: torch.Tensor, start=0, end=-1) -> torch.Tensor:
  """
  Our torch.flatten that does not trigger symbolic shape usage and therefore
  is safe on a platform that doesn't support dynamic shapes.
  """

  return torch.ops.shir_intrinsic.flatten(tensor, start, end)

def qadd(
  lhs: torch.Tensor, s1: float,
  rhs: torch.Tensor, s2: float,
  z: int
) -> torch.Tensor:
  """
  Performs broadcast on the operands before forwarding it to the intrinsic
  (which does not perform broadcast)
  """

  lhs, rhs = torch.broadcast_tensors(lhs, rhs)
  return torch.ops.shir_intrinsic.qadd(lhs, s1, rhs, s2, z)
