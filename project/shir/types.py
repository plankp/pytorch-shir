import torch
from typing import Tuple, Optional
from dataclasses import dataclass

"""
signature SHIRType:
  def name(self) -> str
"""

@dataclass(frozen=True)
class SI:
  bits: int   # should be at least 2

  def name(self) -> str:
    return f"algo.SignedIntType({self.bits})"

  def minval(self) -> int:
    return -1 << (self.bits - 1)

  def maxval(self) -> int:
    return (1 << (self.bits - 1)) - 1

  def cast(self, i: int) -> int:
    i &= ((1 << self.bits) - 1)
    if i & (1 << (self.bits - 1)):
      return -((1 << self.bits) - i)
    return i

  def to_signed(self):
    return self

@dataclass(frozen=True)
class UI:
  bits: int   # must be at least 1

  def name(self) -> str:
    return f"algo.IntType({self.bits})"

  def minval(self) -> int:
    return 0

  def maxval(self) -> int:
    return (1 << self.bits) - 1

  def cast(self, i: int) -> int:
    return i & ((1 << self.bits) - 1)

  def to_signed(self):
    # give one extra bit for the sign
    return SI(self.bits + 1)

def unpack_int_type(ty) -> Tuple[bool, int]:
  match ty:
    case SI(bits):
      return (True, bits)
    case UI(bits):
      return (False, bits)
    case _:
      return None

@dataclass(frozen=True)
class Seq:
  elts: int         # must be at least 1
  of: 'typing.Any'  # really should be a type defined here

  def name(self) -> str:
    return f"algo.SeqType({self.of.name()}, {self.elts})"

_table = {
  torch.uint8: UI(8),
  torch.int8: SI(8),
  torch.int16: SI(16),
  torch.int32: SI(32),
  torch.int64: SI(64),
}

def get_scalar_type(torch_ty):
  return _table.get(torch_ty)

def get_element_type(node: torch.fx.Node):
  tinfo = node.meta.get("val")
  return get_scalar_type(tinfo.dtype)

def has_shir_type(node: torch.fx.Node) -> bool:
  return get_element_type(node) is not None

def get_tensor_type(node: torch.fx.Node, dimslice=slice(None)):
  acc = get_element_type(node)
  if acc is not None:
    tinfo = node.meta.get("val")
    for i in reversed(tinfo.shape[dimslice]):
      acc = Seq(acc, i)
  return acc
