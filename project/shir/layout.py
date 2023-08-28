"""
Deals with everything memory image / memory layout related
(Hopefully...)
"""

import torch
from typing import List, Optional, Tuple
from . import types, config
from functools import reduce
from dataclasses import dataclass
import mmap
import os

_SUPPORTED_TORCH_TYPES = {
  torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
}

# this is good enough for our purposes
_MEMORY_LAYOUT_TYPE_MAP = {
  "u8": (types.UI(8), torch.uint8),
  "s8": (types.SI(8), torch.int8),
  "s16": (types.SI(16), torch.int16),
  "s32": (types.SI(32), torch.int32),
  "s64": (types.SI(64), torch.int64),
}

"""
Our representation of SHIR's MemoryLayout class.
"""

@dataclass(frozen=True)
class LayoutEntry:
  name      : str
  _ty       : str   # the undecoded type (must be valid)
  address   : int   # where the data starts at (cachelines)
  outer     : int   # number of rows
  inner     : int   # number of cachelines for each row

  def get_shir_type(self):
    return _MEMORY_LAYOUT_TYPE_MAP[self._ty][0]

  def get_torch_type(self):
    return _MEMORY_LAYOUT_TYPE_MAP[self._ty][1]

  def cachelines(self) -> int:
    return self.outer * self.inner

  def from_buffer(self, buffer) -> torch.Tensor:
    bytes_per_cl = config.CACHELINE_BITS // 8
    return torch.frombuffer(
      buffer,
      dtype=torch.int8,
      offset=self.address * bytes_per_cl,
      count=self.cachelines() * bytes_per_cl,
    ).view(self.outer, -1).view(self.get_torch_type())

class MemoryLayout:
  def __init__(self, entries: List[LayoutEntry]):
    self._entries = entries
    self._cached_cachelines = None

  def get_entry(self, name: str) -> Optional[LayoutEntry]:
    # linear search for now, could cache a LUT if needed
    for entry in self._entries:
      if entry.name == name:
        return entry
    return None

  def cachelines(self) -> int:
    result = self._cached_cachelines
    if result is None:
      result = 0
      for entry in self._entries:
        result = max(result, entry.address + entry.cachelines())
      self._cached_cachelines = result
    return result

  def bytes_needed(self, round_to_page=True) -> int:
    n = config.CACHELINE_BITS // 8 * self.cachelines()
    if round_to_page:
      n = (n + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE

    return n

def reshape_size_to_matrix(t: torch.Size) -> Tuple[int, int]:
  ndim = len(t)
  inner = outer = 1
  if ndim == 1:
    inner = t[0]
  elif ndim > 1:
    outer = t[0]
    inner = reduce(lambda x, y: x * y, t[1:])
  return (outer, inner)

def reshape_to_matrix(t: torch.Tensor) -> torch.Tensor:
  if t.ndim < 2:
    return t.reshape((1, -1))
  if t.ndim > 2:
    return t.reshape((t.size(0), -1))
  return t

def tensor_to_matrix_csv(t: torch.Tensor, f):
  assert t.dtype in _SUPPORTED_TORCH_TYPES, "Tensor dtype is not supported"

  t = reshape_to_matrix(t)
  outer_len = t.size(0)
  inner_len = t.size(1)

  for i in range(outer_len):
    for j in range(inner_len - 1):
      print(t[i, j].item(), ",", sep="", end="", file=f)
    print(t[i, inner_len - 1].item(), file=f)

def read_layout_file(fname: str) -> MemoryLayout:
  entries = []
  with open(fname, "r") as f:
    while True:
      line1 = f.readline()
      if not line1:
        break

      line2 = f.readline()
      (name, addr, ty) = line1.rstrip().split("\t")
      (inner, outer) = [int(d) for d in line2.rstrip().split(",")]
      entries.append(LayoutEntry(name, ty, int(addr, 16), outer, inner))
  return MemoryLayout(entries)

def read_memory_dump(fname: str, entry: LayoutEntry, inner_len: int) -> torch.Tensor:
  result = torch.empty((entry.outer, inner_len), dtype=entry.get_torch_type())
  ety = entry.get_shir_type()

  # a memory dump (may) start off with a few lines of comments,
  # it's all lines of cachelines as hex nibbles (so divide by 4) + '\n'
  chars_per_line = config.CACHELINE_BITS // 4 + 1

  # use binary mode for fseek sanity (even if it may not be needed).
  with open(fname, "rb") as f:
    # it may start with a few lines of comments, so skip over those first.
    # we cheat a bit and claim it's a comment as soon as we see a slash.
    while f.read(1) == b'/':
      f.readline()

    # at this point, we read sth that wasn't a comment.
    # unread that since it must be a line of data.
    # after the unread, we to jump ahead to where the entry starts.
    f.seek(-1 + chars_per_line * entry.address, os.SEEK_CUR)

    # at this point, we just repeatedly readline and load the data into the
    # result tensor.
    for outer in range(entry.outer):
      inner = 0
      for line in range(entry.inner):
        line = f.readline().rstrip()
        line_data = int(line, base=16)

        consumed_bits = 0
        while inner < inner_len and consumed_bits < config.CACHELINE_BITS:
          # the line goes from high memory to low memory.
          # that means we just need to extract the low part.
          result[outer, inner] = ety.cast(line_data)
          inner += 1
          line_data >>= ety.bits
          consumed_bits += ety.bits

  return result
