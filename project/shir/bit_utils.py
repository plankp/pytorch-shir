"""
A whole bunch of bit-related utilities for lowering
"""

from typing import Tuple, List
import math
import struct

def to_signed(v: int, bits=32) -> int:
  v &= (1 << bits) - 1
  if v & (1 << (bits - 1)):
    return -((1 << bits) - v)
  return v

def f32_to_bits(x: float) -> int:
  return struct.unpack(">l", struct.pack(">f", x))[0]

def is_valid_qscale(x: float) -> bool:
  return x > 0 and math.isfinite(x)

def unpack_qscale(x: float) -> Tuple[int, int]:
  bits = f32_to_bits(x)
  frac = (1 << 23) | (bits & ((1 << 23) - 1))
  shamt = 127 + 23 - (bits >> 23)   # 23 comes from (normal) mantissa
  return frac, shamt

def qscale_to_fixpoint(x: List[float]) -> Tuple[List[int], int, int]:
  # one restriction we impose is for the number of fractional bits
  # (in other words, the rounding shift amount) to be non-negative.

  def gen_shortest_qvalue(x):
    for f in x:
      assert is_valid_qscale(f), "Invalid qscale"
      frac, shamt = unpack_qscale(f)
      assert shamt >= 0, "Invalid shift amount"

      # normalize the representation by aggressively shifting right
      # while making sure the shift amount is still valid.
      while shamt >= 0 and (frac & 1) == 0:
        frac >>= 1
        shamt -= 1
      yield frac, shamt

  qvalues = list(gen_shortest_qvalue(x))
  final_scale = max((x[1] for x in qvalues))

  # with values being as "short" as possible, we now try to expand values that
  # are too short by doing the reverse: shifting left.
  N = len(qvalues)
  max_width = 0
  for i in range(N):
    frac, shamt = qvalues[i]
    # generally speaking, this shift only works because Python uses bigints.
    frac <<= final_scale - shamt
    qvalues[i] = frac
    max_width = max(max_width, frac.bit_length())
  return qvalues, max_width, final_scale
