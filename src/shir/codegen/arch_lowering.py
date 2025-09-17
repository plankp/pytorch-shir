from shir import types, layout, config, bit_utils
import torch
from typing import Tuple, Optional, Dict
from functools import reduce
from itertools import chain

_supported_ops = {}

def register_lowering(key):
  def _magic(lowering):
    assert key not in _supported_ops, f"Operation {key} is repeatedly registered"
    _supported_ops[key] = lowering
    return lowering   # allows stacking this decorator
  return _magic

def fetch_lowering(key):
  return _supported_ops.get(key)

shin = torch.ops.shir_intrinsic
aten = torch.ops.aten
prims = torch.ops.prims

@register_lowering(shin.lstm.default)
class OperatorLSTM:
  @staticmethod
  def supports(x, ihs, hhs, bs) -> bool:
    wii, wif, wig, wio = ihs
    whi, whf, whg, who = hhs
    bi, bf, bg, bo = bs

    # TODO: if extra validation is necessary...
    return True

  @staticmethod
  def lower(x, ihs, hhs, bs) -> str:
    wii, wif, wig, wio = ihs
    whi, whf, whg, who = hhs
    bi, bf, bg, bo = bs

    # this is a 3D tensor: [batch x sequence length x input size]
    image_shape = x.meta.get("val").shape
    return "⟨TODO⟩"

@register_lowering(aten.view.default)
class OperatorView:
  @staticmethod
  def supports(a, shape) -> bool:
    return True

  @staticmethod
  def lower(a, shape) -> str:
    # reshape the metatensor to get the resulting shape.
    #
    # do this instead of using shape directly because we might have unresolved
    # (-1) lengths...
    fk = a.meta.get("val")
    nd = fk.ndim
    ys = fk.reshape(shape).shape

    # the shir expression is just a simple join-all + split-all
    se = str(a)
    for _ in range(1, nd):
      se = f"JoinOrderedStream({se})"
    for y in reversed(ys[1:]):
      se = f"SplitOrderedStream({se}, {y})"
    return se

