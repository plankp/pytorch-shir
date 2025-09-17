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

@register_lowering(shin.host_buffer_hint.default)
class LowerShirHostBufferHint:
  @staticmethod
  def supports(a) -> bool:
    return True

  @staticmethod
  def lower(a) -> str:
    # we could just wrap it under a SolverGuidedBuffer and be done with it,
    # but add extra permutes to be consistent with the input/output behaviour.
    annot_typ = types.get_element_type(a)
    ndim = a.meta.get("val").ndim
    shape = a.meta.get("val").shape

    transpose, (h, w) = layout.pack_host_shape(shape)
    shape = [shape[x] for x in transpose]
    itr = layout.inverse_transpose(transpose)

    node = f"sg.SolverGuidedPermute({a.name}, Seq({', '.join((str(d) for d in transpose))}))"
    node = f"sg.SolverGuidedReshape(sg.SolverGuidedRebalance({node}), Seq({h}, {w}))"
    node = f"sg.SolverGuidedBuffer({node})"
    node = f"sg.SolverGuidedRebalance(sg.SolverGuidedReshape({node}, Seq({', '.join((str(d) for d in shape))})))"
    return f"sg.SolverGuidedPermute({node}, Seq({', '.join((str(d) for d in itr))}))"


