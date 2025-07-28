#
# Currently only contains the instruction selection logic for Tiny YOLO v2
#

import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Callable
from functools import reduce
from . import config, rewrites, bit_utils, types
from .isel_utils import (
    extract_qconv_leaky,
    mk_requant_param
)

def _adjust_requant_param(scales, zp):
  return mk_requant_param(scales, zp, rshamt=33)

def select(gm: fx.GraphModule):
  # since we destructively rewrite the graph,
  # try to keep it so that the types of names do not change.
  # (at least for the runtime values)

  counter = 0

  def create_new_param():
    from torch._dynamo.source import NNModuleSource, LocalSource, AttrSource
    nonlocal counter, gm

    counter += 1
    name = f"_isel_param{counter}"
    assert not hasattr(gm, name)
    assert name not in gm._param_name_to_source

    gm.register_parameter(name, None)
    gm._param_name_to_source[name] = NNModuleSource(AttrSource(LocalSource("self"), name))
    return name

  # the variations tend to happen near the end of a sequence of nodes (e.g.,
  # difference between qconv and qconv + pooling). thus, it is better to
  # traverse the fx graph in reverse order.
  graph = gm.graph
  for n in reversed(graph.nodes):
    if (n.op == "call_function" and n.target == torch.ops.shir_intrinsic.int_max_pool2d and
        len(n.args[0].users) == 1 and
        n.args[1] == [2, 2] and n.args[2] == [1, 1] and n.args[3] == [0, 0] and n.args[4] == [1, 1] and
        n.args[0].op == "call_function" and n.args[0].target == torch.ops.aten.pad and
        len(n.args[0].args) == 3 and n.args[0].kwargs == {} and
        len(n.args[0].args[0].users) == 1 and
        n.args[0].args[1] == [0, 1, 0, 1] and n.args[0].args[2] in {"replicate", "reflect"} and
        (qrc := extract_qconv_leaky(n.args[0].args[0], 3)) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 24):

      refl_pad = n.args[0]
      requant, leaky, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])

      # here, we match for the exact shape (because our slow pool impl is a bit different)
      if (adjusted is not None and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] == 13 and
          images.meta.get("val").shape[3] == 13):

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        batch, ich, ih, iw = images.meta.get("val").shape
        och, _, _, _ = getattr(gm, kernel.target).shape
        packfactor = 1
        with graph.inserting_before(n):
          if ich <= 32 and iw % (2 * 13) == 0:
            packfactor = 2
          if ich <= 16 and iw % (4 * 13) == 0:
            packfactor = 4
          if ich <= 8 and iw % (8 * 13) == 0:
            packfactor = 8

          if packfactor == 1:
            ni = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))
          else:
            dwidth = 64 // packfactor
            n1 = graph.call_function(torch.ops.aten.pad, (images, [0, 0, 0, 0, 0, dwidth - ich]))
            n2 = graph.call_function(torch.ops.aten.reshape, (n1, [batch, dwidth, ih, packfactor, iw // packfactor]))
            n3 = graph.call_function(torch.ops.aten.permute, (n2, [0, 2, 4, 3, 1]))
            ni = graph.call_function(torch.ops.aten.reshape, (n3, [batch, ih, iw // packfactor, 64]))

            n5 = graph.call_function(torch.ops.aten.pad, (kernel, [0, 0, 0, 0, 0, dwidth - ich]))
            n6 = graph.call_function(torch.ops.aten.repeat, (n5, [1, packfactor, 1, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (n6, [0, 2, 3, 1]))

          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.tiny_yolo_v2, (ni, zp, nk, bias, n3, requant.args[2], 1, packfactor, leaky is not None))

          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))

        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        graph.erase_node(refl_pad)
        graph.erase_node(requant)
        if leaky is not None: graph.erase_node(leaky)
        graph.erase_node(conv)
        continue

    if (n.op == "call_function" and n.target == torch.ops.shir_intrinsic.int_max_pool2d and
        len(n.args[0].users) == 1 and
        (qrc := extract_qconv_leaky(n.args[0], 3)) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 24):

      requant, leaky, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])

      if (adjusted is not None and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          images.meta.get("val").shape[2] % 13 == 0 and
          images.meta.get("val").shape[3] % 13 == 0):

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        batch, ich, ih, iw = images.meta.get("val").shape
        och, _, _, _ = getattr(gm, kernel.target).shape
        packfactor = 1
        outpack = 1
        with graph.inserting_before(n):
          if ich <= 32 and iw % (2 * 13) == 0:
            packfactor = 2
          if ich <= 16 and iw % (4 * 13) == 0:
            packfactor = 4
          if ich <= 8 and iw % (8 * 13) == 0:
            packfactor = 8

          if och == 16:
            outpack = 4
          if och == 32:
            outpack = 2

          if packfactor == 1:
            ni = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))
          else:
            dwidth = 64 // packfactor
            n1 = graph.call_function(torch.ops.aten.pad, (images, [0, 0, 0, 0, 0, dwidth - ich]))
            n2 = graph.call_function(torch.ops.aten.reshape, (n1, [batch, dwidth, ih, packfactor, iw // packfactor]))
            n3 = graph.call_function(torch.ops.aten.permute, (n2, [0, 2, 4, 3, 1]))
            ni = graph.call_function(torch.ops.aten.reshape, (n3, [batch, ih, iw // packfactor, 64]))

            n5 = graph.call_function(torch.ops.aten.pad, (kernel, [0, 0, 0, 0, 0, dwidth - ich]))
            n6 = graph.call_function(torch.ops.aten.repeat, (n5, [1, packfactor, 1, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (n6, [0, 2, 3, 1]))

          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.tiny_yolo_v2, (ni, zp, nk, bias, n3, requant.args[2], 2, packfactor, leaky is not None))

          if outpack == 1:
            n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
          else:
            n5 = graph.call_function(torch.ops.aten.reshape, (n4, [batch, rs[2], rs[3] // outpack, outpack, och]))
            n5 = graph.call_function(torch.ops.aten.permute, (n5, [0, 4, 1, 3, 2]))
            n5 = graph.call_function(torch.ops.aten.reshape, (n5, rs))

        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        graph.erase_node(requant)
        if leaky is not None: graph.erase_node(leaky)
        graph.erase_node(conv)

    elif ((qrc := extract_qconv_leaky(n, 3)) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 24):

      requant, leaky, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])

      if (adjusted is not None and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          images.meta.get("val").shape[2] % 13 == 0 and
          images.meta.get("val").shape[3] % 13 == 0):

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        batch, ich, ih, iw = images.meta.get("val").shape
        och, _, _, _ = getattr(gm, kernel.target).shape
        packfactor = 1
        outpack = 1
        with graph.inserting_before(n):
          if ich <= 32 and iw % (2 * 13) == 0:
            packfactor = 2
          if ich <= 16 and iw % (4 * 13) == 0:
            packfactor = 4
          if ich <= 8 and iw % (8 * 13) == 0:
            packfactor = 8

          if och == 16:
            outpack = 4
          if och == 32:
            outpack = 2

          if packfactor == 1:
            ni = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))
          else:
            dwidth = 64 // packfactor
            n1 = graph.call_function(torch.ops.aten.pad, (images, [0, 0, 0, 0, 0, dwidth - ich]))
            n2 = graph.call_function(torch.ops.aten.reshape, (n1, [batch, dwidth, ih, packfactor, iw // packfactor]))
            n3 = graph.call_function(torch.ops.aten.permute, (n2, [0, 2, 4, 3, 1]))
            ni = graph.call_function(torch.ops.aten.reshape, (n3, [batch, ih, iw // packfactor, 64]))

            n5 = graph.call_function(torch.ops.aten.pad, (kernel, [0, 0, 0, 0, 0, dwidth - ich]))
            n6 = graph.call_function(torch.ops.aten.repeat, (n5, [1, packfactor, 1, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (n6, [0, 2, 3, 1]))

          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.tiny_yolo_v2, (ni, zp, nk, bias, n3, requant.args[2], 0, packfactor, leaky is not None))

          if outpack == 1:
            n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
          else:
            n5 = graph.call_function(torch.ops.aten.reshape, (n4, [batch, rs[2], rs[3] // outpack, outpack, och]))
            n5 = graph.call_function(torch.ops.aten.permute, (n5, [0, 4, 1, 3, 2]))
            n5 = graph.call_function(torch.ops.aten.reshape, (n5, rs))

        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        if leaky is not None: graph.erase_node(leaky)
        graph.erase_node(conv)

      elif (adjusted is not None and
          stride == [1, 1] and padding == [0, 0] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [1, 1] and
          images.meta.get("val").shape[2] % 13 == 0 and
          images.meta.get("val").shape[3] % 13 == 0):

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        batch, ich, ih, iw = images.meta.get("val").shape
        och, _, _, _ = getattr(gm, kernel.target).shape
        rounded_och = (och + (64 - 1)) // 64 * 64

        # in TinyYOLO v2, this is the last layer, which is dense
        with graph.inserting_before(n):
          ni = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))

          n5 = graph.call_function(torch.ops.aten.pad, (kernel, [1, 1, 1, 1, 0, 0, 0, rounded_och - och]))
          nk = graph.call_function(torch.ops.aten.permute, (n5, [0, 2, 3, 1]))

          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.tiny_yolo_v2, (ni, zp, nk, bias, n3, requant.args[2], 0, 1, leaky is not None))

          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
          n5 = graph.call_function(torch.ops.aten.pad, (n5, [0, 0, 0, 0, 0, -(rounded_och - och)]))

        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        if leaky is not None: graph.erase_node(leaky)
        graph.erase_node(conv)

  graph.lint()
  gm.recompile()

