#
# Currently only contains the instruction selection logic for VGG
#

import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Callable
from functools import reduce
from . import config, rewrites, bit_utils, types
from .isel_utils import (
    extract_qconv_relu,
    extract_qaddmm_relu,
    mk_requant_param
)

def _adjust_requant_param(scales, zp):
  return mk_requant_param(scales, zp)

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
        (qrc := extract_qconv_relu(n.args[0])) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 24 and
        n.args[1] == [2, 2] and n.args[2] == [2, 2] and n.args[3] == [0, 0] and n.args[4] == [1, 1]):
      requant, relu, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])
      if (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] % 14 == 0 and
          images.meta.get("val").shape[3] % 14 == 0):

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        batch, ich, ih, iw = images.meta.get("val").shape
        packfactor = 1
        with graph.inserting_before(n):
          if ich <= 32 and iw % (2 * 14) == 0:
            packfactor = 2
          if ich <= 16 and iw % (4 * 14) == 0:
            packfactor = 4
          if ich <= 8 and iw % (8 * 14) == 0:
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
          n4 = graph.call_function(torch.ops._shir.conv3x3p1b14x64, (ni, zp, nk, bias, n3, requant.args[2], True, packfactor))
          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        graph.erase_node(requant)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      elif (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] % 8 == 0 and
          images.meta.get("val").shape[3] % 8 == 0):
        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        with graph.inserting_before(n):
          n1 = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          n2 = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))
          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.conv3x3p1b8x64, (n1, zp, n2, bias, n3, requant.args[2], True))
          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        graph.erase_node(requant)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

    elif ((qrc := extract_qconv_relu(n)) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 24):
      requant, relu, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])
      if (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] % 14 == 0 and
          images.meta.get("val").shape[3] % 14 == 0):

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        batch, ich, ih, iw = images.meta.get("val").shape
        packfactor = 1
        with graph.inserting_before(n):
          if ich <= 32 and iw % (2 * 14) == 0:
            packfactor = 2
          if ich <= 16 and iw % (4 * 14) == 0:
            packfactor = 4
          if ich <= 8 and iw % (8 * 14) == 0:
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
          n4 = graph.call_function(torch.ops._shir.conv3x3p1b14x64, (ni, zp, nk, bias, n3, requant.args[2], False, packfactor))
          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      elif (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] % 8 == 0 and
          images.meta.get("val").shape[3] % 8 == 0):
        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        with graph.inserting_before(n):
          n1 = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          n2 = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))
          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.conv3x3p1b8x64, (n1, zp, n2, bias, n3, requant.args[2], False))
          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      else:
        print(f"driver::isel: skipping qconv node {requant}:")

    elif ((qrm := extract_qaddmm_relu(n)) is not None and
        qrm[2].args[0].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrm[2].args[0].target)).to_signed().bits <= 24):
      requant, relu, addmm = qrm
      bias = addmm.args[0]
      images = addmm.args[1]
      weight = addmm.args[2]
      adjusted = _adjust_requant_param([requant.args[1]], requant.args[2])
      if (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          weight.op == "get_attr"):
        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        j, k = getattr(gm, weight.target).shape
        i, _ = n.meta.get("val").shape

        # ensure the k dimension has form 3 x 3 x (i_tiles * 64)
        # and    the j dimension has form o_tiles * 64
        i_tiles = (k + (3 * 3 * 64 - 1)) // (3 * 3 * 64)
        o_tiles = (j + (64 - 1)) // 64
        i_pad = i_tiles * 3 * 3 * 64 - k
        o_pad = o_tiles * 64 - j

        with graph.inserting_before(n):
          # try to push the batch dimension inwards to allow weight reusage
          # push to the height dimension first to avoid transpose.
          extra_h = 1
          extra_w = 1
          leftover_i = i
          while extra_h < (14 // 3) and leftover_i % 2 == 0:
            extra_h *= 2
            leftover_i //= 2
          while extra_w < (14 // 3) and leftover_i % 2 == 0:
            extra_w *= 2
            leftover_i //= 2

          n1 = graph.call_function(torch.ops.aten.pad, (images, [0, i_pad]))
          n2 = graph.call_function(torch.ops.aten.view, (n1, [leftover_i, extra_h, extra_w, 3, 3, i_tiles * 64]))
          n2 = graph.call_function(torch.ops.aten.permute, (n2, [0, 1, 3, 2, 4, 5]))
          n2 = graph.call_function(torch.ops.aten.reshape, (n2, [leftover_i, extra_h * 3, extra_w * 3, i_tiles * 64]))
          n3 = graph.call_function(torch.ops.aten.pad, (weight, [0, i_pad, 0, o_pad]))
          n4 = graph.call_function(torch.ops.aten.view, (n3, [o_tiles * 64, 3, 3, i_tiles * 64]))
          n5 = graph.get_attr(sclattr)
          n6 = graph.call_function(torch.ops._shir.conv3x3p1b14x64, (n2, None, n4, bias, n5, requant.args[2], False, 1))
          n7 = graph.call_function(torch.ops.aten.pad, (n6, [0, -o_pad]))
          slice_h = None
          if extra_h > 1:
            slice_h = graph.call_function(torch.arange, (0, extra_h * 3, 3))
          slice_w = None
          if extra_w == extra_h:
            slice_w = slice_h
          elif extra_w > 1:
            slice_w = graph.call_function(torch.arange, (0, extra_w * 3, 3))
          n8 = n7
          if slice_h is not None:
            n8 = graph.call_function(torch.ops.aten.index_select, (n8, 1, slice_h))
          if slice_w is not None:
            n8 = graph.call_function(torch.ops.aten.index_select, (n8, 2, slice_w))

        n.target = torch.ops.aten.view
        n.args = (n8, [i, j])
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(addmm)

  graph.lint()
  gm.recompile()
