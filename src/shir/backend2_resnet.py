#
# Currently only contains the instruction selection logic for resnet
#

import torch
import torch.nn as nn
import torch.fx as fx
from functools import reduce
from . import bit_utils
from .isel_utils import (
    extract_qconv_relu,
    extract_qaddmm_relu,
    mk_requant_param
)

def _adjust_requant_param(scales, zp):
  return mk_requant_param(scales, zp, rshamt=35)

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
    if ((qrc := extract_qconv_relu(n)) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 24):
      requant, relu, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])
      if adjusted is None:
        continue
      if requant.args[2] != -128 and relu is not None:
        continue
      if dilation != [1, 1] or groups != 1:
        continue

      if (kernel.op != "get_attr" or
          len(getattr(gm, kernel.target).shape) != 4 or
          getattr(gm, kernel.target).shape[0] % 64 != 0):
        continue

      if (padding == [3, 3] and
          list(getattr(gm, kernel.target).shape)[2:4] == [7, 7]):
        kernpad = 0
      elif (padding == [1, 1] and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3]):
        kernpad = 2
      elif (padding == [0, 0] and
          list(getattr(gm, kernel.target).shape)[2:4] == [1, 1]):
        kernpad = 3
      else:
        continue

      sclattr = create_new_param()
      setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

      rs = n.meta.get("val").shape
      batch, ich, ih, iw = images.meta.get("val").shape
      packfactor = 1
      with graph.inserting_before(n):
        # only perform input packing if doing so reduces the number of input
        # tiles to be processed.
        if ich <= 32 and iw % 2 == 0 and iw > 14:
          packfactor = 2
        if ich <= 16 and iw % 4 == 0 and iw > 14:
          packfactor = 4
        if ich <= 8 and iw % 8 == 0 and iw > 14:
          packfactor = 8

        if packfactor == 1:
          ni = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          pd = graph.call_function(torch.ops.aten.pad, (kernel, [kernpad] * 4))
          nk = graph.call_function(torch.ops.aten.permute, (pd, [0, 2, 3, 1]))

        else:
          dwidth = 64 // packfactor
          n1 = graph.call_function(torch.ops.aten.pad, (images, [0, 0, 0, 0, 0, dwidth - ich]))
          n2 = graph.call_function(torch.ops.aten.reshape, (n1, [batch, dwidth, ih, packfactor, iw // packfactor]))
          n3 = graph.call_function(torch.ops.aten.permute, (n2, [0, 2, 4, 3, 1]))
          ni = graph.call_function(torch.ops.aten.reshape, (n3, [batch, ih, iw // packfactor, 64]))

          n5 = graph.call_function(torch.ops.aten.pad, (kernel, [kernpad] * 4 + [0, dwidth - ich]))
          n6 = graph.call_function(torch.ops.aten.repeat, (n5, [1, packfactor, 1, 1]))
          nk = graph.call_function(torch.ops.aten.permute, (n6, [0, 2, 3, 1]))

        n3 = graph.get_attr(sclattr)
        n4 = graph.call_function(torch.ops._shir.resnet7x7, (ni, zp, nk, bias, n3, requant.args[2], packfactor))
        if stride[0] > 1:
          sh = graph.call_function(torch.arange, (0, ih, stride[0]))
          n4 = graph.call_function(torch.ops.aten.index_select, (n4, 1, sh))
        if stride[1] > 1:
          sw = graph.call_function(torch.arange, (0, iw, stride[1]))
          n4 = graph.call_function(torch.ops.aten.index_select, (n4, 2, sw))
        n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
      n.target = torch.ops.aten.contiguous
      n.args = (n5,)
      if relu is not None:
        graph.erase_node(relu)
      graph.erase_node(conv)

  graph.lint()
  gm.recompile()

