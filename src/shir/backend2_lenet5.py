#
# Currently only contains the instruction selection logic for LeNet 5
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
    if ((qrm := extract_qaddmm_relu(n)) is not None and
        qrm[2].args[0].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrm[2].args[0].target)).to_signed().bits <= 20):
      requant, relu, addmm = qrm
      bias = addmm.args[0]
      images = addmm.args[1]
      weight = addmm.args[2]
      adjusted = _adjust_requant_param([requant.args[1]], requant.args[2])
      if (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          weight.op == "get_attr" and
          getattr(gm, weight.target).shape[0] <= 10 and
          getattr(gm, weight.target).shape[1] <= 128):
        m = getattr(gm, weight.target)
        w = m.shape[0]
        m = torch.nn.functional.pad(m, (0, 128 - m.shape[1], 0, 10 - m.shape[0]))
        setattr(gm, weight.target, torch.nn.Parameter(m, False))

        m = getattr(gm, bias.target)
        m = torch.nn.functional.pad(m, (0, 10 - m.shape[0]))
        setattr(gm, bias.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted[0], dtype=torch.int32), False))

        with graph.inserting_before(n):
          n1 = graph.get_attr(sclattr)
          n2 = graph.call_function(torch.ops._shir.lenet5_linear3, (images, weight, bias, n1, requant.args[2]))
        n.target = torch.ops.aten.pad
        n.args = (n2, [0, w - 10])
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(addmm)

      elif (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          weight.op == "get_attr" and
          getattr(gm, weight.target).shape[0] <= 90 and
          getattr(gm, weight.target).shape[1] <= 128):
        m = getattr(gm, weight.target)
        w = m.shape[0]
        m = torch.nn.functional.pad(m, (0, 128 - m.shape[1], 0, 90 - m.shape[0]))
        setattr(gm, weight.target, torch.nn.Parameter(m, False))

        m = getattr(gm, bias.target)
        m = torch.nn.functional.pad(m, (0, 90 - m.shape[0]))
        setattr(gm, bias.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted[0], dtype=torch.int32), False))

        with graph.inserting_before(n):
          n1 = graph.get_attr(sclattr)
          n2 = graph.call_function(torch.ops._shir.lenet5_linear2, (images, weight, bias, n1, requant.args[2]))
        n.target = torch.ops.aten.pad
        n.args = (n2, [0, w - 90]) # negative padding / removes the padded entries
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(addmm)

      elif (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          weight.op == "get_attr" and
          getattr(gm, weight.target).shape[0] == 120 and
          getattr(gm, weight.target).shape[1] == 400):
        m = getattr(gm, weight.target).reshape([120, 5, 80])
        m = torch.nn.functional.pad(m, (0, 128 - m.shape[2]))
        setattr(gm, weight.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted[0], dtype=torch.int32), False))

        batch = images.meta.get("val").shape[0]
        with graph.inserting_before(n):
          n1 = graph.get_attr(sclattr)
          n2 = graph.call_function(torch.ops.aten.view, (images, [batch, 5, 80]))
        n.target = torch.ops._shir.lenet5_linear1
        n.args = (n2, weight, bias, n1, requant.args[2])
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(addmm)

    elif (n.op == "call_function" and n.target == torch.ops.shir_intrinsic.int_avg_pool2d and
        len(n.args[0].users) == 1 and
        (qrc := extract_qconv_relu(n.args[0])) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 20):
      requant, relu, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])
      if (adjusted is not None and
          zp == -128 and requant.args[2] == -128 and
          stride == [1, 1] and padding == [2, 2] and dilation == [1, 1] and groups == 1 and
          list(images.meta.get("val").shape[1:]) == [1, 28, 28] and
          kernel.op == "get_attr" and
          list(getattr(gm, kernel.target).shape) == [6, 1, 5, 5]):
        m = getattr(gm, kernel.target)
        m = m.permute([0, 2, 3, 1]).reshape([6, 5 * 5 * 1])
        setattr(gm, kernel.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        batch = images.meta.get("val").shape[0]
        with graph.inserting_before(n):
          i1 = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          i2 = graph.call_function(torch.ops.aten.view, (i1, [batch, 28, 28 * 1]))
          n1 = graph.get_attr(sclattr)
          u = graph.call_function(torch.ops._shir.lenet5_conv_pool1, (i2, kernel, bias, n1, requant.args[2]))
          r = graph.call_function(torch.ops.aten.view, (u, [batch, 14, 14, 6]))
          r = graph.call_function(torch.ops.aten.permute, (r, [0, 3, 1, 2]))
        n.target = torch.ops.aten.reshape
        n.args = (r, [batch, 6, 14, 14])
        graph.erase_node(requant)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      elif (adjusted is not None and
          zp == -128 and requant.args[2] == -128 and
          stride == [1, 1] and padding == [0, 0] and dilation == [1, 1] and groups == 1 and
          list(images.meta.get("val").shape[1:]) == [6, 14, 14] and
          kernel.op == "get_attr" and
          list(getattr(gm, kernel.target).shape) == [16, 6, 5, 5]):
        m = getattr(gm, kernel.target)
        m = m.permute([0, 2, 3, 1]).reshape([16, 5 * 5 * 6])
        setattr(gm, kernel.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        batch = images.meta.get("val").shape[0]
        with graph.inserting_before(n):
          i1 = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          i2 = graph.call_function(torch.ops.aten.reshape, (i1, [batch, 14, 14 * 6]))
          n1 = graph.get_attr(sclattr)
          u = graph.call_function(torch.ops._shir.lenet5_conv_pool2, (i2, kernel, bias, n1, requant.args[2]))
          # this nasty sequence avoids the permute from messing up the
          # "contiguity" of the tensor
          r = graph.call_function(torch.ops.aten.view, (u, [batch, 5, 5, 16]))
          r = graph.call_function(torch.ops.aten.permute, (r, [0, 3, 1, 2]))
          r = graph.call_function(torch.ops.aten.reshape, (r, [batch, 16 * 5 * 5]))
        n.target = torch.ops.aten.view
        n.args = (r, [batch, 16, 5, 5])
        graph.erase_node(requant)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

  graph.lint()
  gm.recompile()
