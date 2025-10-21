#
# Currently only contains the instruction selection logic for resnet using 3x3 convolution
#

import torch
import torch.nn as nn
import torch.fx as fx
from functools import reduce
from torch.ao.quantization.pt2e.utils import _get_all_arguments
from . import bit_utils
from .isel_utils import (
    match_quant_per_tensor,
    match_dequant_per_tensor,
    match_quant_per_channel,
    match_dequant_per_channel,
    match_qlinear,
    match_qconv2d,
    match_qconv2d_add,
    extract_attr,
    mk_requant_param
)

# match the quantize operations ourselves
REQUIRE_QUANT_REWRITE = False

def _mk_act_rqparam(scales, zp):
  return mk_requant_param(scales, zp, qw=28, rshamt=33)

def _mk_rsd_rqparam(scales, zp):
  return mk_requant_param(scales, zp, qw=28, rshamt=24)

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
    if m := match_qconv2d_add(gm, n):
      n_dq_img, n_dq_wgt, n_conv, n_dq_rsd, n_add, n_relu = m
      _, _, n_bias, stride, padding, dilation, groups = _get_all_arguments(
          n_conv.args, n_conv.kwargs, n_conv.target._schema.arguments)

      qo_info = match_quant_per_tensor(gm, n, -128, 127, torch.int8)
      qx_info = match_dequant_per_tensor(gm, n_dq_img, -128, 127, torch.int8)
      qr_info = match_dequant_per_tensor(gm, n_dq_rsd, -128, 127, torch.int8)
      if qr_info is None:
        continue
      if qw_info := match_dequant_per_tensor(gm, n_dq_wgt, -127, 127, torch.int8):
        per_tensor = True
      elif qw_info := match_dequant_per_channel(gm, n_dq_wgt, 0, -127, 127, torch.int8):
        per_tensor = False
      else:
        continue

      if (qw_info[1] != 0 if per_tensor else torch.any(qw_info[1]).item()):
        continue

      k = qx_info[0] * qw_info[0]
      w = extract_attr(gm, n_dq_wgt.args[0])
      if n_bias is None:
        q_bias = torch.zeros([], dtype=torch.int32)
      else:
        q_bias = torch.round(extract_attr(gm, n_bias) / k).int()
      q_bias = q_bias - qx_info[1] * torch.sum(torch.flatten(w, 1), dim=1, dtype=torch.int32)

      if n_relu is not None and n_relu.target != torch.ops.aten.relu.default:
        continue
      if dilation != [1, 1] or groups != 1:
        continue
      if stride != [1, 1]:
        continue

      och, ich, kh, kw = n_dq_wgt.meta.get("val").shape
      if och % 64 != 0:
        continue
      if kh != kw or kw not in {1, 3}:  # 1x1 or 3x3 kernel
        continue
      if kw == 3 and any((p != 1 for p in padding)):
        continue
      if kw == 1 and any((p != 0 for p in padding)):
        continue

      if bit_utils.get_narrow_type(q_bias).to_signed().bits > 24:
        continue
      if per_tensor:
        adjusted = _mk_act_rqparam([k / qo_info[0]], qo_info[1])
      else:
        adjusted = _mk_act_rqparam(k / qo_info[0], qo_info[1])
      if adjusted is None:
        continue
      if qo_info[1] != -128 and n_relu is not None:
        continue

      adjusted_rsd = _mk_rsd_rqparam([qr_info[0] / qo_info[0]], qo_info[1])
      if adjusted_rsd is None:
        continue

      if n_bias is None:
        biasattr = create_new_param()
      else:
        biasattr = n_bias.target
      setattr(gm, biasattr, nn.Parameter(q_bias, False))

      sclattr = create_new_param()
      setattr(gm, sclattr, nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

      sclrsdattr = create_new_param()
      setattr(gm, sclrsdattr, nn.Parameter(torch.tensor(adjusted_rsd, dtype=torch.int32), False))

      rs = n.meta.get("val").shape
      batch, ich, ih, iw = n_dq_img.meta.get("val").shape

      with graph.inserting_before(n):
        ni = graph.call_function(torch.ops.aten.permute, (n_dq_img.args[0], [0, 2, 3, 1]))
        nk = graph.call_function(torch.ops.aten.permute, (n_dq_wgt.args[0], [0, 2, 3, 1]))
        nr = graph.call_function(torch.ops.aten.permute, (n_dq_rsd.args[0], [0, 2, 3, 1]))
        ns1 = graph.get_attr(sclattr)
        ns2 = graph.get_attr(sclrsdattr)
        nb = graph.get_attr(biasattr)
        n1 = graph.call_function(torch.ops._shir.resnet_weird_residual, (ni, qx_info[1], nk, nb, ns1, qo_info[1], nr, ns2, qr_info[1]))
        n2 = graph.call_function(torch.ops.aten.permute, (n1, [0, 3, 1, 2]))
      n.target = torch.ops.aten.contiguous
      n.args = (n2,)

      if n_relu: graph.erase_node(n_relu)
      graph.erase_node(n_add)
      graph.erase_node(n_dq_rsd)
      graph.erase_node(n_conv)
      if n_bias: graph.erase_node(n_bias)
      graph.erase_node(n_dq_wgt)
      graph.erase_node(n_dq_img)
      continue

    elif m := match_qconv2d(gm, n):
      n_dq_img, n_dq_wgt, n_conv, n_relu = m
      _, _, n_bias, stride, padding, dilation, groups = _get_all_arguments(
          n_conv.args, n_conv.kwargs, n_conv.target._schema.arguments)

      qo_info = match_quant_per_tensor(gm, n, -128, 127, torch.int8)
      qx_info = match_dequant_per_tensor(gm, n_dq_img, -128, 127, torch.int8)
      if qw_info := match_dequant_per_tensor(gm, n_dq_wgt, -127, 127, torch.int8):
        per_tensor = True
      elif qw_info := match_dequant_per_channel(gm, n_dq_wgt, 0, -127, 127, torch.int8):
        per_tensor = False
      else:
        continue

      if (qw_info[1] != 0 if per_tensor else torch.any(qw_info[1]).item()):
        continue

      k = qx_info[0] * qw_info[0]
      w = extract_attr(gm, n_dq_wgt.args[0])
      if n_bias is None:
        q_bias = torch.zeros([], dtype=torch.int32)
      else:
        q_bias = torch.round(extract_attr(gm, n_bias) / k).int()
      q_bias = q_bias - qx_info[1] * torch.sum(torch.flatten(w, 1), dim=1, dtype=torch.int32)

      if n_relu is not None and n_relu.target != torch.ops.aten.relu.default:
        continue
      if dilation != [1, 1] or groups != 1:
        continue

      och, ich, kh, kw = n_dq_wgt.meta.get("val").shape
      if och % 64 != 0:
        continue
      if kh != kw or kw not in {1, 3}:  # 1x1 or 3x3 kernel
        continue
      if kw == 3 and any((p != 1 for p in padding)):
        continue
      if kw == 1 and any((p != 0 for p in padding)):
        continue

      if bit_utils.get_narrow_type(q_bias).to_signed().bits > 24:
        continue
      if per_tensor:
        adjusted = _mk_act_rqparam([k / qo_info[0]], qo_info[1])
      else:
        adjusted = _mk_act_rqparam(k / qo_info[0], qo_info[1])
      if adjusted is None:
        continue
      if qo_info[1] != -128 and n_relu is not None:
        continue

      if n_bias is None:
        biasattr = create_new_param()
      else:
        biasattr = n_bias.target
      setattr(gm, biasattr, nn.Parameter(q_bias, False))

      sclattr = create_new_param()
      setattr(gm, sclattr, nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

      rs = n.meta.get("val").shape
      batch, ich, ih, iw = n_dq_img.meta.get("val").shape

      with graph.inserting_before(n):
        ni = graph.call_function(torch.ops.aten.permute, (n_dq_img.args[0], [0, 2, 3, 1]))
        nk = graph.call_function(torch.ops.aten.permute, (n_dq_wgt.args[0], [0, 2, 3, 1]))
        ns = graph.get_attr(sclattr)
        nb = graph.get_attr(biasattr)
        n1 = graph.call_function(torch.ops._shir.resnet_weird, (ni, qx_info[1], nk, nb, ns, qo_info[1], stride))
        n2 = graph.call_function(torch.ops.aten.permute, (n1, [0, 3, 1, 2]))
      n.target = torch.ops.aten.contiguous
      n.args = (n2,)

      if n_relu: graph.erase_node(n_relu)
      graph.erase_node(n_conv)
      if n_bias: graph.erase_node(n_bias)
      graph.erase_node(n_dq_wgt)
      graph.erase_node(n_dq_img)
      continue

    elif m := match_qlinear(gm, n):
      n_dq_img, n_dq_wgt, n_linear, n_relu = m
      _, _, n_bias = _get_all_arguments(n_linear.args, n_linear.kwargs, n_linear.target._schema.arguments)

      qo_info = match_quant_per_tensor(gm, n, -128, 127, torch.int8)
      qx_info = match_dequant_per_tensor(gm, n_dq_img, -128, 127, torch.int8)
      qw_info = match_dequant_per_tensor(gm, n_dq_wgt, -127, 127, torch.int8)

      if qw_info[1] != 0:
        continue

      k = qx_info[0] * qw_info[0]
      w = extract_attr(gm, n_dq_wgt.args[0])
      if n_bias is None:
        q_bias = torch.zeros([], dtype=torch.int32)
      else:
        q_bias = torch.round(extract_attr(gm, n_bias) / k).int()
      q_bias = q_bias - qx_info[1] * torch.sum(w, dim=1, dtype=torch.int32)

      if n_relu is not None and n_relu.target != torch.ops.aten.relu.default:
        continue

      if bit_utils.get_narrow_type(q_bias).to_signed().bits > 24:
        continue

      adjusted = _mk_act_rqparam([k / qo_info[0]], qo_info[1])
      if adjusted is None:
        continue
      if qo_info[1] != -128 and n_relu is not None:
        continue

      if n_bias is None:
        biasattr = create_new_param()
      else:
        biasattr = n_bias.target
      setattr(gm, biasattr, nn.Parameter(q_bias, False))

      sclattr = create_new_param()
      setattr(gm, sclattr, nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

      j, k = n_dq_wgt.meta.get("val").shape
      i, _ = n.meta.get("val").shape

      i_tiles = (k + (64 - 1)) // 64
      o_tiles = (j + (64 - 1)) // 64
      i_pad = i_tiles * 64 - k
      o_pad = o_tiles * 64 - j

      with graph.inserting_before(n):
        # TODO: pack batch dimension inwards
        ni = graph.call_function(torch.ops.aten.pad, (n_dq_img.args[0], [0, i_pad]))
        ni = graph.call_function(torch.ops.aten.view, (ni, [i, 1, 1, i_tiles * 64]))
        nk = graph.call_function(torch.ops.aten.pad, (n_dq_wgt.args[0], [0, i_pad, 0, o_pad]))
        nk = graph.call_function(torch.ops.aten.view, (nk, [o_tiles * 64, 1, 1, i_tiles * 64]))
        ns = graph.get_attr(sclattr)
        nb = graph.get_attr(biasattr)
        n1 = graph.call_function(torch.ops._shir.resnet_weird, (ni, qx_info[1], nk, nb, ns, qo_info[1], [1, 1]))
        n2 = graph.call_function(torch.ops.aten.pad, (n1, [0, -o_pad]))
      n.target = torch.ops.aten.view
      n.args = (n2, [i, j])

      if n_relu: graph.erase_node(n_relu)
      graph.erase_node(n_linear)
      if n_bias: graph.erase_node(n_bias)
      graph.erase_node(n_dq_wgt)
      graph.erase_node(n_dq_img)
      continue

  graph.lint()
  gm.recompile()

