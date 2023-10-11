"""
Things needed to "teach" PyTorch how to quantize your model.

For more information, check the following:
*   https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html
*   existing PyTorch quantizers located around torch.ao.quantization (both legacy and pt2e)
"""

import operator
import itertools
from typing import Dict, List, Optional, Callable

import torch
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.ao.quantization.pt2e.utils import (
  _get_tensor_constant_from_node,
  _get_all_arguments,
)
from torch.ao.quantization.quantizer.utils import (
  _annotate_input_qspec_map,
  _annotate_output_qspec,
)
from torch.fx.passes.utils.source_matcher_utils import (
  get_source_partitions,
  SourcePartition
)
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.quantizer.quantizer import (
  QuantizationSpec,
  Quantizer,
  QuantizationAnnotation,
  SharedQuantizationSpec,
  FixedQParamsQuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
  OperatorConfig,
  QuantizationConfig,
  get_input_act_qspec,
  get_output_act_qspec,
  get_bias_qspec,
  get_weight_qspec,
  _is_annotated,
)
from torch.ao.quantization.observer import (
  HistogramObserver,
  MinMaxObserver,
  PerChannelMinMaxObserver,
  PlaceholderObserver,
)

"""
Quantization specs that are used by SHIR
"""

# the quantization spec for activations must be signed and the min and max
# must use up all possible values (as in [-2**(b-1), 2**(b-1)-1]) because this
# is how rounding and clipping works in SHIR.
#
# use per tensor here because per channel makes things complicated.
_act_qspec = QuantizationSpec(
  dtype=torch.int8,
  quant_min=-128,
  quant_max=127,
  qscheme=torch.per_tensor_affine,
  is_dynamic=False,
  observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
)

# occasionally, it makes sense for some operators (like hardsigmoid) to have a
# predefined quantization parameter that spans across the whole int8 range.
_fixed_i8_qspec = FixedQParamsQuantizationSpec(
  dtype=torch.int8,
  quant_min=-128,
  quant_max=127,
  qscheme=torch.per_tensor_affine,
  scale=1.0 / 256.0,
  zero_point=-128,
)

# for weights, we want to use symmetric scheme to avoid doing too much weight
# adjustment (due to the zero point) at runtime. besides, PyTorch only allows
# symmetric for weights anyways.
_weight_qspec_per_tensor = QuantizationSpec(
  dtype=torch.int8,
  quant_min=-127,
  quant_max=127,
  qscheme=torch.per_tensor_symmetric,
  is_dynamic=False,
  observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
)

# sometimes it makes sense to use per channel quantization for weights
# (convolution weights can certainly make use of this)
_weight_qspec_per_channel = QuantizationSpec(
  dtype=torch.int8,
  quant_min=-127,
  quant_max=127,
  qscheme=torch.per_channel_symmetric,
  ch_axis=0,
  is_dynamic=False,
  observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2**-12),
)

# bias is not quantized because we can derive the qparams from the weights and
# input but also because PyTorch does not allow it.
_bias_qspec = QuantizationSpec(
  dtype=torch.float,
  observer_or_fake_quant_ctr=PlaceholderObserver,
)

"""
Quantization configs
(since it's easier to pass these around than individual qspecs)
"""

_qconfig_per_tensor = QuantizationConfig(
  _act_qspec,
  _act_qspec,
  _weight_qspec_per_tensor,
  _bias_qspec,
)

_qconfig_per_channel = QuantizationConfig(
  _act_qspec,
  _act_qspec,
  _weight_qspec_per_channel,
  _bias_qspec,
)

_qconfig_fixed_output = QuantizationConfig(
  _act_qspec,
  _fixed_i8_qspec,
  _weight_qspec_per_tensor,
  _bias_qspec,
)

"""
Utility functions
(they come from the in-tree XNNPACK)
"""

def _mark_nodes_as_annotated(nodes: List[torch.fx.Node]):
  for node in nodes:
    if node is not None:
      if "quantization_annotation" not in node.meta:
        node.meta["quantization_annotation"] = QuantizationAnnotation()
      node.meta["quantization_annotation"]._annotated = True

def _extract_linear_fields(gm: torch.fx.GraphModule, p: SourcePartition):
  input_node = p.input_nodes[0]
  output_node = p.output_nodes[0]
  weight_node = None
  bias_node = None
  for param in p.params:
    weight_or_bias = getattr(gm, param.target)
    if weight_or_bias.ndim == 2:
      weight_node = param
    if weight_or_bias.ndim == 1:
      bias_node = param
  assert weight_node is not None, "Expected linear layer to have weight"

  input_use_node = None
  for node in p.nodes:
    if node in input_node.users:
      input_use_node = node
      break
  assert input_use_node is not None, "Expect linear layer to use the input"

  return (input_use_node, input_node, weight_node, bias_node, output_node)

"""
The actual magic behind deciding where each qspec goes
"""

# list of single-input nodes where the input and output should share
# quantization parameters
#
# XXX:
# some are already covered by PyTorch's late annotation propagate phase.
# also, since these require inputs to be annotated, we might want to
# match individual fx nodes (as in the rewrite phase) instead.
_OPS_IN_OUT_SHARING = [
  torch.nn.ReLU6,
  torch.nn.Hardtanh,
  torch.nn.MaxPool2d,
  torch.nn.AdaptiveAvgPool2d,
  torch.nn.functional.avg_pool2d,
  torch.ops.shir_intrinsic.flatten,
]

class BackendQuantizer(Quantizer):

  def __init__(self, allow_per_channel=True):
    super().__init__()
    self.allow_per_channel = allow_per_channel

  # where the magic happens
  def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    self._fuse_bn_weights(gm)

    # allow per channel for convolution
    qconfig = _qconfig_per_channel if self.allow_per_channel else _qconfig_per_tensor
    self._annotate_conv_relu(gm, qconfig)
    self._annotate_conv(gm, qconfig)

    # otherwise it's just per tensor
    qconfig = _qconfig_per_tensor
    self._annotate_linear_relu(gm, qconfig)
    self._annotate_linear(gm, qconfig)
    self._annotate_bin_op(gm, qconfig)
    self._annotate_activation([torch.nn.Hardswish], gm, qconfig)

    # hardsigmoid (and friends?) have fixed output range
    qconfig = _qconfig_fixed_output
    self._annotate_activation([torch.nn.Hardsigmoid], gm, qconfig)

    # ops that may depend (and even override) earlier annotations
    self._annotate_in_out_shared_qspec(_OPS_IN_OUT_SHARING, gm)
    return gm

  # validate the annotated graph is supported by the backend
  def validate(self, gm: torch.fx.GraphModule) -> None:
    pass

  # according to pytorch/pytorch PR#99063, it's supposed to return a list of
  # patterns that this quantizer quantizes. (which is nothing like what is
  # written in the base Quantizer class)
  #
  # it looks like this is more like documentation than actually being useful
  @classmethod
  def get_supported_operators(cls) -> List[OperatorConfig]:
    # XXX: this is out of date, but no one seems to use it
    # TODO: update it :sweatsmile:
    return [
      OperatorConfig(_qconfig_per_tensor, [[torch.nn.Linear, torch.nn.ReLU]]),
      OperatorConfig(_qconfig_per_tensor, [[torch.nn.Linear]]),

      # for simplicity, we only claim to support Conv2d.
      # we actually support every non-transpoing convolution,
      # it also isn't too difficult to extend the current stuff
      OperatorConfig(_qconfig_per_tensor, [[torch.nn.Conv2d, torch.nn.ReLU]]),
      OperatorConfig(_qconfig_per_tensor, [[torch.nn.Conv2d]]),
    ]

  def _fuse_bn_weights(self, gm: torch.fx.GraphModule):
    # conv-bn fusion case is already handled by the prepare routine that
    # drives this annotation process.

    # however, we need to handle linear-bn fusion ourselves (likely because
    # linear layers are a more painful to deal with)
    #
    # turn:
    # wt = aten.t.default(w)
    # m  = addmm.default(b, x, wt)    || mm.default(x, wt)
    # _  = _native_bn(m)
    #
    # into:
    # wt = aten.t.default(w')
    # m  = addmm.default(b', x, wt)   <-- batch norm introduces bias
    # _  = m
    for n in gm.graph.nodes:
      if n.op != "call_function" or n.target != torch.ops.aten._native_batch_norm_legit_no_training.default:
        continue
      bn_node = n
      n = bn_node.args[0]

      if n.op != "call_function":
        continue
      if n.target == torch.ops.aten.addmm.default:
        mm_node = n
        mm_bias_node = mm_node.args[0]
        mm_input_node = mm_node.args[1]
        mm_wt_node = mm_node.args[2]
      elif n.target == torch.ops.aten.mm.default:
        mm_node = n
        mm_bias_node = None
        mm_input_node = mm_node.args[0]
        mm_wt_node = mm_node.args[1]
      else:
        continue
      n = mm_wt_node

      if n.op != "call_function" or n.target != torch.ops.aten.t.default:
        continue
      mm_weight_node = n.args[0]

      linear_w = _get_tensor_constant_from_node(mm_weight_node, gm)
      linear_b = _get_tensor_constant_from_node(mm_bias_node, gm)

      bn_args_schema = bn_node.target._schema.arguments
      bn_args = _get_all_arguments(bn_node.args, bn_node.kwargs, bn_args_schema)
      bn_w = _get_tensor_constant_from_node(bn_args[1], gm)
      bn_b = _get_tensor_constant_from_node(bn_args[2], gm)
      bn_rm = _get_tensor_constant_from_node(bn_args[3], gm)
      bn_rv = _get_tensor_constant_from_node(bn_args[4], gm)
      bn_eps = bn_args[6]

      fused_weight, fused_bias = fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b)
      weight_attr_name = mm_weight_node.target
      setattr(gm, weight_attr_name, fused_weight)
      if mm_bias_node is not None:
        bias_attr_name = mm_bias_node.target
      else:
        bias_attr_name = weight_attr_name + "_bias"
        with m.graph.inserting_before(mm_node):
          mm_bias_node = m.graph.get_attr(bias_attr_name)
      setattr(gm, bias_attr_name, fused_bias)
      mm_node.target = torch.ops.aten.addmm.default
      mm_node.args = (mm_bias_node, mm_input_node, mm_wt_node)

      for user in bn_node.users:
        if user.op != "call_function" or user.target != operator.getitem or user.args[1] != 0:
          continue
        user.replace_all_uses_with(mm_node)

    gm.graph.eliminate_dead_code()
    gm.recompile()

  def _annotate_linear_relu(self, gm: torch.fx.GraphModule, qconfig: QuantizationConfig):
    input_qspec = get_input_act_qspec(qconfig)
    output_qspec = get_output_act_qspec(qconfig)
    weight_qspec = get_weight_qspec(qconfig)
    bias_qspec = get_bias_qspec(qconfig)

    fused_partitions = find_sequential_partitions(gm, [torch.nn.Linear, torch.nn.ReLU])
    for linear_p, relu_p in fused_partitions:
      (inp_use, inp, weight, bias, out) = _extract_linear_fields(gm, linear_p)
      relu = relu_p.output_nodes[0]

      # if any of the nodes are annotated, then leave it alone
      if _is_annotated([relu, inp_use, bias, weight, out]):
        continue

      _annotate_input_qspec_map(inp_use, inp, input_qspec)
      _annotate_output_qspec(weight, weight_qspec)
      if bias:
        _annotate_output_qspec(bias, bias_qspec)

      _annotate_output_qspec(relu, output_qspec)
      _mark_nodes_as_annotated([*relu_p.nodes, *linear_p.nodes])

  def _annotate_linear(self, gm: torch.fx.GraphModule, qconfig: QuantizationConfig):
    input_qspec = get_input_act_qspec(qconfig)
    output_qspec = get_output_act_qspec(qconfig)
    weight_qspec = get_weight_qspec(qconfig)
    bias_qspec = get_bias_qspec(qconfig)

    all_partitions = get_source_partitions(gm.graph, [torch.nn.Linear])
    partitions = list(itertools.chain(*all_partitions.values()))
    for p in partitions:
      (inp_use, inp, weight, bias, out) = _extract_linear_fields(gm, p)

      if _is_annotated([inp_use, bias, weight, out]):
        continue

      _annotate_input_qspec_map(inp_use, inp, input_qspec)
      _annotate_output_qspec(weight, weight_qspec)
      if bias:
        _annotate_output_qspec(bias, bias_qspec)

      _annotate_output_qspec(out, output_qspec)
      _mark_nodes_as_annotated([*p.nodes])

  def _annotate_conv_relu(self, gm: torch.fx.GraphModule, qconfig: QuantizationConfig):
    input_qspec = get_input_act_qspec(qconfig)
    output_qspec = get_output_act_qspec(qconfig)
    weight_qspec = get_weight_qspec(qconfig)
    bias_qspec = get_bias_qspec(qconfig)

    fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, torch.nn.ReLU])
    for conv_p, relu_p in fused_partitions:
      conv_node = conv_p.output_nodes[0]
      relu = relu_p.output_nodes[0]

      assert (
        conv_node.op == "call_function"
        and conv_node.target == torch.ops.aten.convolution.default
      ), "Expected conv layer to call aten.convolution"

      if _is_annotated([conv_node, relu]):
        continue

      inp = conv_node.args[0]
      weight = conv_node.args[1]
      bias = conv_node.args[2]
      _annotate_input_qspec_map(conv_node, inp, input_qspec)
      _annotate_input_qspec_map(conv_node, weight, weight_qspec)
      if bias:
        _annotate_input_qspec_map(conv_node, bias, bias_qspec)

      _annotate_output_qspec(relu, output_qspec)
      _mark_nodes_as_annotated([*conv_p.nodes, *relu_p.nodes])

  def _annotate_conv(self, gm: torch.fx.GraphModule, qconfig: QuantizationConfig):
    input_qspec = get_input_act_qspec(qconfig)
    output_qspec = get_output_act_qspec(qconfig)
    weight_qspec = get_weight_qspec(qconfig)
    bias_qspec = get_bias_qspec(qconfig)

    all_partitions = get_source_partitions(gm.graph, [torch.nn.Conv2d])
    partitions = list(itertools.chain(*all_partitions.values()))
    for p in partitions:
      conv_node = p.output_nodes[0]
      assert (
        conv_node.op == "call_function"
        and conv_node.target == torch.ops.aten.convolution.default
      ), "Expected conv layer to call aten.convolution"

      if _is_annotated([conv_node]):
        continue

      inp = conv_node.args[0]
      weight = conv_node.args[1]
      bias = conv_node.args[2]
      _annotate_input_qspec_map(conv_node, inp, input_qspec)
      _annotate_input_qspec_map(conv_node, weight, weight_qspec)
      if bias:
        _annotate_input_qspec_map(conv_node, bias, bias_qspec)

      _annotate_output_qspec(conv_node, output_qspec)
      _mark_nodes_as_annotated([*p.nodes])

  def _annotate_bin_op(self, gm: torch.fx.GraphModule, qconfig: QuantizationConfig):
    input_qspec = get_input_act_qspec(qconfig)
    output_qspec = get_output_act_qspec(qconfig)

    all_partitions = get_source_partitions(gm.graph, [
      torch.add,
      operator.add,
      operator.iadd,  # some models accumulate residuals with +=

      torch.mul,
      operator.mul,
    ])
    partitions = list(itertools.chain(*all_partitions.values()))
    for p in partitions:
      node = p.output_nodes[0]
      if _is_annotated([node]):
        continue

      lhs = node.args[0]
      rhs = node.args[1]

      _annotate_input_qspec_map(node, lhs, input_qspec)
      _annotate_input_qspec_map(node, rhs, input_qspec)
      _annotate_output_qspec(node, output_qspec)
      _mark_nodes_as_annotated([*p.nodes])

  def _annotate_activation(self, ops: List[Callable], gm: torch.fx.GraphModule, qconfig: QuantizationConfig):
    input_qspec = get_input_act_qspec(qconfig)
    output_qspec = get_output_act_qspec(qconfig)

    all_partitions = get_source_partitions(gm.graph, ops)
    partitions = list(itertools.chain(*all_partitions.values()))
    for p in partitions:
      out = p.output_nodes[0]
      inp = p.input_nodes[0]
      if _is_annotated([out]):
        continue

      _annotate_input_qspec_map(out, inp, input_qspec)
      _annotate_output_qspec(out, output_qspec)
      _mark_nodes_as_annotated([*p.nodes])

  def _annotate_in_out_shared_qspec(self, ops: List[Callable], gm: torch.fx.GraphModule):
    all_partitions = get_source_partitions(gm.graph, ops)
    partitions = list(itertools.chain(*all_partitions.values()))
    for p in partitions:
      out = p.output_nodes[0]
      inp = p.input_nodes[0]
      if _is_annotated([out]):
        continue

      # only proceed if the input has an annotation
      if not _is_annotated([inp]):
        continue
      if inp.meta["quantization_annotation"].output_qspec is None:
        continue

      # following the discussion on sharing qparams, we not only share with
      # the output, we also share with the current input!
      shared_qspec = SharedQuantizationSpec(inp)
      _annotate_input_qspec_map(out, inp, shared_qspec)
      _annotate_output_qspec(out, shared_qspec)
      _mark_nodes_as_annotated([*p.nodes])
