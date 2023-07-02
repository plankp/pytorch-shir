# based on the following:
# *   https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html
# *   Gist linked near the end of the wikipage^
# *   existing quantizer implementations

import itertools
from typing import Dict, List, Optional, Callable

import torch

from torch.ao.quantization._pt2e.quantizer.utils import (
  _annotate_input_qspec_map,
  _annotate_output_qspec,
  get_input_act_qspec,
  get_output_act_qspec,
  get_bias_qspec,
  get_weight_qspec,
)

from torch.fx.passes.utils.source_matcher_utils import (
  get_source_partitions,
  SourcePartition
)
from torch.ao.quantization._pt2e.graph_utils import find_sequential_partitions

from torch.ao.quantization._pt2e.quantizer.quantizer import (
  OperatorConfig,
  QuantizationConfig,
  QuantizationSpec,
  Quantizer,
  QuantizationAnnotation,
  SharedQuantizationSpec,
)
from torch.ao.quantization.observer import (
  HistogramObserver,
  MinMaxObserver,
  PlaceholderObserver,
)

def get_symmetric_quantization_config():
  act_quantization_spec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
  )

  # this one has to have a symmetric qscheme
  weight_quantization_spec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-127,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
  )

  # assume we don't quantize the bias (we haven't been doing so anyway)
  bias_quantization_spec = QuantizationSpec(
    dtype=torch.float,
    observer_or_fake_quant_ctr=PlaceholderObserver,
  )
  quantization_config = QuantizationConfig(
    act_quantization_spec,
    act_quantization_spec,
    weight_quantization_spec,
    bias_quantization_spec,
    False,
  )
  return quantization_config

def _mark_nodes_as_annotated(nodes: List[torch.fx.Node]):
  for node in nodes:
    if node is not None:
      if "quantization_annotation" not in node.meta:
        node.meta["quantization_annotation"] = QuantizationAnnotation()
      node.meta["quantization_annotation"]._annotated = True

def _is_annotated(nodes: List[torch.fx.Node]):
  return any(((
    "quantization_annotation" in node.meta
    and node.meta["quantization_annotation"]._annotated
  ) for node in nodes if node is not None))

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

class BackendQuantizer(Quantizer):

  def __init__(self):
    super().__init__()
    self.global_config: QuantizationConfig = None
    self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

  def set_global(self, quantization_config: QuantizationConfig):
    self.global_config = quantization_config
    return self

  # where the magic happens
  def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # say we just annotate linear layers
    qconfig = self.global_config
    self._annotate_linear_relu(gm, qconfig)
    self._annotate_linear(gm, qconfig)

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
    qconfig = get_symmetric_quantization_config()
    return [
      OperatorConfig(qconfig, [[torch.nn.Linear, torch.nn.ReLU]]),
      OperatorConfig(qconfig, [[torch.nn.Linear]]),
    ]

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
