# adopted from https://pytorch.org/vision/main/models.html

"""
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights

weights = ResNet50_QuantizedWeights.DEFAULT
model = resnet50(weights=weights, quantize=True)
model.eval()

print(model)
"""

import copy
import torch
from torchvision.models import vgg11

import torch._dynamo as torchdynamo
from torch.ao.quantization._quantize_pt2e import (
  convert_pt2e,
  prepare_pt2e_quantizer,
)

import shir_backend
import shir_quantizer

model = vgg11(weights='DEFAULT')
model.eval()
example_inputs = (torch.randn(10, 3, 32, 32),)

# program capture
model, guards = torchdynamo.export(
  model,
  *copy.deepcopy(example_inputs),
  aten_graph=True,
)
quantizer = shir_quantizer.BackendQuantizer()

model = prepare_pt2e_quantizer(model, quantizer)
model = convert_pt2e(model)

torchdynamo.reset()
model = torch.compile(backend=shir_backend.compiler)(model)
model(*example_inputs)
