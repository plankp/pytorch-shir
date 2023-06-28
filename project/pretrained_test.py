# adopted from https://pytorch.org/vision/main/models.html

"""
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights

weights = ResNet50_QuantizedWeights.DEFAULT
model = resnet50(weights=weights, quantize=True)
model.eval()

print(model)
"""

import torch
from torchvision.models import vgg11
import shir_backend
from torch.ao.quantization import (
  QConfig,
  QConfigMapping,
)
from torch.ao.quantization.quantize_fx import (
  prepare_fx,
  _convert_to_reference_decomposed_fx,  # XXX: private API
)

model = vgg11(weights='DEFAULT')
model.eval()

example_inputs = torch.randn(10, 3, 32, 32)
qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_qconfig)
model = prepare_fx(model, qconfig_mapping, example_inputs)
model = _convert_to_reference_decomposed_fx(model)
print(model)

torch._dynamo.reset()
model = torch.compile(backend=shir_backend.compiler)(model)
model(example_inputs)
