# adopted from https://pytorch.org/vision/main/models.html

import copy
import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization.quantize_pt2e import (
  convert_pt2e,
  prepare_pt2e,
)

import shir_backend
import shir_quantizer
import shir_intrinsic

## uncomment out one of the following set of lines
# from torchvision.models import vgg11
# model = vgg11(weights='DEFAULT')
# from torchvision.models import resnet50
# model = resnet50(weights='DEFAULT')
from torchvision.models import mobilenet_v2
model = mobilenet_v2(weights='DEFAULT')
# from torchvision.models import mobilenet_v3_large
# model = mobilenet_v3_large(weights='DEFAULT')

model.eval()

example_inputs = (torch.randn(10, 3, 224, 224),)

# program capture
model, guards = torchdynamo.export(
  model,
  *copy.deepcopy(example_inputs),
  aten_graph=True,
)
quantizer = shir_quantizer.BackendQuantizer()

model = prepare_pt2e(model, quantizer)
model(*example_inputs)
model = convert_pt2e(model)

torchdynamo.reset()
model = torch.compile(backend=shir_backend.compiler)(model)
model(*example_inputs)
