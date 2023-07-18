# based on the PyTorch tutorial:
# https://pytorch.org/tutorials/beginner/basics/intro.html

from routine_mnist_digits import (
  reload_cached,
  test_loop,
  test_dataloader,
  loss_fn,
  get_example_input,
)
import copy
import torch
from torch import nn

import torch._dynamo as torchdynamo
from torch.ao.quantization._quantize_pt2e import (
  convert_pt2e,
  prepare_pt2e_quantizer,
)

import shir_backend
import shir_quantizer

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.a = nn.Linear(28*28, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.a(x)
    return x

# The accuracy is 86.9% (which is better than the dense net!?)

SAVED_MODEL_PATH = "./data/model_simple_slp.pth"

model = reload_cached(SAVED_MODEL_PATH, Net)
test_loop(test_dataloader, model, loss_fn)

print(model)

example_inputs = (get_example_input(),)

model, guards = torchdynamo.export(
  model,
  *copy.deepcopy(example_inputs),
  aten_graph=True,
)

quantizer = shir_quantizer.BackendQuantizer()

model = prepare_pt2e_quantizer(model, quantizer)
model(*example_inputs)  # calibration
model = convert_pt2e(model)

torchdynamo.reset()
model = torch.compile(backend=shir_backend.compiler)(model)
model(*example_inputs)

test_loop(test_dataloader, model, loss_fn)
