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
from torch.ao.quantization.quantize_pt2e import (
  convert_pt2e,
  prepare_pt2e,
)

import shir

SAVED_MODEL_PATH = "./data/model_simple_conv.pth"

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 1, 5)
    self.act1 = nn.ReLU()

    self.fc = nn.Linear(576, 10)

  def forward(self, x):
    x = self.act1(self.conv1(x))
    x = torch.ops.shir_intrinsic.flatten(x, 1, -1)
    x = self.fc(x)
    return x

# the accuracy is around 92.2%

model = reload_cached(SAVED_MODEL_PATH, Net, learning_rate=0.01)
test_loop(test_dataloader, model, loss_fn)

print(model)

example_inputs = (get_example_input(),)

model, guards = torchdynamo.export(
  model,
  *copy.deepcopy(example_inputs),
  aten_graph=True,
)

quantizer = shir.BackendQuantizer()

model = prepare_pt2e(model, quantizer)
model(*example_inputs)  # calibration
model = convert_pt2e(model)

torchdynamo.reset()
model = torch.compile(backend=shir.compiler)(model)
model(*example_inputs)

# test_loop(test_dataloader, model, loss_fn)
