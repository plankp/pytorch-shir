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

SAVED_MODEL_PATH = "./data/model_simple_conv.pth"

# model from:
# https://machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)
    self.act1 = nn.ReLU()
    self.drop1 = nn.Dropout(0.3)

    self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
    self.act2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

    self.flat = nn.Flatten()

    self.fc3 = nn.Linear(6272, 512)
    self.act3 = nn.ReLU()
    self.drop3 = nn.Dropout(0.5)

    self.fc4 = nn.Linear(512, 10)

  def forward(self, x):
    x = self.act1(self.conv1(x))
    x = self.drop1(x)

    x = self.act2(self.conv2(x))
    x = self.pool2(x)

    x = self.flat(x)

    x = self.act3(self.fc3(x))
    x = self.drop3(x)

    x = self.fc4(x)
    return x

# the accuracy is around 94.2%

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
operator_config = shir_quantizer.get_symmetric_quantization_config()
quantizer.set_global(operator_config)

model = prepare_pt2e_quantizer(model, quantizer)
model(*example_inputs)  # calibration
model = convert_pt2e(model)

torchdynamo.reset()
model = torch.compile(backend=shir_backend.compiler)(model)
model(*example_inputs)

test_loop(test_dataloader, model, loss_fn)
