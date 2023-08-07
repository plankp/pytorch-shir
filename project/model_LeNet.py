# Almost LeNet:
# It looks like the original definition (according to Wikipedia at least) uses
# sigmoid activation. The network literally does not learn for some reason,
# hence we use relu here!

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
import torch.nn.functional as F

import torch._dynamo as torchdynamo
from torch.ao.quantization.quantize_pt2e import (
  convert_pt2e,
  prepare_pt2e,
)

import shir

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
    x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
    x = torch.ops.shir_intrinsic.flatten(x, 1, -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

SAVED_MODEL_PATH = "./data/model_LeNet.pth"

# the accuracy is around 98.9%

model = reload_cached(SAVED_MODEL_PATH, Net, learning_rate=0.1)
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

test_loop(test_dataloader, model, loss_fn)
