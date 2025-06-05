# Almost LeNet:
# It looks like the original definition (according to Wikipedia at least) uses
# sigmoid activation. The network literally does not learn for some reason,
# hence we use relu here!

from routine_mnist_digits import (
  reload_cached,
  test_loop,
  test_dataloader,
  time_inference,
  loss_fn,
  get_example_input,
)
import copy
import torch
from torch import nn
import torch.nn.functional as F

import torch._dynamo as torchdynamo
import torch.export
from torch.ao.quantization.quantize_pt2e import (
  convert_pt2e,
  prepare_pt2e,
)
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.utils.data import DataLoader, TensorDataset
import shir

PROFILE = "shir"
PROBLEM_SIZE_N = 65536
PROBLEM_TRIPS  = 1000 #1
PROBLEM_INSTS  = 1    #1000
KEEP_DEQUANT = False

dummy_dataloader = DataLoader(TensorDataset(
  torch.zeros(PROBLEM_SIZE_N * PROBLEM_INSTS, 1, 28, 28),
  torch.zeros(PROBLEM_SIZE_N * PROBLEM_INSTS, 10),
), batch_size=PROBLEM_SIZE_N)

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
    x = torch.flatten(x, 1, -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

SAVED_MODEL_PATH = "./data/model_LeNet.pth"

# the accuracy is around 98.9%

model = reload_cached(SAVED_MODEL_PATH, Net, learning_rate=0.1)
#print("SOFT", test_loop(test_dataloader, model, loss_fn))

#print(model)

example_inputs = (get_example_input(),)

torchdynamo.reset()

if PROFILE in {"shir", "shir2"}:
  quantizer = shir.BackendQuantizer()
elif PROFILE == "x86":
  quantizer = xiq.X86InductorQuantizer()
  quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

with torch.no_grad():
  model = torch.export.export(model, example_inputs).module()

  if PROFILE in {"shir", "shir2", "x86"}:
    model = prepare_pt2e(model, quantizer)
    model(*example_inputs)
    model = convert_pt2e(model)

  # right now, the model roughly looks like:
  # input -> quantize -> quantized model -> dequantize -> output
  #
  # here we forcefully remove the final dequantize layer.
  if not KEEP_DEQUANT:
    maybe_output_node = next(iter(reversed(model.graph.nodes)))
    assert maybe_output_node.op == "output", "Expected last node to be an output node"
    assert len(maybe_output_node.args) == 1 and len(maybe_output_node.args[0]) == 1, "Expected single output node"

    maybe_dequant_node = maybe_output_node.args[0][0]
    assert maybe_dequant_node.op == "call_function" and maybe_dequant_node.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default, "Expected output to be a dequant node"

    maybe_output_node.args = ([maybe_dequant_node.args[0]],)
    model.graph.erase_node(maybe_dequant_node)

    model.graph.lint()
    model.recompile()

  # print(model)

  if PROFILE == "shir":
    model = torch.compile(model, backend=shir.compiler)
  elif PROFILE == "shir2":
    import shir.backend2
    model = torch.compile(model, backend=shir.backend2.compiler)
  else:
    model = torch.compile(model)

"""
shir.config.FPGA_PRINT_RTINFO = False
for i in range(PROBLEM_TRIPS):
  time_inference(dummy_dataloader, model)
"""

"""
shir.config.FPGA_PRINT_RTINFO = False
with open(f"./metrics/LeNet5/{PROFILE}_{PROBLEM_SIZE_N}.log", "w") as f:
  for i in range(PROBLEM_TRIPS):
    for w in time_inference(dummy_dataloader, model):
      print(w, file=f)
"""

# print(model(example_inputs[0][0, None]))
# print(model(example_inputs[0][0:16]))

print("FINAL", test_loop(test_dataloader, model, loss_fn))
