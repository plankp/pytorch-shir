# This is a sub-part of the full LeNet model, namely the first two layers

import torch
import torch.nn as nn
import torch.nn.functional as F

from routine_mnist_digits import (
  reload_cached,
  test_loop,
  time_inference,
  test_dataloader,
  loss_fn,
  get_example_input,
)
import copy
import torch._dynamo as torchdynamo
import torch.export
import shir
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.utils.data import DataLoader, TensorDataset
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

# torchdynamo.config.automatic_dynamic_shapes = False

# shir | x86 | fp32
PROFILE = "shir"
PROBLEM_SIZE_N = 64
PROBLEM_TRIPS  = 1    # TRIPS and INSTS is just to avoid allocating *big* tensors
PROBLEM_INSTS  = 1000
KEEP_DEQUANT = True

test_dataloader = DataLoader(TensorDataset(
  torch.zeros(PROBLEM_SIZE_N * PROBLEM_INSTS, 1, 28, 28),
  torch.zeros(PROBLEM_SIZE_N * PROBLEM_INSTS, 6, 14, 14)
), batch_size=PROBLEM_SIZE_N)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5, padding=2)

  def forward(self, x):
    x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
    return x

model = Net()

# copy the parameters from the full model
curr_st = model.state_dict()
full_st = torch.load("./data/model_LeNet.pth")
for k in curr_st:
  curr_st[k].copy_(full_st[k])
curr_st = full_st = None

# mark it as trained already (since we filled the weights)
model.eval()

example_inputs = (get_example_input(),)

torchdynamo.reset()

if PROFILE == "shir":
  quantizer = shir.BackendQuantizer()
elif PROFILE == "x86":
  quantizer = xiq.X86InductorQuantizer()
  quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

with torch.no_grad():
  model = torch.export.export(model, example_inputs).module()

  if PROFILE in ["shir", "x86"]:
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
  else:
    model = torch.compile(model)

"""
for i in range(PROBLEM_TRIPS):
  time_inference(test_dataloader, model)
"""

"""
shir.config.FPGA_PRINT_RTINFO = False
with open(f"./metrics/lenet_2/{PROFILE}_{PROBLEM_SIZE_N}.log", "w") as f:
  for i in range(PROBLEM_TRIPS):
    for w in time_inference(test_dataloader, model):
      print(w, file=f)
"""


print(model(example_inputs[0]))

