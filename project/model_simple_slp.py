# about the simplest model you can ever make...

import pandas as pd
from timeit import timeit
import time
from routine_mnist_digits import (
  reload_cached,
  test_loop,
  time_inference,
  test_data,
  loss_fn,
  get_example_input,
)
import copy
import torch
from torch import nn
from torch.utils.data import (
  DataLoader,
  TensorDataset,
)

import torch._dynamo as torchdynamo
import torch.export
from torch.ao.quantization.quantize_pt2e import (
  convert_pt2e,
  prepare_pt2e,
)
import shir

# always use static shapes
torchdynamo.config.automatic_dynamic_shapes = False

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.a = nn.Linear(28*28, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.a(x)
    return x

# custom test data loader (to drop extra entries)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True, drop_last=True)

# dummy data... each run contains 1000 batches
test_dataloader_16 = DataLoader(TensorDataset(
  torch.randn(16 * 1000, 1, 28, 28), torch.randn(16 * 1000, 1)
), batch_size=16)
test_dataloader_32 = DataLoader(TensorDataset(
  torch.randn(32 * 1000, 1, 28, 28), torch.randn(32 * 1000, 1)
), batch_size=32)
test_dataloader_64 = DataLoader(TensorDataset(
  torch.randn(64 * 1000, 1, 28, 28), torch.randn(64 * 1000, 1)
), batch_size=64)
test_dataloader_128 = DataLoader(TensorDataset(
  torch.randn(128 * 1000, 1, 28, 28), torch.randn(128 * 1000, 1)
), batch_size=128)
test_dataloader_256 = DataLoader(TensorDataset(
  torch.randn(256 * 1000, 1, 28, 28), torch.randn(256 * 1000, 1)
), batch_size=256)
test_dataloader_512 = DataLoader(TensorDataset(
  torch.randn(512 * 1000, 1, 28, 28), torch.randn(512 * 1000, 1)
), batch_size=512)
test_dataloader_1024 = DataLoader(TensorDataset(
  torch.randn(1024 * 1000, 1, 28, 28), torch.randn(1024 * 1000, 1)
), batch_size=1024)
test_dataloader_2048 = DataLoader(TensorDataset(
  torch.randn(2048 * 1000, 1, 28, 28), torch.randn(2048 * 1000, 1)
), batch_size=2048)
test_dataloader_4096 = DataLoader(TensorDataset(
  torch.randn(4096 * 1000, 1, 28, 28), torch.randn(4096 * 1000, 1)
), batch_size=4096)

# The accuracy is 86.9% (which is better than the dense net!?)

SAVED_MODEL_PATH = "./data/model_simple_slp.pth"

model = reload_cached(SAVED_MODEL_PATH, Net)
model.eval()
print(model)

"""
with open("./metrics/simple_slp/fp32_eager_b16.log", "w") as f:
  for w in time_inference(test_dataloader_16, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_eager_b32.log", "w") as f:
  for w in time_inference(test_dataloader_32, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_eager_b64.log", "w") as f:
  for w in time_inference(test_dataloader_64, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_eager_b128.log", "w") as f:
  for w in time_inference(test_dataloader_128, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_eager_b256.log", "w") as f:
  for w in time_inference(test_dataloader_256, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_eager_b512.log", "w") as f:
  for w in time_inference(test_dataloader_512, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_eager_b1024.log", "w") as f:
  for w in time_inference(test_dataloader_1024, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_eager_b2048.log", "w") as f:
  for w in time_inference(test_dataloader_2048, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_eager_b4096.log", "w") as f:
  for w in time_inference(test_dataloader_4096, model):
    print(w, file=f)
"""

example_inputs = (get_example_input(),)

model = torch.export.export(model, example_inputs).module()

print(model)

"""
with open("./metrics/simple_slp/fp32_fx_b16.log", "w") as f:
  for w in time_inference(test_dataloader_16, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_fx_b32.log", "w") as f:
  for w in time_inference(test_dataloader_32, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_fx_b64.log", "w") as f:
  for w in time_inference(test_dataloader_64, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_fx_b128.log", "w") as f:
  for w in time_inference(test_dataloader_128, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_fx_b256.log", "w") as f:
  for w in time_inference(test_dataloader_256, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_fx_b512.log", "w") as f:
  for w in time_inference(test_dataloader_512, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_fx_b1024.log", "w") as f:
  for w in time_inference(test_dataloader_1024, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_fx_b2048.log", "w") as f:
  for w in time_inference(test_dataloader_2048, model):
    print(w, file=f)
with open("./metrics/simple_slp/fp32_fx_b4096.log", "w") as f:
  for w in time_inference(test_dataloader_4096, model):
    print(w, file=f)
"""

"""
import torch.ao.quantization.quantizer.x86_inductor_quantizer as qqq
quantizer = qqq.X86InductorQuantizer()
opconf = qqq.get_default_x86_inductor_quantization_config()
quantizer.set_global(opconf)
model = prepare_pt2e(model, quantizer)
model(*example_inputs)
model = convert_pt2e(model)

print(model)
"""

quantizer = shir.BackendQuantizer()

model = prepare_pt2e(model, quantizer)
model(*example_inputs)  # calibration
model = convert_pt2e(model)
test_loop(test_dataloader, model, loss_fn)

"""
with open("./metrics/simple_slp/shir_fx_b16.log", "w") as f:
  for w in time_inference(test_dataloader_16, model):
    print(w, file=f)
with open("./metrics/simple_slp/shir_fx_b32.log", "w") as f:
  for w in time_inference(test_dataloader_32, model):
    print(w, file=f)
with open("./metrics/simple_slp/shir_fx_b64.log", "w") as f:
  for w in time_inference(test_dataloader_64, model):
    print(w, file=f)
with open("./metrics/simple_slp/shir_fx_b128.log", "w") as f:
  for w in time_inference(test_dataloader_128, model):
    print(w, file=f)
with open("./metrics/simple_slp/shir_fx_b256.log", "w") as f:
  for w in time_inference(test_dataloader_256, model):
    print(w, file=f)
with open("./metrics/simple_slp/shir_fx_b512.log", "w") as f:
  for w in time_inference(test_dataloader_512, model):
    print(w, file=f)
with open("./metrics/simple_slp/shir_fx_b1024.log", "w") as f:
  for w in time_inference(test_dataloader_1024, model):
    print(w, file=f)
with open("./metrics/simple_slp/shir_fx_b2048.log", "w") as f:
  for w in time_inference(test_dataloader_2048, model):
    print(w, file=f)
with open("./metrics/simple_slp/shir_fx_b4096.log", "w") as f:
  for w in time_inference(test_dataloader_4096, model):
    print(w, file=f)
"""

# shir.config.TRY_NARROW_TYPE = False
# shir.config.TRY_COPY_AOT = False
torchdynamo.reset()
model = torch.compile(backend=shir.compiler)(model)
# model(*example_inputs)

# use the model
shir.config.FPGA_PRINT_RTINFO = False

time_inference(test_dataloader_4096, model)
"""
time_inference(test_dataloader_16, model) # compile
with open("./metrics/simple_slp/shir_fpga_b16.log", "w") as f:
  for w in time_inference(test_dataloader_16, model):
    print(w, file=f)
time_inference(test_dataloader_32, model) # compile
with open("./metrics/simple_slp/shir_fpga_b32.log", "w") as f:
  for w in time_inference(test_dataloader_32, model):
    print(w, file=f)
time_inference(test_dataloader_64, model) # compile
with open("./metrics/simple_slp/shir_fpga_b64.log", "w") as f:
  for w in time_inference(test_dataloader_64, model):
    print(w, file=f)
time_inference(test_dataloader_128, model) # compile
with open("./metrics/simple_slp/shir_fpga_b128.log", "w") as f:
  for w in time_inference(test_dataloader_128, model):
    print(w, file=f)
time_inference(test_dataloader_256, model) # compile
with open("./metrics/simple_slp/shir_fpga_b256.log", "w") as f:
  for w in time_inference(test_dataloader_256, model):
    print(w, file=f)
time_inference(test_dataloader_512, model) # compile
with open("./metrics/simple_slp/shir_fpga_b512.log", "w") as f:
  for w in time_inference(test_dataloader_512, model):
    print(w, file=f)
time_inference(test_dataloader_1024, model) # compile
with open("./metrics/simple_slp/shir_fpga_b1024.log", "w") as f:
  for w in time_inference(test_dataloader_1024, model):
    print(w, file=f)
time_inference(test_dataloader_2048, model) # compile
with open("./metrics/simple_slp/shir_fpga_b2048.log", "w") as f:
  for w in time_inference(test_dataloader_2048, model):
    print(w, file=f)
time_inference(test_dataloader_4096, model) # compile
with open("./metrics/simple_slp/shir_fpga_b4096.log", "w") as f:
  for w in time_inference(test_dataloader_4096, model):
    print(w, file=f)
"""

