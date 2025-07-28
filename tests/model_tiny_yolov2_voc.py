# model and weight loading code based on gh:miladlink/TinyYoloV2
#
# the link for weights seems to be dead, but wayback machine seems to have
# saved a copy of it (around 60 MB):
#   https://pjreddie.com/media/files/yolov2-tiny-voc.weights

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch._dynamo as torchdynamo
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import shir

def load_conv_bn(buf, start, conv_layer, bn_layer):
  num_w = conv_layer.weight.numel()
  num_b = bn_layer.bias.numel()

  bn_layer.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
  start += num_b
  bn_layer.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
  start += num_b
  bn_layer.running_mean.data.copy_(torch.from_numpy(buf[start:start + num_b]))
  start += num_b
  bn_layer.running_var.data.copy_(torch.from_numpy(buf[start:start + num_b]))
  start += num_b

  conv_layer.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape_as(conv_layer.weight))
  start += num_w

  return start

def load_conv(buf, start, conv_layer):
  num_w = conv_layer.weight.numel()
  num_b = conv_layer.bias.numel()

  conv_layer.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
  start += num_b
  conv_layer.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape_as(conv_layer.weight))
  start += num_w

  return start

class TinyYoloV2(nn.Module):
  def __init__(self):
    super(TinyYoloV2, self).__init__()

    self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)

    self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(32)

    self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
    self.bn3 = nn.BatchNorm2d(64)

    self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
    self.bn4 = nn.BatchNorm2d(128)

    self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
    self.bn5 = nn.BatchNorm2d(256)

    self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
    self.bn6 = nn.BatchNorm2d(512)

    self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
    self.bn7 = nn.BatchNorm2d(1024)

    self.conv8 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
    self.bn8 = nn.BatchNorm2d(1024)

    # 125 comes from (5 + #classes) * #anchors = (5 + 20) * 5
    self.output = nn.Conv2d(1024, 125, 1, 1, 0)

  def forward(self, x):
    x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x)), 0.1), 2)
    x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x)), 0.1), 2)
    x = F.max_pool2d(F.leaky_relu(self.bn3(self.conv3(x)), 0.1), 2)
    x = F.max_pool2d(F.leaky_relu(self.bn4(self.conv4(x)), 0.1), 2)
    x = F.max_pool2d(F.leaky_relu(self.bn5(self.conv5(x)), 0.1), 2)
    x = F.max_pool2d(F.pad(F.leaky_relu(self.bn6(self.conv6(x)), 0.1), (0, 1, 0, 1), mode='replicate'), 2, 1)
    x = F.leaky_relu(self.bn7(self.conv7(x)), 0.1)
    x = F.leaky_relu(self.bn8(self.conv8(x)), 0.1)
    x = self.output(x)
    return x

  def load_weights(self, file):
    buf = np.fromfile(file, dtype=np.float32)

    # apparently the first four entries are actually int32 fields
    # skip over those
    start = 4

    start = load_conv_bn(buf, start, self.conv1, self.bn1)
    start = load_conv_bn(buf, start, self.conv2, self.bn2)
    start = load_conv_bn(buf, start, self.conv3, self.bn3)
    start = load_conv_bn(buf, start, self.conv4, self.bn4)
    start = load_conv_bn(buf, start, self.conv5, self.bn5)
    start = load_conv_bn(buf, start, self.conv6, self.bn6)
    start = load_conv_bn(buf, start, self.conv7, self.bn7)
    start = load_conv_bn(buf, start, self.conv8, self.bn8)
    load_conv(buf, start, self.output)

img_resolution = 416
batch_size = 64

transforms = v2.Compose([
  v2.ToImage(),
  v2.Resize(img_resolution),
  v2.CenterCrop(img_resolution),
  v2.ToDtype(torch.float32, scale=True),
])

training_data = torchvision.datasets.VOCSegmentation(
  root="data",
  image_set="train",
  download=True,
  transform=transforms,
  target_transform=transforms,
)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True)

def get_example_input():
  for X, _ in train_dataloader:
    return X

def time_inference(data, model):
  import time
  times = []
  with torch.no_grad():
    for X in data:
      _start = time.perf_counter_ns()
      model(X)
      _end = time.perf_counter_ns()
      times.append(_end - _start)
  return times

model = TinyYoloV2()
model.load_weights("data/yolov2-tiny-voc.weights")
model.eval()

PROFILE = "shir"
PROBLEM_SIZE_N = 1
PROBLEM_TRIPS  = 1
PROBLEM_INSTS  = 1000

_qex = get_example_input()[:PROBLEM_SIZE_N, :, :, :]
_qex = torch.concat([_qex] * ((PROBLEM_SIZE_N + (batch_size - 1)) // batch_size), axis=0)
example_inputs = (_qex,)

torchdynamo.reset()

if PROFILE == "shir":
  quantizer = shir.BackendQuantizer()
elif PROFILE == "x86":
  quantizer = xiq.X86InductorQuantizer()
  quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

with torch.no_grad():
  model = torch.export.export(model, example_inputs).module()

  if PROFILE in {"shir", "x86"}:
    model = prepare_pt2e(model, quantizer)
    model(*example_inputs)
    model = convert_pt2e(model)

  if PROFILE == "shir":
    import shir.backend2
    model = torch.compile(model, backend=shir.backend2.compiler)
  else:
    model = torch.compile(model)

"""
print(model(example_inputs[0]))
"""

shir.config.FPGA_PRINT_RTINFO = False
dummy_data = torch.zeros(PROBLEM_INSTS, PROBLEM_SIZE_N, 3, img_resolution, img_resolution)
with open(f"./metrics/tiny_yolov2_voc/{PROFILE}_{PROBLEM_SIZE_N}.log", "w") as f:
  for i in range(PROBLEM_TRIPS):
    for w in time_inference(dummy_data, model):
      print(w, file=f)

