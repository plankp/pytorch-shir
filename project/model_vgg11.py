# Using a pretrained vgg11 (batchnorm) model from
# chenyaofo/pytorch-cifar-models
#
# this is because those ones are trained on cifar,
# which is an easier dataset to setup (compared to imagenet)

from routine_cifar_10 import (
  test_loop,
  test_dataloader,
  loss_fn,
  get_example_input,
)
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

# prepare to monkey patch this call to flatten
_old_torch_flatten_impl = torch.flatten
torch.flatten = shir_intrinsic.flatten_bridge

# this model has 90.5% accuracy for what it's worth
# (since we don't know if we are mixing training data)

model = torch.hub.load(
  "chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", 
  pretrained=True,
  #force_reload=True,   # in case you run into strange http errors
  trust_repo=True
)
model.eval()
test_loop(test_dataloader, model, loss_fn)

example_inputs = (get_example_input(),)

model, guards = torchdynamo.export(
  model,
  *copy.deepcopy(example_inputs),
  aten_graph=True,
)
torch.flatten = _old_torch_flatten_impl

quantizer = shir_quantizer.BackendQuantizer()
model = prepare_pt2e(model, quantizer)
model(*example_inputs)
model = convert_pt2e(model)

torchdynamo.reset()
model = torch.compile(backend=shir_backend.compiler)(model)
model(*example_inputs)

# test_loop(test_dataloader, model, loss_fn)
