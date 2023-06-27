# based on the PyTorch tutorial:
# https://pytorch.org/tutorials/beginner/basics/intro.html

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from pathlib import Path

SAVED_MODEL_PATH = "./data/model_mnist.pth"

training_data = datasets.MNIST(
  root="data",
  train=True,
  download=True,
  transform=ToTensor()
)

test_data = datasets.MNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor()
)

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10),
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  # Set the model to training mode - important for batch normalization and dropout layers
  # Unnecessary in this situation but added for best practices
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
  # Set the model to evaluation mode - important for batch normalization and dropout layers
  # Unnecessary in this situation but added for best practices
  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
  # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

batch_size = 64
loss_fn = nn.CrossEntropyLoss()

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

if not Path(SAVED_MODEL_PATH).is_file():
  print("Model does not exist, training!")

  learning_rate = 1e-3
  epochs = 10

  model = Net()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  train_features, train_labels = next(iter(train_dataloader))
  print(f"Feature batch shape: {train_features.size()}")
  print(f"Labels batch shape: {train_labels.size()}")

  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
  print("Done!")

  torch.save(model.state_dict(), SAVED_MODEL_PATH)

print("Loading saved model:")
model = Net()
model.load_state_dict(torch.load(SAVED_MODEL_PATH))
model.eval()

test_loop(test_dataloader, model, loss_fn)

print(model)

for X, _ in train_dataloader:
  example_inputs = X
  break

import shir_backend
from torch.ao.quantization import (
  QConfig,
  QConfigMapping,
)
from torch.ao.quantization.quantize_fx import (
  prepare_fx,
  _convert_to_reference_decomposed_fx,  # XXX: private API
)

qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_qconfig)
model = prepare_fx(model, qconfig_mapping, example_inputs)
# calibration
model(example_inputs)
# XXX: this is private API
model = _convert_to_reference_decomposed_fx(model)
print(model)

torch._dynamo.reset()
model = torch.compile(backend=shir_backend.compiler)(model)
test_loop(test_dataloader, model, loss_fn)
