# based on the PyTorch tutorial:
# https://pytorch.org/tutorials/beginner/basics/intro.html

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from pathlib import Path
import time

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

def get_example_input():
  for X, _ in train_dataloader:
    return X

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
  return correct, test_loss

def time_inference(dataloader, model):
  times = []
  with torch.no_grad():
    for X, _ in dataloader:
      _start = time.perf_counter_ns()
      model(X)
      _end = time.perf_counter_ns()
      times.append(_end - _start)
  return times

batch_size = 64
loss_fn = nn.CrossEntropyLoss()

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

def reload_cached(
  file_saved_weights: str, model_ctor,
  learning_rate=1e-3, epochs=10
):
  if Path(file_saved_weights).is_file():
    print("Loading saved model")
  else:
    print("Model does not exist, training!")

    model = model_ctor()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train_loop(train_dataloader, model, loss_fn, optimizer)
      test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), file_saved_weights)

  model = model_ctor()

  model.load_state_dict(torch.load(file_saved_weights))
  model.eval()

  return model
