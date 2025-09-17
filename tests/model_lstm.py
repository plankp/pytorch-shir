import torch
import torch.nn as nn
import torch.fx as fx
import shir

class TwoPatterns(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(1, 32, batch_first=True)
    self.fc1 = nn.Linear(32, 16)
    self.fc2 = nn.Linear(16, 4)

  def forward(self, x):
    # XXX: lstm seems to allow optional hidden states as argument
    #      (defaults to zero)
    output, (h, c) = self.lstm(x)
    # XXX: accelerator only outputs the last sequence
    # XXX: this is the same as h[-1]?
    last = output[:, -1]

    return self.fc2(self.fc1(last))

# do the training and whatnot here
model = TwoPatterns()
model.eval()

with torch.no_grad():
  # inputs is Batch by Hidden Units by Input Size
  #
  # XXX:
  # LSTM has default batch_first=False, which means it accepts inputs of
  # Hidden Units by Batch by Input Size...
  x = torch.zeros([1, 128, 1])
  if False:
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(torch.ops.aten.lstm.input._schema)

  # you must have some dummy input in order to export
  gfx = torch.export.export(model, (x,)).module()
  # gfx.print_readable()

  # lazy compile
  import shir.backend_lstm
  model = torch.compile(gfx, backend=shir.backend_lstm.compiler)
  # force compile to happen
  print(model(x))

