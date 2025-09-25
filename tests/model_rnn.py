import torch
import torch.nn as nn
import torch.fx as fx
import shir

# took the TwoPatterns for LSTM, replaced it with RNN
class TwoPatternsRNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.RNN(1, 32, batch_first=True)
    self.fc1 = nn.Linear(32, 16)
    self.fc2 = nn.Linear(16, 4)

  def forward(self, x):
    # XXX: lstm seems to allow optional hidden states as argument
    #      (defaults to zero)
    output, _ = self.rnn(x)
    # XXX: accelerator only outputs the last sequence
    # XXX: this is the same as h[-1]?
    last = output[:, -1]

    return self.fc2(self.fc1(last))

# do the training and whatnot here
model = TwoPatternsRNN()
model.eval()

x = torch.zeros([1, 128, 1])
if True:
  y = model(x)
  print(f"Input shape:  {x.shape}")
  print(f"Output shape: {y.shape}")
  print(torch.ops.aten.rnn_relu.input._schema)
  print(torch.ops.aten.rnn_tanh.input._schema)

# you must have some dummy input in order to export
gfx = torch.export.export(model, (x,)).module()
gfx.print_readable()

import shir.backend_lstm as I
if False:
  # for testing purposes, this one is better since it's (marginally) faster
  # and the stacktraces are not as nasty (in case something goes wrong)
  model = I.compiler(gfx, (x,))
else:
  # lazy compile: compilation happens on first use
  model = torch.compile(gfx, backend=I.compiler)

print(model(x))

