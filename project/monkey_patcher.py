import torch

aten = torch.ops.aten

_patched_calls = {}

def register_patched_fn(original_ns, attr):
  original = getattr(original_ns, attr)
  def wrapped(f):
    def unpack(*args):
      return f(original, *args)

    assert original not in _patched_calls, "Original function is already patched"
    _patched_calls[original] = (original_ns, attr, unpack)
    return f
  return wrapped

class Patcher:
  def __enter__(self):
    for ns, attr, v in _patched_calls.values():
      setattr(ns, attr, v)

  def __exit__(self, type, value, rb):
    for original, (ns, attr, _) in _patched_calls.items():
      setattr(ns, attr, original)

@register_patched_fn(aten.max_pool2d, "default")
def patched_max_pool2d(old_fn, x, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False):
  return old_fn(x.float(), kernel_size, stride, padding, dilation, ceil_mode).int()
