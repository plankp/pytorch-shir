"""
Where the various different SHIR graph modules + compilation logic are defined
"""

import torch
from torch.fx import GraphModule, Node
from typing import Tuple, List, Optional, Any
from . import types, layout, config
from pathlib import Path
import shutil
import subprocess

def _collect_inout_nodes(gm: GraphModule) -> Tuple[List[Node], Node]:
  placeholders = []
  output = None
  for n in gm.graph.nodes:
    if n.op == "placeholder":
      tinfo = n.meta.get("val")
      assert tinfo is not None, "Placeholder must be a tensor"
      assert all((isinstance(d, int) for d in tinfo.shape)), "Dynamic shapes are not supported"

      placeholders.append(n)
    elif n.op == "output":
      assert len(n.args) == 1, "Only single output node is supported"
      node = n.args[0]
      tinfo = node.meta.get("val")
      assert tinfo is not None, "Output must be a tensor"
      assert all((isinstance(d, int) for d in tinfo.shape)), "Dynamic shapes are not supported"

      if output is None:
        output = node
      assert output == node, "Two output nodes returning different values"
  return (placeholders, output)

def _reshape_region(t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
  row_stride = t.stride(0)
  ndims = len(shape)

  if ndims == 0:
    # scalars have no stride
    stride = ()
  elif ndims == 1:
    # it is contiguous along the row
    stride = (1,)
  elif ndims == 2:
    # the column is contiguous along each row
    stride = (row_stride, 1)
  else:
    # decide how many (inner) dimensions are allocated on a single column.
    dims_on_col = 2 if ndims == 4 else ndims - 1

    # compute the physical shape,
    # which may be different when channel-last format is used.
    phys_shape = shape
    if config.USE_CHANNEL_LAST:
      phys_shape = [shape[0], *shape[2:], shape[1]]

    # compute the strides in reverse
    acc = 1
    stride = []
    for i, y in enumerate(reversed(phys_shape)):
      if i == dims_on_col:
        # beyond this point, we are skipping over rows
        acc = row_stride

      stride.append(acc)
      acc = y * acc

    # then reverse and realign with the logical shape as needed
    if config.USE_CHANNEL_LAST:
      stride = (stride[-1], stride[0], *stride[-2:0:-1])
    else:
      stride = tuple(reversed(stride))

  return t.as_strided(shape, stride)

# as the FPGA wills it, we actually preallocate memory for input and output
# data. the caller does not pass data via __call__. instead, they should use
# get_in_tensor to copy the values. for outputs, it will always return the
# same memory location, so the caller is expected to do a copy (if needed).
class SHIRGraphFpgaModule(torch.nn.Module):
  _driver: Any  # something that behaves like driver.py
  _layout: layout.MemoryLayout
  _gbs_file: Path
  _buffer: Any  # buffer allocated by the driver
  _inputs: List[torch.Tensor]
  _output: torch.Tensor

  def __init__(self, input_mapping, output_shape, driver, layout_file, gbs_file):
    super().__init__()
    self._driver = driver
    self._layout = layout.read_layout_file(layout_file)
    self._gbs_file = gbs_file

    # allocate the buffer
    sz = self._layout.bytes_needed(round_to_page=True)
    meminfo = self._driver.alloc_buffer(sz)
    self._buffer = meminfo

    inputs = [None] * len(input_mapping)
    output = None

    for entry in self._layout._entries:
      # some inputs have reduced bitwidth and are not representable by
      # PyTorch. leave the cell as None in those cases.
      #
      # it is technically possible to have a int32 buffer reduced as s8.
      # in this case, the input will have a corresponding tensor since
      # we can use int8 for that.
      region = None
      if entry.get_torch_type() is not None:
        region = entry.from_buffer(meminfo)

      if entry.name == "result":
        output = _reshape_region(region, output_shape)
      elif region is not None and entry.name in input_mapping:
        # then this must be an input (not host-buffered intermediate data)
        (node_id, node_shape) = input_mapping[entry.name]
        inputs[node_id] = _reshape_region(region, node_shape)

    self._inputs = inputs
    self._output = output

  def __call__(self) -> torch.Tensor:
    # reconfigure the fpga if needed
    fpga = self._driver.configure_gbs(self._gbs_file)
    fpga.reset()

    with fpga.prepare_buffer(self._buffer, len(self._buffer)) as wsid:
      fpga.start_computation()
      while not fpga.is_complete():
        pass  # spin

      if config.FPGA_PRINT_RTINFO:
        cycles = fpga.read_mmio64(0, 0x88)
        readreq = fpga.read_mmio64(0, 0xC0)
        readpending = fpga.read_mmio64(0, 0xD0)
        readaf = fpga.read_mmio64(0, 0xE0)
        writereq = fpga.read_mmio64(0, 0xC8)
        writepending = fpga.read_mmio64(0, 0xD8)
        writeaf = fpga.read_mmio64(0, 0xE8)

        print(
          "Execution time (cycles): ", cycles, "\n"
          "Read requests          : ", readreq, " (of which ", readpending, " pending)\n"
          "Write requests         : ", writereq, " (of which ", writepending, " pending)\n"
          "Read request buffer  ", readaf, " times almost full\n"
          "Write request buffer ", writeaf, " times almost full",
          sep="",
        )

    return self._output

  def get_in_tensor(self, index) -> torch.Tensor:
    return self._inputs[index]

  def __del__(self):
    if self._buffer is not None:
      self._driver.free_buffer(self._buffer)
      self._buffer = None

