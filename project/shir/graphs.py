"""
Where the SHIRGraphModule is defined
"""

import torch
from typing import Any, Tuple, List, Dict, Optional, Union
from . import lowering, types
from functools import reduce
from itertools import count
import os
import shutil
import subprocess

# some debug flags and configuration stuff
_CONF_EMIT_SHIR_CODE = True
_CONF_EMIT_DATA_FILES = True
_CONF_EMIT_OUTPUT_DIR = "./data/generated"
_CONF_TEMPLATE_DIR = "./template"
_CONF_PERFORM_SIMULATION = False
_CONF_FPGA_CACHELINE_BITS = 512

def _tensor_to_matrix_csv(t: torch.Tensor, f):
  # XXX: decoding int64's is a bit annoying, ignore it!
  assert t.dtype in {torch.int8, torch.uint8, torch.int16, torch.int32}

  if t.ndim == 0:
    t = torch.unsqueeze(t, 0)
  if t.ndim == 1:
    t = torch.unsqueeze(t, 0)
  if t.ndim > 2:
    t = torch.flatten(t, 1)

  for row in t:
    d = row.shape[0]
    for i in range(d - 1):
      print(row[i].item(), ",", sep="", end="", file=f)
    print(row[d - 1].item(), file=f)

def _load_memory_layout_file(fname: str) -> List[Tuple[str, int, Union[types.SI, types.UI], List[int]]]:
  # this is actually good enough for our purposes
  # since we're limited to the types supported by PyTorch!
  supported_types = {
    "u8": types.UI(8),
    "s8": types.SI(8),
    "s16": types.SI(16),
    "s32": types.SI(32),
    "s64": types.SI(64),
  }
  entries = []
  with open(fname, "r") as f:
    while True:
      line1 = f.readline()
      if not line1:
        break

      line2 = f.readline()
      (name, addr, ty) = line1.rstrip().split("\t")
      dims = [int(d) for d in line2.rstrip().split(",")]
      entries.append((name, int(addr, 16), supported_types[ty], dims))
  return entries

def _load_result_from_memfile(fname: str, result: torch.Tensor) -> torch.Tensor:
  original_shape = result.shape
  if result.ndim < 2:
    result = result.reshape((1, -1))
  if result.ndim > 2:
    result = result.reshape((original_shape[0], -1))

  outer_dim = result.size(0)
  inner_dim = result.size(1)
  dtype = types.get_scalar_type(result.dtype)

  # in our case (since the widest possible data we handle is i64),
  # each outer_dim starts on a new cacheline.
  #
  # occassionally, your inner_dim might be so large to the point that the data
  # cannot fit on a single cacheline (e.g. N rows of 20 i32's).
  # in which case, it will fill as many "full" pieces of data as possible,
  # move on to the next cacheline, then repeat until all data is there.
  #
  # then the next outer_dim starts NOT on the next available slot,
  # but on the next cacheline.
  cachelines_per_row = (inner_dim * dtype.bits + _CONF_FPGA_CACHELINE_BITS - 1) // _CONF_FPGA_CACHELINE_BITS

  # read from the memory file from end to start!
  # we need to open it as a binary file since we read the file from the end!
  with open(fname, "rb") as f:
    # assumptions: the memory file starts with a few lines of comments,
    # therefore our file is never empty or just one line.
    f.seek(0, os.SEEK_END)
    for outer in range(outer_dim - 1, -1, -1):
      # recall that each outer_dim may span across multiple cachelines
      line_data = 0
      for line in range(cachelines_per_row - 1, -1, -1):
        # in hex mode, each line has 128 data nibbles + '\n'
        f.seek(-129, os.SEEK_CUR)
        line = f.read(128)
        f.seek(-128, os.SEEK_CUR)

        # the line goes from high memory to low memory
        line_data <<= _CONF_FPGA_CACHELINE_BITS
        line_data |= int(line, base=16)

      for inner in range(inner_dim):
        result[outer, inner] = dtype.cast(line_data)
        line_data >>= dtype.bits

  return result.reshape(original_shape)

def _collect_inout_nodes(gm: torch.fx.GraphModule) -> Tuple[List[str], torch.fx.Node]:
  placeholders = []
  output = None
  for n in gm.graph.nodes:
    if n.op == "placeholder":
      tinfo = n.meta.get("val")
      assert tinfo is not None, "Placeholder must be a tensor"
      assert all((isinstance(d, int) for d in tinfo.shape)), "Dynamic shapes are not supported"

      placeholders.append(n.target)
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

def _has_many_uses(node: torch.fx.Node) -> bool:
  user_count = len(node.users)
  if user_count > 1:
    return True

  used = False
  for user in node.users:
    # each user may have multiple occurrences / uses of a single node
    for n in user.all_input_nodes:
      if n != node:
        continue
      if used:
        return True
      used = True

  return False

# a counter for counting instances of SHIRGraphModule
# (it's not perfect, but at least it reduces the amount of collisions)
_iid_counter = count()

class SHIRGraphModule(torch.nn.Module):
  # assumption: graph does not change (it really shouldn't!)
  gm: torch.fx.GraphModule
  _inst_id: int
  _call_id: int
  _compiled: bool
  _inout_nodes: Optional[Tuple[List[str], torch.fx.Node]]

  def __init__(self, gm: torch.fx.GraphModule):
    super().__init__()
    self._inst_id = next(_iid_counter)
    self.gm = gm
    self._compiled = False
    self._call_id = 0
    self._inout_nodes = None

    self._clname = f"Module{self._inst_id}"
    self._output_dir = os.path.join(_CONF_EMIT_OUTPUT_DIR, f"module{self._inst_id}")

  def __call__(self, *args):
    self._call_id += 1
    self.compile()

    if not _CONF_EMIT_DATA_FILES:
      # if you aren't going to emit data files, then obviously the compiled
      # model won't have anything to work with.
      #
      # in that case, we have Python evaluate the result using the fx graph.
      return self.gm(*args)

    data_dir = os.path.join(self._output_dir, f"data_{self._call_id}")
    if not os.path.exists(data_dir):
      os.mkdir(data_dir)

    for i, arg in enumerate(args):
      with open(os.path.join(data_dir, f"arg{i}.csv"), "w", encoding="utf-8") as argf:
        _tensor_to_matrix_csv(arg, argf)

    if not _CONF_PERFORM_SIMULATION:
      return self.gm(*args)

    # at this point, we want to generate data and simulate,
    # but we don't want to generating VHDL.
    #
    # here, since the current working directory is already inside the output
    # directory, reference the particular data folder name directly instead of
    # using data_dir (since data_dir may contain relative paths, which is bad)
    subprocess.run(['sbt', f'run --no-gen --sim "data_{self._call_id}"'], check=True, cwd=self._output_dir)

    # assume it didn't crash, then we would have ended up with a memory dump,
    # from which, we can recover the result.
    memory_dump_file = os.path.join(self._output_dir, "out", self._clname, "memory.dump")
    info = self._inout_nodes[1].meta.get("val")
    return _load_result_from_memfile(
      memory_dump_file,
      torch.empty(info.shape, dtype=info.dtype)
    )

  def compile(self):
    if self._compiled:
      return

    self._compiled = True
    if _CONF_EMIT_SHIR_CODE or _CONF_EMIT_DATA_FILES:
      self._prepare_directory()
    if _CONF_EMIT_SHIR_CODE:
      self._emit_source()
      subprocess.run(['sbt', 'run --gen --no-sim'], check=True, cwd=self._output_dir)

  def _prepare_directory(self):
    # purge the output directory
    if os.path.exists(self._output_dir):
      shutil.rmtree(self._output_dir)

    # copy the template directory into the output directory
    shutil.copytree(_CONF_TEMPLATE_DIR, self._output_dir)

  def _emit_source(self):
    self._inout_nodes = _collect_inout_nodes(self.gm)
    with open(os.path.join(
      self._output_dir, "src", "main", "scala",
      f"{self._clname}.scala"
    ), "w", encoding="utf-8") as f:
      print("// This file is autogenerated", file=f)
      print("import core._", file=f)
      print("import algo._", file=f)
      print("import java.nio.file.Paths", file=f)

      print(file=f)
      print("object", self._clname, "extends GeneratedModel {", file=f)

      print(file=f)
      print("  val name: String = \"", self._clname, "\"", sep="", file=f)

      print(file=f)
      print("  def main(args: Array[String]): Unit = Util.drive(this, args)", file=f)

      print(file=f)
      self._emit_method_generate_ir(f)

      print(file=f)
      self._emit_method_load_data(f)

      print("}", file=f)

  def _emit_method_load_data(self, f):
    print("  def loadData(folder: String): Predef.Map[String, Seq[Seq[Int]]] = Predef.Map(", file=f)

    # inputs are always taken from .csv's
    for i, arg in enumerate(self._inout_nodes[0]):
      print("    \"", arg, "\" -> Util.readIntCSV(Paths.get(folder, \"arg", i, ".csv\").toFile()),", sep="", file=f)

    # result is not taken from .csv, but we still need to allocate dummy data
    # so that the RAM size is correctly calculated during simulation.
    shape = self._inout_nodes[1].meta.get("val").shape
    inner = reduce(lambda x, y: x * y, shape[1:], 1)
    print("    \"result\" -> new UniformSeq(new UniformSeq(0, ", inner, "), ", shape[0], ")", sep="", file=f)

    print("  )", file=f)

  def _emit_method_generate_ir(self, f):
    print("  def generateIR(): Expr = {", file=f)

    lets_needed = 0
    for n in self.gm.graph.nodes:
      # assume every node that has many uses needs to be let-bound,
      # which is definitely the case for tensors (which are SeqType's)
      has_many_uses = _has_many_uses(n)
      if has_many_uses:
        lets_needed += 1

      # furthermore, input (placeholder) and output nodes must be 2D in SHIR.
      match n.op:
        case "placeholder":
          typ = types.get_element_type(n)
          shape = ", ".join((str(d) for d in n.meta.get("val").shape))

          if has_many_uses:
            print(
              "  { val _init = core.TypeChecker.check(algo.torch.TInput(",
              typ.name(), ", \"", n.target, "\", Seq(", shape, ")))\n",
              "    val _param = core.ParamDef(_init.t)\n",
              "    core.Let(_param,\n",
              "  { val ", a.name, " = core.ParamUse(_param)",
              sep="", file=f
            )
          else:
            print(
              "    val ", n.name,
              " = core.TypeChecker.check(algo.torch.TInput(",
              typ.name(), ", \"", n.target, "\", Seq(", shape, ")))",
              sep="", file=f
            )

        case "output":
          # sometimes, due to unfortunate graph slicing, we may end up with
          # multiple outputs, which we cannot handle
          [retv] = n.args
          assert isinstance(retv, torch.fx.Node), "Only single node output is allowed"
          assert not has_many_uses  # not sure what this failing would mean...

          shape = retv.meta.get("val").shape
          dims = len(shape)
          v = retv.name

          if dims < 1:
            v = f"algo.Repeat({v}, 1)"
          if dims < 2:
            v = f"algo.Repeat({v}, 1)"
          if dims > 2:
            v = f"algo.torch.TFlatten({v}, 1, {dims - 1})"
          print("    core.TypeChecker.check(", v, ")", sep="", file=f)

        case "call_function":
          obj = lowering.fetch_lowering(n.target)
          expr = obj.lower(*n.args, **n.kwargs)
          if has_many_uses:
            print(
              "    val _init = core.TypeChecker.check(", expr, ")\n",
              "    val _param = core.ParamDef(_init.t)\n",
              "    core.Let(_param,\n",
              "  { val ", n.name, " = core.ParamUse(_param)",
              sep="", file=f
            )
          else:
            print(
              "    val ", n.name, " = core.TypeChecker.check(", expr, ")",
              sep="", file=f
            )

        case _:
          assert False, "Unhandled fx node type when emitting"

    for _ in range(lets_needed):
      print("  }, _init)", file=f)

    print("  }", file=f)
