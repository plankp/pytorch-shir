"""
Where the SHIRGraphModule is defined
"""

import torch
from torch.fx import GraphModule, Node
from typing import Tuple, List, Optional, Any
from . import lowering, types, layout, config, driver
from functools import reduce
from itertools import count
from pathlib import Path
import mmap
import ctypes
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

def has_many_uses(node: Node) -> bool:
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

def _reshape_region(t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
  inner = outer = 1
  ndim = len(shape)
  if ndim == 1:
    inner = shape[0]
  elif ndim > 1:
    outer = shape[0]
    inner = reduce(lambda x, y: x * y, shape[1:])

  return t.as_strided((outer, inner), (t.stride(0), 1)).view(shape)

# a counter for counting instances of SHIRGraphModule
# (it's not perfect, but at least it reduces the amount of collisions)
_iid_counter = count()

# since we use CapabilityBasedPartitioner when extracting the subgraph,
# making this an nn.Module is easier to work with.
#
# aside from that, this does not behave like a conventional module:
# usually the module receives its imput via the __call__ method where it will
# returns the result as a tensor.
# in our case, the input values are expected to be provided beforehand by
# mutating preallocated tensors and the output always returns the same memory
# location.
class SHIRGraphModule(torch.nn.Module):
  # we assume the graph does not change (which it shouldn't...)
  gm: GraphModule
  _inst_id: int
  _call_id: int
  _clname: str
  _output_dir: Path
  _inout_nodes: Optional[Tuple[List[Node], Node]]
  _inout_tensors: Optional[Tuple[List[torch.Tensor], torch.Tensor]]
  _layout: Optional[layout.MemoryLayout]
  _buffer: Optional[Tuple[mmap.mmap, int]]

  def __init__(self, gm: GraphModule):
    super().__init__()
    self._inst_id = next(_iid_counter)
    self._call_id = 0
    self.gm = gm
    self._clname = f"Module{self._inst_id}"
    self._output_dir = Path(config.EMIT_OUTPUT_DIR) / f"module{self._inst_id}"

    self._inout_nodes = None
    self._inout_tensors = None
    self._layout = None
    self._buffer = None

  def __del__(self):
    self._inout_tensors = None
    if self._buffer:
      self._buffer[0].close()
      self._buffer = None  # probably redundant

  def get_in_tensor(self, index) -> torch.Tensor:
    return self.get_inout_tensors()[0][index]

  def get_inout_tensors(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
    t = self._inout_tensors
    if t is None:
      meminfo = self._buffer
      if meminfo is None:
        # only enforce page length requirement when running with FPGA.
        sz = self._layout.bytes_needed(round_to_page=config.PERFORM_SYNTHESIS)
        meminfo = (mmap.mmap(-1, sz), sz)
        self._buffer = meminfo

      arg_mapping = {arg.target: (i, arg) for i, arg in enumerate(self._inout_nodes[0])}
      inputs = [None] * len(self._inout_nodes[0])
      output = None

      for entry in self._layout._entries:
        region = entry.frombuffer(meminfo[0])
        if entry.name == "result":
          output = _reshape_region(
            region,
            self._inout_nodes[1].meta.get("val").shape
          )
        else:
          (arg_id, arg_node) = arg_mapping[entry.name]
          inputs[arg_id] = _reshape_region(
            region,
            arg_node.meta.get("val").shape
          )

      t = (inputs, output)
      self._inout_tensors = t
    return t

  def __call__(self) -> torch.Tensor:
    # call this first to ensure the memory buffer exists!
    # (it's also easier for non-synthesis targets to do it here)
    args, result = self.get_inout_tensors()
    self._call_id += 1

    if config.PERFORM_SYNTHESIS:
      with self._driver.find_and_open_fpga(config.ACCEL_UUID) as handle:
        buffer, sz = self._buffer
        with self._driver.prepare_buffer(handle, buffer, sz) as wsid:
          self._driver.start_computation(handle)
          while not self._driver.is_complete(handle):
            pass  # spin

          cycles = self._driver.read_mmio64(handle, 0, 0x88)
          readreq = self._driver.read_mmio64(handle, 0, 0xC0)
          readpending = self._driver.read_mmio64(handle, 0, 0xD0)
          readaf = self._driver.read_mmio64(handle, 0, 0xE0)
          writereq = self._driver.read_mmio64(handle, 0, 0xC8)
          writepending = self._driver.read_mmio64(handle, 0, 0xD8)
          writeaf = self._driver.read_mmio64(handle, 0, 0xE8)

          print(
            "Execution time (cycles): ", cycles, "\n",
            "Read requests          : ", readreq, " (of which ", readpending, " pending)\n"
            "Write requests         : ", writereq, " (of which ", writepending, " pending)\n"
            "Read request buffer  ", readaf, " times almost full\n"
            "Write request buffer ", writeaf, " times almost full",
            sep="",
          )

    elif config.PERFORM_SIMULATION:
      data_dir = self._output_dir / f"data_{self._call_id}"
      if not data_dir.exists():
        data_dir.mkdir()

      for i, arg in enumerate(args):
        with (data_dir / f"arg{i}.csv").open("w", encoding="utf-8") as argf:
          layout.tensor_to_matrix_csv(arg, argf)

      # here, since the current working directory is already inside the output
      # directory, reference the particular data folder name directly instead of
      # using data_dir (since data_dir may contain relative paths, which is bad)
      subprocess.run(['sbt', f'run --no-gen --sim "data_{self._call_id}"'], check=True, cwd=self._output_dir)

      # assume it didn't crash, then we would have ended up with a memory dump,
      # from which, we can recover the result.
      memory_dump_file = self._output_dir / "out" / self._clname / "memory.dump"
      layout.read_memory_dump(memory_dump_file, self._layout.get_entry("result"), result)

    else:
      # clearly, the copy here is not needed,
      # but we do it anyway to keep the __call__ behavior uniform.
      result.copy_(self.gm(*args))

    return result

  def compile(self):
    self._prepare_directory()
    self._emit_source()
    self._generate_hardware_files()

  def _prepare_directory(self):
    if self._output_dir.exists():
      shutil.rmtree(self._output_dir)

    shutil.copytree(config.TEMPLATE_DIR, self._output_dir)

  def _generate_hardware_files(self):
    # the bare minimum of VHDL and layout file (and maybe others) generation
    subprocess.run(['sbt', 'run --gen --no-sim'], check=True, cwd=self._output_dir)
    memory_layout_file = self._output_dir / "out" / self._clname / "memory.layout"
    self._layout = layout.read_layout_file(memory_layout_file)

    if not config.PERFORM_SYNTHESIS:
      return

    synth_dir = self._output_dir / "synthesis"
    self._driver = driver.Wrapper(ctypes.cdll.LoadLibrary(
      synth_dir / "build_driver" / "libdriver.so"
    ))
    subprocess.run(
      ['./synthesize.sh', str(Path("..") / "out" / self._clname)],
      check=True, cwd=synth_dir
    )

  def _emit_source(self):
    self._inout_nodes = _collect_inout_nodes(self.gm)
    with (
      self._output_dir / "src" / "main" / "scala" / f"{self._clname}.scala"
    ).open("w", encoding="utf-8") as f:
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
      print("    \"", arg.target, "\" -> Util.readIntCSV(Paths.get(folder, \"arg", i, ".csv\").toFile()),", sep="", file=f)

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
      many_uses = has_many_uses(n)
      if many_uses:
        lets_needed += 1

      # furthermore, input (placeholder) and output nodes must be 2D in SHIR.
      match n.op:
        case "placeholder":
          typ = types.get_element_type(n)
          shape = ", ".join((str(d) for d in n.meta.get("val").shape))

          if many_uses:
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
          assert not many_uses  # not sure what this failing would mean...

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
          if many_uses:
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
