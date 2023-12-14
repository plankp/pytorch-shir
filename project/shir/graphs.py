"""
Where the various different SHIR graph modules + compilation logic are defined
"""

import torch
from torch.fx import GraphModule, Node
from typing import Tuple, List, Optional, Any
from . import lowering, types, layout, config
from pathlib import Path
import shutil
import subprocess
import tempfile

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
  return t.as_strided(
    layout.reshape_size_to_matrix(shape),
    (t.stride(0), 1)
  ).view(shape)

# a helper class that deals with interacting with sbt project being generated.
class SHIRProject:
  clname: str
  output_dir: Path

  def __init__(self, clname: str, output_dir: Path):
    self.clname = clname
    self.output_dir = output_dir

  def prepare_directory(self):
    if self.output_dir.exists():
      shutil.rmtree(self.output_dir)
    shutil.copytree(config.TEMPLATE_DIR, self.output_dir)

  def generate_hardware_files(self):
    subprocess.run(['sbt', 'run --gen --no-sim'], check=True, cwd=self.output_dir)

  def load_layout_file(self) -> layout.MemoryLayout:
    memory_layout_file = self.output_dir / "out" / self.clname / "memory.layout"
    return layout.read_layout_file(memory_layout_file)

  def read_memory_dump(self, entry: layout.LayoutEntry, inner_len: int) -> torch.Tensor:
    memory_dump_file = self.output_dir / "out" / self.clname / "memory.dump"
    return layout.read_memory_dump(memory_dump_file, entry, inner_len)

  def get_gbs_file(self) -> Path:
    return self.output_dir / "synthesis" / "build_synth" / "hello_afu_unsigned_ssl.gbs"

  def synthesize(self):
    synth_dir = self.output_dir / "synthesis"
    subprocess.run(
      ['./synthesize.sh', str(Path("..") / "out" / self.clname)],
      check=True, cwd=synth_dir
    )

  def emit_source(self, gm, arg_elt_types):
    with (
      self.output_dir / "src" / "main" / "scala" / f"{self.clname}.scala"
    ).open("w", encoding="utf-8") as f:
      print("// This file is autogenerated", file=f)
      print("import core._", file=f)
      print("import algo._", file=f)
      print("import java.nio.file.Paths", file=f)

      print(file=f)
      print("object", self.clname, "extends GeneratedModel {", file=f)

      print(file=f)
      print("  override val name: String = \"", self.clname, "\"", sep="", file=f)

      print(file=f)
      print("  def main(args: Array[String]): Unit = Util.drive(this, args)", file=f)

      print(file=f)
      self._emit_method_load_data(f, gm)

      print(file=f)
      self._emit_method_extra_rewrites(f, gm, arg_elt_types)

      print(file=f)
      self._emit_method_generate_ir(f, gm, arg_elt_types)

      print("}", file=f)

  def _emit_method_load_data(self, f, gm):
    print("  override def loadData(folder: String): Predef.Map[String, Seq[Seq[Int]]] = Predef.Map(", file=f)

    output_node = None
    placeholder_id = 0
    for n in gm.graph.nodes:
      if n.op == "placeholder":
        # only inputs come from csv's
        print(
          "    \"arg", placeholder_id,
          "\" -> Util.readIntCSV(Paths.get(folder, \"arg", placeholder_id,
          ".csv\").toFile()),",
          sep="", file=f
        )
        placeholder_id += 1

      elif n.op == "output":
        assert output_node is None, f"Multi-output node not supported"
        assert isinstance(n.args[0], Node), "Multi-valued output node not supported"
        output_node = n.args[0].meta.get("val")

    # for the result, we still need to give it some dummy value for simulation
    # to determine the RAM size.
    (outer, inner) = layout.reshape_size_to_matrix(output_node.shape)
    print(
      "    \"result\" -> new UniformSeq(new UniformSeq(0, ",
      inner, "), ", outer, ")",
      sep="", file=f
    )

    print("  )", file=f)

  def _emit_method_extra_rewrites(self, f, gm, arg_elt_types):
    print("  override def extraRewrites(): Seq[(core.compile.CompilerPhase, core.rewrite.RewriteStep)] = {", file=f)
    print("    import core.compile.CompilerPhase", file=f)
    print("    import core.rewrite.{RewriteAll, RewriteStep, RewriteTargeted}", file=f)
    print("    import backend.hdl.arch.{ArchCompiler, MapCompiler}", file=f)
    print("    import backend.hdl.arch.device.DeviceSpecificCompiler", file=f)
    print("    import backend.hdl.arch.rewrite.{InputBufferingRules, ParallelizeDotProductRules}", file=f)
    print("    import backend.hdl.arch.mem.MemFunctionsCompiler", file=f)
    print("    Seq(", file=f)

    # double buffer every input
    # the target is a descending sequence of integers...
    idx = ", ".join((str(x - 1) for x in range(len(arg_elt_types), 0, -1)))
    print(f"(MemFunctionsCompiler.phaseAfter, RewriteStep(RewriteTargeted({idx}), Seq(", file=f)
    print("  InputBufferingRules.doubleBufferRead", file=f)
    print("))),", file=f)

    # fully parallelize the dot product for matrix multiplications
    for n in gm.graph.nodes:
      if n.op == "call_function" and n.target == torch.ops.shir_intrinsic.int_addmm.default:
        # for n*k cross m*k, we want to parallelize by k
        k = n.args[1].meta.get("val").shape[1]
        print("(ArchCompiler.phaseAfter, RewriteStep(RewriteAll(),", file=f)
        print(f"  ParallelizeDotProductRules.all(Some({k}))", file=f)
        print(")),", file=f)

    print("    )", file=f)
    print("  }", file=f)

  def _emit_method_generate_ir(self, f, gm, arg_elt_types):
    print("  override def generateIR(): Expr = {", file=f)

    lets_needed = 0
    placeholder_id = 0
    for n in gm.graph.nodes:
      # assume every node that has many uses needs to be let-bound,
      # which is definitely the case for tensors (which are SeqType's)
      many_uses = has_many_uses(n)
      if many_uses:
        lets_needed += 1

      # furthermore, input (placeholder) and output nodes must be 2D in SHIR.
      if n.op == "placeholder":
        # since these are all tensors, we would have gotten type and shape
        # annotations on all of them.
        #
        # the actual SHIR type may be different from the tensors' metadata:
        # -  the signedness has to match
        # -  the width may be narrower than the annotation
        #
        # the users can assume input tensors already satisfy these properties
        # and are expected to do the corresponding extension if necessary.

        real_typ = arg_elt_types[placeholder_id] or types.get_element_type(n)
        shape = ", ".join((str(d) for d in n.meta.get("val").shape))

        if many_uses:
          print(
            "  { val _init = core.TypeChecker.check(",
            "algo.torch.Input(", real_typ.name(),
            ", \"arg", placeholder_id, "\", Seq(", shape, "))",
            ")\n",
            "    val _param = core.ParamDef(_init.t)\n",
            "    core.Let(_param,\n",
            "  { val ", a.name, " = core.ParamUse(_param)",
            sep="", file=f
          )
        else:
          print(
            "    val ", n.name, " = core.TypeChecker.check(",
            "algo.torch.Input(", real_typ.name(),
            ", \"arg", placeholder_id, "\", Seq(", shape, "))",
            ")",
            sep="", file=f
          )
        placeholder_id += 1

      elif n.op == "output":
        # sometimes, due to unfortunate graph slicing, we may end up with
        # multiple outputs, which we cannot handle
        [retv] = n.args
        assert isinstance(retv, Node), "Only single node output is allowed"
        assert not many_uses  # not sure what this failing would mean...

        annot_typ = types.get_element_type(retv)
        dims = len(retv.meta.get("val").shape)
        v = retv.name

        if dims < 1:
          v = f"algo.Repeat({v}, 1)"
        if dims < 2:
          v = f"algo.Repeat({v}, 1)"
        if dims > 2:
          v = f"algo.torch.Flatten({v}, 1, {dims - 1})"
        print(
          "    core.TypeChecker.check(algo.Map(2,",
          " algo.ResizeInteger.asFunction(types = Seq(", annot_typ.bits, ")),",
          " ", v, "))",
          sep="", file=f
        )

      elif n.op == "call_function":
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

      else:
        assert False, "Unhandled fx node type when emitting"

    for _ in range(lets_needed):
      print("  }, _init)", file=f)

    print("  }", file=f)

class SHIRGraphSimModule(torch.nn.Module):
  project: SHIRProject
  _result_entry: Optional[layout.LayoutEntry]
  _result_inner: int
  _result_shape: torch.Size

  def __init__(self, gm: GraphModule, project: SHIRProject):
    super().__init__()
    self.project = project
    self._result_entry = project.load_layout_file().get_entry("result")
    output_node = _collect_inout_nodes(gm)[1]
    self._result_shape = shape = output_node.meta.get("val").shape
    self._result_inner = layout.reshape_size_to_matrix(shape)[1]

  def __call__(self, *args) -> torch.Tensor:
    with tempfile.TemporaryDirectory() as data_dir:
      for i, arg in enumerate(args):
        with (
          Path(data_dir) / f"arg{i}.csv"
        ).open("w", encoding="utf-8") as f:
          layout.tensor_to_matrix_csv(arg, f)

      subprocess.run(
        ['sbt', f'run --no-gen --sim "{data_dir}"'],
        check=True, cwd=self.project.output_dir
      )
      return self.project.read_memory_dump(
        self._result_entry,
        self._result_inner
      ).reshape(self._result_shape)

_last_flashed_gbs = None
_last_opened_fpga = None

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

  def __init__(self, gm: GraphModule, project: SHIRProject, driver):
    super().__init__()
    self._driver = driver
    self._layout = project.load_layout_file()
    self._gbs_file = project.get_gbs_file()

    # allocate the buffer
    sz = self._layout.bytes_needed(round_to_page=True)
    meminfo = self._driver.alloc_buffer(sz)
    self._buffer = meminfo

    (in_nodes, out_node) = _collect_inout_nodes(gm)
    arg_mapping = {f"arg{i}": (i, arg) for i, arg in enumerate(in_nodes)}
    inputs = [None] * len(in_nodes)
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
        output = _reshape_region(region, out_node.meta.get("val").shape)
      elif region is not None:
        (id, node) = arg_mapping[entry.name]
        inputs[id] = _reshape_region(region, node.meta.get("val").shape)

    self._inputs = inputs
    self._output = output

  def __call__(self) -> torch.Tensor:
    # reconfigure the fpga if needed
    global _last_flashed_gbs
    global _last_opened_fpga
    if _last_flashed_gbs is None or not self._gbs_file.samefile(_last_flashed_gbs):
      # first, close the currently opened FPGA if applicable
      if _last_opened_fpga is not None:
        _last_opened_fpga.close()
        _last_opened_fpga = None

      # reconfigure
      subprocess.run(['fpgaconf', '-v', self._gbs_file])
      _last_flashed_gbs = self._gbs_file

      # then open a new FPGA
      _last_opened_fpga = self._driver.find_and_open_fpga(config.ACCEL_UUID)

    fpga = _last_opened_fpga
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
