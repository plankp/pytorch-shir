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

  def consult_cache(self):
    # XXX: Newer Python platforms have file_digest.
    # but the server uses an older one that doesn't...
    # so we use the one suggested here:
    # https://stackoverflow.com/a/44873382/11416644
    import hashlib
    with (self.output_dir / f"{self.clname}.scala").open("rb", buffering=0) as f:
      h = hashlib.sha256()
      b = bytearray(128*1024)
      mv = memoryview(b)
      while n := f.readinto(mv):
        h.update(mv[:n])
      digest = h.hexdigest()
      return Path(config.MODEL_CACHE_DIR) / digest

  def prepare_directory(self):
    if (self.output_dir / "project").exists():
      shutil.rmtree(self.output_dir / "project")
    shutil.copytree(config.TEMPLATE_DIR, self.output_dir / "project")
    shutil.copyfile(self.output_dir / f"{self.clname}.scala", self.output_dir / "project" / "src" / "main" / "scala" / f"{self.clname}.scala")

  def generate_hardware_files(self):
    subprocess.run(['sbt', 'run --gen --no-sim'], check=True, cwd=self.output_dir / "project")

  def get_source_file(self) -> Path:
    return self.output_dir / f"{self.clname}.scala"

  def get_layout_file(self) -> Path:
    return self.output_dir / "project" / "out" / self.clname / "memory.layout"

  def get_gbs_file(self) -> Path:
    return self.output_dir / "project" / "synthesis" / "build_synth" / "hello_afu_unsigned_ssl.gbs"

  def synthesize(self):
    synth_dir = self.output_dir / "project" / "synthesis"
    subprocess.run(
      ['./synthesize.sh', str(Path("..") / "out" / self.clname)],
      check=True, cwd=synth_dir
    )

  def emit_source(self, gm, host_mapping):
    with (self.output_dir / f"{self.clname}.scala").open("w", encoding="utf-8") as f:
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

      # determine the buffering strategy here.
      # it's needed because doing so might uncover extra host mappings
      # (due to buffering on host).
      buffer_strategy = self._determine_buffering(gm, host_mapping)

      print(file=f)
      self._emit_method_load_data(f, gm, host_mapping)

      print(file=f)
      self._emit_method_extra_rewrites(f, gm, buffer_strategy, host_mapping)

      print(file=f)
      self._emit_method_generate_ir(f, gm, host_mapping)

      print("}", file=f)

  def _determine_buffering(self, gm, host_mapping):
    buffer_id = 0
    buffer_strategy = {}
    for n in gm.graph.nodes:
      if n.op != "call_function":
        continue

      obj = lowering.fetch_lowering(n.target)
      try:
        if callable(obj.should_buffer):
          hint = obj.should_buffer(*n.args, **n.kwargs)
          for node, flag in hint.items():
            original_flag = buffer_strategy.get(node, None)
            if original_flag is not None and flag != original_flag:
              # make sure we continue to buffer matrix
              flag = True
            buffer_strategy[node] = flag

            # right now, assume that we really want it buffered on host.
            # so add a new entry if it wasn't there already
            if node not in host_mapping:
              host_mapping[node] = (f"buffer{buffer_id}_tag", types.get_element_type(node))
              buffer_id += 1
      except:
        pass
    return buffer_strategy

  def _emit_method_load_data(self, f, gm, host_mapping):
    print("  override def loadData(folder: String): Predef.Map[String, Seq[Seq[Int]]] = Predef.Map(", file=f)

    output_node = None
    for n in gm.graph.nodes:
      if n.op == "output":
        assert output_node is None, f"Multi-output node not supported"
        assert isinstance(n.args[0], Node), "Multi-valued output node not supported"
        output_node = n.args[0].meta.get("val")
        continue

      # it is either a placeholder or a intermediate node.
      # in either case, consult the host_mapping table.
      if n in host_mapping:
        host_id = host_mapping[n][0]

        # if it is an input, read the values from csv files.
        # otherwise, just create a bunch of 0's (like in the result case)
        if n.op == "placeholder":
          print(
            "    \"", host_id, "\" -> Util.readIntCSV(Paths.get(folder, \"",
            host_id, ".csv\").toFile()),",
            sep="", file=f
          )
        else:
          (outer, inner) = layout.reshape_size_to_matrix(n.meta.get("val").shape)
          print(
            "    \"", host_id, "\" -> new UniformSeq(new UniformSeq(0, ",
            inner, "), ", outer, "),",
            sep="", file=f
          )

    # for the result, we still need to give it some dummy value for simulation
    # to determine the RAM size.
    (outer, inner) = layout.reshape_size_to_matrix(output_node.shape)
    print(
      "    \"result\" -> new UniformSeq(new UniformSeq(0, ",
      inner, "), ", outer, ")",
      sep="", file=f
    )

    print("  )", file=f)

  def _emit_method_extra_rewrites(self, f, gm, buffer_strategy, host_mapping):
    print("  override def extraRewrites(): Seq[(core.compile.CompilerPhase, core.rewrite.RewriteStep)] = {", file=f)
    print("    import core.compile.CompilerPhase", file=f)
    print("    import core.rewrite.{RewriteAll, RewriteStep, RewriteTargeted}", file=f)
    print("    import backend.hdl.arch.{ArchCompiler, MapCompiler}", file=f)
    print("    import backend.hdl.arch.device.DeviceSpecificCompiler", file=f)
    print("    import backend.hdl.arch.rewrite.{InputBufferingRules, ParallelizeDotProductRules}", file=f)
    print("    import backend.hdl.arch.mem.MemFunctionsCompiler", file=f)
    print("    Seq(", file=f)

    # and we emit the rewrite rules based on the collected hints.
    #
    # XXX:
    # the buffering rewrites MUST be applied outside in. so for a graph
    # of { tmp = input1 * input2; result = tmp * input3 }, tmp must be
    # buffered before input1 and input2.
    #
    # (and the current implementation is obeys that by walking the
    # dependency graph backwards)
    for n in reversed(gm.graph.nodes):
      if n not in host_mapping or n not in buffer_strategy:
        continue

      host_id, real_typ = host_mapping[n]
      strat = buffer_strategy[n]
      if strat is None:
        continue

      # buffering is based on cachelines, so estimate it.
      # (the rewrite will crash later if there's a mismatch...)
      _, lines = layout.guess_line_layout(n.meta.get("val").shape, real_typ)
      print("(ArchCompiler.phaseAfter, RewriteStep(RewriteAll(), Seq(", end='', file=f)
      if strat:
        print("InputBufferingRules.bufferInputMatrix(\"", host_id, "\", ", lines, ")", sep='', end='', file=f)
      else:
        print("InputBufferingRules.bufferInputRow(\"", host_id, "\", ", lines, ")", sep='', end='', file=f)
      print("))),", file=f)

    # add other operation specific rewrites
    for n in gm.graph.nodes:
      if n.op != "call_function":
        continue

      obj = lowering.fetch_lowering(n.target)
      try:
        if callable(obj.should_rewrite):
          hint = obj.should_rewrite(*n.args, **n.kwargs)
          if hint is not None:
            print(hint, ",", sep="", file=f)
      except:
        pass

    # double buffer every input
    # the target is a descending sequence of integers...
    idx = ", ".join((str(x - 1) for x in range(len(host_mapping), 0, -1)))
    print(f"(MemFunctionsCompiler.phaseAfter, RewriteStep(RewriteTargeted({idx}), Seq(", file=f)
    print("  InputBufferingRules.doubleBufferRead", file=f)
    print("))),", file=f)

    print("    )", file=f)
    print("  }", file=f)

  def _emit_method_generate_ir(self, f, gm, host_mapping):
    print("  override def generateIR(): Expr = {", file=f)

    lets_needed = 0
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

        host_id, real_typ = host_mapping[n]
        shape = ", ".join((str(d) for d in n.meta.get("val").shape))

        if many_uses:
          print(
            "  { val _init = core.TypeChecker.check(",
            "algo.torch.Input(", real_typ.name(),
            ", \"", host_id, "\", Seq(", shape, "))",
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
            ", \"", host_id, "\", Seq(", shape, "))",
            ")",
            sep="", file=f
          )

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
        if n in host_mapping:
          # add buffering if necessary.
          expr = f"algo.BufferHost({expr}, core.TextType(\"{host_mapping[n][0]}\"))"

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

