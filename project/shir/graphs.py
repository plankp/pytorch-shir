"""
Where the SHIRGraphModule is defined
"""

import torch
from torch.fx import GraphModule, Node
from typing import Tuple, List, Optional
from . import lowering, types, layout, config
from functools import reduce
from itertools import count
import os
import glob
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

def _has_many_uses(node: Node) -> bool:
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
  gm: GraphModule
  _inst_id: int
  _call_id: int
  _compiled: bool
  _inout_nodes: Optional[Tuple[List[Node], Node]]
  _layout: Optional[layout.MemoryLayout]
  _driver: Optional[ctypes.CDLL]

  def __init__(self, gm: GraphModule):
    super().__init__()
    self._inst_id = next(_iid_counter)
    self.gm = gm
    self._compiled = False
    self._call_id = 0
    self._inout_nodes = None
    self._layout = None
    self._driver = None

    self._clname = f"Module{self._inst_id}"
    self._output_dir = os.path.join(config.EMIT_OUTPUT_DIR, f"module{self._inst_id}")

  def __call__(self, *args):
    self._call_id += 1
    self.compile()

    if config.PERFORM_SYNTHESIS:
      info = self._inout_nodes[1].meta.get("val")
      result = torch.empty(info.shape, dtype=info.dtype)
      r = self._driver.compute(
        config.ACCEL_UUID,
        ctypes.c_void_p(result.data_ptr()),
        *(ctypes.c_void_p(arg.contiguous().data_ptr()) for arg in args)
      )
      if r == 0:
        return result

      assert False, f"Driver returned non-zero exit code: {r}"

    elif config.PERFORM_SIMULATION:
      data_dir = os.path.join(self._output_dir, f"data_{self._call_id}")
      if not os.path.exists(data_dir):
        os.mkdir(data_dir)

      for i, arg in enumerate(args):
        with open(os.path.join(data_dir, f"arg{i}.csv"), "w", encoding="utf-8") as argf:
          layout.tensor_to_matrix_csv(arg, argf)

      # here, since the current working directory is already inside the output
      # directory, reference the particular data folder name directly instead of
      # using data_dir (since data_dir may contain relative paths, which is bad)
      subprocess.run(['sbt', f'run --no-gen --sim "data_{self._call_id}"'], check=True, cwd=self._output_dir)

      # assume it didn't crash, then we would have ended up with a memory dump,
      # from which, we can recover the result.
      memory_dump_file = os.path.join(self._output_dir, "out", self._clname, "memory.dump")
      info = self._inout_nodes[1].meta.get("val")
      return layout.read_memory_dump(
        memory_dump_file, self._layout.get_entry("result"),
        torch.empty(info.shape, dtype=info.dtype)
      )

    else:
      return self.gm(*args)

  def compile(self):
    if self._compiled:
      return

    self._compiled = True
    if (
      config.FORCE_GENERATE_FILES
      or config.PERFORM_SIMULATION or config.PERFORM_SYNTHESIS
    ):
      self._prepare_directory()
      self._emit_source()
      self._generate_hardware_files()

  def _prepare_directory(self):
    # purge the output directory
    if os.path.exists(self._output_dir):
      shutil.rmtree(self._output_dir)

    # copy the template directory into the output directory
    shutil.copytree(config.TEMPLATE_DIR, self._output_dir)

  def _generate_hardware_files(self):
    # the bare minimum of VHDL and layout file (and maybe others) generation
    subprocess.run(['sbt', 'run --gen --no-sim'], check=True, cwd=self._output_dir)
    files_dir = os.path.join(self._output_dir, "out", self._clname)
    memory_layout_file = os.path.join(files_dir, "memory.layout")
    self._layout = layout.read_layout_file(memory_layout_file)

    if not config.PERFORM_SYNTHESIS:
      return

    synth_dir = os.path.join(self._output_dir, "synthesis")
    with open(
      os.path.join(synth_dir, "driver.c"), "w", encoding="utf-8"
    ) as f:
      self._emit_fpga_driver(f)

    for vhdf in glob.glob(os.path.join(files_dir, "*.vhd")):
      shutil.move(vhdf, os.path.join(synth_dir, "hw", "rtl", "generated"))

    subprocess.run(['./synthesize.sh'], check=True, cwd=synth_dir)
    self._driver = ctypes.cdll.LoadLibrary(os.path.join(
      synth_dir, "build_driver", "libdriver.so"
    ))

  def _emit_fpga_driver(self, f):
    print(r'''/* This file is autogenerated */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <uuid/uuid.h>
#include <opae/fpga.h>
#include "helper.h"

#ifdef __cplusplus
extern "C"
#endif /* __cplusplus */
int compute(
  const char *restrict accel_uuid,
  char *restrict result,''', file=f)

    arg_mapping = {}
    for i, arg in enumerate(self._inout_nodes[0]):
      print("  const char *restrict arg", i, ", /* ", arg.target, " */", sep="", file=f)
      arg_mapping[arg.target] = (i, arg)

    print(r''') {
  fpga_handle handle;
  fpga_result r;
  if ((r = find_and_open_fpga(accel_uuid, &handle)))
    goto err_open_fpga;

  /* XXX: Always allocate 100MB for now. */
  char *buffer; uint64_t wsid;
  if ((r = fpgaPrepareBuffer(handle, 1024 * 1024 * 100, &buffer, &wsid, 0)))
    goto err_prepare_buffer;
  uint64_t io_addr;
  if ((r = fpgaGetIOAddress(handle, wsid, &io_addr)))
    goto err_start_routine;''', file=f)

    def emit_memcpy_helper(entry: layer.LayoutEntry, node: Node, emitted_name: str, node_is_source: bool):
      shape = node.meta.get("val").shape
      if len(shape) == 1:
        inner = shape[0]
      else:
        inner = reduce(lambda x, y: x * y, shape[1:], 1)
      length = inner * types.get_element_type(node).bits // 8

      # it's just memcpy's at this stage.
      # the tricky part is that whereas the nodes are contiguous,
      # each row is allocated on a fresh cacheline
      cacheline = entry.address
      offset = 0
      for row in entry.outer:
        if node_is_source:
          print("  memcpy(&buffer[CL(", cacheline, ")], &", emitted_name, "[", offset, "], ", length, ");", sep="", file=f)
        else:
          print("  memcpy(&", emitted_name, "[", offset, "], &buffer[CL(", cacheline, ")], ", length, ")", sep="", file=f)
        cacheline += entry.inner
        offset += length

    result_entry = None
    for entry in self._layout._entries:
      if entry.name == "result":
        result_entry = entry
        continue

      # compute the number of bytes on each "row" of data
      (arg_id, arg_node) = arg_mapping[entry.name]
      print(file=f)
      emit_memcpy_helper(entry, arg_node, f"arg{arg_id}", True)

    print(r'''
  if ((r = fpgaWriteMMIO64(handle, 0, 0x00, io_addr / CL(1))))
    goto err_start_routine;
  if ((r = fpgaWriteMMIO64(handle, 0, 0x08, 1)))
    goto err_start_routine;

  for (;;) {
#ifndef NDEBUG
    usleep(1000000);
    uint64_t write_req;
    if (fpgaReadMMIO64(handle, 0, 0xC8, &write_req))
      fprintf(stderr, "Could not fetch writes requested\n");
    else
      printf("%" PRu64 " writes requested\n", write_req);
#else
    usleep(100);
#endif /* !NDEBUG */

    uint64_t done = 0;
    fpgaReadMMIO64(handle, 0, 0x80, &done);
    if (done == 1)
      break;
  }
''', file=f)

    emit_memcpy_helper(result_entry, self._inout_nodes[1], "result", False)

    print(r'''
#ifndef NDEBUG
  uint64_t cycles, readreq, readpending, writereq, writepending, readaf, writeaf;
  if ((r = fpgaReadMMIO64(handle, 0, 0x88, &cycles)))
    goto err_start_routine;
  if ((r = fpgaReadMMIO64(handle, 0, 0xC0, &readreq)))
    goto err_start_routine;
  if ((r = fpgaReadMMIO64(handle, 0, 0xD0, &readpending)))
    goto err_start_routine;
  if ((r = fpgaReadMMIO64(handle, 0, 0xE0, &readaf)))
    goto err_start_routine;
  if ((r = fpgaReadMMIO64(handle, 0, 0xC8, &writereq)))
    goto err_start_routine;
  if ((r = fpgaReadMMIO64(handle, 0, 0xD8, &writepending)))
    goto err_start_routine;
  if ((r = fpgaReadMMIO64(handle, 0, 0xE8, &writeaf)))
    goto err_start_routine;

  printf(
    "Execution time (cycles): %" PRu64 "\n"
    "Read requests          : %" PRu64 " (of which %" PRu64 "pending)\n"
    "Write requests         : %" PRu64 " (of which %" PRu64 "pending)\n"
    "Read request buffer  %" PRu64 "times almost full\n"
    "Write request buffer %" PRu64 "times almost full\n",
    cycles, readreq, readpending, writereq, writepending, readaf, writeaf
  );
#endif /* !NDEBUG */

err_start_routine:
  fpgaReleaseBuffer(handle, wsid);
err_prepare_buffer:
  close_fpga(handle);
err_open_fpga:
  return r;
}''', file=f)

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
