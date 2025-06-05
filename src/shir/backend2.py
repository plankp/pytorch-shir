import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._subclasses.fake_tensor import FakeTensorMode
from typing import List, Callable
from functools import reduce
from . import config, rewrites, bit_utils, types
import weakref

from torch.library import Library, impl
shir_LeNet5_lib = Library("_shir", "DEF")
shir_LeNet5_lib.define("lenet5_linear3(Tensor images, Tensor weights, Tensor bias, Tensor scale, int z) -> Tensor")
shir_LeNet5_lib.define("lenet5_linear2(Tensor images, Tensor weights, Tensor bias, Tensor scale, int z) -> Tensor")
shir_LeNet5_lib.define("lenet5_linear1(Tensor images, Tensor weights, Tensor bias, Tensor scale, int z) -> Tensor")
shir_LeNet5_lib.define("lenet5_conv_pool2(Tensor images, Tensor kernel, Tensor bias, Tensor scale, int z) -> Tensor")
shir_LeNet5_lib.define("lenet5_conv_pool1(Tensor images, Tensor kernel, Tensor bias, Tensor scale, int z) -> Tensor")

@impl(shir_LeNet5_lib, "lenet5_linear3", "Meta")
def lenet5_linear3_meta(images, weights, bias, scale, zp):
  return torch.empty(images.shape[0], 10, dtype=torch.int8, device='meta')

@impl(shir_LeNet5_lib, "lenet5_linear2", "Meta")
def lenet5_linear2_meta(images, weights, bias, scale, zp):
  return torch.empty(images.shape[0], 90, dtype=torch.int8, device='meta')

@impl(shir_LeNet5_lib, "lenet5_linear1", "Meta")
def lenet5_linear1_meta(images, weights, bias, scale, zp):
  return torch.empty(images.shape[0], 120, dtype=torch.int8, device='meta')

@impl(shir_LeNet5_lib, "lenet5_conv_pool2", "Meta")
def lenet5_conv_pool1_meta(images, kernel, bias, scales, zp):
  return torch.empty(images.shape[0], 5, 5 * 16, dtype=torch.int8, device='meta')

@impl(shir_LeNet5_lib, "lenet5_conv_pool1", "Meta")
def lenet5_conv_pool1_meta(images, kernel, bias, scales, zp):
  return torch.empty(images.shape[0], 14, 14 * 6, dtype=torch.int8, device='meta')

def extract_qaddmm_relu(n: fx.Node):
  requant = None
  relu = None

  if (n.op != "call_function" or n.target != torch.ops.shir_intrinsic.requantize or
      len(n.args[0].users) != 1):
    return None

  requant, n = n, n.args[0]
  if (n.op == "call_function" and n.target == torch.ops.aten.relu and
      len(n.args[0].users) == 1):
    relu, n = n, n.args[0]

  if n.op != "call_function" or n.target != torch.ops.shir_intrinsic.int_addmm:
    return None

  return (requant, relu, n)

def extract_qconv_relu(n: fx.Node):
  requant = None
  relu = None
  if (n.op != "call_function" or n.target != torch.ops.shir_intrinsic.requantize_channel or
      len(n.args[0].users) != 1):
    return None

  requant, n = n, n.args[0]
  if (n.op == "call_function" and n.target == torch.ops.aten.relu and
      len(n.args[0].users) == 1):
    relu, n = n, n.args[0]

  if n.op != "call_function" or n.target != torch.ops.shir_intrinsic.qconv:
    return None

  return (requant, relu, n)

def _adjust_requant_param(scales, zp):
  # XXX: hardcode zp in [-128, 128) and 28 bit scales, shamt ≤ 35
  if zp < -128 or zp > 127:
    return None

  q, w, shamt = bit_utils.qscale_to_fixpoint(scales)

  if shamt > 35:
    print(f"backend::_adjust_requant_param: shamt is larger, trying to continue by truncating")
    q = [x >> (shamt - 35) for x in q]
    w = max(1, w - 35)
    shamt = 35

  lsl = 35 - shamt
  if w + lsl > 28:
    return None

  return [x << lsl for x in q]

def isel(gm: fx.GraphModule):
  # since we destructively rewrite the graph,
  # try to keep it so that the types of names do not change.
  # (at least for the runtime values)

  counter = 0

  def create_new_param():
    from torch._dynamo.source import NNModuleSource, LocalSource, AttrSource
    nonlocal counter, gm

    counter += 1
    name = f"_isel_param{counter}"
    assert not hasattr(gm, name)
    assert name not in gm._param_name_to_source

    gm.register_parameter(name, None)
    gm._param_name_to_source[name] = NNModuleSource(AttrSource(LocalSource("self"), name))
    return name

  graph = gm.graph
  for n in graph.nodes:
    if ((qrm := extract_qaddmm_relu(n)) is not None and
        qrm[2].args[0].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrm[2].args[0].target)).to_signed().bits <= 20):
      requant, relu, addmm = qrm
      bias = addmm.args[0]
      images = addmm.args[1]
      weight = addmm.args[2]
      adjusted = _adjust_requant_param([requant.args[1]], requant.args[2])
      if (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          weight.op == "get_attr" and
          getattr(gm, weight.target).shape[0] <= 10 and
          getattr(gm, weight.target).shape[1] <= 128):
        m = getattr(gm, weight.target)
        w = m.shape[0]
        m = torch.nn.functional.pad(m, (0, 128 - m.shape[1], 0, 10 - m.shape[0]))
        setattr(gm, weight.target, torch.nn.Parameter(m, False))

        m = getattr(gm, bias.target)
        m = torch.nn.functional.pad(m, (0, 10 - m.shape[0]))
        setattr(gm, bias.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted[0], dtype=torch.int32), False))

        with graph.inserting_before(n):
          n1 = graph.get_attr(sclattr)
          n2 = graph.call_function(torch.ops._shir.lenet5_linear3, (images, weight, bias, n1, requant.args[2]))
        n.target = torch.ops.aten.pad
        n.args = (n2, [0, w - 10])
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(addmm)

      elif (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          weight.op == "get_attr" and
          getattr(gm, weight.target).shape[0] <= 90 and
          getattr(gm, weight.target).shape[1] <= 128):
        m = getattr(gm, weight.target)
        w = m.shape[0]
        m = torch.nn.functional.pad(m, (0, 128 - m.shape[1], 0, 90 - m.shape[0]))
        setattr(gm, weight.target, torch.nn.Parameter(m, False))

        m = getattr(gm, bias.target)
        m = torch.nn.functional.pad(m, (0, 90 - m.shape[0]))
        setattr(gm, bias.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted[0], dtype=torch.int32), False))

        with graph.inserting_before(n):
          n1 = graph.get_attr(sclattr)
          n2 = graph.call_function(torch.ops._shir.lenet5_linear2, (images, weight, bias, n1, requant.args[2]))
        n.target = torch.ops.aten.pad
        n.args = (n2, [0, w - 90]) # negative padding / removes the padded entries
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(addmm)

      elif (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          weight.op == "get_attr" and
          getattr(gm, weight.target).shape[0] == 120 and
          getattr(gm, weight.target).shape[1] == 400):
        m = getattr(gm, weight.target).reshape([120, 5, 80])
        m = torch.nn.functional.pad(m, (0, 128 - m.shape[2]))
        setattr(gm, weight.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted[0], dtype=torch.int32), False))

        batch = images.meta.get("val").shape[0]
        with graph.inserting_before(n):
          n1 = graph.get_attr(sclattr)
          n2 = graph.call_function(torch.ops.aten.view, (images, [batch, 5, 80]))
        n.target = torch.ops._shir.lenet5_linear1
        n.args = (n2, weight, bias, n1, requant.args[2])
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(addmm)

    elif (n.op == "call_function" and n.target == torch.ops.shir_intrinsic.int_avg_pool2d and
        len(n.args[0].users) == 1 and
        (qrc := extract_qconv_relu(n.args[0])) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 20):
      requant, relu, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])
      if (adjusted is not None and
          zp == -128 and requant.args[2] == -128 and
          stride == [1, 1] and padding == [2, 2] and dilation == [1, 1] and groups == 1 and
          list(images.meta.get("val").shape[1:]) == [1, 28, 28] and
          kernel.op == "get_attr" and
          list(getattr(gm, kernel.target).shape) == [6, 1, 5, 5]):
        m = getattr(gm, kernel.target)
        m = m.permute([0, 2, 3, 1]).reshape([6, 5 * 5 * 1])
        setattr(gm, kernel.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        batch = images.meta.get("val").shape[0]
        with graph.inserting_before(n):
          i1 = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          i2 = graph.call_function(torch.ops.aten.view, (i1, [batch, 28, 28 * 1]))
          n1 = graph.get_attr(sclattr)
          u = graph.call_function(torch.ops._shir.lenet5_conv_pool1, (i2, kernel, bias, n1, requant.args[2]))
          r = graph.call_function(torch.ops.aten.view, (u, [batch, 14, 14, 6]))
          r = graph.call_function(torch.ops.aten.permute, (r, [0, 3, 1, 2]))
        n.target = torch.ops.aten.reshape
        n.args = (r, [batch, 6, 14, 14])
        graph.erase_node(requant)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      elif (adjusted is not None and
          zp == -128 and requant.args[2] == -128 and
          stride == [1, 1] and padding == [0, 0] and dilation == [1, 1] and groups == 1 and
          list(images.meta.get("val").shape[1:]) == [6, 14, 14] and
          kernel.op == "get_attr" and
          list(getattr(gm, kernel.target).shape) == [16, 6, 5, 5]):
        m = getattr(gm, kernel.target)
        m = m.permute([0, 2, 3, 1]).reshape([16, 5 * 5 * 6])
        setattr(gm, kernel.target, torch.nn.Parameter(m, False))

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        batch = images.meta.get("val").shape[0]
        with graph.inserting_before(n):
          i1 = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          i2 = graph.call_function(torch.ops.aten.reshape, (i1, [batch, 14, 14 * 6]))
          n1 = graph.get_attr(sclattr)
          u = graph.call_function(torch.ops._shir.lenet5_conv_pool2, (i2, kernel, bias, n1, requant.args[2]))
          # this nasty sequence avoids the permute from messing up the
          # "contiguity" of the tensor
          r = graph.call_function(torch.ops.aten.view, (u, [batch, 5, 5, 16]))
          r = graph.call_function(torch.ops.aten.permute, (r, [0, 3, 1, 2]))
          r = graph.call_function(torch.ops.aten.reshape, (r, [batch, 16 * 5 * 5]))
        n.target = torch.ops.aten.view
        n.args = (r, [batch, 16, 5, 5])
        graph.erase_node(requant)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

  graph.lint()
  gm.recompile()

def permute_has_equiv_view(shape: torch.Size, perm: List[int]):
  filtered = [i for (i, x) in zip(perm, list(shape)) if x != 1]
  return all((i < j for (i, j) in zip(filtered, filtered[1:])))

def simpl(gm: fx.GraphModule):
  # since we destructively rewrite the graph,
  # try to keep it so that the types of names do not change.
  graph = gm.graph
  changed = True
  while changed:
    changed = False
    for n in graph.nodes:
      if n.op != "call_function":
        continue
      if (n.target in {torch.ops.aten.view, torch.ops.aten.reshape} and
          n.args[0].op == "call_function" and
          n.args[0].target in {torch.ops.aten.view, torch.ops.aten.reshape}):
        flatten = n.args[0]
        if flatten.target != torch.ops.aten.view:
          n.target = torch.ops.aten.reshape
        n.args = (flatten.args[0],) + n.args[1:]
        if not flatten.users:
          graph.erase_node(flatten)
        changed = True

      elif (n.target in {torch.ops.aten.view, torch.ops.aten.reshape} and
          list(n.args[0].meta.get("val").shape) == list(n.meta.get("val").shape)):
        n.replace_all_uses_with(n.args[0])
        graph.erase_node(n)
        changed = True
  
      elif (n.target == torch.ops.aten.permute and
          all((i == j for (i, j) in enumerate(n.args[1])))):
        n.replace_all_uses_with(n.args[0])
        graph.erase_node(n)
        changed = True

      elif (n.target == torch.ops.aten.permute and
          # this really needs to be the result shape, not the input shape
          permute_has_equiv_view(n.meta.get("val").shape, n.args[1])):
        n.target = torch.ops.aten.view
        n.args = (n.args[0], n.meta.get("val").shape)
        changed = True

      elif (n.target == torch.ops.aten.permute and
          n.args[0].op == "call_function" and
          n.args[0].target == torch.ops.aten.permute):
        perm = n.args[0]
        n.args = (perm.args[0], [perm.args[1][i] for i in n.args[1]])
        if not perm.users:
          graph.erase_node(perm)
        changed = True

      elif (n.target == torch.ops.aten.pad and
          all((x == 0 for x in n.args[1]))):
        # also handles the empty pad / no-op case
        n.replace_all_uses_with(n.args[0])
        graph.erase_node(n)
        changed = True

      elif (n.target == torch.ops._shir.lenet5_linear3 and
          n.args[0].op == "call_function" and
          n.args[0].target == torch.ops.aten.pad and
          n.args[0].args[1][0] == 0 and n.args[0].args[1][1] < 0 and
          n.args[1].op == "get_attr"):
        # try to get rid of the negative pad
        npad = n.args[0]
        u = npad.meta.get("val").shape[-1]
        v = u - npad.args[1][1]
        m = getattr(gm, n.args[1].target)
        if torch.all(m[:, u:v] == 0):
          # if we're going to multiply by 0, then no need to negative pad.
          n.args = (npad.args[0],) + n.args[1:]
          if not npad.users:
            graph.erase_node(npad)
          changed = True

  graph.lint()
  gm.recompile()

def compute_layout(gm: fx.GraphModule):
  max_data = 0
  max_inst = 0
  layout = {}

  def lookup(n: fx.Node, shirTy):
    nonlocal max_data, layout, gm

    try:
      meminfo = layout[n]
      assert meminfo[2] == shirTy

    except KeyError:
      tensor = n.meta.get("val")
      inner = 1
      outer = 1
      if tensor.shape:
        inner = tensor.shape[-1]
        outer = reduce(lambda x, y: x * y, tensor.shape[:-1], 1)

      eltTy = types.get_scalar_type(tensor.dtype)
      if eltTy != shirTy:
        assert n.op == "get_attr"
        realTy = bit_utils.get_narrow_type(getattr(gm, n.target))
        assert shirTy.minval() <= realTy.minval() and realTy.maxval() <= shirTy.maxval()

      maxpk = config.CACHELINE_BITS // shirTy.bits
      lines = outer * ((inner + (maxpk - 1)) // maxpk)

      meminfo = (max_data, lines, shirTy)
      max_data += lines
      layout[n] = meminfo

    return meminfo

  for n in gm.graph.nodes:
    if n.op == "call_function":
      if n.target in {torch.ops._shir.lenet5_conv_pool1,
                      torch.ops._shir.lenet5_conv_pool2,
                      torch.ops._shir.lenet5_linear1,
                      torch.ops._shir.lenet5_linear2,
                      torch.ops._shir.lenet5_linear3}:
        images, weights, bias, scl, zp = n.args
        lookup(images, types.SI(8))
        lookup(weights, types.SI(8))
        lookup(bias, types.SI(20))
        lookup(scl, types.SI(28))
        lookup(n, types.SI(8))        # also need to allocate the result!
        batch = n.meta.get("val").shape[0]
        assert batch > 0 and batch % 8 == 0, "backend::isel: batch size must be multiple of 8"
        max_inst += (batch + (8 * 0xF - 1)) // (8 * 0xF)

  return max_inst, layout

def copy_to_buffer(src: torch.Tensor, dst, offs, sz, ty):
  bytes_per_cl = config.CACHELINE_BITS // 8

  # normalize the tensor to 2D
  mm = src.reshape((-1, src.shape[-1]) if src.shape else (1, 1))
  eltTy = types.get_scalar_type(mm.dtype)

  # a quick sanity check to avoid OOB writes.
  maxpk = config.CACHELINE_BITS // ty.bits
  outer = range(0, mm.shape[0])
  inner = range(0, mm.shape[1], maxpk)
  lines = len(outer) * len(inner)
  assert lines <= sz, "backend::copy_to_buffer: copy is larger than reserved size"

  if eltTy == ty:
    # data matches up, so just use torch's copy mechanism
    # (which is likely faster due to less conversion business)
    backed = torch.frombuffer(
        dst,
        dtype=torch.int8,
        offset=offs * bytes_per_cl,
        count=lines * bytes_per_cl,
    ).view(mm.dtype)

    # even if we were to flatten the source tensor, right now, the tensor
    # backed by the buffer might be larger due to trailing cacheline elements
    # (e.g., 64 bytes on a cacheline, but only 25 are meaningful)
    backed = torch.as_strided(backed, mm.shape, (maxpk * len(inner), 1))

    # now that we've dealt with the trailing element, we can use torch's copy
    backed.copy_(mm)

  else:
    # in this case, assume the types are weird (20-bit integers),
    # so do a manual copy

    # XXX: assumes a single tensor element never straddles across cachelines.
    mask = (1 << ty.bits) - 1
    offs *= bytes_per_cl
    for i in outer:
      for j in inner:
        line_data = 0
        shamt = 0
        for k in range(j, min(j + maxpk, mm.shape[1])):
          line_data |= (ty.cast(mm[i, k].item()) & mask) << shamt
          shamt += ty.bits
        dst[offs:offs + bytes_per_cl] = line_data.to_bytes(bytes_per_cl, byteorder="little")
        offs += bytes_per_cl

def copy_from_buffer(dst: torch.Tensor, src, offs, sz, ty) -> torch.Tensor:
  bytes_per_cl = config.CACHELINE_BITS // 8

  # normalize the tensor to 2D
  mm = dst.reshape((-1, dst.shape[-1]) if dst.shape else (1, 1))
  eltTy = types.get_scalar_type(mm.dtype)

  maxpk = config.CACHELINE_BITS // ty.bits
  outer = mm.shape[0]
  inner = (mm.shape[1] + (maxpk - 1)) // maxpk
  lines = outer * inner
  assert lines == sz, "backend::copy_from_buffer: size mismatch"
  assert eltTy == ty, "backend::copy_from_buffer: type mismatch"

  backed = torch.frombuffer(
      src,
      dtype=torch.int8,
      offset=offs * bytes_per_cl,
      count=lines * bytes_per_cl,
  ).view(mm.dtype)

  mm.copy_(torch.as_strided(backed, mm.shape, (maxpk * inner, 1)))

  # if all goes well, dst, mm, and this result share the same buffer.
  # generally this is the case since there are no weird strides.
  return mm.reshape(dst.shape)

def _wrap_buffer(shape: torch.Size, dty, src, offs, sz) -> torch.Tensor:
  bytes_per_cl = config.CACHELINE_BITS // 8
  ty = types.get_scalar_type(dty)
  inner = 1
  outer = 1
  if shape:
    inner = shape[-1]
    outer = reduce(lambda x, y: x * y, shape[:-1], 1)

  maxpk = config.CACHELINE_BITS // ty.bits
  innercl = (inner + (maxpk - 1)) // maxpk
  lines = outer * innercl
  assert lines == sz, "backend::_wrap_buffer: size mismatch"

  backed = torch.frombuffer(
      src,
      dtype=torch.int8,
      offset=offs * bytes_per_cl,
      count=lines * bytes_per_cl,
  ).view(dty)

  return torch.as_strided(backed, (outer, inner), (maxpk * innercl, 1)).reshape(shape)

def _copy_to_buffer(src: torch.Tensor, dst, offs, sz):
  return copy_to_buffer(src, dst, offs, sz, types.get_scalar_type(src.dtype))

def _copy_from_buffer(dst: torch.Tensor, src, offs, sz):
  return copy_from_buffer(dst, src, offs, sz, types.get_scalar_type(dst.dtype))

def emit(gm: fx.GraphModule, max_inst: int, data_layout):
  from . import driver
  import mmap           # for the page size constant

  bytes_per_cl = config.CACHELINE_BITS // 8
  total_sz = max_inst + max((u + v for (u, v, _) in data_layout.values()))
  total_sz = bytes_per_cl * total_sz
  total_sz = (total_sz + (mmap.PAGESIZE - 1)) // mmap.PAGESIZE * mmap.PAGESIZE

  pptr = driver.alloc_buffer(total_sz)
  assert pptr is not None, "backend::emit: Unable to allocate shared buffer"

  # the bootstrapping instruction must be at location 0.
  # afterwards, we put the instruction followed by the data.
  BASEADDR_INST = 1
  BASEADDR_DATA = BASEADDR_INST + max_inst

  # to track where the instruction should be placed
  iptr = BASEADDR_INST

  # when constructing the new fx graph, we need the track a few things:
  # - a mapping from nodes of the old graph to nodes of the new graph
  # - if the result of an instruction exists (we lazily invoke the FPGA)
  #
  # env is the mapping, pending_value holds the pending values, and
  # pending_inst holds the cacheline address to the first instruction to be
  # executed onto the FPGA.
  new_graph = fx.Graph()
  env = {}
  pending_inst = None
  pending_values = set()

  # dummy argument because passing pptr directly causes issues when compiling
  # the graph nodes (which is needed to execute it later)
  #
  # we _could_ map it using an attribute, but then that means the pptr's
  # lifetime is tied to the GraphModule that is eventually created. the issue
  # is that GraphModules have pretty nasty lifetimes (reference cycles?)...
  _pptr = new_graph.placeholder("_pptr")

  def invoke_and_wait(pptr, offs: int, sz: int, gbs: str):
    fpga = driver.configure_gbs(gbs)
    while sz > 0:
      # The alternative way is to issue a hard reset after configuring, then
      # only use fpga.soft_reset() between bootstrapped restarts.
      #
      # unfortunately, using that method requires hardware side changes at the
      # risk of dropping useful debug counters.
      fpga.reset()
      with fpga.prepare_buffer(pptr, len(pptr)) as wsid:
        # generate the bootstrap instruction
        bundle = min(sz, 0xFF)
        bootstrap = (offs & 0xFFFFFF) << 8 | (bundle & 0xFF)
        pptr[:bytes_per_cl] = bootstrap.to_bytes(bytes_per_cl, byteorder="little")

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
              "Execution time (cycles): ", cycles, " (", bundle, " / ", sz, " instruction(s))\n"
              "Read requests          : ", readreq, " (of which ", readpending, " pending)\n"
              "Write requests         : ", writereq, " (of which ", writepending, " pending)\n"
              "Read request buffer  ", readaf, " times almost full\n"
              "Write request buffer ", writeaf, " times almost full",
              sep="",
          )

        offs += bundle
        sz -= bundle

  def flush_pending_inst():
    nonlocal pending_inst, pending_values, iptr
    if pending_inst is None:
      return

    # start computation on the FPGA
    inst_ptr = pending_inst
    inst_len = iptr - pending_inst
    last_ptr = inst_ptr + range(0, inst_len, 0xFF)[-1]
    assert (inst_ptr & 0xFFFFFF) == inst_ptr, "backend::emit: bootstrapping instruction pointer too wide"
    assert (last_ptr & 0xFFFFFF) == last_ptr, "backend::emit: bootstrapping instruction pointer too wide"
    pending_inst = None
    pending_values = set()
    new_graph.call_function(invoke_and_wait, (_pptr, inst_ptr, inst_len, config.PRESYNTH_GBS))

  def mapper(n):
    nonlocal data_layout, env
    if n not in data_layout:
      return env[n]

    if n in pending_values:
      flush_pending_inst()

    tensor = n.meta.get("val")
    offs, sz, ty = data_layout[n]
    n1 = new_graph.call_function(torch.empty, (tensor.shape,), {"dtype": tensor.dtype})
    n2 = new_graph.call_function(_copy_from_buffer, (n1, _pptr, BASEADDR_DATA + offs, sz))
    return n2

  try:
    for n in gm.graph.nodes:
      if n in data_layout:
        offs, sz, ty = data_layout[n]
        if n.op == "get_attr":
          copy_to_buffer(getattr(gm, n.target), pptr, BASEADDR_DATA + offs, sz, ty)

        elif n.op == "call_function" and n.target in {
            torch.ops._shir.lenet5_conv_pool1,
            torch.ops._shir.lenet5_conv_pool2,
            torch.ops._shir.lenet5_linear1,
            torch.ops._shir.lenet5_linear2,
            torch.ops._shir.lenet5_linear3}:
          images, kernel, bias, scl, zp = n.args
          img_ptr = BASEADDR_DATA + data_layout[images][0]
          krn_ptr = BASEADDR_DATA + data_layout[kernel][0]
          bis_ptr = BASEADDR_DATA + data_layout[bias][0]
          scl_ptr = BASEADDR_DATA + data_layout[scl][0]
          res_ptr = BASEADDR_DATA + data_layout[n][0]
          batch = n.meta.get("val").shape[0]
          # asserting the img_ptr and res_ptr happens later
          assert (krn_ptr & 0xFFFFFF) == krn_ptr, "backend::emit: pointer too wide for LeNet5 encoding"
          assert (bis_ptr & 0xFFFFFF) == bis_ptr, "backend::emit: pointer too wide for LeNet5 encoding"
          assert (scl_ptr & 0xFFFFFF) == scl_ptr, "backend::emit: pointer too wide for LeNet5 encoding"
          assert -128 <= zp < 128, "backend::emit: signed zero point too wide for LeNet5 encoding"
          assert batch % 8 == 0, "backend::emit: batch not divisible by 8 for LeNet5 encoding"

          # uses one-cold encoding
          selbits = {
              torch.ops._shir.lenet5_conv_pool1: 0b10111,
              torch.ops._shir.lenet5_conv_pool2: 0b01111,
              torch.ops._shir.lenet5_linear1:    0b11110,
              torch.ops._shir.lenet5_linear2:    0b11101,
              torch.ops._shir.lenet5_linear3:    0b11011,
          }

          pending_inst = iptr if pending_inst is None else pending_inst
          pending_values.add(n)

          batch8 = batch // 8
          imgs_per_bundle = 0xF * (data_layout[images][1] // batch8)
          ress_per_bundle = 0xF * (data_layout[n][1] // batch8)
          while batch8 > 0:
            assert (img_ptr & 0xFFFFFF) == img_ptr, "backend::emit: pointer too wide for LeNet5 encoding"
            assert (res_ptr & 0xFFFFFF) == res_ptr, "backend::emit: pointer too wide for LeNet5 encoding"
            bundle = min(batch8, 0xF)
            inst = ( bundle << 136
                   | (zp & 0xFF) << 128
                   | scl_ptr << 104
                   | bis_ptr << 80
                   | krn_ptr << 56
                   | img_ptr << 32
                   | selbits[n.target] << 24
                   | res_ptr)

            pptr[iptr * bytes_per_cl:(iptr + 1) * bytes_per_cl] = inst.to_bytes(bytes_per_cl, byteorder="little")
            iptr += 1

            img_ptr += imgs_per_bundle
            res_ptr += ress_per_bundle
            batch8 -= bundle

        else:
          m = new_graph.node_copy(n, mapper)
          new_graph.call_function(_copy_to_buffer, (m, _pptr, BASEADDR_DATA + offs, sz))

      else:
        u = new_graph.node_copy(n, mapper)
        env[n] = u

    new_graph.lint()
    return pptr, new_graph
  except:
    # something goes wrong, release the pointer and rethrow
    driver.free_buffer(pptr)
    raise

def peephole(g: fx.Graph):
  for n in g.nodes:
    if n.op != "call_function":
      continue
    if (n.target == _copy_from_buffer and
        n.args[0].op == "call_function" and
        n.args[0].target == torch.empty and
        len(n.args[0].users) == 1 and
        len(n.args[0].args) == 1 and "dtype" in n.args[0].kwargs and
        len(n.users) == 1 and n.next in n.users and
        n.next.op == "call_function" and
        isinstance(n.next.target, torch._ops.OpOverload) and
        not n.next.target._schema.is_mutable):
      # we have the following instruction sequence:
      #   n1 = torch.empty(shape, dtype=ty)
      #   n2 = _copy_from_buffer(n1, _pptr, offs, sz)  <-- n points here
      #   n3 = f(n2, ...)     where f is "not mutable"
      empt, _pptr, offs, sz = n.args
      shape = empt.args[0]
      ty = empt.kwargs["dtype"]
      with g.inserting_before(n):
        n1 = g.call_function(_wrap_buffer, (shape, ty, _pptr, offs, sz))
      n.replace_all_uses_with(n1)
      g.erase_node(n)
      g.erase_node(empt)

    elif (n.target == torch.ops.quantized_decomposed.quantize_per_tensor.default and
        n.args[3] == -128 and n.args[4] == 127 and n.args[5] == torch.int8):
      with g.inserting_before(n):
        n1 = g.call_function(torch.quantize_per_tensor, n.args[:3], {"dtype": torch.qint8})
        n2 = g.call_method("int_repr", (n1,))
      n.replace_all_uses_with(n2)
      g.erase_node(n)

    elif (n.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default and
        n.args[3] == -128 and n.args[4] == 127 and n.args[5] == torch.int8):
      with g.inserting_before(n):
        n1 = g.call_function(torch._make_per_tensor_quantized_tensor, n.args[:3])
        n2 = g.call_method("dequantize", (n1,))
      n.replace_all_uses_with(n2)
      g.erase_node(n)

  g.lint()

class _Wrapper:
  def __init__(self, gm, drv, pptr):
    self._gm = gm
    self._pptr = pptr
    self._cleanup = weakref.finalize(self, drv.free_buffer, self._pptr)

  def dealloc(self):
    self._cleanup()

  def __call__(self, *args, **kwargs):
    assert self._cleanup.alive, "backend::_Wrapper: content already deallocated"
    return self._gm(self._pptr, *args, **kwargs)

def compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
  mode = FakeTensorMode(allow_non_fake_inputs=True)
  FakeTensorProp(gm, mode).propagate(*example_inputs)
  rewrites.rewrite_quantized_ops(gm)

  FakeTensorProp(gm, mode).propagate(*example_inputs)
  isel(gm)

  FakeTensorProp(gm, mode).propagate(*example_inputs)
  simpl(gm)

  max_inst, data_layout = compute_layout(gm)
  pptr, graph = emit(gm, max_inst, data_layout)
  peephole(graph)

  from . import driver
  return _Wrapper(fx.GraphModule(gm, graph), driver, pptr)

