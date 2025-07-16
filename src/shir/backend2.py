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
shir_fpga_inst_lib = Library("_shir", "DEF")
shir_fpga_inst_lib.define("lenet5_linear3(Tensor images, Tensor weights, Tensor bias, Tensor scale, int z) -> Tensor")
shir_fpga_inst_lib.define("lenet5_linear2(Tensor images, Tensor weights, Tensor bias, Tensor scale, int z) -> Tensor")
shir_fpga_inst_lib.define("lenet5_linear1(Tensor images, Tensor weights, Tensor bias, Tensor scale, int z) -> Tensor")
shir_fpga_inst_lib.define("lenet5_conv_pool2(Tensor images, Tensor kernel, Tensor bias, Tensor scale, int z) -> Tensor")
shir_fpga_inst_lib.define("lenet5_conv_pool1(Tensor images, Tensor kernel, Tensor bias, Tensor scale, int z) -> Tensor")
shir_fpga_inst_lib.define("conv3x3p1b8x64(Tensor images, int padvalue, Tensor kernel, Tensor bias, Tensor scale, int z, bool pool) -> Tensor")
shir_fpga_inst_lib.define("""
conv3x3p1b14x64(Tensor images, int? padvalue, Tensor kernel, Tensor bias,
                Tensor scale, int z, bool pool,
                int packfactor) -> Tensor
""")

@impl(shir_fpga_inst_lib, "lenet5_linear3", "Meta")
def lenet5_linear3_meta(images, weights, bias, scale, zp):
  return torch.empty(images.shape[0], 10, dtype=torch.int8, device='meta')

@impl(shir_fpga_inst_lib, "lenet5_linear2", "Meta")
def lenet5_linear2_meta(images, weights, bias, scale, zp):
  return torch.empty(images.shape[0], 90, dtype=torch.int8, device='meta')

@impl(shir_fpga_inst_lib, "lenet5_linear1", "Meta")
def lenet5_linear1_meta(images, weights, bias, scale, zp):
  return torch.empty(images.shape[0], 120, dtype=torch.int8, device='meta')

@impl(shir_fpga_inst_lib, "lenet5_conv_pool2", "Meta")
def lenet5_conv_pool1_meta(images, kernel, bias, scales, zp):
  return torch.empty(images.shape[0], 5, 5 * 16, dtype=torch.int8, device='meta')

@impl(shir_fpga_inst_lib, "lenet5_conv_pool1", "Meta")
def lenet5_conv_pool1_meta(images, kernel, bias, scales, zp):
  return torch.empty(images.shape[0], 14, 14 * 6, dtype=torch.int8, device='meta')

@impl(shir_fpga_inst_lib, "conv3x3p1b8x64", "Meta")
def conv3x3p1b8x64(images, padvalue, kernel, bias, scales, zp, pool):
  # TODO: update design to allow per tensor quantization
  assert scales.ndim == 1 and bias.ndim == 1, "invalid scale and bias dimension"
  assert scales.shape[0] == bias.shape[0], "invalid per tensor or per channel scale shape"

  n, ih, iw, ich1 = images.shape
  och, kh, kw, ich2 = kernel.shape

  assert kw == 3 and kh == 3, "conv3x3p1b8x64: window must be 3x3"
  assert och % 64 == 0, "conv3x3p1b8x64: output channel must be divisible by 64"

  # XXX: ich is implicitly padded to cacheline size BUT
  # there are no guarantees on what the filled values are!
  ich1 = (ich1 + (64 - 1)) // 64 * 64
  ich2 = (ich2 + (64 - 1)) // 64 * 64

  # round the input windows to the next tile size
  iw = (iw + (8 - 1)) // 8 * 8
  ih = (ih + (8 - 1)) // 8 * 8

  x = torch.ops.aten.convolution(
      torch.empty((n, ich1, ih, iw), dtype=torch.float, device='meta'),
      torch.empty((och, ich2, kh, kw), dtype=torch.float, device='meta'),
      torch.empty(bias.shape, dtype=torch.float, device='meta'),
      [1, 1], [1, 1], [1, 1], False, [0], 1
  )
  if pool:
    x = torch.ops.aten.max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1])
  return torch.empty(x.permute([0, 2, 3, 1]).shape, dtype=torch.int8, device='meta')

@impl(shir_fpga_inst_lib, "conv3x3p1b14x64", "Meta")
def conv3x3p1b14x64(images, padvalue, kernel, bias, scales, zp, pool, packfactor):
  assert scales.ndim == 1 and bias.ndim == 1, "invalid scale and bias dimension"
  assert scales.shape[0] == 1 or scales.shape[0] == bias.shape[0], "invalid per tensor or per channel scale shape"

  n, ih, iw, ich1 = images.shape
  och, kh, kw, ich2 = kernel.shape

  assert kw == 3 and kh == 3, "conv3x3p1b14x64: window must be 3x3"
  assert och % 64 == 0, "conv3x3p1b14x64: output channel must be divisible by 64"

  # XXX: ich is implicitly padded to cacheline size BUT
  # there are no guarantees on what the filled values are!
  ich1 = (ich1 + (64 - 1)) // 64 * 64
  ich2 = (ich2 + (64 - 1)) // 64 * 64
  assert ich1 == ich2, "conv3x3p1b14x64: input channel of input and kernel mismatch"

  if packfactor != 1:
    assert packfactor in {2, 4, 8}, "conv3x3p1b14x64: invalid packing factor"
    assert ich1 <= 64, "conv3x3p1b14x64: packed convolution must be shallow"

    # packing affects the width dimension
    iw *= packfactor
    ich1 //= packfactor
    ich2 //= packfactor

  # let torch's impl handle the other validations
  x = torch.ops.aten.convolution(
      torch.empty((n, ich1, ih, iw), dtype=torch.float, device='meta'),
      torch.empty((och, ich2, kh, kw), dtype=torch.float, device='meta'),
      torch.empty(bias.shape, dtype=torch.float, device='meta'),
      [1, 1], [0, 0] if padvalue is None else [1, 1], [1, 1], False, [0], 1
  )
  if pool:
    x = torch.ops.aten.max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1])
  return torch.empty(x.permute([0, 2, 3, 1]).shape, dtype=torch.int8, device='meta')

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
    print(f"backend::_adjust_requant_param: shamt of {shamt} is larger, trying to continue by truncating")
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

  # the variations tend to happen near the end of a sequence of nodes (e.g.,
  # difference between qconv and qconv + pooling). thus, it is better to
  # traverse the fx graph in reverse order.
  graph = gm.graph
  for n in reversed(graph.nodes):
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

      else:
        print(f"driver::isel: skipping qlinear node {requant}")

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

    elif (n.op == "call_function" and n.target == torch.ops.shir_intrinsic.int_max_pool2d and
        len(n.args[0].users) == 1 and
        (qrc := extract_qconv_relu(n.args[0])) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 24 and
        n.args[1] == [2, 2] and n.args[2] == [2, 2] and n.args[3] == [0, 0] and n.args[4] == [1, 1]):
      requant, relu, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])
      if (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] % 14 == 0 and
          images.meta.get("val").shape[3] % 14 == 0):

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        batch, ich, ih, iw = images.meta.get("val").shape
        packfactor = 1
        with graph.inserting_before(n):
          if ich <= 32 and iw % (2 * 14) == 0:
            packfactor = 2
          if ich <= 16 and iw % (4 * 14) == 0:
            packfactor = 4
          if ich <= 8 and iw % (8 * 14) == 0:
            packfactor = 8

          if packfactor == 1:
            ni = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))

          else:
            dwidth = 64 // packfactor
            n1 = graph.call_function(torch.ops.aten.pad, (images, [0, 0, 0, 0, 0, dwidth - ich]))
            n2 = graph.call_function(torch.ops.aten.reshape, (n1, [batch, dwidth, ih, packfactor, iw // packfactor]))
            n3 = graph.call_function(torch.ops.aten.permute, (n2, [0, 2, 4, 3, 1]))
            ni = graph.call_function(torch.ops.aten.reshape, (n3, [batch, ih, iw // packfactor, 64]))

            n5 = graph.call_function(torch.ops.aten.pad, (kernel, [0, 0, 0, 0, 0, dwidth - ich]))
            n6 = graph.call_function(torch.ops.aten.repeat, (n5, [1, packfactor, 1, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (n6, [0, 2, 3, 1]))

          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.conv3x3p1b14x64, (ni, zp, nk, bias, n3, requant.args[2], True, packfactor))
          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        graph.erase_node(requant)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      elif (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] % 8 == 0 and
          images.meta.get("val").shape[3] % 8 == 0):
        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        with graph.inserting_before(n):
          n1 = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          n2 = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))
          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.conv3x3p1b8x64, (n1, zp, n2, bias, n3, requant.args[2], True))
          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        graph.erase_node(requant)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      else:
        print(f"driver::isel: skipping qconv + max_pool node {requant}:")

    elif ((qrc := extract_qconv_relu(n)) is not None and
        qrc[2].args[3].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrc[2].args[3].target)).to_signed().bits <= 24):
      requant, relu, conv = qrc
      images, zp, kernel, bias, stride, padding, dilation, groups = conv.args
      adjusted = _adjust_requant_param(requant.args[1], requant.args[2])
      if (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] % 14 == 0 and
          images.meta.get("val").shape[3] % 14 == 0):

        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        rs = n.meta.get("val").shape
        batch, ich, ih, iw = images.meta.get("val").shape
        packfactor = 1
        with graph.inserting_before(n):
          if ich <= 32 and iw % (2 * 14) == 0:
            packfactor = 2
          if ich <= 16 and iw % (4 * 14) == 0:
            packfactor = 4
          if ich <= 8 and iw % (8 * 14) == 0:
            packfactor = 8

          if packfactor == 1:
            ni = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))

          else:
            dwidth = 64 // packfactor
            n1 = graph.call_function(torch.ops.aten.pad, (images, [0, 0, 0, 0, 0, dwidth - ich]))
            n2 = graph.call_function(torch.ops.aten.reshape, (n1, [batch, dwidth, ih, packfactor, iw // packfactor]))
            n3 = graph.call_function(torch.ops.aten.permute, (n2, [0, 2, 4, 3, 1]))
            ni = graph.call_function(torch.ops.aten.reshape, (n3, [batch, ih, iw // packfactor, 64]))

            n5 = graph.call_function(torch.ops.aten.pad, (kernel, [0, 0, 0, 0, 0, dwidth - ich]))
            n6 = graph.call_function(torch.ops.aten.repeat, (n5, [1, packfactor, 1, 1]))
            nk = graph.call_function(torch.ops.aten.permute, (n6, [0, 2, 3, 1]))

          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.conv3x3p1b14x64, (ni, zp, nk, bias, n3, requant.args[2], False, packfactor))
          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      elif (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          stride == [1, 1] and padding == [1, 1] and dilation == [1, 1] and groups == 1 and
          kernel.op == "get_attr" and
          len(getattr(gm, kernel.target).shape) == 4 and
          list(getattr(gm, kernel.target).shape)[2:4] == [3, 3] and
          getattr(gm, kernel.target).shape[0] % 64 == 0 and
          images.meta.get("val").shape[2] % 8 == 0 and
          images.meta.get("val").shape[3] % 8 == 0):
        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        with graph.inserting_before(n):
          n1 = graph.call_function(torch.ops.aten.permute, (images, [0, 2, 3, 1]))
          n2 = graph.call_function(torch.ops.aten.permute, (kernel, [0, 2, 3, 1]))
          n3 = graph.get_attr(sclattr)
          n4 = graph.call_function(torch.ops._shir.conv3x3p1b8x64, (n1, zp, n2, bias, n3, requant.args[2], False))
          n5 = graph.call_function(torch.ops.aten.permute, (n4, [0, 3, 1, 2]))
        n.target = torch.ops.aten.contiguous
        n.args = (n5,)
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(conv)

      else:
        print(f"driver::isel: skipping qconv node {requant}:")

    elif (qrm is not None and
        qrm[2].args[0].op == "get_attr" and
        bit_utils.get_narrow_type(getattr(gm, qrm[2].args[0].target)).to_signed().bits <= 24):
      requant, relu, addmm = qrm
      bias = addmm.args[0]
      images = addmm.args[1]
      weight = addmm.args[2]
      adjusted = _adjust_requant_param([requant.args[1]], requant.args[2])
      if (adjusted is not None and
          (requant.args[2] == -128 or relu is None) and
          weight.op == "get_attr"):
        sclattr = create_new_param()
        setattr(gm, sclattr, torch.nn.Parameter(torch.tensor(adjusted, dtype=torch.int32), False))

        j, k = getattr(gm, weight.target).shape
        i, _ = n.meta.get("val").shape

        # ensure the k dimension has form 3 x 3 x (i_tiles * 64)
        # and    the j dimension has form o_tiles * 64
        i_tiles = (k + (3 * 3 * 64 - 1)) // (3 * 3 * 64)
        o_tiles = (j + (64 - 1)) // 64
        i_pad = i_tiles * 3 * 3 * 64 - k
        o_pad = o_tiles * 64 - j

        with graph.inserting_before(n):
          # try to push the batch dimension inwards to allow weight reusage
          # push to the height dimension first to avoid transpose.
          extra_h = 1
          extra_w = 1
          leftover_i = i
          while extra_h < (14 // 3) and leftover_i % 2 == 0:
            extra_h *= 2
            leftover_i //= 2
          while extra_w < (14 // 3) and leftover_i % 2 == 0:
            extra_w *= 2
            leftover_i //= 2

          n1 = graph.call_function(torch.ops.aten.pad, (images, [0, i_pad]))
          n2 = graph.call_function(torch.ops.aten.view, (n1, [leftover_i, extra_h, extra_w, 3, 3, i_tiles * 64]))
          n2 = graph.call_function(torch.ops.aten.permute, (n2, [0, 1, 3, 2, 4, 5]))
          n2 = graph.call_function(torch.ops.aten.reshape, (n2, [leftover_i, extra_h * 3, extra_w * 3, i_tiles * 64]))
          n3 = graph.call_function(torch.ops.aten.pad, (weight, [0, i_pad, 0, o_pad]))
          n4 = graph.call_function(torch.ops.aten.view, (n3, [o_tiles * 64, 3, 3, i_tiles * 64]))
          n5 = graph.get_attr(sclattr)
          n6 = graph.call_function(torch.ops._shir.conv3x3p1b14x64, (n2, None, n4, bias, n5, requant.args[2], False, 1))
          n7 = graph.call_function(torch.ops.aten.pad, (n6, [0, -o_pad]))
          slice_h = None
          if extra_h > 1:
            slice_h = graph.call_function(torch.arange, (0, extra_h * 3, 3))
          slice_w = None
          if extra_w == extra_h:
            slice_w = slice_h
          elif extra_w > 1:
            slice_w = graph.call_function(torch.arange, (0, extra_w * 3, 3))
          n8 = n7
          if slice_h is not None:
            n8 = graph.call_function(torch.ops.aten.index_select, (n8, 1, slice_h))
          if slice_w is not None:
            n8 = graph.call_function(torch.ops.aten.index_select, (n8, 2, slice_w))

        n.target = torch.ops.aten.view
        n.args = (n8, [i, j])
        if relu is not None: graph.erase_node(relu)
        graph.erase_node(addmm)

      else:
        print(f"driver::isel: skipping qlinear node {requant}")

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

      elif (n.target == torch.ops.aten.permute and
          n.args[0].op == "call_function" and
          n.args[0].target == torch.ops.aten.contiguous):
        cont = n.args[0]
        n.args = (cont.args[0],) + n.args[1:]
        if not cont.users:
          graph.erase_node(cont)
        changed = True

      elif (n.target == torch.ops.aten.pad and
          all((x == 0 for x in n.args[1]))):
        # also handles the empty pad / no-op case
        n.replace_all_uses_with(n.args[0])
        graph.erase_node(n)
        changed = True

      elif (n.target == torch.ops.aten.adaptive_avg_pool2d.default and
          list(n.args[0].meta.get("val").shape)[2:] == n.args[1]):
        n.replace_all_uses_with(n.args[0])
        graph.erase_node(n)
        changed = True

      elif (n.target == torch.ops.quantized_decomposed.quantize_per_tensor.default and
          n.args[0].target == torch.ops.quantized_decomposed.dequantize_per_tensor.default and
          n.args[1:] == n.args[0].args[1:]):
        dq = n.args[0]
        n.replace_all_uses_with(dq.args[0])
        graph.erase_node(n)
        if not dq.users:
          graph.erase_node(dq)
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

      elif (n.target in {torch.ops.aten.view,
                         torch.ops.aten.pad,
                         torch.ops.aten.permute} and
          n.args[0].op == "get_attr" and
          len(n.args[0].users) == 1):
        h = n.args[0]
        m = n.target(getattr(gm, h.target), *n.args[1:])
        setattr(gm, h.target, torch.nn.Parameter(m, False))
        with graph.inserting_before(n):
          r = graph.get_attr(h.target)
        n.replace_all_uses_with(r, propagate_meta=True)
        graph.erase_node(n)
        graph.erase_node(h)
        changed = True

  graph.lint()
  gm.recompile()

def compute_layout(gm: fx.GraphModule):
  max_data = 0
  max_inst = 0
  layout = {}

  def lookup(n: fx.Node, shirTy, fix_placement=True):
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

      if fix_placement:
        meminfo = (max_data, lines, shirTy)
        max_data += lines
      else:
        meminfo = (None, lines, shirTy)

      layout[n] = meminfo

    return meminfo

  # map out the memory allocations in two phases:
  # the first phase allocates all the parameters (always live).
  # the second phase allocates regions that are potentially reusable.

  for n in gm.graph.nodes:
    if n.op != "call_function":
      continue

    if n.target in {torch.ops._shir.lenet5_conv_pool1,
                    torch.ops._shir.lenet5_conv_pool2,
                    torch.ops._shir.lenet5_linear1,
                    torch.ops._shir.lenet5_linear2,
                    torch.ops._shir.lenet5_linear3}:
      images, weights, bias, scl, zp = n.args
      if images.op == "get_attr":
        lookup(images, types.SI(8))
      lookup(weights, types.SI(8))
      lookup(bias, types.SI(20))
      lookup(scl, types.SI(28))
      batch = n.meta.get("val").shape[0]
      assert batch > 0 and batch % 8 == 0, "backend::isel: batch size must be multiple of 8"
      max_inst += (batch + (8 * 0xF - 1)) // (8 * 0xF)

    elif n.target in {torch.ops._shir.conv3x3p1b8x64}:
      images, padvalue, kernel, bias, scales, zp, pool = n.args
      if images.op == "get_attr":
        lookup(images, types.SI(8))
      lookup(kernel, types.SI(8))
      lookup(bias, types.SI(24))
      lookup(scales, types.UI(28))

      # for now, do the simple thing, which is each batch is one instruction
      batch = n.meta.get("val").shape[0]
      max_inst += batch

    elif n.target in {torch.ops._shir.conv3x3p1b14x64}:
      images, padvalue, kernel, bias, scales, zp, pool, packfactor = n.args
      if images.op == "get_attr":
        lookup(images, types.SI(8))
      lookup(kernel, types.SI(8))
      lookup(bias, types.SI(24))
      lookup(scales, types.UI(28))

      # for now, do the simple thing, which is each batch is one instruction
      batch = n.meta.get("val").shape[0]
      max_inst += batch

  # knowing the optimal case is actually quite tricky, approximate it by
  # looking for the first available space.
  # (is prone to fragmentation, but at least as good as the naive case)

  ephemeral_start = max_data
  live_nodes = set()

  def mark(n: fx.Node, shirTy):
    info = lookup(n, shirTy, fix_placement=False)
    if info[0] is not None:
      # already allocated
      return

    offset = ephemeral_start
    for blk in sorted((layout[m] for m in live_nodes), key=lambda x: x[0]):
      if blk[0] - offset >= info[1]:
        # first available space that is large enough
        break

      # otherwise, move on to the next available space
      offset = blk[0] + blk[1]

    layout[n] = (offset,) + info[1:]
    live_nodes.add(n)

  for n in reversed(gm.graph.nodes):
    if n.op != "call_function":
      continue

    if n.target in {torch.ops._shir.lenet5_conv_pool1,
                    torch.ops._shir.lenet5_conv_pool2,
                    torch.ops._shir.lenet5_linear1,
                    torch.ops._shir.lenet5_linear2,
                    torch.ops._shir.lenet5_linear3}:
      images, weights, bias, scl, zp = n.args
      mark(n, types.SI(8))
      if images.op != "get_attr":
        mark(images, types.SI(8))
      live_nodes.remove(n)

    elif n.target in {torch.ops._shir.conv3x3p1b8x64}:
      images, padvalue, kernel, bias, scales, zp, pool = n.args
      mark(n, types.SI(8))
      if images.op != "get_attr":
        mark(images, types.SI(8))
      live_nodes.remove(n)

    elif n.target in {torch.ops._shir.conv3x3p1b14x64}:
      images, padvalue, kernel, bias, scales, zp, pool, packfactor = n.args
      mark(n, types.SI(8))
      if images.op != "get_attr":
        mark(images, types.SI(8))
      live_nodes.remove(n)

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

ENCTBL_conv3x3p1b14x64 = {
  "ImageOCHTileNum": (0, 7),
  "ImageHTileNum": (7, 12),
  "ImageWTileNum": (12, 17),
  "ImageICHTileNum": (17, 23),
  "ImageHLowerBound": (23, 31),
  "ImageHUpperBound": (31, 39),
  "ImageWLowerBound": (39, 47),
  "ImageWUpperBound": (47, 55),
  "ImageHOffset": (55, 69),
  "ImageWOffset": (69, 75),
  "ImageWLowerOOBVal": (75, 84),
  "ImageWUpperOOBVal": (84, 93),
  "WeightOCHTileNum": (93, 100),
  "WeightHTileNum": (100, 105),
  "WeightWTileNum": (105, 110),
  "WeightICHTileNum": (110, 116),
  "WeightOCHOffset": (116, 125),
  "WeightWinOffset": (125, 131),
  "RealOCHTileSize": (131, 138),
  "ImagePointer": (138, 162),
  "WeightPointer": (162, 186),
  "BiasCacheLines": (186, 194),
  "BiasPointer": (194, 218),
  "PoolingInstSlice": (218, 219),
  "WriteAddrOCHTileNum": (219, 226),
  "WriteAddrHTileNum": (226, 231),
  "WriteAddrWTileNum": (231, 236),
  "WriteAddrHOuterOffset": (236, 250),
  "WriteAddrWOuterOffset": (250, 258),
  "WriteAddrHPoolReverse": (258, 260),
  "WriteAddrWPoolReverse": (260, 262),
  "WriteAddrHPROffset": (262, 275),
  "WriteAddrWPROffset": (275, 282),
  "WriteAddrHRealLimit": (282, 302),
  "WriteAddrWRealLimit": (302, 315),
  "WriteAddrWFoldLen": (315, 319),
  "WriteAddrWFoldOffset": (319, 326),
  "ResultImagePointer": (326, 350),
  "WeightReuseEnabled": (350, 351),
  "ImageWLowerOOBShamt": (351, 355),
  "ImageWUpperOOBShamt": (355, 359),
  "PartialSumInstSlice": (359, 361),
  "ImagePadValue": (361, 369),
  "RequantZeroPoint": (369, 377),
  "RequantCacheLines": (377, 385),
  "RequantPointer": (385, 409),
  "RequantPerTensor": (409, 410),
}

def _encode(name, tbl, x):
  i = 0
  for k, v in x.items():
    lo, hi = tbl[k]
    w = hi - lo
    assert (v >> w) == (-1 if v < 0 else 0), f"backend::emit: field {k} too narrow for {name}: {v} as {w} bits"
    i |= (v & ((1 << w) - 1)) << lo

  for u in tbl:
    assert u in x, f"backend::emit: missing field {u}"
  return i

def emit(gm: fx.GraphModule, max_inst: int, data_layout):
  from . import driver
  import mmap           # for the page size constant

  if max_inst == 0:
    return None

  # the bootstrapping instruction must be at location 0.
  # afterwards, we put the instruction followed by the data.
  BASEADDR_INST = 1
  BASEADDR_DATA = BASEADDR_INST + max_inst

  bytes_per_cl = config.CACHELINE_BITS // 8
  total_sz = 1 + max_inst + max((u + v for (u, v, _) in data_layout.values()))
  total_sz = bytes_per_cl * total_sz
  total_sz = (total_sz + (mmap.PAGESIZE - 1)) // mmap.PAGESIZE * mmap.PAGESIZE

  pptr = driver.alloc_buffer(total_sz)
  assert pptr is not None, "backend::emit: Unable to allocate shared buffer"

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
    n1 = new_graph.call_function(_wrap_buffer, (tensor.shape, tensor.dtype, _pptr, BASEADDR_DATA + offs, sz))
    n2 = new_graph.call_function(torch.clone, (n1,))
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
          assert (krn_ptr & 0xFFFFFF) == krn_ptr, "backend::emit: pointer too wide"
          assert (bis_ptr & 0xFFFFFF) == bis_ptr, "backend::emit: pointer too wide"
          assert (scl_ptr & 0xFFFFFF) == scl_ptr, "backend::emit: pointer too wide"
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
            assert (img_ptr & 0xFFFFFF) == img_ptr, "backend::emit: pointer too wide"
            assert (res_ptr & 0xFFFFFF) == res_ptr, "backend::emit: pointer too wide"
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

        elif n.op == "call_function" and n.target == torch.ops._shir.conv3x3p1b8x64:
          images, padvalue, kernel, bias, scales, zp, pool = n.args
          img_ptr = BASEADDR_DATA + data_layout[images][0]
          krn_ptr = BASEADDR_DATA + data_layout[kernel][0]
          bis_ptr = BASEADDR_DATA + data_layout[bias][0]
          scl_ptr = BASEADDR_DATA + data_layout[scales][0]
          res_ptr = BASEADDR_DATA + data_layout[n][0]

          # asserting the img_ptr and res_ptr happens later
          assert (krn_ptr & 0xFFFFFF) == krn_ptr, "backend::emit: pointer too wide"
          assert (bis_ptr & 0xFFFFFF) == bis_ptr, "backend::emit: pointer too wide"
          assert (scl_ptr & 0xFFFFFF) == scl_ptr, "backend::emit: pointer too wide"
          assert -128 <= padvalue < 128, "backend::emit: signed pad value too wide for conv3x3p1b8x64"
          assert -128 <= zp < 128, "backend::emit: signed zero point too wide for conv3x3p1b8x64"

          batch, h, w, ich = images.meta.get("val").shape
          och, kh, kw, _ = kernel.meta.get("val").shape
          _, rh, rw, _ = n.meta.get("val").shape
          pertensor = scales.meta.get("val").shape[0] == 1

          ochTileNum = (och + (64 - 1)) // 64
          ichTileNum = (ich + (64 - 1)) // 64
          assert (ochTileNum & 0xF) == ochTileNum, "backend::emit: OCH too wide for conv3x3p1b8x64"
          assert (ichTileNum & 0xF) == ichTileNum, "backend::emit: ICH too wide for conv3x3p1b8x64"

          hTileNum = h // 8
          wTileNum = w // 8
          assert (h & (7 << 3)) == h, "backend::emit: H must be 6 bits and divisible by 8 for conv3x3p1b8x64"
          assert (w & (7 << 3)) == w, "backend::emit: W must be 6 bits and divisible by 8 for conv3x3p1b8x64"

          bis_lines = data_layout[bias][1]
          scl_lines = data_layout[scales][1]
          assert (bis_lines & 0x1F) == bis_lines, "backend::emit: Bias spans to many cachelines for conv3x3p1b8x64"
          assert (scl_lines & 0x1F) == scl_lines, "backend::emit: QScale spans to many cachelines for conv3x3p1b8x64"

          weight_reuse = ich <= 64 and h > 8 and w > 8

          imgHOffset = data_layout[images][1] // batch // h
          imgWOffset = data_layout[images][1] // batch // h // w
          assert (imgHOffset & 0x1FF) == imgHOffset, "backend::emit: image H spans too many cachelines for conv3x3p1b8x64"
          assert (imgWOffset & 0xF) == imgWOffset, "backend::emit: image W spans too many cachelines for conv3x3p1b8x64"

          krnOchOffset = data_layout[kernel][1] // och
          krnWinOffset = data_layout[kernel][1] // och // kh // kw
          assert (krnOchOffset & 0x7F) == krnOchOffset, "backend::emit: kernel OCH spans too many cachelines for conv3x3p1b8x64"
          assert (krnWinOffset & 0xF) == krnWinOffset, "backend::emit: kernel W spans too many cachelines for conv3x3p1b8x64"

          resHOffset = data_layout[n][1] // batch // rh
          resWOffset = data_layout[n][1] // batch // rh // rw
          if pool:
            assert (resHOffset & 0xFF) == resHOffset, "backend::emit: result H spans too many cachelines for conv3x3p1b8x64"
            assert (resWOffset & 0xF) == resWOffset, "backend::emit: result W spans too many cachelines for conv3x3p1b8x64"
          else:
            # need to multiply by 2, which is like one significant bit less
            assert (resHOffset & 0x7F) == resHOffset, "backend::emit: result H spans too many cachelines for conv3x3p1b8x64"
            assert (resWOffset & 7) == resWOffset, "backend::emit: result W spans too many cachelines for conv3x3p1b8x64"

          pending_inst = iptr if pending_inst is None else pending_inst
          pending_values.add(n)

          imgs_per_batch = data_layout[images][1] // batch
          ress_per_batch = data_layout[n][1] // batch
          while batch > 0:
            assert (img_ptr & 0xFFFFFF) == img_ptr, "backend::emit: pointer too wide"
            assert (res_ptr & 0xFFFFFF) == res_ptr, "backend::emit: pointer too wide"

            inst = (0
                | ochTileNum << 0   # ImageOCHTileNum
                | hTileNum << 4     # ImageHTileNum
                | wTileNum << 7     # ImageWTileNum
                | ichTileNum << 10  # ImageICHTileNum
                | 1 << 14           # ImageHLowerBound
                | h << 20           # ImageHUpperBound
                | 1 << 26           # ImageWLowerBound
                | w << 32           # ImageWUpperBound
                | imgHOffset << 38  # ImageHOffset
                | imgWOffset << 47  # ImageWOffset
                | ochTileNum << 51  # WeightOCHTileNum
                | (1 if weight_reuse else hTileNum) << 55 # WeightHTileNum
                | (1 if weight_reuse else wTileNum) << 58 # WeightWTileNum
                | ichTileNum << 61  # WeightICHTileNum
                | krnOchOffset << 65 # WeightOCHOffset
                | krnWinOffset << 72 # WeightWinOffset
                | img_ptr << 76     # ImagePointer
                | krn_ptr << 100    # WeightPointer
                | bis_lines << 124  # BiasCacheLines
                | bis_ptr << 129    # BiasPointer
                | (1 if pool else 0) << 153 # PoolEnabled
                | ochTileNum << 154 # WriteAddrOCHTileNum
                | hTileNum << 158 # WriteAddrHTileNum
                | wTileNum << 161 # WriteAddrWTileNum
                | (resHOffset * (1 if pool else 2)) << 164 # WriteAddrHOuterOffset
                | (resWOffset * (1 if pool else 2)) << 172 # WriteAddrWOuterOffset
                | (1 if pool else 2) << 176 # WriteAddrHPoolReverse
                | (1 if pool else 2) << 178 # WriteAddrWPoolReverse
                | resHOffset << 180 # WriteAddrHPROffset
                | resWOffset << 188 # WriteAddrWPROffset
                | res_ptr << 192 # ResultImagePointer
                | (1 if weight_reuse else 0) << 216 # WeightReuseEnabled
                | ochTileNum << 220 # WeightReuseOCHTileNum
                | hTileNum << 223 # WeightReuseHTileNum
                | wTileNum << 226 # WeightReuseWTileNum
                | (padvalue & 0xFF) << 233 # ImagePadValue
                | (zp & 0xFF) << 241 # RequantZeroPoint
                | scl_lines << 249 # RequantCacheLines
                | scl_ptr << 254 # RequantPointer
                | (1 if pertensor else 0) << 278 # RequantPerTensor
                | 0)

            pptr[iptr * bytes_per_cl:(iptr + 1) * bytes_per_cl] = inst.to_bytes(bytes_per_cl, byteorder="little")
            iptr += 1

            img_ptr += imgs_per_batch
            res_ptr += ress_per_batch
            batch -= 1

        elif n.op == "call_function" and n.target == torch.ops._shir.conv3x3p1b14x64:
          images, padvalue, kernel, bias, scales, zp, pool, packfactor = n.args
          img_ptr = BASEADDR_DATA + data_layout[images][0]
          krn_ptr = BASEADDR_DATA + data_layout[kernel][0]
          bis_ptr = BASEADDR_DATA + data_layout[bias][0]
          scl_ptr = BASEADDR_DATA + data_layout[scales][0]
          res_ptr = BASEADDR_DATA + data_layout[n][0]

          # asserting the img_ptr and res_ptr happens later
          assert (krn_ptr & 0xFFFFFF) == krn_ptr, "backend::emit: pointer too wide"
          assert (bis_ptr & 0xFFFFFF) == bis_ptr, "backend::emit: pointer too wide"
          assert (scl_ptr & 0xFFFFFF) == scl_ptr, "backend::emit: pointer too wide"
          assert -128 <= (padvalue or 0) < 128, "backend::emit: signed pad value too wide for conv3x3p1b14x64"
          assert -128 <= zp < 128, "backend::emit: signed zero point too wide for conv3x3p1b14x64"

          batch, h, w, ich = images.meta.get("val").shape
          och, kh, kw, _ = kernel.meta.get("val").shape
          _, rh, rw, _ = n.meta.get("val").shape
          pertensor = scales.meta.get("val").shape[0] == 1

          ochTileNum = (och + (64 - 1)) // 64
          ichTileNum = (ich + (64 - 1)) // 64

          hTileNum = (h + (14 - 1)) // 14
          wTileNum = (w + (14 - 1)) // 14

          bis_lines = data_layout[bias][1]
          scl_lines = data_layout[scales][1]

          weight_reuse = ich <= 64 and h > 14 and w > 14

          imgHOffset = data_layout[images][1] // batch // h
          imgWOffset = data_layout[images][1] // batch // h // w

          krnOchOffset = data_layout[kernel][1] // och
          krnWinOffset = data_layout[kernel][1] // och // kh // kw

          resHOffset = data_layout[n][1] // batch // rh
          resWOffset = data_layout[n][1] // batch // rh // rw

          oob_shamt = 0 if packfactor == 1 else 8 // packfactor
          fold_ofs = w // 2 if pool else w
          psum_bits = oob_shamt.bit_length()

          pending_inst = iptr if pending_inst is None else pending_inst
          pending_values.add(n)

          imgs_per_batch = data_layout[images][1] // batch
          ress_per_batch = data_layout[n][1] // batch
          while batch > 0:
            inst = _encode("conv3x3p1b14x64", ENCTBL_conv3x3p1b14x64, {
              "ImageOCHTileNum": ochTileNum,
              "ImageHTileNum": hTileNum,
              "ImageWTileNum": wTileNum,
              "ImageICHTileNum": ichTileNum,
              "ImageHLowerBound": 1 if padvalue is not None else 0,
              "ImageHUpperBound": h if padvalue is not None else h - 1,
              "ImageWLowerBound": 1 if padvalue is not None else 0,
              "ImageWUpperBound": w if padvalue is not None else w - 1,
              "ImageHOffset": imgHOffset,
              "ImageWOffset": imgWOffset,
              "ImagePointer": img_ptr,
              "ImagePadValue": padvalue or 0,
              "WeightOCHTileNum": ochTileNum,
              "WeightHTileNum": 1 if weight_reuse else hTileNum,
              "WeightWTileNum": 1 if weight_reuse else wTileNum,
              "WeightICHTileNum": ichTileNum,
              "WeightOCHOffset": krnOchOffset,
              "WeightWinOffset": krnWinOffset,
              "WeightPointer": krn_ptr,
              "BiasCacheLines": bis_lines,
              "BiasPointer": bis_ptr,
              "WriteAddrOCHTileNum": ochTileNum,
              "WriteAddrHTileNum": hTileNum,
              "WriteAddrWTileNum": wTileNum,
              "WriteAddrHOuterOffset": resHOffset * (1 if pool else 2),
              "WriteAddrWOuterOffset": resWOffset * (1 if pool else 2),
              "WriteAddrHPoolReverse": 1 if pool else 2,
              "WriteAddrWPoolReverse": 1 if pool else 2,
              "WriteAddrHPROffset": resHOffset,
              "WriteAddrWPROffset": resWOffset,
              "WriteAddrHRealLimit": resHOffset * (rh - 1),
              "WriteAddrWRealLimit": resWOffset * (rw - 1),
              "ResultImagePointer": res_ptr,
              "WeightReuseEnabled": 1 if weight_reuse else 0,
              "RequantZeroPoint": zp,
              "RequantCacheLines": scl_lines,
              "RequantPointer": scl_ptr,
              "RequantPerTensor": 1 if pertensor else 0,

              "ImageWLowerOOBVal": -1 if packfactor == 1 else 0,
              "ImageWUpperOOBVal": -1 if packfactor == 1 else w - 1,
              "ImageWLowerOOBShamt": -oob_shamt,
              "ImageWUpperOOBShamt": oob_shamt,
              "PoolingInstSlice": 1 if pool else 0,
              "PartialSumInstSlice": psum_bits,
              "RealOCHTileSize": 64,
              "WriteAddrWFoldLen": packfactor,
              "WriteAddrWFoldOffset": fold_ofs,
            })

            pptr[iptr * bytes_per_cl:(iptr + 1) * bytes_per_cl] = inst.to_bytes(bytes_per_cl, byteorder="little")
            iptr += 1

            img_ptr += imgs_per_batch
            res_ptr += ress_per_batch
            batch -= 1

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

def _can_omit_copy(f, n: fx.Node) -> bool:
  schemas = []
  if isinstance(f, torch._ops.OpOverload):
    schemas = [f._schema]
  elif isinstance(f, torch._ops.OpOverloadPacket):
    schemas = f._schemas.values()

  if schemas and all((
    not s.is_mutable and not any((x.alias_info for x in s.arguments))
    for s in schemas
  )):
    # if the schema says not mutable and has no alias information
    # then it assume it is safe to omit the copy.
    return True

  # if all else fails, assume it's not safe to omit the copy
  return False

def peephole(g: fx.Graph):
  for n in g.nodes:
    if n.op != "call_function":
      continue
    if (n.target == torch.clone and
        len(n.users) == 1 and n.next in n.users and
        n.next.op == "call_function" and
        _can_omit_copy(n.next.target, n)):
      n.replace_all_uses_with(n.args[0])
      g.erase_node(n)

    elif (n.target == torch.ops.quantized_decomposed.quantize_per_tensor.default and
        n.args[3] == -128 and n.args[4] == 127 and n.args[5] == torch.int8):
      with g.inserting_before(n):
        n1 = g.call_function(torch.quantize_per_tensor, n.args[:3], {"dtype": torch.qint8})
        n2 = g.call_method("int_repr", (n1,))
      n.replace_all_uses_with(n2, propagate_meta=True)
      g.erase_node(n)

    elif (n.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default and
        n.args[3] == -128 and n.args[4] == 127 and n.args[5] == torch.int8):
      with g.inserting_before(n):
        n1 = g.call_function(torch._make_per_tensor_quantized_tensor, n.args[:3])
        n2 = g.call_method("dequantize", (n1,))
      n.replace_all_uses_with(n2, propagate_meta=True)
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
  if max_inst == 0:
    return gm.forward

  pptr, graph = emit(gm, max_inst, data_layout)
  peephole(graph)
  gm2 = fx.GraphModule(gm, graph)

  from . import driver
  return _Wrapper(gm2, driver, pptr)

