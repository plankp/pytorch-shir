"""
The Python side of the FPGA driver,
which is necessary when running with synthesis mode.

This module is expected to be loaded only when synthesis mode is enabled.
"""

import os
from contextlib import contextmanager
import ctypes as C
import weakref
import subprocess
from . import config

_impl = C.cdll.LoadLibrary(config.DRIVER_LIB)

# buffer.h
_impl.alloc_buffer.restype  = C.c_void_p
_impl.alloc_buffer.argtypes = [C.c_size_t]
_impl.free_buffer.restype   = None
_impl.free_buffer.argtypes  = [C.c_void_p]

class Fpga:
  def __init__(self, handle):
    self._hndl = handle
    self._finalizer = weakref.finalize(self, _impl.close_fpga, self._hndl)

  def close(self):
    self._finalizer()

  @property
  def closed(self):
    return not self._finalizer.alive

  @contextmanager
  def prepare_buffer(self, mem, bytes_needed: int):
    wsid = C.c_uint64()
    if r := _impl.prepare_buffer(
      self._hndl, (C.c_char * bytes_needed).from_buffer(mem),
      C.c_uint64(bytes_needed), C.byref(wsid)
    ):
      raise Exception(f"_impl.prepare_buffer failed: {r}")
    try:
      yield wsid
    finally:
      _impl.release_buffer(self._hndl, wsid)

  def reset(self):
    if r := _impl.fpgaReset(self._hndl):
      raise Exception(f"_impl.fpgaReset failed: {r}")

  def read_mmio64(self, ionum: int, offset: int) -> int:
    result = C.c_uint64()
    res = _impl.fpgaReadMMIO64(self._hndl, C.c_uint32(ionum), C.c_uint64(offset), C.byref(result))
    if res:
      raise Exception(f"driver: failed to read accelerator register {res}")
    return result.value

  def write_mmio64(self, ionum: int, offset: int, value: int) -> int:
    res = _impl.fpgaWriteMMIO64(self._hndl, C.c_uint32(ionum), C.c_uint64(offset), C.c_uint64(value))
    if res:
      raise Exception(f"driver: failed to write accelerator register {res}")

  def start_computation(self):
    # don't consume the completion flag (otherwise is_complete can't poll it)
    self.write_mmio64(0, 0x10, 0)
    # mark the input as valid (which effectively starts the FPGA)
    self.write_mmio64(0, 0x08, 1)

  def stop_computation(self):
    # mark input as invalid to stop the FPGA
    self.write_mmio64(0, 0x08, 0)
    # consume the completion flag if set
    self.write_mmio64(0, 0x10, 1)

  def is_complete(self) -> bool:
    return self.read_mmio64(0, 0x80)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

def find_and_open_fpga(uuid):
  handle = C.POINTER(C.c_void_p)()
  if r := _impl.find_and_open_fpga(uuid, C.byref(handle)):
    raise Exception(f"_impl.find_and_open_fpga failed: {r}")
  return Fpga(handle)

def alloc_buffer(length: int):
  addr = _impl.alloc_buffer(length)
  if addr == 0:
    return None

  return (C.c_char * length).from_address(addr)

def free_buffer(buf):
  _impl.free_buffer(buf)

_last_flashed_gbs = None
_last_opened_fpga = None

def release_fpga():
  global _last_opened_fpga, _last_flashed_gbs
  if _last_opened_fpga is None:
    return
  _last_opened_fpga.close()
  _last_opened_fpga = None
  _last_flashed_gbs = None

def configure_gbs(gbs_file):
  global _last_opened_fpga, _last_flashed_gbs
  if _last_opened_fpga is not None and _last_flashed_gbs is not None and os.path.samefile(_last_flashed_gbs, gbs_file):
    return _last_opened_fpga

  # otherwise, since we noly have one FPGA on our server, release the old one
  release_fpga()

  # and then reconfigure it
  subprocess.run(['fpgaconf', '-v', gbs_file])

  inst = find_and_open_fpga(config.ACCEL_UUID)
  _last_flashed_gbs = gbs_file
  _last_opened_fpga = inst
  return inst
