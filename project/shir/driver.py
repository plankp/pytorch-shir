"""
The Python side of the FPGA driver,
which is necessary when running with synthesis mode.

This module is expected to be loaded only when synthesis mode is enabled.
"""

from contextlib import contextmanager
from typing import Optional
import ctypes as C
import weakref
from . import config

_impl = C.cdll.LoadLibrary(config.DRIVER_LIB)

# buffer.h
_impl.alloc_buffer.restype  = C.c_void_p
_impl.alloc_buffer.argtypes = [C.c_void_p, C.c_size_t]
_impl.free_buffer.restype   = C.c_int
_impl.free_buffer.argtypes  = [C.c_void_p, C.c_size_t]

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

  def start_computation(self):
    if r := _impl.start_computation(self._hndl):
      raise Exception(f"_impl.start_computation failed: {r}")

  def is_complete(self) -> bool:
    done = C.c_uint64(0)
    if r := _impl.poll_for_completion(self._hndl, C.byref(done)):
      raise Exception(f"_impl.poll_for_completion failed: {r}")
    return done.value

  def read_mmio64(self, ionum: int, offset: int) -> Optional[int]:
    result = C.c_uint64()
    if _impl.fpgaReadMMIO64(
      self._hndl, C.c_uint32(ionum), C.c_uint64(offset), C.byref(result)
    ):
      return None
    return result.value

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
  addr = _impl.alloc_buffer(None, length)
  if addr == -1:
    # because mmap uses -1 for errors
    return None

  return (C.c_char * length).from_address(addr)

def free_buffer(buf):
  _impl.free_buffer(buf, len(buf))
