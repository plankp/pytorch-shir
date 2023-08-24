from contextlib import contextmanager
from typing import Optional
import ctypes as C

class Wrapper:
  _impl: C.CDLL

  def __init__(self, dll: C.CDLL):
    self._impl = dll

  @contextmanager
  def find_and_open_fpga(self, uuid):
    handle = C.POINTER(C.c_void_p)()
    if r := self._impl.find_and_open_fpga(uuid, C.byref(handle)):
      raise Exception(f"driver.find_and_open_fpga failed: {r}")
    try:
      yield handle
    finally:
      self._impl.close_fpga(handle)

  @contextmanager
  def prepare_buffer(self, handle, mem, bytes_needed):
    wsid = C.c_uint64()
    if r := self._impl.prepare_buffer(handle, mem, bytes_needed, C.byref(wsid)):
      raise Exception(f"driver.prepare_buffer failed: {r}")
    try:
      yield wsid
    finally:
      self._impl.release_buffer(handle, wsid)

  def start_computation(handle):
    if r := self._impl.start_computation(handle):
      raise Exception(f"driver.start_computation failed: {r}")

  def is_complete(handle) -> bool:
    done = C.c_uint64(0)
    if r := self._impl.poll_for_completion(handle, C.byref(done)):
      raise Exception(f"driver.poll_for_completion failed: {r}")
    return done.value

  def read_mmio64(handle, ionum: int, offset: int) -> Optional[int]:
    result = C.c_uint64()
    if self._impl.fpgaReadMMIO64(
      handle, C.c_uint32(ionum), C.c_uint64(offset), C.byref(result)
    ):
      return None
    return result.value
