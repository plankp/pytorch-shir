"""
Utility for dealing with the intermediate scala project and synthesis
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
import shutil
import subprocess
from shir import config

def hashfile(filename):
  # XXX: Newer Python platforms have file_digest.
  # but the server uses an older one that doesn't...
  # so we use the one suggested here:
  # https://stackoverflow.com/a/44873382/11416644
  import hashlib
  with open(filename, "rb", buffering=0) as f:
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    while n := f.readinto(mv):
      h.update(mv[:n])
    return h.hexdigest()

class SHIRProject(metaclass=ABCMeta):
  clname: str
  output_dir: Path

  def __init__(self, clname: str, output_dir: Path):
    self.clname = clname
    self.output_dir = output_dir

  def consult_cache(self):
    fname = self.output_dir / f"{self.clname}.scala"
    digest = hashfile(fname)
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

  @abstractmethod
  def emit_source(self, gm, host_mapping):
    with (self.output_dir / f"{self.clname}.scala").open("w", encoding="utf-8") as f:
      pass

