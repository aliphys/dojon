"""Expose only the supplemental libraries absent from the pinned JetPack base."""

from pathlib import Path
import subprocess
import sysconfig

root = Path(sysconfig.get_path("purelib")) / "nvidia"
paths = [root / name / "lib" for name in ("nccl", "cusparselt", "nvshmem")]
for path in paths:
    if not path.is_dir():
        raise RuntimeError(f"Missing supplemental library directory: {path}")
Path("/etc/ld.so.conf.d/pytorch-extras.conf").write_text(
    "".join(f"{path}\n" for path in paths)
)
subprocess.run(["ldconfig"], check=True)
