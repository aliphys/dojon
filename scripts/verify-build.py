"""Verify locked requirements, dependency closure and base imports without a GPU."""

import importlib
import importlib.metadata as metadata
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

requirements = []
for lock_file in sorted(Path("/opt/locks").glob("requirements-*.lock")):
    for raw in lock_file.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith(("#", "--")):
            requirements.append(Requirement(line))

installed = {canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions()}
managed = {canonicalize_name(requirement.name) for requirement in requirements}
for requirement in requirements:
    name = canonicalize_name(requirement.name)
    if name not in installed:
        raise RuntimeError(f"Missing locked requirement: {requirement}")
    actual = installed[name].version
    if requirement.specifier and actual not in requirement.specifier:
        raise RuntimeError(f"{requirement}, but installed {actual}")

# These two wheel requirements are intentionally satisfied by JetPack packages.
# Their pip counterparts must never shadow the base CUDA/cuDNN libraries.
system_supplied = {"cuda-toolkit", "nvidia-cudnn-cu13"}
for name in system_supplied:
    if name in installed:
        raise RuntimeError(f"Unexpected replacement of a base library: {name}")
for name in managed:
    for raw in metadata.requires(name) or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        dependency = canonicalize_name(requirement.name)
        if name == "torch" and dependency in system_supplied:
            continue
        if dependency not in installed:
            raise RuntimeError(f"{name}: missing {requirement}")
        actual = installed[dependency].version
        if requirement.specifier and actual not in requirement.specifier:
            raise RuntimeError(f"{name}: {requirement}, but installed {actual}")

for name in (
    "torch", "torchvision", "torchaudio", "triton", "tensorrt", "cv2",
    "numpy", "PIL", "scipy", "tqdm", "pandas", "matplotlib", "seaborn",
    "onnx", "onnxruntime", "jtop", "wandb", "jupyterlab", "ipykernel",
):
    importlib.import_module(name)
import cv2
import numpy
import tensorrt
import torch
assert sys.version_info[:2] == (3, 12), sys.version
assert cv2.__version__ == "4.8.0", cv2.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
assert tensorrt.__version__ == "10.16.2.10", tensorrt.__version__
assert torch.version.cuda == "13.2", torch.version.cuda

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
figure, axis = plt.subplots()
sns.lineplot(x=[0, 1], y=[0, 1], ax=axis)
figure.canvas.draw()
plt.close(figure)

print(f"Build checks passed: {len(managed)} locked distributions; PyTorch {torch.__version__}.")
