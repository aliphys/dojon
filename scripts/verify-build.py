"""Verify project additions and NGC-provided framework imports without a GPU."""

import importlib
import importlib.metadata as metadata
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

requirements = []
for raw in (Path("/opt/locks/requirements-python.lock")).read_text().splitlines():
    line = raw.strip()
    if line and not line.startswith(("#", "--")):
        requirements.append(Requirement(line))

installed = {canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions()}
managed = {canonicalize_name(requirement.name) for requirement in requirements}
for requirement in requirements:
    name = canonicalize_name(requirement.name)
    if name not in installed:
        raise RuntimeError(f"Missing locked requirement: {requirement}")

for name in managed:
    for raw in metadata.requires(name) or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        dependency = canonicalize_name(requirement.name)
        if dependency not in installed:
            raise RuntimeError(f"{name}: missing {requirement}")

for name in (
    "torch", "torchvision", "triton", "tensorrt",
    "numpy", "PIL", "scipy", "tqdm", "pandas", "matplotlib", "seaborn",
    "onnx", "onnxruntime", "jtop", "wandb", "jupyterlab", "ipykernel",
):
    importlib.import_module(name)
import numpy
import tensorrt
import torch
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.version.cuda is not None, "NGC PyTorch must be CUDA-enabled"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
figure, axis = plt.subplots()
sns.lineplot(x=[0, 1], y=[0, 1], ax=axis)
figure.canvas.draw()
plt.close(figure)

print(
    f"Build checks passed: {len(managed)} project distributions; "
    f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, "
    f"cuDNN {torch.backends.cudnn.version()}, TensorRT {tensorrt.__version__}."
)
