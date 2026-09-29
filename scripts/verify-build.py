"""Verify locked versions, Python dependencies and base imports without a GPU."""

import importlib
import importlib.metadata as metadata
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

expected = json.loads(Path(__file__).with_name("tested-versions.json").read_text())
for name, version in expected.items():
    actual = metadata.version(name)
    if actual != version:
        raise RuntimeError(f"{name}: expected {version}, found {actual}")

# These two wheel requirements are intentionally satisfied by JetPack packages.
# Their pip counterparts must never shadow the base CUDA/cuDNN libraries.
system_supplied = {"cuda-toolkit", "nvidia-cudnn-cu13"}
installed = {canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions()}
for name in system_supplied:
    if name in installed:
        raise RuntimeError(f"Unexpected replacement of a base library: {name}")
for name in expected:
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
    "torch", "triton", "tensorrt", "cv2", "numpy", "onnx", "onnxruntime",
    "jtop", "wandb", "jupyterlab", "ipykernel",
):
    importlib.import_module(name)
import cv2
import torch
assert cv2.__version__ == "4.8.0", cv2.__version__
assert torch.version.cuda == "13.2", torch.version.cuda
print(f"Build checks passed: {len(expected)} locked/base distributions; PyTorch {torch.__version__}.")
