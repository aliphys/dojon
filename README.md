# dojon

**dojon** is a machine-learning training dojo for the NVIDIA Jetson Orin Nano.
It provides a reproducible, GPU-enabled environment for learning, experimenting,
and getting practical ML work done on the edge. The name combines **do** with
**JON** (Jetson Orin Nano), while *dojon* is Swedish for “the dojo.” 😎

The `dojon` container image extends
[`whitesscott/l4t-jetpack:r39.2.1`](https://hub.docker.com/r/whitesscott/l4t-jetpack)
with PyTorch, Torchvision, TorchAudio, ONNX, Pillow, SciPy, tqdm, Matplotlib,
Seaborn, JupyterLab, Weights & Biases, and jtop. The base image remains
the source of CUDA, cuDNN, TensorRT, OpenCV, and the JetPack multimedia stack.

- [Quick start](#quick-start)
- [Common commands](#common-commands)
- [Verification commands](#verification-commands)
- [Optional tools](#optional-tools)
- [Troubleshooting](#troubleshooting)
- [Dependency policy](#dependency-policy)
- [Updating dependencies](#updating-dependencies)

## Tested stack

| Component | Version |
| --- | --- |
| Board | Jetson Orin Nano |
| JetPack / L4T | 7.2.1 / 39.2.1 |
| Python | 3.12 |
| CUDA | 13.2 |
| cuDNN | 9.20.0.46 |
| TensorRT | 10.16.2 |
| PyTorch | 2.13.0+cu132 |
| Torchvision / TorchAudio | 0.28.0+cu132 / 2.11.0+cu132 |
| Triton | 3.7.1 |
| SciPy / tqdm | 1.17.1 / 4.70.1 |
| Matplotlib / Seaborn | 3.11.2 / 0.13.2 |
| ONNX / ONNX Runtime | 1.23.0 / 1.30.0 |
| JupyterLab | 4.6.4 |
| W&B | 0.30.0 |
| jtop | 7.2.2 |

PyTorch 2.13 is used because it targets the cuDNN 9.20 family supplied by this
JetPack image. The newer PyTorch 2.14 build requires cuDNN 9.24. TensorRT is
always inherited from the base image and is never upgraded with pip.

## Quick start

Run these commands on an ARM64 Jetson with JetPack 7.2.1, Docker, and NVIDIA
Container Runtime.

Check the host:

```bash
cat /etc/nv_tegra_release
uname -m
docker info --format '{{json .Runtimes}}'
```

Expect R39 revision 2.1, `aarch64`, and an `nvidia` runtime.

Build the image from this repository:

```bash
docker build --pull --platform linux/arm64 \
  -f Dockerfile.jetpack_721 -t dojon:jp721 .
```

Start an interactive shell:

```bash
docker run --rm -it --runtime=nvidia --shm-size=1g \
  -v "$PWD":/workspace -w /workspace \
  dojon:jp721
```

Verify CUDA, PyTorch, ONNX, and TensorRT:

```bash
docker run --rm --runtime=nvidia --shm-size=256m dojon:jp721 \
  python3 /opt/image-checks/verify-gpu.py
```

Use `sudo docker` if your account does not have Docker access.

## Common commands

### Start JupyterLab

```bash
docker run --rm -it --runtime=nvidia --shm-size=1g \
  -e ENABLE_JUPYTER=1 -e JUPYTER_PORT=8888 \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD":/workspace -w /workspace \
  dojon:jp721
```

`ENABLE_JUPYTER=1` starts the JupyterLab server. A Python kernel starts when a
notebook is opened. Jupyter is disabled by default; set `ENABLE_JUPYTER=0`
explicitly if desired. Accepted values are `0/1`, `false/true`, and `no/yes`.

`JUPYTER_PORT` defaults to `8888`. If it is changed, update both sides of
`-p HOST_PORT:CONTAINER_PORT`. Authentication remains enabled; use the token
printed in the container logs.

To access Jupyter from another computer:

```bash
ssh -N -L 8888:127.0.0.1:8888 YOUR_USER@JETSON_HOST
```

Then open <http://localhost:8888>.

### Run a command without Jupyter

```bash
docker run --rm --runtime=nvidia -e ENABLE_JUPYTER=0 \
  dojon:jp721 python3 -c 'import torch; print(torch.__version__)'
```

### Mount a project

```bash
docker run --rm -it --runtime=nvidia --shm-size=1g \
  -v /path/to/project:/workspace -w /workspace \
  dojon:jp721
```

Changes under `/workspace` are stored on the host. Increase shared memory only
when the workload and available Orin Nano RAM justify it.

## Verification commands

Run hardware checks separately to keep memory use low:

```bash
# PyTorch CUDA, cuDNN, ONNX Runtime, and TensorRT
docker run --rm --runtime=nvidia --shm-size=256m dojon:jp721 \
  python3 /opt/image-checks/verify-gpu.py

# torch.cond early exit through PyTorch, ONNX, and TensorRT on CUDA
docker run --rm --runtime=nvidia --shm-size=256m dojon:jp721 \
  python3 /opt/image-checks/verify-torch-cond.py

# Jupyter server and notebook kernel
docker run --rm --runtime=nvidia --shm-size=256m dojon:jp721 \
  python3 /opt/image-checks/verify-tools.py notebook

# W&B offline logging
docker run --rm dojon:jp721 \
  python3 /opt/image-checks/verify-tools.py wandb

# jtop connection to the host service
docker run --rm --runtime=nvidia \
  --mount type=bind,src=/run/jtop.sock,dst=/run/jtop.sock \
  dojon:jp721 python3 /opt/image-checks/verify-tools.py jtop
```

The conditional test should include:

```text
PASS torch.cond selects both branches on CUDA
PASS TensorRT runs both outputs on CUDA and skips the large branch on early exit
```

The image build runs the version and import checks automatically. Runtime checks
exit with a nonzero status when a required check fails.

## Optional tools

### jtop

Install and start
[`jetson-stats`](https://rnext.it/jetson_stats/) on the Jetson host and verify
that `/run/jtop.sock` exists. The container pins jtop 7.2.2 to the upstream
commit matching the tested host service.

```bash
docker run --rm -it --runtime=nvidia \
  --mount type=bind,src=/run/jtop.sock,dst=/run/jtop.sock \
  dojon:jp721 jtop
```

Add the same socket mount to shell or Jupyter commands when they need jtop.

### Weights & Biases

Run `wandb login` inside the container for online use. For local logging:

```bash
docker run --rm -it -e WANDB_MODE=offline \
  -v "$PWD":/workspace -w /workspace \
  dojon:jp721
```

Do not store API keys in the image.

### X11

For a Jetson desktop using X11/Xwayland, add:

```bash
-e DISPLAY="$DISPLAY" \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e XAUTHORITY=/tmp/.Xauthority \
-v "$XAUTHORITY":/tmp/.Xauthority:ro
```

Graphical behavior depends on the host desktop and its X authorization setup.

## Troubleshooting

### PyTorch reports unsupported Orin compute capability

PyTorch 2.13 may warn about Orin compute capability 8.7 even when its CUDA kernels
execute successfully. The verification scripts run real matrix multiplication,
convolution, attention, Triton, and TensorRT workloads instead of relying only on
`torch.cuda.is_available()`. NVIDIA discusses the warning in this
[JetPack 7.2 thread](https://forums.developer.nvidia.com/t/how-do-i-correctly-install-pytorch-on-jetpack-7-2/372773/7).

### pip reports missing CUDA or cuDNN packages

`pip check` reports missing `cuda-toolkit` and `nvidia-cudnn-cu13` wheel
metadata because JetPack supplies those libraries as system packages. Do not
install replacement CUDA, cuDNN, or TensorRT wheels to silence the warning.

### jtop raises `KeyError: 'online'`

The tested host exposes a VIC engine entry without the field expected by jtop's
compact statistics API. Raw GPU and memory readings work. This is a known jtop
compatibility issue; the image does not patch the host service.

### The container runs out of memory

Run GPU and notebook tests in separate containers, close unused desktop
applications, and avoid unnecessarily large `--shm-size` values. A combined
GPU and notebook test exceeded a 1.5 GiB memory limit during validation.

### Jupyter is unreachable

Confirm that `ENABLE_JUPYTER=1` is set, the host and container ports match
`JUPYTER_PORT`, and SSH port forwarding remains open when connecting remotely.

## Included files

| File | Purpose |
| --- | --- |
| `Dockerfile.jetpack_721` | Builds the PyTorch extension image |
| `requirements-torch.lock` | PyTorch, Triton, and supplemental GPU libraries |
| `requirements-python.lock` | ONNX, Jupyter, W&B, jtop, and their dependencies |
| `scripts/start-container.sh` | Selects shell or Jupyter startup mode |
| `scripts/tested-versions.json` | Expected package versions |
| `scripts/verify-build.py` | Build-time version, dependency, and import checks |
| `scripts/verify-gpu.py` | CUDA, cuDNN, ONNX, and TensorRT runtime checks |
| `scripts/verify-torch-cond.py` | Conditional early-exit check on CUDA |
| `scripts/verify-tools.py` | Jupyter, W&B, and jtop checks |

## Dependency policy

The NVIDIA stack is the compatibility anchor:

- CUDA, cuDNN, TensorRT, OpenCV, and JetPack libraries come from the base image.
- The official CUDA 13.2 Torchvision wheel keeps its compiled operators and
  private CUDA runtime namespaced inside the package; it does not replace the
  system CUDA runtime, cuDNN, or TensorRT.
- PyTorch and every added Python dependency are pinned by exact version or
  immutable source URL.
- Pip installs use `--no-deps` so dependency resolution cannot replace the
  NVIDIA system stack.
- NCCL 2.29.7, cuSPARSELt 0.8.1, and NVSHMEM 3.4.5 are added because PyTorch
  requires them and the base image does not provide them.

The former Jetson AI Lab `/sbsa/cu132` URL redirected to generic PyPI during
testing and selected a CUDA 13.0 wheel. This image therefore uses the official
PyTorch CUDA 13.2 ARM64 wheel directly. NVIDIA confirms that upstream SBSA wheels
work on Orin with JetPack 7.2 in this
[forum response](https://forums.developer.nvidia.com/t/how-do-i-correctly-install-pytorch-on-jetpack-7-2/372773/5).

Ultralytics is not included.

## Updating dependencies

Treat the base image and GPU dependencies as one compatibility set:

1. Select a base version and compatible PyTorch, Triton, NCCL, cuSPARSELt, and
   NVSHMEM versions.
2. Keep CUDA, cuDNN, and TensorRT supplied by the base image.
3. Update exact versions or immutable URLs in the appropriate lock file.
4. Update `scripts/tested-versions.json`.
5. Rebuild and run every verification command above on the Jetson.

BuildKit caches pip downloads outside the image. Reusing the same Docker builder
avoids downloading large PyTorch and GPU packages again.

## Validated behavior

The image was tested on Jetson Orin Nano with L4T 39.2.1:

- PyTorch CUDA matrix multiplication, cuDNN convolution, FP16, attention, and
  `torch.compile` with Triton.
- ONNX export and validation, ONNX Runtime ARM64 CPU inference, and TensorRT GPU
  inference.
- `torch.cond` early and full branches in PyTorch and TensorRT on CUDA.
  TensorRT profiling confirmed that the large FC branch was skipped on early exit.
- Authenticated JupyterLab startup and GPU execution from a notebook kernel.
- W&B offline metric logging.
- jtop host connection and raw GPU and memory readings.

Runtime library paths confirmed that CUDA 13.2, cuDNN 9.20, and TensorRT 10.16.2
came from the base image.

## Upstream

Thanks to whitesscott for the published JetPack base image:

```bash
docker image inspect whitesscott/l4t-jetpack:r39.2.1
docker history --no-trunc whitesscott/l4t-jetpack:r39.2.1
```

Original NVIDIA container sources:
<https://gitlab.com/nvidia/container-images/l4t-jetpack>.
Retain applicable upstream license notices when redistributing source or images.
