# dojon

**dojon** is a machine-learning training dojo for the NVIDIA Jetson Orin Nano:
a reproducible, GPU-enabled environment for learning, experimenting, and getting
practical ML work done at the edge. The name combines **do** with **JON**
(Jetson Orin Nano), while *dojon* is Swedish for “the dojo.” 😎

The image extends
[`whitesscott/l4t-jetpack:r39.2.1`](https://hub.docker.com/r/whitesscott/l4t-jetpack)
with PyTorch, Torchvision, TorchAudio, ONNX, SciPy, Matplotlib, JupyterLab,
Weights & Biases, and jtop. CUDA, cuDNN, TensorRT, OpenCV, and the multimedia
stack remain supplied by JetPack.

## Compatibility

Tested on Jetson Orin Nano with:

| Component | Version |
| --- | --- |
| JetPack / L4T | 7.2.1 / 39.2.1 |
| Python / CUDA | 3.12 / 13.2 |
| cuDNN / TensorRT | 9.20.0.46 / 10.16.2 |
| PyTorch | 2.13.0+cu132 |
| Torchvision / TorchAudio | 0.28.0+cu132 / 2.11.0+cu132 |
| ONNX / ONNX Runtime | 1.23.0 / 1.30.0 |
| JupyterLab / W&B / jtop | 4.6.4 / 0.30.0 / 7.2.2 |

PyTorch 2.13 matches the cuDNN 9.20 family in this JetPack image. CUDA, cuDNN,
TensorRT, and OpenCV are never replaced with pip packages.

## Quick start

Run on an ARM64 Jetson with JetPack 7.2.1, Docker, and NVIDIA Container Runtime.

```bash
docker build --pull --platform linux/arm64 -t dojon:jp721 .

docker run --rm -it --runtime=nvidia --shm-size=1g \
  -v "$PWD":/workspace -w /workspace \
  dojon:jp721
```

Use `sudo docker` if your account does not have Docker access.

### JupyterLab

```bash
docker run --rm -it --runtime=nvidia --shm-size=1g \
  -e ENABLE_JUPYTER=1 -e JUPYTER_PORT=8888 \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD":/workspace -w /workspace \
  dojon:jp721
```

Authentication remains enabled; use the token printed in the container logs.
For access from another computer, forward the port over SSH:

```bash
ssh -N -L 8888:127.0.0.1:8888 YOUR_USER@JETSON_HOST
```

### Optional host integrations

For jtop, install and start
[`jetson-stats`](https://rnext.it/jetson_stats/) on the host, then mount its
socket:

```bash
docker run --rm -it --runtime=nvidia \
  --mount type=bind,src=/run/jtop.sock,dst=/run/jtop.sock \
  dojon:jp721 jtop
```

For W&B online use, run `wandb login` inside the container. Do not store API
keys in the image.

For a Jetson desktop using X11/Xwayland, add the following arguments:

```bash
-e DISPLAY="$DISPLAY" \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e XAUTHORITY=/tmp/.Xauthority \
-v "$XAUTHORITY":/tmp/.Xauthority:ro
```

## Verification

The build checks locked versions, dependency closure, imports, and plotting.
Run hardware and tool checks separately to limit memory use:

```bash
# PyTorch CUDA, Torchvision, TorchAudio, ONNX Runtime, and TensorRT
docker run --rm --runtime=nvidia --shm-size=256m dojon:jp721 \
  python3 /opt/image-checks/verify-gpu.py

# torch.cond early exit through PyTorch, ONNX, and TensorRT
docker run --rm --runtime=nvidia --shm-size=256m dojon:jp721 \
  python3 /opt/image-checks/verify-torch-cond.py

# Jupyter server and GPU-backed notebook kernel
docker run --rm --runtime=nvidia --shm-size=256m dojon:jp721 \
  python3 /opt/image-checks/verify-tools.py notebook

# W&B offline logging
docker run --rm dojon:jp721 \
  python3 /opt/image-checks/verify-tools.py wandb

# jtop host connection
docker run --rm --runtime=nvidia \
  --mount type=bind,src=/run/jtop.sock,dst=/run/jtop.sock \
  dojon:jp721 python3 /opt/image-checks/verify-tools.py jtop
```

## Troubleshooting

**Unsupported compute capability warning:** PyTorch may warn about Orin compute
capability 8.7 even when CUDA works. Use `verify-gpu.py`, which executes real GPU
workloads instead of relying only on `torch.cuda.is_available()`.

**pip reports missing CUDA or cuDNN packages:** JetPack supplies these as system
packages, so their wheel metadata is intentionally absent. Do not install pip
replacements to silence the warning.

**jtop raises `KeyError: 'online'`:** The tested host exposes a VIC entry without
the field expected by jtop's compact statistics API. Raw GPU and memory readings
still work.

**Out of memory:** Run GPU and notebook checks in separate containers and avoid
oversized `--shm-size` values.

## Dependency policy

- Exact versions or immutable source URLs live in the two lock files.
- PyTorch and its large GPU dependencies are installed separately for better
  Docker build caching.
- Pip uses `--no-deps`; the lock files explicitly contain added dependencies.
- `verify-build.py` reads the lock files directly, so there is one source of
  truth for dependency versions.

When updating dependencies, change the lock files, rebuild, and run every
verification command on the Jetson.

## License and upstream

dojon is licensed under the [MIT License](LICENSE).

The base image is maintained by whitesscott. Original NVIDIA container sources
are available at <https://gitlab.com/nvidia/container-images/l4t-jetpack>.
Retain all applicable upstream license notices when redistributing images.
