# Image details

Use cases addressed by this image:
1) Ablility to build and run CUDA samples
2) Ablility to build and run CuDNN samples
3) Ablility to build and run TensorRT samples
4) Ablility to build and run VPI samples
5) Ablility to build and run OpenCV samples

# JetPack 7.2.1 image (L4T r39.2.1, Ubuntu 24.04, Thor / Orin)

Build:
```bash
make image_jp72

# On a host where render GID differs from Thor default (993):
make image_jp72 RENDER_GID=$(getent group render | cut -d: -f3)
```

Quick sanity check after build:
```bash
make smoke_jp72
```

Run:
```bash
docker run -it --rm --name jetpack --network host \
  --runtime=nvidia --gpus=all \
  --privileged --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  nvcr.io/nvidia/l4t-jetpack:r39.2.1
```
Substitute `whitesscott/l4t-jetpack:jp7.2.1-thor` for the Docker Hub-pulled image instead of the locally-built tag.

Confirmed working on NVIDIA Thor (SM 11.0):

- **CUDA 13.2.2** — `nvcc` compiles + on-device kernel executes (`atomicAdd` returns expected 65536)
- **cuDNN 9.20.0.46** — `cudnnGetVersion()` returns 92000 via real link
- **TensorRT 10.16.2** — `getInferLibVersion()` returns 101602; builder resources present for sm_110, sm_100, sm_120, sm_75-89
- **cuDLA 13.2** — headers + libs installed (runtime not exercised)
- **VPI 4.1.4** — Python `import vpi` succeeds
- **OpenCV 4.8.0** — Python `import cv2` succeeds
- **PyTorch** — installable via `pip install --index-url https://pypi.jetson-ai-lab.io/sbsa/cu132 torch`; `torch.mm` on `device="cuda"` runs on Thor
- **Multimedia** — GStreamer 1.24.2 + NVIDIA plugins (`nvarguscamerasrc`, `nvv4l2camerasrc`, `nvv4l2decoder`, `nvv4l2h264enc`, `nvv4l2h265enc`, `nvvidconv`); H.264 hardware encode with NVMM zero-copy verified end-to-end
- **Auto-arch** — entrypoint sets `TORCH_CUDA_ARCH_LIST` from `__nvcc_device_query` (Thor → `11.0`, Orin → `8.7`)

Also included: `build-essential`, `git`, `rsync`, `openssh-client`, `xauth`, `python3` + `pip` + `numpy`, locale set to `en_US.UTF-8`, `render`/`video` groups pre-created at Thor default GIDs.

Not included (add downstream if needed): DeepStream, Triton, Isaac ROS, DALI, FFmpeg.

## Inspect the published image

The Docker Hub overview intentionally doesn't paste the full Dockerfile — it goes stale on every rebuild. Inspect the actual pushed image directly:

```bash
docker pull whitesscott/l4t-jetpack:latest
docker history whitesscott/l4t-jetpack:latest --no-trunc   # every RUN/COPY layer with its command
docker inspect whitesscott/l4t-jetpack:latest | jq '.[0].Config'   # ENV, ENTRYPOINT, CMD, exposed ports
```

These always reflect the manifest of the exact image you're about to run — no drift possible.

# Container path
Update the container path $L4T_JETPACK_REGISTRY as applicable

# Building with new Jetpack release
Bump the TAG variable to appropriate value

# Size Estimates
* ~10.4GB if composed of all developer packages
* ~4.7GB if composed of all runtime packages

# Running container and samples
See NGC page for Jetpack container at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack

```
