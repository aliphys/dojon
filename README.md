# l4t-jetpack — JetPack 7.2.1 image (L4T r39.2.1, Ubuntu 24.04, Thor / Orin)

A working, corrected build of the NVIDIA L4T JetPack developer image for
**JetPack 7.2.1 (L4T r39.2.1)** on Jetson AGX Thor and Orin devkits.
Ubuntu 24.04 Noble, aarch64. Based on `nvcr.io/nvidia/base/ubuntu:24.04`
plus the `repo.download.nvidia.com/jetson/{common,som,ffmpeg} r39.2` apt repos.

Published image: [`whitesscott/l4t-jetpack`](https://hub.docker.com/r/whitesscott/l4t-jetpack) on Docker Hub.

## Build

```bash
make image_jp72

# On a host where render GID differs from Thor default (993):
make image_jp72 RENDER_GID=$(getent group render | cut -d: -f3)
```

## Sanity check

```bash
make smoke_jp72
```

## Run
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

## Push to Docker Hub

```bash
docker login
make push_jp72   # tags r39.2.1, jp7.2.1-thor, latest and pushes all three
```

Override the Hub namespace on the fly:
```bash
make push_jp72 HUB_REGISTRY=docker.io/your-org/l4t-jetpack
```

## Confirmed working on NVIDIA Thor (SM 11.0)

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

## Image size

- ~10 GB uncompressed (developer packages: CUDA toolkit + cuDNN + TRT + VPI + OpenCV all with headers).
- ~3-4 GB compressed on the wire when pulling from Docker Hub.

## Upstream

Original NVIDIA source: https://gitlab.com/nvidia/container-images/l4t-jetpack
NGC catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack

Pull upstream changes into this fork:
```bash
git fetch upstream
git log HEAD..upstream/master --oneline   # see what's new
git merge upstream/master                  # or rebase
```
