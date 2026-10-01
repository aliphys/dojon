# dojon

**dojon** is a machine-learning training dojo for the NVIDIA Jetson Orin Nano:
a reproducible, GPU-enabled environment for learning, experimenting, and getting
practical ML work done at the edge. The name combines **do** with **JON**
(Jetson Orin Nano), while *dojon* is Swedish for “the dojo.” 😎

The image extends NVIDIA's
[`nvcr.io/nvidia/pytorch:26.09-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
container with project tools including ONNX Runtime, SciPy, Matplotlib,
Weights & Biases, and jtop. PyTorch, CUDA, cuDNN, TensorRT, JupyterLab, and
other framework libraries come from the NGC base image; project package lists
do not reinstall those components.

## Compatibility

Tested on a Jetson Orin Nano with JetPack/L4T R39.2.1. The NGC tag publishes an
ARM64 image and the GPU, ONNX, TensorRT, and `torch.cond` checks pass on the
device. Camera, display, and multimedia integrations remain separate host
integration concerns.

| Component | Version/status |
| --- | --- |
| NGC image | `nvcr.io/nvidia/pytorch:26.09-py3` (ARM64 manifest confirmed) |
| Host JetPack / L4T | R39.2.1 / JetPack 7.2.1 |
| Python / CUDA / PyTorch | 3.12 / 13.4 / 2.14.0a0+b2c75dd062.nv26.09 |
| TensorRT / cuDNN | 11.3.0.99 / 9.26 |
| JupyterLab | Supplied by NGC |
| ONNX / ONNX Runtime / W&B / jtop | Project tool set |

The NGC image owns the framework and CUDA software stack. The Jetson host still
provides the kernel driver and device integration through NVIDIA Container
Runtime. Host/container driver compatibility and GPU execution must be checked
on the Orin Nano.

## Quick start

Build for ARM64 and run on a Jetson with Docker and NVIDIA Container Runtime:

```bash
docker build --pull --platform linux/arm64 -t dojon:26.09 .

docker run --rm -it --runtime=nvidia --shm-size=1g \
  -v "$PWD":/workspace -w /workspace \
  dojon:26.09
```

Use `sudo docker` if your account does not have Docker access. NGC access may
require accepting NVIDIA's container license and logging in with `docker login
nvcr.io`.

### JupyterLab

```bash
docker run --rm -it --runtime=nvidia --shm-size=1g \
  -e ENABLE_JUPYTER=1 -e JUPYTER_PORT=8888 \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD":/workspace -w /workspace \
  dojon:26.09
```

Authentication remains enabled; use the token printed in the container logs.
The server listens only on the Jetson's loopback interface. From another
computer, create an SSH tunnel to forward the local port:

```bash
ssh -i ~/.ssh/jetson_build \
  -N -L 8888:127.0.0.1:8888 \
  jetson@jetson-001.aliphys.org
```

Leave this SSH command running. On the computer, open the token URL printed by
Jupyter in a browser, changing the host and port to
`http://127.0.0.1:8888/?token=YOUR_TOKEN`.

#### Connect from VS Code

Install the **Jupyter** extension in VS Code, then:

1. Start the container and SSH tunnel above.
2. Open a `.ipynb` file in VS Code.
3. Open the Command Palette with `Ctrl+Shift+P` and run **Jupyter: Specify
   local or remote Jupyter server for connections**.
4. Select **Existing** and enter the complete token URL:
   `http://127.0.0.1:8888/?token=YOUR_TOKEN`.
5. Select the Python kernel offered by the remote Jupyter server.

Test the connection in a notebook cell:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.ones(1, device="cuda"))
```

The expected output includes `True`, `Orin`, and a CUDA tensor. The SSH tunnel
must remain open while VS Code uses the kernel.

### Optional host integrations

For jtop, install and start
[`jetson-stats`](https://rnext.it/jetson_stats/) on the host, then mount its
socket:

```bash
docker run --rm -it --runtime=nvidia \
  --mount type=bind,src=/run/jtop.sock,dst=/run/jtop.sock \
  dojon:26.09 jtop
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

The build checks project additions, core imports, and plotting. Run hardware and
tool checks separately to limit memory use:

```bash
# PyTorch CUDA, Torchvision, ONNX Runtime, and TensorRT
docker run --rm --runtime=nvidia --shm-size=256m dojon:26.09 \
  python3 /opt/image-checks/verify-gpu.py

# torch.cond early exit through PyTorch, ONNX, and TensorRT
docker run --rm --runtime=nvidia --shm-size=256m dojon:26.09 \
  python3 /opt/image-checks/verify-torch-cond.py

# Jupyter server and GPU-backed notebook kernel
docker run --rm --runtime=nvidia --shm-size=256m dojon:26.09 \
  python3 /opt/image-checks/verify-tools.py notebook

# W&B offline logging
docker run --rm dojon:26.09 \
  python3 /opt/image-checks/verify-tools.py wandb

# jtop host connection
docker run --rm --runtime=nvidia \
  --mount type=bind,src=/run/jtop.sock,dst=/run/jtop.sock \
  dojon:26.09 python3 /opt/image-checks/verify-tools.py jtop
```

## FAQ and troubleshooting

**`jtop` says it cannot access `jtop.service`:** Check the host service and
socket permissions:

```bash
systemctl status jtop.service
ls -l /run/jtop.sock
```

The socket should normally be owned by `root:jtop` with group read/write
permissions. Add the login user to the `jtop` group, then start a new login
session or reboot:

```bash
sudo usermod -aG jtop "$USER"
sudo reboot
```

After logging in again, confirm the group is active with `groups`, then run
`jtop` or `sudo jtop`. When using the container integration, mount the same host
socket at `/run/jtop.sock`.

## Other troubleshooting

**CUDA or driver errors:** The Jetson host supplies the kernel driver while NGC
provides CUDA userspace. Confirm the host JetPack/L4T version and NVIDIA
Container Runtime setup, then run `verify-gpu.py` on the device. A successful
build does not validate runtime compatibility.

**pip reports dependency conflicts:** NGC pins framework packages through
`/etc/pip/constraint.txt`. Keep its PyTorch/CUDA stack intact; adjust
constraints deliberately if a project tool requires a conflicting package.

**Out of memory:** Run GPU and notebook checks in separate containers and avoid
oversized `--shm-size` values.

## Dependency policy

- `requirements-python.lock` pins project-added Python tools; installation uses
  `--no-deps` to avoid changing NGC's preinstalled stack.
- PyTorch and CUDA dependencies come from the pinned NGC base image and are not
  duplicated in project package lists.
- `verify-build.py` checks project additions and imports core NGC packages.
  Recheck dependency closure when changing either the base tag or tool pins.

When updating the NGC tag or project tools, rebuild and run every verification
command on the Jetson.

## License and upstream

dojon is licensed under the [MIT License](LICENSE). The base image is maintained
by NVIDIA and published through NGC. Retain all applicable upstream license
notices when redistributing images.
