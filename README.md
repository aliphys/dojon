# Build instructions for Jetson

```bash
sudo make image
```

# Image details

Use cases addressed by this image:
1) Ablility to build and run CUDA samples
2) Ablility to build and run CuDNN samples
3) Ablility to build and run TensorRT samples
4) Ablility to build and run VPI samples
5) Ablility to build and run OpenCV samples

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

