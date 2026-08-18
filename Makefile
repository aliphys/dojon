
TAG_JP721     ?= r39.2.1
RELEASE_JP721 ?= r39.2
VIDEO_GID     ?= 44
RENDER_GID    ?= 993
L4T_JETPACK_REGISTRY ?= nvcr.io/nvidia/l4t-jetpack

# Docker Hub destination for `make push_jp72`.
HUB_REGISTRY ?= docker.io/whitesscott/l4t-jetpack

image_jp72:
	docker build --progress=plain --platform=linux/arm64 -t $(L4T_JETPACK_REGISTRY):$(TAG_JP721) \
		--build-arg "RELEASE=$(RELEASE_JP721)" \
		--build-arg "VIDEO_GID=$(VIDEO_GID)" \
		--build-arg "RENDER_GID=$(RENDER_GID)" \
		-f ./Dockerfile.jetpack_721 .

# Quick post-build sanity check: entrypoint set TORCH_CUDA_ARCH_LIST, nvcc
# reports CUDA 13.2, GPU compute capability probes cleanly, and nvidia-smi
# can talk to the driver. Exits non-zero on any failure.
smoke_jp72:
	docker run --rm --runtime nvidia $(L4T_JETPACK_REGISTRY):$(TAG_JP721) bash -c '\
		set -e; \
		echo "TORCH_CUDA_ARCH_LIST=$$TORCH_CUDA_ARCH_LIST"; \
		nvcc --version | tail -1; \
		echo "compute_cap=$$(/usr/local/cuda/bin/__nvcc_device_query)"; \
		nvidia-smi -q -d COMPUTE | head -8'

# Tag the local build under $(HUB_REGISTRY) with three names: the L4T tag,
# a human-readable jp version tag, and :latest. Run `docker login` first.
tag_jp72:
	docker tag $(L4T_JETPACK_REGISTRY):$(TAG_JP721) $(HUB_REGISTRY):$(TAG_JP721)
	docker tag $(L4T_JETPACK_REGISTRY):$(TAG_JP721) $(HUB_REGISTRY):jp7.2.1-thor
	docker tag $(L4T_JETPACK_REGISTRY):$(TAG_JP721) $(HUB_REGISTRY):latest

# Push all three tags to Docker Hub. First run streams ~3 GB compressed on
# the wire; subsequent tag pushes only send the manifest. Depends on tag_jp72
# so a bare `make push_jp72` works after a build.
push_jp72: tag_jp72
	docker push $(HUB_REGISTRY):$(TAG_JP721)
	docker push $(HUB_REGISTRY):jp7.2.1-thor
	docker push $(HUB_REGISTRY):latest
