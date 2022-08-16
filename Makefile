
TAG     ?= r35.1.0
L4T_JETPACK_REGISTRY ?= "nvcr.io/nvidia/l4t-jetpack"

image:
	docker build -t $(L4T_JETPACK_REGISTRY):$(TAG) \
		--build-arg "TAG=$(TAG)" \
		-f ./Dockerfile.jetpack .
