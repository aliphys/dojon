
TAG     ?= r36.4.0
RELEASE ?= r36.4
L4T_JETPACK_REGISTRY ?= "nvcr.io/nvidia/l4t-jetpack"

image_jp5:
	docker build --platform=linux/arm64 -t $(L4T_JETPACK_REGISTRY):$(TAG) \
		--build-arg "TAG=$(TAG)" \
		-f ./Dockerfile.jetpack_5 .

image_jp6:
	docker build --platform=linux/arm64 -t $(L4T_JETPACK_REGISTRY):$(TAG) \
		--build-arg "RELEASE=$(RELEASE)" \
		-f ./Dockerfile.jetpack_6 .
