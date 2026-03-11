.PHONY: build-gpu build-tpu run-gpu run-tpu

# Variables
IMAGE_NAME = afanthomme/jax-draw

build-gpu:
	docker build -f Dockerfile.gpu -t $(IMAGE_NAME):gpu-latest .
	
build-tpu:
	docker build -f Dockerfile.tpu -t $(IMAGE_NAME):tpu-latest .

