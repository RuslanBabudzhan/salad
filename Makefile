-include .env

IMAGE ?= salad-training
HF_CACHE ?= $(HOME)/.cache/huggingface

DOCKER_RUN = docker run --rm --gpus all --shm-size=16g \
	--user "$(shell id -u):$(shell id -g)" \
	-e HF_HOME=/cache/huggingface \
	-e HF_TOKEN \
	-v "$(CURDIR):/workspace" \
	-v "$(DATA_ROOT):/data:ro" \
	-v "$(HF_CACHE):/cache/huggingface"

.PHONY: build train shell test

build:
	docker build -t $(IMAGE) .

shell:
	$(DOCKER_RUN) -it $(IMAGE) bash

test:
	$(DOCKER_RUN) $(IMAGE) python -m unittest discover -s tests -v
