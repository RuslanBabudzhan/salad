-include .env

IMAGE ?= salad-training

HF_CACHE ?= $(HOME)/.cache/huggingface
UV_CACHE ?= $(HOME)/.cache/uv


DOCKER_RUN = docker run --rm --gpus all --shm-size=16g \
    --user "$(shell id -u):$(shell id -g)" \
    -e HOME=/workspace \
    -e HF_HOME=/cache/huggingface \
    -e HF_TOKEN \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v "$(CURDIR):/workspace" \
    -v "$(DATA_ROOT):/data:ro" \
    -v "$(HF_CACHE):/cache/huggingface"

.PHONY: build train shell test

build:
	docker build -t $(IMAGE) .

shell:
	mkdir -p "$(HF_CACHE)" "$(UV_CACHE)"
	$(DOCKER_RUN) -it $(IMAGE) bash

test:
	mkdir -p "$(HF_CACHE)" "$(UV_CACHE)"
	$(DOCKER_RUN) $(IMAGE) python -m unittest discover -s tests -v
