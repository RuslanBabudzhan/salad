IMAGE ?= salad-training
DATA_ROOT ?= /mnt/e/D_WORK/Localizer/data/datasets/processed/MegaLoc/train
DATA ?= /data/configs/v1_d_vanilla.yaml
CONFIG ?= configs/train_megaloc.yaml
LOGS ?= logs
WORKERS ?= 4
PRECISION ?= 16-mixed
HF_CACHE ?= $(HOME)/.cache/huggingface

DOCKER_RUN = docker run --rm --gpus all --shm-size=16g \
	--user "$(shell id -u):$(shell id -g)" \
	-e HF_HOME=/cache/huggingface \
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
