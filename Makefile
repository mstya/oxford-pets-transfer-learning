SHELL := /bin/bash

# Single source of truth for which experiments/<N> checkpoint gets served -
# see configs/serving_config.json. Everything Docker-related (Dockerfile,
# docker-compose.yml) takes this as a build-arg instead of hardcoding it.
MODEL_EXPERIMENT := $(shell python3 -c "import json; print(json.load(open('configs/serving_config.json'))['model_experiment'])")

.PHONY: train serve build

train:
	python -m src.train

## Build the API image (with the configured checkpoint baked in) and run it.
serve:
	MODEL_EXPERIMENT=$(MODEL_EXPERIMENT) docker compose up --build

## Build the API image without starting it.
build:
	MODEL_EXPERIMENT=$(MODEL_EXPERIMENT) docker compose build
