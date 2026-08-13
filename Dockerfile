# syntax=docker/dockerfile:1
# Serves the Oxford Pets breed classifier API. The trained checkpoint is baked
# into the image at build time (see MODEL_EXPERIMENT below) - training itself
# happens natively on the host (`make train` / `python -m src.train`), not in
# this image.
FROM python:3.12-slim

# Which experiments/<N> checkpoint to bake into the image. No default here on
# purpose: configs/serving_config.json is the single source of truth for this
# number, and `make serve` / docker-compose.yml read it from there - this ARG
# just exists for `docker build` to receive it. Pass it explicitly if you're
# invoking `docker build` directly, e.g. --build-arg MODEL_EXPERIMENT=37.
ARG MODEL_EXPERIMENT

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_EXPERIMENT_DIR=/app/model

WORKDIR /app

# CPU-only torch/torchvision wheels - the container has no GPU/MPS access, and
# the CPU build is a fraction of the size of the default CUDA-enabled one.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.12.0" "torchvision>=0.27.0"

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/ ./src/
COPY configs/ ./configs/

# A single RUN with a bind mount (rather than
# `COPY experiments/${MODEL_EXPERIMENT}/...`) so a missing/blank
# MODEL_EXPERIMENT fails with a clear, custom message - a plain COPY would
# just say "not found". classes.json is optional: older experiments (trained
# before it existed) simply won't have one, and src/api.py falls back to
# OXFORD_PETS_CLASSES then.
RUN --mount=type=bind,source=experiments,target=/build/experiments <<EOF
set -eu
if [ -z "${MODEL_EXPERIMENT:-}" ]; then
    echo "ERROR: --build-arg MODEL_EXPERIMENT is required. Run 'make serve' (reads configs/serving_config.json), or pass --build-arg MODEL_EXPERIMENT=<experiment number> explicitly." >&2
    exit 1
fi
if [ ! -f "/build/experiments/${MODEL_EXPERIMENT}/model.pt" ]; then
    echo "ERROR: experiments/${MODEL_EXPERIMENT}/model.pt not found - is MODEL_EXPERIMENT=${MODEL_EXPERIMENT} correct?" >&2
    exit 1
fi
mkdir -p ./model
cp "/build/experiments/${MODEL_EXPERIMENT}/model.pt" "/build/experiments/${MODEL_EXPERIMENT}/config.json" ./model/
if [ -f "/build/experiments/${MODEL_EXPERIMENT}/classes.json" ]; then
    cp "/build/experiments/${MODEL_EXPERIMENT}/classes.json" ./model/
fi
EOF

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
