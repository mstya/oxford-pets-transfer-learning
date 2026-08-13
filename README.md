# Oxford Pets Transfer Learning

An educational project for pet-breed image classification (37 breeds, Oxford-IIIT
Pet dataset) exploring CNNs from scratch and transfer learning, ending with a
MobileNetV3-Large model fine-tuned to **94.20% best validation accuracy /
89.94% held-out test accuracy**.

## Quick start

```bash
pip install -r requirements.txt

# 1. Train the model configured in configs/base_config.json
make train                  # same as: python -m src.train

# 2. Serve the checkpoint configured in configs/serving_config.json as an API
make serve                  # same as: docker compose up --build

# 3. Classify a photo
curl -X POST http://localhost:8000/predict -F "file=@path/to/photo.jpg"
```

See [Training](#training) and [Serving the API](#serving-the-api) below for
details (quick smoke tests, running the API without Docker, switching which
checkpoint is served, etc).

## Project structure

```text
configs/         # base_config.json (training) and serving_config.json (which
                  # experiment the API/Docker/notebook serve)
data/             # Oxford-IIIT Pet images + annotations (auto-downloaded)
notebooks/        # EDA, from-scratch CNNs, transfer learning, inference
src/               # dataset, model, training loop, metrics, prediction, API, train.py
experiments/      # run artifacts (config.json, model.pt, classes.json, metrics.csv,
                  # results.png per run) and summary.csv with every run's hyperparameters
Dockerfile        # packages src/api.py + a trained checkpoint into a servable image
docker-compose.yml
Makefile          # `make train` / `make serve` / `make build`
```

## Data and evaluation protocol

- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/): 37 breeds
  (cats and dogs), auto-downloaded by `torchvision.datasets.OxfordIIITPet`.
- `trainval` split (3,680 images) is further split 70% train / 30% validation
  (`configs/base_config.json`'s `data.train_split`), with a fixed seed.
- `test` split (3,669 images) is never touched during training or
  hyperparameter selection — used exactly once, at the end, for the headline
  test accuracy below.
- Images resized to `224x224`, normalized with standard ImageNet statistics
  (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) to match what
  the pretrained MobileNetV3 backbone expects.
- Primary metric during tuning: `Best val accuracy`. Final reported metric:
  test accuracy of the served checkpoint.

**Validation correctness note:** `train.py` builds `train_dataset` and
`val_dataset` as two separate `OxfordIIITPet` instances (one per transform)
before splitting into matching index subsets. This matters — a single
dataset split via `Subset` alone would make validation images inherit the
*training* dataset's random-augmentation transform, silently applying random
crops/flips to every "validation" image on every epoch. An earlier version of
this project had exactly that bug; fixing it moved best validation accuracy
from ~79% to ~92-94% for the same model and hyperparameters (see
[Key experiments](#key-experiments)).

## Training

```bash
pip install -r requirements.txt
make train                            # or: python -m src.train
python -m src.train --epochs 5        # quick smoke test
python -m src.train --config configs/base_config.json --early-stopping-patience 0
```

This downloads the dataset (cached by `torchvision` after the first run),
trains the model, and writes a new `experiments/<N>/` directory containing
`model.pt`, `config.json`, `classes.json`, `metrics.csv`, and `results.png` —
the same artifacts produced by `notebooks/03_mobilenet_transfer_learning.ipynb`,
but from a single non-interactive command.

### Early stopping

`src/training.py`'s `training_loop` always keeps the checkpoint from whichever
epoch had the best validation accuracy so far. Early stopping adds one more
thing on top: it **ends training early** once validation accuracy hasn't
improved for a run of consecutive epochs, instead of always running the full
epoch budget. Configured under `training.early_stopping` in
`configs/base_config.json`:

```json
"early_stopping": { "patience": 20, "min_delta": 0.0 }
```

- **`patience`** — epochs in a row without improvement to tolerate before
  stopping. Deliberately set higher than the LR scheduler's own `patience`
  (4), so `ReduceLROnPlateau` gets a few chances to lower the learning rate —
  which often lets validation accuracy recover — before early stopping gives
  up.
- **`min_delta`** — minimum improvement to count as "improved".

Override per run without touching the config: `--early-stopping-patience N`
(`--early-stopping-patience 0` disables it, always running the full
`--epochs`).

## Serving the API

`src/api.py` is a FastAPI app exposing `POST /predict` (multipart image
upload → predicted breed + per-class probabilities) and `GET /health`. It's
packaged into a Docker image with the trained checkpoint baked in at build
time — no GPU needed to run it, only to train.

**Which checkpoint gets served is controlled by one file:
[`configs/serving_config.json`](configs/serving_config.json)** (currently
`{"model_experiment": 37}` — the best single run, see
[Best result](#best-result)). It's the single source of truth read by
`src/api.py`, the `Dockerfile` (via the `Makefile`/`docker-compose.yml`), and
`notebooks/04_inference.ipynb` — point it at a new experiment (e.g. the one
`src/train.py` just produced) and every one of those picks it up
automatically.

```bash
make serve
# ...equivalent to:
MODEL_EXPERIMENT=$(python3 -c "import json; print(json.load(open('configs/serving_config.json'))['model_experiment'])") \
  docker compose up --build
```

Then:

```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/photo.jpg"
```

Example response:

```json
{
  "label": "Abyssinian",
  "confidence": 0.7422,
  "probabilities": { "Abyssinian": 0.7422, "Miniature Pinscher": 0.0344, "...": "..." }
}
```

Running it without Docker (e.g. during development, with `requirements.txt`
already installed) works the same way, and also defaults to
`configs/serving_config.json`:

```bash
uvicorn src.api:app --reload
# ...or override without touching the config:
MODEL_EXPERIMENT_DIR=experiments/<N> uvicorn src.api:app --reload
```

(`requirements-api.txt` is the slim, Docker-only dependency set used by the
`Dockerfile` — it omits training-only packages (matplotlib, pandas, fastai)
and torch/torchvision, which the `Dockerfile` installs separately from the
CPU-only wheel index, since the container has no GPU/MPS access anyway.)

**Class names:** each experiment's `classes.json` (written by `src/train.py`
via `src/experiment.py:save_classes`) records the exact index → breed-name
mapping the model was trained with. Experiments trained before this was added
don't have one — `src/api.py` falls back to a hardcoded `OXFORD_PETS_CLASSES`
list (`src/predict.py`) in that case.

## Model architecture

**MobileNetV3-Large** (`torchvision.models.mobilenet_v3_large`, ImageNet
`IMAGENET1K_V2` pretrained weights) with a resized classifier head:

```text
model.features[0:15]   -> frozen (generic ImageNet convolutional filters)
model.features[15:17]  -> fine-tuned (2 inverted-residual blocks, ~953K params)
model.classifier[0]     Linear(960, 1280)          -> fine-tuned
model.classifier[1]     Hardswish()
model.classifier[2]     Dropout(p=0.5)              -> replaces pretrained 0.2
model.classifier[3]     Linear(1280, 37)            -> replaces pretrained 1000-way head
```

Trained with:
- **Discriminative learning rates** — backbone params at `lr/10` (pretrained
  weights need gentler updates), head params at the full `lr`, weight decay
  `3x` higher for the backbone group. See `src/model.py` / `src/train.py`.
- **Frozen BatchNorm statistics** in the still-frozen blocks
  (`src/model.py:freeze_bn_in_frozen_blocks`) — without this, BatchNorm's
  running mean/var keep drifting away from their pretrained ImageNet values
  on every forward pass in training mode, even though the layer's weights
  never receive a gradient.
- **Label smoothing** (`0.1`) and **dropout** (`0.5`) as regularization —
  tuned after the validation-leak fix above; further tuning past these values
  didn't move the needle (see [Key experiments](#key-experiments)).

## Key experiments

| Stage | Change | Best val accuracy |
| --- | --- | ---: |
| `SimpleCNNConv3` from scratch | 3-block CNN, various augmentation/dropout tuning | up to 55.34% |
| `MyAlexNet` from scratch | AlexNet-style architecture | 41.58% |
| MobileNetV3, linear probe | Pretrained backbone fully frozen, only classifier retrained | 75.99% |
| + unfrozen classifier head, dropout/label smoothing tuning | Still fully frozen `features` | ~79% (plateau) |
| + partial fine-tuning (`features[14:]` unfrozen) | Discriminative LR, BN freezing | ~85-88% |
| + **fixed validation-transform leak** | `val_dataset` now uses the deterministic transform instead of inheriting train-time random augmentation | **~92-94%** |
| Unfreeze-depth sweep (block 13 vs 14 vs 15, 3 seeds each) | Block 15 (fewest trainable backbone params) statistically tied with 13/14 | 92.45-92.75% mean |
| Regularization re-tuning (dropout 0.3-0.5, label smoothing 0.05-0.1) | No further improvement — diminishing returns confirmed | 93.30-93.84% |

Two structural fixes (partial fine-tuning, and the validation-transform leak
fix) accounted for nearly all of the improvement; further hyperparameter
tuning past that point moved results by less than run-to-run seed variance
(~1-1.3 points).

## Best result

```text
Model:              MobileNetV3-Large (backbone features[15:] + classifier fine-tuned)
Batch size:         64
Augmentation:       RandomResizedCrop(224) + RandomHorizontalFlip(p=0.5)
Dropout:            0.5
Label smoothing:    0.1
Optimizer:          Adam (discriminative LR: backbone lr/10, head lr, backbone weight_decay x3)
Learning rate:      0.0005
Weight decay:       0.0004
Scheduler:          ReduceLROnPlateau(mode="min", factor=0.5, patience=4, min_lr=1e-6)
Epochs:             50 (early stopping patience=20)
```

| | Value |
| --- | ---: |
| Experiment | **Exp #37** (`configs/serving_config.json` → `model_experiment: 37`) |
| Seed | 2 |
| Best val accuracy | **94.20%** (epoch 42) |
| **Test accuracy** (true held-out `test` split, evaluated once) | **89.94%** |

`configs/base_config.json` currently reproduces this exact configuration —
running `make train` retrains it from scratch (modulo the usual run-to-run
seed noise on GPU/MPS non-determinism).

## Next steps

- [DONE] Transfer learning with a pretrained backbone (MobileNetV3-Large).
- [DONE] Partial fine-tuning of backbone layers with discriminative learning rates.
- [DONE] Find and fix the validation-transform leak (biggest single accuracy jump).
- [DONE] Multi-seed comparison to separate real effects from run-to-run noise.
- [DONE] FastAPI serving endpoint + Docker packaging.
- Retrain a couple more seeds at the final configuration to get a proper mean
  ± stdev for the headline number (currently a single best-of-N run).
- `experiments/summary.csv` records early experiments (before `classes.json`
  was added) that lack it — `src/api.py` handles this via a fallback class
  list, but backfilling `classes.json` for older runs (or pruning them) would
  make `experiments/` fully self-describing.
