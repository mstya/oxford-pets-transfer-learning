"""FastAPI service exposing the trained MobileNetV3 pet-breed classifier.

POST /predict - multipart image upload -> predicted breed + per-class probabilities.
GET  /health  - liveness/readiness check.

Run without Docker (e.g. during development, with requirements.txt installed):
    uvicorn src.api:app --reload
See the Dockerfile / Makefile (`make serve`) for the containerized version.
"""
import io
import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.model import build_model
from src.predict import OXFORD_PETS_CLASSES, build_predict_transform, predict_proba
from src.training import device

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVING_CONFIG_PATH = _REPO_ROOT / "configs" / "serving_config.json"


def _default_experiment_dir():
    """The experiment served when MODEL_EXPERIMENT_DIR isn't set - read from
    configs/serving_config.json, the single source of truth for which
    checkpoint to deploy (also used by the Dockerfile/docker-compose.yml/
    Makefile and notebooks/04_inference.ipynb)."""
    with open(_SERVING_CONFIG_PATH, encoding="utf-8") as file:
        serving_config = json.load(file)
    return _REPO_ROOT / "experiments" / str(serving_config["model_experiment"])


# Experiment whose saved model (state_dict) is served by the API. Override
# with the MODEL_EXPERIMENT_DIR env var to serve a different checkpoint
# without touching configs/serving_config.json.
EXPERIMENT_DIR = Path(os.environ.get("MODEL_EXPERIMENT_DIR") or _default_experiment_dir())
MODEL_PATH = EXPERIMENT_DIR / "model.pt"
MODEL_CONFIG_PATH = EXPERIMENT_DIR / "config.json"
CLASSES_PATH = EXPERIMENT_DIR / "classes.json"


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


def load_class_names():
    """Prefer the class list training persists alongside the checkpoint (see
    src/experiment.py:save_classes); fall back to the hardcoded
    OXFORD_PETS_CLASSES for older experiments that predate classes.json."""
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH, encoding="utf-8") as file:
            return json.load(file)
    return OXFORD_PETS_CLASSES


def load_model(num_classes):
    with open(MODEL_CONFIG_PATH, encoding="utf-8") as file:
        model_config = json.load(file)
    dropout = float(model_config["model"]["dropout"])

    model = build_model(num_classes=num_classes, dropout=dropout, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


app = FastAPI(title="Oxford Pets Breed Classifier")

class_names = load_class_names()
model = load_model(num_classes=len(class_names))
transform = build_predict_transform()


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    probabilities = predict_proba(image, model, transform=transform, device=device)
    predicted_index = int(probabilities.argmax().item())

    return PredictionResponse(
        label=class_names[predicted_index],
        confidence=float(probabilities[predicted_index]),
        probabilities={
            class_name: float(probability)
            for class_name, probability in zip(class_names, probabilities.tolist())
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
