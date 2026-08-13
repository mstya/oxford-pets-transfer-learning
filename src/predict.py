"""Inference helpers for the trained MobileNetV3 pet-breed classifier."""
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Fallback index -> breed name mapping for experiments trained before
# save_classes() was wired into the training pipeline (see
# src/experiment.py:save_classes) and so have no classes.json alongside their
# model.pt. Matches OxfordIIITPet(...).classes exactly (torchvision derives it
# by sorting the annotation file's breed names).
OXFORD_PETS_CLASSES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
    "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair",
    "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter",
    "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin",
    "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard",
    "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx",
    "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier",
]


def build_predict_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD, image_size=IMG_SIZE):
    """Same deterministic preprocessing as define_transformations()'s
    val_transformations (src/dataset.py) - no random augmentation at inference time."""
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


def predict_proba(image, model, transform=None, device=None):
    """Run a trained model on a single image and return per-class probabilities.

    Args:
        image: a PIL.Image.Image, or a path (str/Path) to an image file.
        model: a trained torch.nn.Module (e.g. from src.model.build_model)
            already loaded with trained weights.
        transform: optional preprocessing transform. Defaults to
            build_predict_transform().
        device: torch.device to run inference on. Defaults to the model's
            current device.

    Returns:
        torch.Tensor: 1D tensor of class probabilities (length = num_classes).
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            image = img.convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if transform is None:
        transform = build_predict_transform()

    if device is None:
        device = next(model.parameters()).device

    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    return probabilities


def predict(image, model, class_names=OXFORD_PETS_CLASSES, transform=None, device=None):
    """Predict the breed of a single image using a trained model.

    Returns:
        str: the predicted class name.
    """
    probabilities = predict_proba(image, model, transform=transform, device=device)
    predicted_index = int(probabilities.argmax().item())

    return class_names[predicted_index]
