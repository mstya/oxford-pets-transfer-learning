"""Train the MobileNetV3 transfer-learning pipeline end to end, from the
Oxford-IIIT Pet dataset (auto-downloaded by torchvision) to a saved checkpoint.

Usage (run from the repo root - needed for the `from src...` imports below to
resolve; `make train` does this for you):
    python -m src.train
    python -m src.train --config configs/base_config.json --epochs 5

Mirrors the pipeline built up in notebooks/03_mobilenet_transfer_learning.ipynb,
but as a single reproducible script so a fresh checkout can be trained with one
command.
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never try to pop up a GUI window

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import OxfordIIITPet

from src.dataset import define_transformations
from src.experiment import prepare_results_dir, save_classes, save_config, save_model, save_results
from src.metrics import plot_training_metrics
from src.model import build_model, freeze_backbone, freeze_bn_in_frozen_blocks
from src.training import device, training_loop

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base_config.json"),
                         help="Path to the training config JSON.")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override the number of epochs from the config.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                         help="Override the config's training.early_stopping.patience. "
                              "Pass 0 to disable early stopping (always run the full --epochs).")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "experiments"),
                         help="Directory experiments are written to.")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"),
                         help="Directory the Oxford-IIIT Pet dataset lives in (downloaded if missing).")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as file:
        config_json = json.load(file)

    seed = int(config_json["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    batch_size = int(config_json["data"]["batch_size"])
    train_split = float(config_json["data"]["train_split"])
    augmentation = config_json["data"]["augmentation"]

    model_name = config_json["model"]["name"]
    dropout = float(config_json["model"]["dropout"])
    unfreeze_from_block = int(config_json["model"]["unfreeze_from_block"])

    learning_rate = float(config_json["training"]["learning_rate"])
    weight_decay = float(config_json["training"]["weight_decay"])
    label_smoothing = float(config_json["training"].get("label_smoothing", 0.0))
    scheduler_config = config_json["training"]["scheduler"]
    epochs = args.epochs if args.epochs is not None else int(config_json["training"]["epochs"])

    # early_stopping is optional in the config - absent means "disabled",
    # same as training_loop's own default.
    early_stopping_config = config_json["training"].get("early_stopping", {})
    if args.early_stopping_patience is not None:
        early_stopping_patience = args.early_stopping_patience or None  # 0 -> disabled
    else:
        early_stopping_patience = early_stopping_config.get("patience")
    early_stopping_min_delta = float(early_stopping_config.get("min_delta", 0.0))

    print("--- Downloading / locating Oxford-IIIT Pet dataset ---")
    train_transform, val_transform = define_transformations(augmentation, IMAGENET_MEAN, IMAGENET_STD)

    # Two separate dataset instances (not one dataset split via random_split
    # alone) so train/val get different transforms - a Subset of a single
    # dataset would inherit that dataset's transform for both, silently
    # running the random train-time augmentation on validation images too.
    train_dataset = OxfordIIITPet(root=args.data_dir, split="trainval", target_types="category",
                                   download=True, transform=train_transform)
    val_dataset = OxfordIIITPet(root=args.data_dir, split="trainval", target_types="category",
                                 download=True, transform=val_transform)
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"{num_classes} classes found.")

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(train_dataset, [train_split, 1 - train_split], generator=generator)

    train_loader = DataLoader(Subset(train_dataset, train_indices.indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(val_dataset, val_indices.indices), batch_size=batch_size, shuffle=False)

    print(f"--- Building {model_name} (unfreeze_from_block={unfreeze_from_block}, dropout={dropout}) ---")
    model = build_model(num_classes=num_classes, dropout=dropout, pretrained=True)
    model = freeze_backbone(model, unfreeze_from_block)
    model = model.to(device)
    model = freeze_bn_in_frozen_blocks(model, unfreeze_from_block)

    backbone_params = [p for block in model.features[unfreeze_from_block:] for p in block.parameters()]
    head_params = list(model.classifier.parameters())

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": learning_rate / 10, "weight_decay": weight_decay * 3},
        {"params": head_params, "lr": learning_rate, "weight_decay": weight_decay},
    ])
    loss_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config["mode"],
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["min_lr"]),
    )

    model, metrics = training_loop(
        model, train_loader, val_loader, loss_function, optimizer, epochs, device, scheduler,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    print("--- Saving experiment artifacts ---")
    results_dir, exp_number, exp_path = prepare_results_dir(args.results_dir)

    # metrics[0] (train_losses) has one entry per epoch actually run, which is
    # <= epochs when early stopping cut training short.
    epochs_trained = len(metrics[0])

    save_config(exp_path, config_json)
    save_classes(exp_path, class_names)
    save_model(exp_path, model)
    save_results(results_dir, exp_path, exp_number, learning_rate, metrics, model.__class__.__name__,
                 epochs_trained, batch_size, optimizer.__class__.__name__, weight_decay, seed, str(augmentation),
                 dropout, scheduler_config, label_smoothing, unfreeze_from_block)
    plot_training_metrics(metrics, exp_path)

    print(f"\nDone. Model and artifacts saved to {exp_path}/")
    print(f"To serve this checkpoint, set \"model_experiment\": {exp_number} in configs/serving_config.json,")
    print("then run `make serve` (or `uvicorn src.api:app --reload` without Docker).")
    print(f"(Or try it immediately without touching the config: "
          f"MODEL_EXPERIMENT_DIR={exp_path} uvicorn src.api:app --reload)")


if __name__ == "__main__":
    main()
