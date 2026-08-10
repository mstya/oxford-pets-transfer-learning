import csv
import json
import os
import time
from os import listdir
from os.path import isdir, join
from pathlib import Path

import pandas as pd
import torch


def prepare_results_dir(results_dir="experiments"):
    """Create (if needed) the experiments dir and allocate the next experiment number."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    experiment_numbers = [
        int(name) for name in listdir(results_dir)
        if name.isdigit() and isdir(join(results_dir, name))
    ]
    exp_number = max(experiment_numbers, default=0) + 1
    exp_path = os.path.join(results_dir, str(exp_number))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    return results_dir, exp_number, exp_path


def save_config(path, config_json):
    config_path = os.path.join(path, "config.json")
    with open(config_path, "w", newline="", encoding="utf-8") as f:
        json.dump(config_json, f)


def save_classes(path, class_names):
    """Persist the index -> class name mapping the model was trained with."""
    classes_path = os.path.join(path, "classes.json")
    with open(classes_path, "w", newline="", encoding="utf-8") as f:
        json.dump(class_names, f)


def save_metrics(path, metrics):
    train_losses, val_losses, val_accuracies, train_accuracy = metrics
    metrics_path = os.path.join(path, "metrics.csv")
    data = {
        "Train Losses": train_losses,
        "Val Losses": val_losses,
        "Val Accuracies": val_accuracies,
        "Train Accuracies": train_accuracy,
    }
    pd.DataFrame(data).to_csv(metrics_path, index_label="epoch")


def save_summary(save_path,
                  exp_number,
                  learning_rate,
                  val_accuracy,
                  model_name,
                  epoch,
                  batch_size,
                  optimizer,
                  weight_decay,
                  seed,
                  augmentation,
                  val_loss,
                  best_val_accuracy,
                  best_val_loss,
                  best_epoch,
                  dropout,
                  scheduler_config):
    summary = {
        "Experiment": f"Exp #{str(exp_number)}",
        "Timestamp": time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()),
        "Model": model_name,
        "Epochs Trained": epoch,
        "Batch Size": batch_size,
        "Optimizer": optimizer,
        "Learning Rate": learning_rate,
        "Weight Decay": weight_decay,
        "Dropout": dropout,
        "Seed": seed,
        "Final Val accuracy": val_accuracy,
        "Final Val loss": val_loss,
        "Best val accuracy": best_val_accuracy,
        "Best val loss": best_val_loss,
        "Best val epoch": best_epoch,
        "Augmentation": augmentation,
        "Scheduler": scheduler_config["name"],
        "Scheduler Mode": scheduler_config["mode"],
        "Scheduler Factor": scheduler_config["factor"],
        "Scheduler Patience": scheduler_config["patience"],
        "Scheduler Min LR": scheduler_config["min_lr"],
    }
    summary_path = os.path.join(save_path, "summary.csv")

    file_exists = os.path.isfile(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary)


def save_results(results_dir,
                  exp_path,
                  exp_number,
                  learning_rate,
                  metrics,
                  model_name,
                  epoch,
                  batch_size,
                  optimizer,
                  weight_decay,
                  seed,
                  augmentation,
                  dropout,
                  scheduler_config):
    train_losses, val_losses, val_accuracies, train_accuracy = metrics
    best_index = max(range(len(val_accuracies)), key=val_accuracies.__getitem__)
    best_val_accuracy = val_accuracies[best_index]
    best_val_loss = val_losses[best_index]
    best_epoch = best_index + 1

    final_val_accuracy = val_accuracies[-1]
    final_val_loss = val_losses[-1]

    save_summary(results_dir, exp_number, learning_rate, final_val_accuracy, model_name, epoch, batch_size,
                 optimizer, weight_decay, seed, augmentation, final_val_loss, best_val_accuracy, best_val_loss,
                 best_epoch, dropout, scheduler_config)
    save_metrics(exp_path, metrics)


def save_model(exp_path, model):
    torch.save(
        model.state_dict(),
        os.path.join(exp_path, "model.pt"),
    )
