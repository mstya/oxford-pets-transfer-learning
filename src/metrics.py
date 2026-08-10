import os

import matplotlib.ticker as mticker
import torch
from matplotlib import pyplot as plt


def plot_training_metrics(metrics, save_path):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies, train_accuracy = metrics

    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot train accuracy data
    ax2.plot(epochs, train_accuracy, color='#085c75', linewidth=2.5, marker='o', markersize=5,
             label='Training Accuracy')
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5,
             label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Training & Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)

        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()
    fig.savefig(os.path.join(save_path, 'results.png'))


def calculate_per_class_accuracy(model, data_loader, class_names, device):
    model.eval()
    correct = [0] * len(class_names)
    total = [0] * len(class_names)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)

            for class_index in range(len(class_names)):
                class_mask = labels == class_index
                total[class_index] += class_mask.sum().item()
                correct[class_index] += (predictions[class_mask] == labels[class_mask]).sum().item()

    return {
        class_name: 100 * correct[index] / total[index]
        for index, class_name in enumerate(class_names)
    }


def calculate_confusion_matrix(model, data_loader, num_classes, device):
    model.eval()
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)

            for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                matrix[true_label, predicted_label] += 1

    return matrix


def plot_confusion_matrix(matrix, class_names):
    normalized_matrix = matrix.float() / matrix.sum(dim=1, keepdim=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(normalized_matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Fraction of true class")

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized Confusion Matrix")

    for true_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            value = normalized_matrix[true_index, predicted_index].item()
            color = "white" if value > 0.5 else "black"
            ax.text(predicted_index, true_index, f"{value:.2f}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.show()
