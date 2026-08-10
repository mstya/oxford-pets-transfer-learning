import copy

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")

def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100.0 * correct / total
    return epoch_loss, epoch_accuracy

def validate_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            val_loss = loss_function(outputs, labels)
            running_val_loss += val_loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total

    return epoch_val_loss, epoch_accuracy

def training_loop(model,
                  train_loader,
                  val_loader,
                  loss_function,
                  optimizer,
                  num_epochs,
                  device,
                  scheduler,
                  early_stopping_patience=None,
                  early_stopping_min_delta=0.0):
    """Train for up to `num_epochs`, always keeping the best-val-accuracy
    checkpoint (as before).

    If `early_stopping_patience` is set, training stops early once
    `early_stopping_patience` consecutive epochs pass without validation
    accuracy improving by more than `early_stopping_min_delta`. Leaving it
    `None` (the default) disables early stopping and keeps the old
    run-for-num_epochs behavior.
    """
    model.to(device)

    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    train_losses, val_losses, val_accuracies, train_accuracies = [], [], [], []

    print("--- Training Started ---")

    for epoch in range(num_epochs):
        epoch_loss, train_accuracy = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)

        epoch_val_loss, val_accuracy = validate_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy + early_stopping_min_delta:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"--- Early stopping: no val accuracy improvement for {early_stopping_patience} epochs "
                  f"(best {best_val_accuracy:.2f}% at epoch {best_epoch}) ---")
            break

    print("--- Finished Training ---")

    if best_model_state:
        print(f"\n--- Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)

    metrics = [train_losses, val_losses, val_accuracies, train_accuracies]

    return model, metrics