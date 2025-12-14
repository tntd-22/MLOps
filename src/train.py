"""Training loop with MLflow tracking."""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from src.model import get_model, count_parameters
from src.data import get_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    experiment_name: str,
    model_type: str = "cnn",
    use_batchnorm: bool = False,
    dropout_rate: float = 0.0,
    use_augmentation: bool = False,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    epochs: int = 10,
    description: str = "",
    save_model: bool = True
):
    """
    Train a model with MLflow tracking.

    Args:
        experiment_name: Name for this experiment run
        model_type: "cnn" or "mlp"
        use_batchnorm: Whether to use batch normalization
        dropout_rate: Dropout rate (0.0 = no dropout)
        use_augmentation: Whether to use data augmentation
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        description: Description of the experiment
        save_model: Whether to save the model after training

    Returns:
        Dictionary with training results
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        use_augmentation=use_augmentation
    )

    # Create model
    model = get_model(
        model_type=model_type,
        use_batchnorm=use_batchnorm,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    num_params = count_parameters(model)
    print(f"Model: {model_type.upper()}, Parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start MLflow run
    with mlflow.start_run(run_name=experiment_name):
        # Log parameters
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("description", description)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("use_batchnorm", use_batchnorm)
        mlflow.log_param("use_augmentation", use_augmentation)
        mlflow.log_param("num_parameters", num_params)
        mlflow.log_param("device", str(device))

        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Store history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            # Track best model in memory
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log final metrics
        final_train_acc = history["train_acc"][-1]
        final_val_acc = history["val_acc"][-1]
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        overfit_gap = final_train_acc - final_val_acc

        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("overfit_gap", overfit_gap)
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Log best model to MLflow
        if save_model and best_model_state is not None:
            model.load_state_dict(best_model_state)
            mlflow.pytorch.log_model(model, "model")

        # Get run ID
        run_id = mlflow.active_run().info.run_id

        # Prepare results
        results = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "model_type": model_type,
            "final_train_accuracy": final_train_acc,
            "final_val_accuracy": final_val_acc,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_accuracy": best_val_acc,
            "overfit_gap": overfit_gap,
            "hyperparameters": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "dropout_rate": dropout_rate,
                "use_batchnorm": use_batchnorm,
                "use_augmentation": use_augmentation
            }
        }

        print(f"\n{'='*50}")
        print(f"Experiment: {experiment_name}")
        print(f"Final Train Accuracy: {final_train_acc:.4f}")
        print(f"Final Val Accuracy: {final_val_acc:.4f}")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"Overfit Gap: {overfit_gap:.4f}")
        print(f"{'='*50}\n")

        return results, model
