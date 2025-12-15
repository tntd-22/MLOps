"""Configuration for MLOps project."""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local MLflow Configuration with SQLite
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_ARTIFACT_ROOT = MLFLOW_DIR

# Experiment and Model Registry
MLFLOW_EXPERIMENT_NAME = "fashion-mnist"
MLFLOW_MODEL_NAME = "fashion-mnist-cnn"

# Data paths
DATA_DIR = "data"

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 10
DEFAULT_SUBSET_SIZE = 5000  # Use subset for local training (0 for full dataset)

# Fashion MNIST classes
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def setup_mlflow():
    """Configure MLflow with local SQLite backend."""
    import mlflow

    # Create mlruns directory if it doesn't exist
    os.makedirs(MLFLOW_DIR, exist_ok=True)

    # Set tracking URI to local SQLite database
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return mlflow


def get_production_model_from_registry():
    """
    Get the Production model from MLflow Model Registry using aliases.

    Returns:
        Tuple of (model_uri, model_info) or (None, None) if no production model found
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    setup_mlflow()
    client = MlflowClient()

    try:
        # Get model version by "champion" alias (new MLflow approach)
        model_version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "champion")
        model_uri = f"models:/{MLFLOW_MODEL_NAME}@champion"

        # Get run info for additional metadata
        run = client.get_run(model_version.run_id)

        model_info = {
            "model_name": MLFLOW_MODEL_NAME,
            "version": model_version.version,
            "alias": "champion",
            "run_id": model_version.run_id,
            "experiment_name": run.data.params.get("experiment_name", "unknown"),
            "best_val_accuracy": run.data.metrics.get("best_val_accuracy", 0),
        }

        return model_uri, model_info

    except Exception as e:
        print(f"Error getting production model: {e}")
        return None, None


def promote_model_to_production(run_id: str):
    """
    Register a model from a run and set it as the champion (production) model.
    Uses MLflow aliases instead of deprecated stages.

    Args:
        run_id: The MLflow run ID containing the model to promote

    Returns:
        The registered model version
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    setup_mlflow()
    client = MlflowClient()

    # Register the model (or get existing)
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)

    # Set "champion" alias to the new version (automatically moves from previous version)
    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias="champion",
        version=result.version
    )

    print(f"Set model v{result.version} as 'champion'")
    return result
