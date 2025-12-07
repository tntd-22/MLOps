"""Configuration for MLOps project."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DagsHub MLflow Configuration
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "your_username")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "your_token")
DAGSHUB_REPO_NAME = "MLOps"

# MLflow Tracking URI
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"

# Model paths
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "best_model_info.json")

# Results paths
RESULTS_DIR = "results"
EXPERIMENTS_SUMMARY_PATH = os.path.join(RESULTS_DIR, "experiments_summary.csv")

# Data paths
DATA_DIR = "data"

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 10

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
    """Configure MLflow with DagsHub credentials."""
    import mlflow

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Set authentication via environment variables
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    return mlflow
