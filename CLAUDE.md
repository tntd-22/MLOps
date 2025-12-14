# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end MLOps pipeline for Fashion MNIST image classification using PyTorch, MLflow (local SQLite + Model Registry), Flask, and Docker.

**Pipeline Flow:** Local Training → MLflow Tracking → Model Registry (Production) → Docker Build → Inference

## Common Commands

```bash
# Run experiments (trains models, registers best to Production)
python -m src.experiments

# View MLflow UI locally (experiments + model registry)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Run Flask app locally (loads Production model)
python app.py

# Build Docker image (includes MLflow artifacts)
docker build -t mlops:latest .

# Run Docker container
docker run -p 5000:5000 mlops:latest
```

## Architecture

### Source Code (`src/`)
- **config.py** - MLflow configuration, Model Registry helpers (`get_production_model_from_registry()`, `promote_model_to_production()`)
- **data.py** - Data loading and augmentation; provides `get_dataloaders()` and `get_inference_transform()`
- **model.py** - Neural network architectures: `CNN` (with optional BatchNorm/Dropout) and `MLP`; use `get_model()` factory
- **train.py** - Training loop with MLflow tracking; logs models to MLflow runs
- **experiments.py** - Runs 5 experiments, then promotes the best model to Production stage

### Web Application
- **app.py** - Flask app that loads Production model from Model Registry; endpoints: `/predict`, `/health`, `/model_info`
- **templates/** - HTML templates
- **static/** - CSS styles

### MLflow Storage
- **mlflow.db** - SQLite database (experiments, runs, model registry)
- **mlruns/** - Model artifacts (PyTorch models via `mlflow.pytorch`)

## Key Patterns

### Model Registry
Models are versioned and staged in MLflow Model Registry:
- **Model name:** `fashion-mnist-cnn`
- **Stages:** None → Production (best model) / Archived (old versions)
- **Load Production model:** `models:/fashion-mnist-cnn/Production`

```python
# Promote a model to Production (done automatically in experiments.py)
from src.config import promote_model_to_production
promote_model_to_production(run_id)

# Load Production model (used by app.py)
from src.config import get_production_model_from_registry
model_uri, info = get_production_model_from_registry()
model = mlflow.pytorch.load_model(model_uri)
```

### MLflow Tracking
- Tracking URI: `sqlite:///mlflow.db`
- Artifacts: `mlruns/`
- Models logged via `mlflow.pytorch.log_model()`

### Inference
Flask app loads `models:/fashion-mnist-cnn/Production` at startup. The model version and metadata are available at `/model_info`.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci-cd.yml`) builds and pushes Docker images to Docker Hub on push to `main`. The Docker image includes MLflow artifacts (`mlflow.db` and `mlruns/`) with the Production model. Requires GitHub Secrets: `DOCKER_USERNAME`, `DOCKER_TOKEN`.
