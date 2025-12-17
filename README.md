# MLOps Project - Fashion MNIST Classification

End-to-end MLOps pipeline with PyTorch, MLflow (local with Model Registry), Flask, and Docker.

## Architecture

```
Local Training → MLflow Tracking (SQLite) → Model Registry → ONNX Export → Docker Build → Inference
```

**Docker image size: ~540 MB** (using ONNX Runtime instead of PyTorch for inference)

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/tntd-22/MLOps.git
cd MLOps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Run all 5 experiments (trains models and promotes best to Production)
python -m src.experiments
```

This will:
- Train 5 different model configurations
- Log all experiments to MLflow (SQLite)
- Automatically register the best model to Model Registry
- Promote it to "champion" alias

### 3. Export Model to ONNX

```bash
# Export the champion model for lightweight Docker inference
python export_onnx.py
```

This creates:
- `model.onnx` - Lightweight ONNX model (~1.6 MB)
- `model_metadata.json` - Model metadata for inference

### 4. View MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Browse experiments in the "Experiments" tab and model versions in the "Models" tab.

### 5. Run Flask App

**Local development (with PyTorch):**
```bash
python app.py
# Open http://localhost:5001
```

**Local with ONNX (lightweight):**
```bash
python app_onnx.py
# Open http://localhost:5001
```

### 6. Docker

```bash
# Build lightweight image (~540 MB with ONNX)
docker build -t mlops:latest .

# Run
docker run -p 5001:5001 mlops:latest

# Or pull from Docker Hub
docker pull dungtnt/mlops:latest
docker run -p 5001:5001 dungtnt/mlops:latest
```

## Experiments

| # | Experiment | Description |
|---|------------|-------------|
| 1 | Baseline CNN | No regularization - observe overfitting |
| 2 | CNN + Regularization | BatchNorm + Dropout - reduce overfitting |
| 3 | CNN + Augmentation | Data enrichment - best generalization |
| 4 | Hyperparameter Tuning | Optimized settings (15 epochs) |
| 5 | Simple MLP | Demonstrate underfitting |

## Model Registry

Models are versioned in MLflow Model Registry using aliases:

| Alias | Description |
|-------|-------------|
| champion | Best performing model (auto-promoted) |

```python
# Load champion model programmatically
import mlflow.pytorch
model = mlflow.pytorch.load_model("models:/fashion-mnist-cnn@champion")
```

## Project Structure

```
MLOps/
├── .github/workflows/ci-cd.yml   # GitHub Actions
├── src/                          # Source code
│   ├── config.py                 # MLflow & Model Registry config
│   ├── data.py                   # Data loading & augmentation
│   ├── model.py                  # CNN & MLP architectures
│   ├── train.py                  # Training loop with MLflow
│   └── experiments.py            # Run experiments & promote model
├── templates/                    # Flask templates
├── static/                       # CSS styles
├── app.py                        # Flask application (PyTorch)
├── app_onnx.py                   # Flask application (ONNX Runtime)
├── export_onnx.py                # Export champion model to ONNX
├── Dockerfile                    # Docker configuration (ONNX)
├── requirements.txt              # Full dependencies (training)
├── requirements-inference.txt    # Lightweight dependencies (Docker)
├── model.onnx                    # Exported ONNX model
├── model_metadata.json           # Model metadata for inference
├── mlflow.db                     # MLflow SQLite database
└── mlruns/                       # MLflow artifacts (models)
```

## CI/CD

Push to `main` branch triggers:
1. Build lightweight Docker image (~540 MB with ONNX)
2. Push to Docker Hub

**Note:** `model.onnx` and `model_metadata.json` must be committed to the repo for CI/CD to work.

Required GitHub Secrets:
- `DOCKER_USERNAME`
- `DOCKER_TOKEN`

## Links

- **GitHub**: https://github.com/tntd-22/MLOps
- **Docker Hub**: https://hub.docker.com/r/dungtnt/mlops
