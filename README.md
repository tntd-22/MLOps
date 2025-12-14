# MLOps Project - Fashion MNIST Classification

End-to-end MLOps pipeline with PyTorch, MLflow (local with Model Registry), Flask, and Docker.

## Architecture

```
Local Training → MLflow Tracking (SQLite) → Model Registry (Production) → Docker Build → Inference
```

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/<username>/MLOps.git
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
- Promote it to "Production" stage

### 3. View MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Browse experiments in the "Experiments" tab and model versions in the "Models" tab.

### 4. Run Flask App

```bash
python app.py
# Open http://localhost:5000
```

The app automatically loads the Production model from the Model Registry.

### 5. Docker

```bash
# Build (includes MLflow artifacts with Production model)
docker build -t mlops:latest .

# Run
docker run -p 5000:5000 mlops:latest

# Or pull from Docker Hub
docker pull <username>/mlops:latest
docker run -p 5000:5000 <username>/mlops:latest
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

Models are versioned and staged in MLflow Model Registry:

| Stage | Description |
|-------|-------------|
| None | Newly registered models |
| Production | Best performing model (auto-promoted) |
| Archived | Previous production models |

```python
# Load Production model programmatically
import mlflow.pytorch
model = mlflow.pytorch.load_model("models:/fashion-mnist-cnn/Production")
```

## Project Structure

```
MLOps/
├── .github/workflows/ci-cd.yml  # GitHub Actions
├── src/                          # Source code
│   ├── config.py                 # MLflow & Model Registry config
│   ├── data.py                   # Data loading & augmentation
│   ├── model.py                  # CNN & MLP architectures
│   ├── train.py                  # Training loop with MLflow
│   └── experiments.py            # Run experiments & promote model
├── templates/                    # Flask templates
├── static/                       # CSS styles
├── app.py                        # Flask application
├── Dockerfile                    # Docker configuration (Python 3.12)
├── requirements.txt              # Python dependencies
├── mlflow.db                     # MLflow SQLite database
└── mlruns/                       # MLflow artifacts (models)
```

## CI/CD

Push to `main` branch triggers:
1. Build Docker image (includes MLflow artifacts)
2. Push to Docker Hub

Required GitHub Secrets:
- `DOCKER_USERNAME`
- `DOCKER_TOKEN`

## Links

- **Docker Hub**: `https://hub.docker.com/r/<username>/mlops`
