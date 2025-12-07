# MLOps Project - Fashion MNIST Classification

End-to-end MLOps pipeline with PyTorch, MLflow, Flask, and Docker.

## Architecture

```
Training (Colab + DagsHub) → Model Export → GitHub → CI/CD → Docker Hub → Inference
```

## Quick Start

### 1. Training (Google Colab)

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Set your DagsHub credentials
3. Run all cells to train 5 experiments
4. Download `best_model.pt` and `best_model_info.json`

### 2. Local Setup

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

### 3. Run Flask App Locally

```bash
python app.py
# Open http://localhost:5000
```

### 4. Docker

```bash
# Build
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
| 4 | Hyperparameter Tuning | Optimized settings |
| 5 | Simple MLP | Demonstrate underfitting |

## Project Structure

```
MLOps/
├── .github/workflows/ci-cd.yml  # GitHub Actions
├── models/                       # Trained model (committed)
├── notebooks/train_colab.ipynb   # Training notebook
├── results/                      # Experiment results
├── src/                          # Source code
├── templates/                    # Flask templates
├── static/                       # CSS styles
├── app.py                        # Flask application
├── Dockerfile                    # Docker configuration
└── requirements.txt              # Python dependencies
```

## Links

- **DagsHub MLflow**: `https://dagshub.com/<username>/MLOps`
- **Docker Hub**: `https://hub.docker.com/r/<username>/mlops`

## CI/CD

Push to `main` branch triggers:
1. Build Docker image
2. Push to Docker Hub

Required GitHub Secrets:
- `DOCKER_USERNAME`
- `DOCKER_TOKEN`
