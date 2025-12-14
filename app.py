"""Flask web application for Fashion MNIST prediction."""

import torch
import mlflow.pytorch
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

from src.data import get_inference_transform
from src.config import (
    FASHION_MNIST_CLASSES,
    get_production_model_from_registry,
    setup_mlflow
)

app = Flask(__name__)

# Global model variable
model = None
model_info = None


def load_model():
    """Load the Production model from MLflow Model Registry."""
    global model, model_info

    # Setup MLflow
    setup_mlflow()

    # Load Production model from registry
    model_uri, registry_info = get_production_model_from_registry()

    if model_uri:
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        model_info = registry_info
        print(f"Model loaded from registry: {registry_info['model_name']}")
        print(f"Version: {registry_info['version']} (@{registry_info['alias']})")
        print(f"Experiment: {registry_info['experiment_name']}")
        print(f"Best validation accuracy: {registry_info['best_val_accuracy']:.4f}")
        return model

    raise RuntimeError(
        "No champion model found. Please run experiments first: python -m src.experiments"
    )


def predict_image(image_bytes):
    """
    Predict the class of an uploaded image.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Load and preprocess image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale

    # Apply transforms
    transform = get_inference_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = FASHION_MNIST_CLASSES[predicted_idx.item()]
    confidence_value = confidence.item()

    # Get all class probabilities
    all_probs = {
        FASHION_MNIST_CLASSES[i]: float(probabilities[0][i])
        for i in range(len(FASHION_MNIST_CLASSES))
    }

    return predicted_class, confidence_value, all_probs


@app.route("/")
def index():
    """Home page with upload form."""
    return render_template("index.html", model_info=model_info)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read image bytes
        image_bytes = file.read()

        # Get prediction
        predicted_class, confidence, all_probs = predict_image(image_bytes)

        # Sort probabilities
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": f"{confidence * 100:.2f}%",
            "all_probabilities": sorted_probs[:5]  # Top 5 predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info")
def get_model_info():
    """Return model metadata."""
    return jsonify(model_info)


@app.route("/health")
def health():
    """Health check endpoint for Docker."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


if __name__ == "__main__":
    # Load model on startup
    load_model()

    # Run Flask app (port 5001 to avoid macOS AirPlay conflict)
    app.run(host="0.0.0.0", port=5001, debug=False)
