"""Flask web application for Fashion MNIST prediction using ONNX Runtime."""

import json
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Paths
ONNX_MODEL_PATH = "model.onnx"
MODEL_METADATA_PATH = "model_metadata.json"

# Global variables
session = None
model_info = None
classes = None


def load_model():
    """Load the ONNX model and metadata."""
    global session, model_info, classes

    # Load metadata
    with open(MODEL_METADATA_PATH, "r") as f:
        model_info = json.load(f)

    classes = model_info["classes"]

    # Create ONNX Runtime session
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    print(f"Model loaded: {model_info['model_name']} v{model_info['version']}")
    print(f"Best validation accuracy: {model_info['best_val_accuracy']:.4f}")

    return session


def preprocess_image(image_bytes):
    """Preprocess image for inference."""
    # Load image and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]

    # Add batch and channel dimensions: (1, 1, 28, 28)
    img_array = img_array.reshape(1, 1, 28, 28)

    return img_array


def softmax(x):
    """Compute softmax values."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict_image(image_bytes):
    """
    Predict the class of an uploaded image.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Preprocess
    input_tensor = preprocess_image(image_bytes)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})[0]

    # Apply softmax to get probabilities
    probabilities = softmax(outputs[0])

    # Get prediction
    predicted_idx = np.argmax(probabilities)
    confidence = float(probabilities[predicted_idx])
    predicted_class = classes[predicted_idx]

    # Get all class probabilities
    all_probs = {
        classes[i]: float(probabilities[i])
        for i in range(len(classes))
    }

    return predicted_class, confidence, all_probs


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
        "model_loaded": session is not None
    })


if __name__ == "__main__":
    # Load model on startup
    load_model()

    # Run Flask app (port 5001 to avoid macOS AirPlay conflict)
    app.run(host="0.0.0.0", port=5001, debug=False)
