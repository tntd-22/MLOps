"""Export Production model from MLflow Registry to ONNX format."""

import os
import json
import torch
import mlflow.pytorch

from src.config import (
    get_production_model_from_registry,
    setup_mlflow,
    FASHION_MNIST_CLASSES
)

# Output paths
ONNX_MODEL_PATH = "model.onnx"
MODEL_METADATA_PATH = "model_metadata.json"


def export_to_onnx():
    """Export the Production model to ONNX format."""
    setup_mlflow()

    # Load Production model from registry
    model_uri, model_info = get_production_model_from_registry()

    if model_uri is None:
        raise RuntimeError(
            "No champion model found. Please run experiments first: python -m src.experiments"
        )

    print(f"Loading model: {model_info['model_name']} v{model_info['version']}")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    # Create dummy input (Fashion MNIST: 1 channel, 28x28)
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export to ONNX (use dynamo=False for compatibility with Python 3.14)
    print(f"Exporting to {ONNX_MODEL_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        dynamo=False  # Use legacy exporter for compatibility
    )

    # Save model metadata for the inference app
    metadata = {
        "model_name": model_info["model_name"],
        "version": model_info["version"],
        "alias": model_info["alias"],
        "run_id": model_info["run_id"],
        "experiment_name": model_info["experiment_name"],
        "best_val_accuracy": model_info["best_val_accuracy"],
        "classes": FASHION_MNIST_CLASSES,
        "input_shape": [1, 1, 28, 28],
        "onnx_file": ONNX_MODEL_PATH
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {MODEL_METADATA_PATH}")

    # Verify the ONNX model
    import onnx
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification: OK")

    # Print file sizes
    onnx_size = os.path.getsize(ONNX_MODEL_PATH) / 1024 / 1024
    print(f"\nONNX model size: {onnx_size:.2f} MB")

    return ONNX_MODEL_PATH, MODEL_METADATA_PATH


if __name__ == "__main__":
    export_to_onnx()
