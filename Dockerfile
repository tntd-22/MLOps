FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install lightweight inference dependencies (NO PyTorch!)
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy ONNX model and metadata (exported via export_onnx.py)
COPY model.onnx .
COPY model_metadata.json .

# Copy application
COPY src/config.py src/config.py
COPY templates/ templates/
COPY static/ static/
COPY app_onnx.py .

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run ONNX inference application
CMD ["python", "app_onnx.py"]
