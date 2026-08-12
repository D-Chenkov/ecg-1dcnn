# Non-root, cache-ordered image for the ECG 1D-CNN API (mirrors k8s-fastapi-starter).
FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch first (smaller image), then the rest.
# Ordering deps before code keeps this layer cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir fastapi uvicorn numpy pandas scikit-learn mlflow

COPY src/ src/
COPY app/ app/
COPY ecg_1dcnn.pth .          
# trained weights (produced by src/train.py)

# run as non-root
RUN useradd -u 1000 --create-home appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
