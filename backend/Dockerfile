FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal OS runtime libraries required by OpenCV & EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# STEP 1: Install CPU-only PyTorch FIRST
# This satisfies the torch dependency for easyocr and timm so pip NEVER downloads 2.5 GB of CUDA packages
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install remaining dependencies (easyocr and timm will recognize torch is already present)
COPY requirements-no-torch.txt .
RUN pip install --no-cache-dir -r requirements-no-torch.txt

# Copy application entrypoint
COPY predict.py .

# Copy data assets (models & metadata; raw datasets excluded via .dockerignore)
COPY data/ ./data/

# Cloud Run injects $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT} --workers 1"]
