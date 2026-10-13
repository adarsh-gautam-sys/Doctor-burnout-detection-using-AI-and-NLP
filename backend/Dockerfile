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

# Install non-torch Python dependencies
COPY requirements-no-torch.txt .
RUN pip install --no-cache-dir -r requirements-no-torch.txt

# Install CPU-only PyTorch & Torchvision (saves ~2GB container size)
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy application entrypoint
COPY predict.py .

# Copy data assets (models & metadata; raw datasets excluded via .dockerignore)
COPY data/ ./data/

# Cloud Run injects $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT} --workers 1"]
