FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/data SAMPLES_DIR=/samples
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only PyTorch keeps the image far smaller than the default CUDA wheels
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app app
COPY scripts scripts
# Bake model weights into the image so runtime needs no network or paid credentials
RUN python scripts/warmup.py || echo "model warm-up skipped; weights download on first run"
COPY tests tests
COPY pytest.ini .
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
