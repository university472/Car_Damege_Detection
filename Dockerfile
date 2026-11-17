# ---------- Stage 1: build the React frontend ----------
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty base URL => the app calls the API on the same domain (served by FastAPI below).
ENV VITE_API_BASE_URL=
RUN npm run build

# ---------- Stage 2: Python API + models ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# System libraries OpenCV needs to decode video / run headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Railway has no GPU: install the small CPU-only PyTorch build first so
# ultralytics does not pull the ~2.5 GB CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY models/car_parts.pt models/car_damage.pt ./models/
COPY --from=frontend /frontend/dist ./frontend/dist

RUN mkdir -p data/uploads data/artifacts

# Railway injects PORT. Use ONE worker: job state lives in process memory.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75"]
