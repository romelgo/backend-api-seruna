# ============================================================
# BACKEND — Control de Asistencia Facial
# Stack: FastAPI + InsightFace (SCRFD-34GF + ArcFace ResNet100)
# ============================================================

# ETAPA 1: Base con CUDA 12.1 + cuDNN (para onnxruntime-gpu)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Variables de entorno para build limpio
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencias del sistema: Python 3.10, libGL (OpenCV), libgomp (insightface)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Apuntar python3 → python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ── Directorio de trabajo ──────────────────────────────────
WORKDIR /app

# ── Instalar dependencias Python primero (cache layer) ────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Copiar el código de la aplicación ─────────────────────
# Estructura esperada en /app:
#   /app/backend/  → código del backend
#   /app/dataset/  → embeddings (montado como volumen)
#   /app/config.py → configuración del proyecto
COPY . /app/backend/
COPY config.py /app/config.py 2>/dev/null || true

# Crear directorio del dataset (será montado como volumen en producción)
RUN mkdir -p /app/dataset

# ── Variables de entorno por defecto ──────────────────────
ENV CUDA_VISIBLE_DEVICES=0 \
    INSIGHTFACE_CTX_ID=0 \
    DATASET_DIR=/app/dataset \
    DB_URL=/app/backend/attendance.db \
    CORS_ORIGINS=http://localhost:3000 \
    PORT=8000

# ── Exponer el puerto de la API ─────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Arranque del servidor ─────────────────────────────────
# uvicorn corre el módulo backend.main desde /app
CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
