"""
FastAPI — Entry point del backend de reconocimiento facial.

Arquitectura:
  - Lifespan: carga modelos insightface una sola vez al arranque
  - Estado global en app.state (recognizer, embedding_manager, db, index)
  - CORS habilitado para frontend Next.js
  - Routers: /enroll, /identify, /students, /attendance
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Cargar variables de entorno ANTES de importar routers
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Añadir el root del proyecto al path para importar dataset_manager, config, etc.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.face_recognizer import FaceRecognizer
from backend.core.embedding_manager import EmbeddingManager
from backend.core.attendance import AttendanceDB
from backend.routers import enrollment, identification, students


# ----------------------------------------------------------------
# Variables de entorno (con defaults)
# ----------------------------------------------------------------
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(PROJECT_ROOT / "dataset")))
DB_PATH = os.getenv("DB_URL", str(PROJECT_ROOT / "backend" / "attendance.db"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")


# ----------------------------------------------------------------
# Lifespan: inicialización y cleanup
# ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga los modelos insightface y abre la DB al arranque.
    Los modelos quedan en memoria durante toda la vida del servidor.
    """
    print("=" * 60)
    print("  BACKEND — Control de Asistencia Facial")
    print("  Stack: SCRFD-34GF + ArcFace ResNet100 (insightface)")
    print("=" * 60)

    # Contexto CUDA (0=primera GPU, -1=CPU)
    ctx_id = int(os.getenv("INSIGHTFACE_CTX_ID", "0"))

    # Motor de reconocimiento facial (carga y descarga modelos buffalo_l)
    app.state.recognizer = FaceRecognizer(ctx_id=ctx_id)

    # Gestor de embeddings en disco
    app.state.embedding_manager = EmbeddingManager(DATASET_DIR)

    # Cargar índice de embeddings en memoria para búsqueda rápida
    app.state.embeddings_index = app.state.embedding_manager.load_all_embeddings()
    print(f"[main] Índice cargado: {len(app.state.embeddings_index)} estudiantes")

    # Base de datos SQLite
    app.state.db = AttendanceDB(DB_PATH)
    await app.state.db.init()

    import config as _cfg
    print(f"[main] Dataset:             {DATASET_DIR}")
    print(f"[main] DB:                  {DB_PATH}")
    print(f"[main] Umbral seguro:       {_cfg.THRESHOLD_SECURE}")
    print(f"[main] Umbral zona gris:    {_cfg.THRESHOLD_GREY_LOW}")
    print(f"[main] Votos galería:       {_cfg.GALLERY_VOTE_RATIO}")
    print("[main] Servidor listo.\n")

    yield  # Servidor corriendo

    # Cleanup (si fuera necesario)
    print("[main] Apagando servidor...")


# ----------------------------------------------------------------
# App
# ----------------------------------------------------------------
app = FastAPI(
    title="Control de Asistencia Facial",
    description=(
        "API REST para registro e identificación de estudiantes "
        "mediante reconocimiento facial con SCRFD-34GF + ArcFace ResNet100."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permitir llamadas desde el frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(enrollment.router, tags=["Enrollment"])
app.include_router(identification.router, tags=["Identificación"])
app.include_router(students.router, tags=["Estudiantes"])


@app.get("/", tags=["Health"])
async def health_check():
    """Health check — verifica que el servidor está activo."""
    return {
        "status": "ok",
        "service": "Control de Asistencia Facial",
        "students_loaded": len(app.state.embeddings_index),
    }
