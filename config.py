"""
Configuración global del sistema de enrollment facial.
Optimizado para CUDA 12.9 + PyTorch 2.8+
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# Rutas base
# ============================================================
# PyInstaller pone los archivos en _internal/, pero el .exe está un nivel arriba.
# En desarrollo, __file__ apunta al directorio del proyecto directamente.
if getattr(sys, 'frozen', False):
    # Ejecutando como .exe empaquetado con PyInstaller
    # sys.executable = dist/EnrollmentFacial/EnrollmentFacial.exe
    BASE_DIR = Path(sys.executable).parent
else:
    # Ejecutando como script Python normal
    BASE_DIR = Path(__file__).parent

# Cargar variables desde .env ubicado junto a este archivo (backend/.env)
load_dotenv(BASE_DIR / ".env")

# DATASET_DIR: se lee del .env; si es ruta relativa se resuelve desde BASE_DIR
_dataset_env = os.getenv("DATASET_DIR", "dataset")
DATASET_DIR = (BASE_DIR / _dataset_env).resolve()

MODELS_DIR = BASE_DIR / "models"

# ============================================================
# GPU / CUDA
# ============================================================
# Detección defensiva: PyInstaller puede no empaquetar torch.cuda
CUDA_AVAILABLE = False
DEVICE = "cpu"
GPU_NAME = "N/A"
GPU_MEMORY = "N/A"
CUDA_VERSION = "N/A"
USE_HALF_PRECISION = False

try:
    import torch
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        CUDA_AVAILABLE = True
        DEVICE = "cuda"
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        CUDA_VERSION = torch.version.cuda or "N/A"
        USE_HALF_PRECISION = True
except (ImportError, ModuleNotFoundError, RuntimeError, Exception):
    # Si torch no está disponible o CUDA falla, seguimos en CPU
    pass

# ============================================================
# Detección facial (SCRFD via InsightFace buffalo_l)
# ============================================================
SCRFD_DET_THRESHOLD = 0.85      # Confianza mínima de detección SCRFD
SCRFD_NMS_THRESHOLD  = 0.40     # Umbral NMS

# ============================================================
# Calidad de imagen (quality_gate)
# ============================================================
MIN_FACE_AREA    = 112 * 112    # Área mínima del bbox en px² para ArcFace
MAX_ROLL_DEGREES = 25           # Inclinación máxima tolerada (estimada con kps)
BRIGHTNESS_MIN   = 40           # Brillo medio mínimo (rango 0-255)
BRIGHTNESS_MAX   = 220          # Brillo medio máximo (rango 0-255)
BLUR_THRESHOLD   = 80.0         # Varianza Laplaciano mínima (< = borrosa)

# ============================================================
# Enrollment — galería multi-frame
# ============================================================
MIN_SAMPLES_PER_STUDENT  = 5    # Mínimo de rostros para enrollment completo
ENROLLMENT_FRAMES        = 8    # Frames objetivo a capturar por alumno
ENROLLMENT_MIN_SIM       = 0.70 # Similitud mínima para no descartar embedding como outlier

# Poses objetivo para enrollment guiado multiaxial
ENROLLMENT_POSES = [
    {"label": "Mira al frente",                "yaw":  0,  "pitch":  0, "key": "frontal", "count": 3},
    {"label": "Gira levemente a la derecha",   "yaw": 15,  "pitch":  0, "key": "right",   "count": 2},
    {"label": "Gira levemente a la izquierda", "yaw":-15,  "pitch":  0, "key": "left",    "count": 2},
    {"label": "Levanta un poco la cabeza",     "yaw":  0,  "pitch": 10, "key": "up",      "count": 1},
]

# Tolerancia angular para aceptar una muestra como "en pose"
POSE_YAW_TOLERANCE   = 8   # grados
POSE_PITCH_TOLERANCE = 8   # grados

# Multi-centroide — rangos de clasificación por cluster
POSE_FRONTAL_MAX_YAW    = 10  # |yaw| < 10  → cluster frontal
POSE_SIDE_MIN_YAW       = 10  # |yaw| >= 10 → cluster right/left
POSE_VERTICAL_MIN_PITCH =  8  # |pitch| >= 8 → cluster up/down

# ============================================================
# Reconocimiento — ArcFace L2-normalizado
# ============================================================
THRESHOLD_SECURE    = 0.72   # Coincidencia directa segura (marcar asistencia inmediata)
THRESHOLD_GREY_LOW  = 0.65   # Límite inferior de zona gris (validar con galería)
GALLERY_VOTE_RATIO  = 0.60   # Fracción mínima de votos en galería para aceptar match

# ============================================================
# Preprocesamiento (legacy — mantener compatibilidad)
# ============================================================
MIN_FACE_SIZE = 80              # Píxeles mínimos (ancho o alto) para aceptar un rostro
MAX_FACES_PER_FRAME = 1         # Para enrollment solo 1 rostro por frame
FACE_OUTPUT_SIZE = (224, 224)   # Tamaño final del rostro recortado (W, H)
APPLY_CLAHE = True              # Ecualización adaptativa de histograma
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# ============================================================
# Data Augmentation
# ============================================================
ENABLE_AUGMENTATION      = True
AUGMENTATIONS_PER_IMAGE  = 5     # Variantes generadas por cada imagen original
AUGMENT_FLIP_HORIZONTAL  = True  # Simula ángulo opuesto leve
AUGMENT_YAW_RANGE        = 15    # Rotación 2D simulada ±15°
AUGMENT_BRIGHTNESS_RANGE = 0.20  # Variación de brillo ±20%



