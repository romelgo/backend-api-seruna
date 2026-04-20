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
# Modelo YOLO-Face
# ============================================================
# Modelo pre-entrenado en WIDERFace (se descarga automáticamente)
# Con GPU: usar yolov8m-face.pt (más preciso, GPU lo maneja sin problema)
# Sin GPU: usar yolov8n-face.pt (más liviano para CPU)
YOLO_MODEL_NAME = "yolov8m-face.pt" if CUDA_AVAILABLE else "yolov8n-face.pt"
YOLO_MODEL_REPO = "https://github.com/akanametov/yolo-face"
YOLO_CONFIDENCE = 0.5          # Umbral mínimo de confianza
YOLO_IOU_THRESHOLD = 0.45      # Non-Maximum Suppression IoU
YOLO_IMG_SIZE = 640             # Tamaño de entrada del modelo

# ============================================================
# Captura y detección
# ============================================================
MIN_FACE_SIZE = 80              # Píxeles mínimos (ancho o alto) para aceptar un rostro
MAX_FACES_PER_FRAME = 1         # Para enrollment solo 1 rostro por frame
FACE_PADDING = 0.25             # 25% padding alrededor del bbox del rostro

# ============================================================
# Preprocesamiento
# ============================================================
FACE_OUTPUT_SIZE = (224, 224)   # Tamaño final del rostro recortado (W, H)
APPLY_CLAHE = True              # Ecualización adaptativa de histograma
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# ============================================================
# Calidad de imagen
# ============================================================
BLUR_THRESHOLD = 80.0           # Varianza del Laplaciano mínima (< = borrosa)
BRIGHTNESS_MIN = 40             # Brillo medio mínimo aceptable
BRIGHTNESS_MAX = 220            # Brillo medio máximo aceptable

# ============================================================
# Enrollment
# ============================================================
MIN_SAMPLES_PER_STUDENT = 5     # Mínimo de rostros para considerar enrollment completo
MAX_SAMPLES_PER_STUDENT = 20    # Máximo de capturas por estudiante
WEBCAM_CAPTURE_DELAY_MS = 500   # Delay entre capturas automáticas (ms)

# ============================================================
# Data Augmentation
# ============================================================
ENABLE_AUGMENTATION = True
AUGMENTATIONS_PER_IMAGE = 3     # Variantes generadas por cada imagen original



