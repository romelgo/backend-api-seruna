# PLAN: Asistencia Precisión ≥ 98% en condiciones de aula reales Tiempo Real
> **Versión:** 2.0 — Optimizado para IA  
> **Stack:** Python · InsightFace · SCRFD · ArcFace (ResNet100) · NumPy  
> **Objetivo:** Precisión ≥ 98% en condiciones de aula reales

---

## 0. CONTEXTO Y PROBLEMA

```
ENTRADA  → Cámara en tiempo real (web)
PROCESO  → Detectar rostro → Generar embedding → Comparar con BD
SALIDA   → Marcar asistencia del alumno identificado
PROBLEMA → Baja precisión por: embeddings ruidosos, umbral fijo, sin alineación
```

El sistema actual falla porque:
1. Guarda 1 solo `.npy` sin alinear el rostro
2. No normaliza los vectores antes de guardar
3. Usa distancia euclidiana cruda sin normalización L2
4. No filtra calidad en el registro

---

## 1. ARQUITECTURA DEL SISTEMA (2 MÓDULOS)

```
┌─────────────────────────────────────────────────────────┐
│  MÓDULO A: ENROLAMIENTO (registro de alumno)            │
│                                                          │
│  Cámara → SCRFD (detectar) → Alinear (kps afín)         │
│        → ArcFace (embedding 512d) → Normalizar L2       │
│        → Repetir 5-10 veces → Guardar galería + media   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  MÓDULO B: RECONOCIMIENTO (asistencia tiempo real web)   │
│                                                          │
│  Cámara → SCRFD → Filtro calidad → Alinear → ArcFace    │
│        → Normalizar → Cosine Similarity con BD          │
│        → Umbral → Marcar asistencia con timestamp       │
└──────────────────────────────────────────────────────────┘
```

---

## 2. PIPELINE DETALLADO — MÓDULO A: ENROLAMIENTO

### 2.1 Detección con SCRFD
```python
# SCRFD entrega por cada rostro:
face.bbox   # [x1, y1, x2, y2] bounding box
face.kps    # 5 keypoints: ojo_izq, ojo_der, nariz, boca_izq, boca_der
face.det_score  # confianza de detección (float 0-1)
```

**FILTRO DE CALIDAD — rechazar si:**
| Condición | Umbral de rechazo | Razón |
|-----------|-------------------|-------|
| `det_score` bajo | `< 0.90` | Detección insegura |
| Área del rostro | `< 112 × 112 px` | Muy pequeño para ArcFace |
| Pose lateral (yaw) | `> 30°` | Embedding inconsistente |
| Iluminación | histograma < 50 o > 200 | Sub/sobre-exposición |

### 2.2 Alineación Facial (CRÍTICO)
```python
# InsightFace provee esta función internamente:
from insightface.utils import face_align

# Plantilla canónica de 5 puntos para 112x112
TEMPLATE_112 = numpy.array([
    [38.2946, 51.6963],  # ojo izquierdo
    [73.5318, 51.5014],  # ojo derecho
    [56.0252, 71.7366],  # nariz
    [41.5493, 92.3655],  # boca izquierda
    [70.7299, 92.2041],  # boca derecha
], dtype=numpy.float32)

aligned_face = face_align.norm_crop(img, landmark=face.kps, image_size=112)
# Resultado: imagen 112×112 con rostro centrado y ojos horizontales
```

**POR QUÉ ES CRÍTICO:**  
ResNet100 fue entrenado con rostros alineados. Un rostro inclinado 15° puede bajar la similitud coseno de 0.85 → 0.60 con el mismo individuo.

### 2.3 Generación del Embedding
```python
# ArcFace ResNet100 genera vector de 512 dimensiones
embedding_raw = arcface_model.get_feat(aligned_face)  
# Shape: (512,) — dtype: float32
```

### 2.4 Normalización L2 (OBLIGATORIO)
```python
import numpy as np

def normalize_l2(embedding: np.ndarray) -> np.ndarray:
    """Normaliza el vector a norma unitaria."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

embedding_norm = normalize_l2(embedding_raw)
# ||embedding_norm|| == 1.0  ← garantizado
```

**POR QUÉ:** Si el vector no está normalizado, la similitud coseno varía con la magnitud del vector, no solo con el ángulo. Al normalizar, cos_sim = dot_product, que es O(n) y muy rápido.

### 2.5 Estrategia de Enrolamiento Multi-Frame (HÍBRIDA)
```
Capturar 10 frames válidos del alumno
         │
         ├─→ Calcular embedding de cada uno (10 vectores)
         │
         ├─→ Filtrar outliers: descartar embeddings con similitud 
         │   < 0.70 respecto a la mediana del grupo
         │
         ├─→ Calcular CENTROIDE (media de los vectores restantes)
         │
         ├─→ Normalizar el centroide → embedding_mean.npy
         │
         └─→ Guardar los N vectores individuales → gallery.npy
```

**Estructura de archivos por alumno:**
```
dataset/
└── 00001(codigo del alumno)/
    ├── embedding_mean.npy   # centroide normalizado (512,)
    ├── gallery.npy          # galería individual (N, 512)
    └── metadata.json        # nombre, ID, fecha registro, umbral personal
```

---

## 3. PIPELINE DETALLADO — MÓDULO B: RECONOCIMIENTO

### 3.1 Loop de Tiempo Real
```
CADA FRAME:
  1. SCRFD detecta rostros (todos los presentes en escena)
  2. Por cada rostro detectado:
     a. Filtro calidad rápido (det_score > 0.85)
     b. Alineación afín con kps
     c. ArcFace → embedding → normalizar L2
     d. Comparar con BD → similitud coseno
     e. Si similitud > umbral → candidato identificado

```

### 3.2 Función de Similitud Coseno
```python
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Requiere vectores NORMALIZADOS L2.
    Si ambos están normalizados: cos_sim = dot product (más rápido).
    Rango: [-1, 1] → para rostros: [0.0, 1.0]
    """
    return float(np.dot(vec_a, vec_b))
```

### 3.3 Estrategia de Comparación Híbrida (Velocidad + Precisión)
```python
UMBRAL_RAPIDO = 0.72    # comparación con centroide
UMBRAL_ZONA_GRIS_INF = 0.65  # inicio zona de duda
UMBRAL_ZONA_GRIS_SUP = 0.72  # fin zona de duda

def identify_face(query_emb, student_db):
    for student_id, data in student_db.items():
        
        # PASO 1: comparar con centroide (O(1) por alumno)
        sim_mean = cosine_similarity(query_emb, data['embedding_mean'])
        
        if sim_mean >= UMBRAL_RAPIDO:
            # Coincidencia segura → marcar asistencia
            return student_id, sim_mean, "SEGURO"
        
        elif sim_mean >= UMBRAL_ZONA_GRIS_INF:
            # PASO 2: zona gris → validar con galería completa
            sims = [cosine_similarity(query_emb, g) for g in data['gallery']]
            votos_positivos = sum(1 for s in sims if s >= UMBRAL_ZONA_GRIS_INF)
            
            if votos_positivos >= len(sims) * 0.6:  # 60% de votos
                return student_id, max(sims), "GALERIA"
    
    return None, 0.0, "DESCONOCIDO"
```

### 3.4 Tabla de Umbrales Recomendados
| Escenario | Umbral Coseno | Acción |
|-----------|--------------|--------|
| `>= 0.75` | Coincidencia segura | Marcar asistencia inmediata |
| `0.65 – 0.75` | Zona gris | Validar con galería |
| `< 0.65` | No reconocido | Ignorar o loguear |

> **Nota:** Estos umbrales son para ArcFace ResNet100 con vectores L2-normalizados. Ajustar tras pruebas con el grupo de alumnos específico.

---

## 4. CONTROL DE CALIDAD EN TIEMPO REAL

### 4.1 Filtros Pre-Embedding (rápidos, antes de ArcFace)
```python
def quality_gate(face, img) -> tuple[bool, str]:
    """Retorna (pasa, motivo_rechazo)"""
    
    # 1. Confianza de detección SCRFD
    if face.det_score < 0.85:
        return False, "det_score_bajo"
    
    # 2. Tamaño mínimo del bounding box
    x1, y1, x2, y2 = face.bbox
    area = (x2 - x1) * (y2 - y1)
    if area < (112 * 112):
        return False, "rostro_pequeno"
    
    # 3. Estimación de pose (usando keypoints)
    eye_dx = face.kps[1][0] - face.kps[0][0]  # delta x entre ojos
    eye_dy = face.kps[1][1] - face.kps[0][1]  # delta y entre ojos
    roll = np.degrees(np.arctan2(eye_dy, eye_dx))
    if abs(roll) > 25:
        return False, "pose_rotada"
    
    # 4. Iluminación (histograma del ROI)
    roi = img[int(y1):int(y2), int(x1):int(x2)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < 40 or mean_brightness > 220:
        return False, "iluminacion"
    
    return True, "ok"
```

### 4.2 Anti-Duplicados (Ventana Temporal)
```python
# Evitar marcar asistencia múltiple del mismo alumno

```

---

## 5. ESTRUCTURA DE CÓDIGO RECOMENDADA

```
proyecto_asistencia/
├── models/
│   ├── scrfd_10g_bnkps.onnx       # detector SCRFD
│   └── w600k_r50.onnx             # ArcFace ResNet100
│
├── dataset/
│       └── {id_alumno}/
│           ├── embedding_mean.npy
│           ├── gallery.npy
│           └── metadata.json
│


```

---

## 6. CARGA EFICIENTE DE BASE DE DATOS DE EMBEDDINGS

```python
import numpy as np
import os, json

def load_student_database(data_dir: str) -> dict:
    """
    Carga todos los embeddings en memoria al inicio.
    Estructura resultado:
    {
        "alumno_001": {
            "name": "Juan Pérez",
            "embedding_mean": np.array (512,),
            "gallery": np.array (N, 512)
        },
        ...
    }
    """
    db = {}
    for student_id in os.listdir(data_dir):
        student_path = os.path.join(data_dir, student_id)
        
        mean_path = os.path.join(student_path, "embedding_mean.npy")
        gallery_path = os.path.join(student_path, "gallery.npy")
        meta_path = os.path.join(student_path, "metadata.json")
        
        if not os.path.exists(mean_path):
            continue
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        db[student_id] = {
            "name": meta["name"],
            "embedding_mean": np.load(mean_path),
            "gallery": np.load(gallery_path) if os.path.exists(gallery_path) else None
        }
    
    return db

# Vectorizar comparaciones con toda la BD (más rápido con numpy)
def batch_cosine_similarity(query: np.ndarray, db_matrix: np.ndarray) -> np.ndarray:
    """
    query:     (512,)   — vector del rostro detectado
    db_matrix: (N, 512) — matriz con todos los centroides
    retorna:   (N,)     — similitudes coseno
    """
    return db_matrix @ query  # dot product (vectores ya normalizados L2)
```








## 9. PARÁMETROS DE CONFIGURACIÓN GLOBALES

```python
# config.py — todos los hiperparámetros en un lugar

CONFIG = {
    # Modelos
    "scrfd_model": "models/scrfd_10g_bnkps.onnx",
    "arcface_model": "models/w600k_r50.onnx",
    
    # Detección
    "scrfd_det_threshold": 0.85,      # confianza mínima SCRFD
    "scrfd_nms_threshold": 0.40,
    
    # Calidad
    "min_face_area": 112 * 112,       # píxeles²
    "max_roll_degrees": 25,            # grados de inclinación
    "min_brightness": 40,              # 0-255
    "max_brightness": 220,
    
    # Enrolamiento
    "enrollment_frames": 8,            # frames a capturar
    "enrollment_min_similarity": 0.70, # filtro outliers en galería
    
    # Reconocimiento
    "threshold_secure": 0.72,          # coincidencia segura
    "threshold_grey_zone": 0.65,       # inicio zona gris
    "gallery_vote_ratio": 0.60,        # % votos para aceptar en zona gris
 
}
```



