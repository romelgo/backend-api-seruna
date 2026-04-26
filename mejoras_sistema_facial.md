# Mejoras y Correcciones — Sistema de Reconocimiento Facial
> Sistema: InsightFace `buffalo_l` + ArcFace ResNet100 + SCRFD-34GF  
> Objetivo principal: Reconocimiento robusto en **ángulos diferentes**

---

## 🔴 Prioridad Alta — Crítico para reconocimiento multi-ángulo

### 1. Enrollment guiado por pose (captura multiaxial)

**Problema actual:**  
La ráfaga de 8 frames con intervalo de 350ms captura poses casi idénticas (frontal). No hay variedad de ángulo real en la galería del alumno.

**Por qué importa:**  
ArcFace ResNet100 puede manejar hasta ±30° de yaw, pero solo si el embedding fue construido con muestras de esos ángulos. Sin diversidad angular en enrollment, el reconocimiento de lado falla aunque el modelo sea capaz.

**Solución:**  
Reemplazar la ráfaga ciega por un flujo guiado que espere activamente una pose antes de capturar cada muestra.

```python
# config.py — agregar poses objetivo
ENROLLMENT_POSES = [
    {"label": "Mira al frente",              "yaw": 0,   "pitch": 0},
    {"label": "Gira levemente a la derecha", "yaw": 15,  "pitch": 0},
    {"label": "Gira levemente a la izquierda","yaw": -15, "pitch": 0},
    {"label": "Levanta un poco la cabeza",   "yaw": 0,   "pitch": 10},
    {"label": "Baja un poco la cabeza",      "yaw": 0,   "pitch": -10},
]

# Tolerancia angular para aceptar una muestra como "en pose"
POSE_YAW_TOLERANCE   = 8   # grados
POSE_PITCH_TOLERANCE = 8   # grados
```

**En el backend (enrollment):**  
InsightFace devuelve `pose` (yaw, pitch, roll) por cada rostro detectado. Usar esos valores para validar que la captura corresponde a la pose solicitada antes de guardar el embedding.

**En el frontend:**  
Mostrar instrucción visual al alumno ("Gira a la derecha") y confirmar visualmente cuando se captura cada ángulo. Avanzar a la siguiente pose automáticamente.

---

### 2. Multi-centroide por cluster de pose

**Problema actual:**  
El sistema calcula un único centroide (`mean` de todos los embeddings) por alumno. Promediar embeddings frontales con embeddings de perfil degrada la calidad de cada centroide.

**Por qué importa:**  
Un centroide mezclado tiene menor similitud con cualquier vista específica. Usar centroides separados por rango de pose mejora el score de similitud entre 5–10 puntos porcentuales.

**Solución:**  
Al guardar embeddings en enrollment, etiquetar cada uno con su rango de pose y calcular un centroide por cluster.

```python
# Al guardar en dataset
student_data = {
    "centroids": {
        "frontal": mean(embeddings donde abs(yaw) < 10),
        "right":   mean(embeddings donde yaw >= 10),
        "left":    mean(embeddings donde yaw <= -10),
        "up":      mean(embeddings donde pitch >= 8),
        "down":    mean(embeddings donde pitch <= -8),
    },
    "gallery": all_embeddings  # galería completa sin cambios
}

# Al identificar: comparar contra el centroide más cercano a la pose actual
def get_best_centroid(student_data, current_pose):
    # Seleccionar centroide según yaw/pitch del rostro detectado
    ...
```

---

## 🟡 Prioridad Media — Mejoras de precisión y usabilidad

### 3. Usar `pose` de SCRFD en el kiosk para retroalimentación

**Problema actual:**  
El kiosk intenta identificar cualquier rostro sin considerar si el ángulo es demasiado extremo para el modelo. Ángulos >35° producen falsos negativos silenciosos.

**Solución:**  
Devolver `yaw` y `pitch` en la respuesta de `/identify` y mostrar en el frontend un aviso si el ángulo supera el umbral reconocible.

```python
# Respuesta de /identify — agregar campo pose
{
  "match": true,
  "student_id": "A001",
  "similarity": 0.78,
  "pose": {"yaw": 22.5, "pitch": -5.1, "roll": 3.0},  # ← nuevo
  "bbox": [x1, y1, x2, y2]
}
```

```javascript
// Frontend kiosk — mostrar aviso si ángulo es muy extremo
if (Math.abs(response.pose.yaw) > 30) {
  showHint("Mira más hacia la cámara");
}
```

---

### 4. Reducir cooldown y usar posición del bbox

**Problema actual:**  
El cooldown fijo de 8 segundos es excesivo en pasillos con varios alumnos pasando seguido. Puede bloquear la identificación de la siguiente persona si se acerca antes de que expire.

**Solución:**  
Bajar a 2-3 segundos Y agregar detección de "nueva persona" basada en la posición del bounding box. Si el centroide del bbox se desplaza más de N píxeles entre frames, reiniciar el cooldown anticipadamente.

```javascript
// config frontend
const COOLDOWN_MS = 3500;
const BBOX_DISPLACEMENT_THRESHOLD = 120; // píxeles

function isNewPerson(prevBbox, currentBbox) {
  const prevCx = (prevBbox[0] + prevBbox[2]) / 2;
  const currCx = (currentBbox[0] + currentBbox[2]) / 2;
  return Math.abs(currCx - prevCx) > BBOX_DISPLACEMENT_THRESHOLD;
}
```

---

### 5. Verificar que `BLUR_THRESHOLD` se aplica también en enrollment

**Problema actual:**  
`BLUR_THRESHOLD = 80.0` está definido en `config.py` pero no se menciona explícitamente como parte del quality gate durante el enrollment, solo durante reconocimiento.

**Acción:**  
Confirmar que el filtro de varianza Laplaciana se ejecuta en el flujo de `/enroll` antes de guardar cada embedding. Si no está activo, agregar:

```python
# En quality_gate.py o equivalente — aplicar en enrollment también
laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
if laplacian_var < BLUR_THRESHOLD:
    return {"rejected": True, "reason": "imagen_borrosa"}
```

---

## 🟢 Prioridad Baja — Optimizaciones incrementales

### 6. Aumentar augmentaciones con rotación para compensar falta de ángulos

**Contexto:**  
Mientras se implementa la captura multiaxial, se puede mejorar la cobertura angular de forma sintética aumentando las augmentaciones.

```python
# config.py
AUGMENTATIONS_PER_IMAGE = 5  # subir de 3 a 5

# En el módulo de augmentación, agregar:
# - Flip horizontal (simula ángulo opuesto leve)
# - Rotación aleatoria ±15° en yaw (simulado con transformación 2D)
# - Variación de brillo ±20%
```

> ⚠️ Esto es un parche temporal. No reemplaza tener muestras reales de múltiples ángulos.

---

### 7. Logging de rechazos del quality gate

**Sugerencia:**  
Registrar en un log (archivo o DB) cada vez que el quality gate rechaza una muestra durante enrollment, incluyendo el motivo (`blur`, `roll`, `brightness`, `area`, `detector_confidence`). Esto permite identificar si los alumnos con mal reconocimiento tienen pocas muestras por causas técnicas.

```python
# Estructura de log sugerida
{
  "student_id": "A001",
  "timestamp": "2025-04-25T10:32:00",
  "rejected_frames": 3,
  "reasons": {"blur": 2, "brightness": 1},
  "accepted_frames": 5
}
```

---

## Resumen de Prioridades

| # | Mejora | Área | Impacto en ángulos | Esfuerzo |
|---|--------|------|--------------------|----------|
| 1 | Enrollment guiado por pose | Backend + Frontend | 🔴 Alto | Alto |
| 2 | Multi-centroide por cluster de pose | Backend | 🔴 Alto | Medio |
| 3 | Devolver `pose` en `/identify` + hint visual | Backend + Frontend | 🟡 Medio | Bajo |
| 4 | Reducir cooldown + detección por bbox | Frontend | 🟡 Medio | Bajo |
| 5 | Verificar blur en enrollment | Backend | 🟡 Medio | Muy bajo |
| 6 | Más augmentaciones con rotación | Backend | 🟢 Bajo-Medio | Bajo |
| 7 | Logging de rechazos | Backend | 🟢 Mantenimiento | Bajo |

---

## Notas de Arquitectura

- El stack actual (SCRFD + ArcFace ResNet100 `buffalo_l`) **es capaz** de manejar ángulos de hasta ±30° yaw. El problema no es el modelo, es la **falta de muestras angulares en enrollment**.
- Los umbrales actuales (72% seguro / 65% zona gris) son conservadores y correctos. No cambiarlos hasta tener datos reales post-mejora.
- El flujo híbrido (centroide + votación de galería) es una buena arquitectura. El multi-centroide la extiende sin romperla.
