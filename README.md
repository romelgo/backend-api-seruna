# Backend - Control de Asistencia Facial

Este módulo comprende el backend desarrollado en [FastAPI](https://fastapi.tiangolo.com/) y Python, diseñado para gestionar el registro, la identificación por reconocimiento facial y el control de asistencia de los estudiantes.

## Arquitectura

El servidor utiliza una arquitectura basada en eventos (Lifespan) que carga los pesados modelos de machine learning en memoria (en la VRAM/GPU o CPU) una sola vez al inicio del sistema, para lograr máxima velocidad durante la inferencia en tiempo real. 

El estado global (`app.state`) mantiene el acceso centralizado a:
1. `FaceRecognizer`: Motor de detección y reconocimiento (InsightFace).
2. `EmbeddingManager`: Indexación y gestión de datos faciales almacenados en disco.
3. `AttendanceDB`: Base de datos SQLite asíncrona para rastrear la asistencia.
4. `embeddings_index`: Estructura cargada en memoria que agiliza el matcheo o comparación de los rostros en el momento.

## Tecnologías Principales

- **FastAPI**: Framework web rápido (ASGI).
- **InsightFace**: Biblioteca de reconocimiento facial de última generación (Modelos: SCRFD-34GF para Detección y ArcFace ResNet100 para incrustación de características).
- **ONNXRuntime**: Inferencia rápida e interoperable para los modelos de InsightFace (Soporta GPU `onnxruntime-gpu`).
- **AioSQLite**: ORM asíncrono básico para interactuar con SQLite.
- **OpenCV**: Manejo subyacente y preprocesamiento de imágenes.

## Estructura del Código

- **`main.py`**: El punto de entrada central, inicializa la app, ensambla los routers, configura variables de entorno y maneja el ciclo de vida (carga inicial de todos los modelos de IA y DB).
- **`core/`**:
  - `face_recognizer.py`: Envuelve la lógica de InsightFace para abstraer la red neuronal (extrae rostros y genera `embeddings` o vectores matemáticos de los rostros).
  - `embedding_manager.py`: Controla la lectura, indexación y guardado de los rostros enrolados como archivos de numpy en el directorio físico del `.dataset/`.
  - `attendance.py`: Lógica principal conectada a la BD de asistencias, manejando el registro histórico y horarios de la llegada.
- **`routers/`**: Submódulos que dividen y exponen los endpoints funcionales de la API:
  - `enrollment.py`: Rutas (`/enroll`) para añadir nuevos estudiantes mediante análisis inicial de su fotografía y guardarlos en el indexado.
  - `identification.py`: Rutas (`/identify`) responsables de tomar un frame (video/foto en tiempo real), mandar a inferir las caras vs el índice y anotar/marcar la asistencia validada del estudiante.
  - `students.py`: Rutas secundarias para manejo administrativo y visualización de la lista de estudiantes enrolados y reportes.

## Variables de Entorno

El `.env` o la configuración del sistema suele requerir parámetros como:
- `CUDA_VISIBLE_DEVICES` y `INSIGHTFACE_CTX_ID`: Configuración y selector de GPU si hay hardware dedicado disponible.
- `DATASET_DIR`: Path donde residen los rostros guardados (Embeddings `.npy` o imágenes crudas).
- `DB_URL`: Ruta del archivo `.db` para SQLite local.
- `SIMILARITY_THRESHOLD`: Nivel matemático mínimo (ejemplo 0.5) para que el servidor dictamine que el rostro en cámara "hace match positivo" con un alumno registrado.
- `CORS_ORIGINS`: Dominio o IPs del panel o interfaz web permitidos.

## Inicialización Local y Dependencias

Para levantar este módulo en un entorno de desarrollo:

```bash
# Recomendable configurar un entorno virtual venv previamente
pip install -r requirements.txt

# Para correr en modo dev con auto-recarga:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
La interfaz de tipo Swagger estará automáticamente disponible en `http://localhost:8000/docs` una vez levantada.
