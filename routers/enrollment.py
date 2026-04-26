"""
POST /enroll            — Registro de nuevo estudiante (v3.0: enrollment guiado por pose).
POST /enroll/check-pose — Verifica si la pose actual cumple con el ángulo objetivo.

Pipeline v3.0:
  1. Recibir name, code, images[]
  2. Para cada imagen:
     a. SCRFD-34GF → detectar rostro (get_largest_face)
     b. quality_gate(check_blur=True) → rechazar frames de baja calidad Y borrosos
     c. get_pose() → etiquetar embedding con cluster de pose
     d. norm_crop 112×112 → guardar crop en dataset/{code}/faces/
     e. ArcFace ResNet100 → embedding (512,) ya L2-normalizado
  3. Filtrar outliers entre los embeddings acumulados (ENROLLMENT_MIN_SIM)
  4. Calcular centroide global + centroides por pose → guardar
     embedding_mean.npy, gallery.npy, centroids.npz
  5. Guardar personal_threshold en metadata.json
  6. Insertar/actualizar estudiante en SQLite
  7. Notificar a Next.js para actualizar face_registered en Supabase
"""
import io
import json
import os
import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import config

from dataset_manager import DatasetManager

NEXT_APP_URL = os.getenv("NEXT_APP_URL", "http://localhost:3000")
AI_SERVER_SECRET = os.getenv("AI_SERVER_SECRET", "")

router = APIRouter()


def get_recognizer(request: Request):
    return request.app.state.recognizer


def get_embedding_manager(request: Request):
    return request.app.state.embedding_manager


def get_attendance_db(request: Request):
    return request.app.state.db


def _log_enrollment_rejection(codigo: str, reason: str):
    """Registra en enrollment_log.json cada frame rechazado durante el enrollment."""
    try:
        log_path = config.DATASET_DIR / codigo / "enrollment_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.datetime.now().isoformat(), "reason": reason}
        entries = []
        if log_path.exists():
            try:
                entries = json.loads(log_path.read_text())
            except Exception:
                entries = []
        entries.append(entry)
        log_path.write_text(json.dumps(entries, indent=2))
    except Exception as e:
        print(f"[enroll] Advertencia log: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# POST /enroll/check-pose
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/enroll/check-pose")
async def check_pose(
    request: Request,
    target_yaw: float = Form(...),
    target_pitch: float = Form(...),
    image: UploadFile = File(...),
):
    """
    Recibe UN frame y verifica si la pose actual del rostro coincide con la pose objetivo.

    Usado por el frontend de enrollment guiado para confirmar cada ángulo antes de capturar.

    Returns:
        {
          "pose_ok":      bool,
          "face_found":   bool,
          "current_pose": {yaw, pitch, roll} | None,
          "target":       {yaw, pitch},
          "delta":        {yaw, pitch}  — cuánto le falta para llegar a la pose objetivo
        }
    """
    recognizer = get_recognizer(request)

    raw = await image.read()
    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(
            status_code=422,
            content={"pose_ok": False, "face_found": False, "reason": "frame_invalido"},
        )

    face = recognizer.get_largest_face(frame)

    if face is None:
        return JSONResponse(content={
            "pose_ok":      False,
            "face_found":   False,
            "current_pose": None,
            "target":       {"yaw": target_yaw, "pitch": target_pitch},
            "delta":        {"yaw": 0.0, "pitch": 0.0},
        })

    pose = recognizer.get_pose(face)
    delta_yaw   = round(pose["yaw"]   - target_yaw,   1)
    delta_pitch = round(pose["pitch"] - target_pitch, 1)

    yaw_ok   = abs(delta_yaw)   <= config.POSE_YAW_TOLERANCE
    pitch_ok = abs(delta_pitch) <= config.POSE_PITCH_TOLERANCE

    return JSONResponse(content={
        "pose_ok":      bool(yaw_ok and pitch_ok),
        "face_found":   True,
        "current_pose": pose,
        "target":       {"yaw": target_yaw, "pitch": target_pitch},
        "delta":        {"yaw": delta_yaw, "pitch": delta_pitch},
    })


# ──────────────────────────────────────────────────────────────────────────────
# POST /enroll
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/enroll")
async def enroll_student(
    request: Request,
    name: str = Form(...),
    code: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """
    Registra un nuevo estudiante con sus imágenes faciales.

    - **name**:   Nombre completo del estudiante
    - **code**:   Código único del estudiante (ej: EST-2024001)
    - **images**: 5–10 imágenes JPEG/PNG del rostro (idealmente desde enrollment guiado)
    """
    recognizer   = get_recognizer(request)
    emb_manager  = get_embedding_manager(request)
    db           = get_attendance_db(request)

    if not images:
        raise HTTPException(status_code=422, detail="Se requiere al menos 1 imagen.")

    dm = DatasetManager(config.DATASET_DIR)
    dm.create_student(code, name)

    embeddings_collected: List[np.ndarray] = []
    poses_collected: List[dict] = []
    faces_saved      = 0
    errors           = []
    quality_rejected = 0

    for i, upload in enumerate(images):
        raw = await upload.read()
        nparr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            errors.append(f"Imagen {i+1}: no se pudo decodificar")
            continue

        # Detectar rostro con SCRFD
        face = recognizer.get_largest_face(frame)

        if face is None:
            errors.append(f"Imagen {i+1}: no se detectó rostro")
            continue

        # ── Filtro de calidad con blur check activo ────────────────────
        ok, reason = recognizer.quality_gate(face, frame, check_blur=True)
        if not ok:
            errors.append(f"Imagen {i+1}: rechazada por calidad — {reason}")
            quality_rejected += 1
            _log_enrollment_rejection(code, reason)
            continue

        if not hasattr(face, "embedding") or face.embedding is None:
            errors.append(f"Imagen {i+1}: no se pudo extraer embedding")
            continue

        # ── Estimación de pose ─────────────────────────────────────────
        pose_info = recognizer.get_pose(face)
        poses_collected.append(pose_info)

        # Crop alineado 112×112
        aligned = recognizer.get_aligned_crop(frame, face, image_size=112)

        # Guardar usando dataset_manager
        quality_score = float(
            cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        )
        dm.save_face_image(code, aligned, quality_score)
        faces_saved += 1

        # Acumular embedding
        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        embeddings_collected.append(emb)

    if not embeddings_collected:
        raise HTTPException(
            status_code=422,
            detail=f"No se procesó ningún rostro válido. Errores: {errors}",
        )

    # ── Galería + centroide global + centroides por pose (v3.0) ────────
    _, _, centroid = emb_manager.save_embeddings(
        code,
        embeddings_collected,
        poses_list=poses_collected,
        filter_outliers=True,
    )

    # ── Umbral personal ─────────────────────────────────────────────────
    if len(embeddings_collected) >= 3:
        gallery      = np.stack(embeddings_collected)
        pair_sims    = gallery @ centroid
        mean_sim     = float(pair_sims.mean())
        std_sim      = float(pair_sims.std())
        personal_threshold = float(np.clip(
            mean_sim - 1.5 * std_sim,
            config.THRESHOLD_GREY_LOW,
            config.THRESHOLD_SECURE,
        ))
    else:
        personal_threshold = config.THRESHOLD_SECURE

    dm.update_student_metadata(code, {"personal_threshold": round(personal_threshold, 4)})
    print(f"[enroll] {code}: personal_threshold={personal_threshold:.4f}")

    # Registrar en SQLite
    await db.upsert_student(code, name, face_count=faces_saved)

    # Recargar índice en memoria
    request.app.state.embeddings_index = emb_manager.load_all_embeddings()

    # Notificar a Next.js → actualiza face_registered en Supabase
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{NEXT_APP_URL}/api/enroll",
                json={"student_code": code, "name": name, "faces_saved": faces_saved},
                headers={"x-ai-server-secret": AI_SERVER_SECRET},
            )
    except Exception as e:
        print(f"[enroll] Advertencia: no se pudo notificar a Next.js: {e}")

    # Resumen de clusters de pose capturados
    pose_summary: dict = {}
    for p in poses_collected:
        key = emb_manager._classify_pose(p)
        pose_summary[key] = pose_summary.get(key, 0) + 1

    return JSONResponse(
        status_code=200,
        content={
            "status":             "enrolled",
            "student_code":       code,
            "name":               name,
            "faces_saved":        faces_saved,
            "quality_rejected":   quality_rejected,
            "personal_threshold": round(personal_threshold, 4),
            "pose_clusters":      pose_summary,
            "errors":             errors,
        },
    )
