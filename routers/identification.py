import os
import sys
import base64
import time
import uuid
import datetime
from pathlib import Path

import cv2
import numpy as np
import httpx
from fastapi import APIRouter, BackgroundTasks, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

import config

PROJECT_ROOT = Path(__file__).parent.parent.parent

NEXT_APP_URL = os.getenv("NEXT_APP_URL", "http://localhost:3000")
AI_SERVER_SECRET = os.getenv("AI_SERVER_SECRET", "")

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Cache anti-duplicados en memoria (refuerzo rápido: 60 s por alumno)
# La verificación canónica se hace en Supabase (check_marked_today).
# ─────────────────────────────────────────────────────────────────────────────
_recent_attendances: dict[str, float] = {}
_CACHE_TTL_SECONDS = 60


async def _bg_notify_nextjs(
    student_code: str,
    name: str,
    confidence: float,
    frame_bytes: bytes | None = None,
):
    payload: dict = {
        "student_code": student_code,
        "name": name,
        "confidence": confidence,
    }
    if frame_bytes:
        payload["photo_base64"] = base64.b64encode(frame_bytes).decode()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{NEXT_APP_URL}/api/attendance",
                json=payload,
                headers={"x-ai-server-secret": AI_SERVER_SECRET},
            )
            print(f"[identify] Supabase notify: {resp.status_code}")
    except Exception as e:
        print(f"[identify] Error notificando a Next.js: {e}")


async def _check_marked_today_supabase(student_code: str) -> bool:
    """
    Consulta Supabase para verificar si el alumno ya tiene asistencia hoy.
    Requiere SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY en el entorno.
    Retorna True si ya está marcado, False si no o si hay error.
    """
    supa_url = os.getenv("SUPABASE_URL")
    supa_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supa_url or not supa_key:
        return False

    headers = {"apikey": supa_key, "Authorization": f"Bearer {supa_key}"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # 1. Obtener ID del estudiante
            res_stu = await client.get(
                f"{supa_url}/rest/v1/students"
                f"?student_code=eq.{student_code}&select=id,first_name,last_name",
                headers=headers,
            )
            if res_stu.status_code != 200 or not res_stu.json():
                return False

            student_id = res_stu.json()[0]["id"]

            # 2. Verificar asistencia de hoy
            today_start = (
                datetime.datetime.now()
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .isoformat()
            )
            res_att = await client.get(
                f"{supa_url}/rest/v1/attendance"
                f"?student_id=eq.{student_id}&timestamp=gte.{today_start}&select=id",
                headers=headers,
            )
            return res_att.status_code == 200 and bool(res_att.json())

    except Exception as e:
        print(f"[identify] Error Supabase check: {e}")
        return False


@router.post("/identify")
async def identify_frame(
    request: Request,
    background_tasks: BackgroundTasks,
    frame: UploadFile = File(...),
    course_id: str = Query(default="default", description="ID del curso/sesión"),
):
    recognizer      = request.app.state.recognizer
    emb_manager     = request.app.state.embedding_manager
    embeddings_index = request.app.state.embeddings_index  # Dict con estructura enriquecida

    raw = await frame.read()
    nparr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        return JSONResponse(
            status_code=422,
            content={"faces": [], "error": "No se pudo decodificar el frame"},
        )

    height, width, _ = bgr.shape

    # Detectar todos los rostros del frame
    faces = recognizer.get_faces(bgr)
    results = []

    for face in faces:
        if not hasattr(face, "embedding") or face.embedding is None:
            continue

        bbox = face.bbox.tolist()  # [x1, y1, x2, y2]

        # ── Filtro de calidad pre-embedding (NUEVO) ──────────────────
        ok, reason = recognizer.quality_gate(face, bgr)
        if not ok:
            results.append({
                "bbox": bbox,
                "match": False,
                "face_detected": True,
                "quality_rejected": True,
                "quality_reason": reason,
                "similarity": 0.0,
                "student_code": None,
                "name": "Baja calidad",
                "already_marked": False,
            })
            continue

        query_emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # ── Comparación híbrida (NUEVO) centroide → galería ──────────
        best_code, similarity = emb_manager.find_best_match(
            query_emb,
            embeddings=embeddings_index,
        )

        face_result = {
            "bbox": bbox,
            "match": False,
            "face_detected": True,
            "quality_rejected": False,
            "similarity": round(similarity, 4) if best_code else 0.0,
            "student_code": None,
            "name": "Desconocido",
            "already_marked": False,
        }

        if best_code is not None:
            # Obtener nombre desde el índice en memoria (evita I/O extra)
            student_name = (
                embeddings_index.get(best_code, {}).get("name")
                or emb_manager.get_nombre(best_code)
                or best_code
            )

            # ── Anti-duplicados: cache rápido en memoria ──────────────
            now_time = time.time()
            already_marked = (
                best_code in _recent_attendances
                and (now_time - _recent_attendances[best_code]) < _CACHE_TTL_SECONDS
            )

            # ── Anti-duplicados: verificación canónica en Supabase ────
            if not already_marked:
                already_marked = await _check_marked_today_supabase(best_code)

            face_result.update({
                "match": True,
                "student_code": best_code,
                "name": student_name,
                "already_marked": already_marked,
            })

            if not already_marked:
                # Dibujar bbox en la foto que se enviará a Supabase
                frame_bytes_to_send: bytes | None = None
                try:
                    draw_img = bgr.copy()
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        draw_img, student_name,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                    )
                    # Reducir tamaño para no superar 1 MB en Next.js
                    h, w = draw_img.shape[:2]
                    if w > 800:
                        draw_img = cv2.resize(draw_img, (800, int(h * (800 / w))))
                    _, buf = cv2.imencode(
                        ".jpg", draw_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    frame_bytes_to_send = buf.tobytes()
                except Exception as e:
                    print(f"[identify] Error dibujando bbox: {e}")
                    frame_bytes_to_send = raw  # fallback

                background_tasks.add_task(
                    _bg_notify_nextjs,
                    best_code,
                    student_name,
                    round(similarity, 4),
                    frame_bytes_to_send,
                )

                # Actualizar cache anti-duplicados
                _recent_attendances[best_code] = time.time()

        results.append(face_result)

    return JSONResponse(content={
        "faces": results,
        "frame_width": width,
        "frame_height": height,
    })
