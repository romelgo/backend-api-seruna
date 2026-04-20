import os
import sys
import base64
from pathlib import Path
from datetime import date

import cv2
import numpy as np
import httpx
from fastapi import APIRouter, BackgroundTasks, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).parent.parent.parent

NEXT_APP_URL = os.getenv("NEXT_APP_URL", "http://localhost:3000")
AI_SERVER_SECRET = os.getenv("AI_SERVER_SECRET", "")

router = APIRouter()



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

import time
import uuid
import datetime

# Cache to prevent rapid bursts of attendance marking for the same student
_recent_attendances = {}

@router.post("/identify")
async def identify_frame(
    request: Request,
    background_tasks: BackgroundTasks,
    frame: UploadFile = File(...),
    course_id: str = Query(default="default", description="ID del curso/sesión"),
):
    recognizer = request.app.state.recognizer
    emb_manager = request.app.state.embedding_manager
    embeddings_index = request.app.state.embeddings_index
    threshold = request.app.state.similarity_threshold

    raw = await frame.read()
    nparr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        return JSONResponse(status_code=422, content={"faces": [], "error": "No se pudo decodificar el frame"})

    height, width, _ = bgr.shape

    # Detectar todas las caras
    faces = recognizer.get_faces(bgr)
    
    results = []

    for face in faces:
        if not hasattr(face, "embedding") or face.embedding is None:
            continue
            
        bbox = face.bbox.tolist() # [x1, y1, x2, y2]
        
        query_emb = face.embedding.astype(np.float32)
        best_code, similarity = emb_manager.find_best_match(query_emb, threshold=threshold, embeddings=embeddings_index)

        face_result = {
            "bbox": bbox,
            "match": False,
            "face_detected": True,
            "similarity": round(similarity, 4) if best_code else 0.0,
            "student_code": None,
            "name": "Desconocido",
            "already_marked": False
        }

        if best_code is not None:
            student_name = emb_manager.get_nombre(best_code) or best_code
            already_marked = False
            
            # Consultar en cache local primero para evitar rafagas y envíos duplicados de WhatsApp
            now_time = time.time()
            if best_code in _recent_attendances and (now_time - _recent_attendances[best_code]) < 60:
                already_marked = True
            
            # Consultar con Supabase asincrónicamente
            supa_url = os.getenv("SUPABASE_URL")
            supa_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            if supa_url and supa_key:
                headers = {"apikey": supa_key, "Authorization": f"Bearer {supa_key}"}
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        # 1. Obtener datos del estudiante
                        res_stu = await client.get(f"{supa_url}/rest/v1/students?student_code=eq.{best_code}&select=id,first_name,last_name", headers=headers)
                        if res_stu.status_code == 200 and res_stu.json():
                            stu = res_stu.json()[0]
                            student_id = stu["id"]
                            student_name = f"{stu['first_name']} {stu['last_name']}"
                            
                            # 2. Verificar asistencia de hoy
                            today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                            res_att = await client.get(f"{supa_url}/rest/v1/attendance?student_id=eq.{student_id}&timestamp=gte.{today_start}&select=id", headers=headers)
                            if res_att.status_code == 200 and res_att.json():
                                already_marked = True
                except Exception as e:
                    print(f"[identify] Error Supabase check: {e}")
            
            face_result.update({
                "match": True,
                "student_code": best_code,
                "name": student_name,
                "already_marked": already_marked,
            })

            if not already_marked:

                frame_bytes_to_send: bytes | None = None
                try:
                    draw_img = bgr.copy()
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(draw_img, student_name, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Reducir tamaño para no superar el límite de payload de Next.js (1MB)
                    h, w = draw_img.shape[:2]
                    if w > 800:
                        new_h = int(h * (800 / w))
                        draw_img = cv2.resize(draw_img, (800, new_h))
                        
                    _, buf = cv2.imencode(".jpg", draw_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    frame_bytes_to_send = buf.tobytes()
                except Exception as e:
                    print(f"Error drawing bbox: {e}")
                    frame_bytes_to_send = raw  # fallback

                background_tasks.add_task(
                    _bg_notify_nextjs,
                    best_code,
                    student_name,
                    round(similarity, 4),
                    frame_bytes_to_send,
                )
                
                # Registrar en el cache local para bloquear futuros frames inmediatos
                _recent_attendances[best_code] = time.time()
                
        results.append(face_result)

    # Retornamos el frame size para simplificar el frontend positioning
    return JSONResponse(content={
        "faces": results,
        "frame_width": width,
        "frame_height": height
    })
