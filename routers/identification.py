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

async def _bg_mark_attendance(db, student_code: str, course_id: str):
    try:
        await db.mark_attendance(student_code, course_id)
    except Exception as e:
        print(f"[identify] Error marcando asistencia SQLite: {e}")

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

async def _bg_send_whatsapp_fastapi(student_code: str, name: str, frame_bytes: bytes):
    try:
        from supabase import create_client, Client as SC
        import httpx
        
        supa_url = os.getenv("SUPABASE_URL")
        supa_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supa_url or not supa_key: return
        
        supa_client: SC = create_client(supa_url, supa_key)
        
        # 1. Obtener student ID
        stu_res = supa_client.table("students").select("id").eq("student_code", student_code).execute()
        if not stu_res.data: 
            print(f"[ident_bg] Aborting: No student profile found for code {student_code}")
            return
        student_id = stu_res.data[0]["id"]
        
        # 2. Subir imagen a storage
        filename = f"attendance/{student_id}/{int(time.time()*1000)}_py.jpg"
        print(f"[ident_bg] Subiendo foto a supabase storage {filename}...")
        supa_client.storage.from_("attendance-photos").upload(
            file=frame_bytes,
            path=filename,
            file_options={"content-type": "image/jpeg", "upsert": "false"}
        )
        photo_url = supa_client.storage.from_("attendance-photos").get_public_url(filename)
        print(f"[ident_bg] Photo uploaded: {photo_url}")
        
        # 3. Buscar tutores
        parent_res = supa_client.table("parent_student").select("parent_id").eq("student_id", student_id).execute()
        if not parent_res.data: 
            print("[ident_bg] No parents linked. Aborting WhatsApp.")
            return
        parent_ids = [p["parent_id"] for p in parent_res.data]
        print(f"[ident_bg] Parent IDs for student: {parent_ids}")
        
        parents_res = supa_client.table("parents").select("whatsapp_number, first_name").in_("id", parent_ids).execute()
        if not parents_res.data: 
            print("[ident_bg] No whatsapp numbers found for parents. Aborting.")
            return
        
        time_str = datetime.datetime.now().strftime("%I:%M %p")
        body_msg = f"Hola 👋\n\n*{name}* acaba de registrar su asistencia al instante.\n🕐 Hora: *{time_str}*\n\n_Validación vía IA completada exitosamente desde el Backend._"
        
        for p in parents_res.data:
            if not p.get("whatsapp_number"): continue
            to_num = p["whatsapp_number"]
            # Limpiar número si traía formato de twilio u otro
            if to_num.startswith("whatsapp:"):
                to_num = to_num.replace("whatsapp:", "")
            if to_num.startswith("+"):
                to_num = to_num.replace("+", "")
                
            try:
                # Llamada al servicio local de Baileys
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        "http://localhost:3001/send",
                        json={
                            "number": to_num,
                            "message": body_msg,
                            "media_url": photo_url
                        }
                    )
                    print(f"[identify] Baileys service response para {to_num}: {resp.status_code}")
            except Exception as e:
                print(f"[identify] Error enviando a servicio Baileys para {to_num}: {e}")
                
    except Exception as e:
         print(f"[identify] Error general en WhatsApp FASTAPI loop: {e}")

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
    db = request.app.state.db
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
            already_marked = await db.is_already_marked(best_code, course_id, date.today())
            student = await db.get_student(best_code)
            student_name = student["name"] if student else emb_manager.get_nombre(best_code) or best_code
            
            face_result.update({
                "match": True,
                "student_code": best_code,
                "name": student_name,
                "already_marked": already_marked,
            })

            if not already_marked:
                # Marcamos para que no se vuelva a marcar en la misma petición si hay un bug o doble cara
                # aunque esto requeriría actualizar el DB, que es lo que hacemos en bg
                background_tasks.add_task(_bg_mark_attendance, db, best_code, course_id)

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
                
                # Enviar WhatsApp directamente desde FastAPI si hay imagen
                if frame_bytes_to_send:
                    background_tasks.add_task(
                        _bg_send_whatsapp_fastapi,
                        best_code,
                        student_name,
                        frame_bytes_to_send
                    )
                
        results.append(face_result)

    # Retornamos el frame size para simplificar el frontend positioning
    return JSONResponse(content={
        "faces": results,
        "frame_width": width,
        "frame_height": height
    })
