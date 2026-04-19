"""
GET  /students           — Lista todos los estudiantes registrados.
GET  /students/{code}    — Detalle de un estudiante.
DELETE /students/{code}  — Elimina un estudiante y sus datos.
GET  /attendance/today   — Asistencia del día actual.
"""
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager import DatasetManager

router = APIRouter()


@router.get("/students")
async def list_students(request: Request):
    """
    Retorna todos los estudiantes con su info de registro.
    """
    db = request.app.state.db
    students = await db.get_all_students()
    return JSONResponse(content=students)


@router.get("/students/{code}")
async def get_student(code: str, request: Request):
    """
    Retorna el detalle de un estudiante específico,
    incluyendo el estado de enrollment del dataset.
    """
    db = request.app.state.db
    student = await db.get_student(code)

    if not student:
        raise HTTPException(status_code=404, detail=f"Estudiante '{code}' no encontrado.")

    # Enriquecer con info del dataset (faces, etc.)
    dataset_dir = PROJECT_ROOT / "dataset"
    dm = DatasetManager(dataset_dir)
    if dm.student_exists(code):
        verification = dm.verify_student(code)
        student["dataset_status"] = verification["status"]
        student["total_faces"] = verification["total_faces"]
        student["avg_quality"] = verification["avg_quality"]
    else:
        student["dataset_status"] = "NOT_IN_DATASET"

    return JSONResponse(content=student)


@router.delete("/students/{code}")
async def delete_student(code: str, request: Request):
    """
    Elimina un estudiante: borra de SQLite, dataset en disco y embedding.
    """
    db = request.app.state.db
    emb_manager = request.app.state.embedding_manager

    # Eliminar de SQLite
    deleted_db = await db.delete_student(code)

    # Eliminar embedding
    emb_manager.delete_embedding(code)

    # Eliminar dataset del disco
    dataset_dir = PROJECT_ROOT / "dataset"
    dm = DatasetManager(dataset_dir)
    deleted_dataset = dm.delete_student(code)

    if not deleted_db and not deleted_dataset:
        raise HTTPException(status_code=404, detail=f"Estudiante '{code}' no encontrado.")

    # Recargar índice en memoria
    request.app.state.embeddings_index = emb_manager.load_all_embeddings()

    return JSONResponse(
        content={"status": "deleted", "student_code": code}
    )


@router.get("/attendance/today")
async def today_attendance(
    request: Request,
    course_id: str = Query(default="default", description="ID del curso/sesión"),
):
    """
    Retorna la lista de asistencia del día actual para un curso.
    """
    db = request.app.state.db
    records = await db.get_today_attendance(course_id)
    return JSONResponse(content=records)
