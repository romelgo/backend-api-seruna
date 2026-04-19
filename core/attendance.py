"""
Gestión de asistencia con SQLite (async via aiosqlite).

Schema:
    students   — registro de estudiantes enrolados
    attendance — registro de asistencia por sesión/fecha
"""
import aiosqlite
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict


# ----------------------------------------------------------------
# Schema SQL
# ----------------------------------------------------------------

_CREATE_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    code        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_count  INTEGER DEFAULT 0
);
"""

_CREATE_ATTENDANCE = """
CREATE TABLE IF NOT EXISTS attendance (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_code TEXT REFERENCES students(code),
    course_id    TEXT NOT NULL DEFAULT 'default',
    marked_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class AttendanceDB:
    """
    Acceso async a la base de datos de asistencia SQLite.

    Uso:
        db = AttendanceDB("./attendance.db")
        await db.init()
        await db.upsert_student("EST-001", "Juan Pérez", face_count=8)
        already = await db.is_already_marked("EST-001", "COMP101", date.today())
    """

    def __init__(self, db_path: str = "./attendance.db"):
        self.db_path = db_path

    async def init(self):
        """Crea las tablas si no existen."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(_CREATE_STUDENTS)
            await db.execute(_CREATE_ATTENDANCE)
            await db.commit()
        print(f"[AttendanceDB] Inicializado en {self.db_path}")

    # ----------------------------------------------------------------
    # Estudiantes
    # ----------------------------------------------------------------

    async def upsert_student(
        self, code: str, name: str, face_count: int = 0
    ) -> None:
        """
        Inserta o actualiza un estudiante en la tabla students.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO students (code, name, face_count)
                VALUES (?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    name       = excluded.name,
                    face_count = excluded.face_count
                """,
                (code, name, face_count),
            )
            await db.commit()

    async def get_student(self, code: str) -> Optional[Dict]:
        """Retorna datos del estudiante o None si no existe."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT code, name, enrolled_at, face_count FROM students WHERE code = ?",
                (code,),
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_all_students(self) -> List[Dict]:
        """Retorna todos los estudiantes registrados."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT code, name, enrolled_at, face_count FROM students ORDER BY name"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def delete_student(self, code: str) -> bool:
        """Elimina un estudiante y su historial de asistencia."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM attendance WHERE student_code = ?", (code,))
            cursor = await db.execute("DELETE FROM students WHERE code = ?", (code,))
            await db.commit()
            return cursor.rowcount > 0

    # ----------------------------------------------------------------
    # Asistencia
    # ----------------------------------------------------------------

    async def mark_attendance(
        self, student_code: str, course_id: str = "default"
    ) -> int:
        """
        Marca asistencia de un estudiante.

        Returns:
            ID del registro creado.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO attendance (student_code, course_id) VALUES (?, ?)",
                (student_code, course_id),
            )
            await db.commit()
            return cursor.lastrowid

    async def is_already_marked(
        self,
        student_code: str,
        course_id: str = "default",
        check_date: Optional[date] = None,
    ) -> bool:
        """
        Verifica si el estudiante ya fue marcado en la sesión actual.

        Una 'sesión' = mismo course_id + mismo día calendario.
        """
        if check_date is None:
            check_date = date.today()

        date_str = check_date.strftime("%Y-%m-%d")

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT COUNT(*) FROM attendance
                WHERE student_code = ?
                  AND course_id    = ?
                  AND DATE(marked_at) = ?
                """,
                (student_code, course_id, date_str),
            ) as cursor:
                row = await cursor.fetchone()
                return (row[0] > 0) if row else False

    async def get_today_attendance(self, course_id: str = "default") -> List[Dict]:
        """Retorna la asistencia del día actual para un curso."""
        today = date.today().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT a.id, a.student_code, s.name, a.course_id, a.marked_at
                FROM attendance a
                JOIN students s ON a.student_code = s.code
                WHERE a.course_id = ? AND DATE(a.marked_at) = ?
                ORDER BY a.marked_at DESC
                """,
                (course_id, today),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
