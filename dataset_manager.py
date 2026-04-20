"""
Gestor del dataset facial en disco.
Organiza imágenes por estudiante con metadata JSON.
"""
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config


class DatasetManager:
    """
    Gestiona la estructura del dataset de rostros en disco.
    
    Estructura:
        dataset/
        └── {codigo_alumno}/
            ├── metadata.json      # Info del estudiante
            ├── raw/               # Imágenes originales capturadas
            ├── faces/             # Rostros recortados, alineados, normalizados
            └── augmented/         # Versiones augmentadas
    """

    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir = dataset_dir or config.DATASET_DIR
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Gestión de estudiantes
    # ----------------------------------------------------------------

    def create_student(self, codigo: str, nombre: str) -> Path:
        """
        Crea la estructura de directorios para un nuevo estudiante.
        
        Args:
            codigo: Código único del estudiante (ej: EST-2024001).
            nombre: Nombre completo del estudiante.
        
        Returns:
            Path al directorio del estudiante.
        """
        student_dir = self.dataset_dir / codigo

        # Crear subdirectorios
        (student_dir / "raw").mkdir(parents=True, exist_ok=True)
        (student_dir / "faces").mkdir(parents=True, exist_ok=True)
        (student_dir / "augmented").mkdir(parents=True, exist_ok=True)

        # Crear o actualizar metadata
        metadata_path = student_dir / "metadata.json"
        if metadata_path.exists():
            metadata = self._load_metadata(codigo)
            metadata["updated_at"] = datetime.now().isoformat()
        else:
            metadata = {
                "codigo": codigo,
                "nombre": nombre,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "total_raw": 0,
                "total_faces": 0,
                "total_augmented": 0,
                "enrollment_complete": False,
                "quality_scores": [],
            }

        self._save_metadata(codigo, metadata)
        print(f"[Dataset] Estudiante creado: {codigo} - {nombre}")
        return student_dir

    def student_exists(self, codigo: str) -> bool:
        """Verifica si un estudiante ya está registrado."""
        return (self.dataset_dir / codigo / "metadata.json").exists()

    def get_student_dir(self, codigo: str) -> Path:
        """Retorna el directorio de un estudiante."""
        return self.dataset_dir / codigo

    def list_students(self) -> List[Dict]:
        """Lista todos los estudiantes registrados con su metadata."""
        students = []
        for student_dir in sorted(self.dataset_dir.iterdir()):
            if student_dir.is_dir():
                metadata_path = student_dir / "metadata.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    students.append(metadata)
        return students

    def delete_student(self, codigo: str) -> bool:
        """Elimina un estudiante y todos sus datos."""
        student_dir = self.dataset_dir / codigo
        if student_dir.exists():
            shutil.rmtree(student_dir)
            print(f"[Dataset] Estudiante eliminado: {codigo}")
            return True
        return False

    # ----------------------------------------------------------------
    # Guardado de imágenes
    # ----------------------------------------------------------------

    def save_raw_image(self, codigo: str, image: np.ndarray) -> Path:
        """
        Guarda una imagen original capturada.
        
        Args:
            codigo: Código del estudiante.
            image: Imagen BGR.
        
        Returns:
            Path al archivo guardado.
        """
        raw_dir = self.dataset_dir / codigo / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Generar nombre con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"raw_{timestamp}.jpg"
        filepath = raw_dir / filename

        cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Actualizar conteo
        metadata = self._load_metadata(codigo)
        metadata["total_raw"] = len(list(raw_dir.glob("*.jpg")))
        metadata["updated_at"] = datetime.now().isoformat()
        self._save_metadata(codigo, metadata)

        return filepath

    def save_face_image(
        self,
        codigo: str,
        face_image: np.ndarray,
        quality_score: float,
    ) -> Path:
        """
        Guarda un rostro procesado (recortado, alineado, normalizado).
        
        Args:
            codigo: Código del estudiante.
            face_image: Imagen BGR del rostro procesado.
            quality_score: Score de calidad (blur variance).
        
        Returns:
            Path al archivo guardado.
        """
        faces_dir = self.dataset_dir / codigo / "faces"
        faces_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"face_{timestamp}.jpg"
        filepath = faces_dir / filename

        cv2.imwrite(str(filepath), face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Actualizar metadata
        metadata = self._load_metadata(codigo)
        metadata["total_faces"] = len(list(faces_dir.glob("*.jpg")))
        metadata["quality_scores"].append(round(quality_score, 2))
        metadata["updated_at"] = datetime.now().isoformat()

        # Verificar si enrollment está completo
        if metadata["total_faces"] >= config.MIN_SAMPLES_PER_STUDENT:
            metadata["enrollment_complete"] = True

        self._save_metadata(codigo, metadata)

        return filepath

    def save_augmented_image(self, codigo: str, image: np.ndarray, index: int) -> Path:
        """Guarda una imagen augmentada."""
        aug_dir = self.dataset_dir / codigo / "augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"aug_{timestamp}_{index:02d}.jpg"
        filepath = aug_dir / filename

        cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Actualizar conteo
        metadata = self._load_metadata(codigo)
        metadata["total_augmented"] = len(list(aug_dir.glob("*.jpg")))
        metadata["updated_at"] = datetime.now().isoformat()
        self._save_metadata(codigo, metadata)

        return filepath

    # ----------------------------------------------------------------
    # Lectura de imágenes
    # ----------------------------------------------------------------

    def get_face_images(self, codigo: str) -> List[np.ndarray]:
        """Carga todas las imágenes de rostros procesados de un estudiante."""
        faces_dir = self.dataset_dir / codigo / "faces"
        images = []
        for filepath in sorted(faces_dir.glob("*.jpg")):
            img = cv2.imread(str(filepath))
            if img is not None:
                images.append(img)
        return images

    def get_all_face_images(self, codigo: str) -> List[np.ndarray]:
        """Carga rostros procesados + augmentados."""
        images = self.get_face_images(codigo)

        aug_dir = self.dataset_dir / codigo / "augmented"
        if aug_dir.exists():
            for filepath in sorted(aug_dir.glob("*.jpg")):
                img = cv2.imread(str(filepath))
                if img is not None:
                    images.append(img)

        return images

    # ----------------------------------------------------------------
    # Verificación
    # ----------------------------------------------------------------

    def verify_student(self, codigo: str) -> Dict:
        """
        Verifica el estado del enrollment de un estudiante.
        
        Returns:
            Diccionario con diagnóstico completo.
        """
        if not self.student_exists(codigo):
            return {"status": "NOT_FOUND", "message": f"Estudiante {codigo} no encontrado"}

        metadata = self._load_metadata(codigo)
        faces_dir = self.dataset_dir / codigo / "faces"
        face_count = len(list(faces_dir.glob("*.jpg")))

        avg_quality = 0.0
        if metadata["quality_scores"]:
            avg_quality = sum(metadata["quality_scores"]) / len(metadata["quality_scores"])

        return {
            "status": "COMPLETE" if face_count >= config.MIN_SAMPLES_PER_STUDENT else "INCOMPLETE",
            "codigo": codigo,
            "nombre": metadata.get("nombre", ""),
            "total_raw": metadata.get("total_raw", 0),
            "total_faces": face_count,
            "total_augmented": metadata.get("total_augmented", 0),
            "min_required": config.MIN_SAMPLES_PER_STUDENT,
            "avg_quality": round(avg_quality, 2),
            "message": (
                f"Enrollment completo ({face_count}/{config.MIN_SAMPLES_PER_STUDENT})"
                if face_count >= config.MIN_SAMPLES_PER_STUDENT
                else f"Faltan {config.MIN_SAMPLES_PER_STUDENT - face_count} muestras más"
            ),
        }

    def get_dataset_summary(self) -> Dict:
        """Resumen general del dataset completo."""
        students = self.list_students()
        total_faces = sum(s.get("total_faces", 0) for s in students)
        total_augmented = sum(s.get("total_augmented", 0) for s in students)
        complete = sum(1 for s in students if s.get("enrollment_complete", False))

        return {
            "total_students": len(students),
            "enrollment_complete": complete,
            "enrollment_pending": len(students) - complete,
            "total_face_images": total_faces,
            "total_augmented_images": total_augmented,
            "dataset_path": str(self.dataset_dir),
        }

    # ----------------------------------------------------------------
    # Metadata helpers
    # ----------------------------------------------------------------

    def _load_metadata(self, codigo: str) -> Dict:
        """Carga metadata JSON de un estudiante."""
        path = self.dataset_dir / codigo / "metadata.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save_metadata(self, codigo: str, metadata: Dict):
        """Guarda metadata JSON de un estudiante."""
        path = self.dataset_dir / codigo / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

