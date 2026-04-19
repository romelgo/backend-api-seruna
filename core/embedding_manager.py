"""
Gestor de embeddings faciales en disco.

Sigue el mismo lineamiento que dataset_manager.py:
  - Un archivo .npy por estudiante: dataset/{codigo}/embedding.npy
  - Promedio de todos los embeddings capturados durante el enroll
  - Identidad como metadata en dataset/{codigo}/metadata.json

Implementa cosine similarity para búsqueda O(n).
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EmbeddingManager:
    """
    Gestiona los embeddings ArcFace persistidos en disco.

    Estructura (dentro de dataset_dir):
        dataset/
        └── {codigo}/
            ├── metadata.json       # Gestionado por DatasetManager
            └── embedding.npy      # Vector (512,) promedio — gestionado aquí
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    # ----------------------------------------------------------------
    # Escritura
    # ----------------------------------------------------------------

    def save_embedding(self, codigo: str, embedding: np.ndarray) -> Path:
        """
        Guarda el embedding promedio de un estudiante en disco.

        Args:
            codigo: Código del estudiante.
            embedding: Vector (512,) L2-normalizado.

        Returns:
            Path al archivo guardado.
        """
        student_dir = self.dataset_dir / codigo
        student_dir.mkdir(parents=True, exist_ok=True)

        emb_path = student_dir / "embedding.npy"
        np.save(str(emb_path), embedding.astype(np.float32))
        print(f"[EmbeddingManager] Embedding guardado: {emb_path}")
        return emb_path

    def update_embedding(self, codigo: str, new_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Promedia una lista de embeddings y guarda el resultado.

        Carga el embedding existente (si lo hay) y promedia con los nuevos.

        Args:
            codigo: Código del estudiante.
            new_embeddings: Lista de vectores (512,) nuevos.

        Returns:
            Embedding promedio guardado.
        """
        all_embeddings = new_embeddings.copy()

        # Cargar embedding existente si hay uno
        existing = self.load_embedding(codigo)
        if existing is not None:
            all_embeddings.append(existing)

        # Promediar y re-normalizar
        averaged = np.mean(np.stack(all_embeddings), axis=0)
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        return self.save_embedding(codigo, averaged), averaged

    # ----------------------------------------------------------------
    # Lectura
    # ----------------------------------------------------------------

    def load_embedding(self, codigo: str) -> Optional[np.ndarray]:
        """
        Carga el embedding de un estudiante desde disco.

        Returns:
            np.ndarray (512,) o None si no existe.
        """
        emb_path = self.dataset_dir / codigo / "embedding.npy"
        if emb_path.exists():
            return np.load(str(emb_path)).astype(np.float32)
        return None

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Carga todos los embeddings disponibles en memoria.

        Returns:
            Diccionario {codigo: embedding_vector}.
        """
        embeddings = {}
        for student_dir in sorted(self.dataset_dir.iterdir()):
            if not student_dir.is_dir():
                continue
            emb_path = student_dir / "embedding.npy"
            if emb_path.exists():
                codigo = student_dir.name
                embeddings[codigo] = np.load(str(emb_path)).astype(np.float32)
        print(f"[EmbeddingManager] {len(embeddings)} embeddings cargados.")
        return embeddings

    # ----------------------------------------------------------------
    # Búsqueda
    # ----------------------------------------------------------------

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.5,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Busca el estudiante más similar al embedding dado.

        Usa cosine similarity (dot product de vectores L2-normalizados).

        Args:
            query_embedding: Vector (512,) de consulta, L2-normalizado.
            threshold: Similitud mínima para considerar un match.
            embeddings: Índice pre-cargado (evita lecturas de disco repetidas).
                        Si None, carga desde disco en cada llamada.

        Returns:
            Tupla (codigo, similarity). codigo=None si no hay match sobre threshold.
        """
        if embeddings is None:
            embeddings = self.load_all_embeddings()

        if not embeddings:
            return None, 0.0

        best_code = None
        best_sim = -1.0

        for codigo, stored_emb in embeddings.items():
            sim = float(np.dot(query_embedding, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_code = codigo

        if best_sim >= threshold:
            return best_code, best_sim
        return None, best_sim

    # ----------------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------------

    def has_embedding(self, codigo: str) -> bool:
        """Verifica si un estudiante tiene embedding guardado."""
        return (self.dataset_dir / codigo / "embedding.npy").exists()

    def delete_embedding(self, codigo: str) -> bool:
        """Elimina el embedding de un estudiante."""
        emb_path = self.dataset_dir / codigo / "embedding.npy"
        if emb_path.exists():
            emb_path.unlink()
            print(f"[EmbeddingManager] Embedding eliminado: {codigo}")
            return True
        return False

    def get_nombre(self, codigo: str) -> Optional[str]:
        """Lee el nombre del estudiante desde su metadata.json."""
        meta_path = self.dataset_dir / codigo / "metadata.json"
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            return data.get("nombre")
        return None
