"""
Gestor de embeddings faciales en disco — v2.0.

Mejoras vs v1.0:
  - Galería multi-frame: guarda embedding_mean.npy (centroide) + gallery.npy (N, 512)
  - Filtro de outliers en enrollment (similitud < ENROLLMENT_MIN_SIM descartados)
  - Comparación híbrida: centroide rápido → galería en zona gris
  - Batch cosine similarity vectorizado con NumPy (O(N·512) one-shot)
  - Índice en memoria con estructura enriquecida (mean + gallery + threshold + name)

Estructura en disco:
    dataset/
    └── {codigo}/
        ├── metadata.json         # Gestionado por DatasetManager
        ├── embedding_mean.npy    # Centroide L2-normalizado (512,)
        └── gallery.npy           # Galería individual (N, 512) — L2-normalizado
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config


class EmbeddingManager:
    """
    Gestiona los embeddings ArcFace persistidos en disco.

    Estructura (dentro de dataset_dir):
        dataset/
        └── {codigo}/
            ├── metadata.json          # Gestionado por DatasetManager
            ├── embedding_mean.npy     # Centroide (512,) — para comparación rápida
            └── gallery.npy            # Galería (N, 512) — para validación zona gris
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    # ----------------------------------------------------------------
    # Escritura
    # ----------------------------------------------------------------

    def save_embeddings(
        self,
        codigo: str,
        embeddings_list: List[np.ndarray],
        filter_outliers: bool = True,
    ) -> Tuple[Path, Path, np.ndarray]:
        """
        Procesa una lista de embeddings crudos y persiste la galería + centroide.

        Pipeline:
          1. Calcular centroide provisional de todos los embeddings.
          2. Filtrar outliers: descartar embeddings con similitud al centroide < ENROLLMENT_MIN_SIM.
          3. Re-calcular centroide con los embeddings limpios y normalizar L2.
          4. Guardar gallery.npy  (N_clean, 512)
          5. Guardar embedding_mean.npy (512,)

        Args:
            codigo: Código del estudiante.
            embeddings_list: Lista de vectores (512,) L2-normalizados capturados.
            filter_outliers: Si True, descarta embeddings alejados del centroide.

        Returns:
            (path_mean, path_gallery, centroide_final)
        """
        if not embeddings_list:
            raise ValueError(f"[EmbeddingManager] Lista de embeddings vacía para {codigo}")

        stack = np.stack([e.astype(np.float32) for e in embeddings_list])  # (N, 512)

        if filter_outliers and len(stack) > 2:
            # Centroide provisional (sin normalizar) solo para medir distancias
            provisional_mean = stack.mean(axis=0)
            norm = np.linalg.norm(provisional_mean)
            if norm > 0:
                provisional_mean = provisional_mean / norm

            # Similitudes de cada embedding respecto al centroide provisional
            sims = stack @ provisional_mean  # (N,)
            mask = sims >= config.ENROLLMENT_MIN_SIM
            kept = stack[mask]

            n_discarded = len(stack) - mask.sum()
            if n_discarded > 0:
                print(f"[EmbeddingManager] {codigo}: {n_discarded} outlier(s) descartado(s) "
                      f"(sim < {config.ENROLLMENT_MIN_SIM})")

            # Si el filtro elimina demasiado, conservar al menos 3 o el total
            if len(kept) < min(3, len(stack)):
                print(f"[EmbeddingManager] {codigo}: demasiados outliers; se conservan todos.")
                kept = stack
        else:
            kept = stack

        # Centroide final normalizado
        centroid = kept.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        student_dir = self.dataset_dir / codigo
        student_dir.mkdir(parents=True, exist_ok=True)

        mean_path    = student_dir / "embedding_mean.npy"
        gallery_path = student_dir / "gallery.npy"

        np.save(str(mean_path),    centroid.astype(np.float32))
        np.save(str(gallery_path), kept.astype(np.float32))

        print(f"[EmbeddingManager] {codigo}: galería guardada "
              f"({len(kept)} embeddings, centroide normalizado)")
        return mean_path, gallery_path, centroid

    # Alias de compatibilidad con código anterior
    def save_embedding(self, codigo: str, embedding: np.ndarray) -> Path:
        """Guarda un único embedding como centroide (sin galería). Compatibilidad v1.0."""
        student_dir = self.dataset_dir / codigo
        student_dir.mkdir(parents=True, exist_ok=True)
        mean_path = student_dir / "embedding_mean.npy"
        np.save(str(mean_path), embedding.astype(np.float32))
        print(f"[EmbeddingManager] Embedding (centroide) guardado: {mean_path}")
        return mean_path

    def update_embeddings(self, codigo: str, new_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Agrega nuevos embeddings a la galería existente y recalcula el centroide.

        Carga la galería existente (si la hay) y la combina con los nuevos.

        Args:
            codigo: Código del estudiante.
            new_embeddings: Lista de vectores (512,) nuevos.

        Returns:
            Centroide actualizado (512,).
        """
        all_embeddings = list(new_embeddings)

        existing_gallery = self.load_gallery(codigo)
        if existing_gallery is not None:
            all_embeddings.extend(existing_gallery.tolist())

        _, _, centroid = self.save_embeddings(codigo, all_embeddings)
        return centroid

    # ----------------------------------------------------------------
    # Lectura
    # ----------------------------------------------------------------

    def load_mean(self, codigo: str) -> Optional[np.ndarray]:
        """Carga el centroide de un estudiante. Soporta ambos nombres de archivo."""
        for filename in ("embedding_mean.npy", "embedding.npy"):
            path = self.dataset_dir / codigo / filename
            if path.exists():
                return np.load(str(path)).astype(np.float32)
        return None

    # Alias de compatibilidad
    def load_embedding(self, codigo: str) -> Optional[np.ndarray]:
        return self.load_mean(codigo)

    def load_gallery(self, codigo: str) -> Optional[np.ndarray]:
        """Carga la galería (N, 512) de un estudiante. Retorna None si no existe."""
        path = self.dataset_dir / codigo / "gallery.npy"
        if path.exists():
            return np.load(str(path)).astype(np.float32)
        return None

    def load_all_embeddings(self) -> Dict[str, dict]:
        """
        Carga todos los embeddings en memoria con estructura enriquecida.

        Retorna:
            {
                "codigo": {
                    "mean":    np.ndarray (512,),
                    "gallery": np.ndarray (N, 512) | None,
                    "name":    str,
                    "threshold": float,  # umbral personal o default del config
                }
            }
        """
        db: Dict[str, dict] = {}

        for student_dir in sorted(self.dataset_dir.iterdir()):
            if not student_dir.is_dir():
                continue

            codigo = student_dir.name
            mean_emb = self.load_mean(codigo)
            if mean_emb is None:
                continue

            gallery = self.load_gallery(codigo)

            # Leer nombre y umbral personal desde metadata.json
            name = codigo
            personal_threshold = config.THRESHOLD_SECURE
            meta_path = student_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    name = meta.get("nombre", codigo)
                    personal_threshold = meta.get(
                        "personal_threshold", config.THRESHOLD_SECURE
                    )
                except Exception:
                    pass

            db[codigo] = {
                "mean":      mean_emb,
                "gallery":   gallery,
                "name":      name,
                "threshold": personal_threshold,
            }

        print(f"[EmbeddingManager] {len(db)} estudiantes cargados en índice.")
        return db

    # ----------------------------------------------------------------
    # Búsqueda — comparación híbrida vectorizada
    # ----------------------------------------------------------------

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = None,
        embeddings: Optional[Dict[str, dict]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Busca el estudiante más similar usando comparación híbrida:
          1. Batch cosine similarity contra todos los centroides (vectorizado).
          2. Si el mejor supera el umbral seguro  → coincidencia inmediata.
          3. Si cae en zona gris [THRESHOLD_GREY_LOW, THRESHOLD_SECURE) → validar con galería.

        Args:
            query_embedding: Vector (512,) L2-normalizado.
            threshold: Umbral seguro (default: config.THRESHOLD_SECURE).
            embeddings: Índice pre-cargado por load_all_embeddings(). Si None, carga del disco.

        Returns:
            (codigo, similarity) — codigo=None si no supera ningún umbral.
        """
        if threshold is None:
            threshold = config.THRESHOLD_SECURE

        if embeddings is None:
            embeddings = self.load_all_embeddings()

        if not embeddings:
            return None, 0.0

        codigos = list(embeddings.keys())
        means = np.stack([embeddings[c]["mean"] for c in codigos])  # (N, 512)

        # ── Paso 1: batch cosine similarity contra centroides ──────────
        sims = means @ query_embedding  # (N,) — vectorizado O(N·512)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_code = codigos[best_idx]

        # Umbral personal del alumno con mejor similitud
        student_threshold = embeddings[best_code].get("threshold", threshold)

        # ── Paso 2: coincidencia segura ────────────────────────────────
        if best_sim >= student_threshold:
            return best_code, best_sim

        # ── Paso 3: zona gris → validar con galería ────────────────────
        if best_sim >= config.THRESHOLD_GREY_LOW:
            gallery = embeddings[best_code].get("gallery")
            if gallery is not None and len(gallery) > 0:
                gallery_sims = gallery @ query_embedding        # (M,)
                votes = int(np.sum(gallery_sims >= config.THRESHOLD_GREY_LOW))
                vote_ratio = votes / len(gallery_sims)

                if vote_ratio >= config.GALLERY_VOTE_RATIO:
                    print(f"[EmbeddingManager] Zona gris → {best_code} "
                          f"(votos={votes}/{len(gallery_sims)}, sim={best_sim:.3f})")
                    return best_code, float(np.max(gallery_sims))

        return None, best_sim

    # ----------------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------------

    def has_embedding(self, codigo: str) -> bool:
        """Verifica si un estudiante tiene embedding guardado."""
        for filename in ("embedding_mean.npy", "embedding.npy"):
            if (self.dataset_dir / codigo / filename).exists():
                return True
        return False

    def delete_embedding(self, codigo: str) -> bool:
        """Elimina todos los archivos de embedding de un estudiante."""
        deleted = False
        for filename in ("embedding_mean.npy", "embedding.npy", "gallery.npy"):
            path = self.dataset_dir / codigo / filename
            if path.exists():
                path.unlink()
                deleted = True
        if deleted:
            print(f"[EmbeddingManager] Embeddings eliminados: {codigo}")
        return deleted

    def get_nombre(self, codigo: str) -> Optional[str]:
        """Lee el nombre del estudiante desde su metadata.json."""
        meta_path = self.dataset_dir / codigo / "metadata.json"
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            return data.get("nombre")
        return None
