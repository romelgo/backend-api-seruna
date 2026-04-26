"""
Gestor de embeddings faciales en disco — v3.0.

Mejoras vs v2.0:
  - Multi-centroide por cluster de pose (frontal/right/left/up/down)
  - save_embeddings acepta poses_list opcional para etiquetar cada embedding
  - Centroides por pose guardados en centroids.npz
  - find_best_match usa centroide de pose específica si query_pose es proporcionado
  - Compatibilidad total hacia atrás: alumnos sin centroids.npz usan embedding_mean.npy

Estructura en disco:
    dataset/
    └── {codigo}/
        ├── metadata.json          # Gestionado por DatasetManager
        ├── embedding_mean.npy     # Centroide global L2-normalizado (512,)
        ├── centroids.npz          # Centroides por pose {frontal,right,left,up,down} (nuevo)
        └── gallery.npy            # Galería individual (N, 512) — L2-normalizado
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config


class EmbeddingManager:
    """
    Gestiona los embeddings ArcFace persistidos en disco.

    v3.0: Añade multi-centroide por cluster de pose para mejorar
    el reconocimiento en ángulos distintos al frontal.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    # ----------------------------------------------------------------
    # Clasificación de pose
    # ----------------------------------------------------------------

    @staticmethod
    def _classify_pose(pose: dict) -> str:
        """
        Clasifica un dict {yaw, pitch, roll} en un cluster de pose.

        Returns:
            "frontal" | "right" | "left" | "up" | "down"
        """
        yaw   = pose.get("yaw",   0.0)
        pitch = pose.get("pitch", 0.0)

        # Prioridad vertical: si pitch es significativo, clasificar en up/down
        if abs(pitch) >= config.POSE_VERTICAL_MIN_PITCH:
            return "up" if pitch > 0 else "down"

        # Lateral
        if abs(yaw) >= config.POSE_SIDE_MIN_YAW:
            return "right" if yaw > 0 else "left"

        return "frontal"

    # ----------------------------------------------------------------
    # Escritura
    # ----------------------------------------------------------------

    def save_embeddings(
        self,
        codigo: str,
        embeddings_list: List[np.ndarray],
        poses_list: Optional[List[dict]] = None,
        filter_outliers: bool = True,
    ) -> Tuple[Path, Path, np.ndarray]:
        """
        Procesa una lista de embeddings crudos y persiste la galería + centroide(s).

        Pipeline:
          1. Filtrar outliers (similitud < ENROLLMENT_MIN_SIM).
          2. Calcular centroide global normalizado.
          3. Si poses_list está disponible, agrupar por cluster y calcular
             un centroide por grupo → guardar en centroids.npz.
          4. Guardar gallery.npy  (N_clean, 512)
          5. Guardar embedding_mean.npy (512,)

        Args:
            codigo:          Código del estudiante.
            embeddings_list: Lista de vectores (512,) L2-normalizados capturados.
            poses_list:      Lista de dicts {yaw, pitch, roll} paralela a embeddings_list.
                             Si None, no se generan centroides por pose.
            filter_outliers: Si True, descarta embeddings alejados del centroide.

        Returns:
            (path_mean, path_gallery, centroide_final)
        """
        if not embeddings_list:
            raise ValueError(f"[EmbeddingManager] Lista de embeddings vacía para {codigo}")

        stack = np.stack([e.astype(np.float32) for e in embeddings_list])  # (N, 512)

        # ── Filtro de outliers ─────────────────────────────────────────
        if filter_outliers and len(stack) > 2:
            provisional_mean = stack.mean(axis=0)
            norm = np.linalg.norm(provisional_mean)
            if norm > 0:
                provisional_mean = provisional_mean / norm

            sims = stack @ provisional_mean  # (N,)
            mask = sims >= config.ENROLLMENT_MIN_SIM
            kept = stack[mask]

            n_discarded = int(len(stack) - mask.sum())
            if n_discarded > 0:
                print(f"[EmbeddingManager] {codigo}: {n_discarded} outlier(s) descartado(s) "
                      f"(sim < {config.ENROLLMENT_MIN_SIM})")

            if len(kept) < min(3, len(stack)):
                print(f"[EmbeddingManager] {codigo}: demasiados outliers; se conservan todos.")
                kept = stack
                mask = np.ones(len(stack), dtype=bool)
        else:
            kept = stack
            mask = np.ones(len(stack), dtype=bool)

        # ── Centroide global normalizado ────────────────────────────────
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

        # ── Multi-centroide por pose ────────────────────────────────────
        if poses_list is not None and len(poses_list) == len(embeddings_list):
            # Alinear poses con la máscara de outliers
            poses_arr = [poses_list[i] for i in range(len(poses_list)) if mask[i]]

            clusters: Dict[str, List[np.ndarray]] = {
                "frontal": [], "right": [], "left": [], "up": [], "down": []
            }
            for emb, pose in zip(kept, poses_arr):
                key = self._classify_pose(pose)
                clusters[key].append(emb)

            centroids_data = {}
            for cluster_key, cluster_embs in clusters.items():
                if cluster_embs:
                    c = np.stack(cluster_embs).mean(axis=0)
                    c_norm = np.linalg.norm(c)
                    centroids_data[cluster_key] = (
                        (c / c_norm) if c_norm > 0 else c
                    ).astype(np.float32)
                    print(f"[EmbeddingManager] {codigo}: cluster '{cluster_key}' "
                          f"→ {len(cluster_embs)} muestras")

            if centroids_data:
                centroids_path = student_dir / "centroids.npz"
                np.savez(str(centroids_path), **centroids_data)
                print(f"[EmbeddingManager] {codigo}: {len(centroids_data)} centroides "
                      f"por pose guardados en centroids.npz")

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

    def load_centroids(self, codigo: str) -> Dict[str, np.ndarray]:
        """Carga los centroides por pose desde centroids.npz. Retorna {} si no existe."""
        path = self.dataset_dir / codigo / "centroids.npz"
        if path.exists():
            data = np.load(str(path))
            return {k: data[k].astype(np.float32) for k in data.files}
        return {}

    def load_all_embeddings(self) -> Dict[str, dict]:
        """
        Carga todos los embeddings en memoria con estructura enriquecida.

        Retorna:
            {
                "codigo": {
                    "mean":      np.ndarray (512,),
                    "centroids": dict[str, np.ndarray],  # por pose, puede ser {}
                    "gallery":   np.ndarray (N, 512) | None,
                    "name":      str,
                    "threshold": float,
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

            gallery   = self.load_gallery(codigo)
            centroids = self.load_centroids(codigo)

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
                "centroids": centroids,
                "gallery":   gallery,
                "name":      name,
                "threshold": personal_threshold,
            }

        print(f"[EmbeddingManager] {len(db)} estudiantes cargados en índice.")
        return db

    # ----------------------------------------------------------------
    # Búsqueda — comparación híbrida vectorizada con multi-centroide
    # ----------------------------------------------------------------

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        query_pose: Optional[dict] = None,
        threshold: float = None,
        embeddings: Optional[Dict[str, dict]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Busca el estudiante más similar usando comparación híbrida multi-centroide:
          1. Batch cosine similarity contra todos los centroides globales.
          2. Si hay query_pose, también comparar contra el centroide de pose apropiado
             y tomar el mejor score entre global y pose-específico.
          3. Si el mejor supera el umbral seguro  → coincidencia inmediata.
          4. Si cae en zona gris [THRESHOLD_GREY_LOW, THRESHOLD_SECURE) → validar con galería.

        Args:
            query_embedding: Vector (512,) L2-normalizado.
            query_pose:      Dict {yaw, pitch, roll} de la pose actual (opcional).
            threshold:       Umbral seguro (default: config.THRESHOLD_SECURE).
            embeddings:      Índice pre-cargado por load_all_embeddings().

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
        means   = np.stack([embeddings[c]["mean"] for c in codigos])  # (N, 512)

        # ── Paso 1: batch cosine similarity contra centroides globales ──
        sims     = means @ query_embedding  # (N,) vectorizado
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_code = codigos[best_idx]

        # ── Paso 2: centroide de pose específica (si disponible) ────────
        if query_pose is not None:
            pose_key = self._classify_pose(query_pose)
            pose_sims = np.full(len(codigos), -1.0, dtype=np.float32)

            for i, codigo in enumerate(codigos):
                c = embeddings[codigo].get("centroids", {}).get(pose_key)
                if c is not None:
                    pose_sims[i] = float(c @ query_embedding)

            if pose_sims.max() > -1.0:  # al menos un alumno tiene centroide de pose
                pose_best_idx  = int(np.argmax(pose_sims))
                pose_best_sim  = float(pose_sims[pose_best_idx])
                pose_best_code = codigos[pose_best_idx]

                # Tomar el mejor entre centroide global y centroide de pose
                if pose_best_sim > best_sim:
                    best_sim  = pose_best_sim
                    best_code = pose_best_code
                    best_idx  = pose_best_idx
                    print(f"[EmbeddingManager] Pose '{pose_key}' mejoró score: "
                          f"{best_code} sim={best_sim:.3f}")

        # Umbral personal del alumno con mejor similitud
        student_threshold = embeddings[best_code].get("threshold", threshold)

        # ── Paso 3: coincidencia segura ────────────────────────────────
        if best_sim >= student_threshold:
            return best_code, best_sim

        # ── Paso 4: zona gris → validar con galería ────────────────────
        if best_sim >= config.THRESHOLD_GREY_LOW:
            gallery = embeddings[best_code].get("gallery")
            if gallery is not None and len(gallery) > 0:
                gallery_sims = gallery @ query_embedding        # (M,)
                votes        = int(np.sum(gallery_sims >= config.THRESHOLD_GREY_LOW))
                vote_ratio   = votes / len(gallery_sims)

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
        for filename in ("embedding_mean.npy", "embedding.npy", "gallery.npy", "centroids.npz"):
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
