"""
Motor de reconocimiento facial usando insightface buffalo_l.

Stack:
  - Detección:     SCRFD-34GF  (det_10g.onnx)   — via insightface
  - Reconocimiento: ArcFace ResNet100 (w600k_r50.onnx) — via insightface
  - Preprocesamiento: norm_crop() → 112×112

Los modelos se descargan automáticamente en ~/.insightface/models/buffalo_l/
"""
import numpy as np
from typing import List, Optional, Tuple
import cv2

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class FaceRecognizer:
    """
    Motor unificado de detección + reconocimiento facial.

    Uso:
        recognizer = FaceRecognizer()
        faces = recognizer.get_faces(bgr_frame)
        for face in faces:
            emb = face.embedding   # np.ndarray (512,) L2-normalizado
    """

    def __init__(self, ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)):
        """
        Inicializa el motor insightface con buffalo_l.

        Args:
            ctx_id: ID del dispositivo CUDA (0 = primera GPU). Usa -1 para CPU.
            det_size: Tamaño de entrada del detector SCRFD (W, H).
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "insightface no está instalado. "
                "Ejecuta: pip install insightface onnxruntime-gpu"
            )

        self.ctx_id = ctx_id
        self.det_size = det_size

        print("[FaceRecognizer] Cargando buffalo_l (SCRFD-34GF + ArcFace R100)...")
        print(f"[FaceRecognizer] Device ctx_id={ctx_id} | det_size={det_size}")

        # FaceAnalysis carga automáticamente SCRFD (detección) + ArcFace (reconocimiento)
        # Descarga en ~/.insightface/models/buffalo_l/ si no existe
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

        print("[FaceRecognizer] Listo. SCRFD-34GF + ArcFace ResNet100 activos.")

    def get_faces(self, bgr_frame: np.ndarray) -> list:
        """
        Detecta y reconoce todos los rostros en un frame.

        Args:
            bgr_frame: Imagen BGR (cualquier resolución).

        Returns:
            Lista de objetos Face de insightface. Cada uno tiene:
              - face.bbox:      [x1, y1, x2, y2]
              - face.kps:       np.ndarray (5, 2) — landmarks
              - face.embedding: np.ndarray (512,) — L2-normalizado
              - face.det_score: float — confianza de detección
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return []
        return self._app.get(bgr_frame)

    def get_largest_face(self, bgr_frame: np.ndarray) -> Optional[object]:
        """
        Retorna solo el rostro más grande (mayor área de bbox).

        Args:
            bgr_frame: Imagen BGR.

        Returns:
            Objeto Face o None si no se detecta ninguno.
        """
        faces = self.get_faces(bgr_frame)
        if not faces:
            return None

        # Ordenar por área de bbox (descendente)
        def bbox_area(f):
            x1, y1, x2, y2 = f.bbox
            return (x2 - x1) * (y2 - y1)

        return max(faces, key=bbox_area)

    def get_aligned_crop(
        self,
        bgr_frame: np.ndarray,
        face,
        image_size: int = 112,
    ) -> np.ndarray:
        """
        Obtiene el recorte alineado 112×112 usando norm_crop de insightface.

        Args:
            bgr_frame: Imagen BGR original.
            face: Objeto Face retornado por get_faces().
            image_size: Tamaño del crop de salida (default 112 para ArcFace).

        Returns:
            Imagen BGR alineada de tamaño (image_size, image_size, 3).
        """
        aligned = face_align.norm_crop(bgr_frame, landmark=face.kps, image_size=image_size)
        return aligned

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula similitud coseno entre dos embeddings L2-normalizados.

        Args:
            a, b: Vectores (512,) L2-normalizados.

        Returns:
            Similitud en [-1, 1]. Típicamente >0.6 = mismo estudiante.
        """
        # insightface ya normaliza los embeddings, así que dot product = cosine sim
        return float(np.dot(a, b))
