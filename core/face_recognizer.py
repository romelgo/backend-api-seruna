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

import config

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
            ok, reason = recognizer.quality_gate(face, bgr_frame)
            if ok:
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

    def quality_gate(self, face, bgr_frame: np.ndarray) -> Tuple[bool, str]:
        """
        Filtra rostros de baja calidad antes de llamar a ArcFace.

        Criterios (todos configurables en config.py):
          1. det_score >= SCRFD_DET_THRESHOLD     (confianza SCRFD)
          2. area bbox  >= MIN_FACE_AREA           (rostro suficientemente grande)
          3. |roll|     <= MAX_ROLL_DEGREES        (pose: estimado con eye keypoints)
          4. brillo medio del ROI en [BRIGHTNESS_MIN, BRIGHTNESS_MAX]

        Args:
            face:      Objeto Face de insightface.
            bgr_frame: Frame BGR original (para extraer ROI de iluminación).

        Returns:
            (True, "ok") si supera todos los filtros.
            (False, motivo) si es rechazado.
        """
        # 1. Confianza de detección SCRFD
        if hasattr(face, "det_score") and face.det_score < config.SCRFD_DET_THRESHOLD:
            return False, f"det_score_bajo ({face.det_score:.2f} < {config.SCRFD_DET_THRESHOLD})"

        x1, y1, x2, y2 = face.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 2. Área mínima del bounding box
        area = (x2 - x1) * (y2 - y1)
        if area < config.MIN_FACE_AREA:
            return False, f"rostro_pequeno ({area}px² < {config.MIN_FACE_AREA}px²)"

        # 3. Estimación de roll con keypoints (ojo_izq → ojo_der)
        if face.kps is not None and len(face.kps) >= 2:
            eye_dx = float(face.kps[1][0] - face.kps[0][0])
            eye_dy = float(face.kps[1][1] - face.kps[0][1])
            roll = abs(np.degrees(np.arctan2(eye_dy, eye_dx)))
            if roll > config.MAX_ROLL_DEGREES:
                return False, f"pose_rotada (roll={roll:.1f}° > {config.MAX_ROLL_DEGREES}°)"

        # 4. Iluminación del ROI
        h, w = bgr_frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        roi = bgr_frame[y1c:y2c, x1c:x2c]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            if brightness < config.BRIGHTNESS_MIN or brightness > config.BRIGHTNESS_MAX:
                return False, (
                    f"iluminacion ({brightness:.1f} fuera de "
                    f"[{config.BRIGHTNESS_MIN}, {config.BRIGHTNESS_MAX}])"
                )

        return True, "ok"

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula similitud coseno entre dos embeddings L2-normalizados.

        Args:
            a, b: Vectores (512,) L2-normalizados.

        Returns:
            Similitud en [-1, 1]. Tipicamente >0.6 = mismo estudiante.
        """
        # insightface ya normaliza los embeddings, asi que dot product = cosine sim
        return float(np.dot(a, b))
