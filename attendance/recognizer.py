"""
attendance/recognizer.py
-------------------------
Face recognition using DeepFace (no dlib / no Visual Studio required).
Works on Windows out of the box.
"""

import os
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TensorFlow noise

try:
    from deepface import DeepFace
    DF_AVAILABLE = True
except ImportError:
    DF_AVAILABLE = False
    print("[Recognizer] WARNING: deepface not installed.\n"
          "  Run: pip install deepface tf-keras")


MODEL_NAME  = "Facenet"          # fast + accurate; alternatives: "VGG-Face", "ArcFace"
DETECTOR    = "opencv"           # no extra deps; alternatives: "retinaface", "mtcnn"
METRIC      = "cosine"


class FaceRecognizer:
    def __init__(self, threshold: float = 0.6, model: str = MODEL_NAME):
        self.threshold = threshold
        self.model     = model
        # Warm up the model on first init so the first real call is fast
        if DF_AVAILABLE:
            try:
                DeepFace.build_model(self.model)
                print(f"[Recognizer] DeepFace '{self.model}' model loaded.")
            except Exception as e:
                print(f"[Recognizer] Model warm-up failed: {e}")

    def encode(self, frame_rgb: np.ndarray, bounding_box: tuple):
        """
        Generate a face embedding for the face at bounding_box.

        Args:
            frame_rgb    : Full frame in RGB.
            bounding_box : (x, y, w, h) in OpenCV format.

        Returns:
            1-D numpy embedding array, or None on failure.
        """
        if not DF_AVAILABLE:
            return None

        x, y, w, h = bounding_box
        # Crop and convert to BGR for DeepFace
        face_crop = frame_rgb[y:y+h, x:x+w]
        face_bgr  = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

        try:
            result = DeepFace.represent(
                img_path         = face_bgr,
                model_name       = self.model,
                detector_backend = "skip",   # we already have the crop
                enforce_detection= False,
            )
            if result:
                return np.array(result[0]["embedding"])
        except Exception as e:
            print(f"[Recognizer] Encode error: {e}")
        return None

    def identify(self, encoding, known_encodings: list, known_ids: list):
        """
        Find the closest match in known_encodings.

        Returns:
            (student_id: str | None, confidence: float 0-1)
        """
        if encoding is None or not known_encodings:
            return None, 0.0

        # Cosine similarity: 1 - cosine_distance
        enc  = encoding / (np.linalg.norm(encoding) + 1e-9)
        sims = []
        for k in known_encodings:
            k_norm = k / (np.linalg.norm(k) + 1e-9)
            sims.append(float(np.dot(enc, k_norm)))

        best_idx  = int(np.argmax(sims))
        best_sim  = sims[best_idx]

        # threshold is stored as max-distance; convert: sim >= (1 - threshold)
        if best_sim >= (1.0 - self.threshold):
            return known_ids[best_idx], round(best_sim, 4)
        return None, round(best_sim, 4)
