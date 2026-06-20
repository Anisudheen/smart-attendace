"""
attendance/detector.py
-----------------------
Face detection using OpenCV.

Two backends:
  - 'haar'  : Haar Cascade (lightweight, CPU-only)
  - 'dnn'   : SSD + MobileNet via OpenCV DNN (more accurate)

Returns a list of bounding boxes [(x, y, w, h), ...].
"""

import cv2
import numpy as np
import os


# ── Haar Cascade path ──────────────────────────────────────────────────────
HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── DNN model paths (download once and place in models/) ──────────────────
DNN_PROTOTXT = "models/deploy.prototxt"
DNN_MODEL    = "models/res10_300x300_ssd_iter_140000.caffemodel"


class FaceDetector:
    def __init__(self, backend: str = "haar", min_confidence: float = 0.5):
        """
        Args:
            backend        : 'haar' or 'dnn'
            min_confidence : DNN confidence threshold (ignored for Haar)
        """
        self.backend        = backend
        self.min_confidence = min_confidence
        self._net           = None
        self._cascade       = None
        self._load()

    # ── Initialisation ──────────────────────────────────────────────────
    def _load(self):
        if self.backend == "dnn":
            if not os.path.exists(DNN_PROTOTXT) or not os.path.exists(DNN_MODEL):
                print(
                    "[Detector] DNN model files not found. "
                    "Falling back to Haar Cascade.\n"
                    "  Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
                )
                self.backend = "haar"
            else:
                self._net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_MODEL)
                print("[Detector] DNN backend loaded.")

        if self.backend == "haar":
            self._cascade = cv2.CascadeClassifier(HAAR_XML)
            print("[Detector] Haar Cascade backend loaded.")

    # ── Public API ──────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces in a BGR frame.

        Returns:
            List of (x, y, w, h) bounding boxes.
        """
        if self.backend == "dnn":
            return self._detect_dnn(frame)
        return self._detect_haar(frame)

    # ── Haar ────────────────────────────────────────────────────────────
    def _detect_haar(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improve contrast
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    # ── DNN ─────────────────────────────────────────────────────────────
    def _detect_dnn(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.min_confidence:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
