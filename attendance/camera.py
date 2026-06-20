"""
attendance/camera.py
--------------------
Handles video frame acquisition via OpenCV.
Supports webcam index or RTSP/HTTP stream URL.
"""

import cv2


class CameraManager:
    def __init__(self, source=0):
        """
        Args:
            source: int (webcam index) or str (RTSP / HTTP URL)
        """
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")
        # Set preferred resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"[Camera] Opened source: {self.source}")

    def read_frame(self):
        """Returns (success: bool, frame: np.ndarray)."""
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            print("[Camera] Released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
