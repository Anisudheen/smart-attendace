"""
attendance/session.py
----------------------
Orchestrates a single attendance session:
  Camera → Detection → Recognition → Duplicate Guard → DB Log → Annotated Display
"""

import uuid
import datetime
import numpy as np
import cv2

from .camera      import CameraManager
from .detector    import FaceDetector
from .recognizer  import FaceRecognizer
from .database    import Database
from .reports     import ReportGenerator


class AttendanceSession:
    PROCESS_EVERY_N_FRAMES = 5
    PREVIEW_WIDTH = 960

    def __init__(
        self,
        class_id:   str,
        camera_src  = 0,
        threshold:  float = 0.6,
        show_feed:  bool  = True,
        detector_backend: str = "haar",
    ):
        self.class_id   = class_id
        self.session_id = str(uuid.uuid4())
        self.show_feed  = show_feed

        self.camera     = CameraManager(camera_src)
        self.detector   = FaceDetector(backend=detector_backend)
        self.recognizer = FaceRecognizer(threshold=threshold)
        self.db         = Database()
        self.reporter   = ReportGenerator(self.db)

        # Session state
        self.seen: set[str] = set()          # student_ids marked this session
        self.frame_count   = 0
        self.start_time    = None
        self._report_written = False

        # Cache class embeddings once at start
        self._known_encodings: list[np.ndarray] = []
        self._known_ids:       list[str]         = []

    # ── Setup ────────────────────────────────────────────────────────────
    def _load_embeddings(self):
        students = self.db.get_students_by_class(self.class_id)
        self._known_encodings = []
        self._known_ids       = []
        for s in students:
            if s["embedding"] is not None:
                self._known_encodings.append(s["embedding"])
                self._known_ids.append(s["student_id"])
        print(f"[Session] Loaded {len(self._known_ids)} student embeddings for class '{self.class_id}'.")

    # ── Main loop ────────────────────────────────────────────────────────
    def run(self):
        self._load_embeddings()
        self.start_time = datetime.datetime.now()

        with self.camera:
            while True:
                ret, frame = self.camera.read_frame()
                if not ret:
                    print("[Session] Frame read failed — stopping.")
                    break

                self.frame_count += 1
                display_frame = self._resize_frame(frame)

                # Process fewer frames to keep the preview responsive.
                if self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                    self._process_frame(display_frame)

                if self.show_feed:
                    cv2.imshow("Attendance System  [Q to quit]", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cv2.destroyAllWindows()
        print(f"[Session] Ended. Marked {len(self.seen)} students present.")

    def _process_frame(self, frame: np.ndarray):
        boxes  = self.detector.detect(frame)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in boxes:
            encoding = self.recognizer.encode(rgb, (x, y, w, h))
            student_id, confidence = self.recognizer.identify(
                encoding, self._known_encodings, self._known_ids
            )

            if student_id and student_id not in self.seen:
                self.seen.add(student_id)
                self.db.mark_attendance(student_id, self.session_id,
                                        self.class_id, confidence)
                print(f"[Session] PRESENT: {student_id} (confidence={confidence:.2f})")
                self._write_report_snapshot()

            # Annotate frame
            if self.show_feed:
                label = student_id if student_id else "Unknown"
                color = (0, 200, 80) if student_id else (0, 60, 220)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.PREVIEW_WIDTH:
            return frame

        scale = self.PREVIEW_WIDTH / float(width)
        resized_height = int(height * scale)
        return cv2.resize(frame, (self.PREVIEW_WIDTH, resized_height), interpolation=cv2.INTER_AREA)

    # ── Report ───────────────────────────────────────────────────────────
    def _write_report_snapshot(self):
        self.reporter.generate(
            session_id=self.session_id,
            class_id=self.class_id,
        )
        self._report_written = True

    def generate_report(self):
        if not self._report_written:
            self._write_report_snapshot()
