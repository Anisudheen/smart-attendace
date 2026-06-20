"""
Streamlit app for the Smart Attendance System.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from attendance.camera import CameraManager
from attendance.database import Database
from attendance.detector import FaceDetector
from attendance.recognizer import FaceRecognizer
from attendance.reports import ReportGenerator


DEFAULT_CLASS_ID = "AI111"
DEFAULT_CAMERA = "0"
PROCESS_EVERY_N_FRAMES = 5
PREVIEW_WIDTH = 960


@st.cache_resource
def build_services(threshold: float, detector_backend: str):
    db = Database()
    detector = FaceDetector(backend=detector_backend)
    recognizer = FaceRecognizer(threshold=threshold)
    reporter = ReportGenerator(db)
    return db, detector, recognizer, reporter


def parse_camera_source(raw_value: str):
    value = (raw_value or "").strip()
    if value.isdigit():
        return int(value)
    return value or 0


def resolve_class_id(db: Database, requested_class_id: str) -> str:
    requested = (requested_class_id or "").strip()
    if requested and db.get_students_by_class(requested):
        return requested

    latest = db.get_latest_class_id()
    if latest:
        return latest

    return requested or DEFAULT_CLASS_ID


def resize_frame(frame: np.ndarray, target_width: int = PREVIEW_WIDTH) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame

    scale = target_width / float(width)
    new_height = int(height * scale)
    return cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)


def decode_image_bytes(data: bytes | None) -> np.ndarray | None:
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def average_embedding(frames: list[np.ndarray], detector: FaceDetector, recognizer: FaceRecognizer) -> np.ndarray | None:
    encodings = []
    for index, frame in enumerate(frames):
        boxes = detector.detect(frame)
        if not boxes:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box = max(boxes, key=lambda item: item[2] * item[3])
        embedding = recognizer.encode(rgb, box)
        if embedding is not None:
            encodings.append(embedding)

    if not encodings:
        return None
    return np.mean(encodings, axis=0)


def init_state():
    defaults = {
        "running": False,
        "camera": None,
        "session_id": None,
        "class_id": DEFAULT_CLASS_ID,
        "frame_count": 0,
        "seen": set(),
        "report_generated": False,
        "status_message": "Ready.",
        "last_report_message": "",
        "enroll_message": "",
        "enroll_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def open_session(class_id: str, camera_source, threshold: float, detector_backend: str):
    db, detector, recognizer, reporter = build_services(threshold, detector_backend)
    resolved_class_id = resolve_class_id(db, class_id)

    if not db.get_students_by_class(resolved_class_id):
        st.error(f"No enrolled students found for class '{resolved_class_id}'. Enroll at least one student first.")
        return False

    camera = CameraManager(camera_source)
    camera.open()

    st.session_state.running = True
    st.session_state.camera = camera
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.class_id = resolved_class_id
    st.session_state.frame_count = 0
    st.session_state.seen = set()
    st.session_state.report_generated = False
    st.session_state.status_message = f"Running attendance for class '{resolved_class_id}'."
    st.session_state.last_report_message = ""
    st.session_state.db = db
    st.session_state.detector = detector
    st.session_state.recognizer = recognizer
    st.session_state.reporter = reporter
    st.session_state.known_students = db.get_students_by_class(resolved_class_id)
    st.session_state.known_encodings = []
    st.session_state.known_ids = []

    for student in st.session_state.known_students:
        if student["embedding"] is not None:
            st.session_state.known_encodings.append(student["embedding"])
            st.session_state.known_ids.append(student["student_id"])

    if not st.session_state.known_ids:
        st.warning(f"Class '{resolved_class_id}' has no student embeddings yet. Enroll students first.")

    return True


def close_session(generate_report: bool = True):
    if generate_report and st.session_state.get("running") and not st.session_state.get("report_generated"):
        try:
            st.session_state.reporter.generate(
                session_id=st.session_state.session_id,
                class_id=st.session_state.class_id,
            )
            st.session_state.report_generated = True
            st.session_state.last_report_message = "Final report generated."
        except Exception as exc:
            st.session_state.last_report_message = f"Report generation failed: {exc}"

    camera = st.session_state.get("camera")
    if camera is not None:
        camera.release()

    st.session_state.camera = None
    st.session_state.running = False


def process_frame(frame: np.ndarray):
    st.session_state.frame_count += 1
    display_frame = resize_frame(frame)

    if st.session_state.frame_count % PROCESS_EVERY_N_FRAMES != 0:
        return display_frame, []

    boxes = st.session_state.detector.detect(display_frame)
    rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    newly_marked = []

    for (x, y, w, h) in boxes:
        encoding = st.session_state.recognizer.encode(rgb, (x, y, w, h))
        student_id, confidence = st.session_state.recognizer.identify(
            encoding,
            st.session_state.known_encodings,
            st.session_state.known_ids,
        )

        label = student_id if student_id else "Unknown"
        color = (0, 200, 80) if student_id else (0, 60, 220)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            display_frame,
            f"{label} {confidence:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        if student_id and student_id not in st.session_state.seen:
            st.session_state.seen.add(student_id)
            st.session_state.db.mark_attendance(
                student_id,
                st.session_state.session_id,
                st.session_state.class_id,
                confidence,
            )
            newly_marked.append((student_id, confidence))

    if newly_marked:
        try:
            st.session_state.reporter.generate(
                session_id=st.session_state.session_id,
                class_id=st.session_state.class_id,
            )
            st.session_state.report_generated = True
        except Exception as exc:
            st.session_state.last_report_message = f"Live report update failed: {exc}"

    return display_frame, newly_marked


def enroll_student(class_id: str, student_id: str, name: str, uploaded_files, camera_snapshot):
    db, detector, recognizer, _ = build_services(0.6, "haar")
    resolved_class_id = (class_id or "").strip() or DEFAULT_CLASS_ID

    frames: list[np.ndarray] = []
    for uploaded in uploaded_files or []:
        frame = decode_image_bytes(uploaded.getvalue())
        if frame is not None:
            frames.append(frame)

    snapshot_frame = decode_image_bytes(camera_snapshot.getvalue()) if camera_snapshot is not None else None
    if snapshot_frame is not None:
        frames.append(snapshot_frame)

    if not frames:
        st.error("Add at least one face photo from upload or the camera snapshot.")
        return False

    embedding = average_embedding(frames, detector, recognizer)
    if embedding is None:
        st.error("No face could be detected in the provided image(s).")
        return False

    db.add_student(student_id, name, resolved_class_id, embedding)
    st.session_state.enroll_message = f"Enrolled {name} ({student_id}) in {resolved_class_id}."
    st.session_state.enroll_error = ""
    return True


def enrolled_students_table(db: Database):
    all_students = db.get_all_students()
    if not all_students:
        st.info("No students enrolled yet.")
        return

    cleaned = []
    for student in all_students:
        row = dict(student)
        row.pop("embedding", None)
        cleaned.append(row)
    st.dataframe(cleaned, use_container_width=True, hide_index=True)


def student_removal_panel(db: Database):
    all_students = db.get_all_students()
    if not all_students:
        st.info("No students available to remove.")
        return

    options = [f"{student['student_id']} | {student['name']} | {student['class_id']}" for student in all_students]
    selected = st.selectbox("Select a student to remove", options)

    if st.button("Remove selected student", type="primary"):
        student_id = selected.split(" | ", 1)[0]
        removed = db.delete_student(student_id)
        if removed:
            st.success(f"Removed student {student_id}.")
            st.rerun()
        else:
            st.error("Student not found.")


def main():
    st.set_page_config(page_title="Smart Attendance", layout="wide")
    init_state()

    st.title("Smart Attendance System")
    st.caption("Use one app for live attendance and student enrollment.")

    tabs = st.tabs(["Attendance", "Enroll", "Students", "Reports"])

    with tabs[0]:
        control_col, live_col = st.columns([1, 2], gap="large")

        with control_col:
            st.subheader("Session Control")
            class_id_input = st.text_input("Class ID", value=DEFAULT_CLASS_ID, key="attendance_class_id")
            camera_input = st.text_input("Camera source", value=DEFAULT_CAMERA, key="attendance_camera")
            threshold = st.slider("Match threshold", min_value=0.3, max_value=0.9, value=0.6, step=0.05, key="attendance_threshold")
            detector_backend = st.selectbox("Detector backend", ["haar", "dnn"], index=0, key="attendance_detector_backend")

            if st.button("Start live session", use_container_width=True, disabled=st.session_state.running):
                started = open_session(class_id_input, parse_camera_source(camera_input), threshold, detector_backend)
                if started:
                    st.rerun()

            if st.button("Stop session", use_container_width=True, disabled=not st.session_state.running):
                close_session(generate_report=True)
                st.rerun()

            st.divider()
            st.metric("Running", "Yes" if st.session_state.running else "No")
            st.metric("Recognized", len(st.session_state.seen))
            st.metric("Frames", st.session_state.frame_count)
            st.write(st.session_state.status_message)
            if st.session_state.last_report_message:
                st.info(st.session_state.last_report_message)

        if st.session_state.running:
            camera = st.session_state.camera
            if camera is None:
                st.error("Camera is not open.")
                close_session(generate_report=False)
                st.stop()

            ret, frame = camera.read_frame()
            if not ret:
                st.error("Could not read from the camera source.")
                close_session(generate_report=True)
                st.stop()

            display_frame, newly_marked = process_frame(frame)

            with live_col:
                st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), caption="Live camera", use_container_width=True)

                session_records = st.session_state.db.get_session_attendance(st.session_state.session_id)
                present_rows = [dict(row) for row in session_records]

                st.subheader("Present students table")
                if present_rows:
                    st.dataframe(present_rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No students marked present yet.")

                if newly_marked:
                    names = ", ".join(student_id for student_id, _ in newly_marked)
                    st.success(f"Marked present: {names}")

                st.subheader("Session details")
                details_col1, details_col2 = st.columns(2)
                with details_col1:
                    st.write(f"Class: {st.session_state.class_id}")
                    st.write(f"Session ID: {st.session_state.session_id}")
                with details_col2:
                    st.write(f"Recognized: {len(st.session_state.seen)}")
                    st.write(f"Frames processed: {st.session_state.frame_count}")

                st.write("Press Stop session when you are done.")

            time.sleep(0.05)
            st.rerun()
        else:
            st.info("Start a session from the controls on the left. The app will keep running and mark attendance when a face is recognized.")

    with tabs[1]:
        st.subheader("Enroll a student")
        st.write("Add one or more face photos, then save them to the shared roster and database.")

        with st.form("enroll_form", clear_on_submit=False):
            enroll_class_id = st.text_input("Class ID", value=st.session_state.class_id or DEFAULT_CLASS_ID)
            enroll_student_id = st.text_input("Student ID")
            enroll_name = st.text_input("Full name")
            uploaded_files = st.file_uploader(
                "Upload face photos",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
            )
            camera_snapshot = st.camera_input("Or take one snapshot with the camera")
            enroll_submit = st.form_submit_button("Enroll student")

        if enroll_submit:
            if not enroll_student_id.strip() or not enroll_name.strip():
                st.error("Student ID and full name are required.")
            else:
                success = enroll_student(
                    enroll_class_id,
                    enroll_student_id.strip(),
                    enroll_name.strip(),
                    uploaded_files,
                    camera_snapshot,
                )
                if success:
                    st.success(st.session_state.enroll_message)
                    st.rerun()

        if st.session_state.enroll_message:
            st.success(st.session_state.enroll_message)

    with tabs[2]:
        st.subheader("Enrolled students")
        db = st.session_state.get("db") or Database()
        student_removal_panel(db)
        st.divider()
        enrolled_students_table(db)

    with tabs[3]:
        st.subheader("Reports")
        reports_dir = Path("reports")
        if reports_dir.exists():
            report_files = sorted(reports_dir.glob("attendance_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
            if report_files:
                report_names = [path.name for path in report_files]
                selected_report_name = st.selectbox("Select report", report_names)
                selected_report_path = reports_dir / selected_report_name

                try:
                    report_df = pd.read_csv(selected_report_path)
                    st.dataframe(report_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download CSV",
                        data=selected_report_path.read_bytes(),
                        file_name=selected_report_path.name,
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.caption(f"Rows: {len(report_df)} | File: {selected_report_path.name}")
                except Exception as exc:
                    st.error(f"Could not load report: {exc}")
            else:
                st.info("No attendance reports generated yet.")
        else:
            st.info("No reports folder found yet.")


if __name__ == "__main__":
    main()