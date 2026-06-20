"""
enroll.py
----------
CLI tool to register new students and generate their face embeddings.
Uses DeepFace — no dlib or Visual Studio required.

Usage:
    # Capture 10 photos from webcam
    python enroll.py --id S001 --name "Alice Kumar" --class_id CS101

    # Use existing image files
    python enroll.py --id S001 --name "Alice Kumar" --class_id CS101 \
                     --images alice1.jpg alice2.jpg alice3.jpg
"""

import argparse
import os
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from attendance.database   import Database
from attendance.detector   import FaceDetector
from attendance.recognizer import FaceRecognizer


def capture_from_webcam(count: int = 10) -> list[np.ndarray]:
    """Capture `count` face images interactively from the default webcam."""
    cap      = cv2.VideoCapture(0)
    detector = FaceDetector(backend="haar")
    samples  = []

    print(f"[Enroll] Camera opened. Press SPACE to capture, Q to quit.")
    print(f"[Enroll] Make sure your face is clearly visible and well-lit.")

    while len(samples) < count:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        boxes   = detector.detect(frame)

        for (x, y, w, h) in boxes:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 200, 80), 2)

        status = f"Captured: {len(samples)}/{count}   |   Faces detected: {len(boxes)}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2)
        cv2.imshow("Enrollment  [SPACE = capture | Q = quit]", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if boxes:
                samples.append(frame.copy())
                print(f"  Captured {len(samples)}/{count}")
            else:
                print("  No face detected in frame — try again.")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return samples


def load_from_files(paths: list[str]) -> list[np.ndarray]:
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
            print(f"  Loaded: {p}")
        else:
            print(f"  Warning: could not read {p}")
    return frames


def average_embedding(frames: list[np.ndarray],
                      detector: FaceDetector,
                      recognizer: FaceRecognizer) -> np.ndarray | None:
    """Compute the mean embedding across multiple frames for robustness."""
    encodings = []
    for i, frame in enumerate(frames):
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detector.detect(frame)
        if not boxes:
            print(f"  Frame {i+1}: no face detected, skipping.")
            continue
        # Use the largest detected face
        box = max(boxes, key=lambda b: b[2] * b[3])
        enc = recognizer.encode(rgb, box)
        if enc is not None:
            encodings.append(enc)
            print(f"  Frame {i+1}: embedding OK")
        else:
            print(f"  Frame {i+1}: embedding failed, skipping.")

    if not encodings:
        return None
    return np.mean(encodings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Enroll a student")
    parser.add_argument("--id",       required=True, help="Student ID (e.g. S001)")
    parser.add_argument("--name",     required=True, help="Full name")
    parser.add_argument("--class_id", required=True, help="Class identifier")
    parser.add_argument("--images",   nargs="*",     help="Paths to existing face images")
    parser.add_argument("--count",    type=int, default=10,
                        help="Number of webcam captures (default 10)")
    args = parser.parse_args()

    print(f"\n[Enroll] Enrolling: {args.name} ({args.id}) in {args.class_id}")
    print("[Enroll] Loading DeepFace model (first run downloads ~90 MB)...\n")

    detector   = FaceDetector(backend="haar")
    recognizer = FaceRecognizer()
    db         = Database()

    # Collect frames
    if args.images:
        frames = load_from_files(args.images)
        print(f"\n[Enroll] Loaded {len(frames)} image(s) from disk.")
    else:
        frames = capture_from_webcam(count=args.count)
        print(f"\n[Enroll] Captured {len(frames)} frame(s).")

    if not frames:
        print("[Enroll] No frames collected. Aborting.")
        return

    print("\n[Enroll] Computing face embeddings...")
    embedding = average_embedding(frames, detector, recognizer)

    if embedding is None:
        print("\n[Enroll] FAILED — no face could be detected in any frame.")
        print("Tips:")
        print("  • Ensure good lighting (face the light, not away from it)")
        print("  • Keep your face centred and close to the camera")
        print("  • Avoid dark rooms or strong backlighting")
        return

    db.add_student(args.id, args.name, args.class_id, embedding)
    print(f"\n[Enroll] SUCCESS: {args.name} enrolled in class {args.class_id}.")


if __name__ == "__main__":
    main()
