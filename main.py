"""
AI-Powered Smart Attendance System
====================================
Entry point — starts the live attendance session.

Usage:
    python main.py --class_id CS101 --camera 0
    python main.py --class_id CS101 --camera rtsp://192.168.1.10/stream
"""

import argparse
from attendance.database import Database
from attendance.session import AttendanceSession


DEFAULT_CLASS_ID = "AI111"


def parse_args():
    parser = argparse.ArgumentParser(description="AI-Powered Smart Attendance System")
    parser.add_argument("--class_id", type=str, default=None, help="Class/subject identifier (optional)")
    parser.add_argument("--camera",   type=str, default="0",  help="Camera index or RTSP URL")
    parser.add_argument("--threshold",type=float, default=0.6, help="Recognition confidence threshold")
    parser.add_argument("--show",     dest="show", action="store_true", default=True,
                        help="Display live annotated feed (default)")
    parser.add_argument("--no-show",   dest="show", action="store_false",
                        help="Run without opening the camera window")
    return parser.parse_args()


def main():
    args = parse_args()
    db = Database()

    class_id = args.class_id or DEFAULT_CLASS_ID
    if not db.get_students_by_class(class_id):
        class_id = db.get_latest_class_id() or class_id
    if not class_id:
        print("[INFO] No class_id provided and no enrolled students found. Enroll at least one student first.")
        return

    camera_src = int(args.camera) if args.camera.isdigit() else args.camera

    session = AttendanceSession(
        class_id=class_id,
        camera_src=camera_src,
        threshold=args.threshold,
        show_feed=args.show,
    )

    print(f"[INFO] Starting session for class '{class_id}'. Press Q to stop.")
    session.run()
    print("[INFO] Session ended. Generating report...")
    session.generate_report()


if __name__ == "__main__":
    main()
