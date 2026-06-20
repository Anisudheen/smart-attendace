"""
attendance/database.py
-----------------------
SQLite-backed storage for:
  - Student records + face embeddings
  - Attendance logs

Schema
------
students   : student_id, name, class_id, embedding (BLOB), enrolled_at
attendance : id, student_id, session_id, class_id, timestamp, confidence, status
"""

import sqlite3
import pickle
import datetime
import numpy as np
import csv
from pathlib import Path


DB_PATH = Path("data/attendance.db")
STUDENTS_CSV_PATH = Path("data/enrolled_students.csv")


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        print(f"[DB] Connected to {db_path}")

    # ── Schema ──────────────────────────────────────────────────────────
    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                student_id   TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                class_id     TEXT NOT NULL,
                embedding    BLOB,
                enrolled_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id   TEXT NOT NULL,
                session_id   TEXT NOT NULL,
                class_id     TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                confidence   REAL,
                status       TEXT DEFAULT 'PRESENT',
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );
        """)
        self.conn.commit()

    # ── Student CRUD ────────────────────────────────────────────────────
    def add_student(self, student_id: str, name: str, class_id: str,
                    embedding: np.ndarray | None = None):
        blob = pickle.dumps(embedding) if embedding is not None else None
        self.conn.execute(
            "INSERT OR REPLACE INTO students (student_id, name, class_id, embedding) VALUES (?,?,?,?)",
            (student_id, name, class_id, blob)
        )
        self.conn.commit()
        self._export_students_csv()
        print(f"[DB] Enrolled student: {name} ({student_id})")

    def _export_students_csv(self):
        STUDENTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows = self.conn.execute(
            "SELECT student_id, name, class_id, enrolled_at FROM students ORDER BY enrolled_at, student_id"
        ).fetchall()

        with STUDENTS_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["student_id", "name", "class_id", "enrolled_at"])
            for row in rows:
                writer.writerow([row["student_id"], row["name"], row["class_id"], row["enrolled_at"]])

    def update_embedding(self, student_id: str, embedding: np.ndarray):
        blob = pickle.dumps(embedding)
        self.conn.execute(
            "UPDATE students SET embedding=? WHERE student_id=?",
            (blob, student_id)
        )
        self.conn.commit()

    def get_all_students(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM students").fetchall()
        students = []
        for row in rows:
            s = dict(row)
            s["embedding"] = pickle.loads(s["embedding"]) if s["embedding"] else None
            students.append(s)
        return students

    def get_students_by_class(self, class_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM students WHERE class_id=?", (class_id,)
        ).fetchall()
        students = []
        for row in rows:
            s = dict(row)
            s["embedding"] = pickle.loads(s["embedding"]) if s["embedding"] else None
            students.append(s)
        return students

    def get_latest_class_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT class_id FROM students ORDER BY enrolled_at DESC, student_id DESC LIMIT 1"
        ).fetchone()
        return row["class_id"] if row else None

    def delete_student(self, student_id: str) -> bool:
        student = self.conn.execute(
            "SELECT student_id FROM students WHERE student_id=?",
            (student_id,)
        ).fetchone()
        if not student:
            return False

        self.conn.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
        self.conn.execute("DELETE FROM students WHERE student_id=?", (student_id,))
        self.conn.commit()
        self._export_students_csv()
        print(f"[DB] Removed student: {student_id}")
        return True

    # ── Attendance ──────────────────────────────────────────────────────
    def mark_attendance(self, student_id: str, session_id: str,
                        class_id: str, confidence: float):
        timestamp = datetime.datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO attendance
               (student_id, session_id, class_id, timestamp, confidence, status)
               VALUES (?,?,?,?,?,'PRESENT')""",
            (student_id, session_id, class_id, timestamp, confidence)
        )
        self.conn.commit()

    def get_session_attendance(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT a.*, s.name
               FROM attendance a
               JOIN students s ON a.student_id = s.student_id
               WHERE a.session_id = ?
               ORDER BY a.timestamp""",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_class_attendance(self, class_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT a.*, s.name
               FROM attendance a
               JOIN students s ON a.student_id = s.student_id
               WHERE a.class_id = ?
               ORDER BY a.timestamp DESC""",
            (class_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
