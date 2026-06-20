"""
attendance/reports.py
----------------------
Generates attendance reports:
  1. CSV / Excel export via Pandas
  2. Natural-language summary via LLM (Anthropic Claude)

The LLM call is optional — if ANTHROPIC_API_KEY is not set,
only the CSV/Excel report is produced.
"""

import os
import json
import datetime
from pathlib import Path

import pandas as pd

from .database import Database


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


class ReportGenerator:
    def __init__(self, db: Database):
        self.db = db

    # ── Main entry ──────────────────────────────────────────────────────
    def generate(self, session_id: str, class_id: str):
        records = self.db.get_session_attendance(session_id)
        all_students = self.db.get_students_by_class(class_id)

        # Build present/absent sets
        present_ids = {r["student_id"] for r in records}
        absent = [
            {"student_id": s["student_id"], "name": s["name"]}
            for s in all_students
            if s["student_id"] not in present_ids
        ]

        # CSV / Excel
        csv_path   = self._export_csv(records, absent, session_id, class_id)
        excel_path = self._export_excel(records, absent, session_id, class_id)

        # LLM summary
        llm_summary = self._llm_summary(records, absent, class_id)

        print("\n" + "=" * 60)
        print("ATTENDANCE REPORT")
        print("=" * 60)
        print(f"Class      : {class_id}")
        print(f"Session    : {session_id[:8]}...")
        print(f"Present    : {len(present_ids)} / {len(all_students)}")
        print(f"Absent     : {len(absent)}")
        print(f"CSV saved  : {csv_path}")
        print(f"Excel saved: {excel_path}")
        if llm_summary:
            print("\nAI SUMMARY")
            print("-" * 60)
            print(llm_summary)
        print("=" * 60)

    # ── CSV export ──────────────────────────────────────────────────────
    def _export_csv(self, records, absent, session_id, class_id) -> Path:
        rows = [
            {
                "student_id": r["student_id"],
                "name":       r["name"],
                "status":     "PRESENT",
                "timestamp":  r["timestamp"],
                "confidence": round(r["confidence"], 4),
                "class_id":   class_id,
                "session_id": session_id,
            }
            for r in records
        ]
        for a in absent:
            rows.append({
                "student_id": a["student_id"],
                "name":       a["name"],
                "status":     "ABSENT",
                "timestamp":  "",
                "confidence": "",
                "class_id":   class_id,
                "session_id": session_id,
            })

        df   = pd.DataFrame(rows)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"attendance_{class_id}_{ts}.csv"
        df.to_csv(path, index=False)
        return path

    # ── Excel export ────────────────────────────────────────────────────
    def _export_excel(self, records, absent, session_id, class_id) -> Path:
        present_df = pd.DataFrame([
            {
                "Student ID": r["student_id"],
                "Name":       r["name"],
                "Status":     "PRESENT",
                "Time":       r["timestamp"],
                "Confidence": round(r["confidence"], 4),
            }
            for r in records
        ])
        absent_df = pd.DataFrame([
            {"Student ID": a["student_id"], "Name": a["name"],
             "Status": "ABSENT", "Time": "", "Confidence": ""}
            for a in absent
        ])

        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"attendance_{class_id}_{ts}.xlsx"

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            present_df.to_excel(writer, sheet_name="Present", index=False)
            absent_df.to_excel(writer,  sheet_name="Absent",  index=False)

        return path

    # ── LLM summary ─────────────────────────────────────────────────────
    def _llm_summary(self, records, absent, class_id) -> str | None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[Report] ANTHROPIC_API_KEY not set — skipping LLM summary.")
            return None

        try:
            import anthropic

            payload = {
                "class_id":        class_id,
                "date":            datetime.date.today().isoformat(),
                "total_enrolled":  len(records) + len(absent),
                "present_count":   len(records),
                "absent_count":    len(absent),
                "absent_students": [a["name"] for a in absent],
                "present_students": [r["name"] for r in records],
            }

            prompt = f"""You are an academic attendance analyst.
Given the session data below, return a JSON object with exactly these keys:
  - "summary"         : 2-3 sentence plain-English overview
  - "attendance_rate" : percentage (number, 0-100)
  - "concerns"        : list of concern strings (empty list if none)
  - "recommendations" : list of action strings for the instructor

Respond with valid JSON only. No preamble, no markdown fences.

Session data:
{json.dumps(payload, indent=2)}
"""

            client   = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 600,
                messages   = [{"role": "user", "content": prompt}],
            )

            raw    = response.content[0].text.strip()
            parsed = json.loads(raw)
            return json.dumps(parsed, indent=2)

        except Exception as e:
            print(f"[Report] LLM summary failed: {e}")
            return None
