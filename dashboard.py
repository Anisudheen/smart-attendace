"""
dashboard.py
-------------
Flask web dashboard for the Smart Attendance System.

Endpoints:
  GET  /                          — Overview / stats
  GET  /api/students              — List all students
  GET  /api/attendance/<class_id> — Full attendance log for a class
  GET  /api/session/<session_id>  — Single session records

Run:
    python dashboard.py
"""

from flask import Flask, jsonify, render_template_string
from attendance.database import Database

app = Flask(__name__)
db  = Database()


# ── HTML template (inline for single-file convenience) ────────────────────
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Smart Attendance Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; background: #f0f4f8; color: #1a1a2e; }
    header {
      background: #1a3a5c; color: white; padding: 1.2rem 2rem;
      display: flex; align-items: center; gap: 1rem;
    }
    header h1 { font-size: 1.4rem; }
    header span { font-size: 0.9rem; opacity: 0.7; }
    .container { max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 1rem; margin-bottom: 2rem; }
    .card { background: white; border-radius: 10px; padding: 1.2rem 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    .card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .card .value { font-size: 2rem; font-weight: 700; color: #1a3a5c; margin-top: 0.3rem; }
    table { width: 100%; background: white; border-radius: 10px; overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,.08); border-collapse: collapse; }
    thead { background: #1a3a5c; color: white; }
    th, td { padding: 0.75rem 1rem; text-align: left; font-size: 0.9rem; }
    tbody tr:nth-child(even) { background: #ebf5fb; }
    .badge {
      display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    }
    .present { background: #d4edda; color: #155724; }
    .absent  { background: #f8d7da; color: #721c24; }
    h2 { margin-bottom: 1rem; color: #1a3a5c; }
  </style>
</head>
<body>
  <header>
    <h1>Smart Attendance System</h1>
    <span>Real-time AI attendance dashboard</span>
  </header>
  <div class="container">
    <div class="cards" id="stats">
      <div class="card"><div class="label">Total Students</div><div class="value" id="total">–</div></div>
      <div class="card"><div class="label">Present Today</div><div class="value" id="present">–</div></div>
      <div class="card"><div class="label">Absent Today</div><div class="value" id="absent">–</div></div>
      <div class="card"><div class="label">Attendance Rate</div><div class="value" id="rate">–</div></div>
    </div>
    <h2>Student List</h2>
    <table>
      <thead><tr><th>Student ID</th><th>Name</th><th>Class</th><th>Enrolled</th></tr></thead>
      <tbody id="student-table"></tbody>
    </table>
  </div>
  <script>
    async function load() {
      const res = await fetch('/api/students');
      const students = await res.json();
      const tbody = document.getElementById('student-table');
      students.forEach(s => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${s.student_id}</td><td>${s.name}</td><td>${s.class_id}</td><td>${s.enrolled_at || ''}</td>`;
        tbody.appendChild(tr);
      });
      document.getElementById('total').textContent = students.length;
      document.getElementById('present').textContent = '–';
      document.getElementById('absent').textContent  = '–';
      document.getElementById('rate').textContent    = '–';
    }
    load();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/students")
def api_students():
    students = db.get_all_students()
    # Remove raw embedding blobs before serialising
    for s in students:
        s.pop("embedding", None)
    return jsonify(students)


@app.route("/api/attendance/<class_id>")
def api_attendance(class_id):
    records = db.get_class_attendance(class_id)
    return jsonify(records)


@app.route("/api/session/<session_id>")
def api_session(session_id):
    records = db.get_session_attendance(session_id)
    return jsonify(records)


if __name__ == "__main__":
    print("[Dashboard] Running at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
