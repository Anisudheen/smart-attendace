# AI-Powered Smart Attendance System

Real-time student attendance via face recognition, with LLM-generated reports.

---

## Project Structure

```
smart_attendance/
├── main.py               # Entry point — run a live attendance session
├── enroll.py             # CLI to register new students
├── dashboard.py          # Flask web dashboard
├── requirements.txt
└── attendance/
    ├── camera.py         # Video capture (OpenCV)
    ├── detector.py       # Face detection (Haar / DNN)
    ├── recognizer.py     # Face recognition (DeepFace / TensorFlow)
    ├── database.py       # SQLite storage
    ├── session.py        # Session orchestrator
    ├── reports.py        # CSV, Excel, LLM report generation
    └── alerts.py         # Email / SMS notifications
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The recognizer uses `deepface` and `tf-keras` for embeddings.
> If you install from scratch, the first model load may download weights.

### 2. Enroll students

Each successful enrollment is saved to one shared roster file at `data/enrolled_students.csv`.

**From webcam (interactive):**
```bash
python enroll.py --id S001 --name "Alice Kumar" --class_id CS101
```

**From existing photos:**
```bash
python enroll.py --id S002 --name "Bob Sharma" --class_id CS101 \
                 --images photos/bob1.jpg photos/bob2.jpg
```

### 3. Run a session

```bash
# Webcam, show live feed
python main.py --class_id CS101 --camera 0 --show

# IP camera
python main.py --class_id CS101 --camera rtsp://192.168.1.10/stream --show

# Adjust confidence threshold (default 0.6)
python main.py --class_id CS101 --threshold 0.55 --show
```

Press **Q** to end the session. A CSV, Excel, and (optionally) an AI summary are saved to `reports/`.

### 4. View dashboard

```bash
python dashboard.py
# Open http://localhost:5000
```

### 5. Run in Streamlit

```bash
streamlit run app.py
```

This opens a browser-based live attendance UI that uses the same detector, recognizer, and SQLite database.

---

## LLM Reports (optional)

Set your Anthropic API key to enable AI-generated summaries:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The system will automatically call Claude at the end of each session and produce a JSON summary with attendance rate, concerns, and recommendations.

---

## Email Alerts (optional)

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=you@gmail.com
export SMTP_PASS=your_app_password
export ALERT_FROM=you@gmail.com
export ALERT_TO=instructor@school.edu
```

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | `0.6` | Max face distance for a match (lower = stricter) |
| `--camera` | `0` | Webcam index or RTSP URL |
| `--show` | off | Display annotated live feed |
| `--count` (enroll) | `10` | Number of webcam captures per student |

---

## Technology Stack

| Layer | Library |
|-------|---------|
| Video capture | OpenCV `VideoCapture` |
| Face detection | OpenCV Haar / DNN SSD |
| Face recognition | `face_recognition` (dlib 128-D) |
| Deep learning | TensorFlow / PyTorch |
| Storage | SQLite via `sqlite3` |
| Data export | Pandas + openpyxl |
| LLM reports | Anthropic Claude API |
| Dashboard | Flask |
| Alerts | SMTP + Twilio |
