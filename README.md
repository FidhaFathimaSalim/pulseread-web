# PulseRead

<p align="center">
  <img src="PulseRead_logo.png" alt="PulseRead Logo" width="180" />
</p>

<p align="center">
  <strong>Contactless Real-Time Physiological & Vitals Monitoring System via Remote Photoplethysmography (rPPG)</strong>
</p>

---

## 📌 Overview

**PulseRead** is an end-to-end contactless vital signs monitoring web platform leveraging computer vision and deep learning (rPPG). By analyzing subtle facial skin color fluctuations across frames, PulseRead estimates physiological markers including:

- **Heart Rate (BPM)**
- **Heart Rate Variability (HRV - SDNN, RMSSD)**
- **Respiration Rate (Breathe/min)**
- **Autonomic Stress & State Assessment**
- **Explainable Diagnostics & Clinical Recommendation Engine**
- **Comprehensive PDF Health Summary Reports**

PulseRead provides both **Video File Upload Analysis** and **Continuous Real-Time Webcam Streaming**.

---

## 🏗️ Architecture

```
pulseread-web/
├── frontend/
│   ├── index.html          # Landing page with interactive presentation
│   ├── dashboard.html      # Vitals dashboard (Upload + Real-time Webcam streaming)
│   ├── css/styles.css      # Core styles & responsive UI
│   ├── js/dashboard.js     # Analysis pipeline, real-time metrics, interactive charts, PDF export
│   └── js/webcam.js        # WebRTC / getUserMedia frame capture & socket stream
├── backend/
│   ├── app.py              # Primary server (TensorFlow / Flask) for UI and file analysis
│   ├── live_server.py      # Real-time WebSocket server (JAX / Flask-SocketIO) for live webcam
│   ├── pulseread_pipeline.py # Deep learning rPPG feature extraction & inference
│   ├── report_builder.py   # PDF diagnostic report generator (ReportLab)
│   ├── face_tracker.py     # Facial landmark detection and ROI tracking
│   ├── recommendations.py  # Health & clinical recommendation logic
│   └── requirements_web.txt # Python dependencies
```

### Dual-Engine Backend Design

Keras 3 backend configuration locks per Python process (TensorFlow vs. JAX). To combine optimal performance for both pipelines:

| Service | Backend | Port | Responsibility |
|---|---|---|---|
| `app.py` | TensorFlow | `5000` | Web application hosting, static assets, asynchronous file upload processing |
| `live_server.py` | JAX | `5001` | High-frequency continuous webcam frame ingestion & sliding-window inference via Socket.IO |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Virtual environments for TensorFlow and JAX runtimes (`venv_pulseread` and `venv_openrppg`)

### 1. Environment Setup

```bash
# Setup dependencies for upload analysis server
python -m venv venv_pulseread
venv_pulseread\Scripts\activate
pip install -r backend/requirements_web.txt

# Setup dependencies for real-time live webcam server
python -m venv venv_openrppg
venv_openrppg\Scripts\activate
pip install -r backend/requirements_web.txt
```

### 2. Running the Servers

**Terminal 1 — Main Web & File Analysis Server (Port 5000):**
```bash
venv_pulseread\Scripts\activate
cd backend
python app.py
```

**Terminal 2 — Real-Time Live Streaming Server (Port 5001):**
```bash
venv_openrppg\Scripts\activate
cd backend
python live_server.py
```

Navigate to **`http://localhost:5000`** in any modern web browser.

---

## 🔬 How Real-Time Webcam Mode Works

1. **Client Frame Stream:** Browser captures camera frames via `navigator.mediaDevices.getUserMedia` and streams downsampled frames over WebSocket to `live_server.py`.
2. **Buffer Management:** `live_server.py` maintains a rolling temporal window per connection.
3. **Sliding Window Inference:** Computes rapid BPM updates once initial frames are buffered, followed by deep HRV / stress parameter passes every refresh cycle.
4. **Instant Diagnostic Telemetry:** Metrics update directly on the dashboard's live pulse chart with real-time feedback.

---

## 📄 License & Attribution

Developed as an advanced health-tech research & physiological computing project.
