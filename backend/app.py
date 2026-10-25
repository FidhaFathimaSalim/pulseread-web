"""
app.py — PulseRead web backend (TensorFlow-side process).

Responsibilities:
  1. Serve the static frontend (frontend/index.html, dashboard.html, css, js).
  2. POST /api/analyze      — accept an uploaded video, run it through the
                               existing main.analyze() pipeline (PulseRead +
                               open-rppg subprocess, unchanged), return JSON.
  3. GET  /api/history      — read logs/comparison_log.jsonl (the same log
                               main.py already writes) and return past
                               sessions for the HRV/HR trend charts.
  4. POST /api/generate-report — build a PDF summary of a result with
                               reportlab, for the "Download PDF Report" button.
  5. GET  /media/explain/<f>   — serve the waveform/ROI snapshot PNGs that
                               explain.py writes, so the dashboard can embed
                               them as <img> tags.

IMPORTANT — this process must stay TensorFlow-side, same as main.py's
docstring explains: don't import anything JAX/open-rppg-model related
directly into THIS process. main.analyze() already isolates open-rppg
into its own subprocess, so importing main.py here is safe. The live
webcam path is handled entirely by the separate live_server.py process
(JAX-side) — see that file for why.

Setup (see ../README_WEB.md for the full walkthrough):
    pip install -r requirements_web.txt
    # Copy/symlink your existing project files into this backend/ folder:
    #   main.py, pulseread_pipeline.py, explain.py, recommendations.py,
    #   openrppg_pipeline.py, face_landmarker.task, models/
    python app.py
"""

import os
import sys
import json
import uuid

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'comparison_log.jsonl')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

# main.py (and everything it imports: pulseread_pipeline, explain,
# recommendations) must live in this same backend/ folder — see README_WEB.md.
sys.path.insert(0, BASE_DIR)
try:
    import main as pulseread_main
except ImportError as e:
    pulseread_main = None
    print(f"[app.py] WARNING: could not import main.py ({e}). "
          f"Copy your project's main.py + its dependencies into {BASE_DIR} "
          f"before running the analyze endpoint.")

app = Flask(__name__, static_folder=None)
CORS(app)  # dashboard.html may be opened from a different origin/port during dev

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


# ══════════════════════════════════════════════════════════════
# Static frontend
# ══════════════════════════════════════════════════════════════

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_frontend_file(filename):
    # Serves dashboard.html, css/*, js/* — anything under frontend/
    full_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.isfile(full_path):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({'error': 'not found'}), 404


# ══════════════════════════════════════════════════════════════
# Video upload → analysis
# ══════════════════════════════════════════════════════════════

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if pulseread_main is None:
        return jsonify({'status': 'error', 'error_message': 'Backend pipeline (main.py) is not wired up yet on the server.'}), 500

    if 'video' not in request.files:
        return jsonify({'status': 'error', 'error_message': 'No video file uploaded.'}), 400

    file = request.files['video']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'status': 'error', 'error_message': f'Unsupported file type "{ext}". Use .mp4, .avi, .mov or .mkv.'}), 400

    session_id = uuid.uuid4().hex[:12]
    save_path = os.path.join(UPLOAD_DIR, f'{session_id}{ext}')
    file.save(save_path)

    try:
        result = pulseread_main.analyze(save_path)
    except Exception as e:
        return jsonify({'status': 'error', 'error_message': f'Pipeline error: {e}'}), 500

    # Turn local file paths from explain.py into URLs the frontend can load.
    if result.get('status') == 'ok':
        if result.get('waveform_plot_path'):
            result['waveform_plot_url'] = f"/media/explain/{os.path.basename(result['waveform_plot_path'])}"
        if result.get('roi_snapshot_path'):
            result['roi_snapshot_url'] = f"/media/explain/{os.path.basename(result['roi_snapshot_path'])}"

    return jsonify(result)


@app.route('/media/explain/<path:filename>')
def serve_explain_media(filename):
    explain_dir = os.path.join(BASE_DIR, 'output', 'explain')
    return send_from_directory(explain_dir, filename)


# ══════════════════════════════════════════════════════════════
# Session history (for HRV / HR trend charts)
# ══════════════════════════════════════════════════════════════

@app.route('/api/history')
def api_history():
    if not os.path.exists(LOG_PATH):
        return jsonify([])

    sessions = []
    with open(LOG_PATH, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                displayed = record.get('displayed_to_user', {})
                if displayed.get('status') == 'ok':
                    sessions.append({
                        'timestamp': record.get('timestamp'),
                        'heart_rate_bpm': displayed.get('heart_rate_bpm'),
                        'hrv_rmssd_ms': displayed.get('hrv_rmssd_ms'),
                        'hrv_sdnn_ms': displayed.get('hrv_sdnn_ms'),
                        'signal_quality_pct': displayed.get('signal_quality_pct'),
                        'stress_level': displayed.get('stress_level'),
                    })
            except json.JSONDecodeError:
                continue
    return jsonify(sessions)


# ══════════════════════════════════════════════════════════════
# PDF report
# ══════════════════════════════════════════════════════════════

@app.route('/api/generate-report', methods=['POST'])
def api_generate_report():
    data = request.get_json(force=True) or {}
    from report_builder import build_report_pdf
    session_id = uuid.uuid4().hex[:8]
    out_path = os.path.join(BASE_DIR, 'output', f'pulseread_report_{session_id}.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    build_report_pdf(data, out_path)
    return send_file(out_path, as_attachment=True, download_name='pulseread_report.pdf')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
