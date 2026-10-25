"""
main.py — runs BOTH pipelines every time, but only open-rppg's result is
shown/returned as the final answer. PulseRead still runs in the background
and gets written to a log file — kept as evidence for your dissertation.

ENHANCEMENTS IN THIS VERSION:
  1. Pre-check: rejects too-short/unreadable videos before running either
     pipeline (saves time, avoids confusing downstream errors).
  2. Retry: open-rppg subprocess call retries once on transient failure/timeout.
  3. Graceful HRV messaging: when open-rppg's internal SQI gate blocks HRV,
     the user sees a clear reason instead of a silent null.
  4. BPM reliability label: extends the same honesty principle you already
     built for HRV (Unavailable/Low/Medium) to BPM too.
  5. Personal baseline tracking: compares this result to the user's own
     rolling history (from logs/comparison_log.jsonl), not generic population
     norms — flags "higher/lower than your usual" once enough history exists.

WHY TWO PROCESSES (unchanged):
  PulseRead's CNN needs TensorFlow. open-rppg's FacePhys needs JAX.
  Keras 3 locks its backend per-process, so open-rppg runs as a subprocess.

USAGE:
    python main.py path/to/video.mp4
"""

import sys
import os
import json
import subprocess
import statistics
from datetime import datetime, timezone

import cv2

import pulseread_pipeline as pr
import explain
import recommendations
import graph_data


PULSEREAD_MODEL_PATH = "models/pulseread_best.keras"
MEDIAPIPE_MODEL_PATH = "face_landmarker.task"
OPENRPPG_SCRIPT = os.path.join(os.path.dirname(__file__), "openrppg_pipeline.py")
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "comparison_log.jsonl")

# Update this if you move the project folder — does not auto-detect.
OPENRPPG_PYTHON = r"D:\pulseread-web\venv_openrppg\Scripts\python.exe"

MIN_DURATION_SEC = 15.0
# A video is rejected if a face is found in fewer than this fraction of sampled frames.
MIN_FACE_FRACTION = 0.25
SUBPROCESS_TIMEOUT_SEC = 120
SUBPROCESS_MAX_RETRIES = 2

EXPLAIN_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "explain")

# How much history is needed before we trust a personal baseline enough
# to comment on deviation from it.
BASELINE_MIN_SESSIONS = 3


# ══════════════════════════════════════════════════════════════
# 1. PRE-CHECK — catch obvious problems before running either pipeline
# ══════════════════════════════════════════════════════════════

_face_cascade = None


def _frame_has_face(frame_bgr):
    """True if OpenCV finds a frontal face (fails open if the detector is unavailable)."""
    global _face_cascade
    try:
        if _face_cascade is None:
            _face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if _face_cascade.empty():
            return True
        h, w = frame_bgr.shape[:2]
        scale = 480.0 / w if w > 480 else 1.0
        if scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, (480, int(h * scale)))
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        return len(_face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))) > 0
    except Exception:
        return True


def precheck_video(video_path):
    """
    Returns None if the video is OK to process, or an error dict to
    return immediately (skipping both pipelines) if it's not.
    """
    if not os.path.exists(video_path):
        return {'status': 'error', 'error_message': 'Video file not found.'}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'status': 'error', 'error_message': 'Cannot open video file — check the format/codec.'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Quick brightness check on a handful of sampled frames.
    brightness_samples = []
    for frac in (0.2, 0.5, 0.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * frac))
        ok, frame = cap.read()
        if ok:
            brightness_samples.append(frame.mean())
    # Sample ~12 frames across the video and count how many contain a face.
    face_hits, face_samples = 0, 0
    for i in range(1, 13):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * i / 13))
        ok, frame = cap.read()
        if ok:
            face_samples += 1
            if _frame_has_face(frame):
                face_hits += 1
    cap.release()

    if duration < MIN_DURATION_SEC:
        return {
            'status': 'error',
            'error_message': f'Video too short ({duration:.1f}s). Record at least {MIN_DURATION_SEC:.0f} seconds.',
        }

    if brightness_samples and statistics.mean(brightness_samples) < 40:
        return {
            'status': 'error',
            'error_message': 'Video appears too dark. Record in better lighting, facing a light source.',
        }

    if face_samples and (face_hits / face_samples) < MIN_FACE_FRACTION:
        return {
            'status': 'error',
            'error_message': ('No face detected in the video. Make sure your whole face is '
                              'visible, facing the camera, and well lit.'),
        }

    return None  # OK to proceed


# ══════════════════════════════════════════════════════════════
# open-rppg subprocess call, with retry
# ══════════════════════════════════════════════════════════════

def run_openrppg_subprocess(video_path):
    """Run open-rppg in its own process (JAX backend), retrying once on failure."""
    last_error = None

    for attempt in range(1, SUBPROCESS_MAX_RETRIES + 1):
        try:
            proc = subprocess.run(
                [OPENRPPG_PYTHON, OPENRPPG_SCRIPT, video_path],
                capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC, check=True,
            )
            return json.loads(proc.stdout.strip())
        except subprocess.TimeoutExpired:
            last_error = f'open-rppg timed out after {SUBPROCESS_TIMEOUT_SEC}s (attempt {attempt}).'
        except subprocess.CalledProcessError as e:
            last_error = f'open-rppg subprocess failed (attempt {attempt}): {e.stderr}'
        except json.JSONDecodeError:
            last_error = f'Could not parse open-rppg output (attempt {attempt}).'

    return {'status': 'error', 'error_message': last_error}


def run_pulseread(video_path):
    """Run your original CNN + ensemble pipeline (TensorFlow backend, this process)."""
    model = pr.load_pulseread_model(PULSEREAD_MODEL_PATH)
    options, FaceLandmarker = pr.setup_mediapipe(MEDIAPIPE_MODEL_PATH)
    snapshot_path = os.path.join(EXPLAIN_OUTPUT_DIR, 'roi_snapshot.jpg')
    return pr.analyze_video(video_path, model, options, FaceLandmarker, snapshot_path=snapshot_path)


# ══════════════════════════════════════════════════════════════
# Stress + reliability labeling
# ══════════════════════════════════════════════════════════════

def classify_stress(rmssd_ms, sdnn_ms):
    if rmssd_ms is None or sdnn_ms is None:
        return 'Insufficient data'
    if rmssd_ms < 15 or sdnn_ms < 15:
        return 'High'
    if rmssd_ms < 35 or sdnn_ms < 40:
        return 'Moderate'
    if rmssd_ms < 70 or sdnn_ms < 80:
        return 'Low'
    return 'Very Low'


def classify_bpm_reliability(sqi):
    """
    Extends the same honesty principle already applied to HRV
    (Unavailable/Low/Medium reliability) to the BPM number itself.
    Most rPPG demos show BPM with full confidence regardless of signal
    quality — this makes that distinction visible instead of hidden.
    """
    if sqi is None:
        return 'Unknown'
    if sqi >= 0.75:
        return 'High'
    if sqi >= 0.5:
        return 'Medium'
    return 'Low'


# ══════════════════════════════════════════════════════════════
# 4. Personal baseline tracking
# ══════════════════════════════════════════════════════════════

def load_history():
    """Read past successful sessions from the log. Returns a list of dicts."""
    if not os.path.exists(LOG_PATH):
        return []
    history = []
    with open(LOG_PATH, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                displayed = record.get('displayed_to_user', {})
                if displayed.get('status') == 'ok' and displayed.get('heart_rate_bpm') is not None:
                    history.append(displayed)
            except json.JSONDecodeError:
                continue
    return history


def compute_baseline_note(current_bpm, current_rmssd, history):
    """
    Compares the current session to the user's OWN past sessions, not
    generic population norms — closer to how wearables frame "recovery"
    than a one-size-fits-all reference range.
    """
    if len(history) < BASELINE_MIN_SESSIONS:
        return {
            'baseline_available': False,
            'baseline_note': (f'Baseline not yet available — need {BASELINE_MIN_SESSIONS} '
                              f'past sessions, have {len(history)}.'),
        }

    past_bpms = [h['heart_rate_bpm'] for h in history if h.get('heart_rate_bpm') is not None]
    avg_bpm = statistics.mean(past_bpms)
    bpm_diff_pct = round((current_bpm - avg_bpm) / avg_bpm * 100, 1) if avg_bpm else None

    note = f'Your average over {len(past_bpms)} past sessions is {avg_bpm:.1f} BPM. '
    if bpm_diff_pct is not None:
        direction = 'higher' if bpm_diff_pct > 0 else 'lower'
        note += f'This session is {abs(bpm_diff_pct):.1f}% {direction} than your usual.'

    return {
        'baseline_available': True,
        'baseline_avg_bpm': round(avg_bpm, 1),
        'baseline_bpm_diff_pct': bpm_diff_pct,
        'baseline_note': note,
    }


# ══════════════════════════════════════════════════════════════
# Result shaping
# ══════════════════════════════════════════════════════════════

def build_display_result(openrppg_result):
    if openrppg_result.get('status') != 'ok':
        return {
            'status': 'error',
            'error_message': openrppg_result.get('error_message', 'Analysis failed.'),
        }

    rmssd = openrppg_result.get('rmssd_ms')
    sdnn = openrppg_result.get('sdnn_ms')
    sqi = openrppg_result.get('signal_quality_score')

    # Breathing: prefer our video-respiration estimate (respiration.py) when it is trustworthy,
    # otherwise fall back to open-rppg's own value (only exists when HRV is available).
    # Both are kept in the log so you can compare the two estimators in your dissertation.
    breathing_hz = openrppg_result.get('breathing_rate')
    openrppg_breathing = round(breathing_hz * 60, 1) if breathing_hz is not None else None
    resp = openrppg_result.get('respiration')
    if resp and resp.get('rate_bpm') is not None and resp.get('confidence') in ('High', 'Medium'):
        breathing_per_min, breathing_source = resp['rate_bpm'], 'video respiration'
    else:
        breathing_per_min = openrppg_breathing
        breathing_source = 'open-rppg' if openrppg_breathing is not None else None

    hrv_available = openrppg_result.get('hrv_available', bool(rmssd))
    hrv_note = ''
    if not hrv_available:
        hrv_note = openrppg_result.get(
            'warning',
            f'HRV unavailable — signal quality ({(sqi or 0) * 100:.0f}%) below the reliability threshold.',
        )

    mean_ibi = openrppg_result.get('mean_ibi_ms')

    # HRV from video is only as good as the signal: 50-60% quality passes open-rppg's gate
    # but gives rough numbers (often too high). Say so, so a "Very Low stress" is not over-trusted.
    if not hrv_available:
        hrv_confidence = None
    elif (sqi or 0) >= 0.75:
        hrv_confidence = 'High'
    elif (sqi or 0) >= 0.6:
        hrv_confidence = 'Medium'
    else:
        hrv_confidence = 'Low'

    return {
        'status': 'ok',
        'heart_rate_bpm': openrppg_result.get('final_bpm'),
        'bpm_reliability': classify_bpm_reliability(sqi),
        'hrv_rmssd_ms': round(rmssd, 1) if rmssd is not None else None,
        'hrv_sdnn_ms': round(sdnn, 1) if sdnn is not None else None,
        'mean_ibi_ms': round(mean_ibi, 1) if mean_ibi is not None else None,
        'hrv_available': hrv_available,
        'hrv_confidence': hrv_confidence,
        'hrv_note': hrv_note,
        'signal_quality_pct': round((sqi or 0) * 100, 1),
        'stress_level': classify_stress(rmssd, sdnn),
        'breathing_rate_per_min': breathing_per_min,
        'breathing_source': breathing_source,
        'breathing_rate_openrppg_per_min': openrppg_breathing,
        'resp': resp,
        # Real waveform + beats + IBI + HR for the three dashboard graphs.
        'graph': openrppg_result.get('graph'),
        'graph_source': 'open-rppg' if openrppg_result.get('graph') else None,
    }


def log_comparison(video_path, pulseread_result, openrppg_result, display_result):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'video_path': video_path,
        # keep the log lean: the graph arrays are large and can be rebuilt from the video
        'displayed_to_user': {k: v for k, v in display_result.items() if k != 'graph'},
        'openrppg_raw': openrppg_result,
        'pulseread_raw': pulseread_result,
    }
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')


def analyze(video_path):
    precheck_error = precheck_video(video_path)
    if precheck_error:
        return precheck_error

    history = load_history()  # read BEFORE this session gets appended

    pulseread_result = run_pulseread(video_path)
    openrppg_result = run_openrppg_subprocess(video_path)

    display_result = build_display_result(openrppg_result)

    if display_result.get('status') == 'ok':
        baseline = compute_baseline_note(
            display_result['heart_rate_bpm'], display_result.get('hrv_rmssd_ms'), history,
        )
        display_result.update(baseline)

        # Fallback: if open-rppg gave no waveform, build graphs from PulseRead's filtered signal.
        if not display_result.get('graph') and pulseread_result.get('pulse_signal'):
            display_result['graph'] = graph_data.build_graph(
                pulseread_result['pulse_signal'],
                pulseread_result.get('bpm_processing_fps') or 10.0)
            display_result['graph_source'] = 'pulseread' if display_result['graph'] else None

        os.makedirs(EXPLAIN_OUTPUT_DIR, exist_ok=True)
        artifacts = explain.build_explainability_artifacts(pulseread_result, EXPLAIN_OUTPUT_DIR)
        display_result.update(artifacts)

        # Wellness recommendation — matched from these EXACT computed
        # numbers (BPM + HRV) against 4 predefined physiological states.
        # See recommendations.py — not a diagnosis, general guidance only.
        display_result['recommendation'] = recommendations.get_recommendation(
            display_result.get('heart_rate_bpm'),
            display_result.get('hrv_rmssd_ms'),
            display_result.get('hrv_available', False),
        )

    log_comparison(video_path, pulseread_result, openrppg_result, display_result)
    return display_result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(json.dumps(result, indent=2))