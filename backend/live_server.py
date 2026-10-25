"""
live_server.py — PulseRead live webcam backend (JAX-side process)  —  v2

Run this in the open-rppg venv (venv_openrppg):
    pip install -r requirements_web.txt
    pip install simple-websocket
    python live_server.py

WHAT IS NEW IN v2 (three features, all optional from the browser):
  1. RESPIRATION   every full analysis window also returns a breathing rate
                   (respiration.py: FM + AM + baseline-wander fusion).
  2. CHALLENGE     baseline -> challenge -> recovery experiment (challenge.py).
                   Every finished window is stamped with its phase; a report is
                   emitted when the experiment ends.
  3. MULTI-PERSON  one webcam, up to MAX_PEOPLE faces. Each face is tracked
                   (face_tracker.py), cropped, buffered and analysed on its own.

HOW "LIVE" WORKS (unchanged idea):
  The browser sends JPEG frames over Socket.IO. For each person ("subject") we keep a
  rolling buffer of the last ~34 s of frames. A background worker re-runs
  model.process_video_tensor() on the newest window every few seconds and emits the
  latest HR / HRV / stress / breathing estimate. The model runs in a small pool
  of independent model copies (MAX_PARALLEL_RUNS, multi mode only), so several people are
  analysed at the same time; the person whose result is oldest goes first.

WINDOW LENGTH:  HR / HRV / graphs use the same 20 s windows as before. Breathing needs
  >= 25 s, so every ~25 s one 30 s window is analysed instead; its breathing value is kept
  until the next one. (Longer HRV windows, e.g. 30 s, would be steadier but slower: change
  HRV_MIN_BUFFER_SEC if you want that.)

Socket.IO events (browser -> server):
  frame {image, ts}        one JPEG frame (ts = browser clock, seconds)
  config {mode}            'single' (default) or 'multi'
  select_person {id}       which person is highlighted (multi mode)
  challenge_start {protocol, ts} / challenge_phase {name, ts} / challenge_end {ts} / challenge_cancel
Server -> browser:
  server_info, live_update (selected person), people_update (multi: boxes + short summary),
  person_update (multi: one FULL result incl. graphs per person, every second), challenge_started,
  challenge_sample, challenge_report, challenge_error, challenge_cancelled
"""

import os
import sys
import math
import time
import base64
import threading
import itertools
from collections import deque

import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request
from flask_socketio import SocketIO

import openrppg_pipeline as orp  # sets KERAS_BACKEND=jax at import time
import recommendations           # pure python, safe in either backend process
import graph_data                # numpy only
import respiration               # numpy only
import challenge as challenge_mod
import face_tracker as ft

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    async_mode='threading',
    max_http_buffer_size=10_000_000,
)

# ---------------- Tuning ----------------
QUICK_MIN_WINDOW_SEC = 20.0   # buffered seconds needed before the FIRST BPM estimate. Was 10.0: short 10 s windows
                              # gave outliers (e.g. 61 / 72 BPM when the real value was ~83) and cost a full model run
                              # every 2 s. Set back to 10.0 for a faster but noisier first reading.
QUICK_REFRESH_SEC = 2.0       # how often BPM is recomputed before HRV is eligible
HRV_MIN_BUFFER_SEC = 20.0     # HR + HRV window — SAME as v1, so tiles and graphs behave as before
HRV_REFRESH_SEC = 5.0         # how often that pass re-runs
RESP_WINDOW_SEC = 30.0        # breathing needs a longer look: every RESP_REFRESH_SEC one 30 s window is run
RESP_REFRESH_SEC = 25.0       #   (it also gives HR/HRV/graphs like any other run); in between, the last
RESP_KEEP_SEC = 120.0         #   breathing value is kept for up to RESP_KEEP_SEC
MAX_MISSING_SEC = 10.0        # multi mode: keep a person (same number, same buffer) this long after the detector loses the face
MAX_FEED_GAP_SEC = 4.0        # ... but stop feeding guessed crops after this long; a longer gap restarts that person's buffer
MAX_BUFFER_SEC = 34.0         # frames older than this are dropped (must be >= RESP_WINDOW_SEC)
MIN_EFFECTIVE_FPS = 12.0      # below this the signal is too sparse to trust
TARGET_FPS = 15               # every model call gets frames evenly resampled to this rate

FRAMES_ARE_RGB = True         # flip to False if BPM looks wrong (see old notes)
SAVE_DEBUG_CLIP = True        # single mode: writes debug_window.mp4; multi mode: debug_personN_best.mp4 and debug_personN_worst.mp4

FACE_DETECTOR = 'auto'        # 'auto' = MediaPipe if it works, else OpenCV Haar | 'mediapipe' | 'haar'
REQUIRE_FACE = True           # single mode face gate
FACE_HIST_LEN = 6
NO_FACE_MAX_HITS = 1
SINGLE_MULTI_FACE_HITS = 3    # single mode measures ONE person: if 2+ faces are seen in this many of the last FACE_HIST_LEN
                              #   checks (one check per second), recording pauses and the buffer is cleared

# ---- multi-person ----
MAX_PEOPLE = 4
CROP_SIZE = 224               # each person's face is cut into a CROP_SIZE x CROP_SIZE clip. 224 made every model run ~1.6x heavier
                              #   for no measurable gain once 3-4 people share the CPU (the model works on small faces anyway)
WARM_MULTI_SHAPE = True       # also pre-compile the model for the crop size at start-up
MULTI_FRAME_WIDTH = 1280       # the browser sends this width in multi mode. 1280 px JPEGs (q 0.95) were ~7 MB/s: the browser, the socket and
                              #   the JPEG decoder could not keep 20 fps. That is only a problem with MANY people; with ONE person in view the
                              #   browser is told to send FULL_FEED_WIDTH instead (see 'want_width'). With 2+ people we need every pixel
                              #   we can get: a face 100 px wide (960 px frame, 4 people) gives the model ~9x fewer skin pixels than the
                              #   300 px face of single mode, which is why multi-person quality stayed at 10-35 % while single mode reached 54 %.
                              #   (faces are DETECTED on a smaller copy, the crops are cut from this frame)
FULL_FEED_WIDTH = 640         # ONE person in view in multi mode: feed the whole frame at this width, exactly like single mode does
                              #   (single mode reaches ~54 % with it). With 2+ people each face gets its own crop.
SINGLE_BACK_SEC = 8.0         # after a 2nd person leaves, wait this long before going back to the whole-frame feed (no flip-flopping)
DETECT_EVERY = 3              # multi mode: run the face detector on every 3rd frame (the smoothed boxes carry on in between)
MAX_PARALLEL_RUNS = 2         # multi mode: how many people are analysed AT THE SAME TIME (each uses its own model copy, ~4 MB).
                              #   2 is a safe default for a 4-6 core laptop; try 3 if your CPU is idle during runs, 1 = old behaviour
MULTI_REFRESH_SEC = 8.0       # multi mode: minimum seconds between two runs of the same person (single mode keeps HRV_REFRESH_SEC)
MULTI_RESP_REFRESH_SEC = 40.0 # multi mode: the heavy 30 s breathing window is repeated this rarely
CROP_CONTEXT = 2.2            # crop side = this x face size. In single mode (which reaches 54-70 %) the face fills ~40 % of the picture and the model
                              #   finds it itself; a 1.7 crop made the face fill ~60 %, an input the model never saw in single mode. 2.2 -> ~45 %.
                              #   A/B test: 1.7 = tight, 2.2 = like single mode. Keep CROP_SIZE >= 224 so the face still has ~100 px.
MIN_VISIBLE = 0.88            # a face with more than 12 % of its box outside the camera picture is not fed (a cut-off face ruins the pulse)
CROP_RELOCK_SHIFT = 0.20      # LOCKED CROP: a person's crop window stays exactly where it is until the face has moved by more than this
                              #   many face-widths (or grown/shrunk by CROP_RELOCK_SCALE). A crop that follows every tiny box change makes the
                              #   skin pixels slide and zoom by a few pixels per frame - far bigger than the 0.1 % colour change of the pulse.
CROP_RELOCK_SCALE = 1.20
WEAK_QUALITY_PCT = 50.0       # a person below this is re-checked first and sooner, using the longer 30 s window
WEAK_REFRESH_SEC = 4.0
MAX_BLEND_GAP = 0.35          # resampling: blend two neighbouring frames, but never across a hole longer than this many seconds
JUMP_RESET = 0.9              # multi mode: if a person's face box jumps by more than this many face-widths between two frames,
                              #   the tracker probably swapped two people: restart that person's buffer instead of mixing two pulses

_model_ready = threading.Event()                     # slot 0 is compiled: single mode and the first person can start
_slot_lock = threading.Lock()
_model_locks = [threading.Lock() for _ in range(MAX_PARALLEL_RUNS)]   # one lock per model copy (a copy is not thread-safe)
_slot_ready = [False] * MAX_PARALLEL_RUNS            # model copy is loaded AND compiled
_slot_busy = [False] * MAX_PARALLEL_RUNS
_n_running = [0]                                      # how many model runs are executing right now (for the log)


def _acquire_slot(limit):
    """Reserve a free, warmed-up model copy among the first `limit` ones. None if all are busy."""
    with _slot_lock:
        for i in range(min(limit, MAX_PARALLEL_RUNS)):
            if _slot_ready[i] and not _slot_busy[i]:
                _slot_busy[i] = True
                return i
    return None


def _release_slot(slot):
    with _slot_lock:
        _slot_busy[slot] = False

sessions = {}
_uid_counter = itertools.count(1)   # every Subject gets a unique uid, so the browser can tell a NEW person from a reused 'Person N' number


# ---------------- Helpers ----------------

def _num(x, nd=1):
    try:
        if x is None:
            return None
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, nd)
    except (TypeError, ValueError):
        return None


def clean(obj):
    """Recursively make a payload JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, 'item'):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def classify_bpm_reliability(sqi):
    if sqi is None:
        return 'Unknown'
    if sqi >= 0.75:
        return 'High'
    if sqi >= 0.5:
        return 'Medium'
    return 'Low'


def decode_frame(data_url):
    """'data:image/jpeg;base64,...' -> uint8 ndarray (H,W,3), or None on failure."""
    try:
        _, b64data = data_url.split(',', 1)
        raw = base64.b64decode(b64data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if frame is None:
            return None
        if FRAMES_ARE_RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        print('decode_frame failed:', e)
        return None


_gate_detector = None
_face_check_broken = False


def count_faces(frame):
    """Single-mode face gate: how many DISTINCT faces are in the frame (duplicate / nested boxes merged).
    Fails open (returns 1) if no detector works here."""
    global _gate_detector, _face_check_broken
    if _face_check_broken:
        return 1
    try:
        if _gate_detector is None:
            _gate_detector = ft.auto_detector_factory(FACE_DETECTOR)
        return len(ft._nms(list(_gate_detector(frame))))
    except Exception as e:
        _face_check_broken = True
        print('[face check DISABLED] no face detector works in this Python environment:', e)
        return 1


def has_face(frame):
    """True if at least one face is found."""
    return count_faces(frame) > 0


# ---------------- Buffers / subjects / sessions ----------------

class FrameBuffer:
    """Rolling buffer of (frame, timestamp) with an evenly-resampled window() getter."""

    def __init__(self):
        self.buffer = deque()
        self.lock = threading.Lock()
        self.frames_received = 0
        self.max_gap = 0.0          # longest hole between two arriving frames since the last [perf] line

    def add(self, frame, ts):
        with self.lock:
            if self.buffer:
                self.max_gap = max(self.max_gap, ts - self.buffer[-1][1])
            self.buffer.append((frame, ts))
            self.frames_received += 1
            cutoff = ts - MAX_BUFFER_SEC
            while self.buffer and self.buffer[0][1] < cutoff:
                self.buffer.popleft()

    def latest(self):
        with self.lock:
            return self.buffer[-1][0] if self.buffer else None

    def buffered_seconds(self):
        with self.lock:
            if len(self.buffer) < 2:
                return 0.0
            return self.buffer[-1][1] - self.buffer[0][1]

    def window(self, seconds, interpolate=False):
        """
        (frames, raw_fps, t_start, t_end): exactly seconds*TARGET_FPS frames evenly spaced
        over the last `seconds`. t_start / t_end are on the browser clock (used by challenge mode).
        (None, None, None, None) if there is not enough history yet.
        """
        with self.lock:
            if len(self.buffer) < 10:
                return None, None, None, None
            items = list(self.buffer)
        ts = np.array([t for _, t in items])
        t_end = ts[-1]
        t_start = t_end - seconds
        if ts[0] > t_start + 0.5:
            return None, None, None, None
        in_win = ts >= t_start
        n_in = int(in_win.sum())
        span = ts[-1] - ts[in_win][0]
        if n_in < 10 or span <= 0:
            return None, None, None, None
        raw_fps = (n_in - 1) / span
        targets = np.linspace(t_start, t_end, int(round(seconds * TARGET_FPS)))
        if not interpolate:
            idx = np.abs(ts[None, :] - targets[:, None]).argmin(axis=1)
            frames = np.stack([items[i][0] for i in idx], axis=0)
            return frames, raw_fps, float(t_start), float(t_end)
        # Frames arrive unevenly (the browser drops frames when busy, the server stalls for a moment while the model runs).
        # Picking the nearest frame then DUPLICATES some frames and skips others, which puts steps into the pulse signal.
        # Instead every output frame is a time-weighted blend of the two real frames on either side of its exact time.
        j = np.clip(np.searchsorted(ts, targets), 1, len(ts) - 1)
        i0, i1 = j - 1, j
        gap = ts[i1] - ts[i0]
        w = np.clip((targets - ts[i0]) / np.maximum(gap, 1e-6), 0.0, 1.0)
        w = np.where(gap > MAX_BLEND_GAP, (w >= 0.5).astype(float), w)     # across a long hole do not invent a cross-fade
        out = np.empty((len(targets),) + items[0][0].shape, dtype=np.uint8)
        for k in range(len(targets)):
            a = items[i0[k]][0]
            if w[k] <= 0.0:
                out[k] = a
            elif w[k] >= 1.0:
                out[k] = items[i1[k]][0]
            else:
                out[k] = (a.astype(np.float32) * (1.0 - w[k]) + items[i1[k]][0].astype(np.float32) * w[k] + 0.5).astype(np.uint8)
        return out, raw_fps, float(t_start), float(t_end)


class Subject:
    """One measured person (single mode: always id 1, whole frame; multi mode: one tracked face)."""

    def __init__(self, sid, label):
        self.id = sid
        self.uid = next(_uid_counter)
        self.label = label
        self.buf = FrameBuffer()
        self.last_run = 0.0
        self.last_result = {}
        self.run_counter = 0
        self.emitted_run = None
        self.emitted_pp_run = None  # same idea, for the per-person dashboards (multi mode)
        self.last_long_run = 0.0   # when the last 30 s (breathing) window finished
        self.last_resp = None      # (resp dict, breathing_rate, breathing_source, finished_at)
        self.box = None            # normalised face box (multi mode)
        self.last_seen = None      # last time the tracker really saw this face
        self.last_fed = None       # last time a frame was added to this person's buffer
        self.crop_lock = None      # (cx, cy, size) of the frozen crop window
        self.best_q = None         # debug clips: quality of the best / worst run saved so far
        self.worst_q = None
        self.last_center = None    # last crop centre in frame pixels + face size (identity-swap guard)
        self.perf_n = 0            # frames_received at the last [perf] line


class LiveSession:
    def __init__(self, sid):
        self.sid = sid
        self.mode = 'single'
        self.lock = threading.RLock()
        self.frame_lock = threading.Lock()     # serialises tracker updates (multi mode)
        self.stop_event = threading.Event()
        self.subjects = {1: Subject(1, 'You')}
        self.selected_id = 1
        self.running = {}                      # subject id -> model slot, for every analysis that is running right now
        self.avg_run_sec = 0.0                 # moving average of how long one model run takes (sets the refresh interval)
        self.perf_t = time.time()
        self.feed_kind = 'full'                # multi mode: 'full' = one person, whole frame | 'crop' = 2+ people, one crop each
        self.single_since = None
        self.tracker = None
        self.frame_wh = (0, 0)
        self.face_hist = deque(maxlen=FACE_HIST_LEN)
        self.last_people_emit = 0.0            # throttles the fast people_update used for the face boxes
        self.challenge = None                  # challenge.ChallengeRecorder or None
        self.challenge_ending = False

    @property
    def inferring(self):
        return bool(self.running)

    def set_mode(self, mode):
        with self.lock:
            if mode == self.mode:
                return
            self.mode = mode
            if mode == 'multi':
                self.subjects = {}
                self.selected_id = 1
                self.tracker = ft.FaceTracker(max_people=MAX_PEOPLE, detect_every=DETECT_EVERY, max_missing_sec=MAX_MISSING_SEC,
                                          detector_kind=FACE_DETECTOR)
            else:
                self.tracker = None
                self.subjects = {1: Subject(1, 'You')}
                self.selected_id = 1
            self.face_hist.clear()

    def primary(self):
        with self.lock:
            if not self.subjects:
                return None
            if self.selected_id not in self.subjects:
                self.selected_id = min(self.subjects)
            return self.subjects[self.selected_id]

    def labels(self):
        with self.lock:
            return {i: s.label for i, s in self.subjects.items()}


# ---------------- Model work ----------------

def analyze_window(frames, eff_fps, want_hrv, buffered, slot=0):
    """Run the model (copy number `slot`) on one window and return the result fields (JSON-safe)."""
    window_sec = frames.shape[0] / eff_fps
    with _model_locks[slot]:
        t0 = time.time()
        _n_running[0] += 1
        model = orp.get_model(slot)
        raw = model.process_video_tensor(frames, fps=eff_fps)
        bvp, bfps = orp._bvp_and_fps(model, window_sec=window_sec)   # grab while we hold the model
        _n_running[0] -= 1
    print(f'[model] slot {slot}: inference took {time.time() - t0:.1f}s ({window_sec:.0f}s window, {_n_running[0] + 1} run(s) overlapped at the end)')

    graph = graph_data.build_graph(bvp, bfps) if bvp is not None else None
    resp = respiration.estimate_respiration(bvp, bfps) if bvp is not None else None

    hr = _num(raw.get('hr'))
    sqi = _num(raw.get('SQI'), 3)
    hrv = raw.get('hrv') or {}

    rmssd = _num(hrv.get('rmssd'))
    sdnn = _num(hrv.get('sdnn'))
    ibi = _num(hrv.get('ibi'))
    breathing_hz = _num(hrv.get('breathingrate'), 4)

    hrv_available = bool(hrv) and want_hrv and rmssd is not None
    if not want_hrv:
        # a short (quick) window is only trusted for heart rate; never show its HRV numbers
        rmssd = sdnn = ibi = None
    if not hrv_available:
        hrv_confidence = None
    elif (sqi or 0) >= 0.75:
        hrv_confidence = 'High'
    elif (sqi or 0) >= 0.6:
        hrv_confidence = 'Medium'
    else:
        hrv_confidence = 'Low'     # passes the model's 50% gate but the numbers are rough
    print(f"[model] window={frames.shape} eff_fps={eff_fps:.1f} hr={hr} sqi={sqi} "
          f"rmssd={rmssd} sdnn={sdnn} resp={resp and resp.get('rate_bpm')}({resp and resp.get('confidence')})")

    # Breathing rate shown to the user: our video-respiration estimate when it is trustworthy,
    # otherwise open-rppg's own value (which only exists when HRV is available).
    hrv_breath = round(breathing_hz * 60, 1) if breathing_hz else None
    if resp and resp.get('rate_bpm') is not None and resp.get('confidence') in ('High', 'Medium'):
        breathing, breathing_source = resp['rate_bpm'], 'video respiration'
    elif hrv_breath is not None:
        breathing, breathing_source = hrv_breath, 'open-rppg'
    else:
        breathing, breathing_source = None, None

    result = {
        'heart_rate_bpm': hr,
        'bpm_reliability': classify_bpm_reliability(sqi),
        'signal_quality_pct': round((sqi or 0) * 100, 1),
        'hrv_available': hrv_available,
        'hrv_confidence': hrv_confidence,
        'full_window': bool(want_hrv),          # False = short window: heart rate only
        'hrv_rmssd_ms': rmssd,
        'hrv_sdnn_ms': sdnn,
        'mean_ibi_ms': ibi,
        'stress_level': orp.classify_stress(rmssd, sdnn) if hrv_available else 'Insufficient data',
        'breathing_rate_per_min': breathing,
        'breathing_source': breathing_source,
        'resp': resp,
        'graph': graph,
    }

    if hr is None:
        result['hrv_note'] = ('No pulse detected yet — keep your whole face in frame, '
                              'well lit from the front, and stay still.')
    elif not hrv_available:
        result['hrv_note'] = (f'Building up signal — {buffered:.0f}s / {HRV_MIN_BUFFER_SEC:.0f}s buffered.'
                              if not want_hrv else _hrv_unavailable_note(result['signal_quality_pct']))
    else:
        result['hrv_note'] = ''

    if hr is not None:
        result['recommendation'] = recommendations.get_recommendation(hr, rmssd, hrv_available)
    return result


def save_debug_clip(frames, name='debug_window.mp4'):
    try:
        h, w = frames.shape[1:3]
        out = cv2.VideoWriter(os.path.join(BASE_DIR, name),
                              cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if FRAMES_ARE_RGB else f)
        out.release()
    except Exception as e:
        print('debug clip failed:', e)


def _merge_breathing(subject, result, window_sec):
    """Breathing needs the long window. Short runs reuse the latest long-run breathing value."""
    now = time.time()
    if window_sec >= RESP_WINDOW_SEC - 0.1:
        subject.last_long_run = now
        subject.last_resp = (result.get('resp'), result.get('breathing_rate_per_min'),
                             result.get('breathing_source'), now)
        return
    last = subject.last_resp
    if last and now - last[3] <= RESP_KEEP_SEC and last[0] is not None:
        result['resp'], result['breathing_rate_per_min'], result['breathing_source'] = last[0], last[1], last[2]


def _infer_job(session, subject, window_sec, want_hrv, buffered, slot=0):
    """Runs in its own thread so the 1-second status updates never freeze."""
    try:
        if session.stop_event.is_set():
            return
        frames, raw_fps, t0, t1 = subject.buf.window(window_sec, interpolate=(session.mode == 'multi'))
        if frames is None:
            return
        if (SAVE_DEBUG_CLIP and session.mode == 'single'
                and frames.shape[0] >= int(HRV_MIN_BUFFER_SEC * TARGET_FPS)):
            save_debug_clip(frames)
        if raw_fps < MIN_EFFECTIVE_FPS:
            subject.last_result = {**subject.last_result,
                                   'hrv_note': f'Frame rate too low ({raw_fps:.1f} fps). Close heavy tabs/apps.'}
            print(f'[warn] arrival fps too low: {raw_fps:.1f}')
            return
        print(f'[model] {subject.label}: arrival_fps={raw_fps:.1f}, resampled to {TARGET_FPS} fps, '
              f'frames={frames.shape[0]}')
        t_run = time.time()
        result = analyze_window(frames, float(TARGET_FPS), want_hrv, buffered, slot)
        took = time.time() - t_run
        session.avg_run_sec = took if session.avg_run_sec <= 0 else 0.6 * session.avg_run_sec + 0.4 * took
        own_resp = result.get('resp') if window_sec >= RESP_WINDOW_SEC - 0.1 else None   # this window's OWN breathing (long runs only)
        if SAVE_DEBUG_CLIP and session.mode == 'multi' and result.get('heart_rate_bpm') is not None:
            # keep the exact frames the model saw for this person's BEST and WORST run, so you can look at them and run
            # check_signal.py on them: debug_person1_best.mp4 / debug_person1_worst.mp4 (overwritten when beaten)
            q = result.get('signal_quality_pct') or 0.0
            if subject.best_q is None or q > subject.best_q:
                subject.best_q = q
                save_debug_clip(frames, f'debug_person{subject.id}_best.mp4')
            if subject.worst_q is None or q < subject.worst_q:
                subject.worst_q = q
                save_debug_clip(frames, f'debug_person{subject.id}_worst.mp4')
        _merge_breathing(subject, result, window_sec)
        if result.get('heart_rate_bpm') is not None:
            subject.run_counter += 1
            result['run_id'] = subject.run_counter
            subject.last_result = result
            ch = session.challenge
            if ch is not None and not ch.cancelled:
                # the challenge must not get a breathing value carried over from an earlier window (it may belong to another phase)
                sample = ch.add_sample(subject.id, t0, t1, {**result, 'resp': own_resp})
                socketio.emit('challenge_sample', clean({'id': subject.id, **sample}), to=session.sid)
        else:
            subject.last_result = {**subject.last_result, 'hrv_note': result['hrv_note']}
    except Exception as e:
        print('[model error]', repr(e))
        subject.last_result = {**subject.last_result, 'hrv_note': f'model error: {e}'}
    finally:
        subject.last_run = time.time()
        with session.lock:
            session.running.pop(subject.id, None)
        _release_slot(slot)


def _pick_subject(session):
    """The subject whose result is oldest and due. Returns (subject, window_sec, want_hrv, buffered) or None.
    People that are being analysed right now are skipped, so several can run side by side."""
    now = time.time()
    best = None
    multi = session.mode == 'multi'
    with session.lock:
        subjects = list(session.subjects.values())
        busy = set(session.running)
    ch = session.challenge
    ready = [s for s in subjects if s.buf.buffered_seconds() >= QUICK_MIN_WINDOW_SEC]
    # multi mode: never ask for more work than the CPU can finish. If one run takes 8 s, 4 people and 2 copies, a person
    # can only be refreshed about every 8 * 4 / 2 * 1.15 = 18 s; asking sooner just starves the frame intake (signal quality collapses).
    load_floor = (session.avg_run_sec * len(ready) / max(1, MAX_PARALLEL_RUNS) * 1.15) if multi else 0.0
    for s in ready:
        if s.id in busy:
            continue
        buffered = s.buf.buffered_seconds()
        want_hrv = buffered >= HRV_MIN_BUFFER_SEC
        if ch is not None and not ch.cancelled and not want_hrv:
            continue                                  # during a challenge only full windows are used
        base = (MULTI_REFRESH_SEC if multi else HRV_REFRESH_SEC) if want_hrv else QUICK_REFRESH_SEC
        q = s.last_result.get('signal_quality_pct')
        weak = multi and q is not None and s.last_result.get('heart_rate_bpm') is not None and q < WEAK_QUALITY_PCT
        if s.last_result.get('heart_rate_bpm') is None:
            interval = QUICK_REFRESH_SEC
        elif weak:
            interval = max(WEAK_REFRESH_SEC, load_floor * 0.6)      # a weak person is checked again soon ...
        else:
            interval = max(base, load_floor)
        if now - s.last_run < interval:
            continue
        resp_every = max(RESP_REFRESH_SEC, MULTI_RESP_REFRESH_SEC if multi else 0, load_floor * 2)
        if want_hrv and buffered >= RESP_WINDOW_SEC + 0.5 and now - s.last_long_run >= resp_every:
            window_sec = RESP_WINDOW_SEC               # breathing run
        else:
            window_sec = HRV_MIN_BUFFER_SEC if want_hrv else QUICK_MIN_WINDOW_SEC
        key = (0 if weak else 1, s.last_run)           # ... and goes first
        if best is None or key < best[4]:
            best = (s, window_sec, want_hrv, buffered, key)
    return best[:4] if best else None


# ---------------- Payload building ----------------

def _hrv_unavailable_note(quality_pct):
    """Only blame signal quality when it really is below the model's 50% gate."""
    if quality_pct is not None and quality_pct < 50:
        return (f'HRV unavailable — signal quality ({quality_pct:.0f}%) is below the 50% the model needs. '
                'Improve lighting and hold still.')
    return 'HRV could not be computed from this window. Hold still, face the light and wait for the next update.'


def build_payload(session, subject, marker='emitted_run', with_graph=True):
    """
    The full result of one person. The big graph arrays are sent once per model run per `marker`
    (so the single-person dashboard and the per-person dashboards never steal each other's graphs).
    with_graph=False never sends graph data (used for the shared live_update in multi mode).
    """
    buffered = subject.buf.buffered_seconds()
    payload = dict(subject.last_result)
    rid = payload.get('run_id')
    if rid is not None:
        if not with_graph or rid == getattr(subject, marker):
            payload.pop('graph', None)       # send the big graph arrays once per run only
        else:
            setattr(subject, marker, rid)
    payload['buffer_sec'] = round(buffered, 1)
    payload['hrv_min_buffer_sec'] = HRV_MIN_BUFFER_SEC
    payload['person_id'] = subject.id
    payload['person_uid'] = subject.uid
    payload['label'] = subject.label
    this_one_running = subject.id in session.running
    if buffered < QUICK_MIN_WINDOW_SEC:
        payload['hrv_note'] = (f'Building up signal — {buffered:.0f}s / '
                               f'{QUICK_MIN_WINDOW_SEC:.0f}s needed for first estimate.')
    elif not _model_ready.is_set():
        payload['hrv_note'] = 'Model warming up (first start only, up to a minute)…'
    elif this_one_running and 'heart_rate_bpm' not in payload:
        payload['hrv_note'] = 'Analysing… the first run can take a while (model warm-up).'
    elif payload.get('heart_rate_bpm') is not None and not payload.get('hrv_available'):
        if buffered < HRV_MIN_BUFFER_SEC:
            payload['hrv_note'] = (f'Building up signal — {buffered:.0f}s / '
                                   f'{HRV_MIN_BUFFER_SEC:.0f}s buffered.')
        elif not payload.get('full_window'):
            # the stored result came from a short window; the full 30 s analysis is just not done yet
            payload['hrv_note'] = (f'Heart rate shown. HRV and stress appear after the next full '
                                   f'{HRV_MIN_BUFFER_SEC:.0f} s analysis…')
        else:
            payload['hrv_note'] = _hrv_unavailable_note(payload.get('signal_quality_pct', 0))
    return payload


def people_payload(session):
    people = []
    with session.lock:
        subjects = list(session.subjects.values())
        selected = session.selected_id
    for s in subjects:
        r = s.last_result
        buffered = s.buf.buffered_seconds()
        if s.id in session.running:
            state = 'analysing'                 # the model is working on THIS person right now
        elif r.get('heart_rate_bpm') is not None:
            state = 'ready'
        elif buffered < QUICK_MIN_WINDOW_SEC:
            state = 'collecting'
        else:
            state = 'waiting'                   # enough video, waiting for a free model copy
        people.append({
            'id': s.id, 'uid': s.uid, 'label': s.label, 'box': s.box, 'selected': s.id == selected, 'state': state,
            'buffer_sec': round(buffered, 1),
            'heart_rate_bpm': r.get('heart_rate_bpm'),
            'bpm_reliability': r.get('bpm_reliability'),
            'hrv_rmssd_ms': r.get('hrv_rmssd_ms'),
            'hrv_sdnn_ms': r.get('hrv_sdnn_ms'),
            'hrv_available': r.get('hrv_available', False),
            'stress_level': r.get('stress_level'),
            'signal_quality_pct': r.get('signal_quality_pct'),
            'breathing_rate_per_min': r.get('breathing_rate_per_min'),
            'resp_confidence': (r.get('resp') or {}).get('confidence'),
        })
    tracker = session.tracker
    return {'people': people, 'frame': {'w': session.frame_wh[0], 'h': session.frame_wh[1]},
            'max_people': MAX_PEOPLE,
            'want_width': FULL_FEED_WIDTH if session.feed_kind == 'full' else MULTI_FRAME_WIDTH,   # the browser follows this
            'feed_kind': session.feed_kind,
            'detector_error': bool(tracker is not None and tracker.detector_broken),
            'detector_error_text': tracker.detector_error_text if tracker is not None else '',
            'detector': tracker.detector_name if tracker is not None else None,
            'model_ready': _model_ready.is_set()}


# ---------------- Worker ----------------

def _finalize_challenge(session):
    ch = session.challenge
    if ch is None:
        return
    report = ch.report(session.labels())
    socketio.emit('challenge_report', clean(report), to=session.sid)
    session.challenge = None
    session.challenge_ending = False


def _perf_line(session):
    """Every 10 s: how many frames per second really arrive for each person, and how busy the model is.
    Arrival fps well below ~15 is the first thing to look at when signal quality is poor."""
    now = time.time()
    if now - session.perf_t < 10.0:
        return
    dt = now - session.perf_t
    session.perf_t = now
    with session.lock:
        subs = list(session.subjects.values())
    parts = []
    for s in subs:
        n = s.buf.frames_received
        with s.buf.lock:
            gap, s.buf.max_gap = s.buf.max_gap, 0.0
        q = s.last_result.get('signal_quality_pct')
        parts.append(f'{s.label}: {(n - s.perf_n) / dt:.1f} fps in, longest gap {gap:.2f}s, quality {q if q is not None else "--"}')
        s.perf_n = n
    print(f'[perf] mode={session.mode} running={dict(session.running)} avg_run={session.avg_run_sec:.1f}s | ' + ' | '.join(parts))


def run_worker(sid):
    """Every second: report status. When due, start a model run in a thread."""
    session = sessions.get(sid)
    if session is None:
        return

    while not session.stop_event.is_set():
        session.stop_event.wait(1.0)
        if session.stop_event.is_set():
            break

        # a finished challenge is reported as soon as the last in-flight analysis is done
        if session.challenge_ending and not session.inferring:
            _finalize_challenge(session)

        subject = session.primary()

        # ---- nobody to measure ----
        if session.mode == 'multi':
            if subject is None:
                socketio.emit('live_update', clean({
                    'no_face': True, 'buffer_sec': 0, 'hrv_min_buffer_sec': HRV_MIN_BUFFER_SEC,
                    'hrv_note': 'No face detected — step into view (up to '
                                f'{MAX_PEOPLE} people).'}), to=sid)
                socketio.emit('people_update', clean(people_payload(session)), to=sid)
                continue
        elif REQUIRE_FACE and subject is not None:
            latest = subject.buf.latest()
            if latest is not None:
                session.face_hist.append(count_faces(latest))
            full = len(session.face_hist) == session.face_hist.maxlen
            # single mode = ONE person. Two or more faces: pause, throw away what was recorded, tell the browser.
            if sum(1 for n in session.face_hist if n >= 2) >= SINGLE_MULTI_FACE_HITS:     # acts after ~3 s, no need to wait for a full history
                subject.last_result = {}
                subject.buf = FrameBuffer()
                socketio.emit('live_update', clean({
                    'no_face': True, 'multi_face': True,
                    'face_count': max(session.face_hist),
                    'buffer_sec': 0,
                    'hrv_min_buffer_sec': HRV_MIN_BUFFER_SEC,
                    'hrv_note': 'More than one face in view — single mode measures ONE person. '
                                'Make sure only you are in front of the camera (or use the multi-person tab).',
                }), to=sid)
                continue
            if (full and sum(1 for n in session.face_hist if n >= 1) <= NO_FACE_MAX_HITS):
                subject.last_result = {}
                socketio.emit('live_update', clean({
                    'no_face': True,
                    'buffer_sec': round(subject.buf.buffered_seconds(), 1),
                    'hrv_min_buffer_sec': HRV_MIN_BUFFER_SEC,
                    'hrv_note': 'No face detected — face the camera with your whole face in view.',
                }), to=sid)
                continue
        if subject is None:
            continue

        # ---- schedule model runs: one in single mode, up to MAX_PARALLEL_RUNS in multi mode ----
        # (never while a finished challenge waits for its report)
        if _model_ready.is_set() and not session.challenge_ending:
            limit = MAX_PARALLEL_RUNS if session.mode == 'multi' else 1
            while len(session.running) < limit:
                slot = _acquire_slot(limit)
                if slot is None:
                    break
                pick = _pick_subject(session)
                if pick is None:
                    _release_slot(slot)
                    break
                s, window_sec, want_hrv, buffered = pick
                with session.lock:
                    session.running[s.id] = slot
                threading.Thread(target=_infer_job, args=(session, s, window_sec, want_hrv, buffered, slot),
                                 daemon=True).start()

        _perf_line(session)

        multi = session.mode == 'multi'
        # multi mode: the shared live_update carries no graphs - every person gets their own person_update instead
        socketio.emit('live_update', clean(build_payload(session, subject, with_graph=not multi)), to=sid)
        if multi:
            socketio.emit('people_update', clean(people_payload(session)), to=sid)
            with session.lock:
                everyone = list(session.subjects.values())
            for s in everyone:
                socketio.emit('person_update', clean(build_payload(session, s, marker='emitted_pp_run')), to=sid)


# ---------------- Warm-up ----------------

def _warmup_frame():
    for path in (os.path.join(BASE_DIR, 'face1.jpg'),
                 os.path.join(BASE_DIR, '..', 'frontend', 'face1.jpg')):
        if os.path.isfile(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (320, 240))
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if FRAMES_ARE_RGB else img
    print('[warmup] face1.jpg not found — using noise (warm-up may be less complete)')
    return np.random.default_rng(0).integers(0, 255, size=(240, 320, 3), dtype=np.uint8)


def _warm_slot(slot, shapes):
    """Load model copy `slot` and compile it for the window sizes we use. Marks the copy usable at the end."""
    rng = np.random.default_rng(1 + slot)
    for img in shapes:
        for secs in sorted({QUICK_MIN_WINDOW_SEC, HRV_MIN_BUFFER_SEC, RESP_WINDOW_SEC}):
            n = int(secs * TARGET_FPS)
            noise = rng.integers(-2, 3, size=(n,) + img.shape)
            clip = np.clip(img[None] + noise, 0, 255).astype(np.uint8)
            t0 = time.time()
            with _model_locks[slot]:
                orp.get_model(slot).process_video_tensor(clip, fps=float(TARGET_FPS))
            print(f'[warmup] slot {slot}: {img.shape[1]}x{img.shape[0]} {secs:.0f}s window ready ({time.time() - t0:.1f}s)')
    with _slot_lock:
        _slot_ready[slot] = True


def _warm_extra_slots(shape):
    """Copies 1.. are only used by multi mode (crop-sized clips), so they are warmed in the background after start-up."""
    for slot in range(1, MAX_PARALLEL_RUNS):
        try:
            _warm_slot(slot, [shape])
        except Exception as e:
            print(f'[warmup] slot {slot} failed, multi mode will use fewer parallel runs:', repr(e))
            return


def warm_up_model():
    """Load + compile the model for the window sizes (and shapes) we will use BEFORE any user connects."""
    crop_shape = None
    try:
        print('[warmup] loading model and compiling — this takes up to a minute or two, once per start...')
        base = _warmup_frame().astype(np.int16)
        shapes = [base]
        crop_shape = cv2.resize(base.astype(np.uint8), (CROP_SIZE, CROP_SIZE)).astype(np.int16)
        if WARM_MULTI_SHAPE:
            shapes.append(crop_shape)
        _warm_slot(0, shapes)
        print('[warmup] done — ready for the webcam.')
    except Exception as e:
        print('[warmup] failed (not fatal, first real run will be slow):', repr(e))
        with _slot_lock:
            _slot_ready[0] = True        # let the first real run try anyway
    finally:
        _model_ready.set()
    if crop_shape is not None and MAX_PARALLEL_RUNS > 1:
        threading.Thread(target=_warm_extra_slots, args=(crop_shape,), daemon=True).start()


# ---------------- Routes / Socket.IO events ----------------

@app.route('/')
def health():
    return 'PulseRead live_server v2 is running (JAX backend).'


@socketio.on('connect')
def on_connect():
    sid = request.sid
    print('client connected', sid)
    sessions[sid] = LiveSession(sid)
    socketio.emit('server_info', clean({
        'protocols': challenge_mod.list_protocols(),
        'max_people': MAX_PEOPLE,
        'multi_frame_width': MULTI_FRAME_WIDTH,
        'full_window_sec': HRV_MIN_BUFFER_SEC,
        'resp_window_sec': RESP_WINDOW_SEC,
        'version': 2,
    }), to=sid)
    socketio.start_background_task(run_worker, sid)


@socketio.on('config')
def on_config(data):
    session = sessions.get(request.sid)
    if session is None or not isinstance(data, dict):
        return
    mode = data.get('mode')
    if mode in ('single', 'multi'):
        session.set_mode(mode)
        print(f'[config] mode={mode}')
        socketio.emit('mode_set', {'mode': mode}, to=request.sid)      # lets the browser confirm the server is v2


@socketio.on('select_person')
def on_select_person(data):
    session = sessions.get(request.sid)
    if session is None or not isinstance(data, dict):
        return
    try:
        pid = int(data.get('id'))
    except (TypeError, ValueError):
        return
    with session.lock:
        if pid in session.subjects and pid != session.selected_id:
            session.selected_id = pid
            session.subjects[pid].emitted_run = None      # re-send this person's graphs


def _locked_crop(subj, frame, box, others=()):
    """Square crop around a face that only moves when the face has really moved (see CROP_RELOCK_SHIFT)."""
    x, y, bw, bh = box
    cx, cy, size = x + bw / 2, y + bh / 2, max(bw, bh)
    lk = subj.crop_lock
    if (lk is None or math.hypot(cx - lk[0], cy - lk[1]) > CROP_RELOCK_SHIFT * lk[2]
            or not (1.0 / CROP_RELOCK_SCALE <= size / lk[2] <= CROP_RELOCK_SCALE)):
        lk = subj.crop_lock = (cx, cy, size)
    h, w = frame.shape[:2]
    # other people's faces that fall inside this person's crop are painted over: two different pulses in one picture confuse the model
    src = frame
    for ob in others:
        ox, oy, ow, oh = ob
        if (min(ox + ow, lk[0] + lk[2] * CROP_CONTEXT / 2) > max(ox, lk[0] - lk[2] * CROP_CONTEXT / 2)
                and min(oy + oh, lk[1] + lk[2] * CROP_CONTEXT / 2) > max(oy, lk[1] - lk[2] * CROP_CONTEXT / 2)):
            if src is frame:
                src = frame.copy()
            x0, y0 = int(max(ox - 0.1 * ow, 0)), int(max(oy - 0.1 * oh, 0))
            x1, y1 = int(min(ox + 1.1 * ow, w)), int(min(oy + 1.1 * oh, h))
            src[y0:y1, x0:x1] = 128
    crop, _ = ft.crop_face(src, (lk[0] - lk[2] / 2, lk[1] - lk[2] / 2, lk[2], lk[2]), out_size=CROP_SIZE, context=CROP_CONTEXT)
    return crop


@socketio.on('frame')
def on_frame(data):
    sid = request.sid
    session = sessions.get(sid)
    if session is None:
        return
    image = data.get('image', '') if isinstance(data, dict) else data
    frame = decode_frame(image)
    if frame is None:
        return

    ts = None
    if isinstance(data, dict):
        try:
            ts = float(data.get('ts'))
        except (TypeError, ValueError):
            ts = None
    if not ts or ts <= 0:
        ts = time.time()

    h, w = frame.shape[:2]
    session.frame_wh = (w, h)

    if session.mode == 'single':
        subj = session.subjects.get(1)
        if subj is None:
            return
        subj.buf.add(frame, ts)
        n = subj.buf.frames_received
        if n == 1 or n % 150 == 0:
            print(f'[frames] received={n} shape={frame.shape}')
        return

    # ---- multi-person: track faces, then give every person their own crop (or, for ONE person, the whole frame) ----
    with session.frame_lock:
        tracker = session.tracker
        if tracker is None:
            return
        live, removed = tracker.update(frame, ts)
        # ONE person: the model gets the whole frame like in single mode (that is what reaches ~54 %). 2+ people: crops.
        want = 'full' if len(live) <= 1 else 'crop'
        if want == 'crop':
            session.single_since = None
            new_kind = 'crop'
        else:
            if session.single_since is None:
                session.single_since = ts
            new_kind = 'full' if (session.feed_kind == 'full' or ts - session.single_since >= SINGLE_BACK_SEC) else 'crop'
        kind_changed = new_kind != session.feed_kind
        session.feed_kind = new_kind
        with session.lock:
            for tid in removed:
                session.subjects.pop(tid, None)
            for t in live:
                subj = session.subjects.get(t.id)
                if subj is None:
                    subj = Subject(t.id, f'Person {t.id}')
                    session.subjects[t.id] = subj
                subj.box = ft.normalised_box(t.box, w, h)
                subj.last_seen = t.last_seen
                if ts - t.last_seen > MAX_FEED_GAP_SEC:
                    continue                      # face lost for a while: do not feed guessed crops of an empty area
                if kind_changed:
                    subj.buf = FrameBuffer()      # whole frames and crops must never share one buffer
                    subj.crop_lock = None
                    print(f'[multi] feed switched to {new_kind!r}: buffers restarted')
                if subj.last_fed is not None and ts - subj.last_fed > MAX_FEED_GAP_SEC:
                    subj.buf = FrameBuffer()      # discontinuity: a buffer with a hole would corrupt the pulse signal
                    subj.crop_lock = None
                size = max(t.box[2], t.box[3])
                cx, cy = t.center
                if subj.last_center is not None and subj.last_fed is not None and ts - subj.last_fed < 1.0:
                    lx, ly, lsize = subj.last_center
                    if math.hypot(cx - lx, cy - ly) > JUMP_RESET * max(size, lsize):
                        subj.buf = FrameBuffer()      # the box jumped: probably two people swapped numbers - never mix two pulses
                        subj.crop_lock = None
                        print(f'[multi] {subj.label}: face box jumped, buffer restarted')
                subj.last_center = (cx, cy, size)
                bx, by, bw_, bh_ = t.box
                vis_w = max(0.0, min(bx + bw_, w) - max(bx, 0.0)) / max(bw_, 1.0)
                vis_h = max(0.0, min(by + bh_, h) - max(by, 0.0)) / max(bh_, 1.0)
                if vis_w * vis_h < MIN_VISIBLE:
                    continue                      # face partly outside the picture: skip (a gap > MAX_FEED_GAP_SEC restarts the buffer)
                if new_kind == 'full':
                    img = frame if w <= FULL_FEED_WIDTH else cv2.resize(
                        frame, (FULL_FEED_WIDTH, int(round(h * FULL_FEED_WIDTH / w))), interpolation=cv2.INTER_AREA)
                else:
                    img = _locked_crop(subj, frame, t.box, [o.box for o in live if o.id != t.id])
                subj.buf.add(img, ts)
                subj.last_fed = ts
        # push the face boxes about 4 times a second so they follow the faces smoothly
        now = time.time()
        if now - session.last_people_emit >= 0.25:
            session.last_people_emit = now
            socketio.emit('people_update', clean(people_payload(session)), to=sid)


# ---- challenge mode ----

@socketio.on('challenge_start')
def on_challenge_start(data):
    sid = request.sid
    session = sessions.get(sid)
    if session is None or not isinstance(data, dict):
        return
    if session.challenge is not None:
        socketio.emit('challenge_error', {'message': 'A challenge is already running.'}, to=sid)
        return
    ready = [s for s in list(session.subjects.values())
             if s.buf.buffered_seconds() >= HRV_MIN_BUFFER_SEC]
    if not ready:
        socketio.emit('challenge_error', {
            'message': f'Wait until the buffer shows {HRV_MIN_BUFFER_SEC:.0f}s / {HRV_MIN_BUFFER_SEC:.0f}s '
                       'with your face visible, then start the challenge.'}, to=sid)
        return
    try:
        ts = float(data.get('ts') or time.time())
        session.challenge = challenge_mod.ChallengeRecorder(data.get('protocol'), ts)
    except ValueError as e:
        socketio.emit('challenge_error', {'message': str(e)}, to=sid)
        return
    session.challenge_ending = False
    socketio.emit('challenge_started', clean(challenge_mod.protocol_public(data['protocol'])), to=sid)
    print('[challenge] started', data.get('protocol'))


@socketio.on('challenge_phase')
def on_challenge_phase(data):
    session = sessions.get(request.sid)
    if session is None or session.challenge is None or not isinstance(data, dict):
        return
    try:
        session.challenge.begin_phase(str(data.get('name')), float(data.get('ts') or time.time()))
    except (TypeError, ValueError):
        pass


@socketio.on('challenge_end')
def on_challenge_end(data):
    session = sessions.get(request.sid)
    if session is None or session.challenge is None:
        return
    ts = time.time()
    if isinstance(data, dict):
        try:
            ts = float(data.get('ts') or ts)
        except (TypeError, ValueError):
            pass
    session.challenge.finish(ts)
    session.challenge_ending = True     # run_worker emits the report once the last analysis is done
    print('[challenge] ended, waiting for the last analysis, then reporting')


@socketio.on('challenge_cancel')
def on_challenge_cancel():
    session = sessions.get(request.sid)
    if session is None or session.challenge is None:
        return
    session.challenge.cancel()
    session.challenge = None
    session.challenge_ending = False
    socketio.emit('challenge_cancelled', {}, to=request.sid)


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    print('client disconnected', sid)
    session = sessions.pop(sid, None)
    if session:
        session.stop_event.set()


if __name__ == '__main__':
    print('PulseRead live_server v2 starting on http://localhost:5001 (JAX backend)')
    print('OpenCV', cv2.__version__, '|', cv2.__file__, '| face detector available:',
          hasattr(cv2, 'CascadeClassifier'))
    try:                                            # tell the user NOW if face detection cannot work
        _probe = ft.auto_detector_factory(FACE_DETECTOR)
        _probe(np.zeros((240, 320, 3), np.uint8))
        print('[face] face detector OK:', _probe.info['name'])
        for _err in _probe.info['errors']:                    # why a better detector (MediaPipe) was skipped
            print('[face] NOTE - skipped detector:', _err)
        if _probe.info['name'] == 'haar':
            print('[face] Haar is the weaker detector (front-facing faces only). For tilted faces / glasses run '
                  'check_face_detector.py and install mediapipe in this environment.')
    except Exception as e:                          # noqa: BLE001
        print('[face] PROBLEM - multi-person cannot find faces:', e)
    threading.Thread(target=warm_up_model, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
