"""
PulseRead — final production pipeline (extracted from Colab notebook).

This is YOUR original deep learning + signal processing work:
  - Custom 1D CNN (build_pulseread_cnn) for BPM regression
  - Multi-ROI (forehead + both cheeks) green-channel extraction via MediaPipe
  - Dual-path extraction: ~10Hz for BPM (CNN+FFT ensemble), full native FPS for HRV
  - Adaptive FFT+CNN ensemble with a val-calibrated confidence threshold
  - Windowed HRV/PRV computation with physiological IBI filtering and
    missed-beat interpolation

Requires:
    pip install tensorflow opencv-python mediapipe scipy numpy

You also need the MediaPipe face landmarker model file. Download it once:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
and place it at MODEL_ASSET_PATH below (or pass your own path into setup_mediapipe()).
"""

import os
import gc
import subprocess
import tempfile

import numpy as np
import cv2
import mediapipe as mp
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ══════════════════════════════════════════════════════════════
# CONFIG — one source of truth (frozen values from your notebook)
# ══════════════════════════════════════════════════════════════

PULSEREAD_CONFIG = {
    # BPM path
    'frame_skip_bpm':        3,
    'bpm_filter_low_hz':     0.70,
    'bpm_filter_high_hz':    4.00,
    'bpm_filter_order':      3,
    'window_size':           150,
    'window_stride':         30,
    'ensemble_threshold':    5.25,
    'bpm_clip_low':          40,
    'bpm_clip_high':         150,
    'fft_physio_low_bpm':    40,
    'fft_physio_high_bpm':   150,

    # HRV path
    'frame_skip_hrv':        1,        # every frame
    'hrv_filter_low_hz':     0.50,
    'hrv_filter_high_hz':    4.00,
    'hrv_filter_order':      3,
    'hrv_window_sec':        15,       # window-based HRV
    'max_interp_gap_sec':    0.5,

    # Physiological limits
    'ibi_min_ms':            300,      # max 200 BPM
    'ibi_max_ms':            1500,     # min 40 BPM
    'max_hr_bpm':            200.0,
    'min_hr_bpm':            40.0,

    # Peak detection
    'peak_min_prominence':   0.15,
    'peak_prominence_factor': 0.30,
    'peak_bpm_prior_factor': 0.75,     # 75% of expected period = min distance

    # IBI cleaning
    'ibi_median_lo_factor':  0.50,
    'ibi_median_hi_factor':  1.75,
    'missed_beat_lo_factor': 1.80,
    'missed_beat_hi_factor': 2.20,
    'missed_beat_half_tol':  0.20,     # half must be within 20% of median

    # HRV reliability
    'hrv_min_valid_ibi':     8,
    'hrv_min_retention_pct': 50.0,
    'hrv_max_sdnn_ms':       90.0,
    'hrv_min_fps':           20.0,
    'hrv_min_duration_sec':  15.0,

    # Signal quality gate
    'quality_gate_score':    0.20,

    'random_seed':           42,
}

C = PULSEREAD_CONFIG  # shorthand

# The CNN works on the BPM path sampled at ~10 Hz (30 fps video / frame_skip_bpm 3).
CNN_NATIVE_FPS = 10.0


def pick_frame_skip(orig_fps):
    """
    frame_skip_bpm = 3 gives the 10 Hz the CNN expects for ~30 fps video. A slower video (e.g. a trimmed clip at
    23.2 fps -> 7.7 Hz) puts the 4 Hz band-pass edge ABOVE the Nyquist limit and scipy stops with
    "Digital filter critical frequencies must be 0 < Wn < 1". For those videos use a smaller skip so the
    BPM path stays at 10 Hz or more (it is resampled to exactly 10 Hz afterwards).
    """
    skip = C['frame_skip_bpm']
    if orig_fps and orig_fps / skip < CNN_NATIVE_FPS - 0.05:
        skip = max(1, int(orig_fps // CNN_NATIVE_FPS))
    return skip


def safe_bandpass(x, fs, lo, hi, order=3):
    """Butterworth band-pass that never asks for a frequency above the Nyquist limit of this sampling rate."""
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    if not (0 < lo < hi):
        raise ValueError(f'The video frame rate is too low to measure a pulse ({fs:.1f} Hz sampling). '
                         f'Use a video of at least about 15 frames per second.')
    b, a = scipy_signal.butter(N=order, Wn=[lo / nyq, hi / nyq], btype='band')
    return scipy_signal.filtfilt(b, a, x)

FINAL_ENSEMBLE_THRESHOLD = C['ensemble_threshold']

# ROI landmark indices (MediaPipe 478-point face mesh)
FOREHEAD_INDICES    = [9, 10, 67, 69, 104, 108, 109, 151, 337, 338, 297, 299]
LEFT_CHEEK_INDICES  = [117, 118, 119, 120, 121, 126, 142, 203, 206, 207]
RIGHT_CHEEK_INDICES = [346, 347, 348, 349, 350, 355, 371, 423, 426, 427]

MODEL_ASSET_PATH = "face_landmarker.task"  # download separately, see module docstring


# ══════════════════════════════════════════════════════════════
# CNN ARCHITECTURE — your original model
# ══════════════════════════════════════════════════════════════

def build_pulseread_cnn(input_length=150):
    """3-block 1D CNN for BPM regression. 46,529 params."""
    inputs = keras.Input(shape=(input_length, 1), name='signal_input')

    x = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', name='conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear', name='bpm_output')(x)

    return keras.Model(inputs, outputs, name='PulseRead_CNN')


def load_pulseread_model(weights_path):
    """Load the frozen V7 model (.h5 or .keras)."""
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={'mae': tf.keras.metrics.MeanAbsoluteError()},
        compile=False,
    )
    model.compile(loss='mae', metrics=['mae'])
    return model


# ══════════════════════════════════════════════════════════════
# MEDIAPIPE SETUP
# ══════════════════════════════════════════════════════════════

def setup_mediapipe(model_asset_path=MODEL_ASSET_PATH):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    return options, FaceLandmarker


# ══════════════════════════════════════════════════════════════
# ROI EXTRACTION
# ══════════════════════════════════════════════════════════════

def get_forehead_box(landmarks, h, w):
    xs = [int(landmarks[idx].x * w) for idx in FOREHEAD_INDICES]
    ys = [int(landmarks[idx].y * h) for idx in FOREHEAD_INDICES]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    margin_x = int((x_max - x_min) * 0.1)
    margin_y = int((y_max - y_min) * 0.15)
    return x_min + margin_x, x_max - margin_x, y_min + margin_y, y_max - margin_y


def get_cheek_box(landmarks, h, w, indices):
    xs = [int(landmarks[idx].x * w) for idx in indices]
    ys = [int(landmarks[idx].y * h) for idx in indices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    margin_x = int((x_max - x_min) * 0.08)
    margin_y = int((y_max - y_min) * 0.08)
    return x_min + margin_x, x_max - margin_x, y_min + margin_y, y_max - margin_y


def extract_roi_green(img_array, box):
    """Average green channel from a bounding box. Returns None if box invalid."""
    x_min, x_max, y_min, y_max = box
    if x_max <= x_min or y_max <= y_min:
        return None
    crop = img_array[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return float(np.mean(crop[:, :, 1]))


# ══════════════════════════════════════════════════════════════
# DUAL-PATH SIGNAL EXTRACTION
#   BPM path:  every 3rd frame (~10Hz) -> CNN/FFT ensemble
#   HRV path:  every frame (native FPS) -> windowed PRV
# ══════════════════════════════════════════════════════════════

def extract_dual_path_signals(frames_dir, options, FaceLandmarker, orig_fps,
                               frame_skip_bpm=None, snapshot_path=None):
    if frame_skip_bpm is None:
        frame_skip_bpm = C['frame_skip_bpm']

    frames = sorted(os.listdir(frames_dir))
    n_total = len(frames)

    bpm_fh, bpm_lc, bpm_rc = [], [], []
    hrv_fh, hrv_lc, hrv_rc = [], [], []
    n_det_bpm, n_det_hrv = 0, 0
    snapshot_saved = False

    with FaceLandmarker.create_from_options(options) as landmarker:
        for i, fname in enumerate(frames):
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)

            if not res.face_landmarks:
                hrv_fh.append(np.nan)
                hrv_lc.append(np.nan)
                hrv_rc.append(np.nan)
                del img, rgb
                continue

            landmarks = res.face_landmarks[0]

            # Visual explainability: save ONE annotated frame showing the
            # actual ROI boxes used for signal extraction, so the output
            # isn't a black-box number — the user/examiner can see what
            # region the model actually looked at.
            if snapshot_path and not snapshot_saved:
                annotated = img.copy()
                for box, color in [
                    (get_forehead_box(landmarks, h, w), (0, 255, 0)),
                    (get_cheek_box(landmarks, h, w, LEFT_CHEEK_INDICES), (255, 200, 0)),
                    (get_cheek_box(landmarks, h, w, RIGHT_CHEEK_INDICES), (255, 200, 0)),
                ]:
                    x_min, x_max, y_min, y_max = box
                    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.imwrite(snapshot_path, annotated)
                snapshot_saved = True
            del img

            arr = mp_image.numpy_view()

            fh_v = extract_roi_green(arr, get_forehead_box(landmarks, h, w))
            lc_v = extract_roi_green(arr, get_cheek_box(landmarks, h, w, LEFT_CHEEK_INDICES))
            rc_v = extract_roi_green(arr, get_cheek_box(landmarks, h, w, RIGHT_CHEEK_INDICES))

            hrv_fh.append(fh_v if fh_v else np.nan)
            hrv_lc.append(lc_v if lc_v else np.nan)
            hrv_rc.append(rc_v if rc_v else np.nan)
            n_det_hrv += 1

            if i % frame_skip_bpm == 0:
                if fh_v: bpm_fh.append(fh_v)
                if lc_v: bpm_lc.append(lc_v)
                if rc_v: bpm_rc.append(rc_v)
                n_det_bpm += 1

            if i % 100 == 0:
                gc.collect()

    # BPM signal
    eff_fps_bpm = orig_fps / frame_skip_bpm
    min_bpm_len = min(len(bpm_fh), len(bpm_lc), len(bpm_rc))
    signal_bpm = None

    if min_bpm_len >= 30:
        def norm(x, n):
            a = np.array(x[:n], dtype=np.float32)
            return (a - a.mean()) / (a.std() + 1e-8)
        raw_bpm = (norm(bpm_fh, min_bpm_len) + norm(bpm_lc, min_bpm_len) + norm(bpm_rc, min_bpm_len)) / 3.0
        trimmed = raw_bpm[20:]
        det = scipy_signal.detrend(trimmed)
        signal_bpm = safe_bandpass(det, eff_fps_bpm, 0.7, 4.0)
        # a slower video was sampled at a different rate than the CNN's 10 Hz: bring it to exactly 10 Hz
        if abs(eff_fps_bpm - CNN_NATIVE_FPS) > 0.05 and len(signal_bpm) > 2:
            t_old = np.arange(len(signal_bpm)) / eff_fps_bpm
            t_new = np.arange(0.0, t_old[-1], 1.0 / CNN_NATIVE_FPS)
            signal_bpm = np.interp(t_new, t_old, signal_bpm).astype(np.float32)
            eff_fps_bpm = CNN_NATIVE_FPS

    # HRV signal
    eff_fps_hrv = orig_fps
    hrv_fh_arr = np.array(hrv_fh, dtype=np.float32)
    hrv_lc_arr = np.array(hrv_lc, dtype=np.float32)
    hrv_rc_arr = np.array(hrv_rc, dtype=np.float32)
    signal_hrv = None

    min_hrv_len = min(len(hrv_fh_arr), len(hrv_lc_arr), len(hrv_rc_arr))
    if min_hrv_len >= 50:
        MAX_INTERP_FRAMES = int(eff_fps_hrv * 0.5)

        def interp_short_gaps(arr, max_gap):
            out = arr.copy()
            nan_m = np.isnan(out)
            if not nan_m.any():
                return out
            in_gap, gap_start = False, 0
            for j in range(len(out)):
                if nan_m[j] and not in_gap:
                    in_gap, gap_start = True, j
                elif not nan_m[j] and in_gap:
                    gap_len = j - gap_start
                    if gap_len <= max_gap:
                        v0 = out[gap_start - 1] if gap_start > 0 else out[j]
                        v1 = out[j]
                        out[gap_start:j] = np.linspace(v0, v1, gap_len + 2)[1:-1]
                    in_gap = False
            return out

        fh_i = interp_short_gaps(hrv_fh_arr[:min_hrv_len], MAX_INTERP_FRAMES)
        lc_i = interp_short_gaps(hrv_lc_arr[:min_hrv_len], MAX_INTERP_FRAMES)
        rc_i = interp_short_gaps(hrv_rc_arr[:min_hrv_len], MAX_INTERP_FRAMES)

        valid_mask = (~np.isnan(fh_i) & ~np.isnan(lc_i) & ~np.isnan(rc_i))

        if valid_mask.sum() >= 50:
            def norm_valid(arr, mask):
                a = arr[mask].copy()
                return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)
            raw_hrv = (norm_valid(fh_i, valid_mask) + norm_valid(lc_i, valid_mask) + norm_valid(rc_i, valid_mask)) / 3.0
            signal_hrv = safe_bandpass(raw_hrv, eff_fps_hrv, 0.5, 4.0)

    return {
        'signal_bpm': signal_bpm,
        'signal_hrv': signal_hrv,
        'eff_fps_bpm': eff_fps_bpm,
        'eff_fps_hrv': eff_fps_hrv,
        'n_detected_bpm': n_det_bpm,
        'n_detected_hrv': n_det_hrv,
        'n_frames_total': n_total,
    }


# ══════════════════════════════════════════════════════════════
# BPM ENSEMBLE (per-window CNN + FFT, physiologically clamped)
# ══════════════════════════════════════════════════════════════

def get_window_ensemble_bpm(w_norm, eff_fps, model, window_size=150,
                             threshold=FINAL_ENSEMBLE_THRESHOLD):
    fft_m = np.abs(np.fft.fft(w_norm))
    freqs = np.fft.fftfreq(window_size, d=1 / eff_fps)
    valid = (freqs >= 0.7) & (freqs <= 4.0)

    if not valid.any():
        return None, None, None, 0.0, 0.0

    vm = fft_m[valid]
    vf = freqs[valid]
    ratio = float(vm.max() / (vm.mean() + 1e-8))
    fft_bpm = float(vf[np.argmax(vm)] * 60)
    fft_valid = (40 <= fft_bpm <= 150)

    cnn_bpm = float(np.clip(model.predict(w_norm.reshape(1, window_size, 1), verbose=0)[0][0], 40, 150))
    conf_01 = float(1.0 / (1.0 + np.exp(-(ratio - 3.0) / 1.5)))

    if fft_valid:
        c = conf_01 * 6
        fw = 0.2 + 0.6 * (c / (c + threshold))
        cw = 1.0 - fw
        ens = float(np.clip(fw * fft_bpm + cw * cnn_bpm, 40, 150))
    else:
        ens = cnn_bpm
        conf_01 = 0.0

    return ens, fft_bpm, cnn_bpm, conf_01, ratio


def compute_signal_quality(combined_signal, eff_fps, n_frames_total, n_detected):
    if n_frames_total == 0:
        return 'Poor', 0.0

    detection_rate = n_detected / n_frames_total
    q_detection = float(min(detection_rate, 1.0))

    sig_std = combined_signal.std()
    q_variance = float(min(sig_std / 0.3, 1.0))

    n = len(combined_signal)
    fft_mag = np.abs(np.fft.fft(combined_signal)) ** 2
    freqs = np.fft.fftfreq(n, d=1 / eff_fps)
    hr_band = (freqs >= 0.7) & (freqs <= 4.0)
    total_power = fft_mag.sum() + 1e-8
    hr_power = fft_mag[hr_band].sum()
    q_spectral = float(min(hr_power / total_power * 5, 1.0))

    quality_score = 0.30 * q_detection + 0.30 * q_variance + 0.40 * q_spectral

    if quality_score < 0.25:
        label = 'Poor'
    elif quality_score < 0.50:
        label = 'Fair'
    elif quality_score < 0.75:
        label = 'Good'
    else:
        label = 'Excellent'

    return label, round(quality_score, 3)


# ══════════════════════════════════════════════════════════════
# HRV / PRV — windowed, physiologically filtered
# ══════════════════════════════════════════════════════════════

def compute_hrv_single(signal_seg, eff_fps, estimated_bpm, signal_quality_score=1.0):
    res = {
        'sdnn_ms': None, 'rmssd_ms': None,
        'mean_ibi_ms': None, 'median_ibi_ms': None,
        'mean_hr_from_ibi': None, 'hrv_available': False,
        'n_peaks': 0, 'n_valid_ibi': 0, 'pct_retained': 0.0,
        'corrected_intervals': 0, 'removed_intervals': 0,
    }

    if len(signal_seg) < 30 or estimated_bpm is None:
        return res

    min_dist_sec = (60.0 / max(estimated_bpm, 40)) * C['peak_bpm_prior_factor']
    min_distance = max(int(min_dist_sec * eff_fps), 2)
    sig_std = signal_seg.std()
    prominence = max(C['peak_min_prominence'], sig_std * C['peak_prominence_factor'])

    peaks, _ = find_peaks(signal_seg, distance=min_distance, prominence=prominence)
    res['n_peaks'] = len(peaks)

    if len(peaks) < 3:
        return res

    ibi_raw = np.diff(peaks) / eff_fps * 1000
    plaus = ibi_raw[(ibi_raw >= C['ibi_min_ms']) & (ibi_raw <= C['ibi_max_ms'])]
    removed = int(len(ibi_raw) - len(plaus))

    if len(plaus) < 2:
        res['removed_intervals'] = removed
        return res

    med = np.median(plaus)
    clean = plaus[(plaus >= med * C['ibi_median_lo_factor']) & (plaus <= med * C['ibi_median_hi_factor'])]
    removed += int(len(plaus) - len(clean))

    if len(clean) < 2:
        res['removed_intervals'] = removed
        return res

    corrected, final_l = 0, []
    med_ref = np.median(clean)
    for ibi in clean:
        if C['missed_beat_lo_factor'] * med_ref < ibi < C['missed_beat_hi_factor'] * med_ref:
            half = ibi / 2.0
            if (C['ibi_min_ms'] <= half <= C['ibi_max_ms'] and
                    abs(half - med_ref) / med_ref < C['missed_beat_half_tol']):
                final_l.extend([half, half])
                corrected += 1
            else:
                removed += 1
        else:
            final_l.append(ibi)

    ibi_f = np.array(final_l)
    if len(ibi_f) < 2:
        return res

    pct = len(ibi_f) / max(len(ibi_raw), 1) * 100
    mean = float(ibi_f.mean())
    med2 = float(np.median(ibi_f))
    sdnn = float(ibi_f.std(ddof=1))
    diffs = np.diff(ibi_f)
    rmss = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) >= 1 else None

    res.update({
        'sdnn_ms': round(sdnn, 1),
        'rmssd_ms': round(rmss, 1) if rmss else None,
        'mean_ibi_ms': round(mean, 1),
        'median_ibi_ms': round(med2, 1),
        'mean_hr_from_ibi': round(60000 / mean, 1),
        'n_valid_ibi': len(ibi_f),
        'pct_retained': round(pct, 1),
        'corrected_intervals': corrected,
        'removed_intervals': removed,
    })
    return res


def compute_hrv_windowed(hrv_signal, eff_fps_hrv, estimated_bpm,
                          signal_quality_score=1.0, window_sec=None):
    if window_sec is None:
        window_sec = C['hrv_window_sec']

    window_samples = int(window_sec * eff_fps_hrv)
    total_samples = len(hrv_signal)

    if total_samples < window_samples:
        return compute_hrv_single(hrv_signal, eff_fps_hrv, estimated_bpm, signal_quality_score)

    win_sdnn, win_rmssd, win_mean_ibi = [], [], []
    win_n_valid, win_retention = [], []
    stride_samples = window_samples // 2

    for start in range(0, total_samples - window_samples + 1, stride_samples):
        win = hrv_signal[start:start + window_samples]
        res = compute_hrv_single(win, eff_fps_hrv, estimated_bpm, signal_quality_score)
        if res['n_valid_ibi'] >= 4:
            win_sdnn.append(res['sdnn_ms'])
            win_rmssd.append(res['rmssd_ms'])
            win_mean_ibi.append(res['mean_ibi_ms'])
            win_n_valid.append(res['n_valid_ibi'])
            win_retention.append(res['pct_retained'])

    if len(win_sdnn) == 0:
        return compute_hrv_single(hrv_signal, eff_fps_hrv, estimated_bpm, signal_quality_score)

    agg_sdnn = float(np.median([s for s in win_sdnn if s is not None]))
    agg_rmssd = float(np.median([r for r in win_rmssd if r is not None]))
    agg_mean_ibi = float(np.median([m for m in win_mean_ibi if m is not None]))
    agg_retention = float(np.median(win_retention))
    agg_n_valid = int(np.median(win_n_valid))
    mean_hr_ibi = 60000.0 / agg_mean_ibi if agg_mean_ibi else None

    sdnn_stability = float(np.std(win_sdnn)) if len(win_sdnn) > 1 else 0

    low_fps = eff_fps_hrv < C['hrv_min_fps']
    bpm_diff = abs(mean_hr_ibi - estimated_bpm) if mean_hr_ibi else 99
    bpm_ok = bpm_diff < 15

    if low_fps:
        reliability_label, hrv_avail = 'Unavailable', False
        warning = f'Sampling rate {eff_fps_hrv:.1f}Hz too low for reliable PRV.'
    elif (agg_n_valid >= C['hrv_min_valid_ibi'] and agg_retention >= C['hrv_min_retention_pct']
          and agg_sdnn < C['hrv_max_sdnn_ms'] and bpm_ok and len(win_sdnn) >= 2):
        reliability_label, hrv_avail, warning = 'Medium', True, ''
    elif agg_n_valid >= 5 and agg_retention >= 35:
        reliability_label, hrv_avail = 'Low', True
        warning = 'Low confidence — indicative only.'
    else:
        reliability_label, hrv_avail = 'Unavailable', False
        warning = 'Insufficient reliable beat intervals.'

    if not hrv_avail:
        wellness, s_conf = 'Insufficient data', 'low'
    elif agg_rmssd < 15 or agg_sdnn < 15:
        wellness, s_conf = 'Higher stress indicator', 'low'
    elif agg_rmssd < 35 or agg_sdnn < 40:
        wellness, s_conf = 'Moderate stress indicator', 'low'
    elif agg_rmssd < 70 or agg_sdnn < 80:
        wellness, s_conf = 'Lower stress indicator', 'low'
    else:
        wellness, s_conf = 'Insufficient data', 'low'
        hrv_avail = False
        reliability_label = 'Unavailable'
        warning += ' PRV values too variable — treating as unreliable.'

    if low_fps or not hrv_avail:
        reliability_score = 0
    else:
        score = 0
        score += min(agg_retention / 100 * 40, 40)
        score += min(agg_n_valid / 20 * 30, 30)
        score += (10 if bpm_ok else 0)
        score += (20 if len(win_sdnn) >= 3 else 10)
        reliability_score = int(min(score, 79))

    return {
        'sdnn_ms': round(agg_sdnn, 1),
        'rmssd_ms': round(agg_rmssd, 1),
        'mean_ibi_ms': round(agg_mean_ibi, 1),
        'median_ibi_ms': round(agg_mean_ibi, 1),
        'mean_hr_from_ibi': round(mean_hr_ibi, 1) if mean_hr_ibi else None,
        'hrv_available': hrv_avail,
        'n_valid_ibi': agg_n_valid,
        'pct_retained': round(agg_retention, 1),
        'corrected_intervals': 0,
        'removed_intervals': 0,
        'hrv_reliability': reliability_label,
        'hrv_reliability_score': reliability_score,
        'n_windows_used': len(win_sdnn),
        'sdnn_stability_ms': round(sdnn_stability, 1),
        'stress_indicator': wellness,
        'stress_confidence': s_conf,
        'warning': warning,
        'hrv_note': (f'Window-based PRV: {len(win_sdnn)} windows of {window_sec}s. '
                     f'RMSSD={agg_rmssd:.1f}ms, SDNN={agg_sdnn:.1f}ms. '
                     f'Rule-based wellness only — not a medical diagnosis.'),
    }


# ══════════════════════════════════════════════════════════════
# FINAL PRODUCTION ENTRY POINT
# ══════════════════════════════════════════════════════════════

def analyze_video(video_path, model, options, FaceLandmarker, orig_fps=None,
                   snapshot_path=None):
    """
    PulseRead — final production function.
    DUAL PATH: BPM ~10Hz (CNN+FFT ensemble), HRV native FPS (windowed PRV).
    Returns a structured result dict.

    snapshot_path: optional path (e.g. 'output/roi_snapshot.jpg') — if given,
    saves one annotated frame showing the actual forehead/cheek ROI boxes
    used for signal extraction (visual explainability).
    """
    result = {
        'status': 'error', 'error_message': '',
        'final_bpm': None, 'cnn_bpm': None, 'fft_bpm': None, 'fft_confidence': 0.0,
        'signal_quality': 'Poor', 'signal_quality_score': 0.0, 'pulse_signal': None,
        'n_frames_total': 0, 'n_frames_detected': 0,
        'duration_sec': 0.0, 'original_fps': 0.0,
        'bpm_processing_fps': 0.0, 'hrv_processing_fps': 0.0,
        'windows_total': 0, 'windows_used': 0, 'windows_rejected': 0, 'window_bpms': [],
        'sdnn_ms': None, 'rmssd_ms': None, 'mean_ibi_ms': None, 'median_ibi_ms': None,
        'mean_hr_from_ibi': None,
        'hrv_available': False, 'hrv_reliability': 'Unavailable', 'hrv_reliability_score': 0,
        'valid_peak_count': 0, 'valid_ibi_count': 0, 'pct_ibi_retained': 0.0,
        'corrected_intervals': 0,
        'stress_indicator': 'Insufficient data', 'stress_confidence': 'low',
        'hrv_note': '', 'warning': '', 'roi_snapshot_path': None,
    }

    if not os.path.exists(video_path):
        result['error_message'] = 'Video file not found.'
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result['error_message'] = 'Cannot open video file.'
        return result

    fps_orig = orig_fps or cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = total_frames / fps_orig if fps_orig > 0 else 0
    result['duration_sec'] = round(duration_sec, 1)
    result['original_fps'] = round(fps_orig, 3)
    bpm_skip = pick_frame_skip(fps_orig)          # 3 for ~30 fps video; smaller for slower videos (see pick_frame_skip)
    result['bpm_processing_fps'] = round(fps_orig / bpm_skip, 3)
    result['hrv_processing_fps'] = round(fps_orig, 3)

    if duration_sec < 10:
        result['error_message'] = f'Video too short ({duration_sec:.1f}s). Record at least 15 seconds.'
        return result

    frames_dir = os.path.join(tempfile.gettempdir(), 'pulseread_frames_full')
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    # Use bundled ffmpeg from imageio-ffmpeg, fall back to system ffmpeg.
    try:
        import imageio_ffmpeg
        _ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _ffmpeg_bin = 'ffmpeg'

    cmd = (
        f'"{_ffmpeg_bin}" -i "{video_path}" -vf "scale=320:-1" '
        f'-vsync vfr "{frames_dir}/frame_%04d.jpg" -y -loglevel error'
    )
    r = subprocess.run(cmd, shell=True, capture_output=True)
    if r.returncode != 0:
        stderr_msg = r.stderr.decode(errors='replace').strip()
        result['error_message'] = f'Frame extraction failed. ffmpeg stderr: {stderr_msg}'
        return result

    frames = sorted(os.listdir(frames_dir))
    result['n_frames_total'] = len(frames)

    if len(frames) < 30:
        result['error_message'] = 'Too few frames extracted.'
        return result

    dual = extract_dual_path_signals(frames_dir, options, FaceLandmarker,
                                      orig_fps=fps_orig, frame_skip_bpm=bpm_skip,
                                      snapshot_path=snapshot_path)
    if snapshot_path and os.path.exists(snapshot_path):
        result['roi_snapshot_path'] = snapshot_path

    result['n_frames_detected'] = dual['n_detected_bpm']

    if dual['n_detected_bpm'] < 20:
        result['error_message'] = 'Face not detected reliably. Ensure good lighting.'
        return result

    detection_rate = dual['n_detected_bpm'] / max(len(frames) // bpm_skip, 1)
    if detection_rate < 0.5:
        result['error_message'] = f'Unstable face detection ({detection_rate*100:.0f}% of frames).'
        return result

    if dual['signal_bpm'] is None:
        result['error_message'] = 'Insufficient ROI data for BPM.'
        return result

    filtered_bpm = dual['signal_bpm']
    result['pulse_signal'] = filtered_bpm.tolist()

    eff_fps_bpm = dual['eff_fps_bpm']
    n_frames_bpm = len(frames) // bpm_skip
    quality_label, quality_score = compute_signal_quality(
        filtered_bpm, eff_fps_bpm, n_frames_total=n_frames_bpm, n_detected=dual['n_detected_bpm'])

    result['signal_quality'] = quality_label
    result['signal_quality_score'] = quality_score

    if quality_score < C['quality_gate_score']:
        result['status'] = 'low_confidence'
        result['error_message'] = 'Signal quality too poor. Try better lighting and remain still.'
        return result

    WINDOW_SIZE, INFER_STRIDE = C['window_size'], C['window_stride']
    w_ens, w_fft, w_cnn, w_conf, w_qual = [], [], [], [], []
    n_windows = n_rejected = 0

    for start in range(0, len(filtered_bpm) - WINDOW_SIZE + 1, INFER_STRIDE):
        window = filtered_bpm[start:start + WINDOW_SIZE]
        n_windows += 1
        w_norm = (window - window.mean()) / (window.std() + 1e-8)

        ens, fft_b, cnn_b, conf, ratio = get_window_ensemble_bpm(w_norm, eff_fps_bpm, model)

        if ens is None or ratio < 1.3:
            n_rejected += 1
            continue

        w_ens.append(ens); w_fft.append(fft_b); w_cnn.append(cnn_b)
        w_conf.append(conf); w_qual.append(ratio)

    result['windows_total'] = n_windows
    result['windows_rejected'] = n_rejected
    result['windows_used'] = len(w_ens)
    result['window_bpms'] = [round(b, 1) for b in w_ens]

    if len(w_ens) == 0:
        result['error_message'] = 'No reliable BPM windows found.'
        return result

    wts = np.array(w_qual, dtype=np.float32)
    wts_n = wts / wts.sum()
    ens_arr = np.array(w_ens)
    sidx = np.argsort(ens_arr)
    cumsum = np.cumsum(wts_n[sidx])
    midx = min(np.searchsorted(cumsum, 0.5), len(ens_arr) - 1)
    final_bpm = float(ens_arr[sidx[midx]])

    result['final_bpm'] = round(final_bpm, 1)
    result['cnn_bpm'] = round(float(np.median(w_cnn)), 1)
    result['fft_bpm'] = (round(float(np.median([b for b in w_fft if b and 40 <= b <= 150])), 1)
                          if any(b and 40 <= b <= 150 for b in w_fft) else None)
    result['fft_confidence'] = round(float(np.mean(w_conf)), 3)

    if dual['signal_hrv'] is not None:
        hrv = compute_hrv_windowed(dual['signal_hrv'], dual['eff_fps_hrv'],
                                    estimated_bpm=final_bpm, signal_quality_score=quality_score)
    else:
        hrv = {'hrv_available': False, 'hrv_reliability': 'Unavailable', 'hrv_reliability_score': 0,
               'stress_indicator': 'Insufficient data', 'stress_confidence': 'low',
               'sdnn_ms': None, 'rmssd_ms': None, 'mean_ibi_ms': None, 'median_ibi_ms': None,
               'mean_hr_from_ibi': None, 'n_valid_ibi': 0, 'pct_retained': 0.0,
               'corrected_intervals': 0, 'n_windows_used': 0,
               'warning': 'HRV path unavailable.', 'hrv_note': ''}

    result.update({
        'sdnn_ms': hrv['sdnn_ms'], 'rmssd_ms': hrv.get('rmssd_ms'),
        'mean_ibi_ms': hrv['mean_ibi_ms'], 'median_ibi_ms': hrv.get('median_ibi_ms'),
        'mean_hr_from_ibi': hrv.get('mean_hr_from_ibi'),
        'hrv_available': hrv['hrv_available'], 'hrv_reliability': hrv['hrv_reliability'],
        'hrv_reliability_score': hrv['hrv_reliability_score'],
        'valid_peak_count': hrv.get('n_peaks', 0), 'valid_ibi_count': hrv['n_valid_ibi'],
        'pct_ibi_retained': hrv['pct_retained'], 'corrected_intervals': hrv['corrected_intervals'],
        'stress_indicator': hrv['stress_indicator'], 'stress_confidence': hrv['stress_confidence'],
        'hrv_note': hrv.get('hrv_note', ''), 'warning': hrv.get('warning', ''),
    })

    result['status'] = 'ok'
    return result


def print_pulseread_result(r, ref_bpm=None):
    print("=" * 55)
    print("PULSEREAD FINAL RESULT")
    print("=" * 55)
    if r['status'] != 'ok':
        print(f"Status: {r['status'].upper()}")
        print(f"Error:  {r['error_message']}")
        return
    err_str = ''
    if ref_bpm is not None:
        err = abs(r['final_bpm'] - ref_bpm)
        err_str = f"  (ref={ref_bpm} BPM, error={err:.1f} BPM)"
    print(f"Heart Rate:          {r['final_bpm']} BPM{err_str}")
    print(f"  CNN estimate:      {r['cnn_bpm']} BPM")
    print(f"  FFT estimate:      {r['fft_bpm']} BPM")
    print(f"Signal Quality:      {r['signal_quality']} ({r['signal_quality_score']:.2f}/1.0)")
    if r['hrv_available']:
        print(f"PRV/HRV:  RMSSD={r['rmssd_ms']}ms SDNN={r['sdnn_ms']}ms "
              f"Reliability={r['hrv_reliability']} ({r['hrv_reliability_score']}/100)")
    else:
        print(f"PRV/HRV:  Unavailable ({r['warning'][:80]})")
    print(f"Wellness Indicator:  {r['stress_indicator']}")
    print("=" * 55)
