"""
open-rppg wrapper — used as the final BPM + HRV output source.

IMPORTANT: open-rppg's FacePhys model uses raw JAX ops internally (its Mamba/
state-space TNM layer). Keras 3's backend must be set to 'jax' BEFORE keras
or tensorflow is imported anywhere in this process. That means:

  - This module must run in a SEPARATE Python process from pulseread_pipeline.py
    (which needs the TensorFlow backend for your CNN).
  - Set the environment variable before any import happens, e.g. by launching
    this as its own process, or setting KERAS_BACKEND=jax in the environment
    before `python` even starts.

Requires:
    pip install open-rppg
"""

import os

# Must happen before rppg (and therefore keras) is imported anywhere in this process.
os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402
import rppg  # noqa: E402  (import after backend env var is set)

import graph_data  # noqa: E402  (numpy-only helper, same folder)
import respiration  # noqa: E402  (numpy-only helper, same folder)


def _to_native(obj):
    """
    Recursively convert numpy scalars/arrays (e.g. float32, int64) into
    plain Python types so the result is JSON-serializable. open-rppg's
    hrv dict returns numpy float32 values, which json.dumps() rejects.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if hasattr(obj, 'item'):  # covers numpy scalar types (float32, int64, bool_, ...)
        return obj.item()
    return obj


_models = {}


def get_model(slot=0):
    """Lazy singleton per slot - avoids reloading weights on every call.
    Slot 0 is the normal model. live_server.py's multi-person mode also asks for slot 1, 2 ...
    so several people can be analysed at the same time (a model object keeps the last pulse signal
    inside itself and is not safe to share between threads; the model is only a few MB)."""
    m = _models.get(slot)
    if m is None:
        m = rppg.Model()  # defaults to FacePhys.rlap
        _models[slot] = m
    return m


def analyze_video_openrppg(video_path):
    """
    Run open-rppg on a video file and return a result dict shaped
    to match pulseread_pipeline.analyze_video()'s output where possible,
    so the two can be swapped/compared easily.
    """
    model = get_model()
    r = model.process_video(video_path)

    hrv = r.get('hrv') or {}

    result = {
        'status': 'ok' if r.get('hr') is not None else 'error',
        'source': 'open-rppg (FacePhys.rlap)',
        'final_bpm': round(r['hr'], 1) if r.get('hr') is not None else None,
        'signal_quality_score': round(r.get('SQI', 0.0), 3),
        'latency_sec': r.get('latency'),

        # HRV fields, pulled out of open-rppg's hrv dict.
        # NOTE: open-rppg only populates `hrv` at all if SQI > 0.5 internally —
        # check signal_quality_score if hrv comes back empty.
        'sdnn_ms': hrv.get('sdnn'),
        'rmssd_ms': hrv.get('rmssd'),
        'mean_ibi_ms': hrv.get('ibi'),
        'bpm_from_hrv': hrv.get('bpm'),
        'lf_hf_ratio': hrv.get('LF/HF') or hrv.get('lf_hf'),
        'breathing_rate': hrv.get('breathingrate'),
        'hrv_available': bool(hrv),
        'graph': _graph_from_model(model),
        # breathing rate from the same pulse signal (FM + AM + baseline wander fusion)
        'respiration': _resp_from_model(model),
    }

    if not hrv:
        result['warning'] = (
            f"HRV empty — SQI={result['signal_quality_score']} "
            f"(open-rppg only computes HRV when SQI > 0.5)."
        )
    else:
        result['warning'] = ''

    return _to_native(result)


def _bvp_and_fps(model, window_sec=None):
    """
    The raw pulse (BVP) signal the model just produced, plus its sampling rate.
    Returns (bvp, fps) or (None, None). Never raises.
    """
    try:
        bvp, ts = model.bvp()
        bvp = np.asarray(bvp, dtype=float).ravel()
        ts = np.asarray(ts, dtype=float).ravel()
        fps = None
        if len(ts) > 2:
            fps = 1.0 / float(np.median(np.diff(ts)))
        if fps is None or not (5.0 <= fps <= 120.0):   # ts not in seconds? use the model's own rate
            fps = float(model.fps)
        if window_sec:
            bvp = bvp[-int(window_sec * fps):]
        return bvp, fps
    except Exception as e:  # noqa: BLE001
        import sys
        print(f'[bvp] could not read the pulse signal: {e!r}', file=sys.stderr)
        return None, None


def _resp_from_model(model, window_sec=None):
    """Breathing-rate estimate for the model's latest BVP. Never raises."""
    bvp, fps = _bvp_and_fps(model, window_sec)
    if bvp is None:
        return None
    return respiration.estimate_respiration(bvp, fps)


def _graph_from_model(model, window_sec=None):
    """
    Pull the BVP the model just produced and turn it into graph data
    (waveform + beats + IBI + HR). Never raises: graphs are optional.
    """
    try:
        bvp, ts = model.bvp()
        bvp = np.asarray(bvp, dtype=float).ravel()
        ts = np.asarray(ts, dtype=float).ravel()
        fps = None
        if len(ts) > 2:
            fps = 1.0 / float(np.median(np.diff(ts)))
        if fps is None or not (5.0 <= fps <= 120.0):   # ts not in seconds? use the model's own rate
            fps = float(model.fps)
        if window_sec:
            bvp = bvp[-int(window_sec * fps):]
        return graph_data.build_graph(bvp, fps)
    except Exception as e:  # noqa: BLE001
        import sys
        print(f'[graph] could not build graph data: {e!r}', file=sys.stderr)
        return None


def get_raw_bvp(video_path):
    """
    Extract the native-frame-rate BVP signal directly, for feeding into
    your OWN compute_hrv_windowed()/compute_hrv_single() from
    pulseread_pipeline.py instead of open-rppg's built-in HRV calc.
    Returns (bvp, timestamps, fps).
    """
    model = get_model()
    model.process_video(video_path)  # populates internal buffer
    bvp, ts = model.bvp()
    return bvp, ts, model.fps


def classify_stress(rmssd_ms, sdnn_ms):
    """
    Simple rule-based stress label from HRV. Same thresholds used for
    the video pipeline in main.py, kept here too so the live webcam
    script (which runs in this same JAX-backed process) can reuse it
    without importing main.py (which pulls in TensorFlow).
    """
    if rmssd_ms is None or sdnn_ms is None:
        return 'Insufficient data'
    if rmssd_ms < 15 or sdnn_ms < 15:
        return 'High'
    if rmssd_ms < 35 or sdnn_ms < 40:
        return 'Moderate'
    if rmssd_ms < 70 or sdnn_ms < 80:
        return 'Low'
    return 'Very Low'


def build_display_result(raw_result):
    """
    Shape a raw model.hr()-style dict (has 'hr', 'SQI', 'hrv') into the
    same clean field names main.py uses for video analysis, so live
    webcam and video-file results look identical to the frontend.
    """
    hrv = raw_result.get('hrv') or {}
    rmssd = hrv.get('rmssd')
    sdnn = hrv.get('sdnn')

    return _to_native({
        'status': 'ok' if raw_result.get('hr') is not None else 'no_signal',
        'heart_rate_bpm': round(raw_result['hr'], 1) if raw_result.get('hr') is not None else None,
        'hrv_rmssd_ms': rmssd,
        'hrv_sdnn_ms': sdnn,
        'signal_quality_pct': round((raw_result.get('SQI') or 0) * 100, 1),
        'stress_level': classify_stress(rmssd, sdnn),
        'breathing_rate_per_min': round((hrv.get('breathingrate') or 0) * 60, 1) if hrv.get('breathingrate') else None,
        'hrv_available': bool(hrv),
    })


if __name__ == '__main__':
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python openrppg_pipeline.py <video_path>", file=sys.stderr)
        sys.exit(1)

    out = analyze_video_openrppg(path)
    # Print ONLY the JSON on stdout so a parent process can capture it cleanly.
    print(json.dumps(out))