"""
graph_data.py — turns a raw pulse (BVP) waveform into everything the three
dashboard graphs need: the waveform itself, detected beats, clean
beat-to-beat intervals (IBI) and instantaneous heart rate.

Pure numpy, no TensorFlow / JAX, so it is safe to import from main.py,
openrppg_pipeline.py and live_server.py.

Why this exists: the old dashboard drew a fake sine placeholder and a
straight-through "history" line. These graphs are drawn from the REAL
signal instead, so they look like what the pulse actually did.
"""

import numpy as np

MIN_IBI_MS = 400      # 150 BPM
MAX_IBI_MS = 1500     # 40 BPM
IBI_TOLERANCE = 0.25  # an interval >25% away from its neighbours' median is rejected


def _ma(x, n):
    """Moving average, same length as x, edges padded (no edge dip)."""
    n = max(1, int(n))
    if n <= 1:
        return x.copy()
    padded = np.pad(x, (n // 2, n - 1 - n // 2), mode='edge')
    return np.convolve(padded, np.ones(n) / n, mode='valid')


def _clean_signal(bvp, fps):
    x = np.asarray(bvp, dtype=float).ravel()
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        x[bad] = np.nanmean(x[~bad])
    x = x - _ma(x, round(1.5 * fps))            # remove slow drift
    sd = x.std()
    if sd < 1e-9:
        return None
    x = x / sd                                  # z-score
    return _ma(x, round(0.08 * fps))            # tiny smoothing (no-op at low fps)


def _dominant_freq_hz(x, fps):
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(max(n, 8) * 4)))
    spec = np.abs(np.fft.rfft(x * np.hanning(n), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fps)
    band = (freqs >= 0.75) & (freqs <= 2.5)     # 45-150 BPM
    if not band.any():
        return 1.2
    return float(freqs[band][np.argmax(spec[band])])


def detect_beats(x, fps):
    """Return beat times in seconds (sub-sample accurate) from a cleaned signal."""
    L = len(x)
    f0 = _dominant_freq_hz(x, fps)
    min_d = max(2, int(round(0.6 * fps / f0)))  # peaks closer than 60% of a beat are the same beat

    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 0))[0] + 1
    if len(cand) == 0:
        return np.array([])
    blocked = np.zeros(L, dtype=bool)
    keep = []
    for i in cand[np.argsort(-x[cand])]:        # strongest first
        if blocked[i]:
            continue
        keep.append(i)
        blocked[max(0, i - min_d + 1): i + min_d] = True
    keep = np.array(sorted(keep))

    # drop weak bumps (dicrotic notch etc.) that are far smaller than the real beats
    heights = x[keep]
    keep = keep[heights >= 0.4 * np.median(heights)]

    # parabolic refinement: at 15-30 fps one sample is 33-67 ms, which would ruin HRV
    times = []
    for i in keep:
        if 0 < i < L - 1:
            a, b, c = x[i - 1], x[i], x[i + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den != 0 else 0.0
            d = float(np.clip(d, -0.5, 0.5))
        else:
            d = 0.0
        times.append((i + d) / fps)
    return np.array(times)


def _validate_ibi(ibi):
    valid = (ibi >= MIN_IBI_MS) & (ibi <= MAX_IBI_MS)
    ok = ibi[valid]
    out = valid.copy()
    for k in range(len(ibi)):
        if not valid[k]:
            continue
        lo, hi = max(0, k - 3), min(len(ibi), k + 4)
        neigh = ibi[lo:hi][valid[lo:hi]]
        med = np.median(neigh) if len(neigh) else np.median(ok)
        out[k] = abs(ibi[k] - med) <= IBI_TOLERANCE * med
    return out


def build_graph(bvp, fps, max_samples=2400):
    """
    bvp: sequence of samples, fps: its sampling rate.
    Returns a JSON-safe dict, or None if the signal is too short / flat.
    """
    if bvp is None or fps is None or fps <= 0:
        return None
    x = _clean_signal(bvp, fps)
    if x is None or len(x) < fps * 5:
        return None

    peaks_t = detect_beats(x, fps)
    t_axis = np.arange(len(x)) / fps
    peak_y = np.interp(peaks_t, t_axis, x) if len(peaks_t) else np.array([])

    beat_t, ibi_out, hr_out, hr_smooth_out = [], [], [], []
    n_valid = 0
    if len(peaks_t) >= 3:
        ibi = np.diff(peaks_t) * 1000.0
        valid = _validate_ibi(ibi)
        hr = 60000.0 / ibi
        vidx = np.where(valid)[0]
        n_valid = int(len(vidx))
        smooth = {}
        for pos, k in enumerate(vidx):          # centred 5-beat mean over VALID beats only
            lo, hi = max(0, pos - 2), min(len(vidx), pos + 3)
            smooth[k] = float(np.mean(hr[vidx[lo:hi]]))
        for k in range(len(ibi)):
            beat_t.append(round(float(peaks_t[k + 1]), 3))
            if valid[k]:
                ibi_out.append(round(float(ibi[k]), 1))
                hr_out.append(round(float(hr[k]), 1))
                hr_smooth_out.append(round(smooth[k], 1))
            else:
                ibi_out.append(None)
                hr_out.append(None)
                hr_smooth_out.append(None)

    stride = int(np.ceil(len(x) / max_samples)) if len(x) > max_samples else 1
    x_out = x[::stride]
    return {
        'bvp': [round(float(v), 3) for v in x_out],
        'bvp_fps': round(fps / stride, 3),
        'peak_t': [round(float(t), 3) for t in peaks_t],
        'peak_y': [round(float(v), 3) for v in peak_y],
        'beat_t': beat_t,
        'ibi_ms': ibi_out,
        'hr_inst': hr_out,
        'hr_smooth': hr_smooth_out,
        'n_beats': int(len(peaks_t)),
        'n_valid_ibi': n_valid,
        'duration_s': round(len(x) / fps, 2),
    }