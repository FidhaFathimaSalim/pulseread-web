"""
respiration.py — breathing-rate estimation from the SAME pulse (BVP) signal
that PulseRead / open-rppg already extract. No extra camera work.

Pure numpy (+ graph_data.py), so it is safe to import in BOTH processes
(TensorFlow side: main.py, and JAX side: openrppg_pipeline.py / live_server.py).

HOW IT WORKS (beginner version)
-------------------------------
Breathing leaves three small "fingerprints" on a pulse signal:

  FM  (frequency modulation)  - your heart speeds up when you breathe in and
                                slows down when you breathe out (this is
                                called RSA). So the beat-to-beat interval goes
                                up and down once per breath.
  AM  (amplitude modulation)  - the height of each pulse wave grows and shrinks
                                with breathing.
  BW  (baseline wander)       - the whole signal drifts slowly up and down
                                with breathing.

We pull each fingerprint out as a slow signal, look for the strongest
rhythm between 6 and 30 breaths/min in each, then COMBINE the three
(adding their normalised spectra). If the methods agree we trust the
answer more; if they disagree we say so (confidence = Low).

HONEST LIMITS
-------------
* Needs >= ~25-30 s of signal (at least 3 breaths at the slow end).
* Resolution of a 30 s window is about 2 breaths/min.
* Heavy head motion, bad light, or a weak pulse -> Low confidence.
* Validate it yourself (paced breathing with a metronome) before you quote
  an accuracy number in your dissertation. See INTEGRATION.md.
"""

import numpy as np

import graph_data

RESP_LO_HZ = 0.10        # 6 breaths/min
RESP_HI_HZ = 0.50        # 30 breaths/min
SERIES_FS = 4.0          # all slow series are resampled to 4 Hz
MIN_SECONDS = 25.0       # shorter signals are refused (not enough breaths)
MIN_BEATS = 15
AGREE_TOL_BPM = 2.5      # two methods "agree" if their rates differ by less than this


# ------------------------------------------------------------------ helpers

def _fill_nan(x):
    x = np.asarray(x, dtype=float).ravel().copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        x[bad] = np.mean(x[~bad])
    return x


def _to_grid(t, y, t0, t1, fs=SERIES_FS):
    """Resample an irregular series (t, y) onto an even grid of fs Hz."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 4 or t[-1] - t[0] < 10:
        return None, None
    grid = np.arange(max(t0, t[0]), min(t1, t[-1]), 1.0 / fs)
    if len(grid) < int(fs * 10):
        return None, None
    return grid, np.interp(grid, t, y)


def _detrend(y):
    n = len(y)
    k = np.arange(n)
    a, b = np.polyfit(k, y, 1)
    return y - (a * k + b)


def _spectrum(y, fs=SERIES_FS):
    """Hann-windowed, zero-padded power spectrum restricted to the breathing band."""
    y = _detrend(np.asarray(y, dtype=float))
    n = len(y)
    nfft = int(2 ** np.ceil(np.log2(n * 16)))
    power = np.abs(np.fft.rfft(y * np.hanning(n), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs >= RESP_LO_HZ) & (freqs <= RESP_HI_HZ)
    return freqs[band], power[band]


def _peak_info(freqs, power, n_samples, fs=SERIES_FS):
    """Peak frequency + 'peak ratio' = share of band power inside the peak's main lobe."""
    total = power.sum()
    if total <= 0:
        return None
    k = int(np.argmax(power))
    f_peak = float(freqs[k])
    half = 2.0 * fs / n_samples                     # Hann main-lobe half width in Hz
    near = np.abs(freqs - f_peak) <= half
    ratio = float(power[near].sum() / total)
    return {'f_hz': f_peak, 'rate_bpm': f_peak * 60.0, 'peak_ratio': ratio}


def _bandpass_fft(y, fs=SERIES_FS):
    """Full 0.1-0.5 Hz band-pass (deliberately NOT narrowed: a narrow filter makes
    pure noise look like a beautiful clean breathing wave, which would be dishonest)."""
    y = _detrend(np.asarray(y, dtype=float))
    n = len(y)
    spec = np.fft.rfft(y * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec[(freqs < RESP_LO_HZ) | (freqs > RESP_HI_HZ)] = 0
    out = np.fft.irfft(spec, n)
    sd = out.std()
    return out / sd if sd > 1e-9 else out


# ------------------------------------------------------------------ main API

def estimate_respiration(bvp, fps, min_seconds=MIN_SECONDS):
    """
    bvp : raw pulse samples (as produced by the model, NOT z-scored / drift-removed)
    fps : sampling rate of bvp
    Returns a JSON-safe dict (never raises):
      rate_bpm       breaths per minute, or None
      confidence     'High' | 'Medium' | 'Low' | 'Unavailable'
      reason         short text explaining a None / Low result
      methods        {'FM': bpm|None, 'AM': bpm|None, 'BW': bpm|None}
      n_agree        how many methods are within AGREE_TOL_BPM of the final rate
      wave           {'t': [...], 'y': [...]} the breathing waveform (4 Hz), or None
      duration_s
    """
    out = {'rate_bpm': None, 'confidence': 'Unavailable', 'reason': '',
           'methods': {'FM': None, 'AM': None, 'BW': None},
           'n_agree': 0, 'wave': None, 'duration_s': None}
    try:
        return _estimate(bvp, fps, min_seconds, out)
    except Exception as e:  # noqa: BLE001 - respiration is optional, never break the pipeline
        out['reason'] = f'respiration failed: {e!r}'
        return out


def _estimate(bvp, fps, min_seconds, out):
    if bvp is None or fps is None or fps <= 0:
        out['reason'] = 'no pulse signal'
        return out
    raw = _fill_nan(bvp)
    if raw is None:
        out['reason'] = 'no pulse signal'
        return out
    duration = len(raw) / fps
    out['duration_s'] = round(duration, 1)
    if duration < min_seconds:
        out['reason'] = f'needs at least {min_seconds:.0f} s of signal (have {duration:.0f} s)'
        return out

    t_axis = np.arange(len(raw)) / fps
    t_lo, t_hi = 0.0, t_axis[-1]

    # --- beats, using the same cleaned signal + detector as the dashboard graphs
    x = graph_data._clean_signal(raw, fps)
    if x is None:
        out['reason'] = 'signal is flat'
        return out
    peaks_t = graph_data.detect_beats(x, fps)

    series = {}      # name -> (grid_t, grid_y)

    if len(peaks_t) >= MIN_BEATS:
        # FM: beat-to-beat interval series
        ibi = np.diff(peaks_t) * 1000.0
        valid = graph_data._validate_ibi(ibi)
        if valid.mean() >= 0.6 and valid.sum() >= MIN_BEATS - 2:
            g = _to_grid(peaks_t[1:][valid], ibi[valid], t_lo, t_hi)
            if g[0] is not None:
                series['FM'] = g

        # AM: pulse amplitude (peak minus the lowest point since the previous beat)
        idx = np.round(peaks_t * fps).astype(int)
        idx = np.clip(idx, 0, len(x) - 1)
        amps, amp_t = [], []
        for k in range(1, len(idx)):
            lo, hi = idx[k - 1], idx[k]
            if hi - lo < 2:
                continue
            amps.append(x[hi] - x[lo:hi + 1].min())
            amp_t.append(peaks_t[k])
        amps = np.asarray(amps)
        if len(amps) >= MIN_BEATS - 2:
            med = np.median(amps)
            mad = np.median(np.abs(amps - med)) + 1e-9
            keep = np.abs(amps - med) <= 4.0 * 1.4826 * mad
            g = _to_grid(np.asarray(amp_t)[keep], amps[keep], t_lo, t_hi)
            if g[0] is not None:
                series['AM'] = g

    # BW: slow baseline of the raw signal (moving average about one pulse long)
    win = max(2, int(round(1.0 * fps)))
    padded = np.pad(raw, (win // 2, win - 1 - win // 2), mode='edge')
    bw = np.convolve(padded, np.ones(win) / win, mode='valid')
    g = _to_grid(t_axis, bw, t_lo, t_hi)
    if g[0] is not None:
        series['BW'] = g

    if not series:
        out['reason'] = 'could not extract any breathing-related series'
        return out

    # --- spectrum of every series, then fuse
    per = {}
    fused = None
    freqs_ref = None
    for name, (gt, gy) in series.items():
        if np.std(gy) < 1e-9:
            continue
        freqs, power = _spectrum(gy)
        info = _peak_info(freqs, power, len(gy))
        if info is None:
            continue
        per[name] = info
        out['methods'][name] = round(info['rate_bpm'], 1)
        norm = power / power.sum()
        fused = norm if fused is None else fused + norm
        freqs_ref = freqs

    if fused is None:
        out['reason'] = 'no usable spectrum'
        return out

    fused = fused / len(per)
    k = int(np.argmax(fused))
    f_final = float(freqs_ref[k])
    rate = f_final * 60.0
    half = 2.0 * SERIES_FS / min(len(series[n][1]) for n in per)
    ratio = float(fused[np.abs(freqs_ref - f_final) <= half].sum() / fused.sum())

    n_agree = sum(1 for n in per if abs(per[n]['rate_bpm'] - rate) <= AGREE_TOL_BPM)
    at_edge = (f_final - RESP_LO_HZ < 0.01) or (RESP_HI_HZ - f_final < 0.01)

    # Thresholds calibrated on synthetic signals: pure noise gives peak_ratio <= ~0.45,
    # real breathing gives >= ~0.6. Re-check them with paced-breathing recordings of your own.
    if at_edge:
        conf, why = 'Low', 'strongest rhythm sits at the edge of the breathing band (probably drift or noise)'
    elif n_agree >= 2 and ratio >= 0.55:
        conf, why = 'High', ''
    elif ratio >= 0.45:
        conf, why = 'Medium', ''
    else:
        conf, why = 'Low', 'no clear breathing rhythm (weak signal or methods disagree)'

    out.update({'rate_bpm': round(rate, 1), 'confidence': conf, 'reason': why, 'n_agree': int(n_agree),
                'peak_ratio': round(ratio, 2)})

    # --- waveform to draw: the method that agrees best with the final rate
    order = [n for n in ('FM', 'AM', 'BW') if n in per]
    order.sort(key=lambda n: (abs(per[n]['rate_bpm'] - rate), -per[n]['peak_ratio']))
    best = order[0]
    gt, gy = series[best]
    wave = _bandpass_fft(gy)
    out['wave_source'] = best
    out['wave'] = {'t': [round(float(v), 2) for v in gt], 'y': [round(float(v), 3) for v in wave]}
    return out
