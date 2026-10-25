"""
challenge.py — Physiological Challenge Mode (baseline -> challenge -> recovery).

Pure Python (no TensorFlow / JAX / numpy), so it runs inside live_server.py.

IDEA (beginner version)
-----------------------
1. BASELINE  : sit still and relax  -> your resting numbers
2. CHALLENGE : do a small mental task (or paced slow breathing)
3. RECOVERY  : relax again          -> how fast do you come back?

live_server.py keeps analysing the video as usual. This module only
  * remembers when each phase started / ended (client clock),
  * stamps every finished analysis window with the phase it belongs to,
  * and at the end turns the stamped windows into a report.

A window is only counted for a phase if at least 75 % of it lies inside that
phase (after a short "settle" time). Windows that straddle two phases too much
would mix them up, so they are kept for the timeline chart but not for the numbers.

This is wellness / research information, NOT a medical test.
"""

import statistics

MIN_QUALITY_PCT = 55.0     # windows with a weaker signal are ignored in the phase numbers (was 40)
MIN_WINDOWS = 3            # fewer good windows than this -> phase marked "insufficient"; also needed in
                           #   baseline, challenge AND recovery before any recovery number is shown
MIN_HRV_WINDOWS = 2        # HRV needs a stronger signal than heart rate, so fewer windows have it;
                           #   a phase needs at least this many with HRV before its RMSSD is shown / compared
WINDOW_OVERLAP_MIN = 0.75  # a window counts for a phase when at least this much of it lies inside that phase (after its
                           #   settle time). Was "the whole window": with one window every ~25 s that threw away most of them.
HR_NOISE_FLOOR_BPM = 2.0   # smaller HR changes are treated as "no measurable change"
HRV_NOISE_FLOOR_PCT = 25.0 # RMSSD jumps ~20-30 % between neighbouring windows even at rest (was 10)

PROTOCOLS = {
    'cognitive': {
        'label': 'Mental arithmetic stress test',
        'expect': 'stress',
        'phases': [
            {'name': 'baseline', 'seconds': 100, 'settle': 10,
             'title': 'Baseline', 'instruction': 'Sit still, relax your shoulders and breathe normally.'},
            {'name': 'challenge', 'seconds': 100, 'settle': 3,
             'title': 'Challenge', 'instruction': 'Count backwards from 1000 in steps of 7 (1000, 993, 986 ...). '
                                                  'If you make a mistake, start again. Keep your face toward the camera.'},
            {'name': 'recovery', 'seconds': 100, 'settle': 5,
             'title': 'Recovery', 'instruction': 'Stop counting. Relax and breathe normally.'},
        ],
        'note': 'A small mental task. Heart rate usually rises a little and HRV usually drops, then both drift back.',
    },
    'paced': {
        'label': 'Slow paced breathing',
        'expect': 'relax',
        'phases': [
            {'name': 'baseline', 'seconds': 100, 'settle': 10,
             'title': 'Baseline', 'instruction': 'Sit still and breathe normally.'},
            {'name': 'challenge', 'seconds': 100, 'settle': 5,
             'title': 'Paced breathing',
             'instruction': 'Breathe in for 5 seconds, out for 5 seconds (6 breaths per minute). Follow the pacer.'},
            {'name': 'recovery', 'seconds': 100, 'settle': 5,
             'title': 'Recovery', 'instruction': 'Go back to normal, relaxed breathing.'},
        ],
        'note': 'Slow breathing usually raises HRV and brings the breathing rate close to 6 per minute.',
    },
}

SAFETY_NOTE = ('Wellness experiment, not a medical test. Keep the task gentle, do not push physical exertion '
               '(movement also ruins the video signal), and stop any time you feel unwell.')


def protocol_public(key):
    """What the browser needs to run the timer UI."""
    p = PROTOCOLS[key]
    return {'key': key, 'label': p['label'], 'note': p['note'], 'safety': SAFETY_NOTE,
            'phases': [{k: ph[k] for k in ('name', 'seconds', 'title', 'instruction')} for ph in p['phases']]}


def list_protocols():
    return [protocol_public(k) for k in PROTOCOLS]


def _med(values):
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def _sd(values):
    v = [x for x in values if x is not None]
    return statistics.pstdev(v) if len(v) >= 2 else None


class ChallengeRecorder:
    """One challenge run. All timestamps are seconds on the CLIENT clock (same clock as the video frames)."""

    def __init__(self, protocol_key, start_ts):
        if protocol_key not in PROTOCOLS:
            raise ValueError(f'unknown protocol {protocol_key!r}')
        self.key = protocol_key
        self.protocol = PROTOCOLS[protocol_key]
        self.start_ts = float(start_ts)
        self.end_ts = None
        self.cancelled = False
        self.phase_log = []          # [{'name', 'start', 'end', 'settle'}]
        self.samples = {}            # subject_id -> list of sample dicts

    # ---- lifecycle -------------------------------------------------
    @property
    def active(self):
        return self.end_ts is None and not self.cancelled

    def begin_phase(self, name, ts):
        ts = float(ts)
        if self.phase_log and self.phase_log[-1]['end'] is None:
            self.phase_log[-1]['end'] = ts
        settle = next((p['settle'] for p in self.protocol['phases'] if p['name'] == name), 0)
        self.phase_log.append({'name': name, 'start': ts, 'end': None, 'settle': settle})

    def finish(self, ts):
        ts = float(ts)
        if self.phase_log and self.phase_log[-1]['end'] is None:
            self.phase_log[-1]['end'] = ts
        self.end_ts = ts

    def cancel(self):
        self.cancelled = True

    # ---- stamping --------------------------------------------------
    def phase_for_window(self, t0, t1):
        """Phase name if at least WINDOW_OVERLAP_MIN of the window [t0, t1] lies inside one phase
        (after its settle time), else None."""
        dur = max(float(t1) - float(t0), 1e-6)
        best, best_ov = None, 0.0
        for ph in self.phase_log:
            end = ph['end'] if ph['end'] is not None else float('inf')
            ov = min(t1, end) - max(t0, ph['start'] + ph['settle'])
            if ov > best_ov:
                best, best_ov = ph['name'], ov
        return best if best is not None and best_ov / dur >= WINDOW_OVERLAP_MIN - 1e-9 else None

    def add_sample(self, subject_id, t0, t1, result):
        """
        result: the dict live_server produced for this window (heart_rate_bpm, hrv_*, resp, ...).
        Returns the stamped sample (also stored).
        """
        resp = result.get('resp') or {}
        sample = {
            't': round(float(t1) - self.start_ts, 1),            # seconds since the challenge started
            'phase': self.phase_for_window(t0, t1),
            'hr': result.get('heart_rate_bpm'),
            'rmssd': result.get('hrv_rmssd_ms') if result.get('hrv_available') else None,
            'sdnn': result.get('hrv_sdnn_ms') if result.get('hrv_available') else None,
            'resp': resp.get('rate_bpm') if resp.get('confidence') in ('High', 'Medium') else None,
            'quality': result.get('signal_quality_pct'),
        }
        self.samples.setdefault(subject_id, []).append(sample)
        return sample

    # ---- report ----------------------------------------------------
    def report(self, labels=None):
        labels = labels or {}
        people = []
        for sid, samples in sorted(self.samples.items()):
            people.append({
                'id': sid,
                'label': labels.get(sid, f'Person {sid}'),
                **summarise(samples, self.protocol['expect']),
                'timeline': samples,
            })
        return {
            'protocol': protocol_public(self.key),
            'phase_bounds': [{'name': p['name'], 'start': round(p['start'] - self.start_ts, 1),
                              'end': None if p['end'] is None else round(p['end'] - self.start_ts, 1)}
                             for p in self.phase_log],
            'people': people,
            'min_quality_pct': MIN_QUALITY_PCT,
            'min_windows': MIN_WINDOWS,
            'min_hrv_windows': MIN_HRV_WINDOWS,
            'safety': SAFETY_NOTE,
        }


# ---------------------------------------------------------------- statistics

def _phase_stats(samples, phase):
    ok = [s for s in samples
          if s['phase'] == phase and s['hr'] is not None
          and s['quality'] is not None and s['quality'] >= MIN_QUALITY_PCT]
    return {
        'n': len(ok),
        'hr': _med([s['hr'] for s in ok]),
        'hr_sd': _sd([s['hr'] for s in ok]),
        'rmssd': _med([s['rmssd'] for s in ok]),
        'rmssd_n': sum(1 for s in ok if s['rmssd'] is not None),
        'sdnn': _med([s['sdnn'] for s in ok]),
        'resp': _med([s['resp'] for s in ok]),
        'resp_n': sum(1 for s in ok if s['resp'] is not None),
        'quality': _med([s['quality'] for s in ok]),
    }


def _r(x, nd=1):
    return None if x is None else round(x, nd)


def _recovered_pct(first, peak, last):
    """How much of the change (first -> peak) came back by `last`, as 0-100 %.
    Going further the same way counts as 0 %; coming back past the start counts as 100 %."""
    change = peak - first
    if not change:
        return None
    return max(0.0, min(100.0, (peak - last) / change * 100.0))


def summarise(samples, expect='stress'):
    """Per-phase medians + reactivity / recovery numbers + a cautious plain-language reading.
    A number that the report itself calls unreliable is never used in a comparison or in the sentence."""
    phases = {n: _phase_stats(samples, n) for n in ('baseline', 'challenge', 'recovery')}
    b, c, r = phases['baseline'], phases['challenge'], phases['recovery']
    ok = {k: v['n'] >= MIN_WINDOWS for k, v in phases.items()}                      # heart rate usable
    hrv_ok = {k: v['rmssd_n'] >= MIN_HRV_WINDOWS for k, v in phases.items()}        # RMSSD usable
    hrv_rec_ok = {k: v['rmssd_n'] >= MIN_WINDOWS for k, v in phases.items()}        # RMSSD usable for recovery
    notes = []
    for name, st in phases.items():
        label = name.capitalize()
        if not ok[name]:
            notes.append(f'{label}: only {st["n"]} good analysis window(s) (signal quality {MIN_QUALITY_PCT:.0f}% or more; '
                         f'{MIN_WINDOWS} needed) - numbers for this phase are not reliable and are left out of the '
                         f'comparison (good light and stillness help).')
        elif not hrv_ok[name]:
            notes.append(f'{label}: HRV could not be measured reliably (HRV needs a stronger signal than heart rate) - '
                         f'no HRV comparison for this phase.')

    if not all(ok.values()):
        ts = sorted(s['t'] for s in samples if s.get('t') is not None)
        gaps = [b2 - a for a, b2 in zip(ts, ts[1:])]
        if gaps:
            g = statistics.median(gaps)
            notes.append(f'This run produced one analysis window about every {g:.0f} s, so a phase needs roughly '
                         f'{MIN_WINDOWS * g:.0f} s to collect {MIN_WINDOWS} good ones. If this is much slower than ~10-15 s, '
                         f'the computer is struggling: close other apps/tabs (the live_server console shows "inference took ...s").')

    m = {}
    # --- heart rate
    if ok['baseline'] and ok['challenge'] and b['hr'] is not None and c['hr'] is not None:
        m['hr_change_bpm'] = _r(c['hr'] - b['hr'])
        m['hr_change_pct'] = _r((c['hr'] - b['hr']) / b['hr'] * 100)
        noise = max(HR_NOISE_FLOOR_BPM, 2 * (b['hr_sd'] or 0))
        m['hr_change_meaningful'] = abs(c['hr'] - b['hr']) >= noise
    # --- HRV (RMSSD)
    if hrv_ok['baseline'] and hrv_ok['challenge'] and b['rmssd'] and c['rmssd'] is not None:
        m['rmssd_change_pct'] = _r((c['rmssd'] - b['rmssd']) / b['rmssd'] * 100)
        m['rmssd_change_meaningful'] = abs(m['rmssd_change_pct']) >= HRV_NOISE_FLOOR_PCT
    # --- breathing
    if ok['baseline'] and ok['challenge'] and b['resp'] is not None and c['resp'] is not None:
        m['resp_change_bpm'] = _r(c['resp'] - b['resp'])
    # --- recovery: only with enough good windows in ALL three phases, only if there WAS a meaningful change,
    #     and always between 0 and 100 %
    if ok['baseline'] and ok['challenge'] and ok['recovery'] and m.get('hr_change_meaningful') and r['hr'] is not None:
        v = _recovered_pct(b['hr'], c['hr'], r['hr'])
        if v is not None:
            m['hr_recovery_pct'] = _r(v, 0)
    if (hrv_rec_ok['baseline'] and hrv_rec_ok['challenge'] and hrv_rec_ok['recovery']
            and m.get('rmssd_change_meaningful') and r['rmssd'] is not None):
        v = _recovered_pct(b['rmssd'], c['rmssd'], r['rmssd'])
        if v is not None:
            m['rmssd_recovery_pct'] = _r(v, 0)
    if m.get('hr_change_meaningful') and 'hr_recovery_pct' not in m and ok['baseline'] and ok['challenge']:
        notes.append(f'Recovery numbers are hidden: they need at least {MIN_WINDOWS} good windows in baseline, '
                     f'challenge and recovery.')

    reading = _reading(m, expect)

    out_phases = {}
    for k, v in phases.items():
        d = {kk: _r(vv, 1) if isinstance(vv, float) else vv for kk, vv in v.items()}
        d['reliable'] = ok[k]
        if not hrv_ok[k]:                       # do not show an RMSSD / SDNN that rests on fewer than MIN_HRV_WINDOWS windows
            d['rmssd'] = None
            d['sdnn'] = None
        out_phases[k] = d
    return {'phases': out_phases, 'metrics': m, 'reading': reading, 'notes': notes}


def _reading(m, expect):
    """Descriptive sentence(s) only. These rules are this app's heuristics, not clinical criteria."""
    hr_up = m.get('hr_change_bpm')
    hrv = m.get('rmssd_change_pct')
    hr_ok = m.get('hr_change_meaningful')
    hrv_ok = m.get('rmssd_change_meaningful')
    if hr_up is None and hrv is None:
        return ('Not enough reliable data to describe a response - too few good analysis windows '
                f'(signal quality {MIN_QUALITY_PCT:.0f}% or more) in baseline and challenge.')

    parts = []
    if hr_ok is not None:
        parts.append(f'Heart rate {"rose" if hr_up > 0 else "fell"} by {abs(hr_up):.0f} BPM during the challenge'
                     if hr_ok else 'Heart rate showed no clear change during the challenge')
    if hrv_ok is not None:
        parts.append(f'HRV (RMSSD) {"rose" if hrv > 0 else "fell"} by about {abs(hrv):.0f}%'
                     if hrv_ok else 'HRV showed no clear change')
    text = '; '.join(parts) + '.'

    if expect == 'stress' and hr_ok and hrv_ok:
        if hr_up > 0 and hrv < 0:
            text += ' This is the usual pattern for a mild mental stressor (faster heart, lower HRV).'
        else:
            text += ' This differs from the usual pattern (faster heart, lower HRV) - could be real, or just measurement noise.'
    elif expect == 'relax' and hrv_ok:
        text += (' HRV went up, which is what slow paced breathing usually does.' if hrv > 0
                 else ' HRV did not go up, so the paced breathing may not have been followed or the signal was noisy.')
    elif expect == 'relax' and hrv is None:
        text += ' HRV could not be measured reliably, so the effect of slow breathing on HRV is unknown.'
    elif not hr_ok and not hrv_ok:
        text += ' The changes were within normal measurement noise.'

    if 'hr_recovery_pct' in m:
        text += f' Heart rate recovered about {m["hr_recovery_pct"]:.0f}% of the way back to baseline.'
    if 'rmssd_recovery_pct' in m:
        text += f' HRV recovered about {m["rmssd_recovery_pct"]:.0f}% of the way back to baseline.'
    return text
