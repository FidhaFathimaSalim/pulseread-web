"""
reference_ranges.py — "is my number high or low?" for a normal person.

Pure Python. Used by report_builder.py. The browser has a copy of the SAME numbers in
frontend/js/ranges.js — if you change a threshold here, change it there too.

WHERE THE NUMBERS COME FROM (be honest about this in your report)
* Heart rate 60-100 BPM and breathing 12-20 per minute are the usual textbook resting ranges
  for healthy adults.
* HRV has no single "normal": it changes with age, fitness, sleep and the way it is measured.
  The HRV bands below are this app's own simple guide (they are the same cut-offs that
  classify_stress() in openrppg_pipeline.py uses), NOT clinical limits.
* Signal-quality bands match how the app labels reliability (>=75 high, >=50 medium).

General wellness information only, not a diagnosis.
"""

# kind -> meaning (colours are chosen in the report / the browser)
#   ok = typical, good = favourable, warn = worth noticing, bad = poor/low, low = below typical, high = above typical
RANGES = {
    'heart_rate': {
        'label': 'Heart rate', 'unit': 'BPM', 'typical': '60-100 BPM',
        'meaning': 'How many times your heart beats in one minute, at rest.',
        'bands': [(None, 60, 'Below typical', 'low'), (60, 100, 'Typical', 'ok'), (100, None, 'Above typical', 'high')],
        'note': 'Very fit people and athletes can normally be below 60.',
    },
    'rmssd': {
        'label': 'HRV (RMSSD)', 'unit': 'ms', 'typical': '35-70 ms',
        'meaning': 'How much the time between heartbeats changes from beat to beat. Higher usually means a more relaxed, recovered body.',
        'bands': [(None, 15, 'Very low', 'bad'), (15, 35, 'Low', 'warn'), (35, 70, 'Typical', 'ok'), (70, None, 'High', 'good')],
        'note': 'Depends a lot on age and fitness. Compare with your own usual values. Video-based HRV is rough.',
    },
    'sdnn': {
        'label': 'HRV (SDNN)', 'unit': 'ms', 'typical': '40-80 ms',
        'meaning': 'The overall spread of the gaps between heartbeats during the recording.',
        'bands': [(None, 15, 'Very low', 'bad'), (15, 40, 'Low', 'warn'), (40, 80, 'Typical', 'ok'), (80, None, 'High', 'good')],
        'note': 'Same caution as RMSSD.',
    },
    'mean_rr': {
        'label': 'Mean RR', 'unit': 'ms', 'typical': '600-1000 ms',
        'meaning': 'The average time between two heartbeats (800 ms is about 75 BPM).',
        'bands': [(None, 600, 'Short (fast heart)', 'high'), (600, 1000, 'Typical', 'ok'), (1000, None, 'Long (slow heart)', 'low')],
        'note': '',
    },
    'breathing': {
        'label': 'Breathing rate', 'unit': '/min', 'typical': '12-20 /min',
        'meaning': 'How many breaths you take in one minute, at rest.',
        'bands': [(None, 12, 'Slower than typical', 'low'), (12, 20, 'Typical', 'ok'), (20, None, 'Faster than typical', 'high')],
        'note': 'Slow, paced breathing (about 6 per minute) is normal during relaxation exercises.',
    },
    'quality': {
        'label': 'Signal quality', 'unit': '%', 'typical': 'above 75 % is good',
        'meaning': 'How clean the pulse signal in the video was. Low quality makes every other number less reliable.',
        'bands': [(None, 50, 'Poor', 'bad'), (50, 75, 'Fair', 'warn'), (75, None, 'Good', 'ok')],
        'note': 'Face the light, keep your face still and fully in view.',
    },
}

STRESS_LEVELS = 'Very Low, Low, Moderate, High'
STRESS_NOTE = ('Worked out from HRV: lower HRV gives a higher stress level. '
               'It is an estimate, not a diagnosis.')


def assess(metric, value):
    """
    Returns {'status': 'Typical', 'kind': 'ok', 'typical': '60-100 BPM'} for a value,
    or status None when there is no value.
    """
    info = RANGES[metric]
    out = {'status': None, 'kind': None, 'typical': info['typical']}
    try:
        v = float(value)
    except (TypeError, ValueError):
        return out
    for lo, hi, status, kind in info['bands']:
        if (lo is None or v >= lo) and (hi is None or v < hi):
            out['status'], out['kind'] = status, kind
            break
    return out
