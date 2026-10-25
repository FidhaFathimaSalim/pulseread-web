"""
recommendations.py — maps a processed video's BPM + HRV (RMSSD) into one
of four physiological states, each with breathing/exercise/sleep/nutrition
guidance. Runs in-process wherever main.py runs (no TF/JAX dependency —
pure Python, safe to import from either backend).

IMPORTANT: this is general wellness information, not medical advice. It
is not a diagnosis and should not be treated as one. Always say so
wherever this is displayed.
"""

STATES = {
    'High_Stress_Low_HRV': {
        'label': 'High Stress & Low HRV',
        'bpm_range': (90, None),       # BPM > 90
        'hrv_range': (None, 35),       # HRV < 35 ms
        'nervous_system': 'Hyper-aroused, acute "fight or flight" response active. Elevated cortisol likely.',
        'breathing': ('The Physiological Sigh. Take two quick inhales through the nose (one deep breath, '
                      'followed immediately by a sharp top-off inhale), then a long, slow sigh out through '
                      'the mouth. Repeat 5 times to help lower heart rate.'),
        'exercise': ('Strictly restorative mobility today — skip intense workouts. Try a 20-minute slow, '
                     'mindful walk or 10 minutes of gentle yoga (child\'s pose, cat-cow) to release tension '
                     'without adding more stress load.'),
        'sleep': ('Thermodynamic reset & PMR. Lower your bedroom temperature to around 18°C (65°F). Before '
                  'sleep, try Progressive Muscle Relaxation — tense and release each muscle group from toe '
                  'to head.'),
        'nutrition': ('Caffeine curfew & hydration. Consider stopping caffeine for the rest of the day. '
                      'Drink water with electrolytes or a pinch of sea salt — dehydration can mimic anxiety '
                      'symptoms in the body.'),
    },
    'Low_Stress_High_HRV': {
        'label': 'Low Stress & High HRV',
        'bpm_range': (60, 80),
        'hrv_range': (55, None),
        'nervous_system': 'Resilient and balanced — a good state for cognitive or physical load ("flow state").',
        'breathing': ('Resonance / coherent breathing. Inhale for 5 seconds, exhale for 5 seconds. Keep this '
                      'smooth rhythm for 2 minutes to support focused calm.'),
        'exercise': ('Your body looks primed for a harder session — a HIIT workout, heavier lifting, or a '
                     'challenging interval run could suit today well.'),
        'sleep': ('Standard wind-down. Stick to your usual routine — dim lights an hour before bed, avoid '
                  'heavy meals within 3 hours of sleep.'),
        'nutrition': ('Good window for focused work. Complex carbohydrates, healthy fats, and some protein '
                      'can help sustain energy through demanding tasks.'),
    },
    'Exhausted_Low_HRV': {
        'label': 'Exhaustion & Low HRV',
        'bpm_range': (None, 60),
        'hrv_range': (None, 35),
        'nervous_system': 'Possible parasympathetic burnout or physical exhaustion — limited capacity for healthy variation right now.',
        'breathing': ('Box breathing. Inhale for 4 seconds, hold 4, exhale 4, hold 4. Repeat for a few '
                      'minutes to build steady focus without extra strain.'),
        'exercise': ('Zone 2 / active recovery only — skip lifting or running. A light, steady walk or easy '
                     'cycling, at a pace where conversation stays easy, fits better today.'),
        'sleep': ('Consider a period of deep rest — a guided NSDR or Yoga Nidra session in the afternoon or '
                  'before bed may help support recovery.'),
        'nutrition': ('Nutrient-dense foods rich in magnesium (spinach, pumpkin seeds) may support muscle and '
                      'nervous system recovery — check with a professional before adding any supplement.'),
    },
    'Calm_High_HRV': {
        'label': 'Calm / Lethargic & High HRV',
        'bpm_range': (None, 60),
        'hrv_range': (55, None),
        'nervous_system': 'Deeply relaxed, possibly under-aroused — good recovery, but alertness may be lower.',
        'breathing': ('An energizing breathing pattern — several quick, powerful inhale/exhale cycles followed '
                      'by a brief breath-hold — may help increase alertness if you need it.'),
        'exercise': ('Some brief, dynamic movement (jumping jacks, a fast-paced short walk) can help wake up '
                     'your system if you\'re feeling sluggish.'),
        'sleep': ('Sleep quality looks good — keep the room dark and quiet to protect this level of recovery.'),
        'nutrition': ('A cold glass of water can help with alertness. A moderate amount of tea or coffee '
                      'should be well tolerated in this state.'),
    },
}

DISCLAIMER = ('This is general wellness information generated from an estimated BPM/HRV reading — '
              'it is not a diagnosis and does not replace professional medical advice.')


def _range_distance(value, low, high):
    """0 if value is within [low, high] (either bound optional), else distance to nearest bound."""
    if value is None:
        return 5.0  # neutral penalty when we don't have the value at all
    if low is not None and value < low:
        return low - value
    if high is not None and value > high:
        return value - high
    return 0.0


def get_recommendation(bpm, hrv_rmssd_ms, hrv_available):
    """
    Returns the best-matching state's guidance, plus how confidently it
    matched. If HRV isn't available, falls back to BPM-only matching with
    reduced confidence — still useful, just flagged as less precise.
    """
    if bpm is None:
        return None

    hrv_value = hrv_rmssd_ms if hrv_available else None

    best_key, best_score = None, float('inf')
    for key, state in STATES.items():
        bpm_dist = _range_distance(bpm, *state['bpm_range'])
        hrv_dist = _range_distance(hrv_value, *state['hrv_range']) if hrv_value is not None else 0.0
        # Normalize roughly onto comparable scales (BPM spans ~40-150, HRV ~10-100+)
        score = (bpm_dist / 10.0) + (hrv_dist / 10.0)
        if score < best_score:
            best_key, best_score = key, score

    state = STATES[best_key]
    confidence = 'High' if (hrv_available and best_score < 1.0) else ('Medium' if hrv_available else 'Low (HRV unavailable — based on BPM only)')

    return {
        'state_key': best_key,
        'state_label': state['label'],
        'nervous_system_note': state['nervous_system'],
        'breathing': state['breathing'],
        'exercise': state['exercise'],
        'sleep': state['sleep'],
        'nutrition': state['nutrition'],
        'match_confidence': confidence,
        'disclaimer': DISCLAIMER,
    }
