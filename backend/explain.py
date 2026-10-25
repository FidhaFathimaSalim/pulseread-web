"""
explain.py — visual explainability helpers.

Turns the pulse_signal array PulseRead already captures into an actual
plot, and packages it alongside the ROI snapshot (from pulseread_pipeline's
analyze_video) so the output isn't just a black-box number — the user or
an examiner can see the extracted waveform and which face regions were used.

Requires: pip install matplotlib
"""

import os
import matplotlib
matplotlib.use('Agg')  # no GUI backend needed — just saving PNGs
import matplotlib.pyplot as plt


def plot_pulse_waveform(pulse_signal, eff_fps, output_path):
    """
    Saves a simple line plot of the filtered rPPG waveform PulseRead
    extracted, so the actual signal behind the BPM number is visible.
    """
    if not pulse_signal:
        return None

    t = [i / eff_fps for i in range(len(pulse_signal))]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(t, pulse_signal, linewidth=1, color='#2563eb')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Filtered signal (a.u.)')
    ax.set_title('Extracted rPPG waveform (multi-ROI, filtered)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def build_explainability_artifacts(pulseread_result, output_dir):
    """
    Given PulseRead's raw result dict (which already has pulse_signal and
    roi_snapshot_path), generate/collect the artifacts and return their
    paths for the dashboard to display.
    """
    waveform_path = os.path.join(output_dir, 'pulse_waveform.png')
    plot_pulse_waveform(
        pulseread_result.get('pulse_signal'),
        pulseread_result.get('bpm_processing_fps') or 10.0,
        waveform_path,
    )

    return {
        'waveform_plot_path': waveform_path if os.path.exists(waveform_path) else None,
        'roi_snapshot_path': pulseread_result.get('roi_snapshot_path'),
    }
