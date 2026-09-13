"""
Stage 5 (data half): Evaluation sweeps.

These functions run the full pipeline many times and return arrays of
results (e.g. distance error per SNR level). They do NOT plot anything —
plotting the results returned here happens in notebooks/ or app/.
"""

import numpy as np
from src.pulse import generate_pulse
from src.channel import simulate_channel, add_noise, simulate_multi_object_channel
from src.receiver import estimate_distance, estimate_multiple_distances
from src.filters import bandpass_filter
from src.pulse import pulse_bandwidth_hz


def run_snr_sweep(
    true_distance_m: float,
    snr_values_db: list[float],
    fs: float,
    pulse_type: str = "gaussian",
    duration_s: float = 0.002,
    freq_hz: float = 4000.0,
    trials_per_snr: int = 20,
    speed_mps: float = 343.0,
    buffer_duration_s: float = 0.1,
    bandpass: bool = False,
    low_hz: float | None = None,
    high_hz: float | None = None,
) -> dict:
    if not snr_values_db:
        raise ValueError("snr_values_db is empty")
    if trials_per_snr < 1:
        raise ValueError(f"trials_per_snr={trials_per_snr} must be >= 1")

    pulse = generate_pulse(pulse_type, duration_s, fs, freq_hz=freq_hz)

    if bandpass and (low_hz is None or high_hz is None):
        measured_bw = pulse_bandwidth_hz(pulse, fs)
        low_hz = freq_hz - measured_bw
        high_hz = freq_hz + measured_bw

    snr_list, rmse_list = [], []
    for snr_db in snr_values_db:
        errors = []
        for trial_index in range(trials_per_snr):
            received = simulate_channel(
                pulse, true_distance_m, fs=fs, speed_mps=speed_mps,
                buffer_duration_s=buffer_duration_s
            )
            noisy = add_noise(received, snr_db, seed=trial_index)
            if bandpass:
                noisy = bandpass_filter(noisy, fs, low_hz, high_hz)
            est, _ = estimate_distance(noisy, pulse, fs=fs, speed_mps=speed_mps)
            errors.append(est - true_distance_m)
        rmse_list.append(float(np.sqrt(np.mean(np.square(errors)))))
        snr_list.append(snr_db)

    return {"snr_db": snr_list, "rmse_m": rmse_list}

def run_bandwidth_resolution_sweep(
    fs: float,
    pulse_durations_s: list[float],
    separation_range_m: tuple[float, float],
    pulse_type: str = "gaussian",
    freq_hz: float = 4000.0,
    speed_mps: float = 343.0,
    n_steps: int = 40,
    base_distance_m: float = 3.0,
    buffer_duration_s: float = 0.08,
) -> dict:
    """
    For each pulse duration (-> bandwidth), find the smallest resolvable
    separation between two echoes.

    Returns:
        dict with keys: "duration_s" (list), "bandwidth_hz" (list), "min_resolution_m" (list).
    """
    if not pulse_durations_s:
        raise ValueError("pulse_durations_s is empty")
    min_sep, max_sep = separation_range_m
    if min_sep <= 0 or max_sep <= min_sep:
        raise ValueError(f"invalid separation_range_m={separation_range_m}")

    durations, bandwidths, min_resolutions = [], [], []
    test_separations = np.linspace(max_sep, min_sep, n_steps)

    for dur in pulse_durations_s:
        pulse = generate_pulse(pulse_type, dur, fs, freq_hz=freq_hz)
        bw = pulse_bandwidth_hz(pulse, fs)
        durations.append(dur)
        bandwidths.append(bw)

        smallest_sep = max_sep
        for sep in test_separations:
            buf = simulate_multi_object_channel(
                pulse,
                [base_distance_m, base_distance_m + sep],
                fs,
                speed_mps=speed_mps,
                buffer_duration_s=buffer_duration_s,
            )
            dets = estimate_multiple_distances(buf, pulse, fs, speed_mps=speed_mps)
            if len(dets) >= 2:
                # check if both targets are resolved near ground truth
                d1, d2 = dets[0], dets[1]
                if abs(d1 - base_distance_m) < 0.05 and abs(d2 - (base_distance_m + sep)) < 0.05:
                    smallest_sep = float(sep)
                else:
                    break
            else:
                break

        min_resolutions.append(smallest_sep)

    return {
        "duration_s": durations,
        "bandwidth_hz": bandwidths,
        "min_resolution_m": min_resolutions,
    }
