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
        if low_hz <= 0:
            raise ValueError(
                f"bandpass filter low_hz={low_hz} <= 0; "
                f"try a longer duration_s or higher freq_hz"
            )
        if high_hz >= fs / 2:
            raise ValueError(
                f"bandpass filter high_hz={high_hz} >= fs/2; "
                f"try a longer duration_s or lower freq_hz"
            )

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


def describe_pulse_config(config: dict) -> str:
    """Short human-readable name for a pulse config, e.g. "chirp, 2 ms, sweep 4000 Hz"."""
    parts = [config["pulse_type"], f"{config['duration_s'] * 1000:g} ms"]
    if config.get("bandwidth_hz"):
        parts.append(f"sweep {config['bandwidth_hz']:g} Hz")
    return ", ".join(parts)


def both_targets_detected(
    pulse: np.ndarray,
    fs: float,
    target_distances_m: list[float],
    tolerance_m: float,
    speed_mps: float,
    buffer_duration_s: float,
    min_separation_samples: int,
    prominence_frac: float,
) -> bool:
    """True when the receiver finds one detection per target, each within tolerance_m."""
    received = simulate_multi_object_channel(
        pulse,
        target_distances_m,
        fs,
        speed_mps=speed_mps,
        buffer_duration_s=buffer_duration_s,
    )
    detected_m = estimate_multiple_distances(
        received,
        pulse,
        fs,
        speed_mps=speed_mps,
        prominence_frac=prominence_frac,
        min_separation_samples=min_separation_samples,
    )
    if len(detected_m) != len(target_distances_m):
        return False
    return all(
        min(abs(estimate - target) for estimate in detected_m) <= tolerance_m
        for target in target_distances_m
    )


def smallest_resolvable_separation_m(
    pulse: np.ndarray,
    fs: float,
    separations_m: list[float],
    base_distance_m: float,
    jitter_m: float,
    trials: int,
    success_frac: float,
    tolerance_frac: float,
    min_separation_samples: int,
    prominence_frac: float,
    speed_mps: float,
    buffer_duration_s: float,
    rng: np.random.Generator,
) -> float | None:
    """
    Closest two targets can sit while still being reported as two.

    Separations are tried smallest first and the first one that succeeds in
    at least success_frac of its trials is returned. Each trial nudges the
    pair to a slightly different absolute distance, so the answer reflects
    the pulse rather than one lucky alignment of the two echoes.

    A detection counts only if it is closer to its own target than to the
    other one, so separations too tight for that test to mean anything are
    skipped rather than allowed to pass on a pair of sidelobes.

    Returns None if no separation in the list ever succeeds.
    """
    sample_resolution_m = speed_mps / (2 * fs)
    trials_needed = success_frac * trials

    for separation_m in separations_m:
        tolerance_m = max(separation_m * tolerance_frac, 2 * sample_resolution_m)
        if tolerance_m >= separation_m / 2:
            continue
        successes = 0
        for _ in range(trials):
            first_target_m = base_distance_m + rng.uniform(-jitter_m, jitter_m)
            targets_m = [first_target_m, first_target_m + separation_m]
            if both_targets_detected(
                pulse,
                fs,
                targets_m,
                tolerance_m=tolerance_m,
                speed_mps=speed_mps,
                buffer_duration_s=buffer_duration_s,
                min_separation_samples=min_separation_samples,
                prominence_frac=prominence_frac,
            ):
                successes += 1
        if successes >= trials_needed:
            return float(separation_m)
    return None


def run_bandwidth_resolution_sweep(
    pulse_configs: list[dict],
    fs: float,
    base_distance_m: float = 5.0,
    separations_m: list[float] | None = None,
    trials: int = 12,
    success_frac: float = 0.5,
    jitter_m: float = 0.05,
    tolerance_frac: float = 0.25,
    min_separation_samples: int = 8,
    prominence_frac: float = 0.4,
    speed_mps: float = 343.0,
    buffer_duration_s: float = 0.12,
    seed: int = 0,
) -> dict:
    """
    Measure range resolution against pulse bandwidth.

    For each pulse in pulse_configs, two targets are placed at shrinking
    separations until the receiver can no longer report them as two. The
    smallest separation that still works is that pulse's measured
    resolution, plotted against its measured bandwidth and compared with
    the textbook prediction c / (2 * bandwidth).

    Each config is a kwargs dict for generate_pulse, e.g.
        {"pulse_type": "gaussian", "duration_s": 0.004, "freq_hz": 4000.0}
        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 8000.0,
         "bandwidth_hz": 4000.0}

    Mixing gaussians and chirps matters: a gaussian's bandwidth is tied to
    its duration, so gaussians alone cannot separate "wider bandwidth helps"
    from "shorter pulse helps". Chirps hold duration fixed while bandwidth
    varies, which breaks that tie.

    min_separation_samples is deliberately small and fixed here. The
    receiver's own default is len(pulse) // 2, which is a floor set by pulse
    length rather than bandwidth, and it flattens the wide-bandwidth end of
    this sweep into a straight line that has nothing to do with physics.

    Returns:
        dict with parallel lists "labels", "bandwidth_hz", "resolved_m" and
        "theory_m". A resolved_m entry is None when that pulse never managed
        to separate the targets at any separation tried.
    """
    if not pulse_configs:
        raise ValueError("pulse_configs is empty")
    if trials < 1:
        raise ValueError(f"trials={trials} must be >= 1")
    if not 0 < success_frac <= 1:
        raise ValueError(f"success_frac={success_frac} must be in (0, 1]")
    if jitter_m < 0:
        raise ValueError(f"jitter_m={jitter_m} must be non-negative")

    if separations_m is None:
        separations_m = [centimetres / 100 for centimetres in range(1, 121)]
    if any(separation <= 0 for separation in separations_m):
        raise ValueError("separations_m must all be positive")

    rng = np.random.default_rng(seed)
    labels, bandwidths_hz, resolved_m, theory_m = [], [], [], []

    for config in pulse_configs:
        pulse = generate_pulse(fs=fs, **config)
        bandwidth_hz = pulse_bandwidth_hz(pulse, fs)
        smallest_m = smallest_resolvable_separation_m(
            pulse,
            fs,
            separations_m,
            base_distance_m=base_distance_m,
            jitter_m=jitter_m,
            trials=trials,
            success_frac=success_frac,
            tolerance_frac=tolerance_frac,
            min_separation_samples=min_separation_samples,
            prominence_frac=prominence_frac,
            speed_mps=speed_mps,
            buffer_duration_s=buffer_duration_s,
            rng=rng,
        )

        labels.append(describe_pulse_config(config))
        bandwidths_hz.append(bandwidth_hz)
        resolved_m.append(smallest_m)
        theory_m.append(speed_mps / (2 * bandwidth_hz) if bandwidth_hz > 0 else None)

    return {
        "labels": labels,
        "bandwidth_hz": bandwidths_hz,
        "resolved_m": resolved_m,
        "theory_m": theory_m,
    }
