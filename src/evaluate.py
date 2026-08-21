"""
Stage 5 (data half): Evaluation sweeps.

These functions run the full pipeline many times and return arrays of
results (e.g. distance error per SNR level). They do NOT plot anything —
plotting the results returned here happens in notebooks/ or app/.
"""

import numpy as np


def run_snr_sweep(
    true_distance_m: float,
    snr_values_db: list[float],
    fs: float,
    pulse_type: str = "gaussian",
    duration_s: float = 0.001,
    trials_per_snr: int = 20,
) -> dict:
    """
    For each SNR value, run the full pipeline `trials_per_snr` times and
    compute the RMSE of the distance estimate.

    Returns:
        dict with keys: "snr_db" (list), "rmse_m" (list) — ready to hand to a plotting function.
    """
    raise NotImplementedError


def run_bandwidth_resolution_sweep(
    fs: float,
    pulse_durations_s: list[float],
    separation_range_m: tuple[float, float],
) -> dict:
    """
    For each pulse duration (-> bandwidth), find the smallest resolvable
    separation between two echoes.

    Returns:
        dict with keys: "duration_s" (list), "bandwidth_hz" (list), "min_resolution_m" (list).
    """
    raise NotImplementedError
