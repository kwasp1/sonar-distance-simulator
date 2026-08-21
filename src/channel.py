"""
Stage 2: Channel Simulator

RULE: Return data only. No plotting or printing in this file.
"""

import numpy as np


def simulate_channel(
    pulse: np.ndarray,
    true_distance_m: float,
    fs: float,
    speed_mps: float = 343.0,
    attenuation: float = 1.0,
    buffer_duration_s: float = 0.05,
) -> np.ndarray:
    """
    Embed a delayed, attenuated copy of `pulse` into a longer silent buffer,
    at the sample offset corresponding to the round-trip delay for true_distance_m.

    Args:
        pulse: transmit pulse samples
        true_distance_m: ground-truth one-way distance to the reflector
        fs: sampling rate in Hz
        speed_mps: propagation speed (343 m/s for sound in air by default)
        attenuation: linear amplitude scale applied to the echo
        buffer_duration_s: total length of the returned buffer (must be > round-trip time)

    Returns:
        1D numpy array: the "received" signal, noiseless (add noise separately with add_noise()).
    """
    raise NotImplementedError


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) to `signal` at the given SNR (dB).

    Returns:
        Noisy signal, same shape as input.
    """
    raise NotImplementedError
