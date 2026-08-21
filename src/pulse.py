"""
Stage 1: Transmit Pulse Generator

RULE: Every function here returns plain numpy arrays / numbers.
Never call plt.plot(), plt.show(), or print() inside this file.
Plotting happens only in notebooks/ or app/.
"""

import numpy as np


def generate_pulse(pulse_type: str, duration_s: float, fs: float, freq_hz: float = None) -> np.ndarray:
    """
    Generate a transmit pulse.

    Args:
        pulse_type: "rect" | "gaussian" | "chirp"
        duration_s: pulse duration in seconds
        fs: sampling rate in Hz
        freq_hz: carrier/center frequency in Hz (used for "gaussian" and "chirp")

    Returns:
        1D numpy array of pulse samples.
    """
    raise NotImplementedError


def pulse_bandwidth_hz(pulse: np.ndarray, fs: float) -> float:
    """
    Estimate the effective bandwidth of a pulse (for the resolution-vs-bandwidth study).

    Returns:
        Bandwidth in Hz.
    """
    raise NotImplementedError
