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
    if pulse_type == "gaussian":
        if freq_hz is None:
            raise ValueError("freq_hz is required for pulse_type='gaussian'")

        n_samples = int(round(duration_s * fs))
        t_s = np.arange(n_samples) / fs
        t_centered_s = t_s - duration_s / 2

        # sigma set so the envelope tapers to ~0 at both edges of duration_s
        sigma_s = duration_s / 6
        envelope = np.exp(-(t_centered_s ** 2) / (2 * sigma_s ** 2))
        carrier = np.cos(2 * np.pi * freq_hz * t_centered_s)
        return envelope * carrier

    raise NotImplementedError(f"pulse_type={pulse_type!r} not yet implemented")


def pulse_bandwidth_hz(pulse: np.ndarray, fs: float) -> float:
    """
    Estimate the effective bandwidth of a pulse (for the resolution-vs-bandwidth study).

    Returns:
        Bandwidth in Hz.
    """
    raise NotImplementedError
