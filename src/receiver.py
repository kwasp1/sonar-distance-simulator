"""
Stages 3 & 4: Receiver (Matched Filter) + Delay/Distance Estimator

RULE: Return data only. No plotting or printing in this file.
"""

import numpy as np
from scipy.signal import hilbert, find_peaks


def matched_filter(received_signal: np.ndarray, pulse: np.ndarray) -> np.ndarray:
    """
    Cross-correlate the received signal with the known transmit pulse.

    Returns:
        1D numpy array: the correlation output.
    """
    received_signal = np.asarray(received_signal, dtype=float)
    pulse = np.asarray(pulse, dtype=float)
    if received_signal.ndim != 1 or pulse.ndim != 1:
        raise ValueError("received_signal and pulse must both be 1D")
    if pulse.size == 0:
        raise ValueError("pulse is empty")
    if pulse.size > received_signal.size:
        raise ValueError(
            f"pulse ({pulse.size} samples) is longer than received_signal "
            f"({received_signal.size} samples)"
        )
    correlation = np.correlate(received_signal, pulse, mode='full')
    return correlation


def estimate_distance(
    received_signal: np.ndarray,
    pulse: np.ndarray,
    fs: float,
    speed_mps: float = 343.0,
) -> tuple[float, np.ndarray]:
    """
    Full delay -> distance pipeline: matched filter, peak detection, distance conversion.

    Returns:
        (estimated_distance_m, correlation_output)
    """
    if fs <= 0:
        raise ValueError(f"fs={fs} must be positive")
    if speed_mps <= 0:
        raise ValueError(f"speed_mps={speed_mps} must be positive") 
    received_signal = np.asarray(received_signal, dtype=float)
    pulse = np.asarray(pulse, dtype=float)
    correlation = matched_filter(received_signal, pulse)
    # Raw argmax, not the Hilbert envelope: the raw correlation peak is sharper,
    # so it localises better under noise. The envelope is only needed for
    # separating multiple echoes (Week 3 resolution study), not for finding one.
    peak_index = np.argmax(correlation)
    delay_samples = peak_index - (len(pulse) - 1)
    round_trip_time_s = delay_samples / fs
    estimated_distance_m = (round_trip_time_s * speed_mps) / 2
    return estimated_distance_m, correlation

def estimate_multiple_distances(received_signal, pulse, fs, speed_mps=343.0,
                                prominence_frac=0.4, min_separation_samples=None):
    correlation = matched_filter(received_signal, pulse)
    envelope = np.abs(hilbert(correlation))
    if min_separation_samples is None:
        min_separation_samples = len(pulse) // 2
    peak_indices, _ = find_peaks(
        envelope,
        prominence=envelope.max() * prominence_frac,
        distance=min_separation_samples,
    )
    distances = [(pk - (len(pulse) - 1)) / fs * speed_mps / 2 for pk in peak_indices]
    return sorted(distances)