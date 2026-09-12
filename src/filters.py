"""
Phase C: Optional band-pass pre-processing filter (robustness stage).

RULE: Return data only. No plotting or printing in this file.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter (use scipy.signal.butter + filtfilt).

    Returns:
        Filtered signal, same shape as input.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    if fs <= 0:
        raise ValueError(f"fs={fs} must be positive")
    if order < 1:
        raise ValueError(f"order={order} must be >= 1")

    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq

    if not 0 < low < high < 1:
        raise ValueError(
            f"cutoffs must satisfy 0 < low_hz < high_hz < {nyq} Hz "
            f"(got low_hz={low_hz}, high_hz={high_hz})"
        )
    
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)
