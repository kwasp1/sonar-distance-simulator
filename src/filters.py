"""
Phase C: Optional band-pass pre-processing filter (robustness stage).

RULE: Return data only. No plotting or printing in this file.
"""

import numpy as np


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
    raise NotImplementedError
