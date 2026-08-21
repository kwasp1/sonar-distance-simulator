"""
Stages 3 & 4: Receiver (Matched Filter) + Delay/Distance Estimator

RULE: Return data only. No plotting or printing in this file.
"""

import numpy as np


def matched_filter(received_signal: np.ndarray, pulse: np.ndarray) -> np.ndarray:
    """
    Cross-correlate the received signal with the known transmit pulse.

    Returns:
        1D numpy array: the correlation output.
    """
    raise NotImplementedError


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
    raise NotImplementedError
