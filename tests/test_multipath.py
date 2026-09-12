"""
tests/test_multipath.py

Correctness gate for multi-object detection: does not need to explore
edges or characterize behaviour (that's check_multipath.py) -- just
confirms the core claim works before anything is built on top of it.
"""

import numpy as np
import pytest

from src.pulse import generate_pulse
from src.channel import simulate_multi_object_channel
from src.receiver import estimate_multiple_distances

FS = 48000.0
SPEED = 343.0


def test_two_well_separated_objects_both_detected():
    pulse = generate_pulse("gaussian", 0.002, FS, freq_hz=4000.0)
    true_distances = [5.0, 6.0]

    received = simulate_multi_object_channel(
        pulse, true_distances, FS, speed_mps=SPEED, buffer_duration_s=0.1
    )
    detected = estimate_multiple_distances(received, pulse, FS, speed_mps=SPEED)

    assert len(detected) == 2
    for true_d, est_d in zip(sorted(true_distances), detected):
        assert est_d == pytest.approx(true_d, abs=0.01)  # within 1 cm


def test_single_object_not_falsely_split():
    pulse = generate_pulse("gaussian", 0.002, FS, freq_hz=4000.0)

    received = simulate_multi_object_channel(
        pulse, [5.0], FS, speed_mps=SPEED, buffer_duration_s=0.1
    )
    detected = estimate_multiple_distances(received, pulse, FS, speed_mps=SPEED)

    assert len(detected) == 1