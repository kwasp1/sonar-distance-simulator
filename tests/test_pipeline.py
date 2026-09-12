"""
Week 1 sanity check (per the project timeline):
Noiseless, no-attenuation case — the estimated distance must exactly
match the true distance before any noise/filtering is introduced.
"""

import numpy as np
import pytest

from src.pulse import generate_pulse
from src.channel import simulate_channel
from src.receiver import estimate_distance

FS = 48000.0
SPEED = 343.0

@pytest.mark.parametrize("delay_samples", [600, 900, 1200, 1800])
def test_noiseless_distance_recovery(delay_samples):
    true_distance_m = delay_samples * SPEED / (2 * FS)
    pulse = generate_pulse("gaussian", 0.002, FS, freq_hz=4000.0)
    received_signal = simulate_channel(
        pulse, true_distance_m, fs=FS, speed_mps=SPEED, buffer_duration_s=0.1
    )
    estimated_distance_m, _ = estimate_distance(
        received_signal, pulse, fs=FS, speed_mps=SPEED
    )
    recovered_samples = round(estimated_distance_m * 2 * FS / SPEED)
    assert recovered_samples == delay_samples