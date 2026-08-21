"""
Week 1 sanity check (per the project timeline):
Noiseless, no-attenuation case — the estimated distance must exactly
match the true distance before any noise/filtering is introduced.
"""

import numpy as np
from src.pulse import generate_pulse
from src.channel import simulate_channel
from src.receiver import estimate_distance


def test_noiseless_delay_recovery():
    fs = 44100.0
    pulse = generate_pulse("gaussian", duration_s=0.001, fs=fs, freq_hz=2000)
    true_distance = 10.0  # meters

    received = simulate_channel(pulse, true_distance_m=true_distance, fs=fs)
    estimated_distance, _ = estimate_distance(received, pulse, fs=fs)

    assert abs(estimated_distance - true_distance) < 0.05  # within a few cm
