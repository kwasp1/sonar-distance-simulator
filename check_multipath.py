"""
Throwaway exploration script for multi-object detection -- not part of the
test suite, just a way to see how the resolution limit and false-positive
rate actually behave, so you can pick prominence_frac with real numbers
instead of guessing.

Run from the repo root:  python check_multipath.py
"""

import sys

import numpy as np

sys.path.insert(0, "src")
from pulse import generate_pulse, pulse_bandwidth_hz          # noqa: E402
from channel import simulate_multi_object_channel, add_noise  # noqa: E402
from receiver import estimate_multiple_distances               # noqa: E402

FS = 48_000.0
SPEED = 343.0
PULSE = generate_pulse("gaussian", 0.002, FS, freq_hz=4000.0)


def detect(distances_m, attenuations=None, snr_db=None, seed=0, **kwargs):
    buf = simulate_multi_object_channel(
        PULSE, distances_m, FS, attenuations=attenuations,
        speed_mps=SPEED, buffer_duration_s=0.1,
    )
    if snr_db is not None:
        buf = add_noise(buf, snr_db, seed=seed)
    return estimate_multiple_distances(buf, PULSE, FS, speed_mps=SPEED, **kwargs)


print("=" * 78)
theoretical_bw = pulse_bandwidth_hz(PULSE, FS)
print(f"pulse bandwidth: {theoretical_bw:.1f} Hz  ->  "
      f"theory: resolvable to ~{SPEED/(2*theoretical_bw)*100:.1f} cm")
print("=" * 78)

print("\n-- shrinking separation, noiseless (find the resolution boundary) --")
print(f"{'sep (cm)':>9} {'# detected':>11}   distances")
for sep_cm in (50, 20, 10, 5, 3, 2, 1):
    dets = detect([5.0, 5.0 + sep_cm / 100])
    print(f"{sep_cm:9.1f} {len(dets):11d}   {[round(d, 3) for d in dets]}")

print("\n-- false-positive check: does a single object ever split into two? --")
print(f"{'SNR dB':>7} {'false splits / 30 trials':>26}")
for snr in (20, 10, 0, -5):
    false_splits = sum(
        len(detect([5.0], snr_db=snr, seed=k)) > 1 for k in range(30)
    )
    print(f"{snr:7.0f} {false_splits:26d}")

print("\n-- true-positive rate at a fixed 30 cm separation, across SNR --")
print(f"{'SNR dB':>7} {'both objects found / 30 trials':>32}")
for snr in (20, 10, 0, -5):
    hits = sum(
        len(detect([5.0, 5.3], attenuations=[1.0, 0.7], snr_db=snr, seed=k)) == 2
        for k in range(30)
    )
    print(f"{snr:7.0f} {hits:32d}")

print("\n" + "=" * 78)
print("If false-positive rate is high: raise prominence_frac.")
print("If true-positive rate is low at a separation you care about: the objects")
print("are genuinely near the resolution limit at that SNR -- this is a real")
print("physical limit, not something prominence_frac tuning can fix.")
print("=" * 78)
