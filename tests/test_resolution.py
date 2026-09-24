"""
Week 3 gate: range resolution should improve as pulse bandwidth grows.

That trend is the claim the bandwidth sweep exists to make, so it is what
these tests check. The exact centimetre values are left to the exploratory
runs -- they shift with detector tuning, while the trend should not.
"""

import pytest

from src.evaluate import run_bandwidth_resolution_sweep

FS = 48000.0
SPEED = 343.0

GAUSSIAN_CONFIGS = [
    {"pulse_type": "gaussian", "duration_s": 0.008, "freq_hz": 4000.0},
    {"pulse_type": "gaussian", "duration_s": 0.004, "freq_hz": 4000.0},
    {"pulse_type": "gaussian", "duration_s": 0.002, "freq_hz": 4000.0},
]
COARSE_SEPARATIONS_M = [centimetres / 100 for centimetres in range(2, 81, 2)]


@pytest.fixture(scope="module")
def sweep():
    return run_bandwidth_resolution_sweep(
        GAUSSIAN_CONFIGS,
        FS,
        separations_m=COARSE_SEPARATIONS_M,
        trials=4,
    )


def test_every_pulse_resolves_something(sweep):
    assert all(resolved is not None for resolved in sweep["resolved_m"])


def test_wider_bandwidth_resolves_closer_targets(sweep):
    pairs = sorted(zip(sweep["bandwidth_hz"], sweep["resolved_m"]))
    separations = [resolved for _, resolved in pairs]
    assert separations == sorted(separations, reverse=True)


def test_measured_resolution_is_the_right_order_as_theory(sweep):
    for measured, theory in zip(sweep["resolved_m"], sweep["theory_m"]):
        assert 0.1 * theory <= measured <= theory


def test_returns_none_when_no_separation_is_resolvable(sweep):
    result = run_bandwidth_resolution_sweep(
        [{"pulse_type": "gaussian", "duration_s": 0.008, "freq_hz": 4000.0}],
        FS,
        separations_m=[0.02, 0.03],
        trials=2,
    )
    assert result["resolved_m"] == [None]


def test_rejects_bad_arguments():
    config = [{"pulse_type": "gaussian", "duration_s": 0.002, "freq_hz": 4000.0}]
    with pytest.raises(ValueError):
        run_bandwidth_resolution_sweep([], FS)
    with pytest.raises(ValueError):
        run_bandwidth_resolution_sweep(config, FS, trials=0)
    with pytest.raises(ValueError):
        run_bandwidth_resolution_sweep(config, FS, success_frac=0.0)
    with pytest.raises(ValueError):
        run_bandwidth_resolution_sweep(config, FS, separations_m=[0.1, -0.2])
