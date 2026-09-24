"""
Feature 8 gate: the acoustic analysis is checked against a real recording.

physical_sonar_check/rx.npy is an actual microphone capture of a laptop
chirping at a wall, with baseline.npy recorded facing open space and
excess.npy the result the original probe script produced. Re-deriving
excess.npy from the raw recording proves this module does what the
hardware script did, and it runs with no microphone attached.
"""

import numpy as np
import pytest

from src.acoustic import (
    align_to_direct_path,
    analyse_recording,
    design_chirp,
    detect_echo,
    fold_frames,
    remove_clutter,
)

FS = 48000.0
SPEED = 343.0
CHIRP_START_HZ, CHIRP_END_HZ, CHIRP_DURATION_S = 2000.0, 8000.0, 0.010
FRAME_SAMPLES = int((CHIRP_DURATION_S + 0.250) * FS)

RECORDINGS = "physical_sonar_check"


@pytest.fixture(scope="module")
def recorded():
    return {
        "rx": np.load(f"{RECORDINGS}/rx.npy"),
        "baseline": np.load(f"{RECORDINGS}/baseline.npy"),
        "excess": np.load(f"{RECORDINGS}/excess.npy"),
    }


@pytest.fixture(scope="module")
def chirp():
    return design_chirp(CHIRP_START_HZ, CHIRP_END_HZ, CHIRP_DURATION_S, FS)


def test_reproduces_the_original_hardware_result(recorded, chirp):
    result = analyse_recording(
        recorded["rx"], chirp, FRAME_SAMPLES, FS,
        baseline_envelope=recorded["baseline"], speed_mps=SPEED,
    )
    assert np.allclose(result["excess"], recorded["excess"], atol=1e-12)


def test_finds_the_wall_in_the_real_recording(recorded, chirp):
    result = analyse_recording(
        recorded["rx"], chirp, FRAME_SAMPLES, FS,
        baseline_envelope=recorded["baseline"], speed_mps=SPEED,
    )
    assert result["distance_m"] == pytest.approx(1.797, abs=0.05)
    assert result["quality_sigma"] > 6.0


def test_chirp_covers_the_requested_band(chirp):
    spectrum = np.abs(np.fft.rfft(chirp, n=8192))
    freqs = np.fft.rfftfreq(8192, d=1 / FS)
    in_band = (freqs >= CHIRP_START_HZ) & (freqs <= CHIRP_END_HZ)
    assert spectrum[in_band].mean() > 10 * spectrum[~in_band].mean()


def test_folding_frames_averages_noise_away():
    rng = np.random.default_rng(0)
    frame_samples = 500
    echo = np.zeros(frame_samples)
    echo[100] = 1.0
    noisy = np.concatenate([echo + rng.normal(0, 0.5, frame_samples) for _ in range(64)])

    folded = fold_frames(noisy, frame_samples)
    noise_floor = np.std(np.delete(folded, 100))
    assert folded[100] == pytest.approx(1.0, abs=0.2)
    assert noise_floor < 0.5 / 4


def test_alignment_puts_the_direct_arrival_at_zero():
    trace = np.zeros(100)
    trace[30] = 5.0
    trace[45] = 1.0

    aligned = align_to_direct_path(trace)
    assert aligned[0] == 1.0
    assert np.argmax(aligned) == 0
    assert aligned[15] == pytest.approx(0.2)


def test_clutter_removal_cancels_an_unchanged_background():
    baseline = np.array([1.0, 0.4, 0.2, 0.1])
    unchanged = remove_clutter(baseline.copy(), baseline)
    assert np.allclose(unchanged, 0.0)


def test_detects_a_planted_echo_at_the_right_distance():
    excess = np.zeros(4000)
    target_index = int(2 * 2.5 / SPEED * FS)
    excess[target_index] = 1.0

    found = detect_echo(excess, FS, speed_mps=SPEED, min_range_m=1.0, max_range_m=5.0)
    assert found["distance_m"] == pytest.approx(2.5, abs=0.01)


def test_rejects_bad_arguments(chirp):
    with pytest.raises(ValueError):
        design_chirp(8000.0, 2000.0, 0.01, FS)
    with pytest.raises(ValueError):
        design_chirp(2000.0, 40000.0, 0.01, FS)
    with pytest.raises(ValueError):
        fold_frames(np.zeros(100), 500)
    with pytest.raises(ValueError):
        remove_clutter(np.zeros(10), np.zeros(11))
    with pytest.raises(ValueError):
        detect_echo(np.zeros(4000), FS, min_range_m=5.0, max_range_m=1.0)
