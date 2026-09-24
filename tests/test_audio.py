"""
Playback helpers: shape, loudness and loop-safety.

These never check what something sounds like, only that the arrays handed to
a player are well formed: in range, the right length, faded at the seam, and
reproducible when a gain is reused.
"""

import io

import numpy as np
import pytest
from scipy.io import wavfile

from src.audio import audible_echo, make_audible, to_wav_bytes
from src.pulse import generate_pulse

FS = 48000.0
PULSE = generate_pulse("gaussian", 0.002, FS, freq_hz=4000.0)


def test_audio_stays_within_speaker_range():
    audio = make_audible(PULSE, FS)["audio"]
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0


def test_peak_is_set_to_the_requested_loudness():
    audio = make_audible(PULSE, FS, peak=0.5)["audio"]
    assert np.max(np.abs(audio)) == pytest.approx(0.5, abs=1e-9)


def test_length_is_every_copy_plus_its_gap():
    result = make_audible(PULSE, FS, gap_s=0.5, repeats=3, slowdown=4.0)
    gap_samples = round(0.5 * result["rate_hz"])
    assert result["audio"].size == 3 * (PULSE.size + gap_samples)


def test_slowdown_only_changes_the_playback_rate():
    fast = make_audible(PULSE, FS, slowdown=1.0)
    slow = make_audible(PULSE, FS, slowdown=4.0, gap_s=0.0)
    assert slow["rate_hz"] == FS / 4
    assert fast["rate_hz"] == FS


def test_edges_are_faded_so_a_loop_does_not_click():
    audio = make_audible(PULSE, FS, fade_s=0.01)["audio"]
    assert abs(audio[0]) < 1e-9
    assert abs(audio[-1]) < 1e-9


def test_reusing_a_gain_reproduces_the_same_audio():
    first = make_audible(PULSE, FS)
    second = make_audible(PULSE, FS, gain=first["gain"])
    assert np.array_equal(first["audio"], second["audio"])


def test_a_shared_gain_keeps_noisy_clips_comparable():
    loud = audible_echo(PULSE, 5.0, FS, snr_db=20.0, frames=2)
    quiet = audible_echo(PULSE, 5.0, FS, snr_db=-5.0, frames=2, gain=loud["gain"])
    assert quiet["gain"] == loud["gain"]
    assert np.max(np.abs(quiet["audio"])) > np.max(np.abs(loud["audio"]))


def test_echo_frames_each_get_their_own_noise():
    result = audible_echo(PULSE, 5.0, FS, snr_db=0.0, frames=2, slowdown=1.0)
    half = result["audio"].size // 2
    assert not np.allclose(result["audio"][:half], result["audio"][half:])


def test_wav_bytes_survive_a_round_trip():
    result = make_audible(PULSE, FS, slowdown=4.0)
    rate, samples = wavfile.read(io.BytesIO(to_wav_bytes(result["audio"], result["rate_hz"])))
    assert rate == round(result["rate_hz"])
    assert samples.size == result["audio"].size
    assert np.max(np.abs(samples / 32767 - result["audio"])) < 1e-3


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        make_audible([], FS)
    with pytest.raises(ValueError):
        make_audible(PULSE, FS, slowdown=0)
    with pytest.raises(ValueError):
        make_audible(PULSE, FS, peak=1.5)
    with pytest.raises(ValueError):
        audible_echo(PULSE, 5.0, FS, snr_db=0.0, frames=0)
    with pytest.raises(ValueError):
        to_wav_bytes(np.zeros((2, 2)), 12000.0)
