"""
Guards on run_snr_sweep's auto-derived band-pass cutoffs.

A pulse that is short relative to its carrier is wider in bandwidth than the
carrier, so the derived lower cutoff falls at or below zero. That used to abort
the whole sweep; it now widens the band instead and reports what it used.
"""

import pytest

from src.evaluate import run_snr_sweep

FS = 48000.0


def test_short_pulse_at_low_carrier_no_longer_aborts():
    result = run_snr_sweep(5.0, [10.0], FS, duration_s=0.001, freq_hz=1000.0,
                           trials_per_snr=2, bandpass=True)
    assert result["rmse_m"][0] == pytest.approx(0.0, abs=0.05)


def test_reported_band_is_always_usable():
    for duration_s, freq_hz in ((0.001, 1000.0), (0.002, 4000.0), (0.001, 9000.0)):
        result = run_snr_sweep(5.0, [10.0], FS, duration_s=duration_s, freq_hz=freq_hz,
                               trials_per_snr=2, bandpass=True)
        assert 0 < result["low_hz"] < result["high_hz"] < FS / 2


def test_band_is_none_when_not_filtering():
    result = run_snr_sweep(5.0, [10.0], FS, trials_per_snr=2, bandpass=False)
    assert result["low_hz"] is None and result["high_hz"] is None


def test_explicit_cutoffs_are_respected():
    result = run_snr_sweep(5.0, [10.0], FS, trials_per_snr=2, bandpass=True,
                           low_hz=3000.0, high_hz=5000.0)
    assert (result["low_hz"], result["high_hz"]) == (3000.0, 5000.0)
