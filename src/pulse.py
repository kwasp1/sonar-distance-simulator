"""
Stage 1: Transmit Pulse Generator

RULE: Every function here returns plain numpy arrays / numbers.
Never call plt.plot(), plt.show(), or print() inside this file.
Plotting happens only in notebooks/ or app/.
"""

from __future__ import annotations

import numpy as np

def _normalize(signal: np.ndarray) -> np.ndarray:
    """Scale a pulse to unit energy so all pulse types are directly comparable."""
    energy = np.sqrt(np.sum(signal ** 2))
    if energy == 0:
        return signal
    return signal / energy

def generate_pulse(
    pulse_type: str,
    duration_s: float,
    fs: float,
    freq_hz: float | None = None,
    bandwidth_hz: float | None = None,
) -> np.ndarray:
    """
    Generate a transmit pulse.

    Args:
        pulse_type: "rect" | "gaussian" | "chirp"
        duration_s: pulse duration in seconds
        fs: sampling rate in Hz
        freq_hz: carrier/center frequency in Hz (required for all pulse types)
        bandwidth_hz: linear-sweep bandwidth in Hz for "chirp". Defaults to freq_hz,
            i.e. a sweep from freq_hz/2 up to 3*freq_hz/2. Ignored for other types.

        Returns:
        1D numpy array of pulse samples (float64), length round(duration_s * fs),
        normalized to unit energy so all pulse types carry equal energy.

    Note:
        For "gaussian", the envelope std dev is duration_s / 6, so the effective
        pulse is roughly duration_s / 3 wide — shorter than duration_s.
    """
    
    if duration_s <= 0:
        raise ValueError(f"duration_s={duration_s} must be positive")
    if fs <= 0:
        raise ValueError(f"fs={fs} must be positive")

    n_samples = int(round(duration_s * fs))
    if n_samples < 1:
        raise ValueError(
            f"duration_s={duration_s} at fs={fs} yields {n_samples} samples; need >= 1"
        )

    # Shared time base, symmetric about t = 0 for both even and odd n_samples.
    t_s = (np.arange(n_samples) - (n_samples - 1) / 2) / fs
    nyquist_hz = fs / 2

    if freq_hz is None:
        raise ValueError("freq_hz is required for pulse_type='rect'")
    if not 0 <= freq_hz <= nyquist_hz:
        raise ValueError(f"freq_hz={freq_hz} must be in [0, {nyquist_hz}]")

    if pulse_type == "rect":
        return _normalize(np.cos(2 * np.pi * freq_hz * t_s))

    if pulse_type == "gaussian":
        # sigma set so the envelope tapers to ~0 at both edges of duration_s
        sigma_s = duration_s / 6
        envelope = np.exp(-(t_s ** 2) / (2 * sigma_s ** 2))
        carrier = np.cos(2 * np.pi * freq_hz * t_s)
        return _normalize(envelope * carrier)

    if pulse_type == "chirp":
        sweep_hz = freq_hz if bandwidth_hz is None else bandwidth_hz
        if sweep_hz < 0:
            raise ValueError(f"bandwidth_hz={sweep_hz} must be non-negative")
        f_start_hz = freq_hz - sweep_hz / 2
        f_end_hz = freq_hz + sweep_hz / 2
        if f_start_hz < 0 or f_end_hz > nyquist_hz:
            raise ValueError(
                f"chirp sweeps [{f_start_hz}, {f_end_hz}] Hz, outside [0, {nyquist_hz}]"
            )
        # Linear frequency sweep: instantaneous phase is the integral of 2*pi*f(t),
        # with f(t) ramping linearly from f_start_hz to f_end_hz over duration_s.
        tau_s = t_s + duration_s / 2  # 0 .. duration_s
        chirp_rate_hz_per_s = sweep_hz / duration_s
        phase = 2 * np.pi * (f_start_hz * tau_s + 0.5 * chirp_rate_hz_per_s * tau_s ** 2)
        return _normalize(np.cos(phase))

    raise ValueError(
        f"unknown pulse_type={pulse_type!r} (expected 'rect', 'gaussian', or 'chirp')"
    )

def pulse_bandwidth_hz(pulse: np.ndarray, fs: float) -> float:
    """
    Estimate the -3 dB (half-power) bandwidth of a pulse from its samples alone.

    Three steps, once the power spectrum P(f) = |rFFT(pulse)|**2 is in hand:

      1. peak power           P_max = max(P)
      2. half-power threshold  P_max / 2          (i.e. -3 dB below the peak)
      3. bandwidth = f_high - f_low, where f_low and f_high are the lowest and
         highest frequencies whose power is at or above that threshold
         (first-to-last crossing).

    Why the -3 dB span, and not the RMS spread:
    it is the conventional, easy-to-read bandwidth figure -- the width of the
    main spectral lobe where the pulse actually carries most of its energy --
    and it maps directly onto how sonar/ultrasound hardware bandwidth is
    quoted. It also tracks range resolution intuitively: a wider -3 dB lobe
    gives a narrower matched-filter correlation peak, hence finer resolution.

    Why measured from the spectrum (not returned from generate_pulse's args):
    the same function must work on any pulse -- including a filtered or
    acoustically-recorded one whose generating parameters are unknown.

    Notes:
      - No extra window is applied. A pulse with hard edges (rect, chirp)
        genuinely has that spectral leakage, and it is part of its bandwidth.
      - The signal is zero-padded before the FFT so the frequency grid is fine
        enough to locate the crossings even for short (tens of samples) pulses.
      - Computed over non-negative frequencies only; for a real pulse the
        spectrum is symmetric, so the positive half carries the full picture.
      - First-to-last crossing: if sidelobes rise above the half-power level
        they widen the estimate. For the pulses here the main lobe dominates,
        so this stays simple; switch to a contiguous-region walk if that
        assumption breaks.

    Args:
        pulse: transmit pulse samples (real, 1D).
        fs: sampling rate in Hz.

    Returns:
        -3 dB bandwidth in Hz. 0.0 for a silent (all-zero) pulse.
    """
    pulse = np.asarray(pulse, dtype=float)
    if pulse.ndim != 1:
        raise ValueError(f"pulse must be 1D, got shape {pulse.shape}")
    if fs <= 0:
        raise ValueError(f"fs={fs} must be positive")

    n_fft = max(4096, 32 * pulse.size)
    power = np.abs(np.fft.rfft(pulse, n=n_fft)) ** 2
    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    if power.sum() == 0:
        return 0.0

    # 1-2. half-power (-3 dB) threshold; 3. first-to-last frequency above it.
    above_half_power = power >= power.max() / 2
    freqs_in_band = freqs_hz[above_half_power]
    return float(freqs_in_band[-1] - freqs_in_band[0])
