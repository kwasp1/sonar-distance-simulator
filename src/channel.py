"""
Stage 2: Channel Simulator

RULE: Return data only. No plotting or printing in this file.
"""

import numpy as np


def simulate_channel(
    pulse: np.ndarray,
    true_distance_m: float,
    fs: float,
    speed_mps: float = 343.0,
    attenuation: float = 1.0,
    buffer_duration_s: float = 0.05,
) -> np.ndarray:
    """
    Embed a delayed, attenuated copy of `pulse` into a longer silent buffer,
    at the sample offset corresponding to the round-trip delay for true_distance_m.

    Args:
        pulse: transmit pulse samples
        true_distance_m: ground-truth one-way distance to the reflector
        fs: sampling rate in Hz
        speed_mps: propagation speed (343 m/s for sound in air by default)
        attenuation: linear amplitude scale applied to the echo
        buffer_duration_s: total length of the returned buffer (must be > round-trip time)

    Returns:
        1D numpy array: the "received" signal, noiseless (add noise separately with add_noise()).
    """
    if true_distance_m <= 0:
        raise ValueError(f"true_distance_m={true_distance_m} must be positive")
    if speed_mps <= 0:
        raise ValueError(f"speed_mps={speed_mps} must be positive")
    if fs <= 0:
        raise ValueError(f"fs={fs} must be positive")
    if attenuation < 0:
        raise ValueError(f"attenuation={attenuation} must be non-negative")

    buffer_length = int(round(buffer_duration_s * fs))
    if buffer_length < len(pulse):
        raise ValueError(
            f"buffer_duration_s={buffer_duration_s} at fs={fs} yields "
            f"{buffer_length} samples, shorter than the {len(pulse)}-sample pulse"
        )
    buffer = np.zeros(buffer_length)
    round_trip_time_s = 2 * true_distance_m / speed_mps
    delay_samples = round(round_trip_time_s * fs)  
    if delay_samples + len(pulse) > buffer_length:
        max_range_m = (buffer_length - len(pulse)) * speed_mps / (2 * fs)
        raise ValueError(
            f"echo from {true_distance_m} m does not fit in a "
            f"{buffer_duration_s} s window (max range {max_range_m:.3f} m)"
        )
    buffer[delay_samples:delay_samples + len(pulse)] += pulse * attenuation
    return buffer

def add_noise(
    signal: np.ndarray,
    snr_db: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) to `signal` at the given SNR (dB).

    Signal power is measured over the echo region only (the non-zero samples),
    not the whole buffer. Averaging over the silent portion would make the
    realized SNR depend on buffer_duration_s: a longer listening window adds
    more zeros, lowering the mean and thinning the noise for the same snr_db.

    This assumes `signal` is the noiseless buffer straight out of
    simulate_channel(), where silence is exactly zero. Calling this on an
    already-noisy buffer finds no zeros and measures the whole array instead.

    Noise is drawn across the entire buffer, not just under the echo.

    Args:
        signal: noiseless received buffer (1D).
        snr_db: signal-to-noise ratio in dB, relative to echo power.
        seed: RNG seed for reproducibility. None gives a fresh draw each call.

    Returns:
        Noisy signal, same shape as input.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    if not np.isfinite(snr_db):
        raise ValueError(f"snr_db={snr_db} must be a finite number")

    echo = signal[signal != 0]
    if echo.size == 0:
        raise ValueError("signal is all zeros; cannot set SNR against a silent signal")

    signal_power = float(np.mean(echo ** 2))
    if signal_power <= 0:
        raise ValueError("echo region has zero power; cannot set SNR")

    noise_power = signal_power / (10 ** (snr_db / 10))

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), signal.shape)

    return signal + noise

def simulate_multi_object_channel(pulse, distances_m, fs, attenuations=None, **kwargs):
    if attenuations is None:
        attenuations = [1.0] * len(distances_m)
    buffer = None
    for d, att in zip(distances_m, attenuations):
        echo = simulate_channel(pulse, d, fs, attenuation=att, **kwargs)
        buffer = echo if buffer is None else buffer + echo
    return buffer