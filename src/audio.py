"""
Playback helpers: turn simulation arrays into something you can listen to.

RULE: Return data only. No plotting, printing, or sound playback in this file.
These functions hand back arrays and .wav bytes; actually playing them is the
caller's job (IPython.display.Audio in a notebook, st.audio in the webapp).

Every duration here is in HEARD seconds -- what the listener experiences after
the slowdown is applied -- not in signal seconds.
"""

from __future__ import annotations

import io

import numpy as np
from scipy.io import wavfile

from src.channel import add_noise, simulate_channel


def make_audible(
    signal: np.ndarray | list[np.ndarray],
    fs: float,
    gap_s: float = 0.6,
    repeats: int = 4,
    slowdown: float = 4.0,
    fade_s: float = 0.005,
    peak: float = 0.9,
    gain: float | None = None,
) -> dict:
    """
    Build a loopable, listenable version of a simulation array.

    A 2 ms pulse on its own is far too short and far too quiet to hear, and
    looping it end to end just produces a harsh buzz. So each copy is followed
    by a gap of silence, which turns the loop into a sonar-like "ping ... ping"
    instead of a drone.

    Nothing is resampled. The slowdown is applied by declaring a lower playback
    rate, which stretches the sound out and drops its pitch at no cost in
    quality. A 2 ms pulse at slowdown=4 is heard as 8 ms, and an echo delay of
    29 ms becomes 116 ms, which is long enough to hear as a separate echo
    rather than a slight doubling.

    Args:
        signal: one array, repeated `repeats` times, or a list of arrays used
            once each in order. Pass a list of independently-noised buffers so
            the hiss does not audibly cycle.
        fs: sampling rate the signal was simulated at, in Hz.
        gap_s: heard silence after each copy.
        repeats: number of copies, ignored when `signal` is a list.
        slowdown: playback stretch factor; 1.0 plays at the original rate.
        fade_s: heard fade applied to the very start and end, so a looping
            player does not click at the seam.
        peak: loudest sample in the result, leaving headroom below clipping.
        gain: reuse a gain returned by an earlier call to keep a set of clips
            at one consistent volume, so that comparing them reveals a change
            in the signal rather than a change in volume. Take it from the
            LOUDEST clip in the set -- for an SNR ladder that is the noisiest,
            lowest-SNR one -- or the rest will clip. Computed from this signal
            when None.

    Returns:
        dict with "audio" (float array in [-1, 1]), "rate_hz" (the rate the
        audio must be played at) and "gain" (pass to later calls to match).
    """
    frames = [signal] if isinstance(signal, np.ndarray) else list(signal)
    if not frames:
        raise ValueError("signal is empty")
    frames = [np.asarray(frame, dtype=float) for frame in frames]
    if any(frame.ndim != 1 for frame in frames):
        raise ValueError("every signal must be 1D")
    if fs <= 0:
        raise ValueError(f"fs={fs} must be positive")
    if slowdown <= 0:
        raise ValueError(f"slowdown={slowdown} must be positive")
    if repeats < 1:
        raise ValueError(f"repeats={repeats} must be >= 1")
    if gap_s < 0 or fade_s < 0:
        raise ValueError("gap_s and fade_s must be non-negative")
    if not 0 < peak <= 1:
        raise ValueError(f"peak={peak} must be in (0, 1]")

    rate_hz = fs / slowdown
    if isinstance(signal, np.ndarray):
        frames = frames * repeats

    gap = np.zeros(round(gap_s * rate_hz))
    audio = np.concatenate([np.concatenate([frame, gap]) for frame in frames])

    fade_samples = min(round(fade_s * rate_hz), audio.size // 2)
    if fade_samples > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
        audio[:fade_samples] *= ramp
        audio[-fade_samples:] *= ramp[::-1]

    if gain is None:
        loudest = float(np.max(np.abs(audio)))
        gain = peak / loudest if loudest > 0 else 1.0
    audio = np.clip(audio * gain, -1.0, 1.0)

    return {"audio": audio, "rate_hz": rate_hz, "gain": gain}


def audible_echo(
    pulse: np.ndarray,
    distance_m: float,
    fs: float,
    snr_db: float,
    frames: int = 4,
    speed_mps: float = 343.0,
    buffer_duration_s: float | None = None,
    seed: int = 0,
    gap_s: float = 0.0,
    **make_audible_kwargs,
) -> dict:
    """
    Listenable version of what the microphone hears: delayed echo plus hiss.

    Each frame is noised with its own seed. Repeating one noisy buffer would
    make the hiss cycle at the loop rate, which is instantly recognisable as
    fake; fresh noise per frame sounds like real background.

    No gap is inserted by default, because the buffer already ends in its own
    (noisy) quiet stretch, and splicing digital silence between noisy frames
    would sound like the hiss being switched on and off.

    Args:
        pulse: transmit pulse, as returned by generate_pulse().
        distance_m: one-way distance to the reflector.
        fs: sampling rate in Hz.
        snr_db: echo loudness relative to the hiss.
        frames: how many independently-noised copies to string together.
        speed_mps: propagation speed.
        buffer_duration_s: listening window per frame. Sized from the distance
            when None.
        seed: first RNG seed; later frames use seed + 1, seed + 2, ...
        gap_s: heard silence after each frame.
        **make_audible_kwargs: slowdown, fade_s, peak, gain.

    Returns:
        Same dict as make_audible().
    """
    if frames < 1:
        raise ValueError(f"frames={frames} must be >= 1")

    if buffer_duration_s is None:
        round_trip_s = 2 * distance_m / speed_mps
        buffer_duration_s = max(0.08, round_trip_s * 1.5)

    received = simulate_channel(
        pulse,
        distance_m,
        fs,
        speed_mps=speed_mps,
        buffer_duration_s=buffer_duration_s,
    )
    noisy_frames = [
        add_noise(received, snr_db, seed=seed + offset) for offset in range(frames)
    ]
    return make_audible(noisy_frames, fs, gap_s=gap_s, **make_audible_kwargs)


def to_wav_bytes(audio: np.ndarray, rate_hz: float) -> bytes:
    """
    Pack audio into .wav bytes, ready for st.audio(), a notebook, or a file.

    The playback rate is stored inside the .wav header, so whoever plays these
    bytes gets the slowdown automatically without being told about it.
    """
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1D, got shape {audio.shape}")
    if rate_hz <= 0:
        raise ValueError(f"rate_hz={rate_hz} must be positive")

    samples = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    stream = io.BytesIO()
    wavfile.write(stream, int(round(rate_hz)), samples)
    return stream.getvalue()
