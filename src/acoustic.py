"""
Feature 8: real acoustic ranging with a laptop speaker and microphone.

RULE: Return data only. No plotting, printing, recording or playback here.
Capturing audio is the caller's job (see physical_sonar_check/); everything
in this file works on arrays that were recorded earlier, so the analysis can
be tested against archived recordings without any hardware attached.

What a real recording needs that the simulator does not:

  * many transmissions averaged together, because one chirp off a wall is
    far weaker than one chirp dropped into a simulated buffer;
  * a time origin, because the microphone hears the speaker directly long
    before any echo arrives, and that direct arrival is t = 0;
  * clutter removal, because the laptop chassis rings for roughly a metre
    of apparent range and swamps anything close in.
"""

from __future__ import annotations

import numpy as np


def design_chirp(
    f_start_hz: float,
    f_end_hz: float,
    duration_s: float,
    fs: float,
) -> np.ndarray:
    """
    Build the sweep to transmit through a real speaker.

    Deliberately not generate_pulse(): this one is windowed, so the speaker is
    not asked to start and stop abruptly, and it is specified by its start and
    end frequency because what matters outdoors of the simulator is the band
    the hardware can actually reproduce. Run band_check.py to find that band
    before choosing the limits.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s={duration_s} must be positive")
    if fs <= 0:
        raise ValueError(f"fs={fs} must be positive")
    if not 0 <= f_start_hz < f_end_hz <= fs / 2:
        raise ValueError(
            f"need 0 <= f_start_hz < f_end_hz <= {fs / 2} Hz "
            f"(got {f_start_hz}, {f_end_hz})"
        )

    n_samples = int(round(duration_s * fs))
    t_s = np.arange(n_samples) / fs
    sweep_rate = (f_end_hz - f_start_hz) / (2 * duration_s)
    return np.sin(2 * np.pi * (f_start_hz * t_s + sweep_rate * t_s ** 2)) * np.hanning(n_samples)


def fold_frames(correlation: np.ndarray, frame_samples: int) -> np.ndarray:
    """
    Average every transmitted frame on top of the others.

    The chirp is sent many times at a fixed spacing, so the echo lands at the
    same place in every frame while the room noise does not. Stacking the
    frames therefore grows the echo and averages the noise away.
    """
    correlation = np.asarray(correlation, dtype=float)
    if correlation.ndim != 1:
        raise ValueError(f"correlation must be 1D, got shape {correlation.shape}")
    if frame_samples < 1:
        raise ValueError(f"frame_samples={frame_samples} must be >= 1")
    if correlation.size < frame_samples:
        raise ValueError(
            f"correlation has {correlation.size} samples, fewer than one "
            f"{frame_samples}-sample frame"
        )

    whole_frames = correlation.size // frame_samples
    usable = correlation[: whole_frames * frame_samples]
    return usable.reshape(whole_frames, frame_samples).sum(axis=0) / whole_frames


def align_to_direct_path(folded: np.ndarray) -> np.ndarray:
    """
    Slide the trace so the direct speaker-to-microphone arrival sits at zero.

    That arrival is by far the loudest thing in the trace and it travels a
    known, near-zero distance, so it is the only sensible time reference.
    Everything after it is measured relative to it, and the result is scaled
    so the direct arrival reads 1.0.
    """
    folded = np.asarray(folded, dtype=float)
    if folded.ndim != 1:
        raise ValueError(f"folded must be 1D, got shape {folded.shape}")
    if folded.size == 0:
        raise ValueError("folded is empty")

    envelope = np.abs(folded)
    envelope = np.roll(envelope, -int(np.argmax(envelope)))
    if envelope[0] == 0:
        raise ValueError("trace is silent; nothing to align to")
    return envelope / envelope[0]


def remove_clutter(envelope: np.ndarray, baseline_envelope: np.ndarray) -> np.ndarray:
    """
    Subtract a recording made facing open space.

    The speaker ringing, the chassis buzzing and the desk reflecting are
    identical every run, so a recording with nothing in front of the laptop
    captures all of it. What survives the subtraction is what changed, which
    is the object you pointed at.
    """
    envelope = np.asarray(envelope, dtype=float)
    baseline_envelope = np.asarray(baseline_envelope, dtype=float)
    if envelope.shape != baseline_envelope.shape:
        raise ValueError(
            f"envelope {envelope.shape} and baseline {baseline_envelope.shape} "
            f"must be the same shape; both must come from the same settings"
        )
    return envelope - baseline_envelope


def detect_echo(
    excess: np.ndarray,
    fs: float,
    speed_mps: float = 343.0,
    min_range_m: float = 1.0,
    max_range_m: float = 5.0,
) -> dict:
    """
    Pick the strongest return inside a range window and say how believable it is.

    min_range_m exists because the speaker is still ringing for the first
    metre or so; anything found in there is the laptop hearing itself.

    Quality is the peak height measured in standard deviations of the rest of
    the window. Around 6 sigma or more is a real object; a few sigma is the
    clutter floor moving about.

    Returns:
        dict with "distance_m", "quality_sigma" and "peak_index".
    """
    excess = np.asarray(excess, dtype=float)
    if excess.ndim != 1:
        raise ValueError(f"excess must be 1D, got shape {excess.shape}")
    if fs <= 0 or speed_mps <= 0:
        raise ValueError("fs and speed_mps must be positive")
    if not 0 <= min_range_m < max_range_m:
        raise ValueError(
            f"need 0 <= min_range_m < max_range_m (got {min_range_m}, {max_range_m})"
        )

    first = int(2 * min_range_m / speed_mps * fs)
    last = min(int(2 * max_range_m / speed_mps * fs), excess.size)
    if first >= last:
        raise ValueError(
            f"range window {min_range_m}-{max_range_m} m is empty for a "
            f"{excess.size}-sample trace at fs={fs}"
        )

    window = excess[first:last]
    peak_index = first + int(np.argmax(window))
    spread = float(np.std(window))
    return {
        "distance_m": speed_mps * peak_index / fs / 2,
        "quality_sigma": float(excess[peak_index] / spread) if spread > 0 else 0.0,
        "peak_index": peak_index,
    }


def analyse_recording(
    recording: np.ndarray,
    chirp: np.ndarray,
    frame_samples: int,
    fs: float,
    baseline_envelope: np.ndarray | None = None,
    speed_mps: float = 343.0,
    min_range_m: float = 1.0,
    max_range_m: float = 5.0,
) -> dict:
    """
    Whole chain from a raw microphone recording to a distance.

    Matched filter, stack the repeated frames, set the time origin at the
    direct arrival, subtract the open-space baseline, then look for the
    strongest thing left inside the range window.

    Pass baseline_envelope=None when recording the baseline itself; the
    returned "envelope" is what you save and feed back in next time.

    Returns:
        dict with "envelope", "excess", "distance_m", "quality_sigma" and
        "peak_index".
    """
    correlation = np.correlate(np.asarray(recording, dtype=float),
                               np.asarray(chirp, dtype=float), mode="full")
    envelope = align_to_direct_path(fold_frames(correlation, frame_samples))
    excess = envelope if baseline_envelope is None else remove_clutter(envelope, baseline_envelope)

    detection = detect_echo(
        excess,
        fs,
        speed_mps=speed_mps,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
    )
    return {"envelope": envelope, "excess": excess, **detection}
