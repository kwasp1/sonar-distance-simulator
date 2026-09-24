# Sonar Distance Simulator — Project Context

CSE220 Signals and Systems course project (Team SevenEight, 2-person team).
Full spec: `docs/Sonar_Distance_Simulator_Documentation.docx`.

## What this project is

A software simulation (Python) of pulse-echo ranging — the principle behind
sonar/radar/LiDAR. Generate a known transmit pulse, simulate a noisy delayed
echo, recover the delay via matched filtering (cross-correlation), and
convert it to a distance estimate. Study how noise (SNR) and pulse
bandwidth affect estimation accuracy.

This is **not a hardware project**. No microcontroller, no physical sensor.
Everything is numpy arrays. The one hardware touchpoint is the acoustic
validation stretch goal — a laptop's own speaker + mic — which is now
**implemented and validated** (`src/acoustic.py`, captured with
`physical_sonar_check/record_acoustic.py`). Two real detections confirm it:
a wall at 1.797 m and a hand at 0.272 m, both above 6 sigma.

## Non-negotiable architecture rule

**Functions in `src/` return data. They never plot or print.**

```python
# GOOD — src/receiver.py
def estimate_distance(received_signal, pulse, fs):
    correlation = np.correlate(received_signal, pulse, mode="full")
    peak_index = np.argmax(correlation)
    distance = ...
    return distance, correlation   # just data

# BAD — never do this inside src/
def estimate_distance(received_signal, pulse, fs):
    ...
    plt.plot(correlation); plt.show()   # NO — breaks reuse in Streamlit later
    print(f"Distance: {distance}")      # NO
```

Why: the same `src/` functions get called from three places —
`notebooks/` (Matplotlib), `tests/` (assertions), and eventually
`app/streamlit_app.py` (Streamlit, added LAST, only if time allows). If
plotting/printing leaks into `src/`, adding the webapp means rewriting
core logic instead of just writing a new thin display file.

## Pipeline stages -> files

| Stage | File | Core function(s) |
|---|---|---|
| 1. Transmit pulse | `src/pulse.py` | `generate_pulse()`, `pulse_bandwidth_hz()` |
| 2. Channel + noise | `src/channel.py` | `simulate_channel()`, `add_noise()` |
| 3. Matched filter | `src/receiver.py` | `matched_filter()` |
| 4. Distance estimate | `src/receiver.py` | `estimate_distance()` |
| Optional filtering | `src/filters.py` | `bandpass_filter()` |
| 5. Evaluation sweeps | `src/evaluate.py` | `run_snr_sweep()`, `run_bandwidth_resolution_sweep()` |
| Playback | `src/audio.py` | `make_audible()`, `audible_echo()`, `to_wav_bytes()` |
| Real acoustic ranging | `src/acoustic.py` | `analyse_recording()`, `design_chirp()` |
| Display (notebook) | `notebooks/` | calls `src/` functions, does the plotting |
| Display (webapp) | `app/streamlit_app.py` | **owned by Farhan — do not edit** |

`src/audio.py` and `src/acoustic.py` follow the same rule: they return arrays
and `.wav` bytes, and never play, record, plot or print. Recording audio
happens only in `physical_sonar_check/record_acoustic.py`.

`app/streamlit_app.py` imports ten functions from `src/`. Keep those
signatures stable; add new functions rather than reshaping existing ones, and
flag it if a breaking change is genuinely needed.

Function signatures/docstrings are already stubbed in each file — implement
against those contracts rather than changing them, so notebook/app code
that calls them doesn't break.

## Conventions

- SI units throughout: seconds, Hz, meters, m/s. Never mix ms/s or cm/m
  silently — keep unit suffixes in variable/arg names (`duration_s`,
  `freq_hz`, `distance_m`) as already done in the stubs.
- Default speed of sound: 343 m/s (air), passed as a parameter, not hardcoded
  in multiple places.
- Type hints on every function signature (already present in stubs — keep them).
- `scipy.signal` for filter design (Butterworth via `butter` + `filtfilt`),
  not hand-rolled filter coefficients.

## Testing approach

29 pytest cases plus 17 checks in `check_pulse.py`. Run with
`python -m pytest -q` and `python check_pulse.py`.

| File | Gates |
|---|---|
| `tests/test_pipeline.py` | Noiseless exact-delay recovery, 4 delays x 3 pulse types |
| `tests/test_multipath.py` | Two objects detected, one object not split |
| `tests/test_resolution.py` | Resolution improves as bandwidth grows |
| `tests/test_audio.py` | Playback shape, loudness, loop-safety, gain reuse |
| `tests/test_acoustic.py` | Re-derives an archived real recording bit-exactly |

`tests/test_acoustic.py` reads `physical_sonar_check/rx.npy`,
`baseline.npy` and `excess.npy`. Do not delete those three.

`src/filters.py` currently has no pytest coverage — the one gap. If you touch
it, be aware that swapping `filtfilt` for `lfilter` would silently shift every
distance estimate by the filter's group delay and nothing would catch it.

## Explicitly out of scope

No ROS2, no MAVLink, no embedded/microcontroller code, no GPIO/serial
hardware calls anywhere in this repo. If a suggestion involves any of
those, it's out of scope for this project — flag it rather than adding it.

## Status

**All nine committed features are implemented and tested.** What remains is
report production: figures into `results/`, then the write-up.

| Week | Planned | Actual |
|---|---|---|
| 1 | `pulse.py` + `channel.py`, noiseless test | done |
| 2 | `receiver.py`, AWGN, SNR sweep | done |
| 3 | `filters.py`, bandwidth/resolution study | done |
| 4 | Evaluation, report, demo | code done; report outstanding |

Two things landed beyond the original plan: `src/audio.py` (makes any signal
listenable) and `src/acoustic.py` (real speaker/microphone ranging, validated
against a wall at 1.797 m and a hand at 0.272 m).

## Team ownership

- **Aditya (2305178)**: all of `src/`, `tests/`, evaluation sweeps, acoustic
  validation
- **Farhan (2305177)**: `app/streamlit_app.py`

Both: report and presentation.

**`app/streamlit_app.py` is Farhan's file — do not edit it.** It imports ten
functions from `src/`, so keep those signatures stable; add new functions
rather than reshaping existing ones, and flag it if a breaking change is
genuinely needed. `TEAMMATE_GUIDE.md` is the API reference written for him.
