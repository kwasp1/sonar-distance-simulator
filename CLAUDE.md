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
Everything is numpy arrays. The only possible hardware touchpoint is an
OPTIONAL stretch goal: an acoustic validation test using a laptop's own
speaker + mic — not required, not core scope.

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
| 5. Evaluation sweeps | `src/evaluate.py` | `run_snr_sweep()` |
| Display (notebook) | `notebooks/` | calls `src/` functions, does the plotting |
| Display (webapp, LAST) | `app/streamlit_app.py` | thin wrapper, calls `src/` functions only |

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

Week 1 priority: `tests/test_pipeline.py` — noiseless, no-attenuation case
must recover the exact true distance before any noise/filtering work starts.
Get this green first; everything else builds on top of a known-correct
delay/distance pipeline.

## Explicitly out of scope

No ROS2, no MAVLink, no embedded/microcontroller code, no GPIO/serial
hardware calls anywhere in this repo. If a suggestion involves any of
those, it's out of scope for this project — flag it rather than adding it.

## Timeline (4-week plan, targeting 3-4 weeks)

| Week | Focus |
|---|---|
| 1 | `pulse.py` + `channel.py`, noiseless sanity test passing |
| 2 | `receiver.py` matched filter, AWGN, SNR sweep + RMSE plot |
| 3 | `filters.py` band-pass, bandwidth/resolution study |
| 4 | Evaluation, report, demo — `app/streamlit_app.py` only if ahead of schedule |

## Team ownership (primary, not exclusive)

- **Aditya (2305178)**: `src/pulse.py`, `src/receiver.py`, bandwidth/resolution analysis
- **Teammate**: `src/channel.py`, `src/filters.py`, `src/evaluate.py`, report compilation

Both: integration testing, final evaluation sweep, report/presentation.
