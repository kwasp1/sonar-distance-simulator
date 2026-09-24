# Sonar Distance Simulator — Comprehensive Project Overview

> **CSE220 Signals and Systems** | **Team SevenEight**  
> **Contributors:** Aditya Tirtho Roy (2305178) & Farhan Ehsas Sami (2305177)

---

## 1. Executive Summary & Core Concept

This project is a **software simulation of pulse-echo ranging**—the fundamental principle powering **Sonar**, **Radar**, and **LiDAR**.

```
[ Transmit Pulse ]  ───►  [ Propagate to Object ]  ───►  [ Reflect / Echo ]  ───►  [ Matched Filter ]
        x(t)                        delay = 2d / c                    y(t) = α·x(t - τ) + n(t)      Peak detection -> Distance d
```

### The Physics in One Equation:
$$\text{distance} = \frac{\text{speed of sound} \times \text{round-trip time}}{2} = \frac{c \cdot \Delta t}{2}$$
- Default speed of sound: $c = 343\text{ m/s}$ (in air).
- The factor of 2 accounts for sound traveling to the target and reflecting back.

### What the Project is Investigating:
1. **Noise Resilience:** How much noise (in dB SNR) can the system endure before matched filtering fails to detect the echo? (Plotted as **SNR vs. RMSE**).
2. **Range Resolution:** How close can two reflectors be before their individual echo peaks merge into a single unrecognizable blob? (Plotted as **Bandwidth vs. Minimum Resolvable Separation**).

---

## 2. Core Architecture & Golden Rule

> **The Non-Negotiable Rule:**  
> **Functions in `src/` take numbers/arrays and return numbers/arrays. They NEVER call `plt.show()` or `print()`.**

Why? Because the exact same functions in `src/` are imported and consumed by:
1. **Unit Tests** (`tests/`) — automated assertions.
2. **Analysis Notebooks** (`notebooks/`) — Matplotlib charts for the report.
3. **Interactive Web App** (`app/streamlit_app.py`) — Streamlit sliders and UI.

---

## 3. Directory Map

```
sonar-distance-simulator/
├── src/                          # Core DSP pipeline (pure functions only)
│   ├── pulse.py                  # Stage 1: Pulse generation & bandwidth measurement
│   ├── channel.py                # Stage 2: Time delay, attenuation, multi-target & AWGN
│   ├── receiver.py               # Stages 3 & 4: Matched filter & distance estimation
│   ├── filters.py                # Phase C: Bandpass pre-processing filter
│   ├── evaluate.py               # Stage 5: Monte Carlo sweeps (SNR vs RMSE, Resolution)
│   ├── audio.py                  # Playback: makes any signal listenable (.wav bytes)
│   └── acoustic.py               # Feature 8: analysis of real speaker/mic recordings
├── tests/                        # Automated validation
│   ├── test_pipeline.py          # Week 1 gate: noiseless delay/distance recovery
│   ├── test_multipath.py         # Multi-object detection sanity tests
│   ├── test_resolution.py        # Week 3 gate: resolution improves with bandwidth
│   ├── test_audio.py             # Playback shape, loudness, loop-safety
│   └── test_acoustic.py          # Re-derives an archived real recording exactly
├── check_pulse.py                # 17 automated checks for pulse math & stability
├── app/                          # Streamlit interactive web application
│   └── streamlit_app.py          # OWNED BY FARHAN -- do not edit
├── notebooks/                    # Jupyter notebooks for report plots
├── physical_sonar_check/         # Real acoustic capture (Mac/PC mic + speaker)
│   ├── record_acoustic.py        # The only file that touches speaker/mic
│   ├── band_check.py             # Measures speaker/mic frequency response
│   ├── band_check.py             # Measures your speaker/mic frequency response
│   └── *.npy                     # Archived recordings the acoustic tests read
└── requirements.txt              # Project dependencies (numpy, scipy, matplotlib, pytest)
```

---

## 4. Module & Function Reference (`src/`)

### 4.1. `src/pulse.py` — Transmit Pulse Generator
Generates the outgoing probing sound and analyzes its frequency characteristics.

| Function | Signature & Inputs | Output | Description & Purpose |
|---|---|---|---|
| `_normalize` | `signal: np.ndarray` | `np.ndarray` | Scales any pulse to **unit energy** ($\sum x[n]^2 = 1$). Ensures that comparing rect, gaussian, and chirp pulses under noise is fair. |
| `generate_pulse` | `pulse_type: str`<br>`duration_s: float`<br>`fs: float`<br>`freq_hz: float`<br>`bandwidth_hz: float \| None` | `1D np.ndarray` | Creates a normalized transmit waveform centered at $t = 0$.<br>• `"rect"`: Abrupt on/off sinusoidal burst.<br>• `"gaussian"`: Tone with Gaussian bell envelope ($\sigma = \frac{\text{duration}}{6}$); smooth fade prevents high-frequency spectral splatter.<br>• `"chirp"`: Linear frequency sweep from $f_{\text{start}}$ to $f_{\text{end}}$. Gives high energy over time while retaining wide bandwidth for sharp resolution. |
| `pulse_bandwidth_hz` | `pulse: np.ndarray`<br>`fs: float` | `float` (Hz) | Calculates the **-3 dB (half-power)** bandwidth using zero-padded real FFT (`np.fft.rfft`). Finds the frequency span where spectral power $\ge P_{\max} / 2$. Stable across sampling rates. Supplies the x-axis for resolution experiments. |

---

### 4.2. `src/channel.py` — Acoustic Propagation & Noise
Simulates what happens to sound as it travels through the air, reflects, and gets captured by a microphone.

| Function | Signature & Inputs | Output | Description & Purpose |
|---|---|---|---|
| `simulate_channel` | `pulse: np.ndarray`<br>`true_distance_m: float`<br>`fs: float`<br>`speed_mps: float = 343.0`<br>`attenuation: float = 1.0`<br>`buffer_duration_s: float = 0.05` | `1D np.ndarray` (buffer) | Embeds a delayed, attenuated copy of `pulse` into a silent buffer of duration `buffer_duration_s`. Delay in samples is $k = \text{round}\left(\frac{2d}{c} \cdot f_s\right)$. Uses `+=` so multiple reflections can be accumulated. |
| `add_noise` | `signal: np.ndarray`<br>`snr_db: float`<br>`seed: int \| None = None` | `1D np.ndarray` | Adds Additive White Gaussian Noise (AWGN). **Crucial detail:** Calculates signal power only over non-zero echo samples (not the entire silent buffer) so the requested SNR remains physically identical regardless of buffer length. Supports a random seed for reproducible tests. |
| `simulate_multi_object_channel` | `pulse: np.ndarray`<br>`distances_m: list[float]`<br>`fs: float`<br>`attenuations: list[float] \| None`<br>`**kwargs` | `1D np.ndarray` | Simulates an acoustic scene with multiple targets at different distances by calling `simulate_channel` for each target and superimposing the received signals. |

---

### 4.3. `src/receiver.py` — Matched Filter & Distance Estimator
Recovers the weak echo buried in noise and calculates target distance.

| Function | Signature & Inputs | Output | Description & Purpose |
|---|---|---|---|
| `matched_filter` | `received_signal: np.ndarray`<br>`pulse: np.ndarray` | `1D np.ndarray` (cross-correlation) | Computes `np.correlate(received_signal, pulse, mode='full')`. Mathematically the optimal linear filter for maximizing SNR in the presence of white noise. |
| `estimate_distance` | `received_signal: np.ndarray`<br>`pulse: np.ndarray`<br>`fs: float`<br>`speed_mps: float = 343.0` | `(estimated_distance_m, correlation)` | Runs matched filter, detects peak with `np.argmax`, adjusts for the `len(pulse) - 1` offset from `'full'` correlation mode, and converts delay samples back to meters: $d = \frac{k}{f_s} \cdot \frac{c}{2}$. |
| `estimate_multiple_distances` | `received_signal: np.ndarray`<br>`pulse: np.ndarray`<br>`fs: float`<br>`speed_mps: float = 343.0`<br>`prominence_frac: float = 0.4`<br>`min_separation_samples: int \| None = None` | `list[float]` (sorted distances) | Detects multiple targets in multi-path / multi-object scenarios. Computes the **analytic envelope** using Hilbert transform (`scipy.signal.hilbert`), then uses `scipy.signal.find_peaks` to identify distinct echo crests. |

---

### 4.4. `src/filters.py` — Bandpass Pre-processing Filter
Rejects out-of-band ambient noise before the signal enters the matched filter.

| Function | Signature & Inputs | Output | Description & Purpose |
|---|---|---|---|
| `bandpass_filter` | `signal: np.ndarray`<br>`fs: float`<br>`low_hz: float`<br>`high_hz: float`<br>`order: int = 4` | `1D np.ndarray` | Implements a zero-phase Butterworth bandpass filter using `scipy.signal.butter` and forward-backward filtering (`scipy.signal.filtfilt`). Cleans up raw audio prior to matched filtering. |

---

### 4.5. `src/evaluate.py` — Automated Evaluation & Sweeps
Runs repeated Monte Carlo experiments to generate the performance curves needed for the course report.

| Function | Signature & Inputs | Output | Description & Purpose |
|---|---|---|---|
| `run_snr_sweep` | `true_distance_m: float`<br>`snr_values_db: list[float]`<br>`fs: float`<br>`pulse_type: str = "gaussian"`<br>`trials_per_snr: int = 20`<br>`bandpass: bool = False`<br>... | `dict` containing:<br>`"snr_db"`: list<br>`"rmse_m"`: list | Simulates repeated noisy transmissions at each SNR level (e.g. from -10 dB to +30 dB). Computes Root Mean Square Error (RMSE) in meters to assess performance degradation under harsh noise. |
| `run_bandwidth_resolution_sweep` | `pulse_configs: list[dict]`<br>`fs: float`<br>`trials: int = 12`<br>`success_frac: float = 0.5`<br>`min_separation_samples: int = 8`<br>... | `dict` containing:<br>`"labels"`, `"bandwidth_hz"`,<br>`"resolved_m"`, `"theory_m"` | Places two targets, shrinks their separation until the receiver can no longer report two, repeats across pulses of different bandwidth. **Must override `min_separation_samples`**: the receiver's `len(pulse)//2` default is a floor set by pulse length, not bandwidth, and pins every chirp at 17.0 cm regardless of sweep width. A detection must also land near a true target, since interference nulls otherwise read as extra peaks. |

**Result:** measured resolution tracks *c*/(2*B*) at roughly **0.55x for gaussians**, monotonically across a 23x bandwidth range. Gaussians alone cannot prove the point — their bandwidth is tied to their duration — so chirps at fixed 2 ms duration carry the argument.

---

### 4.6. `src/audio.py` — Listenable playback
Turns simulation arrays into sound. A 2 ms pulse is too short and too quiet to hear, and looping it raw produces a buzz.

| Function | Signature & Inputs | Output | Description & Purpose |
|---|---|---|---|
| `make_audible` | `signal: np.ndarray \| list`<br>`fs: float`<br>`gap_s: float = 0.6`<br>`repeats: int = 4`<br>`slowdown: float = 4.0`<br>`gain: float \| None = None` | `dict`: `"audio"`,<br>`"rate_hz"`, `"gain"` | Adds a silence gap after each copy so a loop sounds like "ping ... ping" rather than a drone; fades the edges so the loop seam does not click. **Nothing is resampled** — the slowdown is applied by declaring a lower playback rate, which stretches the sound and drops its pitch for free. |
| `audible_echo` | `pulse`, `distance_m`,<br>`fs`, `snr_db`,<br>`frames: int = 4`, `seed: int = 0` | `dict`, as above | The received version: delayed echo plus hiss. Each frame gets **its own noise seed**, because repeating one noisy buffer makes the hiss cycle audibly and sounds fake. |
| `to_wav_bytes` | `audio: np.ndarray`<br>`rate_hz: float` | `bytes` | Packs to `.wav`. The playback rate lives in the header, so the slowdown is invisible to whoever plays it. |

**Gotcha:** when comparing clips at different SNR, pass one shared `gain` taken from the **loudest** (noisiest) clip. Per-clip normalisation erases the difference you are trying to demonstrate; taking the gain from the quietest clip makes every other one clip.

---

### 4.7. `src/acoustic.py` — Real speaker + microphone ranging (Feature 8)
The same ranging maths applied to audio actually recorded in a room.

| Function | Signature & Inputs | Output | Description & Purpose |
|---|---|---|---|
| `design_chirp` | `f_start_hz`, `f_end_hz`,<br>`duration_s`, `fs` | `np.ndarray` | Hanning-windowed sweep specified by its band, because what matters on real hardware is the band the speaker can reproduce. Deliberately not `generate_pulse`, which is unwindowed by design. |
| `fold_frames` | `correlation`, `frame_samples` | `np.ndarray` | Averages every transmitted frame together. The echo lands in the same place each time, room noise does not, so stacking grows the echo and averages the noise away. |
| `align_to_direct_path` | `folded: np.ndarray` | `np.ndarray` | The microphone hears the speaker directly long before any echo. That arrival is the time origin. |
| `remove_clutter` | `envelope`, `baseline_envelope` | `np.ndarray` | Subtracts a recording of the unchanged room. Speaker ringing, chassis buzz and furniture cancel; only what changed survives. |
| `detect_echo` | `excess`, `fs`,<br>`min_range_m`, `max_range_m` | `dict`: `"distance_m"`,<br>`"quality_sigma"` | Strongest return in a range window, with quality in standard deviations. Above ~6 sigma is a real object. |
| `analyse_recording` | all of the above | `dict` | The whole chain in one call. |

**Validated against reality twice:**

| Target | Measured | Quality |
|---|---|---|
| Wall (archived capture) | 1.797 m | 6.7 sigma |
| Hand held ~30 cm above the keyboard | 0.272 m | 6.2 sigma |

`tests/test_acoustic.py` re-derives the archived `excess.npy` from the raw recording to 12 decimal places, so this stays gated with no microphone attached.

---

## 5. Testing & Validation Tools

- **`tests/test_pipeline.py`**: Runs Pytest across multiple sample delays to verify that in a noiseless channel, the receiver recovers the exact true distance.
- **`tests/test_multipath.py`**: Asserts that two separate objects (at 5.0 m and 6.0 m) are both detected within 1 cm accuracy, and that a single object is never falsely split.
- **`check_pulse.py`**: 17 comprehensive automated assertions checking pulse duration, unit energy, Gaussian centroid symmetry, theoretical bandwidth formulas, and sampling-rate stability.

---

## 6. Physical Sonar Check (`physical_sonar_check/`)
*Optional stretch goal / real-world acoustic experimentation:*

- **`record_acoustic.py`**: The capture script. Records a baseline facing open space, then measures against it. All maths is delegated to `src/acoustic.py`; this file only does audio I/O.
- **`band_check.py`**: Plays a slow sweep and measures your machine's actual speaker and microphone response. Run this first if you change the chirp band.
- **`rx.npy`, `baseline.npy`, `excess.npy`**: Archived capture of a wall at 1.797 m. `tests/test_acoustic.py` re-derives `excess.npy` from `rx.npy` exactly, so do not delete these.

The exploratory probe scripts (`sonar_probe_v1`–`v4`, `sonar_detect`, `sonar_live`, `view.py`) were removed once `record_acoustic.py` superseded them; they remain in git history.

---

## 7. Common Issues & Quick Fixes

### "Cannot find module `sounddevice`" / `numpy` / `scipy`
The packages live in the project virtual environment (`sonar-env`), but the
IDE is probably pointing at the global Python interpreter.

1. Press `Ctrl + Shift + P` (or `Cmd + Shift + P`).
2. Type **`Python: Select Interpreter`** and hit Enter.
3. Pick `sonar-env` — `sonar-env/bin/python`, or
   `sonar-env\Scripts\python.exe` on Windows.

`sounddevice` is only needed for real recording. Everything else — the whole
simulation and all 29 tests — runs without it.

### `run_snr_sweep` raises about `low_hz` or `high_hz`
The auto-derived band-pass cutoffs fell outside `0 .. fs/2`. Bandwidth scales
as roughly `1/duration_s`, so a **longer** `duration_s` narrows the band and
fixes a negative `low_hz`; a **lower** `freq_hz` fixes a `high_hz` past Nyquist.

---

## 8. Summary Table: What Each File Does

| File | Primary Role | Output Type |
|---|---|---|
| [`src/pulse.py`](src/pulse.py) | Generates Rect / Gaussian / Chirp pulses; computes -3 dB bandwidth | NumPy arrays / float |
| [`src/channel.py`](src/channel.py) | Simulates delay, attenuation, multiple reflectors, and AWGN | NumPy arrays |
| [`src/receiver.py`](src/receiver.py) | Cross-correlation matched filtering & distance estimation | Distance (m), correlation array |
| [`src/filters.py`](src/filters.py) | Butterworth bandpass pre-processing | Filtered NumPy array |
| [`src/evaluate.py`](src/evaluate.py) | Monte Carlo SNR sweeps and resolution benchmarks | Dictionaries with metric lists |
| [`src/audio.py`](src/audio.py) | Makes any pulse or received buffer listenable | NumPy arrays / .wav bytes |
| [`src/acoustic.py`](src/acoustic.py) | Ranging on real speaker/microphone recordings | Distance (m), quality (sigma) |
| [`check_pulse.py`](check_pulse.py) | 17 pass/fail checks on pulse generator physics | Console test report |
| [`tests/`](tests/) | Pytest test suite for noiseless recovery & multi-object detection | Pytest pass/fail |
| [`app/streamlit_app.py`](app/streamlit_app.py) | Interactive web application — **owned by Farhan, do not edit** | Streamlit web UI |
| [`physical_sonar_check/`](physical_sonar_check/) | Experimental real laptop mic/speaker acoustic sonar | Real-time audio I/O & Matplotlib |
