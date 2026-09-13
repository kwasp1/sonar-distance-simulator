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
│   └── evaluate.py               # Stage 5: Monte Carlo sweeps (SNR vs RMSE, Resolution)
├── tests/                        # Automated validation
│   ├── test_pipeline.py          # Week 1 gate: noiseless delay/distance recovery
│   └── test_multipath.py         # Multi-object detection sanity tests
├── check_pulse.py                # 17 automated checks for pulse math & stability
├── check_multipath.py            # Exploration script for multi-target resolution limits
├── app/                          # Optional Streamlit interactive web application
│   └── streamlit_app.py          # UI sliders for true distance and SNR
├── notebooks/                    # Jupyter notebooks for report plots
├── physical_sonar_check/         # Optional hardware stretch goal (Mac/PC mic + speaker)
│   ├── band_check.py             # Measures speaker/mic frequency response
│   ├── sonar_detect.py           # Real-time ultrasonic target detector
│   ├── sonar_live.py             # Moving-target indication (MTI) display
│   ├── sonar_probe_v*.py         # Physical acoustic chirp echo ranging
│   └── view.py                   # Plotting helper for recorded numpy traces
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
| `run_bandwidth_resolution_sweep` | `fs: float`<br>`pulse_durations_s: list[float]`<br>`separation_range_m: tuple[float, float]` | `dict` (durations, bandwidths, min_resolutions) | *(In progress)* Evaluates the minimum physical separation needed between two targets for both to be resolved as separate peaks, as a function of pulse bandwidth. |

---

## 5. Testing & Validation Tools

- **`tests/test_pipeline.py`**: Runs Pytest across multiple sample delays to verify that in a noiseless channel, the receiver recovers the exact true distance.
- **`tests/test_multipath.py`**: Asserts that two separate objects (at 5.0 m and 6.0 m) are both detected within 1 cm accuracy, and that a single object is never falsely split.
- **`check_pulse.py`**: 17 comprehensive automated assertions checking pulse duration, unit energy, Gaussian centroid symmetry, theoretical bandwidth formulas, and sampling-rate stability.
- **`check_multipath.py`**: Tests resolution limits by gradually bringing two reflectors closer (50 cm down to 1 cm) and measuring false-split rates across SNR.

---

## 6. Physical Sonar Check (`physical_sonar_check/`)
*Optional stretch goal / real-world acoustic experimentation:*

- **`band_check.py`**: Plays a chirp and measures actual hardware speaker and microphone response on your machine.
- **`sonar_detect.py`**: Real-time ultrasonic rangefinder (15–21 kHz) with MTI clutter cancellation and distance threshold detection.
- **`sonar_live.py`**: Continuous MTI radar display; filters static background reflections (desk, walls) and tracks moving hands.
- **`sonar_probe_v1` – `v4.py`**: Acoustic chirp echo ranging. v4 records an open-space baseline to subtract chassis/speaker ringdown, isolating wall echoes.
- **`view.py`**: Quick plotting tool to visualize saved `.npy` traces (`rx.npy`, `corr.npy`, `excess.npy`).

---

## 7. Common Issues & Quick Fixes

### Why did "Cannot find module `sounddevice`" happen?
When you ran `pip install sounddevice` inside your activated terminal, it was installed inside the virtual environment:
`d:\L-2, T-2\CSE 220\Project\sonar-distance-simulator\venv`

However, your IDE / editor was pointing to the global Python interpreter:
`C:\Users\ASUS\AppData\Local\Programs\Python\Python313\python.exe`

#### How to fix in VS Code / IDE:
1. Press `Ctrl + Shift + P` (or `Cmd + Shift + P`).
2. Type **`Python: Select Interpreter`** and hit Enter.
3. Select the virtual environment:
   `.\venv\Scripts\python.exe` (or browse to `d:\L-2, T-2\CSE 220\Project\sonar-distance-simulator\venv\Scripts\python.exe`).
4. The warning will disappear immediately!

---

## 8. Summary Table: What Each File Does

| File | Primary Role | Output Type |
|---|---|---|
| [`src/pulse.py`](src/pulse.py) | Generates Rect / Gaussian / Chirp pulses; computes -3 dB bandwidth | NumPy arrays / float |
| [`src/channel.py`](src/channel.py) | Simulates delay, attenuation, multiple reflectors, and AWGN | NumPy arrays |
| [`src/receiver.py`](src/receiver.py) | Cross-correlation matched filtering & distance estimation | Distance (m), correlation array |
| [`src/filters.py`](src/filters.py) | Butterworth bandpass pre-processing | Filtered NumPy array |
| [`src/evaluate.py`](src/evaluate.py) | Monte Carlo SNR sweeps and resolution benchmarks | Dictionaries with metric lists |
| [`check_pulse.py`](check_pulse.py) | 17 pass/fail checks on pulse generator physics | Console test report |
| [`check_multipath.py`](check_multipath.py) | Multi-target resolution & false-positive exploration | Console table |
| [`tests/`](tests/) | Pytest test suite for noiseless recovery & multi-object detection | Pytest pass/fail |
| [`app/streamlit_app.py`](app/streamlit_app.py) | Interactive web application with sliders | Streamlit web UI |
| [`physical_sonar_check/`](physical_sonar_check/) | Experimental real laptop mic/speaker acoustic sonar | Real-time audio I/O & Matplotlib |
