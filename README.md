# 📡 Sonar Distance Simulator

> **CSE 220: Signals and Systems — Final Project**  
> **Team SevenEight**  
> **Authors:** Aditya Tirtho Roy (2305178) & Farhan Ehsas Sami (2305177)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-33%20passed-brightgreen.svg)]()
[![UI](https://img.shields.io/badge/Streamlit-Interactive%20App-red.svg)](https://streamlit.io/)
[![Course](https://img.shields.io/badge/Course-CSE220%20Signals%20%26%20Systems-purple.svg)]()

---

## 1. Executive Summary & Physics Principle

The **Sonar Distance Simulator** is an end-to-end Digital Signal Processing (DSP) simulation and physical acoustic experiment of **pulse-echo ranging** — the foundational principle behind **Sonar**, **Radar**, and **LiDAR**.

```
[ Transmit Pulse x(t) ] ───► [ Acoustic Channel ] ───► [ Matched Filter Receiver ] ───► [ Peak Detection ]
       (Rect / Gauss / Chirp)        Delay τ = 2d/c             Cross-Correlation R_xy           Distance d = c·τ / 2
                                    Attenuation + AWGN
```

### The Governing Ranging Equation
$$d = \frac{c \cdot \Delta t}{2}$$

* **$c$**: Speed of sound in medium ($343\text{ m/s}$ in ambient air at $20^\circ\text{C}$).
* **$\Delta t$**: Round-trip time of flight (ToF).
* **Division by 2**: Accounts for the two-way propagation to and from the target.

### Core Scientific Questions Investigated
1. **Noise Resilience (SNR vs. RMSE)**: How much additive Gaussian noise can the channel sustain before matched filtering loses lock on the echo?
2. **Range Resolution ($c / 2B$)**: How close can two reflective targets be before their individual echo peaks merge into an indistinguishable single blob?

---

## 2. Key Features

* **Advanced Pulse Generation (`src/pulse.py`)**:
  * **Rectangular Tone Burst**: Basic pulsed sinusoid with high spectral sidelobes.
  * **Gaussian-Windowed Pulse**: Smooth temporal decay minimizing spectral leakage.
  * **Linear Frequency Modulated (LFM) Chirp**: Wideband sweep decouples pulse duration from bandwidth, enabling high transmit energy without sacrificing range resolution.
  * **$-3\text{ dB}$ (Half-Power) Bandwidth Estimator**: Data-driven FFT spectral analysis.
* **Realistic Channel Model (`src/channel.py`)**:
  * Multi-target multipath reflection with independent distance and reflection coefficients ($\alpha_i$).
  * Spherical wave geometric attenuation ($1 / r^2$).
  * Additive White Gaussian Noise (AWGN) calibrated to exact target Signal-to-Noise Ratios (SNR in dB).
* **Optimal Receiver & Matched Filter (`src/receiver.py`)**:
  * Time-domain cross-correlation matching the template pulse against the noisy channel output.
  * Single-target and multi-target peak detection with peak prominence and thresholding.
* **Bandpass Pre-Filtering (`src/filters.py`)**:
  * 4th-order Butterworth bandpass filter with forward-backward zero-phase filtering (`scipy.signal.filtfilt`).
* **Audible Audio Sonification (`src/audio.py`)**:
  * Frequency-shifting and gain normalization to produce listenable `.wav` sonar pings and echoes.
* **Physical Hardware Acoustic Ranging (`src/acoustic.py` & `physical_sonar_check/`)**:
  * Turns a regular laptop speaker and microphone into a working physical sonar transceiver.
  * Uses 2–8 kHz Hann-windowed chirps, synchronous frame averaging, direct-path arrival calibration, and baseline clutter subtraction.
  * **Empirically validated on real targets**:
    * **Concrete Wall**: Measured at **$1.797\text{ m}$** with **$6.75\sigma$** statistical confidence.
    * **Hand Reflection**: Measured above keyboard at **$0.272\text{ m}$** with **$8.1\sigma$** statistical confidence.
* **Interactive Streamlit Web Dashboard (`app/streamlit_app.py`)**:
  * Full real-time web application with parameter sliders, live waveform and spectrum plots, correlation graphs, multi-target scene visualizer, and in-browser audio playback.

---

## 3. System Architecture & Golden Rule

> **The Non-Negotiable Architecture Rule:**  
> **All functions in `src/` are pure mathematical transformations.** They accept NumPy arrays and parameters, and return NumPy arrays and numbers. They **never** call `plt.show()`, `print()`, or hardware playback/record.

This clean separation ensures that the exact same DSP core functions are 100% reusable across:
1. **Automated Unit Tests** (`tests/`)
2. **Batch Simulation Scripts** (`notebooks/make_figures.py`)
3. **Interactive Web Application** (`app/streamlit_app.py`)

```
sonar-distance-simulator/
├── app/                        # Streamlit web application
│   └── streamlit_app.py        # Interactive UI dashboard
├── .streamlit/                 # UI configuration and themes
│   └── config.toml
├── src/                        # Core DSP Library (pure functions)
│   ├── pulse.py                # Pulse generation & bandwidth estimation
│   ├── channel.py              # Delay, attenuation, multipath, AWGN
│   ├── receiver.py             # Matched filter & distance estimation
│   ├── filters.py              # Butterworth bandpass pre-filtering
│   ├── evaluate.py             # SNR and resolution parameter sweeps
│   ├── audio.py                # Sonification & WAV conversion
│   └── acoustic.py             # Physical hardware acoustic sonar engine
├── tests/                      # Pytest automated test suite (33 tests)
├── notebooks/                  # Figure generation
│   └── make_figures.py         # Reproduces report & presentation figures
├── physical_sonar_check/       # Real acoustic experiment data & scripts
│   ├── record_acoustic.py      # Hardware speaker/mic capture script
│   ├── band_check.py           # Hardware frequency response probe
│   ├── baseline.npy            # Archived ambient baseline capture
│   ├── rx.npy                  # Archived target capture (wall)
│   └── excess.npy              # Clutter-subtracted correlation capture
├── presentation/               # Course presentation deliverables
│   ├── presentation.tex        # LaTeX Beamer source
│   ├── main.pdf                # Compiled final slide deck
│   └── *.png                   # High-resolution presentation figures
├── results/                    # Generated simulation figures (fig1 - fig6)
├── docs/                       # Course project documentation (DOCX)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 4. Quick Start & Installation

### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/kwasp1/sonar-distance-simulator.git
cd sonar-distance-simulator

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Web Application
Launch the interactive dashboard in your browser:
```bash
streamlit run app/streamlit_app.py
```

### 3. Run the Automated Test Suite
Run all 33 unit tests:
```bash
python -m pytest -q
```

### 4. Reproduce Benchmark Figures
Regenerate all 6 high-resolution simulation figures into `results/`:
```bash
python notebooks/make_figures.py
```

---

## 5. Mathematical Formulation & DSP Pipeline

### Stage 1: Transmit Pulse Design
The transmitted pulse $s(t)$ has duration $T$ and carrier frequency $f_c$.
* **Gaussian envelope**:
  $$s(t) = \exp\left(-\frac{t^2}{2\sigma^2}\right) \cos(2\pi f_c t)$$
* **Linear Frequency Modulated (LFM) Chirp**:
  $$s(t) = \cos\left(2\pi \left(f_0 t + \frac{B}{2T} t^2\right)\right), \quad 0 \le t \le T$$
  The instantaneous frequency ramps linearly from $f_0$ to $f_0 + B$.

### Stage 2: Acoustic Propagation Channel
For $M$ targets at distances $d_i$ with reflection gains $\alpha_i$:
$$r(t) = \sum_{i=1}^{M} \frac{\alpha_i}{1 + (d_i / d_0)^2} s(t - \tau_i) + w(t), \quad \tau_i = \frac{2 d_i}{c}$$
where $w(t) \sim \mathcal{N}(0, \sigma_n^2)$ is zero-mean Additive White Gaussian Noise.

### Stage 3: Matched Filter Cross-Correlation
The matched filter maximizes output peak SNR in the presence of white noise. The discrete cross-correlation between received signal $y[n]$ and reference pulse $x[n]$ is:
$$R_{xy}[k] = \sum_{n} y[n] \, x[n - k]$$

### Stage 4: Distance Estimation
The estimated time-of-flight $\hat{\tau}$ corresponds to the index of maximum correlation:
$$\hat{\tau} = \frac{\arg\max_k |R_{xy}[k]|}{f_s}, \qquad \hat{d} = \frac{c \cdot \hat{\tau}}{2}$$

### Theoretical Range Resolution
Two objects are resolvable by matched filtering if their separation $\Delta d$ satisfies:
$$\Delta d \ge \frac{c}{2B}$$
where $B$ is the pulse bandwidth. LFM chirps break the classical pulse duration trade-off: duration $T$ can be extended for transmit energy while bandwidth $B$ is independently widened for razor-sharp range resolution.

---

## 6. Physical Acoustic Sonar (Laptop Speaker + Mic)

Beyond simulation, **Feature 8** performs real-world acoustic ranging on consumer hardware.

```
[Laptop Speaker] ──► (Acoustic Chirp 2-8 kHz) ──► [Target Wall / Hand]
                                                         │
[Laptop Mic]     ◄── Direct Path + Wall Echo ────────────┘
       │
       ▼
 1. Cross-Correlate with Analytic Chirp Template
 2. Synchronous Frame Stacking (Averages N pulses to cancel random ambient noise)
 3. Align Direct Speaker-to-Mic Arrival at t = 0
 4. Subtract Ambient Baseline: Excess = Target - Baseline (Removes room clutter & chassis ringing)
 5. Peak Finding: Estimate distance d with quality metric σ (standard deviations above clutter floor)
```

To run a fresh live acoustic measurement using your computer hardware:
```bash
python physical_sonar_check/record_acoustic.py --frames 30
```

---

## 7. Verification & Test Suite

The project includes **33 unit tests** across 6 test modules covering all edge cases, numerical tolerances, and signal recovery bounds:

| Test File | Focus & Verification Gates |
| :--- | :--- |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | Noiseless round-trip delay recovery across all 3 pulse types ($<1\text{ mm}$ error). |
| [`tests/test_multipath.py`](tests/test_multipath.py) | Resolution of two targets; ensures a single target is never falsely split. |
| [`tests/test_resolution.py`](tests/test_resolution.py) | Validates that resolution monotonically improves as pulse bandwidth increases. |
| [`tests/test_audio.py`](tests/test_audio.py) | Verifies audible conversion, gain scaling, shape integrity, and WAV headers. |
| [`tests/test_acoustic.py`](tests/test_acoustic.py) | Bit-exact re-derivation and validation of archived physical acoustic recordings. |
| [`tests/test_sweeps.py`](tests/test_sweeps.py) | Validates automated SNR sweeps and bandwidth resolution sweep matrices. |

---

## 8. Authors & Acknowledgements

* **Aditya Tirtho Roy** (Student ID: 2305178)
* **Farhan Ehsas Sami** (Student ID: 2305177)

*Department of Computer Science and Engineering, Bangladesh University of Engineering and Technology (BUET)*  
*Course: CSE 220 — Signals and Systems*
