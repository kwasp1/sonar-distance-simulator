# Physical Acoustic Hardware Sonar Guide (Feature 8)

> **CSE220 Signals and Systems — Team SevenEight**  
> **Aditya Tirtho Roy (2305178) & Farhan Ehsas Sami (2305177)**

---

## 1. Overview & Objective

**Feature 8** turns your standard personal computer or laptop into an active **ultrasonic/acoustic sonar transceiver** using only the built-in speaker and microphone.

Rather than relying purely on simulated mathematical buffers, this feature validates pulse-echo distance estimation against the real physical world by:
1. Transmitting acoustic linear frequency sweeps (chirps) into the ambient environment.
2. Capturing physical sound wave reflections off real room boundaries (walls, doors, flat obstacles).
3. Compensating for non-ideal physical acoustics (direct speaker-to-mic leakage, chassis ringing, room clutter, and frequency rolloff) using signals and systems theory.

Two real-world targets have been physically validated using this system:
- **Concrete Wall**: Measured at **1.797 m** with **6.75 σ** statistical detection confidence.
- **Hand Reflection**: Measured above keyboard at **0.272 m** with **8.1 σ** detection confidence.

---

## 2. Core Signals & Systems Principles

Operating an acoustic sonar on consumer hardware presents challenges not found in ideal simulation. The physical pipeline resolves these with five DSP techniques:

```
[Speaker Output] ---> [Room Reflection] ---> [Microphone Capture]
                                                      │
┌─────────────────────────────────────────────────────┘
▼
[1. Matched Filter Correlation] (R_xy with analytic chirp replica)
│
▼
[2. Synchronous Frame Stacking] (Averages N repeated frames; noise cancels, echo reinforces)
│
▼
[3. Direct-Path Alignment] (Direct speaker-to-mic arrival calibrated as t = 0)
│
▼
[4. Clutter & Ringing Subtraction] (Subtracts open-space baseline: Excess = Target - Baseline)
│
▼
[5. Statistical Peak Echo Detection] (Picks argmax in search window, computes SNR quality sigma)
```

### 1. Transmit Probing Chirp Design (`design_chirp`)
- **Bandwidth**: 2,000 Hz to 8,000 Hz linear chirp over a 10 ms duration ($T = 0.010\text{ s}$).
- **Hann Windowing**: The chirp is smoothed using a Hann window. A rectangular pulse would demand instantaneous speaker cone acceleration, producing unwanted acoustic transients and distortion.
- **Why 2–8 kHz?** Running `band_check.py` reveals that laptop speakers suffer high-pass attenuation below 2 kHz, while built-in microphone pre-amps roll off sharply above 8 kHz (dropping past -12 dB). Operating strictly between 2 kHz and 8 kHz maximizes radiated acoustic energy and receiver sensitivity.

### 2. Synchronous Frame Stacking (`fold_frames`)
- A single reflected echo off a distant wall is orders of magnitude weaker than direct speaker coupling and is often submerged beneath ambient room noise.
- To recover it, the system transmits $N$ chirps (typically 8 to 16) spaced at uniform intervals of $260\text{ ms}$ ($10\text{ ms}$ chirp + $250\text{ ms}$ quiet listening period).
- Slicing the full correlation buffer into $N$ equal frames and averaging them together yields:
  - Echoes sum **coherently**: $A_{\text{echo}} \propto N$
  - Uncorrelated room noise sums **incoherently**: $A_{\text{noise}} \propto \sqrt{N}$
  - Net SNR improvement: $\approx 10 \log_{10}(N)\text{ dB}$ (a **+12 dB** gain for 16 repeats).

### 3. Direct-Path Time Origin Alignment (`align_to_direct_path`)
- In physical hardware, there is no shared electrical clock wire between the OS audio output buffer and input buffer; operating system audio latency varies between runs.
- However, sound travels across the short laptop chassis distance between speaker and microphone almost instantaneously ($< 0.5\text{ ms}$). This direct-arrival crosstalk is by far the loudest event in the correlation buffer.
- The pipeline locates this global maximum, rolls the buffer so that the direct arrival sits precisely at index 0 ($t = 0$), and normalizes its amplitude to $1.0$. All obstacle reflections are subsequently measured relative to this direct path.

### 4. Baseline Clutter & Ringing Cancellation (`remove_clutter`)
- Laptop speaker diaphragms and plastic chassis vibrate and ring for approximately 1.0 to 1.5 meters of apparent travel time. This stationary ringing swamps nearby reflections.
- Because chassis ringing and desk reflections remain identical across consecutive runs, an **open-space baseline** (captured facing empty space) records this stationary response.
- Subtracting the baseline from the target measurement leaves only the **Excess Return**:
  $$\text{Excess}(\tau) = \text{Envelope}_{\text{target}}(\tau) - \text{Envelope}_{\text{baseline}}(\tau)$$

### 5. Statistical Detection Metric (`detect_echo`)
- The detector searches inside a user-defined range window (e.g., 1.0 m to 5.0 m).
- The detected target is the maximum excess peak:
  $$d = \frac{c \cdot \tau_{\text{peak}}}{2}$$
- **Quality Metric ($\sigma$)**: Peak height divided by the standard deviation of all other points within the search window:
  $$\text{Quality} = \frac{\text{Excess}[\text{peak}]}{\text{std}(\text{Excess}[\text{window}])}$$
  - $\sigma \ge 6.0$: **Confirmed real target reflection** (extremely high statistical confidence, $>99.9\%$).
  - $\sigma < 6.0$: Weak return or background noise fluctuation.

---

## 3. Preparation & Physical Setup Guidelines

For reliable physical measurements, configure your environment as follows:

| Parameter | Recommended Setting | Rationale |
|---|---|---|
| **System Volume** | **60% – 75%** | Sufficient acoustic power without driving speaker amplifier into nonlinear clipping. |
| **Volume Stability** | **Strictly unchanged** | **Crucial**: Changing volume between baseline and measurement prevents clutter cancellation. |
| **Headphones** | **Unplugged** | Audio must emanate from speakers and be received by the microphone. |
| **Reflecting Target** | **Flat, hard surface** | Concrete wall, closed wooden door, dry-erase board, or large hardcover binder. |
| **Target Orientation** | **Perpendicular ($\approx 90^\circ$)** | Specular acoustic reflection reflects directly back into the microphone. |
| **Range Distance** | **1.0 m to 4.5 m** | Avoids near-field chassis ringing ($< 1.0\text{ m}$) and room attenuation ($> 5.0\text{ m}$). |
| **Room Clearance** | **$\ge 3.5\text{ m}$ for baseline** | Ensures no premature echoes contaminate the reference baseline. |

---

## 4. Method 1: Using the Web Application (Interactive UI)

The web dashboard provides an interactive hardware transceiver deck under **Studio 4**.

### Step-by-Step Instructions:

1. **Launch the Web Application**:
   ```powershell
   venv\Scripts\python -m streamlit run app/streamlit_app.py
   ```
   Open `http://localhost:8503` in your browser.

2. **Select Studio 4**:
   In the left sidebar, click **`4. Real Acoustic Hardware`**.

3. **Choose Hardware Operating Mode**:
   At the top of the Control Deck, select:
   - **`Archived Physical Benchmark (Concrete Wall @ 1.797 m, 6.75 σ)`**:
     Instantly displays the laboratory benchmark capture with full interactive control over sound speed and range window sliders.
   - **`Live Physical Hardware Test (Speaker & Microphone Transceiver)`**:
     Enables live audio recording and measurement directly through your hardware.

4. **Calibrate the Baseline (Step 1)**:
   - Aim your laptop away from walls into the open room (or open corridor).
   - Click **`Record Room Baseline`**.
   - Your laptop will play an audible sequence of chirps (~2.2 to 4.4 seconds). Once complete, a green **`READY`** indicator will appear.
   - *(Optional)* If your room is cramped, click **`Use Archived Baseline`** to load the pre-recorded reference.

5. **Measure Target Distance (Step 2)**:
   - Point your laptop speaker and microphone directly facing the wall or door you wish to measure.
   - Do **not** adjust system volume.
   - Click **`Transmit Chirps & Measure Echo`**.

6. **Inspect Live Results & Telemetry**:
   - **KPI Deck**: Displays detected distance (e.g. `1.842 m`), confidence ($\sigma$), and status (`Confirmed Detection`).
   - **Live Audio Player**: Play back the actual audio recorded by your microphone.
   - **Plot 1 (Direct-Path Aligned Stacked Trace)**: Visualizes the direct arrival alignment at $t = 0$ overlaid against the clutter baseline.
   - **Plot 2 (Clutter-Cancelled Excess Return)**: Shows the isolated obstacle echo with a prominent peak marker.

7. **Hardware Passband Sweep**:
   - Scroll to the expander at the bottom titled **`Hardware Frequency Response Analysis & Terminal CLI Guide`**.
   - Click **`Run Live Audio Passband Sweep (2-22 kHz)`** to measure your laptop's actual speaker/microphone frequency response curve.

---

## 5. Method 2: Command-Line (CLI) Terminal Execution

If you prefer testing directly from PowerShell or Command Prompt, use the standalone scripts in [`physical_sonar_check/`](file:///d:/L-2,%20T-2/CSE%20220/Project/sonar-distance-simulator/physical_sonar_check/).

### Step 1: Check Laptop Speaker/Microphone Passband
Before ranging, run the frequency sweep to see what frequencies your laptop can reproduce:
```powershell
venv\Scripts\python physical_sonar_check/band_check.py
```
*Output*: A text bar chart showing relative dB response from 2 kHz to 22 kHz. Frequencies within -12 dB of the peak represent your hardware's usable band.

### Step 2: Record Empty-Room Baseline
Point your laptop into open space with at least 3.5 m clearance:
```powershell
venv\Scripts\python physical_sonar_check/record_acoustic.py baseline --run live
```
*Output*:
```
transmitting 16 chirps, 2-8 kHz, 4.4 s total ...
input peak level : 0.428
baseline saved to live_baseline.npy. put the target in place and run again without 'baseline'.
```

### Step 3: Point at Wall and Measure Distance
Aim your laptop at the target obstacle without adjusting volume:
```powershell
venv\Scripts\python physical_sonar_check/record_acoustic.py --run live --min-range 1.0 --max-range 5.0
```
*Output*:
```
transmitting 16 chirps, 2-8 kHz, 4.4 s total ...
input peak level : 0.435
searched           : 1.00 - 5.00 m
distance           : 1.812 m
echo quality       : 7.2 sigma (real detection)
```

---

## 6. Architecture & File Reference

| File | Purpose | Key Symbols / Exports |
|---|---|---|
| [`src/acoustic.py`](file:///d:/L-2,%20T-2/CSE%20220/Project/sonar-distance-simulator/src/acoustic.py) | Pure DSP mathematical pipeline (no I/O, no plotting). | `design_chirp()`, `fold_frames()`, `align_to_direct_path()`, `remove_clutter()`, `detect_echo()`, `analyse_recording()` |
| [`physical_sonar_check/record_acoustic.py`](file:///d:/L-2,%20T-2/CSE%20220/Project/sonar-distance-simulator/physical_sonar_check/record_acoustic.py) | Standalone CLI audio capture tool. | `record()`, `main()` |
| [`physical_sonar_check/band_check.py`](file:///d:/L-2,%20T-2/CSE%20220/Project/sonar-distance-simulator/physical_sonar_check/band_check.py) | Hardware passband sweep evaluator. | Plays 2–22 kHz sweep, prints dB response table. |
| [`app/streamlit_app.py`](file:///d:/L-2,%20T-2/CSE%20220/Project/sonar-distance-simulator/app/streamlit_app.py) | Full-featured web dashboard Studio 4 interface. | `record_live_acoustic()`, `run_live_band_check()`, Studio 4 UI deck. |
| [`physical_sonar_check/rx.npy`](file:///d:/L-2,%20T-2/CSE%20220/Project/sonar-distance-simulator/physical_sonar_check/rx.npy) | Archived concrete wall raw microphone capture. | Float64 array ($1{,}697{,}408$ bytes, 48 kHz). |
| [`physical_sonar_check/baseline.npy`](file:///d:/L-2,%20T-2/CSE%20220/Project/sonar-distance-simulator/physical_sonar_check/baseline.npy) | Archived reference open-space clutter baseline. | Float64 array ($12{,}480$ envelope samples). |
| [`physical_sonar_check/excess.npy`](file:///d:/L-2,%20T-2/CSE%20220/Project/sonar-distance-simulator/physical_sonar_check/excess.npy) | Archived clutter-cancelled excess return. | Float64 array ($12{,}480$ excess samples). |

---

## 7. Troubleshooting & FAQs

### Q1: The detection quality is below 6.0 σ ("too weak to trust").
- **Check Obstacle Angle**: Ensure the laptop is pointed directly perpendicular ($90^\circ$) to the wall. At oblique angles, acoustic energy bounces away from the microphone like light off a mirror.
- **Surface Material**: Curtains, sofas, carpeted partitions, and soft clothing absorb 2–8 kHz ultrasound. Test against concrete, glass, a wooden door, or a whiteboard.
- **Volume Change**: Did the system volume change between baseline and measurement? Even a 5% volume drift ruins clutter subtraction.
- **Search Window**: Ensure `--min-range` is set to at least $1.0\text{ m}$ to avoid catching lingering chassis resonance.

### Q2: I get `PortAudio` / `sounddevice` errors.
- Ensure microphone access is permitted in Windows Settings:  
  *Settings $\rightarrow$ Privacy & Security $\rightarrow$ Microphone $\rightarrow$ Allow desktop apps to access your microphone*.
- Ensure your output device is set to your laptop's real speakers, not Bluetooth headphones or virtual streaming inputs.

### Q3: How do I adjust for cold or hot rooms?
Sound speed in dry air varies with temperature $T$ (in $^\circ\text{C}$):
$$c \approx 331.3 \cdot \sqrt{1 + \frac{T}{273.15}}\text{ m/s}$$
- At $20^\circ\text{C}$ ($68^\circ\text{F}$): $c = 343.4\text{ m/s}$
- At $25^\circ\text{C}$ ($77^\circ\text{F}$): $c = 346.3\text{ m/s}$
- At $30^\circ\text{C}$ ($86^\circ\text{F}$): $c = 349.2\text{ m/s}$

Adjust the **`Speed of Sound`** input widget in Studio 4 to match your room's temperature for millimeter-accurate ranging.
