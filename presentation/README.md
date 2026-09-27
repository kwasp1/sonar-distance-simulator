# CSE 220 Final Project Presentation — Team SevenEight

**Project:** Sonar Distance Simulator  
**Course:** CSE 220: Signals and Systems Laboratory (Level-2, Term-2)  
**Authors:** Aditya Tirtho Roy (2305178) & Farhan Ehsas Sami (2305177)  

---

## 1. Minimalist & Graphical Design Philosophy

This slide deck (`presentation.tex`) has been redesigned specifically for **high visual impact and minimal cognitive load**:
- **Visuals-First:** 60%–70% of every slide is dedicated to clear TikZ vector diagrams or experimental report figures.
- **Minimal Text:** No walls of text or dense bullet lists. Points are delivered as punchy labels, formula cards, and callout badges.
- **Intuitive Input-Output Model:** Slide 3 gives an immediate, clear visual mapping of what inputs are given (pulse shape, target distance, SNR) and what outputs are returned (distance in meters, correlation spikes, audio clips).
- **Projector Optimized:** High-contrast light background with deep navy, sky blue, and coral accents ensures maximum legibility in university lab rooms.

---

## 2. How to Compile the LaTeX Presentation

The slide deck is written in standard, modern LaTeX Beamer (`presentation.tex`). It uses standard packages (`tikz`, `graphicx`, `booktabs`, `amsmath`, `hyperref`) and is 100% compatible with Overleaf and local LaTeX distributions.

### Option A: Compile on Overleaf (Easiest & Recommended)
1. Zip the `presentation/` folder together with the `results/` folder (or upload them directly).
2. In Overleaf, create a **New Project** $\to$ **Upload Project**.
3. Set the compiler to **pdfLaTeX** (default) or **XeLaTeX**.
4. Set the main document to `presentation.tex` and click **Recompile**.
5. *Note:* If images from `results/` are uploaded into a `results/` folder, they will automatically embed via `\graphicspath{{../results/}{results/}{./}}`. Even if no images are uploaded, the deck uses custom safe-graphics wrappers so it will compile without errors!

### Option B: Compile Locally via Terminal
If you have MiKTeX or TeX Live installed:
```bash
cd presentation
pdflatex presentation.tex
pdflatex presentation.tex   # Run twice to resolve references and slide counters
```

Or using `latexmk`:
```bash
latexmk -pdf presentation.tex
```

---

## 3. Slide-by-Slide Outline & Speaking Roles

| Slide # | Title | Visual Focus / Diagram | Speaker |
|:---:|---|---|:---:|
| **1** | Title Slide | Minimalist navy title banner | Joint |
| **2** | The Problem | TikZ transceiver-target physics diagram + 3 problem cards | Aditya |
| **3** | Intuitive System Model | **3-Block Pipeline:** Inputs $\to$ DSP Engine $\to$ Outputs | Aditya |
| **4** | Matched Filtering | Sliding pulse template over noisy buffer diagram | Aditya |
| **5** | Multi-Target & Envelope | Side-by-side: Raw oscillation (15 false peaks) vs Hilbert envelope (2 targets) | Aditya |
| **6** | Range Resolution & Chirps | Chirp pulse compression diagram ($T \times B \gg 1$) | Aditya |
| **7** | Software Architecture | Clean modular architecture flowchart (`pulse` $\to$ `channel` $\to$ `rx` $\to$ `eval`) | Aditya |
| **8** | Pipeline in Action | Full `fig2_pipeline.png` with 3 visual callout cards | Farhan |
| **9** | Pulses & Bandwidth | `fig1_pulse_types.png` with waveform comparison notes | Farhan |
| **10** | Noise Resilience | `fig3_rmse_vs_snr.png` with quantization floor and noise cliff notes | Farhan |
| **11** | Resolution & Multipath | Side-by-side: `fig5_multipath.png` (40 cm vs 8 cm) and `fig4_resolution_vs_bandwidth.png` | Farhan |
| **12** | Real Acoustic Hardware | 4-step hardware pipeline + `fig6_acoustic.png` (wall at $1.797\text{ m}$, $6.7\sigma$) | Farhan |
| **13** | Interactive Streamlit App | 4 operating modes + audio engine callout ($4\times$ slowdown, click-free) | Farhan |
| **14** | Live Demonstration Script | 4 structured steps for the live 5-minute software demo | Farhan |
| **15** | Team Contributions | Balanced division cards for Aditya and Farhan | Joint |
| **16** | Conclusion | 3 large takeaway cards: Matched Filter, Chirp, Reality Verified | Joint |
| **17** | Anticipated Q&A Defense | 4 key examiner questions answered concisely | Joint |

---

## 4. 5-Minute Live Demo Walkthrough (Streamlit)

Launch the app before starting:
```bash
streamlit run app/streamlit_app.py
```

1. **Menu 1 (Single Target):**
   - Start with $d = 5.0\text{ m}$, $\text{SNR} = 10\text{ dB}$. Play the audio ping.
   - **The Punchline:** Drag SNR down to $0\text{ dB}$. Point to Stage 2: *"The echo is completely invisible to human eyes."* Then point to Stage 3: *"Yet the matched filter detects $4.999\text{ m}$ instantly."*
2. **Menu 2 (Two Targets — Range Resolution):**
   - Separate targets by $40\text{ cm}$: two clean peaks in the envelope.
   - Drag separation down to $8\text{ cm}$: the two peaks merge into one blob.
   - **The Resolution Fix:** Switch pulse from Gaussian to **Chirp** ($B = 4\text{ kHz}$). The merged blob immediately splits into two separate peaks!
3. **Menu 3 (Noise Sweep):**
   - Click **Run SNR Sweep** to show automated Monte Carlo execution and the $1.8\text{ mm}$ floor.
4. **Menu 4 (Real Sound Test):**
   - Show the archived wall capture at $1.797\text{ m}$ with $6.7\sigma$ confidence.

---

## 5. Quick Q&A Examiner Cheat Sheet

- **Why ~1.8 mm RMSE floor at high SNR?** Discrete sampling quantization $\Delta d = \frac{c}{4 f_s} = \frac{343}{4 \times 48000} \approx 1.79\text{ mm}$.
- **Why did bandpass pre-filter fail to improve white noise RMSE?** The matched filter is already mathematically optimal for AWGN (Cauchy-Schwarz).
- **Why raw argmax for single target, but Hilbert for multi-target?** Raw argmax gives the steepest, sharpest single peak. For multi-targets, the carrier creates 15+ false peaks, which the Hilbert envelope strips away.
