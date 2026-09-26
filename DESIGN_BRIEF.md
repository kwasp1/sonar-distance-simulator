# Design brief — Sonar Distance Simulator

Context for a redesign of the front end. Written for a designer/design model
that has not seen this codebase.

---

## 1. What the product is

A teaching and demonstration tool for **pulse-echo ranging** — the principle
behind sonar, radar and LiDAR. It sends a known sound pulse, listens for the
echo, and works out how far away an object is from the round-trip time.

It is a university signals-and-systems project (CSE220), shown live to an
instructor and used to produce figures for a written report. So the tone is
**professional engineering instrument**, not consumer app and not gaming/"cyber".
Think oscilloscope software, lab bench equipment, or a flight-test dashboard.

Two audiences at once:
- an **instructor** who knows the theory and is checking the work is correct,
- **classmates** who do not, watching a five-minute live demo.

That tension drives the whole design: correct technical labelling, but the
headline reading on any screen must be obvious without explanation.

## 2. Hard technical constraints — these cannot be negotiated

- **The app is Streamlit** (Python, v1.63), not React, not Next.js, not a static
  site. Any design must be buildable from Streamlit primitives: `st.columns`,
  `st.container`, `st.expander`, `st.tabs`, `st.metric`, `st.plotly_chart`,
  `st.audio`, plus one injected `<style>` block.
- **A `<div>` opened in one `st.markdown` cannot wrap later widgets.** Streamlit
  isolates each block, so the div auto-closes and the stray `</div>` renders as
  an empty styled box. The current app had 18 of these. Any card/panel styling
  must either be fully self-contained inside a single markdown block, or be done
  by styling Streamlit's own containers via CSS.
- **Charts are Plotly** (`plotly.graph_objects`), rendered with `st.plotly_chart`.
  There are 11 of them. They are interactive (hover tooltips) and must stay that way.
- **Dark mode only.** There is no light variant, by decision. The theme is pinned
  in `.streamlit/config.toml`.
- **All maths lives in `src/` and must not move into the UI.** The UI may only
  call existing functions and render what they return. It currently imports 17
  of them. Their signatures must stay stable — a second developer owns other
  code that calls the same functions.

## 3. Current palette and type

| Role | Value |
|---|---|
| Background | `#070b14` (with a radial gradient up to `#0f172a`) |
| Panel surface | `#0f172a` / `rgba(15,23,42,0.7)` |
| Primary accent | `#38bdf8` (cyan) |
| Body text | `#e2e8f0` · secondary `#cbd5e1` · muted `#94a3b8` |
| Good / warn / bad | `#34d399` · `#fbbf24` · `#f87171` |
| Extra accent | `#818cf8` (indigo) |
| Fonts | Inter (UI), JetBrains Mono (numbers), Outfit (headings) |

The palette itself is fine and can be kept. What needs work is the *styling
language*: the current build uses neon glow `text-shadow` on numbers, gradient
text on headings, and heavy `backdrop-filter` blur. That reads as "cyber", not
professional. **Keep the colours, drop the glow.**

## 4. The four screens, and what each actually shows

Navigation is four sections plus a home screen.

### Menu 1 — "Send a Pulse, Find the Echo"
Single target, the core demonstration.

- **Stage 1** — the transmitted pulse. Toggles between waveform (amplitude vs
  time) and spectrum (power vs frequency, showing the −3 dB bandwidth).
- **Stage 2** — what the microphone receives: the echo buried in noise. At low
  SNR the echo is invisible by eye, which is the point.
- **Stage 3** — the matched-filter output. A sharp peak marks the target. This
  is the "aha" chart.
- **Results (numbers)** — True Distance, Estimated Distance, Absolute Error,
  Round-Trip Delay.
- **Results (audio)** — two players: the outgoing ping, and the received echo.
  Slowed 4× so a 2 ms pulse is audible; they loop.
- **Settings** — target distance, SNR, pulse type (gaussian/chirp/rect),
  duration, carrier frequency, chirp sweep width, plus an advanced group
  (speed of sound, sampling rate, attenuation, RNG seed, band-pass filter).

### Menu 2 — "Two Targets — Can We Separate Them?"
Range resolution.

- **Stage 1** — the composite received buffer with two overlapping echoes.
- **Stage 2** — matched filter output plus its Hilbert envelope, showing one
  peak per target *if* they are far enough apart.
- **Dynamic bandwidth-vs-resolution sweep** — a separate study chart: measured
  closest-resolvable separation against pulse bandwidth, compared with the
  theoretical curve c/(2B). Runs on a button press and takes a few seconds.
- **Results (numbers)** — theoretical resolution limit, actual separation,
  objects detected vs placed, and a per-target table (truth, detection, error).
- **Results (audio)** — the two-echo scene.
- **Settings** — target 1 distance and reflectivity, target 2 separation and
  reflectivity, optional target 3, pulse waveform and duration, detector
  sensitivity, minimum gap between detections, SNR.

### Menu 3 — "How Much Noise Before It Fails?"
A Monte Carlo sweep. Structurally simpler: configure, then run.

- **Step 1 — settings**: evaluation distance, trials per step, SNR range and
  step, pulse type, duration, carrier, and a "compare with band-pass" toggle.
- **Step 2 — the graph**: distance error versus SNR on a log axis, with and
  without the band-pass filter, plus the sample-quantisation floor. The shape
  is the finding: flat and near-perfect above 0 dB, then a sharp cliff.
- Runs on a button press; takes a few seconds and shows a progress bar.

### Menu 4 — "Real Sound Test (Speaker + Mic)"
Same layout as Menu 1. Uses the laptop's actual speaker and microphone.

- **Stages** — the aligned correlation trace against an empty-room baseline;
  the excess signal after clutter removal, with the detection marked; and
  (optionally) the measured speaker/mic frequency response.
- **Results (numbers)** — measured distance and a confidence figure in sigma
  (above 6σ counts as a real detection).
- **Results (audio)** — the transmitted chirp and the actual recording.
- **Settings** — choose a saved recording or record live; speed of sound,
  burst repeats, detection window.
- Two real results already verified: a wall at 1.797 m (6.7σ) and a hand held
  above the keyboard at 0.272 m (6.2σ).

## 5. Requested layout (from hand-drawn wireframes)

A persistent **top bar**: `Logo │ Menu 1 │ Menu 2 │ Menu 3 │ Menu 4`.
Horizontal, always visible, current section clearly marked. This replaces the
current vertical radio list.

**Menu 1 and Menu 4** share one layout:
```
┌───────────────────────────────────────────────────────────┐
│ Logo    Menu1 │ Menu2 │ Menu3 │ Menu4                     │
├───────────────────────────────────────────────────────────┤
│  ┌─────────────────────────┐   ┌──────────────┐           │
│  │                         │   │   Stage 2    │           │
│  │        Stage 1          │   └──────────────┘           │
│  │        (large)          │   ┌──────────────┐           │
│  │                         │   │   Stage 3    │           │
│  └─────────────────────────┘   └──────────────┘           │
│ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - │
│  ┌──────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ Settings │ │Results Number│ │ Results Audio│           │
│  └──────────┘ └──────────────┘ └──────────────┘           │
└───────────────────────────────────────────────────────────┘
```

**Menu 2** is the same shell, with the third panel being the bandwidth sweep:
Stage 1 (large), Stage 2, and "Dynamic bandwidth vs resolution sweep", then the
same Settings / Results Number / Results Audio row.

**Menu 3** is the reduced form: a Settings panel marked **Step 1**, then the
graph marked **Step 2**, taking the full width.

**Home screen** — a professional landing view with the project identity and a
single primary **Get Started** call to action. It should state in one line what
the tool does, and give a way into the four sections. No marketing gloss.

## 6. Problems to fix

1. **Empty styled panels.** Caused by the split-`<div>` pattern described in §2.
   18 were removed; the redesign must not reintroduce the pattern.
2. **Visual tone.** Neon glow, gradient text and heavy blur read as a game UI.
   The same palette can look like precision instrumentation instead.
3. **Density.** 28 sliders, 10 number inputs, 7 selectboxes across four screens,
   presented mostly flat. Needs a clear primary/advanced split so a first-time
   viewer sees five controls, not forty.
4. **Hierarchy inside a screen.** Every chart currently carries equal visual
   weight. In Menu 1 the matched-filter result is the payoff and should look
   like it; in Menu 3 there is one chart and it should dominate.
5. **Result readouts.** Four numbers in a row of boxes; the single number that
   matters (estimated distance, or its error) should be unmistakable.
6. **Long-running actions.** Two screens run multi-second computations behind a
   button. They need honest progress, and a defined empty state before the
   first run.
7. **Audio players.** Five of them, currently bare. They need labels that say
   what you are about to hear and why it is slowed down.
8. **Narrow widths.** Behaviour below ~1200 px is undefined. A three-panel row
   must degrade sensibly.

## 7. What "professional" means here

- Restraint over decoration. No glow, no gradient text, no animated accents.
- One accent colour doing real work (state, selection, the active trace), not
  applied decoratively everywhere.
- Numbers in a monospace face, aligned, with units always attached.
- Generous spacing and a consistent vertical rhythm; panels distinguished by
  surface and spacing rather than by borders everywhere.
- Charts are the interface. Chrome should recede.
- Every control labelled in plain language with the technical term retained in
  parentheses where it exists — the instructor needs the real term, the
  classmate needs the plain one.
