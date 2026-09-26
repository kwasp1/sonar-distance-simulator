# Prompt to paste into Claude

Paste everything below the line. Attach `DESIGN_BRIEF.md` alongside it — the
prompt assumes the brief is available.

---

Redesign the front end of a **Streamlit** application called the Sonar Distance
Simulator. The full context — what it does, the data on every screen, the
palette, and the constraints — is in the attached `DESIGN_BRIEF.md`. Read it
first.

**Deliver a high-fidelity, interactive HTML/CSS mockup** (single file, inline
CSS, no build step) covering five screens. Use realistic placeholder charts and
plausible numbers so the layout can be judged properly — but note this is a
*design reference*, and the real implementation is Streamlit, so every layout
you propose must be reproducible with `st.columns`, `st.container`,
`st.expander`, `st.tabs`, `st.metric`, `st.plotly_chart` and one injected
`<style>` block. Do not propose anything that needs a JS framework or arbitrary
absolute positioning.

## Screens

**Persistent top bar on all four working screens:**
`Logo │ Menu 1 │ Menu 2 │ Menu 3 │ Menu 4` — horizontal, always visible, with
the active section clearly marked. Use the real section names from the brief,
not "Menu 1".

**0 — Home.** A professional landing screen: project identity, one line stating
what the tool does, and a single primary **Get Started** button. Restrained and
credible — this is lab instrumentation, not a product launch page. No marketing
copy, no feature grid, no testimonials.

**1 — Send a Pulse, Find the Echo.**
```
┌──────────────────────┐  ┌────────────┐
│                      │  │  Stage 2   │
│      Stage 1         │  └────────────┘
│      (large)         │  ┌────────────┐
│                      │  │  Stage 3   │
└──────────────────────┘  └────────────┘
- - - - - - - - - - - - - - - - - - - - -
┌──────────┐ ┌───────────────┐ ┌──────────────┐
│ Settings │ │ Results Number│ │ Results Audio│
└──────────┘ └───────────────┘ └──────────────┘
```
Stage 1 = transmitted pulse, Stage 2 = noisy received signal, Stage 3 = matched
filter output with the detected peak marked. Results Number = true distance,
estimated distance, absolute error, round-trip delay. Results Audio = two
looping players.

**2 — Two Targets — Can We Separate Them?** Same shell, but the third panel is a
*Dynamic bandwidth vs resolution sweep* chart that runs on a button press. The
Results Number panel also carries a small per-target table.

**3 — How Much Noise Before It Fails?** Reduced layout: a **Step 1 — Settings**
panel, then **Step 2 — The Graph** taking the full width. One chart, and it
should dominate the screen.

**4 — Real Sound Test.** Identical layout to screen 1.

## What matters most

1. **Professional restraint.** Keep the existing dark palette (`#070b14`
   background, `#38bdf8` accent) but remove the neon glow, gradient text and
   heavy blur in the current build. Aim for precision instrument, not game UI.
2. **Hierarchy.** On screen 1 the matched-filter result is the payoff — make it
   read that way. On screen 3 there is one chart and it should own the page.
3. **Tame the density.** There are 28 sliders and 17 other controls across the
   four screens. Show the five that matter and put the rest behind a clear
   "Advanced" disclosure. A first-time viewer should not meet forty controls.
4. **One number should dominate each Results panel** — the estimated distance
   and its error. Monospace, aligned, units always attached.
5. **States.** Two screens run multi-second computations behind a button. Design
   the empty state before the first run, the in-progress state, and the result
   state.
6. **Audio players need context** — a label saying what you are about to hear
   and why it is slowed down 4×.
7. **Degrade sensibly below ~1200 px.** Three-panel rows must reflow, not
   squash.

## Do not

- Do not invent new features, metrics or screens. Every panel must map to
  something in the brief.
- Do not use a light theme. Dark only.
- Do not wrap groups of controls in decorative containers that a Streamlit
  `st.markdown` `<div>` cannot actually produce — see the brief §2.
- Do not replace technical terms with vague ones. Keep the real term and add a
  plain-language gloss.

## Deliverables

1. The five screens as one interactive HTML mockup, navigable via the top bar.
2. A short spec of the design system used: spacing scale, type scale, surface
   levels, the accent's meaning, and chart styling rules.
3. For each screen, a one-line note on how it maps to Streamlit layout
   primitives, so it can be ported without guesswork.
