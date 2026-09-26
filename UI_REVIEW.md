# Streamlit UI — review findings

**App:** `app/streamlit_app.py` (1907 lines, commit `1649b65` "real audio test added")
**Reviewed:** 2026-09-26 · **All ten findings fixed and verified 2026-09-26.**
Verification drove the real widgets through `streamlit.testing.v1.AppTest`,
not just static reading. Re-run with the commands in section F.

Method: extracted all 53 interactive controls with their declared min/max, then
exercised the underlying `src/` call sequences across those ranges. Everything
below is reachable by moving sliders only — no invalid input typed anywhere.

**Health check first:** all 17 `src/` imports resolve, the file compiles, and the
audio + acoustic modules are wired into four tabs. The problems are edge-case
handling and audio playback behaviour, not broken plumbing.

---

## A. Crashes — unhandled, show a red traceback

The main pipeline at **L799–L819** (`generate_pulse` → `simulate_channel` →
`add_noise` → `bandpass_filter` → `estimate_distance`) has **no `try`/`except`**.
Any `ValueError` from `src/` lands on the user as a stack trace.

### A1 — Carrier above Nyquist (Studio 1)

`Carrier Freq` goes to **12000 Hz** (L755); `Sampling Rate fs` goes down to
**16000 Hz** (L771), whose Nyquist limit is 8000 Hz.

```
carrier 12000 Hz @ fs 16000 Hz
  ValueError: freq_hz=12000.0 must be in [0, 8000.0]
```

Repro: Studio 1 → Advanced → set fs to 16000, set Carrier to 12000.

Fix direction: make the `Carrier Freq` control's `max_value` depend on the
chosen `fs` (`fs/2` minus a margin) instead of a fixed 12000.

### A2 — Chirp sweep runs below 0 Hz (Studio 1)

`Chirp Sweep Width` goes to **6000 Hz** (L759) and `Carrier Freq` down to
**500 Hz**. The sweep spans `carrier ± width/2`, so any carrier below 3000 Hz
with a 6000 Hz sweep starts at a negative frequency.

```
chirp sweep 6000 @ carrier 500   -> chirp sweeps [-2500.0, 3500.0] Hz, outside [0, 24000.0]
chirp sweep 6000 @ carrier 2000  -> chirp sweeps [-1000.0, 5000.0] Hz, outside [0, 24000.0]
```

Repro: Studio 1 → Pulse Type = chirp → Carrier 2000 → Sweep Width 6000.

Fix direction: cap the sweep slider at `2 * min(carrier, nyquist - carrier)`.

### A3 — Band-pass auto-band goes negative (Studio 3) — fires at DEFAULT settings

`run_snr_sweep(bandpass=True)` derives its band from the measured pulse
bandwidth. Bandwidth scales as roughly `1/duration`, so a short pulse at a low
carrier drives the lower cutoff below zero.

```
gaussian 1.0 ms, carrier 1000 Hz, bandpass ON
  ValueError: bandpass filter low_hz=-640.625 <= 0; try a longer duration_s or higher freq_hz
```

**This is the worst of the three:** `Pulse Duration` bottoms out at 1.0 ms
(L1390), `Carrier Freq` at 1000 Hz (L1391), and **"Compare With Bandpass Filter"
is checked by default** (L1392). Sliding both to minimum and pressing Run is an
ordinary thing to do. The sweep block (L1401–L1500) has no `try`/`except`.

Fix direction: catch the error around the sweep and surface it with `st.error`,
or clamp the derived band before calling.

### A4 — A1 and A2 compound

`chirp sweep 6000 @ carrier 12000 @ fs 16000` fails on the Nyquist check before
the sweep check is even reached.

---

## B. Handled, but the message misleads

### B1 — Live acoustic detection window can collapse (L1609/L1611)

`Min Detection Window` spans 0.5–2.5 m and `Max Detection Window` spans
2.5–8.0 m, so **both can sit at exactly 2.5**, and `detect_echo` requires
`min < max`:

```
min=max=2.5 -> ValueError: need 0 <= min_range_m < max_range_m (got 2.5, 2.5)
```

It *is* caught (L1664–L1681), but reported as:

> Live measurement failed: … Ensure speaker and microphone are enabled.

which sends the user to check their microphone over a slider problem.

Fix direction: start the max slider above 2.5 (the archived-mode pair at
L1538/L1541 already does this correctly with 3.0), or special-case the message.

---

## C. Audio playback — does not match the documented guidance

`TEAMMATE_GUIDE.md` §6 and §7 cover these; all four were missed.

### C1 — No caching anywhere (`@st.cache` count: **0**)

Every rerun regenerates all audio from scratch. Since Streamlit reruns the whole
script on *any* widget interaction, **moving any slider restarts playback** and
recomputes several seconds of audio for nothing.

Fix direction: wrap each audio build in `@st.cache_data` keyed on the parameters
that affect it.

### C2 — No `loop=True` on any of the five `st.audio` calls

L864, L880, L1182, L1590, L1736 all call `st.audio(wav, format="audio/wav")`.
The original requirement was *"plays until stopped"*; as written each clip plays
once. `st.audio(..., loop=True)` is supported on Streamlit 1.63.

### C3 — `audible_echo` called without a shared `gain` (L878)

Each clip is normalised independently, so **changing the SNR slider does not
change how loud the noise sounds**. A −5 dB clip and a +20 dB clip come out at
the same level, which removes the point of the demonstration — the listener is
supposed to hear the hiss rise around a constant ping.

Fix direction: compute the gain once from the noisiest setting and pass it in.

### C4 — Multi-target audio repeats identical noise (L1180)

`make_audible(m_buffer, repeats=4)` is given an already-noisy buffer and repeats
that single buffer four times, so **the exact same hiss pattern cycles four
times** — instantly recognisable as artificial. This is the specific artifact
`audible_echo` exists to avoid (it re-noises each frame with a fresh seed).

---

## D. Silent behaviour worth knowing

### D1 — Studio 2 drops noise entirely at maximum SNR (L1127)

```python
if m_snr_db < 30.0:
    m_buffer = add_noise(m_buffer, m_snr_db, seed=42)
```

The `Channel SNR` slider's maximum **is** 30.0, so at the top of its range no
noise is added at all. The reading still says "30 dB". If 30 is meant to be
"noise off", the control should say so.

### D2 — Multi-target tab hard-codes fs, speed and carrier (L1108–L1112)

`m_fs = 48000.0`, `m_speed = 343.0`, `freq_hz=4000.0` are fixed, while Studio 1
exposes all three. Not a bug — and it is *why* Studio 2 has no crash paths — but
the two tabs behave inconsistently from the user's point of view.

---

## E. Summary

| ID | Area | Severity | State |
|---|---|---|---|
| A1 | Carrier above Nyquist | **High** — raw traceback | **Fixed** |
| A2 | Chirp sweep below 0 Hz | **High** — raw traceback | **Fixed** |
| A3 | Band-pass band negative, at defaults | **High** — raw traceback | **Fixed** |
| B1 | Window collapse blames the mic | Medium | **Fixed** |
| C1 | No caching, playback restarts | Medium | **Fixed** |
| C2 | No `loop=True` | Medium | **Fixed** |
| C3 | No shared gain, SNR inaudible | Medium | **Fixed** |
| C4 | Repeated identical noise | Low | **Fixed** |
| D1 | Noise silently off at SNR 30 | Low | **Fixed** |
| D2 | Tabs inconsistent in what they expose | Low | **Fixed** |

Coverage: **53 controls, 8 `try`/`except` blocks, 4 user-facing error messages.**
The three unguarded crashes are all in code paths with no error handling at all.

## F. How each fix was verified

`A1`, `A2`, `A4`, `A3`, `D1`, `D2`, `B1` were replayed through the real UI with
`streamlit.testing.v1.AppTest` — setting the same widgets a user would and
asserting no exception is raised:

```
A1  carrier 12000 Hz @ fs 16000 Hz      -> OK, warns "carrier lowered to 7600 Hz"
A2  chirp sweep 6000 Hz @ carrier 2000  -> OK, warns "sweep narrowed to 4000 Hz"
A4  both at once                        -> OK, warns about both
A3  gaussian 1.0 ms @ 1000 Hz, sweep ran-> OK (bandpass checkbox still defaults True)
D1  Studio 2 at SNR 30                  -> OK, noise now applied
D2  Studio 2 exposes fs / speed / carrier
B1  min window 0.5..2.5, max 3.0..8.0   -> cannot collide
--  defaults                            -> OK, no warning (nothing changed for normal use)
```

`C3` and `C4` were checked on the audio itself:

```
C3  one gain (0.2261) across every SNR; peak climbs 0.06 -> 0.90 as hiss grows,
    nothing reaches 1.0, so the ping stays put and only the noise changes
C4  the four repeats of a scene are no longer identical -> hiss stops cycling
```

`C1` and `C2` are structural: `@st.cache_data` now wraps all three audio
builders, and all five `st.audio` calls carry `loop=True`.

**Note:** `app/streamlit_app.py` is Farhan's file. These edits were made with
explicit clearance after confirming he had everything committed (his
`SamiBranch` was two commits behind `main` at the time).
