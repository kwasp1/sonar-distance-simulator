# Sonar Distance Simulator — Where We Are

**Team SevenEight · CSE220 Signals and Systems**

This document explains what has been built so far and why. It assumes no signal
processing background — read it top to bottom and you should be able to pick up
the project from here.

---

## 1. What this project actually does

Imagine standing in a canyon. You shout, and a moment later your shout comes
back. Because you know how fast sound travels, that delay alone tells you how
far away the canyon wall is.

That is the entire idea behind sonar, radar, and LiDAR. Send out a signal, wait
for the echo, and use the round-trip time to work out distance:

```
distance = speed_of_sound × round_trip_time / 2
```

The division by 2 is because the sound went there *and* came back.

This project simulates that whole process in Python. Nothing physical is
involved — no microphone, no sensor, no hardware. Every signal is just a list of
numbers in a NumPy array.

### What we are actually being graded on

The ranging trick itself is 80-year-old, well-understood mathematics. The real
work of this project is **measuring how well it holds up under stress**. Two
questions, which become the two main graphs in the report:

1. **How much noise can it tolerate** before the distance guess goes wrong?
2. **How close can two objects be** before we can no longer tell there are two
   of them instead of one?

Everything in `src/` exists to answer one of those two questions.

---

## 2. Key vocabulary

Four terms are used constantly. They are simpler than they sound.

| Term | What it means |
|---|---|
| **Pulse** | The short "beep" we send out. Just a short list of numbers. |
| **Buffer** | The long recording we listen to afterwards. Mostly silence, with the echo somewhere inside it. |
| **SNR** | Signal-to-Noise Ratio. How loud the echo is compared to the background hiss, in decibels (dB). Higher = easier. |
| **Bandwidth** | How wide a range of frequencies the beep occupies, in Hz. Wider = "sharper" beep = better at separating close objects. |

One thing that trips people up: **the pulse is short and the buffer is long.**
The pulse might be 480 samples; the buffer 2400. We place the small thing inside
the big mostly-empty thing. The *position* it sits at is what encodes distance.

---

## 3. The pipeline

Five stages. Each is a separate file in `src/`.

```
  generate_pulse()      Make the beep                     [pulse.py]    DONE
         ↓
  simulate_channel()    Fake the echo bouncing back       [channel.py]  DONE
         ↓
  add_noise()           Corrupt it, like a real mic       [channel.py]  DONE
         ↓
  matched_filter()      Find the echo in the mess         [receiver.py] DONE
         ↓
  estimate_distance()   Turn its position into metres     [receiver.py] DONE
```

Plus three things built on top, all done:

- `evaluate.py` — runs the chain hundreds of times to build the report graphs
  (noise sweep and bandwidth sweep).
- `audio.py` — turns any of these arrays into sound you can actually listen to.
- `acoustic.py` — the same maths applied to a *real* recording from the laptop
  speaker and microphone, instead of a simulated one.

### The architecture rule

**Functions in `src/` return data. They never plot or print.**

The same functions get called from notebooks (which plot), tests (which assert),
and eventually a Streamlit app (which draws sliders). If plotting leaked into
`src/`, adding the web app would mean rewriting the core logic instead of just
writing a thin display layer.

Practical upshot: because `simulate_channel()` takes distance and SNR as plain
numbers, adding an interactive slider later is trivial — the slider just calls
the same function with a different number.

---

## 4. What is built: `pulse.py`

### `generate_pulse()` — makes the beep

Produces one of three beep shapes as a NumPy array:

- **rect** — a plain tone that switches on and off abruptly.
- **gaussian** — a tone that fades smoothly in and out. The best behaved of the
  three.
- **chirp** — a tone that *slides* in frequency across its duration, low to
  high. Useful because it can be long (lots of energy) while still being sharp.

**Why three shapes?** Not because we are comparing which is "best". They are how
we obtain beeps of different **bandwidth**, which is the input variable for
question 2 (the resolution study).

Design decisions made, and why:

| Decision | Reason |
|---|---|
| Time axis is **centred** on zero | Otherwise the gaussian's smooth fade gets chopped in half, reintroducing the hard edges we chose gaussian to avoid. |
| **Nyquist check** on all inputs | If the sampling rate is too low for the frequency, the signal is silently corrupted (aliasing) with no error. We reject it explicitly. |
| All pulses **normalised to unit energy** | Without this, a gaussian carries far less energy than a rect for the same settings, so "the same SNR" would secretly mean an easier problem for one of them. That would silently bias the results. |

### `pulse_bandwidth_hz()` — measures how sharp a beep is

Returns one number: the beep's bandwidth in Hz.

**How it works, in plain terms.** Every signal is made of a mix of frequencies.
Plotting how much energy sits at each frequency gives a hill shape. But the hill
has no hard edges — it just tapers off. So "how wide is it?" needs a rule.

The rule we use: draw a horizontal line at **half the height of the peak**, find
where the hill crosses it on each side, and measure the gap between those two
crossing points.

That's it. (This is called the "-3 dB bandwidth". "-3 dB" is just decibel
notation for "half" — the two terms mean the same thing.)

**Important: this function is not part of the detection pipeline.** It does not
help find the echo or compute the distance. It exists only to supply the x-axis
for the resolution graph in Week 3.

#### One thing we considered and rejected

An alternative definition (RMS bandwidth) was implemented first. It was
discarded because its answer **changed when the sampling rate changed**, even
though the beep itself had not changed. That would have poisoned the Week 3
sweep, since sampling rate is supposed to be irrelevant there. The half-power
definition has no such problem — verified stable to 0.00% across a 2× change in
sampling rate.

---

## 5. What is built: `channel.py`

This file fakes the physics.

### `simulate_channel()` — places the echo

Three steps:

1. **Make a long buffer of silence.** Its length is your listening window, which
   sets your maximum range. Listening for 50 ms at 343 m/s means you can see
   objects out to about 8.6 m — anything further and its echo arrives after you
   stopped listening.

2. **Work out where the echo goes.** Convert distance to a position:
   ```
   delay_seconds = 2 × distance / speed_of_sound
   delay_samples = round(delay_seconds × sampling_rate)
   ```

3. **Paste a weakened copy of the beep at that position.**

**Multiple objects** are handled by summing: two objects means two echoes added
together into the same buffer. This is why the code uses `+=` rather than `=` —
so a second call adds to the buffer rather than overwriting it.

#### Two bugs found and fixed here

Both were **silent failures** — the worst kind, because nothing looks wrong.

- **Negative distance wrapped around.** In Python, a negative index counts from
  the *end* of the array. A distance of −5 m produced a perfectly healthy-looking
  echo at the wrong place, with no error at all. The system would confidently
  report ~3.6 m. Fixed by rejecting non-positive distances up front.

- **The "object too far" check ran after the write, not before.** It was
  unreachable — the write crashed first with an unhelpful NumPy message about
  array shapes. Fixed by moving the check above the write.

  *General lesson: validate before you act. A check placed after the operation
  it guards cannot guard anything.*

### `add_noise()` — corrupts the signal

Adds random Gaussian noise across the whole buffer at a requested SNR.

**Why add noise deliberately?** Real microphones pick up electrical hiss and
room sound. Testing only on a perfect signal tells you nothing about whether the
method works in reality. So we add noise on purpose to find out how much it can
take before it breaks — which is exactly the first report graph.

Note the noise goes across the **entire** buffer, not just where the echo is.
Noise is everywhere; that is what makes finding the echo a real problem.

#### One bug found and fixed here

The signal loudness was originally measured by averaging over the **whole
buffer** — which is mostly silence. So making the listening window longer added
more zeros, dragged the average down, and made the code use *less* noise.

Same echo, same requested SNR, but the actual difficulty changed by 3× just from
changing an unrelated setting. That would have made the SNR graph meaningless,
because "0 dB" would not have referred to a fixed thing.

Fixed by measuring loudness over the echo region only. Verified: the computed
noise level is now identical to twelve decimal places across a 8× change in
window length.

#### Reproducibility

`add_noise()` takes an optional `seed`. The same seed gives the identical noise
every time. This matters because a test that changes on each run is not a test,
and because comparing pulse types is cleaner against identical noise.

---

## 6. Verification

**29 pytest cases, all green**, plus 17 checks in `check_pulse.py`. Run them
with `python -m pytest -q` and `python check_pulse.py`.

The suite covers the Week 1 noiseless gate, multi-object detection, the
bandwidth-resolution trend, playback shape and loop-safety, and — the one
worth knowing about — **a real microphone recording**. `tests/test_acoustic.py`
re-derives an archived capture of a wall bit-for-bit, so the acoustic code
stays honest with no hardware attached.

`check_pulse.py` runs 17 automated checks against `pulse.py`. It verifies:

- Pulse lengths and unit-energy normalisation
- Gaussian symmetry and centring
- Measured bandwidth against theory (within ~2% for rect and gaussian)
- **Bandwidth stability across sampling rate** — the check that caught the RMS
  problem
- That every invalid input raises a clear error

`channel.py` was verified the same way (11 checks) covering echo placement,
attenuation, superposition of two echoes, SNR accuracy across 20 to −5 dB, seed
reproducibility, and all the guards.

### A useful habit this surfaced

One check initially failed by 1.18%. It turned out the *code* was exactly right
— the discrepancy was random sampling scatter, the same reason 100 dice rolls
average 3.42 rather than exactly 3.5.

When a numeric test fails, ask whether the error is **systematic** (a real bug,
like the 3× above) or just **noisy** (expected scatter). Here the fix was a
better test, not different code. This is also why the SNR sweep must run many
trials per noise level and average them — one trial is one dice roll.

---

## 7. Gotchas to remember

Three things discovered along the way that will matter later:

**Envelope for counting echoes, raw correlation for locating one.** These are
opposite answers to two different questions, and the earlier draft of this
document got it backwards.

*Finding one echo* — `estimate_distance()` uses raw `np.argmax`, deliberately.
The envelope was tried and was **worse at every SNR tested**, because smoothing
widens the peak and a wider peak is easier for noise to nudge. Do not "fix"
this to use the envelope.

*Counting several echoes* — `estimate_multiple_distances()` does use the
Hilbert envelope, because the raw correlation oscillates at the carrier and
counting its crests found 18 "peaks" in a genuine 2-object scene. The envelope
plus `find_peaks` found exactly 2.

**`np.correlate(..., mode="full")` shifts the index.** With the echo placed
*starting* at `delay_samples`, the correlation peaks at
`delay_samples + len(pulse) − 1`. The receiver must subtract `len(pulse) − 1`.

**A detector's own settings can fake a physics result.** The receiver's
`min_separation_samples` defaults to `len(pulse)//2` — a floor set by pulse
*length*, which has nothing to do with bandwidth. Left at that default, the
bandwidth sweep pinned every chirp at exactly 17.0 cm across a 3.5x bandwidth
range: a flat line that looks like a real finding and says the opposite of the
truth. Any measurement has to check that it is measuring the physics and not
its own tuning.

**Chirp bandwidth measures narrower than its nominal sweep.** A chirp's spectral
edges roll off rather than cutting sharply, so a 2000 Hz sweep measures about
1665 Hz. The ratio improves as the pulse gets longer (0.83 at low
time-bandwidth, 0.97 at high). This is correct physics, not a bug — and it does
not matter because the sweep plots against *measured* bandwidth, not nominal.

---

## 8. Distance measurement has a floor

Even with zero noise, distance cannot be measured perfectly. Delay is measured
in whole samples, and `2 × distance × fs / speed` is rarely a whole number, so
rounding introduces up to about `speed / (4 × fs)` of error — roughly **1.8 mm**
at 343 m/s and 48 kHz.

This matters for the Week 1 test: "recovers the exact distance" cannot mean
bit-exact for an arbitrary distance. Either pick a distance that lands on a
whole number of samples (recommended, so a failure means a real bug rather than
mere imprecision), or allow tolerance and document why.

The same floor shows up in the noise sweep: RMSE bottoms out at about 1.8 mm
rather than zero at high SNR. That is sample quantisation, not a bug.

---

## 9. What is next

**All nine committed features are built.** Everything below the line is
report production, not code.

### What got finished after this document was first written

- **`receiver.py`** — `matched_filter()` slides the original beep along the
  noisy recording, scoring the match at every position; the highest score is
  the echo. This is provably the best way to find a known shape hidden in
  random noise, which is why radar, sonar and GPS all use it.
  `estimate_distance()` turns that position into metres.
- **`filters.py`** — band-pass pre-filtering. The honest result is that it
  barely helps, because the matched filter is already optimal against white
  noise. A null result with a theoretical explanation, not a failure.
- **`evaluate.py`** — both sweeps. `run_snr_sweep()` answers "how much noise
  before it breaks"; `run_bandwidth_resolution_sweep()` answers "how close can
  two objects get". The second one needed its own detector settings rather
  than the receiver's defaults — see PROJECT_OVERVIEW for why.
- **`audio.py`** — turns any pulse or received buffer into something you can
  actually listen to. A 2 ms beep is far too short and quiet to hear, so it is
  slowed down, repeated with gaps into a sonar-like ping, and faded at the
  edges so it loops without clicking.
- **`acoustic.py`** — the same ranging maths applied to a real recording from
  the laptop speaker and microphone. Verified twice against reality: a wall at
  1.797 m and a hand held 30 cm above the keyboard, measured at 0.272 m. Both
  above the 6-sigma bar for a real detection.
- **Tests** — 29 pytest cases, including one that re-derives an archived real
  recording bit-for-bit, so the acoustic code stays honest without a mic
  attached.

### What is left

1. **Plots.** `notebooks/` is still empty. Five sweeps and two real acoustic
   measurements are computed but nothing has been drawn yet. This is the
   biggest remaining gap.
2. **The report.**
3. Optional polish: steadier numbers in the bandwidth sweep (per-config random
   seeds), and folding the acoustic result into the written evaluation.

---

## 10. Standard settings used so far

| Setting | Value |
|---|---|
| Sampling rate | 48 000 Hz |
| Speed of sound | 343 m/s (air) |
| Pulse duration | 2–10 ms |
| Carrier frequency | 4 000–5 000 Hz |
| Listening window | 50 ms (≈ 8.6 m max range) |

Units are SI throughout — seconds, Hz, metres, m/s — with the unit kept in the
variable name (`duration_s`, `freq_hz`, `distance_m`). Never mix ms with s or cm
with m silently.
