# The project, end to end

Everything the backend does, why it does it that way, and the numbers to know.
Frontend is not covered here.

---

## 1. The whole idea in one line

Send a sound, time the echo, divide by two:

```
distance = speed_of_sound × round_trip_time / 2
```

The `/2` is because the sound travels there *and* back. That is sonar, radar and
LiDAR — only the wave type changes.

**But the formula is not the project.** The formula is 80 years old. The project
is measuring **how well it survives stress**, which is two questions:

1. How much noise before the distance estimate breaks?
2. How close can two objects be before they look like one?

Those are the two graphs in the report. Everything in `src/` exists to answer one
of them.

---

## 2. The pipeline, in order

```
generate_pulse()   make the beep                    pulse.py
       ↓
simulate_channel() place the echo in a recording    channel.py
       ↓
add_noise()        bury it in hiss                  channel.py
       ↓
matched_filter()   find it again                    receiver.py
       ↓
estimate_distance()turn position into metres        receiver.py
```

Then three things built on top: `evaluate.py` (the two experiments),
`audio.py` (listen to any of it), `acoustic.py` (the same maths on a *real*
microphone recording).

---

## 3. The architecture rule

> **Everything in `src/` takes numbers and returns numbers. It never plots,
> prints, records or plays.**

Why: the same functions are called by the tests, the report figures, and the web
app. If plotting leaked into `src/`, adding the web app would have meant
rewriting the maths instead of writing a display layer.

Practical proof: the 33 tests never import the UI. The whole front end can be
rewritten and the physics is provably untouched.

---

## 4. Every file

### `src/pulse.py` — make the beep

| Function | Does |
|---|---|
| `generate_pulse()` | Builds a rect, gaussian or chirp pulse |
| `pulse_bandwidth_hz()` | Measures how "sharp" a pulse is, from its samples |
| `_normalize()` | Scales every pulse to the same total energy |

**Three pulse types.** *Rect* = abrupt on/off tone. *Gaussian* = same tone faded
in and out smoothly. *Chirp* = a tone sliding from low to high pitch.

**Why unit energy.** A faded gaussian carries less energy than a rect of the same
amplitude. Without normalising, the gaussian would look worse under noise purely
because it was quieter — not because it is worse. This makes comparisons fair.

**Bandwidth is measured, not calculated from the settings.** The function takes
the FFT, finds the peak, and measures the width where power is at least half the
peak (the **−3 dB width**). It has to work on any pulse, including a recorded one
whose settings are unknown.

> **Likely question: why −3 dB and not RMS bandwidth?**
> RMS was tried first and rejected. For a rect pulse it is mathematically
> infinite, and only came out finite because sampling truncates it — so the
> *measurement* changed when `fs` changed even though the pulse had not. The
> −3 dB width was stable to 0.00 % across 48 kHz vs 96 kHz; RMS drifted 14 %.

### `src/channel.py` — send it and corrupt it

| Function | Does |
|---|---|
| `simulate_channel()` | Puts one echo into a silent recording at the right spot |
| `add_noise()` | Adds hiss at a chosen SNR |
| `simulate_multi_object_channel()` | Several objects = several echoes added together |

**The key idea:** the buffer is mostly silence, and **where the pulse sits inside
it is the distance**. Delay in samples is `round(2d/c × fs)`.

**Multiple objects are just addition.** Real sound superposes, so three objects
is literally the sum of three one-object recordings. That is why the function is
six lines and still produces real resolution physics.

> **Likely question: how is SNR set?**
> Signal power is measured over the **echo samples only**, not the whole buffer.
> Averaging over the silence would make the realised SNR depend on how long you
> listened — doubling the window would quietly halve the noise. Fixed: noise
> power is now identical to 12 decimal places across 0.05 s–0.40 s windows.

### `src/receiver.py` — find the echo

| Function | Does |
|---|---|
| `matched_filter()` | Slides the known pulse along the recording, scoring the match |
| `estimate_distance()` | Highest score → position → distance |
| `estimate_multiple_distances()` | Finds several echoes |

**Why matched filtering works.** You already know exactly what you sent. Slide it
along the noisy recording and score the overlap at every position. The echo lines
up and scores high; random hiss does not line up and averages out. This is
provably the optimal detector for a known shape in white noise — which is why the
band-pass filter barely helps (below).

> **Likely question: why raw argmax for one echo but the Hilbert envelope for
> several?**
> They answer different questions. For *where is the one echo*, raw argmax wins:
> the envelope smooths the peak, and a wider peak is easier for noise to nudge —
> tested, and the envelope was worse at every SNR. For *how many echoes*, the raw
> correlation oscillates at the carrier, so counting its crests found 18 "peaks"
> in a genuine 2-object scene. The envelope plus `find_peaks` found exactly 2.

**Offset detail.** `np.correlate(..., mode="full")` shifts the index, so the peak
sits at `delay + len(pulse) − 1`. The receiver subtracts `len(pulse) − 1`.

### `src/filters.py` — optional cleanup

`bandpass_filter()` throws away frequencies outside the pulse's band.

Uses `filtfilt`, **not** `lfilter`. `lfilter` introduces group delay, which would
shift the signal in time and corrupt every distance estimate. This is the one
place a "simplification" would silently break the results.

> **The honest finding: it barely helps.** Against white noise the matched filter
> is already optimal, so a pre-filter has nothing left to remove. A band narrower
> than the pulse's own bandwidth makes things *worse* — it clips the pulse's own
> energy. This is a real result with a theoretical explanation, not a failure.

### `src/evaluate.py` — the two experiments

| Function | Does |
|---|---|
| `run_snr_sweep()` | Repeats the pipeline at many noise levels → RMSE per level |
| `run_bandwidth_resolution_sweep()` | Finds the closest resolvable separation per pulse |
| `both_targets_detected()` | One yes/no trial: were both targets found, in the right places? |
| `smallest_resolvable_separation_m()` | Shrinks separation until detection fails |

**Experiment 1 (noise).** Many trials per SNR, RMSE of the errors. Shape: flat and
near-perfect above 0 dB, then a **sharp cliff**. That cliff is the threshold
effect — below a critical SNR the peak occasionally locks onto a noise sidelobe
instead of the echo, producing huge outliers rather than gradual drift.

**Experiment 2 (resolution).** Two targets, shrink the gap until they merge.
Repeat for pulses of different bandwidth. Compare with theory `c/(2B)`.

> **Likely question: what was hard about experiment 2?**
> Three traps, all found by testing:
> 1. The receiver's default `min_separation_samples = len(pulse)//2` is a floor
>    set by pulse **length**, not bandwidth. Left at the default, all three chirps
>    pinned at exactly 17.0 cm across a 3.5× bandwidth range — a flat line that
>    looks like a finding and says the opposite of the truth. The sweep overrides it.
> 2. Counting peaks is not enough. Two echoes a few cm apart create interference
>    nulls that read as structure — it reported two "targets" 16 cm apart when the
>    real pair was 2 cm apart. So each detection must also land *near* a true target.
> 3. Relative echo phase is frozen by a fixed distance, so one alignment is not a
>    measurement. Each separation is tried over several trials with the pair
>    jittered, and accepted at ≥50 % success.
>
> **Why both gaussians and chirps are needed:** a gaussian's bandwidth is tied to
> its duration, so gaussians alone cannot separate "wider bandwidth helps" from
> "shorter pulse helps". Chirps hold duration fixed at 2 ms and vary only
> bandwidth. That is what pins the credit on bandwidth.

### `src/audio.py` — make it listenable

| Function | Does |
|---|---|
| `make_audible()` | Turns an array into a loopable, hearable clip |
| `audible_echo()` | The received version, with hiss |
| `to_wav_bytes()` | Packs it as a `.wav` |

A 2 ms pulse is a click, and a 29 ms echo delay is below the threshold where the
ear hears two events instead of one. So it is **slowed 4×** — not resampled, just
played at a lower declared rate, which costs nothing. Each copy gets a gap of
silence so a loop sounds like "ping … ping" instead of a buzz, and the edges are
faded so the loop seam does not click.

> **Two traps worth knowing.**
> - Comparing clips at different SNR: use **one shared gain, taken from the
>   loudest (noisiest) clip**. Per-clip normalising makes −20 dB and +20 dB
>   equally loud and erases the point; taking it from a clean clip makes the
>   noisy ones clip and distort.
> - Repeating one noisy buffer makes the *same* hiss pattern cycle audibly, which
>   sounds fake. Each repeat gets its own noise seed.

### `src/acoustic.py` — real speaker and microphone (Feature 8)

| Function | Does |
|---|---|
| `design_chirp()` | The sweep to actually transmit |
| `fold_frames()` | Averages many transmissions together |
| `align_to_direct_path()` | Sets t = 0 at the direct speaker→mic arrival |
| `remove_clutter()` | Subtracts an empty-room recording |
| `detect_echo()` | Strongest return in a range window, plus confidence |
| `analyse_recording()` | All of the above in one call |

**What a real recording needs that a simulation does not:**
- **Averaging** — one chirp off a wall is far weaker than a simulated echo. The
  chirp is sent 16 times; stacking grows the echo and averages the room noise away.
- **A time origin** — the mic hears the speaker *directly*, long before any echo.
  That arrival is t = 0.
- **Clutter removal** — the laptop chassis rings, the desk reflects. Record the
  room with nothing there, subtract it, and what survives is what changed.

Recording audio happens **only** in `physical_sonar_check/record_acoustic.py`.
`src/acoustic.py` stays pure, which is why it can be tested with no microphone.

---

## 5. The nine features

| # | Feature | Where |
|---|---|---|
| 1 | Pulse generator (rect / gaussian / chirp) | `pulse.py` |
| 2 | Channel: delay, attenuation, AWGN | `channel.py` |
| 3 | Matched filter + distance estimate | `receiver.py` |
| 4 | RMSE vs SNR sweep | `evaluate.py` |
| 5 | Resolution vs bandwidth sweep | `evaluate.py` |
| 6 | Band-pass filtering + comparison | `filters.py` |
| 7 | Multipath / multiple echoes | `channel.py` + `receiver.py` |
| 8 | Real acoustic validation | `acoustic.py` |
| 9 | Interactive interface | `app/streamlit_app.py` (Farhan's) |

Beyond the list: `audio.py` (listen to any signal) and the figure generator
`notebooks/make_figures.py`.

---

## 6. Numbers to have ready

| Quantity | Value | Where it comes from |
|---|---|---|
| Sampling rate | 48 000 Hz | standard audio rate |
| Speed of sound | 343 m/s | air, room temperature |
| Distance accuracy floor | **≈ 1.8 mm** | `c / (4·fs)` — half a sample |
| Rect bandwidth | `0.886 / T` | measured, matches theory |
| Gaussian bandwidth | `≈ 1.59 / T` | σ = T/6 |
| Resolution theory | `c / (2B)` | Rayleigh criterion |
| Measured resolution | **≈ 0.55 × theory** | consistent across 23× bandwidth |
| SNR cliff | below **≈ 0 dB** | flat above, catastrophic below |
| Real wall | **1.797 m, 6.7σ** | archived recording |
| Real hand | **0.272 m, 6.2σ** | live recording |
| Detection threshold | **> 6σ** | standard "this is real" bar |

> **Why is measured resolution better than theory (0.55×, not 1.0×)?**
> `c/(2B)` is a deliberately conservative convention. A peak-finder on a clean
> envelope can beat it. Reporting "tracks the scaling law at a consistent factor
> below unity" is a stronger claim than a fudged match.

> **Why does RMSE bottom out at 1.8 mm instead of 0?**
> Delay is measured in whole samples. `2d·fs/c` is rarely an integer, so rounding
> costs up to half a sample. Not a bug — pick a distance that lands on an exact
> sample and the error really is zero.

---

## 7. Tests — 33 of them

| File | Gates |
|---|---|
| `test_pipeline.py` | Noiseless exact recovery, 4 delays × 3 pulse types |
| `test_multipath.py` | Two objects found; one object not split in two |
| `test_resolution.py` | Resolution improves as bandwidth grows |
| `test_sweeps.py` | Band-pass cutoffs stay valid at extreme settings |
| `test_audio.py` | Clip shape, loudness, loop-safety, gain reuse |
| `test_acoustic.py` | **Re-derives a real recording bit-for-bit** |

Plus `check_pulse.py` — 17 checks on pulse maths, including the sampling-rate
stability check that caught the RMS-bandwidth problem.

The acoustic test is the interesting one: it loads the actual microphone capture
and reproduces the original hardware script's output to 12 decimal places. So
Feature 8 stays verified with no hardware attached.

---

## 8. Bugs found along the way (good "what did you learn" answers)

- **Negative distance silently worked.** A negative `true_distance_m` produced a
  negative index, which Python wrapped to a valid-looking spot near the end of
  the buffer. No error. Fixed by rejecting it up front.
- **A check that was dead code.** The buffer-overrun check sat *after* the write,
  so the write always failed first with a confusing NumPy error. Lesson:
  **validate before acting, not after.**
- **SNR depended on how long you listened.** Fixed by measuring signal power over
  the echo only.
- **Silent target dropping.** `zip(distances, attenuations)` truncates when the
  lists differ in length, so a third target just vanished. Now both are checked.
- **A detector setting faking a physics result** — the `min_separation_samples`
  floor described above. The most transferable lesson: *check you are measuring
  the physics and not your own tuning.*

---

## 9. Honest limitations

- The bandwidth sweep is **noiseless** on purpose, to isolate bandwidth. Noise
  only makes resolution worse, so those are best-case numbers.
- At very low time-bandwidth product (a 1000 Hz sweep over 2 ms), the −3 dB
  bandwidth under-reports the pulse's real resolving power, so that one point sits
  off the trend. An x-axis artifact, not a physics result.
- The resolution sweep shares one random generator across pulse configs, so adding
  or removing a config shifts later ones slightly. The trend and conclusion hold;
  the individual centimetre values would be steadier with per-config seeds.
- Detecting a *weaker* second target is harder than an equal one: 22/30 trials
  even at 20 dB SNR. Real, and worth reporting.

---

## 10. The app, section by section

Five sections, reached from the top bar. Sections 1, 2 and 5 are **live** — every
control redraws immediately. Sections 3 and 4 are **studies** — set them up, press
run, read the graph.

Layout is the same throughout: stages on the left, and on the right the reading,
the **Settings** tab and the **Audio** tab. Click **Enlarge** on a small panel to
swap it into the big slot.

---

### Section 1 — Send a Pulse, Find the Echo

**Answers:** can we find one echo, and how accurately?

**The three stages**

| Stage | Shows | Look for |
|---|---|---|
| 1 · Transmit — *The pulse we send* | The outgoing pulse. Toggle **Waveform / Spectrum** | Spectrum view marks the −3 dB width — the "sharpness" number |
| 2 · Receive — *What the microphone hears* | The whole recording: echo plus hiss. Shaded band = where the echo truly is | At 0 dB the echo is invisible by eye. That is the point |
| 3 · Detect — *Where the echo is* | Matched filter output. Dashed line marks the detected peak | One sharp spike out of that mess |

**Reading:** Estimated distance (large), with the error underneath, plus true
distance and round-trip time.

**Controls**

| Control | What it does, in plain words | Watch |
|---|---|---|
| **Target distance** | How far away you are pretending the object is. The simulator places the echo that far out, and the receiver has to find it again without being told. | The peak slides along stage 3 |
| **Noise level (SNR)** | How loud the echo is **compared with** the hiss. It is a ratio, so **higher means cleaner, not noisier** — the left end of the slider is the hard end. | 20 dB clean → 0 dB invisible by eye → −10 dB breaks |
| **Pulse shape** | Which beep to send. *rect* = abrupt on/off tone, *gaussian* = same tone faded in and out, *chirp* = a tone sliding from low to high. | Stage 1 changes shape; bandwidth changes with it |
| **Pulse length** | How long the beep lasts. Shorter beeps cover a wider range of frequencies, which makes them "sharper". | Shorter = wider bandwidth = narrower peak |
| **Carrier frequency** | The **pitch** of the tone inside the beep. 4000 Hz means the wave wiggles 4000 times a second. | Stage 1's wiggles get faster |
| **Chirp sweep width** | *(chirp only)* How far the pitch slides from start to end. This is what gives a chirp its bandwidth without making it shorter. | Wider = sharper correlation peak |
| **Advanced** | | |
| **Speed of sound** | The number used to convert time into distance. 343 m/s in air. | Change it and every estimate is wrong — it *is* the conversion factor |
| **Sampling rate** | How many measurements per second are taken, 16–96 kHz. Finer sampling = finer distance steps. | Sets the 1.8 mm accuracy floor |
| **Echo strength** | How much weaker the echo comes back. Does not move it, only quietens it. | Same position, smaller peak |
| **Noise seed** | Fixes the random generator, so the same number always gives the same hiss. | For demos you can repeat exactly |
| **Audio slowdown** | How much to stretch the sound so a 2 ms beep is audible. | 4× is the sweet spot |
| **Band-pass filter** | Throws away frequencies outside the pulse's band before detecting. | Barely changes anything — and that *is* the finding |

**Audio tab:** the outgoing ping and the received echo, slowed so a 2 ms pulse is
audible. Move the SNR slider and replay: the ping stays the same loudness while the
hiss grows, because both clips share one gain.

> **Demo:** set SNR to 0 dB. Stage 2 looks like pure static; stage 3 still spikes,
> and the reading is still right to a few millimetres.

---

### Section 2 — Two Targets, Can We Separate Them?

**Answers:** when do two objects stop looking like two?

**The two stages**

| Stage | Shows |
|---|---|
| 1 · Receive — *Both echoes arrive on top of each other* | The recording. Dashed lines mark the true targets |
| 2 · Detect — *One peak per target?* | The envelope of the matched filter. Two humps = resolved; one = merged |

**Reading:** Targets found (2/2 or 1/2), the separation you set, and the theory
limit `c/2B` — so you can see whether you are above or below it. Below is a table
with each target's truth, detection and error.

**Controls**

| Control | What it does, in plain words |
|---|---|
| **Target 1 distance** | Where the pair of objects sits. Moving it does not change whether they can be separated — only *where* the pair appears. |
| **Target 2 separation** | **The main dial.** How far behind target 1 the second object sits. Shrink it until the two humps merge into one. |
| **Target 1 / 2 reflectivity** | How strongly each object bounces sound back. A weaker second target is genuinely harder to spot than an equal one. |
| **Pulse shape / length / chirp width** | Together these set the **bandwidth**, and bandwidth is what sets how close two objects can be. Sharper beep, closer objects. |
| **Noise level (SNR)** | Higher = cleaner. Noise makes merging happen sooner. |
| **Advanced** | |
| **Add a third target** | Puts a third object in the scene, with its own distance and reflectivity. |
| **Speed / sampling rate / carrier** | Same meanings as section 1. |
| **Detection sensitivity** | How tall a hump must be to count as an object. **Lower finds more** — including fake ones. See section 12. |
| **Minimum gap between detections** | How close two humps may be before the detector calls them one object. Measured in samples, which is really a distance floor. See section 12. |

> **Demo:** start at 40 cm — two clean humps, 2/2. Shrink to 8 cm — one hump, 1/2.
> The crossover for the default pulse is around 15 cm, against a 21.8 cm theory limit.

> **Likely question: why does it resolve better than theory?** `c/2B` is a
> conservative convention, and a peak-finder on a clean envelope beats it.

---

### Section 3 — Bandwidth vs Resolution *(a study)*

**Answers:** is bandwidth really what sets resolution — not pulse length?

**Step 1 — Settings**, **Step 2 — The graph**. Log–log: measured points against the
dashed `c/2B` line. Parallel lines = same power law. Circles are gaussians, triangles
are chirps. A table underneath gives the ratio per pulse.

**Controls**

| Control | What it does, in plain words |
|---|---|
| **Which pulses to compare** | Both families, gaussians only, or chirps only. Both is the honest choice — see the note below. |
| **Gaussian pulse lengths** | Which durations to test. For a gaussian, changing the length *is* changing the bandwidth; they are locked together. |
| **Chirp sweep widths** | Which bandwidths to test. All chirps stay 2 ms long, so **only** bandwidth changes. That is the controlled experiment. |
| **Trials per separation** | How many times each separation is tested. Each trial nudges the pair to a slightly different distance, so one lucky wave alignment cannot be mistaken for a real result. More trials = steadier answer, slower run. |
| **Advanced** | |
| **Counts as resolved at** | What fraction of trials must succeed before a separation counts. 50% is the default: it has to work more often than not. |
| **Distance to the pair** | How far out the two objects sit. Should not change the answer — useful as a sanity check that it doesn't. |
| **Widest separation / step** | How far out to search and in what increments. A 1 cm step gives a finer answer than 5 cm, and takes five times longer. |
| **Detection sensitivity** | Same knob as section 2. See section 12. |
| **Minimum gap between detections** | **The one to be careful with.** Deliberately small here. Turn it up and you will measure the detector instead of the physics. See section 12. |

> **Why both families matter:** a gaussian's bandwidth is tied to its duration, so
> gaussians alone cannot separate "wider bandwidth helps" from "shorter pulse
> helps". The chirps hold duration fixed at 2 ms and vary only bandwidth. That is
> what pins the credit on bandwidth.

> **Careful with "minimum gap".** Turn it up and the curve flattens — you would be
> measuring the detector's floor, not the physics. That exact trap produced a
> perfectly flat, completely wrong result during development.

> **The numbers move with the settings.** Few trials and coarse steps read ~0.80×;
> the finer settings used for the report read ~0.55×. Coarser is more permissive.

---

### Section 4 — How Much Noise Before It Fails? *(a study)*

**Answers:** how much noise can it take?

**Step 1 — Settings**, **Step 2 — The graph**. Distance error against SNR, log y.
Blue = matched filter alone, purple = with the band-pass filter, dashed amber = the
1.8 mm half-sample limit.

**Shape is the result:** flat and near-perfect above 0 dB, then a **cliff**. Below a
critical SNR the peak occasionally locks onto a noise sidelobe instead of the echo,
so errors jump to metres rather than drifting.

**Controls**

| Control | What it does, in plain words |
|---|---|
| **Evaluation distance** | Where the single target sits for the whole sweep. Pick a distance that lands on a whole number of samples and the error at high SNR really is zero. |
| **Noise range (SNR)** | The span of noise levels to test, as a two-ended slider. **The left end is the difficult end.** |
| **Trials per point** | How many noisy runs at each noise level. The result is the RMS error over those trials, so more trials means a smoother, more trustworthy curve — and a slower run. |
| **Pulse shape** | Which beep to test. The cliff appears for all three, at slightly different places. |
| **Advanced** | Step size between noise levels, plus pulse length and carrier. |
| **Also run with a band-pass filter** | Runs the whole sweep a second time with the filter on and draws it in purple. The two curves landing on top of each other is the result, not a bug. |

Progress is counted honestly while it runs. Change a setting after a run and the old
graph stays with an amber "settings changed" note rather than silently clearing.

> **Demo:** leave the band-pass on. The two curves sit on top of each other —
> pre-filtering buys nothing against white noise, because the matched filter is
> already the optimal detector.

---

### Section 5 — Real Sound Test *(live hardware)*

**Answers:** does any of this work on real sound?

**The three stages**

| Stage | Shows |
|---|---|
| 1 · Compare — *This recording vs the empty room* | Grey = empty room, blue = this recording. Mostly identical — that is the clutter |
| 2 · Detect — *What is new in the room* | After subtracting the empty room. What survives is the target |
| Transmit — *The chirp we send* | The 2–8 kHz, 10 ms sweep |

**Reading:** measured distance and a confidence in **sigma**. Above 6σ is a real
detection.

**Controls**

| Control | What it does, in plain words |
|---|---|
| **Where the recording comes from** | A saved wall recording (1.80 m), a saved hand recording (0.27 m), or record live with your own speaker and microphone. The saved ones always work, so the demo cannot fail. |
| **Speed of sound in the room** | The time-to-distance conversion again. Warmer air is slightly faster. |
| **Ignore closer than** | The laptop's own speaker rings for a moment after each chirp, and the microphone hears it. Anything inside this range is the laptop hearing itself, not an object. |
| **Ignore further than** | The far end of the search. Beyond this, returns are too weak to trust. |
| *(live)* **Chirps per burst** | How many times to transmit and stack. 16 is twice as slow as 8 but roughly √2 cleaner, because the echo adds up while random noise averages out. |
| *(live)* **Record room baseline** | Step 1. Point at open space and record what the room sounds like with nothing in front of you. |
| *(live)* **Measure the echo** | Step 2. Aim at the target and record again. The baseline gets subtracted, so only what *changed* survives. |

> **Live procedure:** baseline facing open space, then measure facing the target,
> **without changing the system volume in between** — the subtraction only cancels
> if the speaker ringing is identical.

> **Two results already banked:** a wall at **1.797 m (6.7σ)** and a hand at
> **0.272 m (6.2σ)**. Both survive in saved recordings, so the demo works even if
> the room is noisy or the microphone is blocked.

---

## 11. Glossary

Every term the project uses, in plain words.

### The signal

**Pulse** — the short "beep" we send. Just a list of numbers.

**Buffer** — the long recording we listen to afterwards. Mostly silence, with the
echo somewhere inside. *Where* the echo sits is the distance.

**Carrier frequency** — the pitch of the tone inside the pulse. 4000 Hz here.

**Envelope (of a pulse)** — the smooth outline the oscillation fits inside. A
gaussian pulse is a tone with a bell-shaped envelope; a rect pulse's envelope is a
flat-topped box.

**Chirp (LFM, linear frequency modulation)** — a pulse whose pitch slides from low
to high. Matters because it **breaks a trade-off**: normally a short pulse gives
good resolution but carries little energy. A chirp can be long (lots of energy) and
still wide in bandwidth (good resolution).

**Sampling rate (`fs`)** — how many measurements per second. 48 000 Hz here. Sets
how finely time — and therefore distance — can be measured.

**Nyquist** — half the sampling rate; the highest frequency that can be represented.
Above it, a frequency *aliases*: it masquerades as a lower one. At `fs` = 48 kHz,
Nyquist is 24 kHz.

**Unit energy** — scaling every pulse so its squared samples sum to 1, so all shapes
carry the same total energy and comparisons under noise are fair.

### Frequency

**FFT / spectrum** — the Fourier transform converts a signal from *amplitude over
time* into *how much of each frequency it contains*. Same signal, different view.

**Bandwidth** — how wide a range of frequencies a pulse occupies. Wider = "sharper"
= better at separating close objects.

**dB (decibel)** — a logarithmic ratio, `10·log₁₀(power ratio)`. Compresses huge
ranges: +10 dB = 10× the power, +20 dB = 100×, −10 dB = one tenth.

**−3 dB** — half power (`10·log₁₀(0.5) ≈ −3`). Bandwidth is measured at this level,
so "−3 dB bandwidth" means the width of the band carrying at least half the peak power.

**Time-bandwidth product** — duration × bandwidth. Low values give a chirp soft
spectral edges, so its measured bandwidth falls short of its nominal sweep.

### Noise

**SNR (signal-to-noise ratio)** — how loud the echo is compared with the hiss, in dB.
**0 dB means equal.** +20 dB = echo 100× stronger. −10 dB = noise 10× stronger. Higher
is easier.

> **The classic confusion: higher SNR means LESS noise, not more.** SNR is a
> *ratio* — echo power divided by noise power — so turning it up makes the job
> easier, not harder. If you want to think in terms of "how much noise", read the
> slider backwards: the **left** end is the noisy, difficult end.
>
> ```
> SNR (dB) = 10 · log₁₀( echo power / noise power )
> ```
>
> This is why the error curve in section 4 falls as you move right: more signal
> relative to noise, less error.

**AWGN (additive white Gaussian noise)** — the standard noise model.
*Additive* = added on top of the signal. *White* = equal power at every frequency.
*Gaussian* = sample values follow a bell curve.

**Seed** — a number that fixes the random generator, so the same "random" noise can
be reproduced exactly. Essential for fair comparisons and repeatable tests.

**Monte Carlo** — running many randomised trials and averaging, because one trial is
one dice roll.

### Finding the echo

**Matched filter** — slide the known transmitted pulse along the noisy recording and
score the overlap at every position. The echo lines up and scores high; hiss does not
line up and averages out. **Provably the best possible detector** for a known shape
in white noise.

**Cross-correlation** — the maths that does that sliding-and-scoring (`np.correlate`).
The matched filter *is* a cross-correlation with the transmitted pulse.

**argmax** — "argument of the maximum": **not the biggest value, but its position**.
`np.argmax([3, 9, 4])` returns `1`, not `9`. Central here because position is what
encodes distance — we don't care how strong the match was, only *where* it was.

**Envelope (Hilbert / analytic envelope)** — the smooth curve traced over the peaks
of an oscillating signal. The raw correlation wobbles at the carrier frequency, so
counting its crests finds dozens of fake "peaks". The envelope smooths that away,
leaving one hump per real echo. Used for *counting* echoes, never for *locating* a
single one — smoothing widens the peak and makes it easier for noise to shift.

**Prominence** — how far a peak rises above the dips around it. Used to ignore small
ripples and keep only real echoes.

**Sidelobe** — the smaller secondary peaks either side of the main correlation peak.
At low SNR, noise can push a sidelobe above the true peak — the detector locks onto
the wrong one, which is exactly what causes the **SNR cliff**.

### Measuring how good it is

**RMSE (root-mean-square error)** — square every error, average them, take the square
root. Keeps the answer in metres and punishes a few big misses more than many small
ones.

**Quantisation** — rounding to whole samples. Delay can only be a whole number of
samples, so there is a floor of about **1.8 mm** (`c / 4·fs`) below which distance
cannot be measured, even with zero noise.

**Range resolution** — the closest two objects can be while still being seen as two.

**Rayleigh criterion** — the conventional rule for when two peaks count as separated.
Here `ΔR ≈ c / (2B)`. Deliberately conservative, which is why measurements beat it.

**Sigma (σ)** — standard deviation, a measure of spread. "6σ" means a peak stands six
standard deviations above the background — far too large to be chance. The usual bar
for "this is a real detection".

### Filtering

**Butterworth filter** — a standard filter design with a maximally flat passband
(no ripple in the frequencies it keeps).

**Band-pass** — keeps a band of frequencies, discards everything above and below.

**Group delay** — the time shift a filter introduces. Fatal here: a shifted signal
means a shifted echo means a wrong distance.

**`filtfilt`** — runs the filter forwards, then backwards. The two passes cancel each
other's phase shift, giving **zero group delay**. This is why `filtfilt` is used and
`lfilter` is not.

### Real-world extras

**Attenuation** — how much weaker the returning echo is than what was sent.

**Superposition** — waves simply add. Three objects produce exactly the sum of three
one-object recordings, which is why multi-target simulation is a few lines.

**Multipath** — several echoes arriving from several reflectors or paths.

**Clutter** — returns from things you do not care about: the desk, the walls, the
laptop's own chassis ringing.

**Direct path** — the speaker's sound reaching the microphone directly, without
bouncing off anything. It arrives first and is used as the **t = 0 reference**.

**Coherent averaging (frame folding)** — transmitting many times and stacking the
recordings. The echo lands in the same place every time so it grows; random noise
does not, so it averages away.

---

## 12. The four settings people get wrong

The rest are self-explanatory. These four are not.

---

### Sampling rate — how often you measure

Think of filming. A camera at 30 fps takes 30 photos a second, and that is all a
video is. Sampling rate is the same for sound: **48 000 Hz means 48 000 snapshots
of the wave every second.** Sound is a continuous wobble in the air; a computer
cannot store "continuous", so it measures the height 48 000 times a second and
keeps those numbers. The whole project is those numbers.

**It decides two things.**

**1. Your accuracy floor.** One sample lasts 20.83 microseconds. Sound covers
7.1 mm in that time — but it is a round trip, so it is **3.573 mm of target
distance per sample**. Delay can only be a whole number of samples, so rounding
costs up to half of that:

```
half-sample error = 1.786 mm      ← this is the 1.8 mm floor
```

That is why RMSE bottoms out at 1.8 mm instead of zero. Double the sampling rate
and the floor halves.

**2. The highest pitch you can record — Nyquist.** You need at least two samples
per wave cycle to see a wave at all: one for the top, one for the bottom. So the
highest frequency you can represent is **half the sampling rate** — 24 000 Hz here.
Above it a frequency *aliases*: it masquerades as a lower one, like wagon wheels
appearing to spin backwards in old films.

That is the rule behind the app's guard: a 12 000 Hz carrier at a 16 000 Hz
sampling rate is impossible, because Nyquist there is only 8 000 Hz. The app
clamps it and says so rather than crashing.

---

### Carrier frequency — the pitch of the beep

4000 Hz means the wave wiggles up and down **4000 times a second**. Low number =
deep hum, high number = shrill whistle.

**A pulse is two things multiplied together:**

- the **carrier** — the fast wiggle, which sets the *pitch*
- the **envelope** — the slow shape fading it in and out, which sets *how long and
  how loud*

Like briefly pressing one piano key. The key you press is the carrier; how long you
hold it is the envelope.

**How it all fits, in real numbers:**

```
sampling rate   48 000 samples/sec
carrier          4 000 Hz   →  48000 / 4000 = 12 samples per wiggle
2 ms pulse      →  96 samples  =  8 complete wiggles
```

So the transmitted pulse is literally **96 numbers**, holding **8 wiggles** of a
4 kHz tone, each drawn with **12 points**. Count the humps in stage 1 and you will
find eight.

> **Sampling rate vs carrier — the distinction that matters.** They sound similar
> and do opposite jobs. Sampling rate is how finely you *record*; carrier is what
> you *send*. Sampling rate is the ruler, carrier is the thing being measured — and
> the ruler must be at least twice as fine as the thing. That is Nyquist.

---

### Detection sensitivity — how tall must a bump be?

Think of a mountain range. A small lump on the side of a big mountain is not its
own mountain. What makes something a peak is how far it rises above the dip beside
it — its **prominence**.

This slider sets the bar, as a fraction of the tallest peak. At `0.4`, a bump must
rise **40% of the biggest peak's height** above the dips around it to count.

```
lower  (0.05)  → very sensitive → finds more peaks, including invented ones
higher (0.80)  → very strict    → only obvious peaks, misses real weak echoes
```

**It is a genuine trade-off, measured rather than guessed.** Calibrated on this
project at 0 dB noise:

| Setting | False alarms | Catches a real weak second echo |
|---|---|---|
| 0.2 | **100%** — one object reported as two, every time | yes |
| **0.4** | ~10% | sometimes — chosen as the default |
| 0.7 | ~0% | almost never |

There is no free setting. Loosen it and you invent objects; tighten it and you miss
real ones.

**Why it is needed at all:** the envelope is never perfectly smooth. Real echoes sit
on a rippling baseline, and without a bar every ripple becomes an "object".

---

### Minimum gap between detections — the one that can fake a result

**How close can two humps be before you decide they are the same object?**

Measured in **samples**, and a sample is a fixed distance — 3.573 mm. So this
setting is really a distance floor in disguise:

```
  8 samples  →   2.9 cm      (bandwidth study default)
 24 samples  →   8.6 cm      (section 2 default, 2 ms pulse)
 48 samples  →  17.1 cm      (the receiver's own default for a 2 ms pulse)
```

Set it to 24 and the detector **physically cannot report two targets closer than
8.6 cm**, whatever the physics says.

**Why it is needed:** a single echo's envelope wobbles slightly, and two adjacent
wobbles on one hump would otherwise be counted as two objects.

**Why it is dangerous.** The receiver's built-in default is `len(pulse) // 2` —
half the pulse length. That sounds sensible until you see what it means:

| Pulse | Bandwidth | Theory limit c/2B | Detector floor |
|---|---|---|---|
| 2 ms gaussian | 785 Hz | 21.8 cm | 17.1 cm |
| 10 ms gaussian | 156 Hz | 109.8 cm | **85.8 cm** |

The floor is set by **pulse length**, which has nothing to do with bandwidth. For
gaussians the two happen to move together, so the problem hides. But for a **chirp**
— duration fixed at 2 ms while bandwidth changes — the floor stays frozen at 17.1 cm
while the true limit falls to 3 cm.

That is why the bandwidth study forces it down to **8 samples**. Left at the default,
all three chirps reported exactly 17.0 cm across a 3.5× bandwidth range: a perfectly
flat line that looked like a real finding and said the *opposite* of the truth.

> **The one-line version:** these last two settings control the **detector**, not the
> physics. Set them wrong and you measure your own tuning.
