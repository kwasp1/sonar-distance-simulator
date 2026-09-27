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
