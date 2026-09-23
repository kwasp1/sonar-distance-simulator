# Working with this codebase

**Sonar Distance Simulator — CSE220, Team SevenEight**

A practical guide to using `src/` from the Streamlit app. It assumes you know
Python but nothing about the signal processing underneath. If you only read one
section, read **The one rule** and **Things that will bite you**.

---

## 1. What the project does, in one paragraph

Send out a short sound. Wait for it to bounce off something. Measure how long
it took, multiply by the speed of sound, divide by two — that is the distance.
The maths is 80 years old and boring. What the project actually measures is how
well it holds up under stress: how much background noise it survives, and how
close two objects can be before they blur into one. Those two questions are the
two graphs in the report.

---

## 2. Getting it running

```bash
python -m venv sonar-env
source sonar-env/bin/activate       # sonar-env\Scripts\activate on Windows
pip install -r requirements.txt

python -m pytest -q                 # 29 tests, all should pass
streamlit run app/streamlit_app.py
```

If an import fails in your editor, it is almost always the interpreter: press
`Ctrl/Cmd + Shift + P`, choose **Python: Select Interpreter**, pick `sonar-env`.

`sounddevice` is only needed to record real audio. Delete that line from
`requirements.txt` if you don't want PortAudio on your machine — everything
else still works.

---

## 3. The one rule

> **Functions in `src/` take numbers and return numbers. They never plot,
> print, play sound, or record.**

All display lives outside `src/`: Matplotlib in `notebooks/`, Streamlit in
`app/`. This is why the app was cheap to build — it calls the same functions
the tests do and just draws the results.

If you ever feel like adding `st.something` or `plt.something` inside `src/`,
that is the signal that the logic belongs in your file instead.

---

## 4. Who owns what

| Area | Owner |
|---|---|
| `app/streamlit_app.py` | **You.** Nobody else edits it. |
| `src/`, `tests/` | Aditya |
| Report, evaluation | Both |

The important consequence runs the other way: **your app imports ten functions
from `src/`**, so those signatures are kept stable. New capability arrives as
new functions, not as changes to existing ones. If something in `src/` ever
has to change shape, you'll be told before it lands.

---

## 5. The functions you can call

Everything below is importable and stable. All distances are metres, all times
seconds, all frequencies Hz — the unit is in the argument name.

### Making a pulse — `src/pulse.py`

```python
from src.pulse import generate_pulse, pulse_bandwidth_hz

pulse = generate_pulse("gaussian", duration_s=0.002, fs=48000.0, freq_hz=4000.0)
pulse = generate_pulse("chirp", 0.002, 48000.0, freq_hz=8000.0, bandwidth_hz=4000.0)

bandwidth = pulse_bandwidth_hz(pulse, 48000.0)      # -> 785.2
```

Types are `"rect"`, `"gaussian"`, `"chirp"`. `freq_hz` is required for all
three. `bandwidth_hz` only does anything for a chirp.

### Sending and corrupting it — `src/channel.py`

```python
from src.channel import simulate_channel, add_noise, simulate_multi_object_channel

clean = simulate_channel(pulse, true_distance_m=5.0, fs=48000.0,
                         speed_mps=343.0, attenuation=1.0, buffer_duration_s=0.1)

noisy = add_noise(clean, snr_db=10.0, seed=42)      # seed=None for a fresh draw

scene = simulate_multi_object_channel(pulse, [5.0, 6.0], 48000.0,
                                      attenuations=[1.0, 0.7],
                                      speed_mps=343.0, buffer_duration_s=0.1)
```

### Finding the echo — `src/receiver.py`

```python
from src.receiver import matched_filter, estimate_distance, estimate_multiple_distances

distance_m, correlation = estimate_distance(noisy, pulse, fs=48000.0, speed_mps=343.0)

distances = estimate_multiple_distances(scene, pulse, fs=48000.0, speed_mps=343.0,
                                        prominence_frac=0.4)
# -> [4.999, 5.999]   a plain sorted list
```

Note the shapes differ: `estimate_distance` returns a **tuple**
`(distance, correlation)` so you can plot the correlation without recomputing
it; `estimate_multiple_distances` returns **just a list**.

### Cleaning up — `src/filters.py`

```python
from src.filters import bandpass_filter

filtered = bandpass_filter(noisy, fs=48000.0, low_hz=3000.0, high_hz=5000.0, order=4)
```

Cutoffs must satisfy `0 < low_hz < high_hz < fs/2` or it raises.

### The report sweeps — `src/evaluate.py`

```python
from src.evaluate import run_snr_sweep, run_bandwidth_resolution_sweep

noise = run_snr_sweep(true_distance_m=5.0, snr_values_db=[20, 10, 0, -5],
                      fs=48000.0, trials_per_snr=20, bandpass=False)
# -> {"snr_db": [...], "rmse_m": [...]}

configs = [
    {"pulse_type": "gaussian", "duration_s": 0.004, "freq_hz": 4000.0},
    {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 8000.0,
     "bandwidth_hz": 4000.0},
]
resolution = run_bandwidth_resolution_sweep(configs, fs=48000.0, trials=12)
# -> {"labels": [...], "bandwidth_hz": [...], "resolved_m": [...], "theory_m": [...]}
```

`resolved_m` entries can be `None` when a pulse never separated the targets —
guard for that before plotting.

### Making it audible — `src/audio.py`

```python
from src.audio import make_audible, audible_echo, to_wav_bytes

clip = make_audible(pulse, fs=48000.0)          # {"audio", "rate_hz", "gain"}
wav = to_wav_bytes(clip["audio"], clip["rate_hz"])

echo = audible_echo(pulse, distance_m=5.0, fs=48000.0, snr_db=0.0, frames=4)
```

A 2 ms pulse is too short and quiet to hear, so this slows it 4x, repeats it
with gaps into a sonar-like ping, and fades the edges so it loops without
clicking.

### Real speaker and microphone — `src/acoustic.py`

```python
from src.acoustic import design_chirp, analyse_recording

chirp = design_chirp(2000.0, 8000.0, duration_s=0.010, fs=48000.0)
result = analyse_recording(recording, chirp, frame_samples=12480, fs=48000.0,
                           baseline_envelope=baseline)
# -> {"envelope", "excess", "distance_m", "quality_sigma", "peak_index"}
```

Recording itself happens in `physical_sonar_check/record_acoustic.py`, the only
file in the project that touches the speaker and microphone. Above 6 sigma is a
real detection. Confirmed twice against reality: a wall at 1.797 m and a hand
held above the keyboard at 0.272 m.

---

## 6. Adding the audio buttons to the app

This is the one piece of `src/` built for you but not yet wired in. Streamlit's
audio widget already has its own play button and loops natively, so you do not
need `st.button` — and using one would fight the framework, since every widget
interaction reruns the script and would restart playback.

```python
import streamlit as st
from src.audio import make_audible, audible_echo, to_wav_bytes

@st.cache_data
def pulse_wav(pulse_type, duration_s, fs, freq_hz):
    pulse = generate_pulse(pulse_type, duration_s, fs, freq_hz=freq_hz)
    clip = make_audible(pulse, fs)
    return to_wav_bytes(clip["audio"], clip["rate_hz"])

st.audio(pulse_wav(pulse_type, duration_s, fs, freq_hz),
         format="audio/wav", loop=True)
```

`@st.cache_data` matters: without it every rerun regenerates identical bytes and
interrupts playback for no reason.

For a set of clips at different SNR, pass one shared `gain` — see the next
section.

---

## 7. Things that will bite you

**Pulses are normalised to equal energy.** A gaussian and a rect of the same
nominal amplitude carry the same total power, so comparing them under identical
noise is fair. It also means peak amplitude is small (~0.14), which is normal.

**`add_noise` must receive the clean buffer.** It measures signal power over
the non-zero samples only, which assumes silence is exactly zero. Calling it
twice on an already-noisy buffer gives a wrong SNR. Always
`simulate_channel` → `add_noise`, never `add_noise` → `add_noise`.

**Sharing a gain across audio clips: take it from the LOUDEST one.** For an SNR
ladder that is the noisiest, lowest-SNR clip. Per-clip volume erases the
difference you are demonstrating; taking the gain from the quietest clip makes
every other one clip and distort.

```python
noisiest = audible_echo(pulse, 5.0, fs, snr_db=-5.0, frames=4)
for snr in (20.0, 10.0, 0.0, -5.0):
    clip = audible_echo(pulse, 5.0, fs, snr_db=snr, frames=4,
                        gain=noisiest["gain"])
```

**`run_snr_sweep(bandpass=True)` can raise.** It derives the band from the
measured pulse bandwidth, and that can fall outside `0 .. fs/2`. Bandwidth goes
roughly as `1/duration_s`, so a **longer** `duration_s` fixes a negative
`low_hz`, and a **lower** `freq_hz` fixes a `high_hz` past Nyquist. Wrap it in a
try/except if a slider can reach those values.

**Distance has a floor of about 1.8 mm.** Delay is measured in whole samples at
48 kHz. RMSE bottoming out near 0.002 m instead of 0 at high SNR is sample
quantisation, not a bug.

**Band-pass filtering barely helps.** That is the correct result, not a broken
filter — the matched filter is already the optimal detector against white noise,
so there is little left for a pre-filter to remove. A band narrower than the
pulse's own bandwidth makes things actively worse.

---

## 8. Where everything lives

```
src/            the pipeline — pure functions, no display
  pulse.py        transmit pulses + bandwidth measurement
  channel.py      delay, attenuation, multiple targets, noise
  receiver.py     matched filter, single and multi-target ranging
  filters.py      Butterworth band-pass
  evaluate.py     the two report sweeps
  audio.py        makes any signal listenable
  acoustic.py     real speaker/microphone ranging
app/            Streamlit app — yours
tests/          29 pytest cases
notebooks/      report plots (still to be written)
results/        generated output; audio and plots are gitignored
physical_sonar_check/   hardware capture script + archived recordings
```

---

## 9. Status

All nine committed features are implemented and tested. What remains is report
production: plots into `results/`, then the write-up. Nothing in `src/` is
waiting on you, and nothing you build in the app is blocked.
