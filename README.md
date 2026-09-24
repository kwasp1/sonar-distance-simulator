# Sonar Distance Simulator

CSE220 Signals and Systems — Team SevenEight
Aditya Tirtho Roy (2305178) & Farhan Ehsas Sami (2305177)

Pulse-echo ranging simulator: matched filtering / cross-correlation based
delay estimation, evaluated under additive noise and pulse-bandwidth variation.

Full spec: see `docs/Sonar_Distance_Simulator_Documentation.docx`.

## Setup

```bash
python -m venv sonar-env
source sonar-env/bin/activate      # or sonar-env\Scripts\activate on Windows
pip install -r requirements.txt
```

`sounddevice` is only needed to record real audio; drop that line from
`requirements.txt` if you just want the simulation and the tests.

```bash
python -m pytest -q                    # 29 tests
python check_pulse.py                  # 17 pulse-maths checks
python notebooks/make_figures.py       # regenerates every report figure
```

## Structure

```
src/            core pipeline — pure functions, no plotting/printing
  pulse.py        transmit pulses + bandwidth measurement
  channel.py      delay, attenuation, multiple targets, AWGN
  receiver.py     matched filter, single and multi-target ranging
  filters.py      Butterworth band-pass
  evaluate.py     SNR and bandwidth sweeps
  audio.py        makes any signal listenable (.wav bytes)
  acoustic.py     analysis of real speaker/microphone recordings
notebooks/      exploration + demo plots (Matplotlib)
app/            Streamlit webapp — owned by Farhan, do not edit
tests/          pytest suite
results/        generated plots for the report (audio/plots are gitignored)
physical_sonar_check/   hardware capture script + archived recordings
docs/           project documentation
```

## Golden rule

Every function in `src/` takes parameters and returns numpy arrays / numbers.
It never calls `plt.show()` or `print()`. All display logic (Matplotlib in
notebooks, Streamlit in app/) lives outside `src/` and just calls these
functions. This is what makes adding the webapp at the end free instead of
a rewrite.

## Timeline

| Week | Focus |
|---|---|
| 1 | Pulse generator + channel model, noiseless sanity test |
| 2 | Matched filter, AWGN, SNR sweep, RMSE plot |
| 3 | Band-pass filtering, bandwidth/resolution study |
| 4 | Evaluation, report, demo (+ stretch goals if ahead of schedule) |

All nine committed features are implemented. Remaining work is report
production: plots into `results/`, then the write-up.
