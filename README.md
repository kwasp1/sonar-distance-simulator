# Sonar Distance Simulator

CSE220 Signals and Systems — Team SevenEight
Aditya Tirtho Roy (2305178) & [Team Member 2]

Pulse-echo ranging simulator: matched filtering / cross-correlation based
delay estimation, evaluated under additive noise and pulse-bandwidth variation.

Full spec: see `docs/Sonar_Distance_Simulator_Documentation.docx`.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Structure

```
src/            core pipeline — pure functions, no plotting/printing
notebooks/      exploration + demo plots (Matplotlib)
app/            optional Streamlit webapp (added last, if time allows)
tests/          sanity checks (e.g. noiseless delay recovery)
results/        generated plots for the report
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
