"""
Throwaway validation probe for src/pulse.py.

Not part of the repo's test suite -- just a quick pass/fail sanity check that
generate_pulse() and pulse_bandwidth_hz() agree with theory before moving on.

Run from the repo root:  python check_pulse.py
"""

import sys

import numpy as np

sys.path.insert(0, "src")
from pulse import generate_pulse, pulse_bandwidth_hz  # noqa: E402

FS = 48_000.0
DURATION_S = 0.01
FREQ_HZ = 5_000.0
CHIRP_SWEEP_HZ = 2_000.0

TOL_PCT = 10.0

results = []


def check(label, measured, expected, tol_pct=TOL_PCT, unit=""):
    err_pct = abs(measured - expected) / expected * 100 if expected else float("inf")
    ok = err_pct <= tol_pct
    results.append(ok)
    print(
        f"[{'PASS' if ok else 'FAIL'}] {label:<34} "
        f"measured {measured:>10.3f}{unit}   "
        f"expected {expected:>10.3f}{unit}   ({err_pct:5.1f}% off)"
    )


def check_bool(label, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {label:<34} {detail}")


print("=" * 92)
print(f"fs={FS:.0f} Hz   duration={DURATION_S*1000:.1f} ms   "
      f"carrier={FREQ_HZ:.0f} Hz   chirp sweep={CHIRP_SWEEP_HZ:.0f} Hz")
print("=" * 92)

n_expected = int(round(DURATION_S * FS))
sigma_s = DURATION_S / 6

rect = generate_pulse("rect", DURATION_S, FS, freq_hz=FREQ_HZ)
gauss = generate_pulse("gaussian", DURATION_S, FS, freq_hz=FREQ_HZ)
chirp = generate_pulse("chirp", DURATION_S, FS, freq_hz=FREQ_HZ,
                       bandwidth_hz=CHIRP_SWEEP_HZ)

print("\n-- generate_pulse ------------------------------------------------------")

for name, pulse in (("rect", rect), ("gaussian", gauss), ("chirp", chirp)):
    check_bool(
        f"{name}: length == {n_expected}",
        len(pulse) == n_expected,
        f"got {len(pulse)}",
    )

for name, pulse in (("rect", rect), ("gaussian", gauss), ("chirp", chirp)):
    energy = float(np.sum(pulse ** 2))
    check_bool(
        f"{name}: unit energy",
        abs(energy - 1.0) < 1e-9,
        f"sum(pulse^2) = {energy:.12f}",
    )

# argmax is unreliable here: the carrier oscillates, so the largest single
# sample lands on whichever carrier crest is nearest the centre, not at the
# centre itself. Energy centroid is carrier-independent.
idx = np.arange(n_expected)
centroid = float(np.sum(idx * gauss ** 2) / np.sum(gauss ** 2))
midpoint = (n_expected - 1) / 2
check_bool(
    "gaussian: envelope centred",
    abs(centroid - midpoint) <= 1.0,
    f"energy centroid {centroid:.2f}, midpoint {midpoint:.1f}",
)
check_bool(
    "gaussian: symmetric",
    np.max(np.abs(gauss - gauss[::-1])) < 1e-12,
    f"max |p - reversed(p)| = {np.max(np.abs(gauss - gauss[::-1])):.2e}",
)

print("\n-- pulse_bandwidth_hz --------------------------------------------------")

check("rect: B ~ 0.886 / T", pulse_bandwidth_hz(rect, FS),
      0.886 / DURATION_S, unit=" Hz")
check("gaussian: B ~ 0.265 / sigma", pulse_bandwidth_hz(gauss, FS),
      0.265 / sigma_s, unit=" Hz")
# A chirp's spectral edges roll off, so the -3 dB width sits INSIDE the
# nominal sweep. The ratio rises toward 1.0 as the time-bandwidth product
# grows (~0.83 at TB=20, ~0.97 at TB=800), so allow a wide tolerance here.
tb_product = DURATION_S * CHIRP_SWEEP_HZ
check("chirp: B ~ sweep width", pulse_bandwidth_hz(chirp, FS),
      CHIRP_SWEEP_HZ, tol_pct=25.0, unit=" Hz")
print(f"       (time-bandwidth product = {tb_product:.0f}; "
      f"expect B/sweep well below 1.0 at low TB)")

print("\n-- fs-stability (the important one) ------------------------------------")

fs_lo, fs_hi = 48_000.0, 96_000.0
b_lo = pulse_bandwidth_hz(
    generate_pulse("rect", DURATION_S, fs_lo, freq_hz=FREQ_HZ), fs_lo)
b_hi = pulse_bandwidth_hz(
    generate_pulse("rect", DURATION_S, fs_hi, freq_hz=FREQ_HZ), fs_hi)
drift_pct = abs(b_hi - b_lo) / b_lo * 100
check_bool(
    "rect: B stable across fs",
    drift_pct <= 5.0,
    f"{b_lo:.3f} Hz @ {fs_lo/1000:.0f} kHz  vs  "
    f"{b_hi:.3f} Hz @ {fs_hi/1000:.0f} kHz   ({drift_pct:.2f}% drift)",
)

print("\n-- input validation ----------------------------------------------------")

bad_calls = [
    ("rejects unknown pulse_type", lambda: generate_pulse("triangle", DURATION_S, FS, freq_hz=FREQ_HZ)),
    ("rejects missing freq_hz", lambda: generate_pulse("rect", DURATION_S, FS)),
    ("rejects aliasing freq_hz", lambda: generate_pulse("rect", DURATION_S, FS, freq_hz=FS)),
    ("rejects negative duration", lambda: generate_pulse("rect", -1.0, FS, freq_hz=FREQ_HZ)),
    ("rejects chirp beyond Nyquist", lambda: generate_pulse("chirp", DURATION_S, FS, freq_hz=FREQ_HZ, bandwidth_hz=FS)),
]

for label, call in bad_calls:
    try:
        call()
    except ValueError:
        check_bool(label, True, "raised ValueError")
    except Exception as exc:
        check_bool(label, False, f"raised {type(exc).__name__} instead of ValueError")
    else:
        check_bool(label, False, "no exception raised")

print("\n" + "=" * 92)
passed, total = sum(results), len(results)
print(f"{passed}/{total} checks passed"
      f"{'  -- all good' if passed == total else '  -- see FAIL lines above'}")
print("=" * 92)
