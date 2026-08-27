"""
sonar_probe.py -- throwaway hardware feasibility probe. NOT project code.

v3: period-folding. The chirps repeat every n_frame samples, so the
correlation is folded on that period -- capture latency only rotates the
folded frame, it cannot misalign the average. No peak-hunting before
averaging, so there is nothing to lock onto the wrong sample.

    python3 sonar_probe.py

System output volume ~70%.
"""

import numpy as np
import sounddevice as sd

FS      = 48_000
C       = 343.0
F0, F1  = 2_000, 8_000
T_CHIRP = 0.010
GAP     = 0.250
N_REP   = 16
MIN_M   = 0.40
MAX_M   = 5.00
AMP     = 0.50

n_chirp = int(T_CHIRP * FS)
n_frame = int((T_CHIRP + GAP) * FS)

# --- transmit ------------------------------------------------------------
t = np.arange(n_chirp) / FS
chirp = np.sin(2 * np.pi * (F0 * t + (F1 - F0) / (2 * T_CHIRP) * t**2))
chirp *= np.hanning(n_chirp)

frame = np.zeros(n_frame)
frame[:n_chirp] = AMP * chirp
tx = np.concatenate([np.tile(frame, N_REP), np.zeros(n_frame)])

stereo = np.zeros((tx.size, 2))
stereo[:, 1] = tx                             # RIGHT channel only

rx = sd.playrec(stereo, samplerate=FS, channels=1, blocking=True)[:, 0]
np.save("rx.npy", rx)                          # raw recording, for diagnosis

# --- correlate, then fold on the repetition period -----------------------
corr = np.correlate(rx, chirp, mode="full")
n_use = (corr.size // n_frame) * n_frame
fold = corr[:n_use].reshape(-1, n_frame).sum(axis=0) / (n_use // n_frame)
env = np.abs(fold)

# --- direct path must be the largest thing in the folded frame ----------
p = int(np.argmax(env))
env = np.roll(env, -p)                         # direct now at index 0
med = np.median(env)
direct_q = env[0] / med

lo = int(2 * MIN_M / C * FS)
hi = int(2 * MAX_M / C * FS)
echo = lo + int(np.argmax(env[lo:hi]))
d = C * echo / FS / 2
echo_q = env[echo] / med

print(f"input peak level : {np.max(np.abs(rx)):.3f}   (want 0.05-0.9)")
print(f"frames folded    : {n_use // n_frame}")
print(f"DIRECT quality   : {direct_q:.1f}x median   (<20 = no usable direct path)")
print(f"echo delay       : +{echo} samples")
print(f"distance         : {d:.3f} m")
print(f"echo quality     : {echo_q:.1f}x median   (>5 = believable)")

np.save("corr.npy", env)
