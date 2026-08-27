"""
sonar_probe.py -- throwaway hardware feasibility probe. NOT project code.

Plays a chirp out the RIGHT channel while recording the built-in mic,
finds the direct-path peak and the wall echo peak in the matched-filter
output, and reports the distance from the DIFFERENCE between them
(so all unknown playback/capture latency cancels).

    pip install sounddevice numpy
    python sonar_probe.py

Point the laptop at a bare flat wall ~1.5 m away. Clear the desk in front.
"""

import numpy as np
import sounddevice as sd

FS      = 48_000
C       = 343.0
F0, F1  = 2_000, 8_000     # chirp band
T_CHIRP = 0.010            # 10 ms
GAP     = 0.100            # silence after each chirp
N_REP   = 16               # coherent averages
MIN_M   = 0.40             # ignore echoes closer than this (direct ringdown)
MAX_M   = 5.00
AMP     = 0.35             # keep modest -- distortion breaks the match

n_chirp = int(T_CHIRP * FS)
n_frame = int((T_CHIRP + GAP) * FS)

# --- transmit ------------------------------------------------------------
t = np.arange(n_chirp) / FS
chirp = np.sin(2 * np.pi * (F0 * t + (F1 - F0) / (2 * T_CHIRP) * t**2))
chirp *= np.hanning(n_chirp)              # taper -> lower correlation sidelobes

frame = np.zeros(n_frame)
frame[:n_chirp] = AMP * chirp
tx = np.tile(frame, N_REP)

stereo = np.zeros((tx.size, 2))
stereo[:, 1] = tx                          # RIGHT channel only

rx = sd.playrec(stereo, samplerate=FS, channels=1, blocking=True)[:, 0]

# --- matched filter, averaged over repeats -------------------------------
corr = np.zeros(n_frame + n_chirp - 1)
for k in range(N_REP):
    seg = rx[k * n_frame:(k + 1) * n_frame]
    corr += np.correlate(seg, chirp, mode="full")
corr /= N_REP
env = np.abs(corr)

# --- two-peak logic ------------------------------------------------------
direct = int(np.argmax(env))
lo = direct + int(2 * MIN_M / C * FS)
hi = min(direct + int(2 * MAX_M / C * FS), env.size)
echo = lo + int(np.argmax(env[lo:hi]))

d = C * (echo - direct) / FS / 2
ratio = env[echo] / np.median(env[lo:hi])

print(f"input peak level : {np.max(np.abs(rx)):.3f}   (want 0.05-0.9)")
print(f"direct idx       : {direct}")
print(f"echo idx         : {echo}   (+{echo - direct} samples)")
print(f"distance         : {d:.3f} m")
print(f"peak / median    : {ratio:.1f}x   (>5 = believable, <3 = noise)")

np.save("corr.npy", env)
print("saved corr.npy -- plot it if the number looks wrong")

import numpy as np, matplotlib.pyplot as plt
env = np.load("corr.npy")
d = 343 * np.arange(env.size) / 48000 / 2
plt.plot(d, env); plt.xlabel("distance (m)"); plt.grid(); plt.show()