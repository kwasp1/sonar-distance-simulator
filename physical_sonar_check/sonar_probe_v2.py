"""
sonar_probe.py -- throwaway hardware feasibility probe. NOT project code.

v2: tolerates arbitrary playrec latency by locating the direct-path peak
in the full recording and aligning every repeat to it.

    python3 sonar_probe.py

Raise system output volume to ~70% first.
"""

import numpy as np
import sounddevice as sd

FS      = 48_000
C       = 343.0
F0, F1  = 2_000, 8_000
T_CHIRP = 0.010
GAP     = 0.250            # was 0.100 -- must exceed capture latency + echo window
N_REP   = 16
MIN_M   = 0.40
MAX_M   = 5.00
AMP     = 0.50             # was 0.35

n_chirp = int(T_CHIRP * FS)
n_frame = int((T_CHIRP + GAP) * FS)
n_win   = int(2 * MAX_M / C * FS) + 200      # echo search span after direct

# --- transmit ------------------------------------------------------------
t = np.arange(n_chirp) / FS
chirp = np.sin(2 * np.pi * (F0 * t + (F1 - F0) / (2 * T_CHIRP) * t**2))
chirp *= np.hanning(n_chirp)

frame = np.zeros(n_frame)
frame[:n_chirp] = AMP * chirp
tx = np.concatenate([np.tile(frame, N_REP), np.zeros(n_frame)])   # tail

stereo = np.zeros((tx.size, 2))
stereo[:, 1] = tx                             # RIGHT channel only

rx = sd.playrec(stereo, samplerate=FS, channels=1, blocking=True)[:, 0]

# --- one correlation over the whole recording ----------------------------
corr = np.correlate(rx, chirp, mode="full")

# first direct-path peak: search a window guaranteed to hold exactly one chirp
p0 = int(np.argmax(np.abs(corr[:n_frame])))
latency = p0 - (n_chirp - 1)

# --- fold N_REP repeats onto each other, aligned to p0 -------------------
acc = np.zeros(n_win)
used = 0
for k in range(N_REP):
    s = p0 + k * n_frame
    if s + n_win <= corr.size:
        acc += corr[s:s + n_win]
        used += 1
acc /= used
env = np.abs(acc)

# --- two-peak logic (direct is now at index 0) --------------------------
lo = int(2 * MIN_M / C * FS)
echo = lo + int(np.argmax(env[lo:]))
d = C * echo / FS / 2
ratio = env[echo] / np.median(env[lo:])

print(f"input peak level : {np.max(np.abs(rx)):.3f}   (want 0.05-0.9)")
print(f"capture latency  : {latency} samples = {latency/FS*1000:.1f} ms")
print(f"repeats averaged : {used}/{N_REP}")
print(f"echo delay       : +{echo} samples")
print(f"distance         : {d:.3f} m")
print(f"peak / median    : {ratio:.1f}x   (>5 = believable, <3 = noise)")

np.save("corr.npy", env)
print("saved corr.npy -- plot it if the number looks wrong")

import numpy as np, matplotlib.pyplot as plt
env = np.load("corr.npy")
d = 343 * np.arange(env.size) / 48000 / 2
plt.plot(d, env); plt.xlabel("distance (m)"); plt.grid(); plt.show()