"""
sonar_live.py -- live moving-target display. NOT project code.

Continuously chirps and plots what CHANGED since the last few sweeps.
Static clutter (speaker ringdown, ceiling, desk) is learned into a running
average and cancelled; anything that moves survives. This is MTI -- moving
target indication -- the same trick weather and air-traffic radars use to
see aircraft through ground clutter.

    python3 sonar_live.py

Then hold your hand ~30 cm above the speaker grille and move it up and down.
Ctrl-C to quit. Needs: pip install matplotlib
"""
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

FS      = 48_000
C       = 343.0
F0, F1  = 2_000, 8_000
T_CHIRP = 0.010
GAP     = 0.100
N_REP   = 4                # fewer repeats -> faster refresh
AMP     = 0.50
MIN_M   = 0.15             # MTI removes the ringdown, so we can look closer
MAX_M   = 3.00
ALPHA   = 0.85             # clutter memory: higher = slower to forget

nc = int(T_CHIRP * FS)
nf = int((T_CHIRP + GAP) * FS)
t  = np.arange(nc) / FS
chirp = np.sin(2*np.pi*(F0*t + (F1-F0)/(2*T_CHIRP)*t**2)) * np.hanning(nc)

frame = np.zeros(nf); frame[:nc] = AMP * chirp
tx = np.concatenate([np.tile(frame, N_REP), np.zeros(nf)])
stereo = np.zeros((tx.size, 2)); stereo[:, 1] = tx


def sweep():
    rx = sd.playrec(stereo, samplerate=FS, channels=1, blocking=True)[:, 0]
    corr = np.correlate(rx, chirp, mode="full")
    n = (corr.size // nf) * nf
    fold = corr[:n].reshape(-1, nf).sum(0) / (n // nf)
    env = np.abs(fold)
    env = np.roll(env, -int(np.argmax(env)))     # direct peak -> index 0
    return env / env[0]


lo, hi = int(2*MIN_M/C*FS), int(2*MAX_M/C*FS)
d = C * np.arange(lo, hi) / FS / 2

plt.ion()
fig, ax = plt.subplots(figsize=(9, 4))
line, = ax.plot(d, np.zeros(hi - lo))
ax.set_xlabel("distance (m)"); ax.set_ylabel("MTI response")
ax.set_title("move something above the speaker grille")
ax.grid(alpha=0.3)

clutter = None
print("listening... Ctrl-C to stop")
try:
    while True:
        env = sweep()
        if clutter is None:
            clutter = env.copy()
            continue
        mti = np.abs(env - clutter)[lo:hi]
        clutter = ALPHA * clutter + (1 - ALPHA) * env
        line.set_ydata(mti)
        ax.set_ylim(0, max(mti.max() * 1.3, 1e-5))
        pk = lo + int(np.argmax(np.abs(env - clutter)[lo:hi]))
        print(f"\rstrongest mover: {C*pk/FS/2:5.2f} m   ", end="")
        plt.pause(0.001)
except KeyboardInterrupt:
    print("\nstopped")