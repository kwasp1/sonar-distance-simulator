"""
sonar_detect.py -- HC-SR04-style near-field detector. NOT project code.

High band (8-20 kHz) for directivity, resolution and shorter ringdown.
MTI clutter cancellation, then FIRST threshold crossing rather than argmax --
same logic an ultrasonic rangefinder uses: nearest object wins, silence when
nothing is there.

    python3 sonar_detect.py

Run band_check.py first and set F1 below where your speaker rolls off.
Needs: pip install matplotlib
"""
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

FS      = 48_000
C       = 343.0
F0, F1  = 15_000, 21_000    # <-- narrow this to whatever band_check.py showed
T_CHIRP = 0.005
GAP     = 0.060
N_REP   = 4
AMP     = 0.70
MIN_M   = 0.10             # HF ringdown is shorter, so we can look closer
MAX_M   = 2.00
ALPHA   = 0.90             # clutter memory
K       = 8.0              # threshold, in multiples of the noise floor

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
    env = np.roll(env, -int(np.argmax(env)))
    return env / env[0]


lo, hi = int(2*MIN_M/C*FS), int(2*MAX_M/C*FS)
d = C * np.arange(lo, hi) / FS / 2

plt.ion()
fig, ax = plt.subplots(figsize=(9, 4))
line,  = ax.plot(d, np.zeros(hi-lo), lw=1)
thr_ln = ax.axhline(0, color="r", ls="--", lw=0.8)
mark,  = ax.plot([], [], "ro", ms=9)
ax.set_xlabel("distance (m)"); ax.set_ylabel("MTI response")
ax.grid(alpha=0.3)

clutter = None
print("scanning... Ctrl-C to stop")
try:
    while True:
        env = sweep()
        if clutter is None:
            clutter = env.copy(); continue

        mti = np.abs(env - clutter)[lo:hi]
        clutter = ALPHA*clutter + (1-ALPHA)*env

        floor = np.median(mti)
        thr = K * floor
        over = np.flatnonzero(mti > thr)          # FIRST crossing, not argmax
        line.set_ydata(mti); thr_ln.set_ydata([thr, thr])
        ax.set_ylim(0, max(mti.max()*1.3, thr*1.5, 1e-6))

        if over.size:
            i = over[0]
            rng = d[i]
            mark.set_data([rng], [mti[i]])
            ax.set_title(f"TARGET  {rng*100:.0f} cm")
            print(f"\rTARGET {rng*100:6.1f} cm   ", end="")
        else:
            mark.set_data([], [])
            ax.set_title("clear")
            print("\rclear            ", end="")
        plt.pause(0.001)
except KeyboardInterrupt:
    print("\nstopped")
