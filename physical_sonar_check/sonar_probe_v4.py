"""
sonar_probe.py -- throwaway hardware feasibility probe. NOT project code.
 
v4: background subtraction. The speaker/chassis ringdown dominates out to
~0.9 m and is identical every run, so it is measured once against open
space and removed from the wall measurement (radar clutter removal).
 
    python3 sonar_probe.py baseline    # aim into open space, >4 m clear
    python3 sonar_probe.py             # aim at wall, ~2 m
 
System output volume ~70%. Do not change it between the two runs.
"""
 
import sys, os
import numpy as np
import sounddevice as sd
 
FS      = 48_000
C       = 343.0
F0, F1  = 2_000, 8_000
T_CHIRP = 0.010
GAP     = 0.250
N_REP   = 16
MIN_M   = 1.00             # was 0.40 -- inside the ringdown
MAX_M   = 5.00
AMP     = 0.50
 
baseline_mode = len(sys.argv) > 1 and sys.argv[1] == "baseline"
 
nc = int(T_CHIRP * FS)
nf = int((T_CHIRP + GAP) * FS)
 
t = np.arange(nc) / FS
chirp = np.sin(2 * np.pi * (F0 * t + (F1 - F0) / (2 * T_CHIRP) * t**2))
chirp *= np.hanning(nc)
 
frame = np.zeros(nf)
frame[:nc] = AMP * chirp
tx = np.concatenate([np.tile(frame, N_REP), np.zeros(nf)])
stereo = np.zeros((tx.size, 2))
stereo[:, 1] = tx
 
rx = sd.playrec(stereo, samplerate=FS, channels=1, blocking=True)[:, 0]
 
corr = np.correlate(rx, chirp, mode="full")
n = (corr.size // nf) * nf
fold = corr[:n].reshape(-1, nf).sum(0) / (n // nf)
env = np.abs(fold)
env = np.roll(env, -int(np.argmax(env)))
env = env / env[0]                             # normalise to direct peak
 
print(f"input peak level : {np.max(np.abs(rx)):.3f}")
print(f"DIRECT quality   : {env[0]/np.median(env):.0f}x median")
 
if baseline_mode:
    np.save("baseline.npy", env)
    print("baseline saved. now aim at the wall and run without arguments.")
    sys.exit()
 
if not os.path.exists("baseline.npy"):
    sys.exit("no baseline.npy -- run 'python3 sonar_probe.py baseline' first")
 
base = np.load("baseline.npy")
excess = env - base                            # clutter removed
 
lo, hi = int(2*MIN_M/C*FS), int(2*MAX_M/C*FS)
k = lo + int(np.argmax(excess[lo:hi]))
d = C * k / FS / 2
q = excess[k] / np.std(excess[lo:hi])
 
print(f"echo delay       : +{k} samples")
print(f"distance         : {d:.3f} m")
print(f"echo quality     : {q:.1f} sigma above clutter-removed floor  (>6 = real)")
 
np.save("excess.npy", excess)
np.save("rx.npy", rx)