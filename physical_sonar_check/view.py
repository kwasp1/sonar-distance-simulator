"""
view_npy.py -- plot any trace the probe saves.

    python3 view_npy.py              # defaults to corr.npy
    python3 view_npy.py excess.npy
    python3 view_npy.py baseline.npy

Needs: pip install matplotlib
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

FS, C = 48_000, 343.0
path = sys.argv[1] if len(sys.argv) > 1 else "rx.npy"
y = np.load(path)
d = C * np.arange(y.size) / FS / 2          # one-way distance in metres

signed = y.min() < 0                         # excess.npy is signed, others aren't

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(d, y, lw=0.8)
ax.set_xlabel("distance to reflector (m)")
ax.set_xlim(0, 5)
ax.grid(alpha=0.3)

if signed:
    ax.axhline(0, color="k", lw=0.8)
    s = np.std(y[int(2*1.0/C*FS):int(2*5.0/C*FS)])
    ax.axhline( 6*s, color="r", ls="--", lw=0.8, label="+6 sigma")
    ax.axhline(-6*s, color="r", ls="--", lw=0.8)
    ax.set_ylabel("excess (clutter removed)")
    ax.legend()
else:
    ax.set_yscale("log")
    ax.set_ylabel("correlation (normalised to direct peak)")
    ax.axvspan(0, 1.0, color="orange", alpha=0.15, label="ringdown - unusable")
    ax.legend()

ax.set_title(path)
plt.tight_layout()
plt.show()