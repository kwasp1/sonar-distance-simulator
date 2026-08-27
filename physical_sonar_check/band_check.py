"""
band_check.py -- measure your MacBook's actual speaker+mic response.

Plays a slow 2-22 kHz sweep, records it, and divides received spectrum by
transmitted spectrum. That ratio IS the hardware response -- everything the
speaker, air path and mic do to your signal. Run this FIRST so you pick a
chirp band the hardware can actually produce instead of guessing.

    python3 band_check.py

Point the laptop into open space. System volume ~70%.
"""
import numpy as np
import sounddevice as sd

FS = 48_000
T, F0, F1 = 0.5, 2_000, 22_000

n = int(T * FS)
t = np.arange(n) / FS
sweep = np.sin(2*np.pi*(F0*t + (F1-F0)/(2*T)*t**2))
fade = int(0.005 * FS)                       # short fades, keeps spectrum flat
sweep[:fade]  *= np.linspace(0, 1, fade)
sweep[-fade:] *= np.linspace(1, 0, fade)

tx = np.concatenate([np.zeros(FS//10), 0.5*sweep, np.zeros(FS//5)])
stereo = np.zeros((tx.size, 2)); stereo[:, 1] = tx

rx = sd.playrec(stereo, samplerate=FS, channels=1, blocking=True)[:, 0]

f = np.fft.rfftfreq(tx.size, 1/FS)
resp = np.abs(np.fft.rfft(rx)) / (np.abs(np.fft.rfft(tx)) + 1e-12)

print(f"input peak: {np.abs(rx).max():.3f}\n")
print(" freq (kHz)   response (dB rel. peak)")
rows = []
for lo in range(2000, 22000, 1000):
    m = (f >= lo) & (f < lo + 1000)
    rows.append((lo/1000, 20*np.log10(np.median(resp[m]) + 1e-12)))
peak = max(r[1] for r in rows)
for khz, db in rows:
    rel = db - peak
    bar = "#" * max(0, int(40 + rel))
    print(f"  {khz:4.0f}-{khz+1:<4.0f}  {rel:7.1f}  {bar}")
print("\nusable band = where response stays within about -12 dB of peak")
