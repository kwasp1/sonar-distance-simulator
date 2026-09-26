# Physical Sonar Hardware Testing (Feature 8)

This directory contains the physical acoustic hardware capture scripts and archived ground-truth benchmark recordings for **Feature 8**.

For the complete, comprehensive guide explaining the theory, setup, web interface, CLI execution, and troubleshooting, see:
👉 **[`../PHYSICAL_TEST_GUIDE.md`](../PHYSICAL_TEST_GUIDE.md)**

---

## Quick Command Reference

```powershell
# 1. Measure speaker + microphone frequency passband (2–22 kHz sweep):
python band_check.py

# 2. Record the open-space baseline (point laptop into empty room with >= 3.5 m clearance):
python record_acoustic.py baseline --run live

# 3. Measure distance to physical target (point laptop directly at wall or door):
python record_acoustic.py --run live --min-range 1.0 --max-range 5.0
```

---

## Directory Contents

- **`record_acoustic.py`**: Standalone Python capture script using `sounddevice` to transmit linear chirps through the speaker and record microphone returns simultaneously.
- **`band_check.py`**: Frequency response calibration script testing laptop speaker/microphone passband between 2 kHz and 22 kHz.
- **`rx.npy`**: Archived raw physical recording of a concrete wall at 1.797 m ($1{,}697{,}408$ bytes, 48 kHz).
- **`baseline.npy`**: Archived reference open-space clutter baseline ($12{,}480$ envelope samples).
- **`excess.npy`**: Archived clutter-cancelled excess reflection signal ($12{,}480$ excess samples).
