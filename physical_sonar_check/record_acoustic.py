"""
Feature 8 capture: chirp at something real and measure how far away it is.

This is the only file in the project that touches the speaker and microphone.
All of the maths lives in src/acoustic.py, so this file just records audio,
hands the array over, and prints what comes back.

    python physical_sonar_check/record_acoustic.py baseline   # face open space
    python physical_sonar_check/record_acoustic.py            # face the wall

Captures are named after the run (default "live"), so the archived v4 files
-- baseline.npy, rx.npy and excess.npy -- are never overwritten. The test
suite checks against those, so keep them as they are:

    python physical_sonar_check/record_acoustic.py baseline --run desk

Run the baseline first, pointing at least four metres of clear air. The
laptop's own speaker ringing is identical every run, so the baseline captures
it once and it gets subtracted from every measurement afterwards. Do not
change the system volume between the two runs or the subtraction will not
cancel.

Needs sounddevice, which is not in requirements.txt because it pulls in
PortAudio and nothing else in the project wants it:

    pip install sounddevice
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acoustic import analyse_recording, design_chirp

FS = 48_000.0
SPEED_MPS = 343.0
CHIRP_START_HZ, CHIRP_END_HZ = 2_000.0, 8_000.0
CHIRP_DURATION_S = 0.010
GAP_S = 0.250
REPEATS = 16
AMPLITUDE = 0.5
MIN_RANGE_M, MAX_RANGE_M = 1.0, 5.0

HERE = Path(__file__).resolve().parent
FRAME_SAMPLES = int(round((CHIRP_DURATION_S + GAP_S) * FS))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="measure",
                        choices=["baseline", "measure"],
                        help="record the empty-room reference, or measure against it")
    parser.add_argument("--run", default="live",
                        help="names the saved files, so separate sessions do not collide")
    parser.add_argument("--min-range", type=float, default=MIN_RANGE_M,
                        help="ignore returns closer than this, in metres")
    parser.add_argument("--max-range", type=float, default=MAX_RANGE_M,
                        help="ignore returns further than this, in metres")
    return parser.parse_args()


def record(chirp):
    import sounddevice as sd

    frame = np.zeros(FRAME_SAMPLES)
    frame[: chirp.size] = AMPLITUDE * chirp
    transmit = np.concatenate([np.tile(frame, REPEATS), np.zeros(FRAME_SAMPLES)])

    stereo = np.zeros((transmit.size, 2))
    stereo[:, 1] = transmit
    return sd.playrec(stereo, samplerate=int(FS), channels=1, blocking=True)[:, 0]


def main():
    args = parse_args()
    baseline_mode = args.mode == "baseline"
    baseline_path = HERE / f"{args.run}_baseline.npy"
    recording_path = HERE / f"{args.run}_rx.npy"
    excess_path = HERE / f"{args.run}_excess.npy"

    chirp = design_chirp(CHIRP_START_HZ, CHIRP_END_HZ, CHIRP_DURATION_S, FS)

    if not baseline_mode and not baseline_path.exists():
        sys.exit(f"no {baseline_path.name} yet -- run with 'baseline' first")

    print(f"transmitting {REPEATS} chirps, "
          f"{CHIRP_START_HZ/1000:.0f}-{CHIRP_END_HZ/1000:.0f} kHz, "
          f"{(REPEATS + 1) * (CHIRP_DURATION_S + GAP_S):.1f} s total ...")
    recording = record(chirp)
    print(f"input peak level : {np.max(np.abs(recording)):.3f}")

    baseline = None if baseline_mode else np.load(baseline_path)
    result = analyse_recording(
        recording, chirp, FRAME_SAMPLES, FS,
        baseline_envelope=baseline, speed_mps=SPEED_MPS,
        min_range_m=args.min_range, max_range_m=args.max_range,
    )

    if baseline_mode:
        np.save(baseline_path, result["envelope"])
        print(f"baseline saved to {baseline_path.name}. "
              f"put the target in place and run again without 'baseline'.")
        return

    np.save(recording_path, recording)
    np.save(excess_path, result["excess"])
    print(f"searched           : {args.min_range:.2f} - {args.max_range:.2f} m")
    print(f"distance           : {result['distance_m']:.3f} m")
    print(f"echo quality       : {result['quality_sigma']:.1f} sigma "
          f"({'real detection' if result['quality_sigma'] > 6 else 'too weak to trust'})")


if __name__ == "__main__":
    main()
