"""
Generates every figure for the report into results/.

    python notebooks/make_figures.py

This is display code, so it lives outside src/ and does all its own plotting.
It imports the same functions the tests and the Streamlit app use, which is the
point of the architecture rule: no simulation maths is duplicated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acoustic import analyse_recording, design_chirp
from src.channel import add_noise, simulate_channel, simulate_multi_object_channel
from src.evaluate import run_bandwidth_resolution_sweep, run_snr_sweep
from src.pulse import generate_pulse, pulse_bandwidth_hz
from src.receiver import estimate_distance, matched_filter

FS = 48000.0
SPEED_MPS = 343.0
OUT = Path(__file__).resolve().parent.parent / "results"

# Reference palette, first three categorical slots, used unchanged.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 150,
    "text.color": INK, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8.5,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/{name}")


def fig_pulse_types():
    """The three transmit pulses, and why 'sharpness' is a spectral property."""
    configs = [
        ("rect", {"pulse_type": "rect", "duration_s": 0.002, "freq_hz": 4000.0}, BLUE),
        ("gaussian", {"pulse_type": "gaussian", "duration_s": 0.002, "freq_hz": 4000.0}, ORANGE),
        ("chirp", {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0,
                   "bandwidth_hz": 4000.0}, AQUA),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.6), height_ratios=[1, 1.2])

    for ax, (name, cfg, colour) in zip(axes[0], configs):
        pulse = generate_pulse(fs=FS, **cfg)
        t_ms = np.arange(pulse.size) / FS * 1000
        ax.plot(t_ms, pulse, color=colour)
        ax.set_title(name, color=INK)
        ax.set_xlabel("time (ms)")
    axes[0][0].set_ylabel("amplitude")

    gs = axes[1][0].get_gridspec()
    for ax in axes[1]:
        ax.remove()
    spectrum_ax = fig.add_subplot(gs[1, :])
    for name, cfg, colour in configs:
        pulse = generate_pulse(fs=FS, **cfg)
        n_fft = 8192
        power = np.abs(np.fft.rfft(pulse, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, 1 / FS)
        bandwidth = pulse_bandwidth_hz(pulse, FS)
        keep = freqs < 12000
        spectrum_ax.plot(freqs[keep] / 1000, 10 * np.log10(power[keep] / power.max() + 1e-12),
                         color=colour, label=f"{name} — {bandwidth:.0f} Hz")
    spectrum_ax.axhline(-3, color=GRID, linestyle="--", linewidth=1.2)
    spectrum_ax.annotate("−3 dB", xy=(0.15, -1.6), color=INK_SOFT, fontsize=8, ha="left")
    spectrum_ax.set_ylim(-45, 3)
    spectrum_ax.set_xlabel("frequency (kHz)")
    spectrum_ax.set_ylabel("power (dB, peak-relative)")
    spectrum_ax.set_title("Spectra — a wider −3 dB lobe is a 'sharper' pulse", color=INK)
    spectrum_ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig1_pulse_types.png")


def fig_pipeline():
    """One pass of the whole pipeline: send, receive in noise, recover."""
    pulse = generate_pulse("gaussian", 0.002, FS, freq_hz=4000.0)
    true_distance_m = 5.0
    clean = simulate_channel(pulse, true_distance_m, FS, speed_mps=SPEED_MPS,
                             buffer_duration_s=0.06)
    noisy = add_noise(clean, snr_db=0.0, seed=3)
    estimate_m, correlation = estimate_distance(noisy, pulse, FS, speed_mps=SPEED_MPS)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 6.0))

    ax1.plot(np.arange(pulse.size) / FS * 1000, pulse, color=BLUE)
    ax1.set_title("1 — transmitted pulse (gaussian, 2 ms, 4 kHz)", color=INK)
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("amplitude")

    ax2.plot(np.arange(noisy.size) / FS * 1000, noisy, color=INK_SOFT, linewidth=0.7)
    echo_ms = 2 * true_distance_m / SPEED_MPS * 1000
    ax2.axvspan(echo_ms, echo_ms + 2.0, color=BLUE, alpha=0.18)
    ax2.annotate("the echo is in here", xy=(echo_ms + 1, ax2.get_ylim()[1] * 0.75),
                 xytext=(echo_ms + 9, ax2.get_ylim()[1] * 0.78), color=INK_SOFT, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=INK_SOFT, linewidth=1))
    ax2.set_title("2 — what the microphone hears at 0 dB SNR (echo invisible by eye)", color=INK)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("amplitude")

    distances_m = (np.arange(correlation.size) - (pulse.size - 1)) / FS * SPEED_MPS / 2
    keep = (distances_m >= 0) & (distances_m <= 9)
    ax3.plot(distances_m[keep], correlation[keep], color=BLUE)
    ax3.axvline(true_distance_m, color=GRID, linestyle="--", linewidth=1.4)
    ax3.plot([estimate_m], [correlation.max()], "o", color=ORANGE, markersize=7,
             markeredgecolor=SURFACE, markeredgewidth=2)
    ax3.annotate(f"detected {estimate_m:.3f} m\n(true {true_distance_m:.3f} m)",
                 xy=(estimate_m, correlation.max()), xytext=(estimate_m + 1.1, correlation.max() * 0.82),
                 color=INK, fontsize=8.5)
    ax3.set_title("3 — matched filter output: the echo is unmistakable", color=INK)
    ax3.set_xlabel("distance (m)")
    ax3.set_ylabel("correlation")

    fig.tight_layout()
    save(fig, "fig2_pipeline.png")


def fig_rmse_vs_snr():
    """Noise robustness, and the threshold effect."""
    snr_values = [float(v) for v in range(-12, 21, 2)]
    plain = run_snr_sweep(5.0, snr_values, FS, trials_per_snr=40, bandpass=False)
    filtered = run_snr_sweep(5.0, snr_values, FS, trials_per_snr=40, bandpass=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    quantisation_m = SPEED_MPS / (4 * FS)
    # Drawn thick-under-thin so the two curves stay distinguishable where they
    # coincide -- which is nearly everywhere, and is itself the result.
    ax.semilogy(plain["snr_db"], plain["rmse_m"], "-", color=BLUE, linewidth=4.5,
                alpha=0.45, solid_capstyle="round")
    ax.semilogy(plain["snr_db"], plain["rmse_m"], "o", color=BLUE, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="matched filter")
    ax.semilogy(filtered["snr_db"], filtered["rmse_m"], "--", color=ORANGE, linewidth=1.8)
    ax.semilogy(filtered["snr_db"], filtered["rmse_m"], "s", color=ORANGE, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label="band-pass, then matched filter")
    ax.axhline(quantisation_m, color=GRID, linestyle=":", linewidth=1.4)
    ax.annotate(f"half-sample quantisation limit ≈ {quantisation_m*1000:.1f} mm",
                xy=(-11.5, quantisation_m * 1.25), color=INK_SOFT, fontsize=8)
    ax.annotate("the two curves coincide:\npre-filtering buys nothing against\nwhite noise, as theory predicts",
                xy=(-6, 1.8), xytext=(1.0, 0.35), color=INK_SOFT, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=INK_SOFT, linewidth=1,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("distance RMSE (m)")
    ax.set_title("Ranging error against noise — flat, then a sharp knee below 0 dB", color=INK)
    ax.legend(loc="upper right")
    save(fig, "fig3_rmse_vs_snr.png")


def fig_resolution_vs_bandwidth():
    """The bandwidth study: measured resolution against c / 2B."""
    gaussians = [{"pulse_type": "gaussian", "duration_s": d, "freq_hz": 4000.0}
                 for d in (0.008, 0.006, 0.004, 0.002, 0.001)]
    chirps = [{"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 8000.0,
               "bandwidth_hz": b} for b in (2000.0, 4000.0, 6000.0)]
    result = run_bandwidth_resolution_sweep(gaussians + chirps, FS, trials=16)

    bandwidth = np.array(result["bandwidth_hz"])
    measured = np.array([np.nan if v is None else v for v in result["resolved_m"]])
    n_gauss = len(gaussians)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    grid_hz = np.logspace(np.log10(bandwidth.min() * 0.8), np.log10(bandwidth.max() * 1.25), 100)
    ax.loglog(grid_hz, SPEED_MPS / (2 * grid_hz) * 100, color=GRID, linestyle="--",
              linewidth=1.6)
    label_hz = grid_hz[30]
    ax.annotate("theory  c / 2B", xy=(label_hz, SPEED_MPS / (2 * label_hz) * 100),
                xytext=(label_hz * 1.05, SPEED_MPS / (2 * label_hz) * 100 * 1.18),
                color=INK_SOFT, fontsize=8.5, rotation=-31, rotation_mode="anchor")
    ax.loglog(bandwidth[:n_gauss], measured[:n_gauss] * 100, "o", color=BLUE,
              markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
              label="gaussian — duration varied")
    ax.loglog(bandwidth[n_gauss:], measured[n_gauss:] * 100, "^", color=ORANGE,
              markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
              label="chirp — duration fixed at 2 ms, bandwidth varied")
    ax.set_xticks([200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["200", "500", "1000", "2000", "5000"])
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("measured −3 dB bandwidth (Hz)")
    ax.set_ylabel("closest separation still resolved (cm)")
    ax.set_title("Range resolution improves with bandwidth, not with pulse length", color=INK)
    ax.legend(loc="lower left")
    save(fig, "fig4_resolution_vs_bandwidth.png")
    return result


def fig_multipath():
    """Two targets: resolved, then merged."""
    pulse = generate_pulse("gaussian", 0.002, FS, freq_hz=4000.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

    for ax, separation_cm, colour in ((axes[0], 40, BLUE), (axes[1], 8, ORANGE)):
        targets = [5.0, 5.0 + separation_cm / 100]
        received = simulate_multi_object_channel(pulse, targets, FS, speed_mps=SPEED_MPS,
                                                 buffer_duration_s=0.08)
        envelope = np.abs(hilbert(matched_filter(received, pulse)))
        distances_m = (np.arange(envelope.size) - (pulse.size - 1)) / FS * SPEED_MPS / 2
        keep = (distances_m > 4.5) & (distances_m < 6.0)
        ax.plot(distances_m[keep], envelope[keep], color=colour)
        for target in targets:
            ax.axvline(target, color=GRID, linestyle="--", linewidth=1.2)
        verdict = "two peaks — resolved" if separation_cm == 40 else "one peak — merged"
        ax.set_title(f"{separation_cm} cm apart: {verdict}", color=INK)
        ax.set_xlabel("distance (m)")
    axes[0].set_ylabel("correlation envelope")
    fig.suptitle("Two targets, same pulse — the limit is about 15 cm here", color=INK, y=1.02)
    save(fig, "fig5_multipath.png")


def fig_acoustic():
    """Real speaker and microphone, two independent detections."""
    here = Path(__file__).resolve().parent.parent / "physical_sonar_check"
    chirp = design_chirp(2000.0, 8000.0, 0.010, FS)
    frame_samples = int(round((0.010 + 0.250) * FS))

    panels = []
    wall = analyse_recording(np.load(here / "rx.npy"), chirp, frame_samples, FS,
                             baseline_envelope=np.load(here / "baseline.npy"),
                             speed_mps=SPEED_MPS, min_range_m=1.0, max_range_m=5.0)
    panels.append(("wall", wall, np.load(here / "baseline.npy"), 0.8, 4.5, BLUE))

    hand_rx = here / "hand_rx.npy"
    if hand_rx.exists():
        hand = analyse_recording(np.load(hand_rx), chirp, frame_samples, FS,
                                 baseline_envelope=np.load(here / "hand_baseline.npy"),
                                 speed_mps=SPEED_MPS, min_range_m=0.10, max_range_m=1.50)
        panels.append(("hand held above the keyboard", hand,
                       np.load(here / "hand_baseline.npy"), 0.1, 1.2, ORANGE))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (label, result, baseline, lo, hi, colour) in zip(axes, panels):
        excess = result["excess"]
        distances_m = SPEED_MPS * np.arange(excess.size) / FS / 2
        keep = (distances_m >= lo) & (distances_m <= hi)
        ax.plot(distances_m[keep], baseline[keep], color=GRID, linewidth=1.4,
                label="empty room (clutter)")
        ax.plot(distances_m[keep], excess[keep], color=colour,
                label="after clutter removal")
        ax.plot([result["distance_m"]], [excess[result["peak_index"]]], "o",
                color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{result['distance_m']:.3f} m\n{result['quality_sigma']:.1f}σ",
                    xy=(result["distance_m"], excess[result["peak_index"]]),
                    xytext=(result["distance_m"] + (hi - lo) * 0.08,
                            excess[result["peak_index"]] * 0.72),
                    color=INK, fontsize=8.5)
        ax.set_title(f"Real recording — {label}", color=INK)
        ax.set_xlabel("distance (m)")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("correlation (direct path = 1.0)")
    fig.tight_layout()
    save(fig, "fig6_acoustic.png")


def main():
    OUT.mkdir(exist_ok=True)
    print("generating report figures ...")
    fig_pulse_types()
    fig_pipeline()
    fig_rmse_vs_snr()
    resolution = fig_resolution_vs_bandwidth()
    fig_multipath()
    fig_acoustic()

    print("\nresolution sweep numbers (for the report text):")
    for label, bw, measured, theory in zip(resolution["labels"], resolution["bandwidth_hz"],
                                           resolution["resolved_m"], resolution["theory_m"]):
        if measured is not None:
            print(f"  {label:<28} B={bw:7.0f} Hz   measured {measured*100:5.1f} cm   "
                  f"theory {theory*100:5.1f} cm   ratio {measured/theory:.2f}")


if __name__ == "__main__":
    main()
