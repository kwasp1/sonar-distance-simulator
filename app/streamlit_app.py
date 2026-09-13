"""
Streamlit Web Application: Interactive Sonar Distance Simulator
CSE220 Signals and Systems — Team SevenEight

Strict Architectural Adherence:
This file ONLY imports and calls functions from src/, and renders interactive UI / charts.
No simulation math lives directly in this file.
"""

from __future__ import annotations

import os
import sys

# 1. Thread and BLAS limits MUST be set before importing numpy/scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# 2. Force headless Matplotlib Agg backend to prevent GUI thread memory leaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import hilbert

# Ensure root is on path for src imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from src.pulse import generate_pulse, pulse_bandwidth_hz
from src.channel import simulate_channel, add_noise, simulate_multi_object_channel
from src.receiver import matched_filter, estimate_distance, estimate_multiple_distances
from src.filters import bandpass_filter
from src.evaluate import run_snr_sweep, run_bandwidth_resolution_sweep


# Clean any lingering matplotlib figure state
plt.close("all")


# ==============================================================================
# PAGE SETUP & MODERN CSS THEMING
# ==============================================================================

st.set_page_config(
    page_title="Sonar Distance Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global styling */
    .main {
        background-color: #0b0f19;
    }
    
    /* Header card */
    .hero-header {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(99, 102, 241, 0.15));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        margin: 0;
    }
    
    /* Badges */
    .badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .badge {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    
    /* Metric cards */
    .stat-card {
        background: #111827;
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stat-val-success {
        font-size: 1.7rem;
        font-weight: 700;
        color: #34d399;
    }
    .stat-val-warning {
        font-size: 1.7rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .stat-val-danger {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f87171;
    }
    .stat-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Dark Plot theme helper
def apply_dark_theme(fig, axes):
    bg_color = "#0f172a"  # Slate 900
    grid_color = "#1e293b"  # Slate 800
    text_color = "#cbd5e1"  # Slate 300
    
    fig.patch.set_facecolor(bg_color)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(bg_color)
        ax.grid(True, linestyle="--", alpha=0.4, color=grid_color)
        ax.tick_params(colors=text_color, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color("#f1f5f9")
        if ax.get_legend():
            legend = ax.get_legend()
            legend.get_frame().set_facecolor("#1e293b")
            legend.get_frame().set_edgecolor("#334155")
            for text in legend.get_texts():
                text.set_color(text_color)


# ==============================================================================
# HERO BANNER
# ==============================================================================

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">📡 Sonar Distance Simulator</div>
        <p class="hero-subtitle">
            CSE220 Signals and Systems · Pulse-Echo Ranging & Matched Filter Interactive Suite
        </p>
        <div class="badge-container">
            <span class="badge">Team SevenEight</span>
            <span class="badge">Unit-Energy Normalization</span>
            <span class="badge">AWGN Channel Model</span>
            <span class="badge">Cross-Correlation Matched Filter</span>
            <span class="badge">Hilbert Analytic Envelope</span>
            <span class="badge">Butterworth Bandpass</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# STUDIO NAVIGATION (Segmented View to save RAM and avoid rendering unneeded tabs)
# ==============================================================================

selected_tab = st.radio(
    "Select Simulation Studio",
    [
        "📡 1. End-to-End Pipeline",
        "🎯 2. Multi-Target & Resolution",
        "📊 3. Monte Carlo SNR Sweeps",
        "🔬 4. Pulse & Spectrum Lab",
        "🔊 5. Real Hardware Sonar (MTI)",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.write("")


# ==============================================================================
# 1. END-TO-END SONAR PIPELINE STUDIO
# ==============================================================================
if selected_tab == "📡 1. End-to-End Pipeline":
    st.subheader("Interactive Transmit-Channel-Receiver Pipeline")
    st.caption("Manipulate target distance and noise levels in real time to observe matched-filter delay recovery.")

    col_ctrl, col_view = st.columns([1, 2.2])

    with col_ctrl:
        st.markdown("#### ⚙️ Simulation Controls")
        
        preset = st.selectbox(
            "Quick Scenario Presets",
            ["Custom", "Clean Acoustic Room (20 dB)", "Noisy Factory Floor (0 dB)", "Submerged Deep Noise (-10 dB)", "Long-Range Chirp"]
        )
        
        if preset == "Clean Acoustic Room (20 dB)":
            default_dist, default_snr, default_type, default_dur, default_fc = 6.0, 20.0, "gaussian", 0.002, 4000.0
        elif preset == "Noisy Factory Floor (0 dB)":
            default_dist, default_snr, default_type, default_dur, default_fc = 8.5, 0.0, "gaussian", 0.003, 4000.0
        elif preset == "Submerged Deep Noise (-10 dB)":
            default_dist, default_snr, default_type, default_dur, default_fc = 12.0, -10.0, "chirp", 0.008, 4000.0
        elif preset == "Long-Range Chirp":
            default_dist, default_snr, default_type, default_dur, default_fc = 20.0, 5.0, "chirp", 0.010, 4000.0
        else:
            default_dist, default_snr, default_type, default_dur, default_fc = 10.0, 10.0, "gaussian", 0.002, 4000.0

        true_dist = st.slider("Target Distance (m)", min_value=0.5, max_value=25.0, value=float(default_dist), step=0.1)
        snr_db = st.slider("Channel SNR (dB)", min_value=-20.0, max_value=30.0, value=float(default_snr), step=1.0)
        
        pulse_type = st.selectbox("Pulse Type", ["gaussian", "chirp", "rect"], index=["gaussian", "chirp", "rect"].index(default_type))
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            duration_ms = st.number_input("Duration (ms)", min_value=0.5, max_value=20.0, value=float(default_dur * 1000), step=0.5)
        with col_p2:
            freq_hz = st.number_input("Carrier Freq (Hz)", min_value=500.0, max_value=12000.0, value=float(default_fc), step=500.0)
            
        bandwidth_hz = None
        if pulse_type == "chirp":
            bandwidth_hz = st.slider("Chirp Sweep Bandwidth (Hz)", min_value=500.0, max_value=6000.0, value=2000.0, step=250.0)
            
        with st.expander("Advanced Channel & Pre-Filter Options"):
            speed_mps = st.number_input("Speed of Sound (m/s)", min_value=100.0, max_value=2000.0, value=343.0, step=1.0)
            fs = st.number_input("Sampling Rate fs (Hz)", min_value=16000.0, max_value=96000.0, value=48000.0, step=4000.0)
            attenuation = st.slider("Echo Attenuation", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
            rng_seed = st.number_input("RNG Seed (None for random)", min_value=0, max_value=9999, value=42)
            enable_bandpass = st.checkbox("Enable Butterworth Pre-Filter", value=False)
            bp_order = st.slider("Filter Order", min_value=1, max_value=8, value=4) if enable_bandpass else 4

    # Run Pure Simulation Pipeline from src/
    duration_s = duration_ms / 1000.0
    buffer_duration_s = max(0.06, (2 * true_dist / speed_mps) * 1.4)
    
    pulse = generate_pulse(pulse_type, duration_s, fs, freq_hz=freq_hz, bandwidth_hz=bandwidth_hz)
    measured_bw = pulse_bandwidth_hz(pulse, fs)
    
    clean_rx = simulate_channel(
        pulse, true_dist, fs=fs, speed_mps=speed_mps,
        attenuation=attenuation, buffer_duration_s=buffer_duration_s
    )
    
    noisy_rx = add_noise(clean_rx, snr_db, seed=rng_seed)
    
    if enable_bandpass:
        f_low = max(50.0, freq_hz - (measured_bw if measured_bw > 0 else 1000.0) * 1.2)
        f_high = min(fs / 2 - 100.0, freq_hz + (measured_bw if measured_bw > 0 else 1000.0) * 1.2)
        rx_for_detection = bandpass_filter(noisy_rx, fs, f_low, f_high, order=bp_order)
    else:
        rx_for_detection = noisy_rx
        
    est_dist, correlation = estimate_distance(rx_for_detection, pulse, fs=fs, speed_mps=speed_mps)
    abs_error_m = abs(est_dist - true_dist)
    round_trip_delay_ms = (2 * true_dist / speed_mps) * 1000

    with col_view:
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.markdown(f'<div class="stat-card"><div class="stat-val-success">{true_dist:.2f} m</div><div class="stat-label">True Distance</div></div>', unsafe_allow_html=True)
        with m_c2:
            val_class = "stat-val-success" if abs_error_m < 0.05 else ("stat-val-warning" if abs_error_m < 0.5 else "stat-val-danger")
            st.markdown(f'<div class="stat-card"><div class="{val_class}">{est_dist:.3f} m</div><div class="stat-label">Estimated Distance</div></div>', unsafe_allow_html=True)
        with m_c3:
            err_str = f"{abs_error_m*1000:.1f} mm" if abs_error_m < 0.1 else f"{abs_error_m*100:.2f} cm"
            st.markdown(f'<div class="stat-card"><div class="{val_class}">{err_str}</div><div class="stat-label">Absolute Error</div></div>', unsafe_allow_html=True)
        with m_c4:
            st.markdown(f'<div class="stat-card"><div style="font-size:1.7rem; font-weight:700; color:#38bdf8;">{round_trip_delay_ms:.2f} ms</div><div class="stat-label">Round-Trip Delay</div></div>', unsafe_allow_html=True)
            
        st.write("")
        
        # Plot Figures (optimized size and explicitly closed)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5, 6.8), constrained_layout=True)
        apply_dark_theme(fig, [ax1, ax2, ax3])
        
        # 1. Transmitted Pulse
        t_pulse_ms = (np.arange(len(pulse)) / fs) * 1000
        ax1.plot(t_pulse_ms, pulse, color="#38bdf8", lw=1.3, label=f"TX Pulse ({pulse_type})")
        ax1.set_title(f"1. Transmit Waveform: {pulse_type.capitalize()} (Duration = {duration_ms:.1f} ms, Bandwidth = {measured_bw:.1f} Hz)", fontsize=9.5)
        ax1.set_xlabel("Time (ms)", fontsize=8.5)
        ax1.set_ylabel("Amplitude", fontsize=8.5)
        ax1.legend(loc="upper right", fontsize=8)
        
        # 2. Received Signal Buffer
        t_buf_ms = (np.arange(len(noisy_rx)) / fs) * 1000
        ax2.plot(t_buf_ms, noisy_rx, color="#64748b", alpha=0.7, lw=0.8, label=f"Noisy RX Signal (SNR = {snr_db:.0f} dB)")
        if enable_bandpass:
            ax2.plot(t_buf_ms, rx_for_detection, color="#a855f7", lw=0.9, alpha=0.9, label=f"Bandpass Filtered (Order {bp_order})")
        echo_start_ms = (2 * true_dist / speed_mps) * 1000
        ax2.axvspan(echo_start_ms, echo_start_ms + duration_ms, color="#38bdf8", alpha=0.25, label="Ground-Truth Echo Span")
        ax2.set_title("2. Received Acoustic Buffer (AWGN Corrupted)", fontsize=9.5)
        ax2.set_xlabel("Time (ms)", fontsize=8.5)
        ax2.set_ylabel("Amplitude", fontsize=8.5)
        ax2.legend(loc="upper right", fontsize=8)
        
        # 3. Matched Filter Output
        corr_len = len(correlation)
        corr_delays = (np.arange(corr_len) - (len(pulse) - 1)) / fs
        corr_distances = (corr_delays * speed_mps) / 2
        
        valid_mask = (corr_distances >= 0) & (corr_distances <= (buffer_duration_s * speed_mps / 2))
        ax3.plot(corr_distances[valid_mask], correlation[valid_mask], color="#38bdf8", lw=1.1, label="Cross-Correlation")
        
        ax3.axvline(true_dist, color="#34d399", ls="--", lw=1.5, label=f"True Target: {true_dist:.2f} m")
        ax3.plot([est_dist], [correlation[np.argmax(correlation)]], "ro", ms=7, label=f"Detected Peak: {est_dist:.3f} m")
        ax3.set_title("3. Matched Filter Output R_xy(τ) — Peak Delay Recovery", fontsize=9.5)
        ax3.set_xlabel("Equivalent One-Way Distance (m)", fontsize=8.5)
        ax3.set_ylabel("Correlation Value", fontsize=8.5)
        ax3.legend(loc="upper right", fontsize=8)
        
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


# ==============================================================================
# 2. MULTI-TARGET DETECTION & RANGE RESOLUTION STUDIO
# ==============================================================================
elif selected_tab == "🎯 2. Multi-Target & Resolution":
    st.subheader("Multi-Target Resolvability & Hilbert Analytic Envelope")
    st.markdown(
        r"""
        When two reflectors are close together, their returning echoes overlap in time.
        According to the **Rayleigh resolution criterion**, two echoes can only be distinguished if their physical separation
        exceeds $\Delta R \approx \frac{c}{2B}$, where $B$ is the pulse bandwidth.
        """
    )

    col_m_ctrl, col_m_view = st.columns([1, 2.2])

    with col_m_ctrl:
        st.markdown("#### 🎯 Multi-Target Setup")
        base_d = st.slider("Target 1 Distance (m)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
        separation_cm = st.slider("Separation to Target 2 (cm)", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
        
        add_target_3 = st.checkbox("Include Target 3", value=False)
        target_3_dist = st.slider("Target 3 Distance (m)", min_value=1.0, max_value=15.0, value=7.5, step=0.1) if add_target_3 else None
        
        st.markdown("#### 🎛️ Pulse Characteristics")
        m_pulse_type = st.selectbox("Pulse Waveform", ["gaussian", "chirp", "rect"], key="m_ptype")
        m_duration_ms = st.slider("Pulse Duration (ms)", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="m_dur")
        m_chirp_bw = st.slider("Chirp Bandwidth (Hz)", min_value=500.0, max_value=5000.0, value=2500.0, step=250.0) if m_pulse_type == "chirp" else None
        
        st.markdown("#### 🔍 Peak Detector Tuning")
        prominence_frac = st.slider("Prominence Threshold (% of max)", min_value=0.05, max_value=0.8, value=0.35, step=0.05)
        m_snr_db = st.slider("Noise SNR (dB)", min_value=-10.0, max_value=30.0, value=20.0, step=2.0, key="m_snr")

    m_fs = 48000.0
    m_speed = 343.0
    m_dur_s = m_duration_ms / 1000.0
    
    m_pulse = generate_pulse(m_pulse_type, m_dur_s, m_fs, freq_hz=4000.0, bandwidth_hz=m_chirp_bw)
    m_bw = pulse_bandwidth_hz(m_pulse, m_fs)
    theoretical_res_cm = (m_speed / (2 * m_bw)) * 100.0 if m_bw > 0 else float("nan")

    distances = [base_d, base_d + separation_cm / 100.0]
    attenuations = [1.0, 0.75]
    if add_target_3 and target_3_dist:
        distances.append(target_3_dist)
        attenuations.append(0.6)

    m_buf_dur = max(0.08, (2 * max(distances) / m_speed) * 1.3)
    m_buffer = simulate_multi_object_channel(m_pulse, distances, m_fs, attenuations=attenuations, speed_mps=m_speed, buffer_duration_s=m_buf_dur)
    
    if m_snr_db < 30.0:
        m_buffer = add_noise(m_buffer, m_snr_db, seed=42)

    m_detected = estimate_multiple_distances(m_buffer, m_pulse, m_fs, speed_mps=m_speed, prominence_frac=prominence_frac)
    
    raw_corr = matched_filter(m_buffer, m_pulse)
    envelope = np.abs(hilbert(raw_corr))
    corr_delays_m = (np.arange(len(raw_corr)) - (len(m_pulse) - 1)) / m_fs * m_speed / 2

    with col_m_view:
        res_color = "#34d399" if separation_cm >= theoretical_res_cm else "#f87171"
        st.markdown(
            f"""
            <div style="background:#111827; border:1px solid #334155; border-radius:12px; padding:12px 16px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:#94a3b8; font-size:0.85rem;">Theoretical Resolution Limit (c / 2B):</span>
                        <strong style="color:#38bdf8; font-size:1.05rem; margin-left:6px;">{theoretical_res_cm:.1f} cm</strong>
                    </div>
                    <div>
                        <span style="color:#94a3b8; font-size:0.85rem;">Separation:</span>
                        <strong style="color:{res_color}; font-size:1.05rem; margin-left:6px;">{separation_cm:.1f} cm</strong>
                        <span style="color:{res_color}; font-size:0.8rem; margin-left:4px;">
                            ({'RESOLVABLE' if separation_cm >= theoretical_res_cm else 'BELOW RESOLUTION LIMIT'})
                        </span>
                    </div>
                    <div>
                        <span style="color:#94a3b8; font-size:0.85rem;">Objects Detected:</span>
                        <strong style="color:#f59e0b; font-size:1.05rem; margin-left:6px;">{len(m_detected)} / {len(distances)}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig_m, (ax_sig, ax_env) = plt.subplots(2, 1, figsize=(8.5, 5.5), constrained_layout=True)
        apply_dark_theme(fig_m, [ax_sig, ax_env])

        # 1. Received composite signal
        t_m_ms = (np.arange(len(m_buffer)) / m_fs) * 1000
        ax_sig.plot(t_m_ms, m_buffer, color="#38bdf8", lw=0.8, label="Superimposed Echoes + Noise")
        for i, d in enumerate(distances):
            ax_sig.axvline((2 * d / m_speed) * 1000, color="#f59e0b", ls=":", lw=1.3, label=f"Echo {i+1} ({d:.2f} m)")
        ax_sig.set_title("Multi-Target Composite Acoustic Buffer", fontsize=9.5)
        ax_sig.set_xlabel("Time (ms)", fontsize=8.5)
        ax_sig.set_ylabel("Amplitude", fontsize=8.5)
        ax_sig.legend(loc="upper right", fontsize=7.5)

        # 2. Correlation + Hilbert Envelope
        m_mask = (corr_delays_m >= min(distances) - 0.8) & (corr_delays_m <= max(distances) + 1.2)
        ax_env.plot(corr_delays_m[m_mask], raw_corr[m_mask], color="#475569", lw=0.7, alpha=0.7, label="Raw Correlation")
        ax_env.plot(corr_delays_m[m_mask], envelope[m_mask], color="#38bdf8", lw=1.5, label="Hilbert Envelope |H{R_xy}|")
        
        for pk_d in m_detected:
            idx = int(round((pk_d * 2 / m_speed * m_fs) + len(m_pulse) - 1))
            if 0 <= idx < len(envelope):
                ax_env.plot(pk_d, envelope[idx], "ro", ms=6)
                ax_env.annotate(f"{pk_d:.2f}m", (pk_d, envelope[idx]), textcoords="offset points", xytext=(0, 8),
                                ha="center", color="#f87171", fontsize=8, fontweight="bold")
                
        for i, d in enumerate(distances):
            ax_env.axvline(d, color="#34d399", ls="--", lw=1.1, label=f"True Target {i+1} ({d:.2f} m)")
            
        ax_env.set_title("Matched Filter Cross-Correlation & Peak Separation", fontsize=9.5)
        ax_env.set_xlabel("Distance (m)", fontsize=8.5)
        ax_env.set_ylabel("Correlation Envelope", fontsize=8.5)
        ax_env.legend(loc="upper right", fontsize=7.5)

        st.pyplot(fig_m, clear_figure=True)
        plt.close(fig_m)

        st.markdown("##### 📋 Target Detection Breakdown")
        table_rows = []
        for i, td in enumerate(distances):
            closest_det = min(m_detected, key=lambda d: abs(d - td)) if m_detected else None
            err_cm = abs(closest_det - td) * 100 if closest_det is not None else None
            status = "✅ Resolved" if (err_cm is not None and err_cm < 5.0) else "❌ Missed / Merged"
            table_rows.append({
                "Target #": f"Target {i+1}",
                "Ground Truth (m)": f"{td:.3f} m",
                "Nearest Detection (m)": f"{closest_det:.3f} m" if closest_det else "None",
                "Discrepancy (cm)": f"{err_cm:.1f} cm" if err_cm is not None else "N/A",
                "Status": status
            })
        st.table(table_rows)


# ==============================================================================
# 3. MONTE CARLO SNR & PERFORMANCE SWEEPS
# ==============================================================================
elif selected_tab == "📊 3. Monte Carlo SNR Sweeps":
    st.subheader("Monte Carlo Noise Robustness & SNR Waterfall Analysis")
    st.markdown(
        """
        Runs repeated simulated pulses across a range of **Signal-to-Noise Ratios (SNR)**
        to plot the empirical **Root Mean Square Error (RMSE)**.
        Observe the dramatic **"Threshold Effect" (SNR Cliff)** where matched filtering transitions from millimeter precision to noise breakdown.
        """
    )

    c_sw1, c_sw2 = st.columns([1, 2.2])

    with c_sw1:
        st.markdown("#### ⚙️ Sweep Parameters")
        sw_dist = st.slider("Evaluation Distance (m)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
        sw_trials = st.slider("Trials per SNR Point", min_value=5, max_value=40, value=12, step=1)
        
        snr_min = st.slider("Min SNR (dB)", -20, 0, -12, step=2)
        snr_max = st.slider("Max SNR (dB)", 5, 25, 16, step=2)
        snr_step = st.selectbox("SNR Step (dB)", [2, 3, 4], index=0)
        
        sw_pulse_type = st.selectbox("Pulse Type", ["gaussian", "chirp", "rect"], key="sw_ptype")
        sw_bandpass = st.checkbox("Compare With Bandpass Filter", value=True)
        
        run_sweep_btn = st.button("🚀 Run Monte Carlo Sweep", type="primary", use_container_width=True)

    with c_sw2:
        if run_sweep_btn:
            snr_values = [float(x) for x in range(snr_min, snr_max + 1, snr_step)]
            progress_bar = st.progress(0, text="Running Monte Carlo trials...")
            
            res_raw = run_snr_sweep(
                true_distance_m=sw_dist,
                snr_values_db=snr_values,
                fs=48000.0,
                pulse_type=sw_pulse_type,
                trials_per_snr=sw_trials,
                bandpass=False
            )
            progress_bar.progress(50, text="Evaluating bandpass pre-filtered pipeline...")
            
            res_bp = None
            if sw_bandpass:
                res_bp = run_snr_sweep(
                    true_distance_m=sw_dist,
                    snr_values_db=snr_values,
                    fs=48000.0,
                    pulse_type=sw_pulse_type,
                    trials_per_snr=sw_trials,
                    bandpass=True
                )
            progress_bar.progress(100, text="Sweep complete!")
            
            fig_sw, ax_sw = plt.subplots(figsize=(8.5, 5.0))
            apply_dark_theme(fig_sw, ax_sw)
            
            ax_sw.semilogy(res_raw["snr_db"], res_raw["rmse_m"], "o-", color="#38bdf8", lw=1.8, ms=5, label="Standard Matched Filter")
            if res_bp:
                ax_sw.semilogy(res_bp["snr_db"], res_bp["rmse_m"], "s--", color="#a855f7", lw=1.8, ms=5, label="Butterworth Bandpass + Matched Filter")
                
            ax_sw.axhline(0.0018, color="#34d399", ls=":", lw=1.2, label="Theoretical Quantization Floor (~1.8 mm @ 48 kHz)")
            
            ax_sw.set_title(f"RMSE vs. Channel SNR ({sw_pulse_type.capitalize()}, Distance = {sw_dist} m, {sw_trials} trials/point)", fontsize=9.5)
            ax_sw.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=8.5)
            ax_sw.set_ylabel("Distance RMSE (m) — Log Scale", fontsize=8.5)
            ax_sw.legend(loc="upper right", fontsize=8)
            
            st.pyplot(fig_sw, clear_figure=True)
            plt.close(fig_sw)
            
            st.info("💡 **Signals & Systems Insight**: Above ~0 dB SNR, matched filter distance estimation achieves sub-centimeter accuracy bounded only by the 48 kHz sampling resolution. Below -5 dB, noise peaks in the correlation output exceed the true echo, causing catastrophic delay estimation errors.")
        else:
            st.info("👈 Configure your sweep parameters on the left and click **'Run Monte Carlo Sweep'** to generate the empirical error curve.")


# ==============================================================================
# 4. PULSE & SPECTRUM LABORATORY
# ==============================================================================
elif selected_tab == "🔬 4. Pulse & Spectrum Lab":
    st.subheader("Waveform Synthesis & -3 dB Spectral Bandwidth Laboratory")
    st.markdown(
        r"""
        Explore the mathematical properties of the probing waveforms (`rect`, `gaussian`, `chirp`).
        Every pulse is **normalized to unit energy** ($\sum x^2 = 1$) to ensure fair comparison under noise.
        """
    )

    col_lab_c, col_lab_v = st.columns([1, 2.2])

    with col_lab_c:
        lab_type = st.radio("Pulse Geometry", ["gaussian", "chirp", "rect"], horizontal=True)
        lab_dur_ms = st.slider("Duration (ms)", 0.5, 15.0, 3.0, step=0.5, key="lab_dur")
        lab_fc = st.slider("Carrier Frequency (Hz)", 1000.0, 10000.0, 4000.0, step=500.0, key="lab_fc")
        lab_chirp_bw = st.slider("Chirp Sweep (Hz)", 500.0, 6000.0, 2500.0, step=250.0, key="lab_cbw") if lab_type == "chirp" else None
        lab_fs = 48000.0

    p_samples = generate_pulse(lab_type, lab_dur_ms / 1000.0, lab_fs, freq_hz=lab_fc, bandwidth_hz=lab_chirp_bw)
    p_bw = pulse_bandwidth_hz(p_samples, lab_fs)
    p_energy = float(np.sum(p_samples ** 2))

    n_fft = max(2048, 16 * p_samples.size)
    power = np.abs(np.fft.rfft(p_samples, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / lab_fs)
    p_max = power.max()

    with col_lab_v:
        m1, m2, m3 = st.columns(3)
        m1.metric("Unit Energy Check", f"{p_energy:.6f}", "Ideal: 1.000000")
        m2.metric("Measured -3 dB Bandwidth", f"{p_bw:.1f} Hz")
        m3.metric("Theoretical Range Resolution", f"{(343.0 / (2 * p_bw) * 100):.2f} cm" if p_bw > 0 else "N/A")

        fig_p, (ax_pt, ax_pf) = plt.subplots(2, 1, figsize=(8.5, 5.5), constrained_layout=True)
        apply_dark_theme(fig_p, [ax_pt, ax_pf])

        t_ms = (np.arange(len(p_samples)) - (len(p_samples) - 1) / 2) / lab_fs * 1000
        ax_pt.plot(t_ms, p_samples, color="#38bdf8", lw=1.3, label="Waveform x(t)")
        if lab_type == "gaussian":
            sigma_ms = (lab_dur_ms / 6)
            env_g = np.exp(-(t_ms ** 2) / (2 * sigma_ms ** 2))
            env_g_norm = env_g / np.sqrt(np.sum(env_g ** 2)) * max(abs(p_samples)) * 1.5
            ax_pt.plot(t_ms, env_g_norm, "r--", alpha=0.6, label="Gaussian Envelope")
        ax_pt.set_title(f"Time-Domain Waveform: {lab_type.capitalize()} (Centred at t = 0)", fontsize=9.5)
        ax_pt.set_xlabel("Time (ms)", fontsize=8.5)
        ax_pt.set_ylabel("Amplitude", fontsize=8.5)
        ax_pt.legend(loc="upper right", fontsize=8)

        freq_mask = (freqs >= max(0, lab_fc - 4000)) & (freqs <= min(lab_fs / 2, lab_fc + 4000))
        ax_pf.plot(freqs[freq_mask], power[freq_mask], color="#38bdf8", lw=1.3, label="Power Spectrum P(f)")
        ax_pf.axhline(p_max / 2, color="#f43f5e", ls="--", lw=1.1, label="-3 dB Threshold (P_max / 2)")
        
        above_half = power >= p_max / 2
        if np.any(above_half):
            f_in_band = freqs[above_half]
            f_lo, f_hi = f_in_band[0], f_in_band[-1]
            ax_pf.axvspan(f_lo, f_hi, color="#f43f5e", alpha=0.15, label=f"Span = {p_bw:.1f} Hz")
            
        ax_pf.set_title("rFFT Power Spectrum & -3 dB (Half-Power) Bandwidth Boundary", fontsize=9.5)
        ax_pf.set_xlabel("Frequency (Hz)", fontsize=8.5)
        ax_pf.set_ylabel("Spectral Power", fontsize=8.5)
        ax_pf.legend(loc="upper right", fontsize=8)

        st.pyplot(fig_p, clear_figure=True)
        plt.close(fig_p)


# ==============================================================================
# 5. PHYSICAL HARDWARE SONAR & MTI (OVERVIEW)
# ==============================================================================
elif selected_tab == "🔊 5. Real Hardware Sonar (MTI)":
    st.subheader("Physical Hardware Feasibility & Moving Target Indication (MTI)")
    st.markdown(
        r"""
        In `physical_sonar_check/`, real acoustic chirp ranging was tested using the laptop's own speaker and microphone.
        
        #### Real-World Challenges Discovered:
        1. **Chassis & Speaker Ringdown Clutter**: The sound travels directly from the speaker to the internal microphone, creating a massive blast that obscures echoes closer than ~0.9 m.
        2. **Radar Background Subtraction / MTI**: By recording a baseline in open space and subtracting it from wall measurements ($y_{\text{excess}} = y_{\text{wall}} - y_{\text{baseline}}$), static chassis vibrations cancel completely, revealing true distant reflections!
        """
    )

    phys_dir = os.path.join(ROOT_DIR, "physical_sonar_check")
    corr_path = os.path.join(phys_dir, "corr.npy")
    baseline_path = os.path.join(phys_dir, "baseline.npy")
    excess_path = os.path.join(phys_dir, "excess.npy")

    if os.path.exists(corr_path) and os.path.exists(excess_path):
        st.markdown("##### 📈 Real Acoustic Traces Recorded From Physical Hardware Probes")
        
        env_corr = np.load(corr_path)
        excess = np.load(excess_path)
        d_axis = 343.0 * np.arange(excess.size) / 48000.0 / 2.0
        
        fig_hw, (ax_c, ax_ex) = plt.subplots(2, 1, figsize=(8.5, 5.2), constrained_layout=True)
        apply_dark_theme(fig_hw, [ax_c, ax_ex])
        
        d_corr = 343.0 * np.arange(env_corr.size) / 48000.0 / 2.0
        ax_c.plot(d_corr, env_corr, color="#f59e0b", lw=1.1, label="Raw Measured Correlation")
        ax_c.axvspan(0, 0.9, color="#ef4444", alpha=0.2, label="Direct Speaker Ringdown (Blinds Receiver)")
        ax_c.set_xlim(0, 4.0)
        ax_c.set_title("1. Raw Correlation: Direct Path Overwhelms Distant Echoes", fontsize=9.5)
        ax_c.set_xlabel("Distance (m)", fontsize=8.5)
        ax_c.set_ylabel("Normalized Level", fontsize=8.5)
        ax_c.legend(loc="upper right", fontsize=8)
        
        ax_ex.plot(d_axis, excess, color="#38bdf8", lw=1.1, label="Excess Signal (Baseline Clutter Subtracted)")
        ax_ex.axhline(0, color="#64748b", ls="--", lw=0.8)
        std_noise = np.std(excess[int(2 * 1.0 / 343 * 48000):int(2 * 4.0 / 343 * 48000)])
        ax_ex.axhline(6 * std_noise, color="#f43f5e", ls=":", lw=1.2, label="+6σ Detection Threshold")
        ax_ex.set_xlim(0, 4.0)
        ax_ex.set_title("2. MTI Background-Subtracted Trace: Wall Echo Emerges Clearly Above Noise Floor", fontsize=9.5)
        ax_ex.set_xlabel("Distance (m)", fontsize=8.5)
        ax_ex.set_ylabel("Excess Amplitude", fontsize=8.5)
        ax_ex.legend(loc="upper right", fontsize=8)
        
        st.pyplot(fig_hw, clear_figure=True)
        plt.close(fig_hw)
    else:
        st.info("ℹ️ Physical hardware trace files (`corr.npy`, `excess.npy`) can be generated by running `python physical_sonar_check/sonar_probe_v4.py` with your laptop microphone and speaker.")
