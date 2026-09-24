"""
Streamlit Web Application: Interactive Sonar Distance Simulator
CSE220 Signals and Systems — Team SevenEight
Contributors: Aditya Tirtho Roy (2305178) & Farhan Ehsas Sami (2305177)

Strict Architectural Rule:
This file ONLY imports pure DSP functions from src/, rendering interactive UI and Plotly vector charts.
No simulation maths is duplicated inside this file.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# 1. Thread and BLAS limits MUST be set before importing numpy/scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Ensure repository root is on sys.path for src/ imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from scipy.signal import hilbert
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import pure DSP functions from src/
from src.pulse import generate_pulse, pulse_bandwidth_hz
from src.channel import simulate_channel, add_noise, simulate_multi_object_channel
from src.receiver import matched_filter, estimate_distance, estimate_multiple_distances
from src.filters import bandpass_filter
from src.evaluate import run_snr_sweep, run_bandwidth_resolution_sweep
from src.audio import make_audible, audible_echo, to_wav_bytes
from src.acoustic import analyse_recording, design_chirp, detect_echo


# ==============================================================================
# PAGE CONFIGURATION & ULTRA-MODERN CYBER-SONAR THEME
# ==============================================================================

st.set_page_config(
    page_title="Sonar Distance Simulator | Team SevenEight",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global canvas reset */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #070b14 70%, #04070e 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #e2e8f0;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    code, pre, .mono-val {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Header Card */
    .hero-banner {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.08) 50%, rgba(168, 85, 247, 0.05) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        margin: 0;
    }

    /* Badges */
    .badge-bar {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 14px;
    }
    .badge-pill {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #cbd5e1;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .badge-pill-cyan {
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
    }
    .badge-pill-purple {
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(192, 132, 252, 0.4);
        color: #c084fc;
    }

    /* Glassmorphism Control Panel Card */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 18px;
    }

    /* Telemetry KPI Cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .stat-box {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 14px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 6px 16px -2px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-box:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.45);
    }
    .stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .val-cyan { color: #38bdf8; text-shadow: 0 0 16px rgba(56, 189, 248, 0.4); }
    .val-emerald { color: #34d399; text-shadow: 0 0 16px rgba(52, 211, 153, 0.4); }
    .val-amber { color: #fbbf24; text-shadow: 0 0 16px rgba(251, 191, 36, 0.4); }
    .val-rose { color: #f87171; text-shadow: 0 0 16px rgba(248, 113, 113, 0.4); }
    .val-indigo { color: #818cf8; text-shadow: 0 0 16px rgba(129, 140, 248, 0.4); }

    .stat-label {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Audio Deck Component */
    .audio-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 16px 20px;
        margin-top: 14px;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.4);
    }
    .audio-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }

    /* Modern Radio Navigation Tab Bar */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 5px;
        gap: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    div[data-testid="stRadio"] label {
        flex: 1;
        padding: 10px 16px !important;
        border-radius: 10px !important;
        background: transparent !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.25s ease !important;
        border: 1px solid transparent !important;
    }
    div[data-testid="stRadio"] label:hover {
        color: #e2e8f0 !important;
        background: rgba(30, 41, 59, 0.5) !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.25), rgba(99, 102, 241, 0.25)) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        color: #38bdf8 !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.2) !important;
    }

    /* Style Streamlit Inputs & Sliders */
    .stSlider > div > div > div > div {
        background-color: #38bdf8 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5) !important;
    }

    /* Modern Table */
    .custom-hud-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(51, 65, 85, 0.6);
        background: rgba(15, 23, 42, 0.6);
        margin-top: 10px;
    }
    .custom-hud-table th {
        background: rgba(30, 41, 59, 0.7);
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 10px 14px;
        border-bottom: 1px solid rgba(51, 65, 85, 0.8);
    }
    .custom-hud-table td {
        padding: 10px 14px;
        font-size: 0.86rem;
        color: #cbd5e1;
        border-bottom: 1px solid rgba(51, 65, 85, 0.3);
    }
    .custom-hud-table tr:last-child td {
        border-bottom: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PLOTLY CYBER-SONAR THEME FACTORY
# ==============================================================================

def create_dark_figure(height: int = 240, x_title: str = "", y_title: str = "") -> go.Figure:
    """Creates a consistent, ultra-modern dark-themed Plotly canvas."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=55, r=25, t=32, b=45),
        paper_bgcolor="rgba(11, 17, 32, 0.0)",
        plot_bgcolor="rgba(11, 17, 32, 0.55)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="rgba(56, 189, 248, 0.4)",
            font=dict(family="JetBrains Mono, monospace", color="#f8fafc", size=11),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=11, color="#94a3b8")),
            gridcolor="rgba(148, 163, 184, 0.1)",
            zerolinecolor="rgba(148, 163, 184, 0.15)",
            showgrid=True,
            linecolor="rgba(51, 65, 85, 0.6)",
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=11, color="#94a3b8")),
            gridcolor="rgba(148, 163, 184, 0.1)",
            zerolinecolor="rgba(148, 163, 184, 0.15)",
            showgrid=True,
            linecolor="rgba(51, 65, 85, 0.6)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(51, 65, 85, 0.5)",
            borderwidth=1,
        ),
    )
    return fig


# ==============================================================================
# HERO BANNER
# ==============================================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">📡 Sonar Distance Simulator</div>
        <p class="hero-sub">
            CSE220 Signals and Systems · Pulse-Echo Ranging & Matched Filter Suite
        </p>
        <div class="badge-bar">
            <span class="badge-pill badge-pill-cyan">Team SevenEight</span>
            <span class="badge-pill">AWGN Channel</span>
            <span class="badge-pill">Unit-Energy Normalization</span>
            <span class="badge-pill">Matched Filter (R_xy)</span>
            <span class="badge-pill">Hilbert Analytic Envelope</span>
            <span class="badge-pill badge-pill-purple">Acoustic Hardware Validated (6.7σ)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# STUDIO NAVIGATION
# ==============================================================================

selected_studio = st.radio(
    "Navigation Studios",
    [
        "📡 1. End-to-End Pipeline & Audio Deck",
        "🎯 2. Multi-Target & Range Resolution",
        "📊 3. Monte Carlo SNR Waterfall",
        "🎙️ 4. Real Acoustic Hardware Validation",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.write("")


# ==============================================================================
# STUDIO 1: END-TO-END PIPELINE & AUDIO DECK
# ==============================================================================
if selected_studio == "📡 1. End-to-End Pipeline & Audio Deck":
    col_ctrl, col_view = st.columns([1, 2.2])

    with col_ctrl:
        st.markdown("#### ⚙️ Transmission & Channel")

        preset = st.selectbox(
            "Scenario Preset",
            ["Custom", "Clean Acoustic Room (+20 dB)", "Noisy Factory Floor (0 dB)", "Submerged Deep Noise (-10 dB)", "Long-Range Chirp (20 m)"],
            index=0,
        )

        if preset == "Clean Acoustic Room (+20 dB)":
            def_dist, def_snr, def_type, def_dur, def_fc = 6.0, 20.0, "gaussian", 0.002, 4000.0
        elif preset == "Noisy Factory Floor (0 dB)":
            def_dist, def_snr, def_type, def_dur, def_fc = 8.5, 0.0, "gaussian", 0.003, 4000.0
        elif preset == "Submerged Deep Noise (-10 dB)":
            def_dist, def_snr, def_type, def_dur, def_fc = 12.0, -10.0, "chirp", 0.008, 4000.0
        elif preset == "Long-Range Chirp (20 m)":
            def_dist, def_snr, def_type, def_dur, def_fc = 20.0, 5.0, "chirp", 0.010, 4000.0
        else:
            def_dist, def_snr, def_type, def_dur, def_fc = 8.0, 15.0, "gaussian", 0.002, 4000.0

        true_dist = st.slider("Target Distance (m)", min_value=0.5, max_value=25.0, value=float(def_dist), step=0.1)
        snr_db = st.slider("Channel SNR (dB)", min_value=-20.0, max_value=30.0, value=float(def_snr), step=1.0)

        pulse_type = st.selectbox("Pulse Type", ["gaussian", "chirp", "rect"], index=["gaussian", "chirp", "rect"].index(def_type))

        cp1, cp2 = st.columns(2)
        with cp1:
            duration_ms = st.number_input("Duration (ms)", min_value=0.5, max_value=20.0, value=float(def_dur * 1000), step=0.5)
        with cp2:
            freq_hz = st.number_input("Carrier Freq (Hz)", min_value=500.0, max_value=12000.0, value=float(def_fc), step=500.0)

        bandwidth_hz = None
        if pulse_type == "chirp":
            bandwidth_hz = st.slider("Chirp Sweep Width (Hz)", min_value=500.0, max_value=6000.0, value=2500.0, step=250.0)

        with st.expander("🛠️ Advanced Channel & Pre-Filter"):
            speed_mps = st.number_input("Speed of Sound (m/s)", min_value=100.0, max_value=2000.0, value=343.0, step=1.0)
            fs = st.number_input("Sampling Rate fs (Hz)", min_value=16000.0, max_value=96000.0, value=48000.0, step=4000.0)
            attenuation = st.slider("Attenuation Factor", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
            rng_seed = st.number_input("RNG Noise Seed", min_value=0, max_value=9999, value=42)
            enable_bandpass = st.checkbox("Enable Butterworth Pre-Filter", value=False)
            bp_order = st.slider("Filter Order", min_value=1, max_value=8, value=4) if enable_bandpass else 4

        st.markdown("#### 🔊 Audio Deck Configuration")
        slowdown = st.select_slider("Playback Slowdown Factor", options=[2.0, 4.0, 8.0], value=4.0)
        st.caption("Stretches ultrasound & millisecond pulses into the human hearing range without resampling.")

    # Pure DSP Execution from src/
    duration_s = duration_ms / 1000.0
    buffer_duration_s = max(0.06, (2 * true_dist / speed_mps) * 1.35)

    pulse = generate_pulse(pulse_type, duration_s, fs, freq_hz=freq_hz, bandwidth_hz=bandwidth_hz)
    measured_bw = pulse_bandwidth_hz(pulse, fs)

    clean_rx = simulate_channel(
        pulse, true_dist, fs=fs, speed_mps=speed_mps,
        attenuation=attenuation, buffer_duration_s=buffer_duration_s
    )

    noisy_rx = add_noise(clean_rx, snr_db, seed=int(rng_seed))

    if enable_bandpass:
        f_low = max(50.0, freq_hz - (measured_bw if measured_bw > 0 else 1000.0) * 1.2)
        f_high = min(fs / 2 - 100.0, freq_hz + (measured_bw if measured_bw > 0 else 1000.0) * 1.2)
        rx_for_detection = bandpass_filter(noisy_rx, fs, f_low, f_high, order=bp_order)
    else:
        rx_for_detection = noisy_rx

    est_dist, correlation = estimate_distance(rx_for_detection, pulse, fs=fs, speed_mps=speed_mps)
    abs_error_m = abs(est_dist - true_dist)
    round_trip_delay_ms = (2 * true_dist / speed_mps) * 1000.0

    with col_view:
        # 1. Telemetry KPI Grid
        err_val_class = "val-emerald" if abs_error_m < 0.05 else ("val-amber" if abs_error_m < 0.3 else "val-rose")
        err_str = f"{abs_error_m * 1000:.1f} mm" if abs_error_m < 0.1 else f"{abs_error_m * 100:.2f} cm"

        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-val val-cyan">{true_dist:.2f} m</div>
                    <div class="stat-label">True Distance</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val {err_val_class}">{est_dist:.3f} m</div>
                    <div class="stat-label">Estimated Distance</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val {err_val_class}">{err_str}</div>
                    <div class="stat-label">Absolute Error</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val val-indigo">{round_trip_delay_ms:.2f} ms</div>
                    <div class="stat-label">Round-Trip Delay</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Interactive Plotly Charts
        # Chart 1: Transmit Pulse
        t_pulse_ms = (np.arange(len(pulse)) / fs) * 1000.0
        fig1 = create_dark_figure(height=180, x_title="Time (ms)", y_title="Amplitude")
        fig1.add_trace(
            go.Scatter(
                x=t_pulse_ms, y=pulse,
                mode="lines", line=dict(color="#38bdf8", width=1.8),
                name=f"TX Pulse ({pulse_type.capitalize()})",
                hovertemplate="Time: %{x:.2f} ms<br>Amplitude: %{y:.4f}<extra></extra>",
            )
        )
        fig1.update_layout(
            title=dict(
                text=f"<b>1. Transmitted Waveform</b>: {pulse_type.capitalize()} (Duration = {duration_ms:.1f} ms, -3 dB BW = {measured_bw:.0f} Hz)",
                font=dict(size=12, color="#f1f5f9"),
            )
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Received Acoustic Buffer
        t_buf_ms = (np.arange(len(noisy_rx)) / fs) * 1000.0
        fig2 = create_dark_figure(height=200, x_title="Buffer Time (ms)", y_title="Amplitude")
        fig2.add_trace(
            go.Scatter(
                x=t_buf_ms, y=noisy_rx,
                mode="lines", line=dict(color="#64748b", width=0.9),
                opacity=0.75, name=f"Noisy RX (SNR {snr_db:.0f} dB)",
                hovertemplate="Time: %{x:.2f} ms<br>RX: %{y:.4f}<extra></extra>",
            )
        )
        if enable_bandpass:
            fig2.add_trace(
                go.Scatter(
                    x=t_buf_ms, y=rx_for_detection,
                    mode="lines", line=dict(color="#c084fc", width=1.2),
                    name=f"Bandpass Filtered (Order {bp_order})",
                    hovertemplate="Time: %{x:.2f} ms<br>Filtered: %{y:.4f}<extra></extra>",
                )
            )
        echo_start_ms = (2 * true_dist / speed_mps) * 1000.0
        fig2.add_vrect(
            x0=echo_start_ms, x1=echo_start_ms + duration_ms,
            fillcolor="rgba(56, 189, 248, 0.15)", layer="below", line_width=1,
            line=dict(color="rgba(56, 189, 248, 0.4)", dash="dot"),
            annotation_text="True Echo Window", annotation_position="top left",
            annotation_font=dict(size=10, color="#38bdf8"),
        )
        fig2.update_layout(
            title=dict(
                text=f"<b>2. Received Channel Buffer</b> (AWGN Corrupted, SNR = {snr_db:.0f} dB)",
                font=dict(size=12, color="#f1f5f9"),
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Matched Filter Cross-Correlation
        corr_len = len(correlation)
        corr_delays = (np.arange(corr_len) - (len(pulse) - 1)) / fs
        corr_distances = (corr_delays * speed_mps) / 2.0
        valid_mask = (corr_distances >= 0) & (corr_distances <= (buffer_duration_s * speed_mps / 2.0))

        fig3 = create_dark_figure(height=230, x_title="Equivalent One-Way Distance (m)", y_title="Correlation R_xy")
        fig3.add_trace(
            go.Scatter(
                x=corr_distances[valid_mask], y=correlation[valid_mask],
                mode="lines", line=dict(color="#38bdf8", width=1.5),
                name="Cross-Correlation",
                hovertemplate="Distance: %{x:.3f} m<br>Correlation: %{y:.3f}<extra></extra>",
            )
        )
        fig3.add_vline(
            x=true_dist, line=dict(color="#34d399", width=1.8, dash="dash"),
            annotation_text=f"True: {true_dist:.2f}m", annotation_position="top left",
            annotation_font=dict(size=10, color="#34d399"),
        )
        peak_idx = int(np.argmax(correlation))
        fig3.add_trace(
            go.Scatter(
                x=[est_dist], y=[correlation[peak_idx]],
                mode="markers+text",
                marker=dict(color="#f43f5e", size=9, line=dict(color="#ffffff", width=1.5)),
                text=[f"Peak: {est_dist:.3f} m"],
                textposition="top center",
                textfont=dict(color="#f43f5e", size=10, family="JetBrains Mono"),
                name="Detected Echo Peak",
                hovertemplate="Detected: %{x:.3f} m<br>Peak Value: %{y:.3f}<extra></extra>",
            )
        )
        fig3.update_layout(
            title=dict(
                text="<b>3. Matched Filter Output</b>: Delay Recovery via Optimal Cross-Correlation R_xy(τ)",
                font=dict(size=12, color="#f1f5f9"),
            )
        )
        st.plotly_chart(fig3, use_container_width=True)

        # 3. Audio Playback Deck
        st.markdown("#### 🔊 Interactive Sonar Audio Deck")
        aud_col1, aud_col2 = st.columns(2)

        with aud_col1:
            st.markdown(
                """
                <div class="audio-card">
                    <div class="audio-title">🔊 Transmitted Probing Ping</div>
                """,
                unsafe_allow_html=True,
            )
            try:
                tx_aud = make_audible(pulse, fs=fs, repeats=4, slowdown=slowdown, gap_s=0.5)
                tx_wav = to_wav_bytes(tx_aud["audio"], tx_aud["rate_hz"])
                st.audio(tx_wav, format="audio/wav")
            except Exception as e:
                st.caption(f"Audio playback note: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with aud_col2:
            st.markdown(
                """
                <div class="audio-card">
                    <div class="audio-title">🎙️ Received Echo & Channel Noise</div>
                """,
                unsafe_allow_html=True,
            )
            try:
                rx_aud = audible_echo(pulse, true_dist, fs=fs, snr_db=snr_db, slowdown=slowdown, frames=4)
                rx_wav = to_wav_bytes(rx_aud["audio"], rx_aud["rate_hz"])
                st.audio(rx_wav, format="audio/wav")
            except Exception as e:
                st.caption(f"Audio playback note: {e}")
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# STUDIO 2: MULTI-TARGET & RANGE RESOLUTION
# ==============================================================================
elif selected_studio == "🎯 2. Multi-Target & Range Resolution":
    st.subheader("Multi-Target Resolvability & Hilbert Analytic Envelope")
    st.markdown(
        r"""
        According to the **Rayleigh Criterion**, two reflecting targets are distinguishable only when their physical separation
        exceeds $\Delta R \approx \frac{c}{2B}$, where $B$ is the pulse bandwidth.
        Close targets cause mutual interference and peak merging unless sufficiently wide bandwidth is transmitted.
        """
    )

    col_m_ctrl, col_m_view = st.columns([1, 2.2])

    with col_m_ctrl:
        st.markdown("#### 🎯 Target Separation Setup")
        base_d = st.slider("Target 1 Distance (m)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
        separation_cm = st.slider("Target 2 Separation (cm)", min_value=1.0, max_value=100.0, value=28.0, step=1.0)

        add_target_3 = st.checkbox("Include Target 3", value=False)
        target_3_dist = st.slider("Target 3 Distance (m)", min_value=1.0, max_value=15.0, value=7.5, step=0.1) if add_target_3 else None

        st.markdown("#### 🎛️ Pulse Characteristics")
        m_pulse_type = st.selectbox("Pulse Waveform", ["gaussian", "chirp", "rect"], index=0, key="m2_ptype")
        m_duration_ms = st.slider("Pulse Duration (ms)", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="m2_dur")
        m_chirp_bw = st.slider("Chirp Bandwidth (Hz)", min_value=500.0, max_value=5000.0, value=2500.0, step=250.0) if m_pulse_type == "chirp" else None

        st.markdown("#### 🔍 Peak Detector Tuning")
        prominence_frac = st.slider("Peak Prominence (% of max)", min_value=0.05, max_value=0.8, value=0.35, step=0.05)
        m_snr_db = st.slider("Noise Level SNR (dB)", min_value=-10.0, max_value=30.0, value=20.0, step=2.0, key="m2_snr")

    m_fs = 48000.0
    m_speed = 343.0
    m_dur_s = m_duration_ms / 1000.0

    m_pulse = generate_pulse(m_pulse_type, m_dur_s, m_fs, freq_hz=4000.0, bandwidth_hz=m_chirp_bw)
    m_bw = pulse_bandwidth_hz(m_pulse, m_fs)
    theoretical_res_cm = (m_speed / (2.0 * m_bw)) * 100.0 if m_bw > 0 else float("nan")

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
    corr_delays_m = (np.arange(len(raw_corr)) - (len(m_pulse) - 1)) / m_fs * m_speed / 2.0

    with col_m_view:
        is_resolvable = separation_cm >= theoretical_res_cm
        res_badge_class = "badge-pill-cyan" if is_resolvable else "badge-pill"
        res_badge_text = "✅ RESOLVABLE (ΔR ≥ c/2B)" if is_resolvable else "⚠️ BELOW RESOLUTION LIMIT (ΔR < c/2B)"

        st.markdown(
            f"""
            <div class="glass-card" style="padding: 14px 20px; margin-bottom: 16px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <span style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Rayleigh Limit:</span>
                        <strong style="color:#38bdf8; font-size:1.15rem; margin-left:6px; font-family:'JetBrains Mono';">{theoretical_res_cm:.1f} cm</strong>
                    </div>
                    <div>
                        <span style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Target Separation:</span>
                        <strong style="color:#f8fafc; font-size:1.15rem; margin-left:6px; font-family:'JetBrains Mono';">{separation_cm:.1f} cm</strong>
                        <span class="badge-pill {res_badge_class}" style="margin-left:8px;">{res_badge_text}</span>
                    </div>
                    <div>
                        <span style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Objects Found:</span>
                        <strong style="color:#fbbf24; font-size:1.15rem; margin-left:6px; font-family:'JetBrains Mono';">{len(m_detected)} / {len(distances)}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plot 1: Composite Signal
        t_m_ms = (np.arange(len(m_buffer)) / m_fs) * 1000.0
        fig_m1 = create_dark_figure(height=180, x_title="Time (ms)", y_title="Amplitude")
        fig_m1.add_trace(
            go.Scatter(
                x=t_m_ms, y=m_buffer,
                mode="lines", line=dict(color="#38bdf8", width=1.0),
                name="Superimposed Echoes + Noise",
                hovertemplate="Time: %{x:.2f} ms<br>Amp: %{y:.4f}<extra></extra>",
            )
        )
        for i, d in enumerate(distances):
            echo_t = (2 * d / m_speed) * 1000.0
            fig_m1.add_vline(
                x=echo_t, line=dict(color="#fbbf24", width=1.2, dash="dot"),
                annotation_text=f"Echo {i+1} ({d:.2f}m)", annotation_position="top left",
                annotation_font=dict(size=9, color="#fbbf24"),
            )
        fig_m1.update_layout(
            title=dict(text="<b>Multi-Target Composite Acoustic Buffer</b>", font=dict(size=12, color="#f1f5f9")),
        )
        st.plotly_chart(fig_m1, use_container_width=True)

        # Plot 2: Hilbert Envelope & Peak Markers
        m_mask = (corr_delays_m >= min(distances) - 0.8) & (corr_delays_m <= max(distances) + 1.2)
        fig_m2 = create_dark_figure(height=240, x_title="Distance (m)", y_title="Correlation Envelope")
        fig_m2.add_trace(
            go.Scatter(
                x=corr_delays_m[m_mask], y=raw_corr[m_mask],
                mode="lines", line=dict(color="#475569", width=0.8),
                opacity=0.5, name="Raw Correlation",
                hovertemplate="Dist: %{x:.3f} m<br>Raw R_xy: %{y:.3f}<extra></extra>",
            )
        )
        fig_m2.add_trace(
            go.Scatter(
                x=corr_delays_m[m_mask], y=envelope[m_mask],
                mode="lines", line=dict(color="#38bdf8", width=1.8),
                name="Hilbert Envelope |H{R_xy}|",
                hovertemplate="Dist: %{x:.3f} m<br>Envelope: %{y:.3f}<extra></extra>",
            )
        )
        for pk_d in m_detected:
            idx = int(round((pk_d * 2.0 / m_speed * m_fs) + len(m_pulse) - 1))
            if 0 <= idx < len(envelope):
                fig_m2.add_trace(
                    go.Scatter(
                        x=[pk_d], y=[envelope[idx]],
                        mode="markers+text",
                        marker=dict(color="#f43f5e", size=8, line=dict(color="#ffffff", width=1.2)),
                        text=[f"{pk_d:.2f}m"],
                        textposition="top center",
                        textfont=dict(color="#f43f5e", size=10, family="JetBrains Mono"),
                        name="Detected Echo Peak",
                        showlegend=False,
                    )
                )
        for i, d in enumerate(distances):
            fig_m2.add_vline(
                x=d, line=dict(color="#34d399", width=1.5, dash="dash"),
                annotation_text=f"Target {i+1} ({d:.2f}m)", annotation_position="top left",
                annotation_font=dict(size=9, color="#34d399"),
            )
        fig_m2.update_layout(
            title=dict(text="<b>Cross-Correlation Envelope & Peak Resolvability</b>", font=dict(size=12, color="#f1f5f9")),
        )
        st.plotly_chart(fig_m2, use_container_width=True)

        # HUD Detection Table
        st.markdown("##### 📋 Target Resolution Breakdown")
        table_html = """
        <table class="custom-hud-table">
            <thead>
                <tr>
                    <th>Target Index</th>
                    <th>Ground Truth</th>
                    <th>Nearest Detected</th>
                    <th>Discrepancy</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        for i, td in enumerate(distances):
            closest_det = min(m_detected, key=lambda d: abs(d - td)) if m_detected else None
            err_cm = abs(closest_det - td) * 100.0 if closest_det is not None else None
            status_html = '<span style="color:#34d399; font-weight:600;">✅ Resolved</span>' if (err_cm is not None and err_cm < 5.0) else '<span style="color:#f87171; font-weight:600;">❌ Merged / Missed</span>'
            det_str = f"{closest_det:.3f} m" if closest_det else "None"
            err_str = f"{err_cm:.1f} cm" if err_cm is not None else "N/A"
            table_html += f"""
                <tr>
                    <td>Target {i+1}</td>
                    <td><code>{td:.3f} m</code></td>
                    <td><code>{det_str}</code></td>
                    <td><code>{err_str}</code></td>
                    <td>{status_html}</td>
                </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # Dynamic Bandwidth vs Resolution Sweep
        with st.expander("📈 Dynamic Bandwidth vs. Resolution Sweep (Week 3 Gate)"):
            st.caption("Executes repeated two-target separation trials across varying pulse bandwidths to verify the theoretical trend.")
            if st.button("🚀 Run Bandwidth Resolution Sweep", type="secondary"):
                with st.spinner("Evaluating empirical range resolutions..."):
                    configs = [
                        {"pulse_type": "gaussian", "duration_s": 0.008, "freq_hz": 4000.0},
                        {"pulse_type": "gaussian", "duration_s": 0.004, "freq_hz": 4000.0},
                        {"pulse_type": "gaussian", "duration_s": 0.002, "freq_hz": 4000.0},
                        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0, "bandwidth_hz": 2000.0},
                        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0, "bandwidth_hz": 4000.0},
                    ]
                    sw_res = run_bandwidth_resolution_sweep(
                        configs, m_fs, trials=6,
                        separations_m=[c / 100.0 for c in range(2, 61, 2)],
                    )

                    fig_res = create_dark_figure(height=260, x_title="Bandwidth (Hz)", y_title="Minimum Resolvable Distance (cm)")
                    bw_vals = [bw for bw in sw_res["bandwidth_hz"]]
                    meas_cm = [res * 100.0 if res else None for res in sw_res["resolved_m"]]
                    theory_cm = [th * 100.0 if th else None for th in sw_res["theory_m"]]

                    fig_res.add_trace(
                        go.Scatter(
                            x=bw_vals, y=theory_cm,
                            mode="lines", line=dict(color="#94a3b8", dash="dash", width=1.5),
                            name="Rayleigh Limit: c / (2B)",
                        )
                    )
                    fig_res.add_trace(
                        go.Scatter(
                            x=bw_vals, y=meas_cm,
                            mode="markers+lines",
                            marker=dict(color="#38bdf8", size=8),
                            line=dict(color="#38bdf8", width=2),
                            name="Empirically Measured",
                        )
                    )
                    fig_res.update_layout(
                        title=dict(text="<b>Empirical Resolution vs. Pulse Bandwidth</b>", font=dict(size=12, color="#f1f5f9")),
                    )
                    st.plotly_chart(fig_res, use_container_width=True)


# ==============================================================================
# STUDIO 3: MONTE CARLO SNR WATERFALL
# ==============================================================================
elif selected_studio == "📊 3. Monte Carlo SNR Waterfall":
    st.subheader("Monte Carlo Noise Robustness & SNR Threshold Waterfall")
    st.markdown(
        """
        Runs repeated stochastic noise trials across a wide span of **Signal-to-Noise Ratios (SNR)**
        to plot the empirical **Root Mean Square Error (RMSE)**.
        Observe the dramatic **"Threshold Cliff"** where matched filtering transitions abruptly from millimeter accuracy to noise peak breakdown.
        """
    )

    c_sw1, c_sw2 = st.columns([1, 2.2])

    with c_sw1:
        st.markdown("#### ⚙️ Sweep Configuration")
        sw_dist = st.slider("Evaluation Distance (m)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
        sw_trials = st.slider("Trials per SNR Step", min_value=5, max_value=35, value=12, step=1)

        snr_min = st.slider("Min SNR (dB)", -20, 0, -12, step=2)
        snr_max = st.slider("Max SNR (dB)", 4, 24, 16, step=2)
        snr_step = st.selectbox("SNR Step Size (dB)", [2, 3, 4], index=0)

        sw_pulse_type = st.selectbox("Pulse Type", ["gaussian", "chirp", "rect"], index=0, key="sw_ptype")
        sw_bandpass = st.checkbox("Compare With Bandpass Filter", value=True)

        run_sweep_btn = st.button("🚀 Run Monte Carlo Sweep", type="primary", use_container_width=True)

    with c_sw2:
        if run_sweep_btn:
            snr_values = [float(x) for x in range(snr_min, snr_max + 1, snr_step)]
            prog_bar = st.progress(0, text="Executing Monte Carlo trials...")

            res_raw = run_snr_sweep(
                true_distance_m=sw_dist,
                snr_values_db=snr_values,
                fs=48000.0,
                pulse_type=sw_pulse_type,
                trials_per_snr=sw_trials,
                bandpass=False,
            )
            prog_bar.progress(50, text="Evaluating pre-filtered pipeline...")

            res_bp = None
            if sw_bandpass:
                res_bp = run_snr_sweep(
                    true_distance_m=sw_dist,
                    snr_values_db=snr_values,
                    fs=48000.0,
                    pulse_type=sw_pulse_type,
                    trials_per_snr=sw_trials,
                    bandpass=True,
                )
            prog_bar.progress(100, text="Monte Carlo sweep complete!")

            fig_sw = create_dark_figure(height=360, x_title="Channel SNR (dB)", y_title="Distance RMSE (m) — Log Scale")
            fig_sw.update_yaxes(type="log")

            fig_sw.add_trace(
                go.Scatter(
                    x=res_raw["snr_db"], y=res_raw["rmse_m"],
                    mode="lines+markers",
                    line=dict(color="#38bdf8", width=2.0),
                    marker=dict(color="#38bdf8", size=6),
                    name="Standard Matched Filter",
                    hovertemplate="SNR: %{x:.0f} dB<br>RMSE: %{y:.4f} m<extra></extra>",
                )
            )

            if res_bp:
                fig_sw.add_trace(
                    go.Scatter(
                        x=res_bp["snr_db"], y=res_bp["rmse_m"],
                        mode="lines+markers",
                        line=dict(color="#c084fc", width=2.0, dash="dash"),
                        marker=dict(color="#c084fc", size=6, symbol="square"),
                        name="Butterworth Pre-Filter + Matched Filter",
                        hovertemplate="SNR: %{x:.0f} dB<br>Filtered RMSE: %{y:.4f} m<extra></extra>",
                    )
                )

            # Theoretical sample quantization floor line: c / (4 * fs) ~ 1.8 mm
            fig_sw.add_hline(
                y=0.0018, line=dict(color="#34d399", width=1.5, dash="dot"),
                annotation_text="Quantization Floor (~1.8 mm @ 48 kHz)",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color="#34d399"),
            )

            fig_sw.update_layout(
                title=dict(
                    text=f"<b>Monte Carlo Distance RMSE vs. SNR</b> ({sw_pulse_type.capitalize()}, Distance = {sw_dist} m, {sw_trials} trials/pt)",
                    font=dict(size=12, color="#f1f5f9"),
                )
            )
            st.plotly_chart(fig_sw, use_container_width=True)

            st.markdown(
                """
                <div class="glass-card" style="margin-top:10px;">
                    <h5 style="color:#38bdf8; margin-top:0;">💡 Signals & Systems Analysis:</h5>
                    <p style="font-size:0.9rem; line-height:1.5; color:#cbd5e1; margin-bottom:0;">
                        <strong>1. The High-SNR Regime (> 0 dB):</strong> Matched filtering is mathematically optimal for maximizing output peak SNR in AWGN (derived via Cauchy-Schwarz inequality). The error plateaus at ~1.8 mm, governed by whole-sample discretization (<code>c / 4fs</code>).<br>
                        <strong>2. The Threshold Breakdown (SNR Cliff, < -5 dB):</strong> As noise power overwhelms echo energy, spurious noise crests in the cross-correlation buffer exceed the true reflection peak, causing catastrophic delay misidentification.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("👈 Set your sweep parameters on the left and click **'Run Monte Carlo Sweep'** to generate the error waterfall.")


# ==============================================================================
# STUDIO 4: REAL ACOUSTIC HARDWARE VALIDATION (FEATURE 8)
# ==============================================================================
elif selected_studio == "🎙️ 4. Real Acoustic Hardware Validation":
    st.subheader("Real Acoustic Hardware Ranging & Clutter Cancellation")
    st.markdown(
        """
        Validation of the matched-filtering pipeline using **physical sound waves in a real room** 
        captured with a laptop's built-in speaker and microphone.
        Demonstrates multi-frame coherent stacking, direct-path alignment, and open-space clutter subtraction.
        """
    )

    col_ac1, col_ac2 = st.columns([1, 2.2])

    with col_ac1:
        st.markdown("#### 🎙️ Hardware Configuration")
        ac_speed = st.number_input("Speed of Sound in Room (m/s)", min_value=320.0, max_value=360.0, value=343.0, step=0.5)
        min_range = st.slider("Min Detection Window (m)", min_value=0.5, max_value=2.5, value=1.0, step=0.1)
        st.caption("Rejects initial speaker chassis ringing swamping the microphone.")
        max_range = st.slider("Max Detection Window (m)", min_value=3.0, max_value=8.0, value=5.0, step=0.5)

        st.markdown("#### 📋 Physical Parameters")
        st.markdown(
            """
            - **Transmit Chirp**: 2 000 Hz → 8 000 Hz (Hanning-windowed)
            - **Chirp Duration**: 10 ms
            - **Frame Period**: 260 ms (stacked over multiple pings)
            - **Archived Target**: Concrete Wall in room
            """
        )

    # Load archived real recordings from physical_sonar_check/
    ac_fs = 48000.0
    chirp_sig = design_chirp(2000.0, 8000.0, 0.010, ac_fs)
    frame_samples = int((0.010 + 0.250) * ac_fs)

    rx_file = ROOT_DIR / "physical_sonar_check" / "rx.npy"
    base_file = ROOT_DIR / "physical_sonar_check" / "baseline.npy"

    if rx_file.exists() and base_file.exists():
        raw_rx = np.load(rx_file)
        base_env = np.load(base_file)

        ac_result = analyse_recording(
            raw_rx, chirp_sig, frame_samples, ac_fs,
            baseline_envelope=base_env, speed_mps=ac_speed,
            min_range_m=min_range, max_range_m=max_range,
        )

        wall_dist = ac_result["distance_m"]
        wall_sigma = ac_result["quality_sigma"]
        excess_env = ac_result["excess"]
        stacked_env = ac_result["envelope"]

        with col_ac2:
            st.markdown(
                f"""
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-val val-cyan">Concrete Wall</div>
                        <div class="stat-label">Reflecting Target</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val val-emerald">{wall_dist:.3f} m</div>
                        <div class="stat-label">Detected Distance</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val val-amber">{wall_sigma:.2f} σ</div>
                        <div class="stat-label">Quality Confidence</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val val-indigo">48.0 kHz</div>
                        <div class="stat-label">Acoustic Sampling</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            dist_axis = np.arange(len(excess_env)) * ac_speed / ac_fs / 2.0
            range_mask = (dist_axis >= 0) & (dist_axis <= max_range + 0.5)

            # Plot 1: Stacked Envelope with Direct Path
            fig_ac1 = create_dark_figure(height=200, x_title="Distance (m)", y_title="Normalized Envelope")
            fig_ac1.add_trace(
                go.Scatter(
                    x=dist_axis[range_mask], y=stacked_env[range_mask],
                    mode="lines", line=dict(color="#64748b", width=1.1),
                    name="Raw Stacked Envelope (with Clutter)",
                    hovertemplate="Dist: %{x:.3f} m<br>Envelope: %{y:.4f}<extra></extra>",
                )
            )
            fig_ac1.add_vline(
                x=0, line=dict(color="#38bdf8", width=1.5),
                annotation_text="Direct Arrival (t=0)", annotation_position="top right",
                annotation_font=dict(size=9, color="#38bdf8"),
            )
            fig_ac1.update_layout(
                title=dict(text="<b>1. Direct-Path Aligned Stacked Correlation Trace</b>", font=dict(size=12, color="#f1f5f9")),
            )
            st.plotly_chart(fig_ac1, use_container_width=True)

            # Plot 2: Clutter-Cancelled Excess Return
            fig_ac2 = create_dark_figure(height=240, x_title="Distance (m)", y_title="Excess Amplitude")
            fig_ac2.add_trace(
                go.Scatter(
                    x=dist_axis[range_mask], y=excess_env[range_mask],
                    mode="lines", line=dict(color="#38bdf8", width=1.8),
                    name="Excess Return (Clutter Subtracted)",
                    hovertemplate="Dist: %{x:.3f} m<br>Excess: %{y:.4f}<extra></extra>",
                )
            )
            fig_ac2.add_trace(
                go.Scatter(
                    x=[wall_dist], y=[excess_env[ac_result["peak_index"]]],
                    mode="markers+text",
                    marker=dict(color="#34d399", size=10, line=dict(color="#ffffff", width=1.5)),
                    text=[f"Wall @ {wall_dist:.3f} m ({wall_sigma:.1f}σ)"],
                    textposition="top center",
                    textfont=dict(color="#34d399", size=10, family="JetBrains Mono"),
                    name="Wall Detection",
                )
            )
            fig_ac2.add_vrect(
                x0=min_range, x1=max_range,
                fillcolor="rgba(56, 189, 248, 0.08)", layer="below", line_width=1,
                line=dict(color="rgba(56, 189, 248, 0.3)", dash="dash"),
                annotation_text="Search Window", annotation_position="top left",
                annotation_font=dict(size=9, color="#38bdf8"),
            )
            fig_ac2.update_layout(
                title=dict(text="<b>2. Clutter-Cancelled Excess Return: Confirmed Wall Detection at 1.797 m</b>", font=dict(size=12, color="#f1f5f9")),
            )
            st.plotly_chart(fig_ac2, use_container_width=True)

            # Audio Player for the Real Recording
            st.markdown(
                """
                <div class="audio-card">
                    <div class="audio-title">🔊 Listen to Physical Chirp Probe</div>
                """,
                unsafe_allow_html=True,
            )
            try:
                chirp_aud = make_audible(chirp_sig, fs=ac_fs, repeats=4, slowdown=1.0, gap_s=0.3)
                chirp_wav = to_wav_bytes(chirp_aud["audio"], chirp_aud["rate_hz"])
                st.audio(chirp_wav, format="audio/wav")
            except Exception as e:
                st.caption(f"Audio note: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Archived physical recordings (rx.npy / baseline.npy) not found in physical_sonar_check/.")
