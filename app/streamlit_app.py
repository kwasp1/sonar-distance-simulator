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
import textwrap
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

# Import pure DSP functions from src/
from src.pulse import generate_pulse, pulse_bandwidth_hz
from src.channel import simulate_channel, add_noise, simulate_multi_object_channel
from src.receiver import matched_filter, estimate_distance, estimate_multiple_distances
from src.filters import bandpass_filter
from src.evaluate import run_snr_sweep, run_bandwidth_resolution_sweep, describe_pulse_config
from src.audio import make_audible, audible_echo, to_wav_bytes
from src.acoustic import analyse_recording, design_chirp


# ==============================================================================
# PAGE CONFIGURATION & ULTRA-MODERN CYBER-SONAR THEME
# ==============================================================================

st.set_page_config(
    page_title="Sonar Distance Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global canvas reset & base typography */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #070b14 70%, #04070e 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 16.5px !important;
        color: #e2e8f0;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    h3 {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin-top: 32px !important;
        margin-bottom: 18px !important;
    }
    h4 {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        margin-bottom: 16px !important;
    }
    h5 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }

    p, span, label {
        font-size: 1.04rem;
    }

    .stMarkdown p {
        font-size: 1.05rem !important;
        line-height: 1.65 !important;
        color: #cbd5e1;
    }

    code, pre, .mono-val {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.96rem !important;
    }

    /* Minimal Clean Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.08) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 18px 26px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 60%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }

    /* Sidebar Navigation Styling */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.95) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.18) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 10px;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        padding: 14px 18px !important;
        border-radius: 12px !important;
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(51, 65, 85, 0.6) !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 1.06rem !important;
        transition: all 0.25s ease !important;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div {
        font-size: 1.06rem !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        color: #f8fafc !important;
        background: rgba(30, 41, 59, 0.85) !important;
        border-color: rgba(56, 189, 248, 0.45) !important;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.25), rgba(99, 102, 241, 0.25)) !important;
        border: 1px solid rgba(56, 189, 248, 0.65) !important;
        color: #38bdf8 !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.3) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    section[data-testid="stSidebar"] .stCaption {
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        line-height: 1.55 !important;
    }

    /* Top Control Deck Container */
    .control-deck {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }

    /* Telemetry KPI Cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: 16px;
        margin-bottom: 8px;
    }
    .stat-box {
        background: rgba(11, 17, 32, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-box:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.5);
    }
    .stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.05rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .val-cyan { color: #38bdf8; text-shadow: 0 0 14px rgba(56, 189, 248, 0.4); }
    .val-emerald { color: #34d399; text-shadow: 0 0 14px rgba(52, 211, 153, 0.4); }
    .val-amber { color: #fbbf24; text-shadow: 0 0 14px rgba(251, 191, 36, 0.4); }
    .val-rose { color: #f87171; text-shadow: 0 0 14px rgba(248, 113, 113, 0.4); }
    .val-indigo { color: #818cf8; text-shadow: 0 0 14px rgba(129, 140, 248, 0.4); }

    .stat-label {
        font-size: 0.90rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Audio Deck Component */
    .audio-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.65));
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
    }
    .audio-title {
        font-size: 1.08rem;
        font-weight: 600;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
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
        margin-top: 12px;
    }
    .custom-hud-table th {
        background: rgba(30, 41, 59, 0.85);
        color: #94a3b8;
        font-size: 0.94rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 12px 18px;
        border-bottom: 1px solid rgba(51, 65, 85, 0.8);
    }
    .custom-hud-table td {
        padding: 12px 18px;
        font-size: 1.04rem;
        color: #e2e8f0;
        border-bottom: 1px solid rgba(51, 65, 85, 0.3);
    }
    .custom-hud-table tr:last-child td {
        border-bottom: none;
    }

    /* Badges */
    .badge-pill {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #e2e8f0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.90rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-pill-cyan {
        background: rgba(14, 165, 233, 0.18);
        border: 1px solid rgba(56, 189, 248, 0.55);
        color: #38bdf8;
    }

    /* Widget Labels & Controls */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] {
        font-size: 1.04rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
        margin-bottom: 6px !important;
    }

    /* Enhanced Slider Styling */
    div[data-testid="stSlider"] {
        padding: 6px 0 !important;
    }
    div[data-testid="stSlider"] label {
        font-size: 1.04rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
        margin-bottom: 6px !important;
    }
    div[data-baseweb="slider"] {
        margin-top: 8px !important;
        margin-bottom: 8px !important;
    }
    div[data-baseweb="slider"] > div {
        background: transparent !important;
    }
    /* Slider Track Background */
    div[data-baseweb="slider"] > div > div {
        background: rgba(30, 41, 59, 0.9) !important;
        border-radius: 9999px !important;
        height: 7px !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    /* Slider Filled Track */
    div[data-baseweb="slider"] > div > div > div:first-child {
        background: linear-gradient(90deg, #0284c7 0%, #38bdf8 60%, #818cf8 100%) !important;
        border-radius: 9999px !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.45) !important;
    }
    /* Slider Thumb Knob */
    div[data-baseweb="slider"] div[role="slider"] {
        background: #090d16 !important;
        border: 2.5px solid #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.85) !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
    }
    div[data-baseweb="slider"] div[role="slider"]:hover {
        transform: scale(1.22) !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 1.0) !important;
        border-color: #7dd3fc !important;
    }
    /* Slider Value Indicator Tag */
    div[data-baseweb="slider"] div[role="slider"] > div {
        background: #0f172a !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        border-radius: 6px !important;
        padding: 3px 8px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6) !important;
    }
    div[data-testid="stSliderTickBar"] + div,
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        font-size: 1.0rem !important;
    }

    /* Selectbox & Text/Number Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: rgba(56, 189, 248, 0.55) !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
    }
    div[data-testid="stNumberInput"] button {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #cbd5e1 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: rgba(56, 189, 248, 0.25) !important;
        color: #ffffff !important;
    }

    /* Radio options & Checkboxes */
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="stCheckbox"] label p {
        font-size: 1.04rem !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="stExpander"] details summary span p {
        font-size: 1.12rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        line-height: 1.55 !important;
    }

    /* Modern Glow Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.02em !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.55) !important;
        border-color: rgba(255, 255, 255, 0.45) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Distinct Plot Instrument Card with Generous Separation */
    div[data-testid="stPlotlyChart"] {
        background: rgba(13, 20, 36, 0.75) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 18px !important;
        padding: 24px 26px 26px 26px !important;
        margin-top: 14px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }
    div[data-testid="stPlotlyChart"]:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7) !important;
    }
    /* Natural visual separation between consecutive plots */
    .plot-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(56, 189, 248, 0.3) 25%, rgba(129, 140, 248, 0.3) 75%, transparent 100%);
        margin: 54px 0 38px 0;
        border: none;
    }
    .plot-header-card {
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .plot-header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }
    .plot-header-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(14, 165, 233, 0.16);
        border: 1px solid rgba(56, 189, 248, 0.45);
        color: #38bdf8;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .plot-header-desc {
        font-size: 1.04rem;
        color: #94a3b8;
        line-height: 1.55;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PLOTLY CYBER-SONAR THEME FACTORY & CONFIG
# ==============================================================================

PLOT_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "doubleClick": False,
    "showAxisDragHandles": False,
    "responsive": True,
}


def create_dark_figure(height: int = 380, x_title: str = "", y_title: str = "") -> go.Figure:
    """Creates a full-width, eye-catching, high-resolution dark-themed Plotly canvas."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        dragmode=False,
        margin=dict(l=75, r=35, t=50, b=55),
        paper_bgcolor="rgba(11, 17, 32, 0.0)",
        plot_bgcolor="rgba(11, 17, 32, 0.65)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=13),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="rgba(56, 189, 248, 0.5)",
            font=dict(family="JetBrains Mono, monospace", color="#f8fafc", size=12),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=14, color="#cbd5e1")),
            tickfont=dict(size=12, color="#94a3b8"),
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.18)",
            showgrid=True,
            linecolor="rgba(51, 65, 85, 0.7)",
            fixedrange=True,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=14, color="#cbd5e1")),
            tickfont=dict(size=12, color="#94a3b8"),
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.18)",
            showgrid=True,
            linecolor="rgba(51, 65, 85, 0.7)",
            fixedrange=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(51, 65, 85, 0.6)",
            borderwidth=1,
        ),
    )
    return fig


def render_chart(fig: go.Figure) -> None:
    """Renders a Plotly figure with locked zoom/pan so mouse wheel scrolls the page normally."""
    fig.update_layout(dragmode=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def render_plot_header(stage_tag: str, title: str, description: str):
    """Renders a visually distinct, high-contrast header card for each plot instrument."""
    st.markdown(
        f"""
        <div class="plot-header-card">
            <div class="plot-header-title">
                <span class="plot-header-badge">{stage_tag}</span>
                <span>{title}</span>
            </div>
            <p class="plot-header-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plot_divider():
    """Renders a subtle luminous divider between consecutive plots to ensure distinct separation."""
    st.markdown('<div class="plot-divider"></div>', unsafe_allow_html=True)


def render_hud_table(headers: list[str], rows: list[list[str]]):
    """Renders a custom glassmorphic HUD table without markdown indentation code-block glitches."""
    th_tags = "".join(f"<th>{h}</th>" for h in headers)
    tr_tags = ""
    for row in rows:
        td_tags = "".join(f"<td>{cell}</td>" for cell in row)
        tr_tags += f"<tr>{td_tags}</tr>"
    table_str = f'<table class="custom-hud-table"><thead><tr>{th_tags}</tr></thead><tbody>{tr_tags}</tbody></table>'
    st.markdown(table_str, unsafe_allow_html=True)


def record_live_acoustic(
    chirp: np.ndarray,
    repeats: int = 16,
    fs: float = 48000.0,
    amplitude: float = 0.4,
    gap_s: float = 0.250,
) -> np.ndarray:
    """Plays chirp sequence through system speakers and records via microphone simultaneously."""
    import sounddevice as sd

    frame_samples = int(round((len(chirp) / fs + gap_s) * fs))
    frame = np.zeros(frame_samples)
    frame[: chirp.size] = amplitude * chirp
    transmit = np.concatenate([np.tile(frame, repeats), np.zeros(frame_samples)])
    stereo = np.zeros((transmit.size, 2))
    stereo[:, 1] = transmit
    recording = sd.playrec(stereo, samplerate=int(fs), channels=1, blocking=True)[:, 0]
    return recording


def run_live_band_check(fs: float = 48000.0) -> tuple[np.ndarray, np.ndarray]:
    """Plays a 2-22 kHz linear sweep and measures speaker+mic frequency response."""
    import sounddevice as sd

    t_dur, f0, f1 = 0.5, 2000.0, 22000.0
    n = int(t_dur * fs)
    t = np.arange(n) / fs
    sweep = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * t_dur) * t ** 2))
    fade = int(0.005 * fs)
    sweep[:fade] *= np.linspace(0, 1, fade)
    sweep[-fade:] *= np.linspace(1, 0, fade)

    tx = np.concatenate([np.zeros(int(fs // 10)), 0.3 * sweep, np.zeros(int(fs // 5))])
    stereo = np.zeros((tx.size, 2))
    stereo[:, 1] = tx
    rx = sd.playrec(stereo, samplerate=int(fs), channels=1, blocking=True)[:, 0]

    f_axis = np.fft.rfftfreq(tx.size, 1 / fs)
    resp = np.abs(np.fft.rfft(rx)) / (np.abs(np.fft.rfft(tx)) + 1e-12)
    freq_centers = []
    resp_db_list = []
    for lo in range(2000, 22000, 1000):
        m = (f_axis >= lo) & (f_axis < lo + 1000)
        freq_centers.append((lo + 500) / 1000.0)
        resp_db_list.append(float(20 * np.log10(np.median(resp[m]) + 1e-12)))
    peak = max(resp_db_list)
    norm_db = [d - peak for d in resp_db_list]
    return np.array(freq_centers), np.array(norm_db)


# ==============================================================================
# SIDEBAR NAVIGATION (Shifted to Left of Screen)
# ==============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 22px 0; text-align: center;">
            <div style="font-size: 1.65rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Sonar Studio
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 5px;">
                Pulse-Echo Suite
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_studio = st.radio(
        "Navigation Studios",
        [
            "1. End-to-End Pipeline & Audio",
            "2. Multi-Target & Resolution",
            "3. Monte Carlo SNR Waterfall",
            "4. Real Acoustic Hardware",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Select any studio below to explore matched filtering, multi-target resolution, SNR sweeps, or acoustic wall ranging.")


# ==============================================================================
# CLEAN HEADER (No redundant subtitle or badge clutter)
# ==============================================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">Sonar Distance Simulator</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# STUDIO 1: END-TO-END PIPELINE & AUDIO DECK
# ==============================================================================
if selected_studio == "1. End-to-End Pipeline & Audio":
    # TOP CONTROL DECK
    with st.container():
        st.markdown('<div class="control-deck">', unsafe_allow_html=True)
        st.markdown("#### Transmission & Channel Parameters")

        # Row 1: Presets & Primary Knobs
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([1.2, 1.4, 1.4, 1.0])
        with r1_c1:
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

        with r1_c2:
            true_dist = st.slider("Target Distance (m)", min_value=0.5, max_value=25.0, value=float(def_dist), step=0.1)
        with r1_c3:
            snr_db = st.slider("Channel SNR (dB)", min_value=-20.0, max_value=30.0, value=float(def_snr), step=1.0)
        with r1_c4:
            pulse_type = st.selectbox("Pulse Type", ["gaussian", "chirp", "rect"], index=["gaussian", "chirp", "rect"].index(def_type))

        # Row 2: Timing, Frequencies & Slowdown
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            duration_ms = st.number_input("Duration (ms)", min_value=0.5, max_value=20.0, value=float(def_dur * 1000), step=0.5)
        with r2_c2:
            freq_hz = st.number_input("Carrier Freq (Hz)", min_value=500.0, max_value=12000.0, value=float(def_fc), step=500.0)
        with r2_c3:
            bandwidth_hz = None
            if pulse_type == "chirp":
                bandwidth_hz = st.slider("Chirp Sweep Width (Hz)", min_value=500.0, max_value=6000.0, value=2500.0, step=250.0)
            else:
                st.caption("Chirp sweep bandwidth applies when Pulse Type is 'chirp'.")
        with r2_c4:
            slowdown = st.select_slider("Audio Slowdown Factor", options=[2.0, 4.0, 8.0], value=4.0)

        # Advanced Settings Expander
        with st.expander("Advanced Settings & Pre-Filter"):
            adv_c1, adv_c2, adv_c3, adv_c4 = st.columns(4)
            with adv_c1:
                speed_mps = st.number_input("Speed of Sound (m/s)", min_value=100.0, max_value=2000.0, value=343.0, step=1.0)
            with adv_c2:
                fs = st.number_input("Sampling Rate fs (Hz)", min_value=16000.0, max_value=96000.0, value=48000.0, step=4000.0)
            with adv_c3:
                attenuation = st.slider("Echo Attenuation", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
            with adv_c4:
                rng_seed = st.number_input("RNG Noise Seed", min_value=0, max_value=9999, value=42)

            st.markdown("##### Butterworth Bandpass Pre-Filter")
            bp_c1, bp_c2, bp_c3 = st.columns([1, 1, 2])
            with bp_c1:
                enable_bandpass = st.checkbox("Enable Butterworth Pre-Filter", value=False)
            with bp_c2:
                bp_order = st.slider("Filter Order", min_value=1, max_value=8, value=4) if enable_bandpass else 4
            with bp_c3:
                manual_cutoffs = st.checkbox("Manual Cutoff Frequencies", value=False) if enable_bandpass else False

            bp_low_user, bp_high_user = None, None
            if enable_bandpass and manual_cutoffs:
                cut_c1, cut_c2 = st.columns(2)
                nyq_limit = fs / 2.0
                with cut_c1:
                    bp_low_user = st.slider("Low Cutoff f_low (Hz)", min_value=50.0, max_value=nyq_limit - 500.0, value=max(50.0, freq_hz - 1500.0), step=50.0)
                with cut_c2:
                    bp_high_user = st.slider("High Cutoff f_high (Hz)", min_value=bp_low_user + 100.0, max_value=nyq_limit - 50.0, value=min(nyq_limit - 100.0, freq_hz + 1500.0), step=50.0)

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
            if manual_cutoffs and bp_low_user and bp_high_user:
                f_low, f_high = bp_low_user, bp_high_user
            else:
                f_low = max(50.0, freq_hz - (measured_bw if measured_bw > 0 else 1000.0) * 1.2)
                f_high = min(fs / 2 - 100.0, freq_hz + (measured_bw if measured_bw > 0 else 1000.0) * 1.2)
            rx_for_detection = bandpass_filter(noisy_rx, fs, f_low, f_high, order=bp_order)
        else:
            rx_for_detection = noisy_rx

        est_dist, correlation = estimate_distance(rx_for_detection, pulse, fs=fs, speed_mps=speed_mps)
        abs_error_m = abs(est_dist - true_dist)
        round_trip_delay_ms = (2 * true_dist / speed_mps) * 1000.0

        # Telemetry KPI Grid
        err_val_class = "val-emerald" if abs_error_m < 0.05 else ("val-amber" if abs_error_m < 0.3 else "val-rose")
        err_str = f"{abs_error_m * 1000:.1f} mm" if abs_error_m < 0.1 else f"{abs_error_m * 100:.2f} cm"

        st.markdown(
            textwrap.dedent(f"""
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
            """),
            unsafe_allow_html=True,
        )

        # Audio Deck Cards
        aud_col1, aud_col2 = st.columns(2)
        with aud_col1:
            st.markdown(
                """
                <div class="audio-card">
                    <div class="audio-title">Transmitted Probing Ping</div>
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
                    <div class="audio-title">Received Echo & Channel Noise</div>
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

        st.markdown("</div>", unsafe_allow_html=True)

    # FULL-WIDTH HIGH SIGNAL PLOTS (Below all controls)
    st.markdown("### Real-Time Acoustic & Matched-Filter Signals")

    render_plot_header(
        "STAGE 1",
        "Transmitted Probing Waveform",
        f"Unit-energy acoustic pulse ({pulse_type.capitalize()}, {duration_ms:.1f} ms duration, {measured_bw:.0f} Hz bandwidth). Inspect in time domain or frequency domain below.",
    )

    pulse_view_mode = st.radio(
        "Transmit Pulse Domain View",
        ["Time-Domain Waveform x[n]", "Frequency Power Spectrum |X(f)|² (-3 dB Bandwidth)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if pulse_view_mode == "Time-Domain Waveform x[n]":
        t_pulse_ms = (np.arange(len(pulse)) / fs) * 1000.0
        fig1 = create_dark_figure(height=380, x_title="Time (ms)", y_title="Amplitude")
        fig1.add_trace(
            go.Scatter(
                x=t_pulse_ms, y=pulse,
                mode="lines", line=dict(color="#38bdf8", width=2.2),
                name=f"TX Pulse ({pulse_type.capitalize()})",
                hovertemplate="Time: %{x:.2f} ms<br>Amplitude: %{y:.4f}<extra></extra>",
            )
        )
        fig1.update_layout(
            title=dict(
                text=f"<b>Transmitted Waveform x[n]</b>: {pulse_type.capitalize()} (Duration = {duration_ms:.1f} ms, -3 dB Span = {measured_bw:.0f} Hz)",
                font=dict(size=14, color="#f1f5f9"),
            )
        )
        render_chart(fig1)
    else:
        n_fft = max(4096, 32 * pulse.size)
        p_spec = np.abs(np.fft.rfft(pulse, n=n_fft)) ** 2
        f_axis = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        p_max = p_spec.max() if p_spec.max() > 0 else 1.0
        p_db = 10 * np.log10(p_spec / p_max + 1e-12)

        above_half = p_spec >= p_max / 2.0
        f_lo = f_axis[above_half][0] if np.any(above_half) else freq_hz
        f_hi = f_axis[above_half][-1] if np.any(above_half) else freq_hz

        fig_spec = create_dark_figure(height=380, x_title="Frequency (kHz)", y_title="Power (dB rel. peak)")
        max_plot_f = min(fs / 2, max(freq_hz + 3 * measured_bw + 2000, 8000.0))
        k_mask = f_axis <= max_plot_f

        fig_spec.add_trace(
            go.Scatter(
                x=f_axis[k_mask] / 1000.0, y=p_db[k_mask],
                mode="lines", line=dict(color="#38bdf8", width=2.2),
                name=f"Power Spectrum P(f)",
                hovertemplate="Freq: %{x:.2f} kHz<br>Power: %{y:.1f} dB<extra></extra>",
            )
        )
        fig_spec.add_hline(
            y=-3.0, line=dict(color="#94a3b8", width=1.8, dash="dash"),
            annotation_text="-3 dB (Half-Power)", annotation_position="top left",
            annotation_font=dict(size=11, color="#94a3b8"),
        )
        fig_spec.add_vrect(
            x0=f_lo / 1000.0, x1=f_hi / 1000.0,
            fillcolor="rgba(52, 211, 153, 0.15)", layer="below", line_width=1.2,
            line=dict(color="#34d399", dash="dot"),
            annotation_text=f"Bandwidth = {measured_bw:.0f} Hz", annotation_position="top left",
            annotation_font=dict(size=11, color="#34d399"),
        )
        fig_spec.update_yaxes(range=[-45, 3])
        fig_spec.update_layout(
            title=dict(
                text=f"<b>Spectral Power Profile |X(f)|²</b> (f_low = {f_lo:.0f} Hz, f_high = {f_hi:.0f} Hz, -3 dB Span = {measured_bw:.0f} Hz)",
                font=dict(size=14, color="#f1f5f9"),
            )
        )
        render_chart(fig_spec)

    render_plot_divider()

    # Plot 2: Full-Width Received Acoustic Buffer
    render_plot_header(
        "STAGE 2",
        "Received Channel Acoustic Buffer",
        f"Simulated propagation through medium with geometric spreading attenuation, absorption, and AWGN noise (SNR = {snr_db:.0f} dB).",
    )
    t_buf_ms = (np.arange(len(noisy_rx)) / fs) * 1000.0
    fig2 = create_dark_figure(height=390, x_title="Buffer Time (ms)", y_title="Amplitude")
    fig2.add_trace(
        go.Scatter(
            x=t_buf_ms, y=noisy_rx,
            mode="lines", line=dict(color="#64748b", width=1.1),
            opacity=0.8, name=f"Noisy RX (SNR {snr_db:.0f} dB)",
            hovertemplate="Time: %{x:.2f} ms<br>RX: %{y:.4f}<extra></extra>",
        )
    )
    if enable_bandpass:
        fig2.add_trace(
            go.Scatter(
                x=t_buf_ms, y=rx_for_detection,
                mode="lines", line=dict(color="#c084fc", width=1.6),
                name=f"Bandpass Filtered (Order {bp_order})",
                hovertemplate="Time: %{x:.2f} ms<br>Filtered: %{y:.4f}<extra></extra>",
            )
        )
    echo_start_ms = (2 * true_dist / speed_mps) * 1000.0
    fig2.add_vrect(
        x0=echo_start_ms, x1=echo_start_ms + duration_ms,
        fillcolor="rgba(56, 189, 248, 0.16)", layer="below", line_width=1.2,
        line=dict(color="rgba(56, 189, 248, 0.5)", dash="dot"),
        annotation_text="True Echo Window", annotation_position="top left",
        annotation_font=dict(size=11, color="#38bdf8"),
    )
    fig2.update_layout(
        title=dict(
            text=f"<b>Propagated Channel Acoustic Signal</b> (AWGN Corrupted, SNR = {snr_db:.0f} dB)",
            font=dict(size=14, color="#f1f5f9"),
        )
    )
    render_chart(fig2)

    render_plot_divider()

    # Plot 3: Full-Width Matched Filter Output
    render_plot_header(
        "STAGE 3",
        "Matched Filter Cross-Correlation Output",
        f"Optimal linear cross-correlation detector R_xy(tau) achieving sub-centimeter delay recovery (Estimated: {est_dist:.3f} m, True: {true_dist:.3f} m).",
    )
    corr_len = len(correlation)
    corr_delays = (np.arange(corr_len) - (len(pulse) - 1)) / fs
    corr_distances = (corr_delays * speed_mps) / 2.0
    valid_mask = (corr_distances >= 0) & (corr_distances <= (buffer_duration_s * speed_mps / 2.0))

    fig3 = create_dark_figure(height=420, x_title="Equivalent One-Way Distance (m)", y_title="Correlation Output R_xy(τ)")
    fig3.add_trace(
        go.Scatter(
            x=corr_distances[valid_mask], y=correlation[valid_mask],
            mode="lines", line=dict(color="#38bdf8", width=1.8),
            name="Matched Filter Cross-Correlation",
            hovertemplate="Distance: %{x:.3f} m<br>Correlation: %{y:.3f}<extra></extra>",
        )
    )
    fig3.add_vline(
        x=true_dist, line=dict(color="#34d399", width=2.0, dash="dash"),
        annotation_text=f"True Target: {true_dist:.2f}m", annotation_position="top left",
        annotation_font=dict(size=11, color="#34d399"),
    )
    peak_idx = int(np.argmax(correlation))
    fig3.add_trace(
        go.Scatter(
            x=[est_dist], y=[correlation[peak_idx]],
            mode="markers+text",
            marker=dict(color="#f43f5e", size=11, line=dict(color="#ffffff", width=1.8)),
            text=[f"Peak: {est_dist:.3f} m"],
            textposition="top center",
            textfont=dict(color="#f43f5e", size=11, family="JetBrains Mono"),
            name="Detected Echo Peak",
            hovertemplate="Detected: %{x:.3f} m<br>Peak Value: %{y:.3f}<extra></extra>",
        )
    )
    fig3.update_layout(
        title=dict(
            text="<b>Matched Filter Output R_xy(τ)</b>: Sub-Centimeter Delay Recovery via Optimal Cross-Correlation",
            font=dict(size=14, color="#f1f5f9"),
        )
    )
    render_chart(fig3)


# ==============================================================================
# STUDIO 2: MULTI-TARGET & RANGE RESOLUTION
# ==============================================================================
elif selected_studio == "2. Multi-Target & Resolution":
    # TOP CONTROL DECK
    with st.container():
        st.markdown('<div class="control-deck">', unsafe_allow_html=True)
        st.markdown("#### Multi-Target Setup & Detector Tuning")

        # Row 1: Target Distances & Attenuations
        t_c1, t_c2, t_c3, t_c4 = st.columns(4)
        with t_c1:
            base_d = st.slider("Target 1 Distance (m)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
            att_1 = st.slider("Target 1 Reflectivity", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
        with t_c2:
            separation_cm = st.slider("Target 2 Separation (cm)", min_value=1.0, max_value=100.0, value=28.0, step=1.0)
            att_2 = st.slider("Target 2 Reflectivity", min_value=0.1, max_value=1.0, value=0.75, step=0.05)
        with t_c3:
            add_target_3 = st.checkbox("Include Target 3", value=False)
            target_3_dist, att_3 = None, None
            if add_target_3:
                target_3_dist = st.slider("Target 3 Distance (m)", min_value=1.0, max_value=15.0, value=7.5, step=0.1)
                att_3 = st.slider("Target 3 Reflectivity", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
            else:
                st.caption("Check to add a 3rd reflector to the acoustic scene.")
        with t_c4:
            m_pulse_type = st.selectbox("Pulse Waveform", ["gaussian", "chirp", "rect"], index=0, key="m2_ptype")
            m_duration_ms = st.slider("Pulse Duration (ms)", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="m2_dur")

        # Row 2: Bandwidth & Detector Settings
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            m_chirp_bw = st.slider("Chirp Bandwidth (Hz)", min_value=500.0, max_value=5000.0, value=2500.0, step=250.0) if m_pulse_type == "chirp" else None
            if m_pulse_type != "chirp":
                st.caption("Gaussian/Rect bandwidth is set by duration.")
        with b_c2:
            prominence_frac = st.slider("Peak Prominence (% of max)", min_value=0.05, max_value=0.8, value=0.35, step=0.05)
        with b_c3:
            pulse_len_preview = int(round((m_duration_ms / 1000.0) * 48000.0))
            min_sep_samples = st.slider(
                "Min Peak Separation (Samples)",
                min_value=2,
                max_value=max(10, pulse_len_preview),
                value=max(4, pulse_len_preview // 4),
                step=2,
                help="Detector floor. Lower values allow resolving overlapping crests for wideband chirps.",
            )
        with b_c4:
            m_snr_db = st.slider("Channel SNR (dB)", min_value=-10.0, max_value=30.0, value=20.0, step=2.0, key="m2_snr")

        # Pure DSP Execution from src/
        m_fs = 48000.0
        m_speed = 343.0
        m_dur_s = m_duration_ms / 1000.0

        m_pulse = generate_pulse(m_pulse_type, m_dur_s, m_fs, freq_hz=4000.0, bandwidth_hz=m_chirp_bw)
        m_bw = pulse_bandwidth_hz(m_pulse, m_fs)
        theoretical_res_cm = (m_speed / (2.0 * m_bw)) * 100.0 if m_bw > 0 else float("nan")

        distances = [base_d, base_d + separation_cm / 100.0]
        attenuations = [att_1, att_2]
        if add_target_3 and target_3_dist and att_3:
            distances.append(target_3_dist)
            attenuations.append(att_3)

        m_buf_dur = max(0.08, (2 * max(distances) / m_speed) * 1.3)
        m_buffer = simulate_multi_object_channel(m_pulse, distances, m_fs, attenuations=attenuations, speed_mps=m_speed, buffer_duration_s=m_buf_dur)

        if m_snr_db < 30.0:
            m_buffer = add_noise(m_buffer, m_snr_db, seed=42)

        m_detected = estimate_multiple_distances(
            m_buffer, m_pulse, m_fs, speed_mps=m_speed,
            prominence_frac=prominence_frac, min_separation_samples=min_sep_samples,
        )

        raw_corr = matched_filter(m_buffer, m_pulse)
        envelope = np.abs(hilbert(raw_corr))
        corr_delays_m = (np.arange(len(raw_corr)) - (len(m_pulse) - 1)) / m_fs * m_speed / 2.0

        # Rayleigh Limit Status Card
        is_resolvable = separation_cm >= theoretical_res_cm
        res_badge_class = "badge-pill-cyan" if is_resolvable else "badge-pill"
        res_badge_text = "RESOLVABLE (ΔR ≥ c/2B)" if is_resolvable else "BELOW RESOLUTION LIMIT (ΔR < c/2B)"

        st.markdown(
            textwrap.dedent(f"""
            <div class="stat-grid" style="margin-top: 10px;">
                <div class="stat-box">
                    <div class="stat-val val-cyan">{theoretical_res_cm:.1f} cm</div>
                    <div class="stat-label">Rayleigh Resolution Limit (c / 2B)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val val-indigo">{separation_cm:.1f} cm</div>
                    <div class="stat-label">Target Separation</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val val-amber">{len(m_detected)} / {len(distances)}</div>
                    <div class="stat-label">Objects Detected</div>
                </div>
                <div class="stat-box">
                    <div style="font-size: 1.15rem; font-weight: 700; margin-top: 6px;" class="val-emerald">
                        <span class="badge-pill {res_badge_class}">{res_badge_text}</span>
                    </div>
                    <div class="stat-label">Rayleigh Status</div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # Multi-Target Audio Deck & HUD Table
        m_t1, m_t2 = st.columns([1.2, 1.8])
        with m_t1:
            st.markdown(
                """
                <div class="audio-card">
                    <div class="audio-title">Multi-Target Composite Echo Audio</div>
                """,
                unsafe_allow_html=True,
            )
            m_slowdown = st.select_slider("Multi-Echo Slowdown", options=[2.0, 4.0, 8.0], value=4.0, key="m_aud_slow")
            try:
                m_aud = make_audible(m_buffer, fs=m_fs, repeats=4, slowdown=m_slowdown, gap_s=0.5)
                m_wav = to_wav_bytes(m_aud["audio"], m_aud["rate_hz"])
                st.audio(m_wav, format="audio/wav")
            except Exception as e:
                st.caption(f"Audio note: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with m_t2:
            st.markdown("##### Target Resolution Breakdown")
            hud_headers = ["Target Index", "Ground Truth", "Nearest Detected", "Discrepancy", "Status"]
            hud_rows = []
            for i, td in enumerate(distances):
                closest_det = min(m_detected, key=lambda d: abs(d - td)) if m_detected else None
                err_cm = abs(closest_det - td) * 100.0 if closest_det is not None else None
                status_html = '<span style="color:#34d399; font-weight:600;">Resolved</span>' if (err_cm is not None and err_cm < 5.0) else '<span style="color:#f87171; font-weight:600;">Merged / Missed</span>'
                det_str = f"<code>{closest_det:.3f} m</code>" if closest_det else "None"
                err_str = f"<code>{err_cm:.1f} cm</code>" if err_cm is not None else "N/A"
                hud_rows.append([
                    f"Target {i+1}",
                    f"<code>{td:.3f} m</code>",
                    det_str,
                    err_str,
                    status_html,
                ])
            render_hud_table(hud_headers, hud_rows)

        st.markdown("</div>", unsafe_allow_html=True)

    # FULL-WIDTH HIGH SIGNAL PLOTS (Below all controls)
    st.markdown("### Multi-Target Acoustic Waveforms & Hilbert Envelopes")

    # Plot 1: Full-Width Composite Signal
    render_plot_header(
        "STAGE 1",
        "Multi-Target Composite Acoustic Buffer",
        "Superposition of reflected acoustic echoes from distinct physical boundaries with independent attenuations and AWGN noise.",
    )
    t_m_ms = (np.arange(len(m_buffer)) / m_fs) * 1000.0
    fig_m1 = create_dark_figure(height=390, x_title="Buffer Time (ms)", y_title="Amplitude")
    fig_m1.add_trace(
        go.Scatter(
            x=t_m_ms, y=m_buffer,
            mode="lines", line=dict(color="#38bdf8", width=1.3),
            name="Superimposed Echoes + Noise",
            hovertemplate="Time: %{x:.2f} ms<br>Amp: %{y:.4f}<extra></extra>",
        )
    )
    for i, d in enumerate(distances):
        echo_t = (2 * d / m_speed) * 1000.0
        fig_m1.add_vline(
            x=echo_t, line=dict(color="#fbbf24", width=1.8, dash="dot"),
            annotation_text=f"Echo {i+1} ({d:.2f}m)", annotation_position="top left",
            annotation_font=dict(size=10, color="#fbbf24"),
        )
    fig_m1.update_layout(
        title=dict(text="<b>Multi-Target Composite Buffer</b>: Superimposed Reflected Echoes in Medium", font=dict(size=14, color="#f1f5f9")),
    )
    render_chart(fig_m1)

    render_plot_divider()

    # Plot 2: Full-Width Hilbert Envelope & Peak Markers
    render_plot_header(
        "STAGE 2",
        "Matched Filter & Hilbert Analytic Envelope Peak Separation",
        "Demodulation of matched filter output via Hilbert transform |H{R_xy}| for envelope peak extraction and resolution testing.",
    )
    m_mask = (corr_delays_m >= min(distances) - 0.8) & (corr_delays_m <= max(distances) + 1.2)
    fig_m2 = create_dark_figure(height=420, x_title="Distance (m)", y_title="Correlation Envelope")
    fig_m2.add_trace(
        go.Scatter(
            x=corr_delays_m[m_mask], y=raw_corr[m_mask],
            mode="lines", line=dict(color="#475569", width=1.0),
            opacity=0.55, name="Raw Correlation R_xy",
            hovertemplate="Dist: %{x:.3f} m<br>Raw R_xy: %{y:.3f}<extra></extra>",
        )
    )
    fig_m2.add_trace(
        go.Scatter(
            x=corr_delays_m[m_mask], y=envelope[m_mask],
            mode="lines", line=dict(color="#38bdf8", width=2.2),
            name="Hilbert Analytic Envelope |H{R_xy}|",
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
                    marker=dict(color="#f43f5e", size=10, line=dict(color="#ffffff", width=1.8)),
                    text=[f"{pk_d:.2f}m"],
                    textposition="top center",
                    textfont=dict(color="#f43f5e", size=11, family="JetBrains Mono"),
                    name="Detected Echo Peak",
                    showlegend=False,
                )
            )
    for i, d in enumerate(distances):
        fig_m2.add_vline(
            x=d, line=dict(color="#34d399", width=1.8, dash="dash"),
            annotation_text=f"Target {i+1} ({d:.2f}m)", annotation_position="top left",
            annotation_font=dict(size=11, color="#34d399"),
        )
    fig_m2.update_layout(
        title=dict(text="<b>Matched Filter Output & Hilbert Analytic Envelope Peak Separation</b>", font=dict(size=14, color="#f1f5f9")),
    )
    render_chart(fig_m2)

    # Dynamic Bandwidth vs Resolution Sweep Expander
    with st.expander("Dynamic Bandwidth vs. Resolution Sweep"):
        st.caption("Executes repeated two-target separation trials across varying pulse bandwidths to verify the theoretical trend.")
        
        sw_c1, sw_c2 = st.columns([2, 1])
        with sw_c1:
            sweep_comparison_mode = st.radio(
                "Sweep Pulse Selection",
                [
                    "Compare Gaussians & Chirps (Balanced)",
                    "Chirps Only (Varying Sweep Width, Fixed 2 ms Duration)",
                    "Gaussians Only (Varying Duration: 8ms, 4ms, 2ms, 1ms)"
                ],
                horizontal=True,
            )
        with sw_c2:
            sw_trials_num = st.slider("Resolution Sweep Trials per Step", min_value=4, max_value=16, value=6, step=2)

        if st.button("Run Bandwidth Resolution Sweep", type="secondary"):
            with st.spinner("Evaluating empirical range resolutions..."):
                if "Chirps Only" in sweep_comparison_mode:
                    configs = [
                        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0, "bandwidth_hz": 1000.0},
                        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0, "bandwidth_hz": 2000.0},
                        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0, "bandwidth_hz": 3000.0},
                        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0, "bandwidth_hz": 4000.0},
                    ]
                elif "Gaussians Only" in sweep_comparison_mode:
                    configs = [
                        {"pulse_type": "gaussian", "duration_s": 0.008, "freq_hz": 4000.0},
                        {"pulse_type": "gaussian", "duration_s": 0.004, "freq_hz": 4000.0},
                        {"pulse_type": "gaussian", "duration_s": 0.002, "freq_hz": 4000.0},
                        {"pulse_type": "gaussian", "duration_s": 0.001, "freq_hz": 4000.0},
                    ]
                else:
                    configs = [
                        {"pulse_type": "gaussian", "duration_s": 0.008, "freq_hz": 4000.0},
                        {"pulse_type": "gaussian", "duration_s": 0.004, "freq_hz": 4000.0},
                        {"pulse_type": "gaussian", "duration_s": 0.002, "freq_hz": 4000.0},
                        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0, "bandwidth_hz": 2000.0},
                        {"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 4000.0, "bandwidth_hz": 4000.0},
                    ]

                sw_res = run_bandwidth_resolution_sweep(
                    configs, m_fs, trials=sw_trials_num,
                    separations_m=[c / 100.0 for c in range(2, 61, 2)],
                )

                fig_res = create_dark_figure(height=400, x_title="Measured Bandwidth (Hz)", y_title="Minimum Resolvable Separation (cm)")
                bw_vals = [bw for bw in sw_res["bandwidth_hz"]]
                meas_cm = [res * 100.0 if res else None for res in sw_res["resolved_m"]]
                theory_cm = [th * 100.0 if th else None for th in sw_res["theory_m"]]
                labels = sw_res["labels"]

                fig_res.add_trace(
                    go.Scatter(
                        x=bw_vals, y=theory_cm,
                        mode="lines", line=dict(color="#94a3b8", dash="dash", width=1.8),
                        name="Rayleigh Limit: c / (2B)",
                        hovertemplate="BW: %{x:.0f} Hz<br>Theoretical Limit: %{y:.1f} cm<extra></extra>",
                    )
                )
                fig_res.add_trace(
                    go.Scatter(
                        x=bw_vals, y=meas_cm,
                        mode="markers+lines",
                        marker=dict(color="#38bdf8", size=10),
                        line=dict(color="#38bdf8", width=2.4),
                        text=labels,
                        name="Empirical Resolution",
                        hovertemplate="<b>%{text}</b><br>BW: %{x:.0f} Hz<br>Resolved: %{y:.1f} cm<extra></extra>",
                    )
                )
                fig_res.update_layout(
                    title=dict(text="<b>Empirical Resolution vs. Pulse Bandwidth</b>", font=dict(size=14, color="#f1f5f9")),
                )
                render_chart(fig_res)


# ==============================================================================
# STUDIO 3: MONTE CARLO SNR WATERFALL
# ==============================================================================
elif selected_studio == "3. Monte Carlo SNR Waterfall":
    # TOP CONTROL DECK
    with st.container():
        st.markdown('<div class="control-deck">', unsafe_allow_html=True)
        st.markdown("#### Monte Carlo Noise Sweep Configuration")

        sw_c1, sw_c2, sw_c3, sw_c4 = st.columns(4)
        with sw_c1:
            sw_dist = st.slider("Evaluation Distance (m)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
            sw_trials = st.slider("Trials per SNR Step", min_value=5, max_value=35, value=12, step=1)
        with sw_c2:
            snr_min = st.slider("Min SNR (dB)", -20, 0, -12, step=2)
            snr_max = st.slider("Max SNR (dB)", 4, 24, 16, step=2)
        with sw_c3:
            snr_step = st.selectbox("SNR Step Size (dB)", [2, 3, 4], index=0)
            sw_pulse_type = st.selectbox("Pulse Type", ["gaussian", "chirp", "rect"], index=0, key="sw_ptype")
        with sw_c4:
            sw_dur_ms = st.slider("Pulse Duration (ms)", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="sw_dur")
            sw_fc = st.number_input("Carrier Freq (Hz)", min_value=1000.0, max_value=10000.0, value=4000.0, step=500.0, key="sw_fc")
            sw_bandpass = st.checkbox("Compare With Bandpass Filter", value=True)

        run_sweep_btn = st.button("Run Monte Carlo Sweep", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # FULL-WIDTH HIGH SIGNAL PLOTS (Below all controls)
    st.markdown("### Empirical Noise Waterfall & SNR Cliff Breakdown")

    if run_sweep_btn:
        snr_values = [float(x) for x in range(snr_min, snr_max + 1, snr_step)]
        prog_bar = st.progress(0, text="Executing Monte Carlo trials...")

        res_raw = run_snr_sweep(
            true_distance_m=sw_dist,
            snr_values_db=snr_values,
            fs=48000.0,
            pulse_type=sw_pulse_type,
            duration_s=sw_dur_ms / 1000.0,
            freq_hz=sw_fc,
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
                duration_s=sw_dur_ms / 1000.0,
                freq_hz=sw_fc,
                trials_per_snr=sw_trials,
                bandpass=True,
            )
        prog_bar.progress(100, text="Monte Carlo sweep complete!")

        fig_sw = create_dark_figure(height=500, x_title="Channel SNR (dB)", y_title="Distance RMSE (m) — Log Scale")
        fig_sw.update_yaxes(type="log")

        fig_sw.add_trace(
            go.Scatter(
                x=res_raw["snr_db"], y=res_raw["rmse_m"],
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.4),
                marker=dict(color="#38bdf8", size=8),
                name="Standard Matched Filter",
                hovertemplate="SNR: %{x:.0f} dB<br>RMSE: %{y:.4f} m<extra></extra>",
            )
        )

        if res_bp:
            fig_sw.add_trace(
                go.Scatter(
                    x=res_bp["snr_db"], y=res_bp["rmse_m"],
                    mode="lines+markers",
                    line=dict(color="#c084fc", width=2.4, dash="dash"),
                    marker=dict(color="#c084fc", size=8, symbol="square"),
                    name="Butterworth Pre-Filter + Matched Filter",
                    hovertemplate="SNR: %{x:.0f} dB<br>Filtered RMSE: %{y:.4f} m<extra></extra>",
                )
            )

        # Theoretical sample quantization floor line: c / (4 * fs) ~ 1.8 mm
        fig_sw.add_hline(
            y=0.0018, line=dict(color="#34d399", width=1.8, dash="dot"),
            annotation_text="Quantization Floor (~1.8 mm @ 48 kHz)",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color="#34d399"),
        )

        fig_sw.update_layout(
            title=dict(
                text=f"<b>Monte Carlo Distance RMSE vs. SNR</b> ({sw_pulse_type.capitalize()}, Distance = {sw_dist} m, {sw_trials} trials/pt)",
                font=dict(size=14, color="#f1f5f9"),
            )
        )
        render_chart(fig_sw)

        st.markdown(
            textwrap.dedent("""
            <div class="control-deck" style="margin-top:14px;">
                <h5 style="color:#38bdf8; margin-top:0;">Signals & Systems Analysis:</h5>
                <p style="font-size:1.05rem; line-height:1.7; color:#cbd5e1; margin-bottom:0;">
                    <strong>1. The High-SNR Regime (> 0 dB):</strong> Matched filtering is mathematically optimal for maximizing output peak SNR in AWGN (derived via Cauchy-Schwarz inequality). The error plateaus at ~1.8 mm, governed by whole-sample discretization (<code>c / 4fs</code>).<br>
                    <strong>2. The Threshold Breakdown (SNR Cliff, < -5 dB):</strong> As noise power overwhelms echo energy, spurious noise crests in the cross-correlation buffer exceed the true reflection peak, causing catastrophic delay misidentification.<br>
                    <strong>3. Bandpass Pre-Filtering Effect:</strong> In pure white noise, bandpass pre-filtering yields negligible benefit because the matched filter already achieves optimal spectral weighting. Its primary value in practical sonar is rejecting out-of-band non-stationary interference.
                </p>
            </div>
            """),
            unsafe_allow_html=True,
        )
    else:
        st.info("Set your sweep parameters above and click **'Run Monte Carlo Sweep'** to generate the error waterfall.")


# ==============================================================================
# STUDIO 4: REAL ACOUSTIC HARDWARE VALIDATION (FEATURE 8)
# ==============================================================================
elif selected_studio == "4. Real Acoustic Hardware":
    ac_fs = 48000.0
    chirp_sig = design_chirp(2000.0, 8000.0, 0.010, ac_fs)
    frame_samples = int((0.010 + 0.250) * ac_fs)

    rx_file = ROOT_DIR / "physical_sonar_check" / "rx.npy"
    base_file = ROOT_DIR / "physical_sonar_check" / "baseline.npy"

    # Initialize live hardware session state
    if "live_baseline_env" not in st.session_state:
        st.session_state["live_baseline_env"] = None
    if "live_rx_audio" not in st.session_state:
        st.session_state["live_rx_audio"] = None
    if "live_ac_result" not in st.session_state:
        st.session_state["live_ac_result"] = None
    if "live_band_curve" not in st.session_state:
        st.session_state["live_band_curve"] = None

    # TOP CONTROL DECK
    with st.container():
        st.markdown('<div class="control-deck">', unsafe_allow_html=True)
        st.markdown("#### Real Acoustic Hardware & Physical Sonar Transceiver (Feature 8)")
        st.caption(
            "Feature 8 turns your physical PC/laptop into an active ultrasonic/acoustic sonar transceiver. "
            "Linear frequency chirps (2–8 kHz) are transmitted through your real speakers, reflections captured by the microphone, "
            "aligned to direct-path acoustic arrival (t = 0), and room/chassis clutter subtracted to isolate real physical boundaries."
        )

        hw_source_mode = st.radio(
            "Acoustic Hardware Operating Mode",
            [
                "Archived Physical Benchmark (Concrete Wall @ 1.797 m, 6.75 σ)",
                "Live Physical Hardware Test (Speaker & Microphone Transceiver)",
            ],
            horizontal=True,
        )

        if hw_source_mode == "Archived Physical Benchmark (Concrete Wall @ 1.797 m, 6.75 σ)":
            if rx_file.exists() and base_file.exists():
                raw_rx = np.load(rx_file)
                base_env = np.load(base_file)

                hw_c1, hw_c2, hw_c3 = st.columns(3)
                with hw_c1:
                    ac_speed = st.number_input("Speed of Sound in Room (m/s)", min_value=320.0, max_value=360.0, value=343.0, step=0.5, key="arch_speed")
                with hw_c2:
                    min_range = st.slider("Min Detection Window (m)", min_value=0.5, max_value=2.5, value=1.0, step=0.1, key="arch_min_r")
                    st.caption("Rejects initial speaker chassis ringing swamping the microphone.")
                with hw_c3:
                    max_range = st.slider("Max Detection Window (m)", min_value=3.0, max_value=8.0, value=5.0, step=0.5, key="arch_max_r")

                ac_result = analyse_recording(
                    raw_rx, chirp_sig, frame_samples, ac_fs,
                    baseline_envelope=base_env, speed_mps=ac_speed,
                    min_range_m=min_range, max_range_m=max_range,
                )

                wall_dist = ac_result["distance_m"]
                wall_sigma = ac_result["quality_sigma"]
                excess_env = ac_result["excess"]
                stacked_env = ac_result["envelope"]

                # Telemetry KPI Grid
                st.markdown(
                    textwrap.dedent(f"""
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
                    """),
                    unsafe_allow_html=True,
                )

                # Audio Player for the Real Recording
                st.markdown(
                    """
                    <div class="audio-card" style="margin-top: 14px;">
                        <div class="audio-title">Physical Benchmark Probe Audio (Transmitted Burst)</div>
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
                ac_result = None

        else:
            # LIVE HARDWARE TRANSCEIVER MODE
            st.markdown("##### Live PC Acoustic Hardware Configuration")

            live_c1, live_c2, live_c3, live_c4 = st.columns(4)
            with live_c1:
                ac_speed = st.number_input("Room Sound Speed (m/s)", min_value=320.0, max_value=360.0, value=343.0, step=0.5, key="live_speed")
            with live_c2:
                live_repeats = st.selectbox("Transmit Burst Repeats", [8, 16], index=0, help="8 repeats takes ~2.2s; 16 repeats takes ~4.4s for higher SNR integration.", key="live_rep")
            with live_c3:
                min_range = st.slider("Min Detection Window (m)", min_value=0.5, max_value=2.5, value=1.0, step=0.1, help="Rejects direct chassis acoustic leakage.", key="live_min_r")
            with live_c4:
                max_range = st.slider("Max Detection Window (m)", min_value=2.5, max_value=8.0, value=5.0, step=0.5, key="live_max_r")

            # 2-Step Live Action Deck
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            act_c1, act_c2 = st.columns(2)

            with act_c1:
                st.markdown(
                    """
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 16px;">
                        <h5 style="color: #38bdf8; margin: 0 0 8px 0;">Step 1: Open-Space Baseline Calibration</h5>
                        <p style="font-size: 0.98rem; color: #94a3b8; margin-bottom: 12px; line-height: 1.5;">
                            Point your laptop into open room air (at least 3–4 meters clearance). Keep volume around 60–75%.
                            This records speaker chassis ringing so it can be subtracted from the measurement.
                        </p>
                    """,
                    unsafe_allow_html=True,
                )
                b_col1, b_col2 = st.columns([1.2, 1])
                with b_col1:
                    if st.button("Record Room Baseline", key="btn_rec_base"):
                        try:
                            with st.spinner(f"Transmitting {live_repeats} chirps & recording room baseline ({live_repeats * 0.26:.1f}s)..."):
                                rec_base = record_live_acoustic(chirp_sig, repeats=live_repeats, fs=ac_fs, amplitude=0.4)
                                base_res = analyse_recording(rec_base, chirp_sig, frame_samples, ac_fs, baseline_envelope=None)
                                st.session_state["live_baseline_env"] = base_res["envelope"]
                                st.toast("Open-space baseline recorded and calibrated successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Audio recording failed: {e}. Check microphone permissions.")
                with b_col2:
                    if base_file.exists() and st.button("Use Archived Baseline", key="btn_use_arch_base"):
                        st.session_state["live_baseline_env"] = np.load(base_file)
                        st.toast("Loaded archived baseline reference!")
                        st.rerun()

                base_status = "READY" if st.session_state["live_baseline_env"] is not None else "NOT RECORDED"
                base_color = "#34d399" if base_status == "READY" else "#fbbf24"
                st.markdown(f'<div style="margin-top: 10px; font-size: 0.92rem; color: #cbd5e1;">Baseline Status: <strong style="color: {base_color};">{base_status}</strong></div></div>', unsafe_allow_html=True)

            with act_c2:
                st.markdown(
                    """
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 16px;">
                        <h5 style="color: #38bdf8; margin: 0 0 8px 0;">Step 2: Physical Echo Distance Measurement</h5>
                        <p style="font-size: 0.98rem; color: #94a3b8; margin-bottom: 12px; line-height: 1.5;">
                            Point your laptop speaker & mic directly facing a solid wall, closed door, or large object (1 to 5 meters away).
                            Do not change the system volume.
                        </p>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Transmit Chirps & Measure Echo", key="btn_rec_target"):
                    try:
                        active_baseline = st.session_state.get("live_baseline_env")
                        if active_baseline is None and base_file.exists():
                            active_baseline = np.load(base_file)
                            st.info("No live baseline recorded yet — using archived reference baseline.")

                        with st.spinner(f"Transmitting {live_repeats} probing chirps & listening for echoes ({live_repeats * 0.26:.1f}s)..."):
                            rec_target = record_live_acoustic(chirp_sig, repeats=live_repeats, fs=ac_fs, amplitude=0.4)
                            res = analyse_recording(
                                rec_target, chirp_sig, frame_samples, ac_fs,
                                baseline_envelope=active_baseline, speed_mps=ac_speed,
                                min_range_m=min_range, max_range_m=max_range,
                            )
                            st.session_state["live_rx_audio"] = rec_target
                            st.session_state["live_ac_result"] = res
                            st.toast(f"Physical echo detected at {res['distance_m']:.3f} m ({res['quality_sigma']:.1f}σ)!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Live measurement failed: {e}. Ensure speaker and microphone are enabled.")

                st.markdown('<div style="margin-top: 10px; font-size: 0.92rem; color: #94a3b8;">Captures live reflections via sounddevice <code>sd.playrec()</code></div></div>', unsafe_allow_html=True)

            # Check if live result exists
            if st.session_state["live_ac_result"] is not None:
                ac_result = st.session_state["live_ac_result"]
                wall_dist = ac_result["distance_m"]
                wall_sigma = ac_result["quality_sigma"]
                excess_env = ac_result["excess"]
                stacked_env = ac_result["envelope"]
                base_env = st.session_state.get("live_baseline_env")
                if base_env is None and base_file.exists():
                    base_env = np.load(base_file)

                # Telemetry KPI Grid
                sig_label = "Confirmed Detection" if wall_sigma >= 6.0 else "Weak Echo (Sub-Threshold)"
                sig_color = "val-emerald" if wall_sigma >= 6.0 else "val-amber"

                st.markdown(
                    textwrap.dedent(f"""
                    <div class="stat-grid" style="margin-top: 18px;">
                        <div class="stat-box">
                            <div class="stat-val val-cyan">Live Room Echo</div>
                            <div class="stat-label">Measurement Mode</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val val-emerald">{wall_dist:.3f} m</div>
                            <div class="stat-label">Detected Distance</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val {sig_color}">{wall_sigma:.2f} σ</div>
                            <div class="stat-label">{sig_label}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val val-indigo">48.0 kHz</div>
                            <div class="stat-label">Live Sampling Rate</div>
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

                # Audio Player for Live Recorded Audio
                if st.session_state.get("live_rx_audio") is not None:
                    st.markdown(
                        """
                        <div class="audio-card" style="margin-top: 14px;">
                            <div class="audio-title">Captured Microphone Audio (Live Acoustic Recording)</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    try:
                        live_wav = to_wav_bytes(st.session_state["live_rx_audio"], int(ac_fs))
                        st.audio(live_wav, format="audio/wav")
                    except Exception as e:
                        st.caption(f"Audio playback note: {e}")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                ac_result = None
                st.info("Click **'Transmit Chirps & Measure Echo'** above to test your laptop's real speaker and microphone in your room.")

        st.markdown("</div>", unsafe_allow_html=True)

    # FULL-WIDTH HIGH SIGNAL PLOTS (Below all controls)
    if ac_result is not None:
        st.markdown("### Real Acoustic Traces & Clutter Cancellation")

        dist_axis = np.arange(len(excess_env)) * ac_speed / ac_fs / 2.0
        range_mask = (dist_axis >= 0) & (dist_axis <= max_range + 0.5)

        # Plot 1: Full-Width Stacked Envelope with Baseline Overlay
        fig_ac1 = create_dark_figure(height=390, x_title="Distance (m)", y_title="Normalized Envelope")
        fig_ac1.add_trace(
            go.Scatter(
                x=dist_axis[range_mask], y=stacked_env[range_mask],
                mode="lines", line=dict(color="#38bdf8", width=1.8),
                name="Physical Recording Envelope (Target + Clutter)",
                hovertemplate="Dist: %{x:.3f} m<br>Target Env: %{y:.4f}<extra></extra>",
            )
        )
        if base_env is not None and len(base_env) == len(dist_axis):
            fig_ac1.add_trace(
                go.Scatter(
                    x=dist_axis[range_mask], y=base_env[range_mask],
                    mode="lines", line=dict(color="#c084fc", width=1.6, dash="dash"),
                    name="Open-Space Baseline (Clutter Only)",
                    hovertemplate="Dist: %{x:.3f} m<br>Baseline: %{y:.4f}<extra></extra>",
                )
            )
        fig_ac1.add_vline(
            x=0, line=dict(color="#38bdf8", width=1.8),
            annotation_text="Direct Arrival (t=0)", annotation_position="top right",
            annotation_font=dict(size=10, color="#38bdf8"),
        )
        render_plot_header(
            "STAGE 1",
            "Direct-Path Aligned Stacked Correlation Trace & Clutter Baseline Overlay",
            "Direct-arrival synchronized acoustic correlation trace plotted alongside background clutter baseline.",
        )
        fig_ac1.update_layout(
            title=dict(text="<b>Direct-Path Aligned Correlation & Clutter Baseline Overlay</b>", font=dict(size=14, color="#f1f5f9")),
        )
        render_chart(fig_ac1)

        render_plot_divider()

        # Plot 2: Full-Width Clutter-Cancelled Excess Return
        render_plot_header(
            "STAGE 2",
            "Clutter-Cancelled Excess Return & Peak Echo Detection",
            f"Physical acoustic reflection isolated via background subtraction, confirming target reflection at {wall_dist:.3f} m ({wall_sigma:.1f}σ).",
        )
        fig_ac2 = create_dark_figure(height=420, x_title="Distance (m)", y_title="Excess Amplitude")
        fig_ac2.add_trace(
            go.Scatter(
                x=dist_axis[range_mask], y=excess_env[range_mask],
                mode="lines", line=dict(color="#38bdf8", width=2.2),
                name="Excess Return (Target − Baseline)",
                hovertemplate="Dist: %{x:.3f} m<br>Excess: %{y:.4f}<extra></extra>",
            )
        )
        fig_ac2.add_trace(
            go.Scatter(
                x=[wall_dist], y=[excess_env[ac_result["peak_index"]]],
                mode="markers+text",
                marker=dict(color="#34d399", size=12, line=dict(color="#ffffff", width=1.8)),
                text=[f"Peak @ {wall_dist:.3f} m ({wall_sigma:.1f}σ)"],
                textposition="top center",
                textfont=dict(color="#34d399", size=11, family="JetBrains Mono"),
                name="Detected Object",
            )
        )
        fig_ac2.add_vrect(
            x0=min_range, x1=max_range,
            fillcolor="rgba(56, 189, 248, 0.08)", layer="below", line_width=1.2,
            line=dict(color="rgba(56, 189, 248, 0.4)", dash="dash"),
            annotation_text="Search Window", annotation_position="top left",
            annotation_font=dict(size=11, color="#38bdf8"),
        )
        fig_ac2.update_layout(
            title=dict(text="<b>Clutter-Cancelled Excess Return: Confirmed Physical Reflection</b>", font=dict(size=14, color="#f1f5f9")),
        )
        render_chart(fig_ac2)

    # Hardware Frequency Response Analyzer & Terminal CLI Guide
    with st.expander("Hardware Frequency Response Analysis & Terminal CLI Guide"):
        st.markdown(
            """
            `physical_sonar_check/band_check.py` plays a 2–22 kHz linear chirp to determine the true hardware passband
            of your computer's built-in speaker and microphone.
            """
        )

        b_btn_col1, b_btn_col2 = st.columns([1.5, 2])
        with b_btn_col1:
            if st.button("Run Live Audio Passband Sweep (2-22 kHz)", key="btn_run_band_live"):
                try:
                    with st.spinner("Playing 0.5s sweep (2-22 kHz) and recording response..."):
                        b_freqs, b_dbs = run_live_band_check(fs=ac_fs)
                        st.session_state["live_band_curve"] = (b_freqs, b_dbs)
                        st.toast("Live electroacoustic passband measured!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Band sweep failed: {e}")

        # Choose curve to display
        if st.session_state.get("live_band_curve") is not None:
            band_freqs, band_resp_db = st.session_state["live_band_curve"]
            trace_title = "Live Measured Hardware Response H_hw(f)"
        else:
            band_freqs = np.array([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5])
            band_resp_db = np.array([-8.2, -3.1, -1.2, 0.0, -1.8, -4.5, -9.8, -14.2, -18.5, -22.1, -25.4, -28.9, -32.5, -36.1, -39.0, -42.5, -45.0, -48.2, -50.0, -52.0])
            trace_title = "Benchmark Hardware Response H_hw(f) (from band_check.py)"

        fig_hw = create_dark_figure(height=380, x_title="Frequency (kHz)", y_title="Relative Response (dB)")
        fig_hw.add_trace(
            go.Scatter(
                x=band_freqs, y=band_resp_db,
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.4),
                marker=dict(color="#38bdf8", size=8),
                name=trace_title,
                hovertemplate="Freq: %{x:.1f} kHz<br>Response: %{y:.1f} dB<extra></extra>",
            )
        )
        fig_hw.add_hline(
            y=-12.0, line=dict(color="#f43f5e", width=1.8, dash="dot"),
            annotation_text="-12 dB Usability Threshold", annotation_position="bottom right",
            annotation_font=dict(size=11, color="#f43f5e"),
        )
        fig_hw.add_vrect(
            x0=2.0, x1=8.0,
            fillcolor="rgba(52, 211, 153, 0.14)", layer="below", line_width=1.2,
            line=dict(color="#34d399", dash="dot"),
            annotation_text="Chosen 2 - 8 kHz Operating Chirp Band", annotation_position="top left",
            annotation_font=dict(size=11, color="#34d399"),
        )
        fig_hw.update_layout(
            title=dict(text=f"<b>Electroacoustic Hardware Transfer Function (2–22 kHz)</b>", font=dict(size=14, color="#f1f5f9")),
        )
        render_chart(fig_hw)

        st.caption(
            "Insight: Frequencies below 2 kHz suffer severe chassis high-pass rolloff, while frequencies above 8 kHz drop beyond -12 dB. "
            "Operating between 2 kHz and 8 kHz maximizes energy transmission and signal-to-noise ratio in ambient room conditions."
        )

        st.markdown("---")
        st.markdown("##### Direct PowerShell / Terminal Command Line Execution")
        st.markdown(
            """
            If you prefer running tests from the command line outside Streamlit, use the dedicated scripts in `physical_sonar_check/`:
            
            ```powershell
            # 1. Measure speaker + microphone frequency passband:
            python physical_sonar_check/band_check.py

            # 2. Record the empty-room reference baseline (point laptop into open space with >3m clearance):
            python physical_sonar_check/record_acoustic.py baseline

            # 3. Point laptop facing a solid wall or obstacle and measure physical distance:
            python physical_sonar_check/record_acoustic.py
            ```
            """
        )
