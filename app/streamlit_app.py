"""
Sonar Distance Simulator — interactive front end.
CSE220 Signals and Systems — Team SevenEight
Aditya Tirtho Roy (2305178) & Farhan Ehsas Sami (2305177)

Architectural rule: this file imports pure functions from src/ and renders
them. No simulation maths is duplicated here.

Layout follows the redesign canvas: a sticky top bar, a home screen, and four
working screens. Each working screen is a stage grid on the left and a fixed
right-hand column holding the reading, the settings and the audio.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import hilbert

from src.acoustic import analyse_recording, design_chirp
from src.audio import audible_echo, make_audible, to_wav_bytes
from src.channel import add_noise, simulate_channel, simulate_multi_object_channel
from src.evaluate import (
    describe_pulse_config,
    run_bandwidth_resolution_sweep,
    run_snr_sweep,
)
from src.filters import bandpass_filter
from src.pulse import generate_pulse, pulse_bandwidth_hz
from src.receiver import estimate_distance, estimate_multiple_distances, matched_filter

# ==============================================================================
# DESIGN TOKENS
# ==============================================================================

L0, L1, L2, L3 = "#070b14", "#0f172a", "#0b1222", "#1e293b"
ACCENT, COMPARE, THRESH = "#38bdf8", "#818cf8", "#fbbf24"
GOOD, BAD = "#34d399", "#f87171"
INK, INK_2, INK_MUTED = "#e2e8f0", "#cbd5e1", "#94a3b8"
HAIRLINE = "rgba(148,163,184,.12)"

TRACE_RAW = dict(color=INK_MUTED, width=1.25)
TRACE_PAYOFF = dict(color=ACCENT, width=1.6)
TRACE_COMPARE = dict(color=COMPARE, width=1.5)
LINE_THRESH = dict(color=THRESH, width=1, dash="dash")

SCREENS = [
    ("m1", "01", "Send a Pulse, Find the Echo", "Single target"),
    ("m2", "02", "Two Targets — Can We Separate Them?", "Range resolution"),
    ("m3", "03", "How Much Noise Before It Fails?", "Monte Carlo sweep"),
    ("m4", "04", "Real Sound Test (Speaker + Mic)", "Laptop hardware"),
]
SCREEN_TITLE = {key: title for key, _, title, _ in SCREENS}

st.set_page_config(page_title="Sonar Distance Simulator", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@500;600&display=swap');

    [data-testid="stAppViewContainer"]{background:#070b14}
    header[data-testid="stHeader"]{display:none}
    .block-container{padding:0 24px 32px 24px!important;max-width:100%!important}
    html,body,[class*="css"]{font-family:Inter,system-ui,sans-serif;color:#e2e8f0}

    /* Panels are Streamlit's own bordered containers, styled in place -- never
       a <div> opened in one st.markdown and closed in another. */
    [data-testid="stVerticalBlockBorderWrapper"]{
        background:#0f172a;border:1px solid rgba(148,163,184,.10)!important;
        border-radius:10px;padding:16px 18px}
    .st-key-payoff{border-color:rgba(56,189,248,.45)!important}

    .st-key-topbar{position:sticky;top:0;z-index:99;background:#070b14;
        border-bottom:1px solid rgba(148,163,184,.14);padding:6px 0 4px 0!important;
        border-radius:0!important;margin-bottom:14px}
    .st-key-topbar [data-testid="stVerticalBlockBorderWrapper"]{
        background:none;border:0!important;padding:0}
    .st-key-topbar button{border:0!important;border-radius:0!important;
        background:none!important;font-size:13.5px!important;padding:6px 2px!important}
    .st-key-topbar button[kind="tertiary"]{color:#94a3b8!important}
    .st-key-topbar button[kind="tertiary"]:hover{color:#e2e8f0!important}
    .st-key-topbar button[kind="primary"]{color:#f1f5f9!important;
        border-bottom:2px solid #38bdf8!important}

    [data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;font-weight:500;
        font-size:20px}
    [data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:11.5px!important}
    .st-key-hero [data-testid="stMetricValue"]{font-size:44px;line-height:1.05}

    [data-testid="stPlotlyChart"]{border-radius:6px;overflow:hidden}
    [data-testid="stExpander"] details{border:0;
        border-top:1px solid rgba(148,163,184,.12);border-radius:0;background:none}
    [data-testid="stExpander"] summary{font-size:12.5px;color:#94a3b8}

    .eyebrow{font:500 11px 'JetBrains Mono',monospace;letter-spacing:.08em;
        text-transform:uppercase;color:#94a3b8;margin:0}
    .ptitle{font:600 15px Outfit,sans-serif;color:#f1f5f9;margin:2px 0 0}
    .pdesc{font-size:12.5px;color:#94a3b8;margin:2px 0 0;line-height:1.5}
    .screen-title{font:600 26px/1.2 Outfit,sans-serif;color:#f8fafc;margin:0}
    .brand{font:600 15px Outfit,sans-serif;color:#f1f5f9;margin:0;padding-top:6px}
    .brand span{color:#38bdf8}
    .empty-state{border:1px dashed rgba(148,163,184,.3);border-radius:8px;
        padding:28px 20px;text-align:center;color:#94a3b8;font-size:13px;line-height:1.6}
    .empty-state b{display:block;font:600 15px Outfit,sans-serif;color:#e2e8f0;
        margin-bottom:6px}
    .stale{color:#fbbf24;font-size:12.5px;margin:0}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CACHED AUDIO BUILDERS
# ==============================================================================
# Streamlit reruns the whole script on every widget interaction. Without a cache
# each rerun rebuilds seconds of audio and restarts whatever is playing.
#
# LOUDNESS_REFERENCE_SNR_DB is the quietest SNR the UI offers. Normalising every
# echo clip to that worst case keeps the ping at a constant level while the hiss
# grows -- the whole point of listening to an SNR sweep. Per-clip normalisation
# would make -20 dB and +30 dB equally loud; taking the gain from a clean clip
# would clip the noisy ones.

LOUDNESS_REFERENCE_SNR_DB = -20.0


@st.cache_data(show_spinner=False)
def cached_transmit_wav(pulse_type, duration_s, fs, freq_hz, bandwidth_hz, slowdown):
    pulse = generate_pulse(pulse_type, duration_s, fs, freq_hz=freq_hz,
                           bandwidth_hz=bandwidth_hz)
    clip = make_audible(pulse, fs=fs, repeats=4, slowdown=slowdown, gap_s=0.5)
    return to_wav_bytes(clip["audio"], clip["rate_hz"])


@st.cache_data(show_spinner=False)
def cached_echo_wav(pulse_type, duration_s, fs, freq_hz, bandwidth_hz,
                    distance_m, snr_db, speed_mps, slowdown):
    pulse = generate_pulse(pulse_type, duration_s, fs, freq_hz=freq_hz,
                           bandwidth_hz=bandwidth_hz)
    loudest = audible_echo(pulse, distance_m, fs=fs, snr_db=LOUDNESS_REFERENCE_SNR_DB,
                           speed_mps=speed_mps, slowdown=slowdown, frames=4)
    clip = audible_echo(pulse, distance_m, fs=fs, snr_db=snr_db, speed_mps=speed_mps,
                        slowdown=slowdown, frames=4, gain=loudest["gain"])
    return to_wav_bytes(clip["audio"], clip["rate_hz"])


@st.cache_data(show_spinner=False)
def cached_scene_wav(pulse_type, duration_s, fs, freq_hz, bandwidth_hz,
                     distances_m, attenuations, snr_db, speed_mps,
                     buffer_duration_s, slowdown):
    pulse = generate_pulse(pulse_type, duration_s, fs, freq_hz=freq_hz,
                           bandwidth_hz=bandwidth_hz)
    clean = simulate_multi_object_channel(
        pulse, list(distances_m), fs, attenuations=list(attenuations),
        speed_mps=speed_mps, buffer_duration_s=buffer_duration_s,
    )
    # Fresh noise per repeat: repeating one noisy buffer makes the hiss cycle
    # audibly at the loop rate, which sounds obviously synthetic.
    frames = [add_noise(clean, snr_db, seed=42 + offset) for offset in range(4)]
    clip = make_audible(frames, fs=fs, slowdown=slowdown, gap_s=0.0)
    return to_wav_bytes(clip["audio"], clip["rate_hz"])


# ==============================================================================
# SHARED RENDERING HELPERS
# ==============================================================================


def base_figure(height: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height, autosize=True, dragmode=False,
        paper_bgcolor=L2, plot_bgcolor=L2,
        font=dict(family="Inter", size=11, color=INK_MUTED),
        margin=dict(l=48, r=14, t=10, b=34),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=L3, font_family="JetBrains Mono", font_size=11),
        legend=dict(orientation="h", x=0, y=1.14, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10.5)),
        xaxis=dict(gridcolor=HAIRLINE, zeroline=False, fixedrange=True,
                   tickfont=dict(family="JetBrains Mono", size=10.5)),
        yaxis=dict(gridcolor=HAIRLINE, zeroline=False, fixedrange=True,
                   tickfont=dict(family="JetBrains Mono", size=10.5)),
    )
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False, "responsive": True})


def panel_header(eyebrow: str, title: str, description: str = "") -> None:
    """One self-contained markdown block -- no split divs."""
    desc = f'<p class="pdesc">{description}</p>' if description else ""
    st.markdown(
        f'<p class="eyebrow">{eyebrow}</p><p class="ptitle">{title}</p>{desc}',
        unsafe_allow_html=True,
    )


def empty_state(title: str, line: str) -> None:
    st.markdown(f'<div class="empty-state"><b>{title}</b>{line}</div>',
                unsafe_allow_html=True)


def goto(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


# ==============================================================================
# TOP BAR + HOME
# ==============================================================================

if "page" not in st.session_state:
    st.session_state["page"] = "home"
page = st.session_state["page"]


def render_topbar(active: str) -> None:
    with st.container(key="topbar"):
        cols = st.columns([2.0, 2.0, 2.6, 2.4, 2.4], vertical_alignment="center")
        with cols[0]:
            if st.button("SONAR · Distance Simulator", key="nav_home",
                         type="tertiary", width="stretch"):
                goto("home")
        for col, (key, number, title, _) in zip(cols[1:], SCREENS):
            with col:
                if st.button(f"{number}  {title}", key=f"nav_{key}",
                             type="primary" if key == active else "tertiary",
                             width="stretch"):
                    goto(key)


def render_home() -> None:
    left, centre, right = st.columns([1, 2.4, 1])
    with centre:
        st.markdown(
            """
            <div style="padding:56px 0 24px 0">
              <p class="eyebrow">CSE220 · Signals and Systems project</p>
              <h1 style="font:600 44px/1.1 Outfit,sans-serif;color:#f8fafc;
                         margin:10px 0 0;letter-spacing:-.01em">Sonar Distance Simulator</h1>
              <p style="font-size:15px;line-height:1.65;color:#cbd5e1;margin:14px 0 0;
                        max-width:620px">
                Sends a known sound pulse, listens for the echo, and works out how far
                away an object is from the round-trip time — the principle behind sonar,
                radar and LiDAR.
              </p>
              <div style="background:#0b1222;border-radius:8px;padding:18px 20px;margin-top:26px">
                <svg width="100%" height="74" viewBox="0 0 620 74" fill="none"
                     xmlns="http://www.w3.org/2000/svg" role="img"
                     aria-label="A pulse is sent, reflects off an object, and returns after a round-trip time">
                  <line x1="14" y1="30" x2="300" y2="30" stroke="#38bdf8" stroke-width="1.6"/>
                  <polygon points="300,30 292,26 292,34" fill="#38bdf8"/>
                  <line x1="300" y1="44" x2="14" y2="44" stroke="#94a3b8"
                        stroke-width="1.4" stroke-dasharray="5 4"/>
                  <polygon points="14,44 22,40 22,48" fill="#94a3b8"/>
                  <rect x="306" y="14" width="10" height="46" rx="2" fill="#475569"/>
                  <text x="14" y="18" fill="#94a3b8"
                        style="font:400 11px Inter,sans-serif">pulse sent</text>
                  <text x="14" y="64" fill="#94a3b8"
                        style="font:400 11px Inter,sans-serif">echo returns</text>
                  <text x="352" y="30" fill="#cbd5e1"
                        style="font:500 12px 'JetBrains Mono',monospace">&#916;t  round-trip time</text>
                  <text x="352" y="52" fill="#38bdf8"
                        style="font:500 13px 'JetBrains Mono',monospace">d = c &#183; &#916;t / 2</text>
                </svg>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Get Started", key="home_start", type="primary"):
            goto("m1")
        st.caption("Starts with the single-target demonstration.")
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        for key, number, title, desc in SCREENS:
            if st.button(f"{number}    {title}    ·  {desc}", key=f"home_{key}",
                         type="tertiary", width="stretch"):
                goto(key)


def stage_grid(screen_key: str, stages: list[dict], payoff_index: int) -> None:
    """
    One large stage beside the others, with the focus swappable.

    Each stage is a render function, so moving one between the big and small
    slots never duplicates code and never touches src/.
    """
    focus = st.session_state.get(f"focus_{screen_key}", payoff_index)
    focus = min(focus, len(stages) - 1)
    big_col, small_col = st.columns([3, 2], gap="small")

    for index, stage in enumerate(stages):
        focused = index == focus
        target = big_col if focused else small_col
        with target:
            key = "payoff" if index == payoff_index else f"{screen_key}_stage{index}"
            with st.container(border=True, key=key):
                panel_header(stage["eyebrow"], stage["title"],
                             stage["description"] if focused else "")
                stage["draw"](340 if focused else 150)
                if not focused:
                    if st.button("Enlarge", key=f"focus_{screen_key}_{index}",
                                 type="tertiary"):
                        st.session_state[f"focus_{screen_key}"] = index
                        st.rerun()


def clamp_to_nyquist(freq_hz: float, bandwidth_hz: float | None, fs: float):
    """
    Keep the carrier and the chirp sweep physically valid.

    The carrier and sweep are chosen before the sampling rate is known, so a low
    fs can leave either above Nyquist or sweeping below 0 Hz. Clamping here
    beats letting generate_pulse raise a traceback at the viewer.
    """
    nyquist_hz = fs / 2.0
    notes = []
    if freq_hz > nyquist_hz * 0.95:
        freq_hz = round(nyquist_hz * 0.95, 1)
        notes.append(f"carrier lowered to {freq_hz:.0f} Hz (Nyquist is {nyquist_hz:.0f} Hz)")
    if bandwidth_hz is not None:
        widest = 2.0 * min(freq_hz, nyquist_hz - freq_hz)
        if bandwidth_hz > widest:
            bandwidth_hz = max(100.0, round(widest, 1))
            notes.append(f"chirp sweep narrowed to {bandwidth_hz:.0f} Hz")
    return freq_hz, bandwidth_hz, notes


# ==============================================================================
# SCREEN 1 — SEND A PULSE, FIND THE ECHO
# ==============================================================================


def render_screen_1() -> None:
    st.markdown(f'<p class="screen-title">{SCREEN_TITLE["m1"]}</p>',
                unsafe_allow_html=True)
    st.caption("One pulse goes out, one echo comes back. The matched filter finds it "
               "even when the recording looks like pure noise.")

    main, side = st.columns([1, 0.42], gap="small")

    # Controls first: the stages need their values before anything can be drawn.
    with side:
        with st.container(key="side_m1"):
            results_slot = st.container(key="results")
            settings_tab, audio_tab = st.tabs(["Settings", "Audio"])

            with settings_tab:
                true_dist = st.slider("Target distance (m)", 0.5, 25.0, 10.0, 0.1)
                snr_db = st.slider("Noise level (SNR, dB)", -20.0, 30.0, 10.0, 1.0)
                pulse_type = st.selectbox("Pulse shape", ["gaussian", "chirp", "rect"])
                duration_ms = st.slider("Pulse length (ms)", 0.5, 20.0, 2.0, 0.5)
                freq_hz = st.slider("Carrier frequency (Hz)", 500.0, 12000.0, 4000.0, 250.0)
                bandwidth_hz = None
                if pulse_type == "chirp":
                    bandwidth_hz = st.slider("Chirp sweep width (Hz)", 500.0, 6000.0,
                                             2500.0, 250.0)

                with st.expander("Advanced"):
                    speed_mps = st.number_input("Speed of sound (m/s)", 100.0, 2000.0,
                                                343.0, 1.0)
                    fs = float(st.selectbox("Sampling rate (Hz)", [16000, 24000, 48000, 96000],
                                            index=2))
                    attenuation = st.slider("Echo strength", 0.1, 1.0, 1.0, 0.05)
                    rng_seed = st.number_input("Noise seed", 0, 9999, 42)
                    slowdown = st.select_slider("Audio slowdown (x slower)",
                                                options=[2.0, 4.0, 8.0], value=4.0)
                    use_bandpass = st.checkbox("Band-pass filter before detecting")
                    bp_order = st.slider("Filter steepness", 1, 8, 4) if use_bandpass else 4

    freq_hz, bandwidth_hz, notes = clamp_to_nyquist(freq_hz, bandwidth_hz, fs)
    duration_s = duration_ms / 1000.0
    buffer_duration_s = max(0.06, (2 * true_dist / speed_mps) * 1.35)

    pulse = generate_pulse(pulse_type, duration_s, fs, freq_hz=freq_hz,
                           bandwidth_hz=bandwidth_hz)
    measured_bw = pulse_bandwidth_hz(pulse, fs)
    clean_rx = simulate_channel(pulse, true_dist, fs=fs, speed_mps=speed_mps,
                                attenuation=attenuation,
                                buffer_duration_s=buffer_duration_s)
    noisy_rx = add_noise(clean_rx, snr_db, seed=int(rng_seed))

    rx_for_detection = noisy_rx
    if use_bandpass:
        span = measured_bw if measured_bw > 0 else 1000.0
        f_low = max(50.0, freq_hz - span * 1.2)
        f_high = min(fs / 2 - 100.0, freq_hz + span * 1.2)
        rx_for_detection = bandpass_filter(noisy_rx, fs, f_low, f_high, order=bp_order)

    est_dist, correlation = estimate_distance(rx_for_detection, pulse, fs=fs,
                                              speed_mps=speed_mps)
    error_m = abs(est_dist - true_dist)
    round_trip_ms = (2 * true_dist / speed_mps) * 1000.0

    # ---- stages -------------------------------------------------------------
    def draw_pulse(height):
        view = st.radio("View", ["Waveform", "Spectrum"], horizontal=True,
                        label_visibility="collapsed", key="m1_view")
        fig = base_figure(height)
        if view == "Waveform":
            fig.add_scatter(x=np.arange(pulse.size) / fs * 1000, y=pulse,
                            mode="lines", line=TRACE_RAW, name="pulse",
                            hovertemplate="%{x:.2f} ms<extra></extra>")
            fig.update_xaxes(title_text="Time (ms)")
        else:
            n_fft = 8192
            power = np.abs(np.fft.rfft(pulse, n=n_fft)) ** 2
            freqs = np.fft.rfftfreq(n_fft, 1 / fs)
            keep = freqs <= min(fs / 2, freq_hz * 3 + 2000)
            db = 10 * np.log10(power[keep] / power.max() + 1e-12)
            fig.add_scatter(x=freqs[keep] / 1000, y=db, mode="lines", line=TRACE_RAW,
                            name="spectrum", hovertemplate="%{x:.2f} kHz<extra></extra>")
            fig.add_hline(y=-3, line=LINE_THRESH,
                          annotation_text=f"−3 dB · {measured_bw:.0f} Hz wide",
                          annotation_font=dict(color=THRESH, size=10.5))
            fig.update_xaxes(title_text="Frequency (kHz)")
            fig.update_yaxes(range=[-45, 4])
        show_chart(fig)

    def draw_received(height):
        fig = base_figure(height)
        fig.add_scatter(x=np.arange(noisy_rx.size) / fs * 1000, y=noisy_rx,
                        mode="lines", line=TRACE_RAW, name="received",
                        hovertemplate="%{x:.1f} ms<extra></extra>")
        echo_ms = round_trip_ms
        fig.add_vrect(x0=echo_ms, x1=echo_ms + duration_ms,
                      fillcolor=ACCENT, opacity=0.09, line_width=0)
        fig.update_xaxes(title_text="Time (ms)")
        show_chart(fig)

    def draw_detection(height):
        distances = (np.arange(correlation.size) - (pulse.size - 1)) / fs * speed_mps / 2
        keep = (distances >= 0) & (distances <= buffer_duration_s * speed_mps / 2)
        fig = base_figure(height)
        fig.add_scatter(x=distances[keep], y=correlation[keep], mode="lines",
                        line=TRACE_PAYOFF, name="match strength",
                        hovertemplate="%{x:.3f} m<extra></extra>")
        fig.add_vline(x=est_dist, line=dict(color=ACCENT, width=1, dash="dash"),
                      annotation_text=f"peak → {est_dist:.3f} m",
                      annotation_font=dict(color=ACCENT, size=10.5,
                                           family="JetBrains Mono"))
        fig.update_xaxes(title_text="Distance (m)")
        show_chart(fig)

    with main:
        if notes:
            st.markdown(f'<p class="stale">Adjusted to stay physical: {"; ".join(notes)}.</p>',
                        unsafe_allow_html=True)
        stage_grid("m1", [
            dict(eyebrow="Stage 1 · Transmit", title="The pulse we send",
                 description=f"{pulse_type}, {duration_ms:.1f} ms, {measured_bw:.0f} Hz wide.",
                 draw=draw_pulse),
            dict(eyebrow="Stage 2 · Receive", title="What the microphone hears",
                 description=f"The echo is in there at {snr_db:.0f} dB SNR.",
                 draw=draw_received),
            dict(eyebrow="Stage 3 · Detect", title="Where the echo is",
                 description="Matched filter output. The peak marks the target.",
                 draw=draw_detection),
        ], payoff_index=2)

    with results_slot:
        with st.container(key="hero"):
            st.metric("Estimated distance", f"{est_dist:.3f} m",
                      delta=f"{error_m * 100:.2f} cm error", delta_color="off")
        c1, c2 = st.columns(2)
        c1.metric("True distance", f"{true_dist:.2f} m")
        c2.metric("Round-trip", f"{round_trip_ms:.2f} ms")

    with audio_tab:
        st.caption(f"Slowed {slowdown:.0f}× so a {duration_ms:.1f} ms pulse is audible. "
                   "Both clips loop.")
        st.markdown("**The pulse we send**")
        try:
            st.audio(cached_transmit_wav(pulse_type, duration_s, fs, freq_hz,
                                         bandwidth_hz, slowdown),
                     format="audio/wav", loop=True)
        except Exception as exc:
            st.caption(f"Audio unavailable: {exc}")
        st.markdown("**What the microphone hears**")
        try:
            st.audio(cached_echo_wav(pulse_type, duration_s, fs, freq_hz, bandwidth_hz,
                                     true_dist, snr_db, speed_mps, slowdown),
                     format="audio/wav", loop=True)
        except Exception as exc:
            st.caption(f"Audio unavailable: {exc}")


# ==============================================================================
# SCREEN 2 — TWO TARGETS, CAN WE SEPARATE THEM?
# ==============================================================================


def render_screen_2() -> None:
    st.markdown(f'<p class="screen-title">{SCREEN_TITLE["m2"]}</p>',
                unsafe_allow_html=True)
    st.caption("Two objects close together blur into one echo. How close they can get "
               "before that happens is set by the pulse bandwidth.")

    main, side = st.columns([1, 0.42], gap="small")

    with side:
        with st.container(key="side_m2"):
            results_slot = st.container(key="results")
            settings_tab, audio_tab = st.tabs(["Settings", "Audio"])

            with settings_tab:
                base_d = st.slider("Target 1 distance (m)", 1.0, 15.0, 5.0, 0.5)
                separation_cm = st.slider("Target 2 separation (cm)", 1.0, 100.0, 28.0, 1.0)
                att_1 = st.slider("Target 1 reflectivity", 0.1, 1.0, 1.0, 0.05)
                att_2 = st.slider("Target 2 reflectivity", 0.1, 1.0, 0.75, 0.05)
                m_pulse_type = st.selectbox("Pulse shape", ["gaussian", "chirp", "rect"],
                                            key="m2_ptype")
                m_duration_ms = st.slider("Pulse length (ms)", 1.0, 10.0, 2.0, 0.5,
                                          key="m2_dur")
                m_chirp_bw = None
                if m_pulse_type == "chirp":
                    m_chirp_bw = st.slider("Chirp sweep width (Hz)", 500.0, 5000.0,
                                           2500.0, 250.0, key="m2_bw")
                m_snr_db = st.slider("Noise level (SNR, dB)", -10.0, 30.0, 20.0, 2.0,
                                     key="m2_snr")

                with st.expander("Advanced"):
                    add_third = st.checkbox("Add a third target")
                    third_d = st.slider("Target 3 distance (m)", 1.0, 15.0, 7.5, 0.1) \
                        if add_third else None
                    att_3 = st.slider("Target 3 reflectivity", 0.1, 1.0, 0.5, 0.05) \
                        if add_third else None
                    m_speed = st.number_input("Speed of sound (m/s)", 100.0, 2000.0,
                                              343.0, 1.0, key="m2_speed")
                    m_fs = float(st.selectbox("Sampling rate (Hz)", [24000, 48000, 96000],
                                              index=1, key="m2_fs"))
                    m_freq = st.slider("Carrier frequency (Hz)", 1000.0, 10000.0,
                                       4000.0, 250.0, key="m2_fc")
                    prominence_frac = st.slider("Detection sensitivity (lower finds more)",
                                                0.05, 0.8, 0.35, 0.05)
                    pulse_len = int(round((m_duration_ms / 1000.0) * m_fs))
                    min_sep_samples = st.slider("Minimum gap between detections (samples)",
                                                2, max(10, pulse_len),
                                                max(4, pulse_len // 4), 2)
                    m_slowdown = st.select_slider("Audio slowdown (x slower)",
                                                  options=[2.0, 4.0, 8.0], value=4.0,
                                                  key="m2_slow")

    m_freq, m_chirp_bw, notes = clamp_to_nyquist(m_freq, m_chirp_bw, m_fs)
    m_dur_s = m_duration_ms / 1000.0

    m_pulse = generate_pulse(m_pulse_type, m_dur_s, m_fs, freq_hz=m_freq,
                             bandwidth_hz=m_chirp_bw)
    m_bw = pulse_bandwidth_hz(m_pulse, m_fs)
    theory_cm = (m_speed / (2.0 * m_bw)) * 100.0 if m_bw > 0 else float("nan")

    distances = [base_d, base_d + separation_cm / 100.0]
    attenuations = [att_1, att_2]
    if add_third and third_d is not None:
        distances.append(third_d)
        attenuations.append(att_3)

    m_buf_dur = max(0.08, (2 * max(distances) / m_speed) * 1.3)
    clean_scene = simulate_multi_object_channel(
        m_pulse, distances, m_fs, attenuations=attenuations,
        speed_mps=m_speed, buffer_duration_s=m_buf_dur)
    # Always noise the scene -- this used to be skipped at exactly 30 dB, the
    # slider's maximum, so the top of the range was silently noiseless.
    m_buffer = add_noise(clean_scene, m_snr_db, seed=42)

    detected = estimate_multiple_distances(
        m_buffer, m_pulse, m_fs, speed_mps=m_speed,
        prominence_frac=prominence_frac, min_separation_samples=min_sep_samples)

    correlation = matched_filter(m_buffer, m_pulse)
    envelope = np.abs(hilbert(correlation))
    corr_distance = (np.arange(correlation.size) - (m_pulse.size - 1)) / m_fs * m_speed / 2

    # ---- stages -------------------------------------------------------------
    def draw_scene(height):
        fig = base_figure(height)
        fig.add_scatter(x=np.arange(m_buffer.size) / m_fs * 1000, y=m_buffer,
                        mode="lines", line=TRACE_RAW, name="received",
                        hovertemplate="%{x:.1f} ms<extra></extra>")
        for target in distances:
            fig.add_vline(x=2 * target / m_speed * 1000, line=LINE_THRESH)
        fig.update_xaxes(title_text="Time (ms)")
        show_chart(fig)

    def draw_envelope(height):
        lo, hi = min(distances) - 0.6, max(distances) + 0.9
        keep = (corr_distance >= lo) & (corr_distance <= hi)
        fig = base_figure(height)
        fig.add_scatter(x=corr_distance[keep], y=correlation[keep], mode="lines",
                        line=dict(color="#475569", width=1), name="correlation",
                        hovertemplate="%{x:.3f} m<extra></extra>")
        fig.add_scatter(x=corr_distance[keep], y=envelope[keep], mode="lines",
                        line=TRACE_PAYOFF, name="envelope",
                        hovertemplate="%{x:.3f} m<extra></extra>")
        for target in distances:
            fig.add_vline(x=target, line=LINE_THRESH)
        for found in detected:
            fig.add_vline(x=found, line=dict(color=ACCENT, width=1, dash="dot"))
        fig.update_xaxes(title_text="Distance (m)")
        show_chart(fig)

    def draw_sweep(height):
        state = st.session_state.get("m2_sweep_state", "idle")
        stored = st.session_state.get("m2_sweep_result")

        if state == "done" and stored is not None:
            bandwidth = np.array(stored["bandwidth_hz"])
            measured = np.array([np.nan if v is None else v * 100
                                 for v in stored["resolved_m"]])
            theory = np.array([np.nan if v is None else v * 100
                               for v in stored["theory_m"]])
            order = np.argsort(bandwidth)
            fig = base_figure(height)
            fig.add_scatter(x=bandwidth[order], y=theory[order], mode="lines",
                            line=LINE_THRESH, name="theory c/2B")
            fig.add_scatter(x=bandwidth[order], y=measured[order], mode="markers",
                            marker=dict(color=ACCENT, size=9), name="measured")
            fig.update_xaxes(title_text="Measured bandwidth (Hz)", type="log")
            fig.update_yaxes(title_text="Closest separation (cm)", type="log")
            show_chart(fig)
            if st.button("Run again", key="m2_rerun", type="tertiary"):
                st.session_state["m2_sweep_state"] = "idle"
                st.rerun()
            return

        empty_state("Bandwidth vs resolution",
                    "Measures how close two targets can get, for pulses of different "
                    "bandwidth, and compares it with c / 2B. Takes a few seconds.")
        if st.button("Run sweep", key="m2_run", type="primary"):
            configs = [{"pulse_type": "gaussian", "duration_s": d, "freq_hz": 4000.0}
                       for d in (0.008, 0.004, 0.002, 0.001)]
            configs += [{"pulse_type": "chirp", "duration_s": 0.002, "freq_hz": 8000.0,
                         "bandwidth_hz": b} for b in (2000.0, 4000.0, 6000.0)]
            progress = st.progress(0.0, text="Measuring resolution…")
            result = run_bandwidth_resolution_sweep(configs, m_fs, trials=8)
            progress.progress(1.0, text="Done")
            st.session_state["m2_sweep_result"] = result
            st.session_state["m2_sweep_state"] = "done"
            st.rerun()

    with main:
        if notes:
            st.markdown(f'<p class="stale">Adjusted to stay physical: {"; ".join(notes)}.</p>',
                        unsafe_allow_html=True)
        stage_grid("m2", [
            dict(eyebrow="Stage 1 · Receive",
                 title="Both echoes arrive on top of each other",
                 description=f"{len(distances)} targets, {separation_cm:.0f} cm apart.",
                 draw=draw_scene),
            dict(eyebrow="Stage 2 · Detect", title="One peak per target?",
                 description="Envelope of the matched filter output.",
                 draw=draw_envelope),
            dict(eyebrow="Study", title="Bandwidth vs resolution",
                 description="Measured limit against the c / 2B prediction.",
                 draw=draw_sweep),
        ], payoff_index=1)

    resolved = len(detected) >= len(distances)
    with results_slot:
        with st.container(key="hero"):
            st.metric("Targets found", f"{len(detected)} / {len(distances)}",
                      delta="resolved" if resolved else "merged",
                      delta_color="normal" if resolved else "inverse")
        c1, c2 = st.columns(2)
        c1.metric("Separation", f"{separation_cm:.0f} cm")
        c2.metric("Theory c/2B", f"{theory_cm:.1f} cm")
        rows = []
        for index, target in enumerate(distances, start=1):
            nearest = min(detected, key=lambda d: abs(d - target)) if detected else None
            rows.append({
                "Target": f"T{index}",
                "Truth": f"{target:.3f} m",
                "Found": f"{nearest:.3f} m" if nearest is not None else "—",
                "Error": f"{abs(nearest - target) * 100:.1f} cm" if nearest is not None else "—",
            })
        st.dataframe(rows, hide_index=True, width="stretch")

    with audio_tab:
        st.caption(f"Slowed {m_slowdown:.0f}×. Each repeat carries its own noise, so the "
                   "hiss does not cycle.")
        st.markdown("**Both echoes together**")
        try:
            st.audio(cached_scene_wav(m_pulse_type, m_dur_s, m_fs, m_freq, m_chirp_bw,
                                      tuple(distances), tuple(attenuations), m_snr_db,
                                      m_speed, m_buf_dur, m_slowdown),
                     format="audio/wav", loop=True)
        except Exception as exc:
            st.caption(f"Audio unavailable: {exc}")


# ==============================================================================
# SCREEN 3 — HOW MUCH NOISE BEFORE IT FAILS?
# ==============================================================================


def render_screen_3() -> None:
    st.markdown(f'<p class="screen-title">{SCREEN_TITLE["m3"]}</p>',
                unsafe_allow_html=True)
    st.caption("Repeats the whole measurement many times at each noise level and plots "
               "how wrong the distance gets.")

    main, side = st.columns([1, 0.42], gap="small")

    with side:
        with st.container(key="side_m3"):
            panel_header("Step 1", "Settings", "")
            sw_dist = st.slider("Evaluation distance (m)", 1.0, 20.0, 5.0, 1.0)
            snr_lo, snr_hi = st.slider("Noise range (SNR, dB)", -20, 24, (-12, 16), 2)
            sw_trials = st.slider("Trials per point", 5, 35, 12, 1)
            sw_pulse_type = st.selectbox("Pulse shape", ["gaussian", "chirp", "rect"],
                                         key="m3_ptype")
            with st.expander("Advanced"):
                snr_step = st.selectbox("Step size (dB)", [2, 3, 4], index=0)
                sw_dur_ms = st.slider("Pulse length (ms)", 1.0, 10.0, 2.0, 0.5,
                                      key="m3_dur")
                sw_fc = st.slider("Carrier frequency (Hz)", 1000.0, 10000.0, 4000.0,
                                  250.0, key="m3_fc")
                sw_bandpass = st.checkbox("Also run with a band-pass filter", value=True)

            snr_values = [float(v) for v in range(snr_lo, snr_hi + 1, snr_step)]
            runs = len(snr_values) * sw_trials * (2 if sw_bandpass else 1)
            st.caption(f"{len(snr_values)} noise levels · {runs:,} simulated pulses")
            run_now = st.button("Run the test", key="m3_run", type="primary",
                                width="stretch")

    config = (sw_dist, tuple(snr_values), sw_trials, sw_pulse_type, sw_dur_ms,
              sw_fc, sw_bandpass)
    stored = st.session_state.get("m3_result")
    stored_config = st.session_state.get("m3_config")

    with main:
        with st.container(border=True, key="step2"):
            panel_header("Step 2", "The graph",
                         "Flat and near-perfect above 0 dB, then a sharp cliff.")

            if run_now:
                progress = st.progress(0.0, text="Starting…")
                plain, filtered = [], []
                for index, snr in enumerate(snr_values, start=1):
                    progress.progress(index / len(snr_values),
                                      text=f"SNR {snr:+.0f} dB · point {index} of "
                                           f"{len(snr_values)}")
                    plain.append(run_snr_sweep(
                        sw_dist, [snr], 48000.0, pulse_type=sw_pulse_type,
                        duration_s=sw_dur_ms / 1000.0, freq_hz=sw_fc,
                        trials_per_snr=sw_trials, bandpass=False)["rmse_m"][0])
                    if sw_bandpass:
                        filtered.append(run_snr_sweep(
                            sw_dist, [snr], 48000.0, pulse_type=sw_pulse_type,
                            duration_s=sw_dur_ms / 1000.0, freq_hz=sw_fc,
                            trials_per_snr=sw_trials, bandpass=True)["rmse_m"][0])
                progress.empty()
                stored = {"snr": snr_values, "plain": plain, "filtered": filtered or None}
                st.session_state["m3_result"] = stored
                st.session_state["m3_config"] = config
                stored_config = config

            if stored is None:
                empty_state("Nothing measured yet",
                            f"Press <b>Run the test</b> to simulate {runs:,} pulses "
                            "across the noise range. It takes a few seconds.")
            else:
                if stored_config != config:
                    st.markdown('<p class="stale">Settings changed since this run — '
                                'press Run the test again to refresh.</p>',
                                unsafe_allow_html=True)
                quantisation_m = 343.0 / (4 * 48000.0)
                fig = base_figure(520)
                fig.add_scatter(x=stored["snr"], y=stored["plain"], mode="lines+markers",
                                line=TRACE_PAYOFF, marker=dict(size=7, color=ACCENT),
                                name="matched filter",
                                hovertemplate="%{x:.0f} dB · %{y:.4f} m<extra></extra>")
                if stored["filtered"]:
                    fig.add_scatter(x=stored["snr"], y=stored["filtered"],
                                    mode="lines+markers", line=TRACE_COMPARE,
                                    marker=dict(size=6, color=COMPARE),
                                    name="with band-pass",
                                    hovertemplate="%{x:.0f} dB · %{y:.4f} m<extra></extra>")
                fig.add_hline(y=quantisation_m, line=LINE_THRESH,
                              annotation_text="half-sample limit ≈ 1.8 mm",
                              annotation_font=dict(color=THRESH, size=10.5))
                fig.update_yaxes(type="log", title_text="Distance error (m)")
                fig.update_xaxes(title_text="Noise level (SNR, dB)")
                show_chart(fig)
                worst = max(stored["plain"])
                best = min(stored["plain"])
                st.caption(f"Best {best * 1000:.1f} mm · worst {worst:.2f} m across the range.")


# ==============================================================================
# SCREEN 4 — REAL SOUND TEST
# ==============================================================================

AC_FS = 48000.0
AC_FRAME_SAMPLES = int((0.010 + 0.250) * AC_FS)
RX_FILE = ROOT_DIR / "physical_sonar_check" / "rx.npy"
BASE_FILE = ROOT_DIR / "physical_sonar_check" / "baseline.npy"
HAND_RX_FILE = ROOT_DIR / "physical_sonar_check" / "hand_rx.npy"
HAND_BASE_FILE = ROOT_DIR / "physical_sonar_check" / "hand_baseline.npy"


def record_live_acoustic(chirp, repeats=16, fs=AC_FS, amplitude=0.4, gap_s=0.250):
    """Plays the chirp train through the speaker and records the microphone."""
    import sounddevice as sd

    frame_samples = int(round((len(chirp) / fs + gap_s) * fs))
    frame = np.zeros(frame_samples)
    frame[: chirp.size] = amplitude * chirp
    transmit = np.concatenate([np.tile(frame, repeats), np.zeros(frame_samples)])
    stereo = np.zeros((transmit.size, 2))
    stereo[:, 1] = transmit
    return sd.playrec(stereo, samplerate=int(fs), channels=1, blocking=True)[:, 0]


def render_screen_4() -> None:
    st.markdown(f'<p class="screen-title">{SCREEN_TITLE["m4"]}</p>',
                unsafe_allow_html=True)
    st.caption("The same maths on real sound: a chirp out of the laptop speaker, the "
               "echo back through its microphone.")

    chirp_sig = design_chirp(2000.0, 8000.0, 0.010, AC_FS)
    for key in ("live_baseline_env", "live_result", "live_rx"):
        st.session_state.setdefault(key, None)

    main, side = st.columns([1, 0.42], gap="small")

    with side:
        with st.container(key="side_m4"):
            results_slot = st.container(key="results")
            settings_tab, audio_tab = st.tabs(["Settings", "Audio"])
            with settings_tab:
                sources = ["Saved recording — wall at 1.80 m"]
                if HAND_RX_FILE.exists():
                    sources.append("Saved recording — hand at 0.27 m")
                sources.append("Record now with your speaker and mic")
                source = st.selectbox("Where the recording comes from", sources)
                ac_speed = st.number_input("Speed of sound (m/s)", 320.0, 360.0, 343.0,
                                           0.5, key="m4_speed")
                min_range = st.slider("Ignore closer than (m)", 0.1, 2.5, 1.0, 0.1,
                                      key="m4_min")
                max_range = st.slider("Ignore further than (m)", 3.0, 8.0, 5.0, 0.5,
                                      key="m4_max")
                live_repeats = 8
                if source.startswith("Record now"):
                    live_repeats = st.selectbox("Chirps per burst", [8, 16], index=0)
                    st.caption("Point the laptop at open space and press Record baseline "
                               "first, then aim at a wall and measure. Do not change the "
                               "system volume in between.")
                    if st.button("1 · Record room baseline", width="stretch"):
                        try:
                            with st.spinner(f"Recording ~{live_repeats * 0.26:.1f} s…"):
                                rec = record_live_acoustic(chirp_sig, repeats=live_repeats)
                                st.session_state["live_baseline_env"] = analyse_recording(
                                    rec, chirp_sig, AC_FRAME_SAMPLES, AC_FS)["envelope"]
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Recording failed: {exc}")
                    if st.button("2 · Measure the echo", width="stretch",
                                 type="primary"):
                        try:
                            baseline = st.session_state["live_baseline_env"]
                            if baseline is None and BASE_FILE.exists():
                                baseline = np.load(BASE_FILE)
                            with st.spinner(f"Recording ~{live_repeats * 0.26:.1f} s…"):
                                rec = record_live_acoustic(chirp_sig, repeats=live_repeats)
                                st.session_state["live_rx"] = rec
                                st.session_state["live_result"] = analyse_recording(
                                    rec, chirp_sig, AC_FRAME_SAMPLES, AC_FS,
                                    baseline_envelope=baseline, speed_mps=ac_speed,
                                    min_range_m=min_range, max_range_m=max_range)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Measurement failed: {exc}")

    # ---- resolve the recording ---------------------------------------------
    result, baseline_env = None, None
    if source.startswith("Saved recording — wall") and RX_FILE.exists():
        baseline_env = np.load(BASE_FILE)
        result = analyse_recording(np.load(RX_FILE), chirp_sig, AC_FRAME_SAMPLES, AC_FS,
                                   baseline_envelope=baseline_env, speed_mps=ac_speed,
                                   min_range_m=min_range, max_range_m=max_range)
    elif source.startswith("Saved recording — hand") and HAND_RX_FILE.exists():
        baseline_env = np.load(HAND_BASE_FILE)
        result = analyse_recording(np.load(HAND_RX_FILE), chirp_sig, AC_FRAME_SAMPLES,
                                   AC_FS, baseline_envelope=baseline_env,
                                   speed_mps=ac_speed,
                                   min_range_m=min(min_range, 0.1), max_range_m=1.5)
    else:
        result = st.session_state.get("live_result")
        baseline_env = st.session_state.get("live_baseline_env")

    def draw_overlay(height):
        if result is None:
            empty_state("No recording yet", "Record a baseline and a measurement.")
            return
        envelope = result["envelope"]
        axis = AC_FS and 343.0 * np.arange(envelope.size) / AC_FS / 2
        keep = (axis >= 0.05) & (axis <= max_range)
        fig = base_figure(height)
        if baseline_env is not None and baseline_env.shape == envelope.shape:
            fig.add_scatter(x=axis[keep], y=baseline_env[keep], mode="lines",
                            line=dict(color="#475569", width=1.2), name="empty room")
        fig.add_scatter(x=axis[keep], y=envelope[keep], mode="lines",
                        line=TRACE_RAW, name="this recording")
        fig.update_xaxes(title_text="Distance (m)")
        show_chart(fig)

    def draw_excess(height):
        if result is None:
            empty_state("Nothing measured yet",
                        "The clutter-removed trace appears here once a recording exists.")
            return
        excess = result["excess"]
        axis = 343.0 * np.arange(excess.size) / AC_FS / 2
        keep = (axis >= min_range * 0.5) & (axis <= max_range)
        fig = base_figure(height)
        fig.add_scatter(x=axis[keep], y=excess[keep], mode="lines", line=TRACE_PAYOFF,
                        name="after clutter removal",
                        hovertemplate="%{x:.3f} m<extra></extra>")
        fig.add_vline(x=result["distance_m"], line=dict(color=ACCENT, width=1, dash="dash"),
                      annotation_text=f"{result['distance_m']:.3f} m",
                      annotation_font=dict(color=ACCENT, size=10.5,
                                           family="JetBrains Mono"))
        fig.update_xaxes(title_text="Distance (m)")
        show_chart(fig)

    def draw_chirp(height):
        fig = base_figure(height)
        fig.add_scatter(x=np.arange(chirp_sig.size) / AC_FS * 1000, y=chirp_sig,
                        mode="lines", line=TRACE_RAW, name="chirp")
        fig.update_xaxes(title_text="Time (ms)")
        show_chart(fig)

    with main:
        stage_grid("m4", [
            dict(eyebrow="Stage 1 · Compare", title="This recording vs the empty room",
                 description="Everything static appears in both traces.",
                 draw=draw_overlay),
            dict(eyebrow="Stage 2 · Detect", title="What is new in the room",
                 description="Subtract the empty room and the target is what is left.",
                 draw=draw_excess),
            dict(eyebrow="Transmit", title="The chirp we send",
                 description="2–8 kHz sweep, 10 ms, Hann windowed.",
                 draw=draw_chirp),
        ], payoff_index=1)

    with results_slot:
        if result is None:
            st.caption("No measurement yet.")
        else:
            sigma = result["quality_sigma"]
            with st.container(key="hero"):
                st.metric("Measured distance", f"{result['distance_m']:.3f} m",
                          delta=f"{sigma:.1f}σ confidence",
                          delta_color="normal" if sigma > 6 else "inverse")
            st.caption("Above 6σ counts as a real detection."
                       if sigma > 6 else "Below 6σ — too weak to trust.")

    with audio_tab:
        st.caption("The outgoing chirp is audible as-is, so it is not slowed down.")
        st.markdown("**The chirp we send**")
        try:
            clip = make_audible(chirp_sig, fs=AC_FS, repeats=4, slowdown=1.0, gap_s=0.3)
            st.audio(to_wav_bytes(clip["audio"], clip["rate_hz"]), format="audio/wav",
                     loop=True)
        except Exception as exc:
            st.caption(f"Audio unavailable: {exc}")
        if st.session_state.get("live_rx") is not None:
            st.markdown("**What the microphone recorded**")
            st.audio(to_wav_bytes(st.session_state["live_rx"], int(AC_FS)),
                     format="audio/wav")


# ==============================================================================
# ROUTER
# ==============================================================================

if page == "home":
    render_home()
else:
    render_topbar(page)
    {"m1": render_screen_1, "m2": render_screen_2,
     "m3": render_screen_3, "m4": render_screen_4}[page]()
