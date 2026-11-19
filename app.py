"""
MA375TGR | Mathematics of Music
Multi-Stem Audio Separation and Parametric Equalization with LLM-Driven Natural Language Control
Interactive Streamlit Application
"""

import os
import sys
import json
import hashlib
import importlib
import numpy as np
import streamlit as st

# Add current workspace to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import src.phase2_llm.llm_engine
importlib.reload(src.phase2_llm.llm_engine)

from src.phase1_separation import (
    STEM_NAMES,
    DEFAULT_SR,
    load_audio,
    save_audio,
    create_stock_preset,
    create_high_contrast_preset,
    StemSeparator
)
from src.phase2_llm import (
    LLMReasoningEngine,
    EQPlan,
    FilterEdit,
    FilterType,
    StemType,
    DESCRIPTOR_SPECS
)
from src.phase3_dsp import BiquadFilter, DSPEngine
from src.visualization import plot_filter_curves, plot_before_after_spectrogram


# Page Configuration
st.set_page_config(
    page_title="Multi-Stem Audio EQ | Real-Time Neural & LLM Control",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #f0f6fc; }
    .stApp { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
    .project-header {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 25px;
    }
    .badge-online {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        background: #10b981;
        color: white;
    }
    .badge-offline {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        background: #ef4444;
        color: white;
    }
    .badge-model {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        background: #0284c7;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "current_track_id" not in st.session_state:
        st.session_state.current_track_id = None
    if "mix_path" not in st.session_state:
        st.session_state.mix_path = None
    if "stem_paths" not in st.session_state:
        st.session_state.stem_paths = {}
    if "eq_plan" not in st.session_state:
        st.session_state.eq_plan = None
    if "dsp_results" not in st.session_state:
        st.session_state.dsp_results = None


init_session_state()

# Initialize engines
separator = StemSeparator(model_name="htdemucs")
llm_engine = LLMReasoningEngine()
ollama_online = llm_engine.is_ollama_available()


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🎛️ System Configuration")

    # 1. Audio Source Selection
    st.subheader("1. Audio Input Track")
    audio_source = st.radio(
        "Choose Audio Source:",
        [
            "High-Contrast Benchmark (Audible Boxy Vocal & Harsh Drums)",
            "Stock Preset: Funk / Lo-Fi Groove",
            "Upload Custom Audio (.wav / .mp3) [Real Neural Demucs]"
        ]
    )

    if audio_source == "High-Contrast Benchmark (Audible Boxy Vocal & Harsh Drums)":
        track_id = "preset_high_contrast"
        preset_dir = os.path.join("data", "presets", "high_contrast_demo")
        if not os.path.exists(os.path.join(preset_dir, "mixture.wav")):
            with st.spinner("Synthesizing High-Contrast Benchmark preset..."):
                mix_p, stems_p = create_high_contrast_preset(preset_dir)
                st.session_state.mix_path = mix_p
                st.session_state.stem_paths = stems_p
        else:
            st.session_state.mix_path = os.path.join(preset_dir, "mixture.wav")
            st.session_state.stem_paths = {
                name: os.path.join(preset_dir, f"{name}.wav") for name in STEM_NAMES
            }
        st.session_state.current_track_id = track_id
        st.success("✔ High-Contrast Benchmark Active")

    elif audio_source == "Stock Preset: Funk / Lo-Fi Groove":
        track_id = "preset_funk_groove"
        preset_dir = os.path.join("data", "presets", "stock_funk_groove")
        if not os.path.exists(os.path.join(preset_dir, "mixture.wav")):
            with st.spinner("Synthesizing 4-stem stock preset..."):
                mix_p, stems_p = create_stock_preset(preset_dir)
                st.session_state.mix_path = mix_p
                st.session_state.stem_paths = stems_p
        else:
            st.session_state.mix_path = os.path.join(preset_dir, "mixture.wav")
            st.session_state.stem_paths = {
                name: os.path.join(preset_dir, f"{name}.wav") for name in STEM_NAMES
            }
        st.session_state.current_track_id = track_id
        st.success("✔ Stock 4-Stem Preset Active")

    else:
        uploaded_file = st.file_uploader("Upload Any Music Track (.wav, .mp3)", type=["wav", "mp3"])
        if uploaded_file is not None:
            upload_name = uploaded_file.name
            file_bytes = uploaded_file.getbuffer()
            file_hash = hashlib.md5(file_bytes[:1024 * 512]).hexdigest()[:8]
            track_id = f"upload_{upload_name}_{file_hash}"

            upload_path = os.path.join("data", "uploads", upload_name)
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            with open(upload_path, "wb") as f:
                f.write(file_bytes)

            st.session_state.mix_path = upload_path

            # Only separate if this is a newly uploaded track
            if st.session_state.current_track_id != track_id:
                st.info(f"New track uploaded: {upload_name}. Running real Demucs 4-stem separation...")
                prog_bar = st.progress(0.0)
                status_txt = st.empty()

                def update_progress(pct, msg):
                    prog_bar.progress(pct)
                    status_txt.text(msg)

                try:
                    separated_stems = separator.separate(
                        upload_path,
                        progress_callback=update_progress
                    )
                    st.session_state.stem_paths = separated_stems
                    st.session_state.current_track_id = track_id
                    status_txt.empty()
                    prog_bar.empty()
                    st.success(f"✔ Real Demucs separation completed for {upload_name}!")
                except Exception as e:
                    st.error(f"Separation failed: {e}")

    st.markdown("---")

    # 2. AI Reasoning Engine Selection
    st.subheader("2. AI Reasoning Engine")
    if ollama_online:
        st.markdown("<span class='badge-online'>OLLAMA DAEMON: CONNECTED</span>", unsafe_allow_html=True)
        if hasattr(llm_engine, "list_installed_ollama_models"):
            installed_models = llm_engine.list_installed_ollama_models()
        else:
            try:
                import ollama
                resp = ollama.Client(host="http://127.0.0.1:11434").list()
                models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
                installed_models = [m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "") for m in models]
            except Exception:
                installed_models = []
        if not installed_models:
            installed_models = ["llama3.2", "qwen2.5"]
    else:
        st.markdown("<span class='badge-offline'>OLLAMA DAEMON: OFFLINE (localhost:11434)</span>", unsafe_allow_html=True)
        installed_models = ["llama3.2", "qwen2.5"]

    engine_options = []
    if ollama_online:
        engine_options.append("Ollama (Local LLM via official SDK)")
    engine_options.append("Google Gemini API (Cloud Neural LLM)")
    engine_options.append("Deterministic Psychoacoustic Rules (Offline Baseline)")

    selected_backend = st.selectbox("LLM Inference Engine:", engine_options)

    selected_ollama_model = "llama3.2"
    gemini_api_key = ""

    if "Ollama" in selected_backend:
        selected_ollama_model = st.selectbox("Select Ollama Model:", installed_models)
        with st.expander("📥 Pull / Download New Ollama Model"):
            new_model = st.text_input("Model Tag (e.g. llama3.2, qwen2.5, mistral):", value="llama3.2")
            if st.button("Pull Model"):
                with st.spinner(f"Pulling {new_model} via Ollama SDK..."):
                    try:
                        if hasattr(llm_engine, "pull_ollama_model"):
                            llm_engine.pull_ollama_model(new_model)
                        else:
                            import ollama
                            ollama.Client(host="http://127.0.0.1:11434").pull(new_model)
                        st.success(f"Successfully pulled {new_model}!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Pull error: {err}")

    elif "Gemini" in selected_backend:
        gemini_api_key = st.text_input("Enter Gemini API Key:", type="password")
        st.caption("Enter a Google Gemini API key to run neural LLM reasoning live without installing Ollama locally.")

    else:
        st.info("Using built-in deterministic psychoacoustic rules (Slide 11 frequency maps).")

    st.markdown("---")

    # 3. DSP Settings
    st.subheader("3. DSP Resynthesis Mode")
    dsp_mode = st.radio(
        "Filter Mode:",
        ["STFT Bin Gain (Hann N=2048, hop=512)", "Time-Domain Biquad SOS"]
    )
    mode_key = "stft" if "STFT" in dsp_mode else "sos"

    st.markdown("---")
    st.caption("MA375TGR Mathematics of Music\nAuthors: Bharath Kumar MS, Dharshan SN, Dheeraj R")


# ----------------- MAIN HEADER -----------------
st.markdown("""
<div class="project-header">
    <div style="font-size: 13px; color: #38bdf8; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;">
        MA375TGR | Mathematics of Music
    </div>
    <h2 style="margin: 4px 0 8px 0; color: #ffffff;">
        Multi-Stem Audio Separation & Parametric EQ with LLM-Driven Natural Language Control
    </h2>
    <p style="margin: 0; color: #94a3b8; font-size: 14px;">
        Real Demucs Neural Stem Separation + LLM Reasoning + Deterministic Biquad STFT Resynthesis.
    </p>
</div>
""", unsafe_allow_html=True)


# ----------------- SECTION 1: STEM MIXER -----------------
st.markdown("### 1. Multi-Stem Audio Input & Stems")
mix_col, stems_col = st.columns([1, 2])

with mix_col:
    st.markdown("#### 🎧 Input Mixed Track $x(t)$")
    if st.session_state.mix_path and os.path.exists(st.session_state.mix_path):
        st.audio(st.session_state.mix_path)
        mix_data, sr = load_audio(st.session_state.mix_path)
        st.caption(f"Channels: Stereo | Sample Rate: {sr} Hz | Duration: {mix_data.shape[-1]/sr:.1f}s")
    else:
        st.info("Select or upload audio in sidebar.")

with stems_col:
    st.markdown("#### 🧩 Isolated Stems (Demucs 4-Stem Model)")
    col_v, col_d, col_b, col_o = st.columns(4)

    cols = {"vocals": col_v, "drums": col_d, "bass": col_b, "other": col_o}
    icons = {"vocals": "🎤 Vocals", "drums": "🥁 Drums", "bass": "🎸 Bass", "other": "🎹 Other"}

    for name in STEM_NAMES:
        with cols[name]:
            st.markdown(f"**{icons[name]}**")
            p = st.session_state.stem_paths.get(name)
            if p and os.path.exists(p):
                st.audio(p)
            else:
                st.caption("Waiting for separation...")

st.markdown("---")


# ----------------- SECTION 2: NATURAL LANGUAGE PROMPT -----------------
st.markdown("### 2. Natural Language Timbre Control (LLM Reasoning Layer)")

example_prompts = [
    "The vocal sounds too boxy and muddy, and I want the bass kick to have more punch.",
    "The drums are too harsh in the cymbals, add air to the vocals, and make the bass warmer.",
    "Tame sibilance on the singer and eliminate sub-rumble below 35 Hz on the mix.",
    "Boost the vocal presence by +4 dB at 2500 Hz and make the keyboards brighter."
]

selected_example = st.selectbox("Quick Preset Prompts (or type custom below):", ["-- Custom Prompt --"] + example_prompts)
default_text = example_prompts[0] if selected_example == "-- Custom Prompt --" else selected_example
user_prompt = st.text_input("Enter natural language mixing instruction:", value=default_text)

col_slider, col_run = st.columns([2, 1])
with col_slider:
    intensity = st.slider("🔊 EQ Intensity Multiplier (Increase if changes are subtle on laptop speakers):", min_value=1.0, max_value=2.5, value=1.6, step=0.1)
with col_run:
    st.write("")
    run_button = st.button("🚀 Process Audio with AI", type="primary", use_container_width=True)

if run_button and user_prompt:
    if not st.session_state.stem_paths:
        st.error("No stems available! Please load an audio track or preset first.")
    else:
        backend_tag = "rules"
        if "Ollama" in selected_backend and ollama_online:
            backend_tag = "ollama"
        elif "Gemini" in selected_backend and gemini_api_key:
            backend_tag = "gemini"

        with st.spinner(f"Running LLM reasoning engine ({selected_backend})..."):
            plan = llm_engine.generate_plan(
                user_prompt,
                backend_mode=backend_tag,
                model_name=selected_ollama_model,
                api_key=gemini_api_key
            )
            # Apply intensity scaling
            if intensity != 1.0:
                for edit in plan.edits:
                    edit.gain_db = max(-12.0, min(12.0, round(edit.gain_db * intensity, 1)))
            st.session_state.eq_plan = plan

        with st.spinner("Executing deterministic DSP & Resynthesizing master mix..."):
            dsp = DSPEngine(mode=mode_key, fs=DEFAULT_SR)
            out_dir = os.path.join("data", "output", f"processed_{st.session_state.current_track_id}")
            results = dsp.process_stems(st.session_state.stem_paths, plan, output_dir=out_dir)
            st.session_state.dsp_results = results

        st.success("✔ Audio successfully equalized and resynthesized!")


# ----------------- DISPLAY LLM REASONING JSON & FILTER CURVES -----------------
if st.session_state.eq_plan is not None:
    plan: EQPlan = st.session_state.eq_plan

    st.markdown(
        f"**Engineering Summary**: *{plan.summary}* &nbsp; <span class='badge-model'>Engine: {plan.source.upper()}</span>",
        unsafe_allow_html=True
    )

    json_col, curve_col = st.columns([1, 1])

    with json_col:
        st.markdown("#### 📋 Validated Biquad Parameter JSON (Slide 10)")
        st.json(plan.model_dump()["edits"])

    with curve_col:
        st.markdown("#### 📈 Cascaded Biquad Transfer Function $H(z)$")
        all_filters = [
            BiquadFilter(
                filter_type=e.filter.value,
                f0_hz=e.f0_hz,
                gain_db=e.gain_db,
                q=e.q,
                fs=DEFAULT_SR
            ) for e in plan.edits
        ]
        if all_filters:
            fig_curve = plot_filter_curves(all_filters)
            st.pyplot(fig_curve)
        else:
            st.info("No active filter curves.")


# ----------------- SECTION 3: RESYNTHESIZED MASTER & A/B COMPARISON -----------------
if st.session_state.dsp_results is not None:
    res = st.session_state.dsp_results

    st.markdown("---")
    st.markdown("### 3. Resynthesized Audio Master (A/B Comparison)")

    orig_col, proc_col = st.columns(2)

    with orig_col:
        st.markdown("#### ⚪ Original Mix $x(t)$")
        st.audio(st.session_state.mix_path)

    with proc_col:
        st.markdown("#### 🔵 Polished Master Mix $x_{\\text{master}}(t)$")
        st.audio(res["master_mix_path"])
        with open(res["master_mix_path"], "rb") as f:
            st.download_button(
                label="⬇️ Download Polished Master (.wav)",
                data=f.read(),
                file_name="polished_master_mix.wav",
                mime="audio/wav"
            )

    st.markdown("#### 🎚️ Direct Stem A/B Solo Comparison (Hear exact differences in isolation)")
    st.caption("Listening to individual stems isolates the EQ change from other instruments masking it.")

    col_orig_stem, col_proc_stem = st.columns(2)
    selected_ab_stem = st.selectbox("Select Stem for Solo A/B Audition:", ["vocals", "drums", "bass", "other"])

    with col_orig_stem:
        st.markdown(f"**Before (Original {selected_ab_stem.capitalize()})**")
        st.audio(st.session_state.stem_paths[selected_ab_stem])

    with col_proc_stem:
        st.markdown(f"**After (Processed {selected_ab_stem.capitalize()})**")
        st.audio(res["stem_paths"][selected_ab_stem])

    # Delta / Null Test
    orig_s, sr_s = load_audio(st.session_state.stem_paths[selected_ab_stem])
    proc_s, _ = load_audio(res["stem_paths"][selected_ab_stem])
    delta_s = orig_s - proc_s

    delta_path = os.path.join("data", "output", f"delta_{selected_ab_stem}.wav")
    save_audio(delta_path, delta_s, sr=sr_s, normalize=True)

    with st.expander(f"🔍 Listen to Delta / Null Test Signal (The exact frequencies removed/altered from {selected_ab_stem.capitalize()})"):
        st.audio(delta_path)
        st.caption("The Delta signal plays ONLY the acoustic difference. If you hear audio here, the EQ definitely reshaped the sound!")


    # ----------------- SECTION 4: LIVE SPECTROGRAM VISUALIZATION -----------------
    st.markdown("---")
    st.markdown("### 4. Live Before / After Spectrogram Analysis (Slide 14)")

    selected_stem_view = st.selectbox(
        "Select Track to Inspect in Spectrogram:",
        ["vocals", "drums", "bass", "other", "master mix"]
    )

    if selected_stem_view == "master mix":
        orig_data, fs = load_audio(st.session_state.mix_path)
        proc_data, _ = load_audio(res["master_mix_path"])
    else:
        orig_data, fs = load_audio(st.session_state.stem_paths[selected_stem_view])
        proc_data, _ = load_audio(res["stem_paths"][selected_stem_view])

    spec_fig = plot_before_after_spectrogram(
        orig_data,
        proc_data,
        fs=fs,
        stem_name=selected_stem_view
    )
    st.pyplot(spec_fig)
