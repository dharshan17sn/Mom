# Multi-Stem Audio Separation and Parametric Equalization with LLM-Driven Natural Language Control

**Course**: MA375TGR | Mathematics of Music  
**Authors**:
- Bharath Kumar MS (1RV23IS034)
- Dharshan SN (1RV23IS044)
- Dheeraj R (1RV23IS045)

---

## 🎵 Project Overview

Shaping the tone of an audio recording normally demands mastering frequency knobs, Q bandwidths, filter topologies, and decibels. This project replaces that steep learning curve with a single natural language sentence:

> *“The vocal sounds too boxy and muddy, and I want the bass kick to have more punch.”*  
> **$\rightarrow$ Vocal: $-4\text{ dB}$ at $300\text{ Hz}$, Bass: $+3\text{ dB}$ at $80\text{ Hz}$**

### Core Philosophy
**AI as the control layer, deterministic DSP as the engine.**
Only the interpretation of natural language timbre adjectives is driven by language reasoning; every audio sample passes through deterministic, mathematically inspectable filter equations.

---

## 🏛️ System Architecture & Implemented Phases

```
1. Input Mix x(t) ───► [Phase 1: Stem Separation] ───► Vocals, Drums, Bass, Other
                             │
2. User Prompt    ───► [Phase 2: LLM Reasoning]   ───► Validated Filter JSON
                             │
3. Stems + JSON   ───► [Phase 3: DSP & Resynth]   ───► Polished Master Mix
                             │
4. Web Dashboard  ───► [Phase 4: Streamlit UI]    ───► Live Spectrograms & A/B Playback
```

### Phase 1: Multi-Stem Data & Separation Pipeline
- **Stock Audio Preset Generator (`stock_generator.py`)**: Synthesizes a high-fidelity 44.1 kHz stereo 4-stem track (`stock_funk_groove`) with psychoacoustic test targets (300 Hz vocal formant, 80 Hz bassline, drum transients, synth pads).
- **Stem Preservation**: Mathematically verified that $x(t) \approx \text{vocals}(t) + \text{drums}(t) + \text{bass}(t) + \text{other}(t)$ with **$72.93\text{ dB}$ SNR** and near-zero MSE ($1.79 \times 10^{-9}$).
- **Demucs Wrapper (`separator.py`)**: Seamless fallback between pre-separated stock stems and neural Demucs separation.

### Phase 2: LLM Reasoning Engine & Semantic Prior
- **Pydantic Schema Validation (`schema.py`)**: Enforces strict boundaries:
  - Gain: $\pm 12.0\text{ dB}$
  - Frequency $f_0$: $20\text{ Hz} - 20,000\text{ Hz}$
  - $Q$: $0.1 - 10.0$
- **Psychoacoustic Prior Knowledge Map (`psychoacoustics.py`)**:
  - *Punchy kick*: Boost 60–100 Hz ($+3\text{ dB}$)
  - *Warm*: Boost 100–250 Hz ($+3\text{ dB}$)
  - *Muddy*: Cut 200–400 Hz ($-4\text{ dB}$)
  - *Boxy*: Cut 300–600 Hz ($-4\text{ dB}$)
  - *Nasal*: Cut 800–1200 Hz ($-3.5\text{ dB}$)
  - *Harsh*: Cut 2–5 kHz ($-4\text{ dB}$)
  - *Sibilant*: Cut 5–8 kHz ($-5\text{ dB}$)
  - *Airy / Bright*: High-shelf boost 10–16 kHz ($+3.5\text{ dB}$)
  - *Sub Rumble*: High-pass filter cut $< 35\text{ Hz}$
- **Dual Engine (`llm_engine.py`)**: Local Ollama support (`llama3.2` / `qwen2.5`) with zero-latency deterministic offline fallback.

### Phase 3: DSP Execution & Overlap-Add Resynthesis
- **Robert Bristow-Johnson (RBJ) Biquad Mathematics (`biquad.py`)**:
  $$A = 10^{G / 40}, \quad \omega_0 = \frac{2\pi f_0}{f_s}, \quad \alpha = \frac{\sin \omega_0}{2Q}$$
  $$H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}$$
  Supports Peaking, Low-Shelf, High-Shelf, High-Pass, and Notch topologies.
- **STFT Domain Processing (`stft_processor.py`)**:
  - Hann window, $N = 2048$, hop $= 512$ ($75\%$ overlap)
  - Bin gain multiplication: $Y_k = |H(\omega_k)| \cdot X_k$
  - Overlap-add inverse STFT (iSTFT) resynthesis.
- **Resynthesis Coordinator (`dsp_engine.py`)**: Applies stem edits and re-sums stems into `polished_master_mix.wav`.

### Phase 4: Interactive Web UI & Spectrogram Analysis
- **Streamlit Application (`app.py`)**:
  - 4-Stem audio player and mixer.
  - Natural language prompt bar with quick presets.
  - Live JSON parameter inspector & $H(z)$ frequency response graph.
  - Side-by-side Before vs. After STFT Spectrograms.
  - Instant A/B comparison and master WAV export.

---

## 🚀 How to Run

### 1. Run Verification Tests
Each phase includes its own standalone unit test:

```powershell
# Phase 1 Test (Audio synthesis, stem loading & SNR check)
.\.venv\Scripts\python.exe test_phase1.py

# Phase 2 Test (LLM prompt parsing & Pydantic safety clamps)
.\.venv\Scripts\python.exe test_phase2.py

# Phase 3 Test (RBJ biquad math, STFT filtering & resynthesis)
.\.venv\Scripts\python.exe test_phase3.py
```

### 2. Launch the Web Application

Double-click `run_app.bat` or run:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser to: **`http://localhost:8501`**

