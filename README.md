# 🎛️ Multi-Stem Audio Separation and Parametric Equalization with LLM-Driven Natural Language Control

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Demucs v4](https://img.shields.io/badge/Separation-HTDemucs%20v4-orange.svg)](https://github.com/facebookresearch/demucs)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Llama--3.2%20%7C%20Ollama-black.svg)](https://ollama.com/)
[![DSP](https://img.shields.io/badge/Audio%20DSP-RBJ%20Biquad%20IIR%20%2B%20STFT-green.svg)](https://www.w3.org/TR/audio-eq-cookbook/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

**Course**: MA375TGR | Mathematics of Music  
**Institution**: RV College of Engineering  
**Authors**:
- **Bharath Kumar MS** (1RV23IS034)
- **Dharshan SN** (1RV23IS044)
- **Dheeraj R** (1RV23IS045)

---

## 🎵 Project Overview

Shaping the tone and balance of a musical mix traditionally demands years of audio engineering training—maneuvering through center frequencies ($f_0$), $Q$ bandwidth factors, decibel gains, filter topologies, and masking thresholds. 

This project bridges that gap by replacing technical audio knobs with natural language:
> *“The vocal sounds too boxy and harsh, and I want the kick drum to have more punch and low end.”*  
> **$\rightarrow$ Vocals: Peaking $-4.0\text{ dB}$ at $300\text{ Hz}$ ($Q=1.5$), Notch $-6.0\text{ dB}$ at $2500\text{ Hz}$**  
> **$\rightarrow$ Drums: Peaking $+4.0\text{ dB}$ at $80\text{ Hz}$ ($Q=1.2$), Low-Shelf $+3.0\text{ dB}$ at $45\text{ Hz}$**

### 🧠 Core Engineering Philosophy
**"AI as the high-level semantic controller, deterministic Digital Signal Processing (DSP) as the audio engine."**
- The neural LLM never touches raw audio samples directly and cannot hallucinate audio artifacts.
- Every sound wave is filtered using closed-form, mathematically provable **Robert Bristow-Johnson (RBJ) Biquad IIR filters** and **STFT Overlap-Add (OLA) resynthesis**.

---

## 🏛️ System Architecture

```
                                  [ User Audio Track x(t) ]
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │  Phase 1: Source Separation   │
                             │  (Demucs v4 Hybrid Conv-TRF)  │
                             └──────────────┬────────────────┘
                                            │
               ┌──────────────┬─────────────┴────────────┬──────────────┐
               ▼              ▼                          ▼              ▼
          [ Vocals ]      [ Drums ]                   [ Bass ]      [ Other ]
               │              │                          │              │
               │              │   [ Natural Language ]   │              │
               │              │        Prompt            │              │
               │              │          │               │              │
               │              │          ▼               │              │
               │              │  ┌───────────────┐       │              │
               │              │  │  Phase 2: LLM │       │              │
               │              │  │  (Llama 3.2)  │       │              │
               │              │  └───────┬───────┘       │              │
               │              │          │ Filter Plan   │              │
               ▼              ▼          ▼ (JSON)        ▼              ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │                 Phase 3: Deterministic DSP Engine                   │
        │      • RBJ Biquad IIR Difference Equations: H(z)                    │
        │      • STFT Overlap-Add Processing (N=2048, Hann, 75% overlap)      │
        │      • Transparent Soft-Limiting & Summing Bus                      │
        └──────────────────────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │    Polished Master Mix      │
                            │  + Null Test Delta Audio    │
                            │  + Before/After Spectrogram │
                            └─────────────────────────────┘
```

---

## 📋 Prerequisites

Ensure your system meets the following requirements before cloning:
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: `3.10` or `3.11` (Python 3.12+ may have PyTorch wheel constraints on some platforms).
- **RAM**: Minimum 8 GB (16 GB recommended for full-track Demucs neural separation).
- **GPU (Optional)**: NVIDIA GPU with CUDA support for 5x faster separation (falls back automatically to CPU).
- **Ollama (Optional for local LLM)**: Download from [ollama.com](https://ollama.com) (or use built-in Gemini API / Offline Expert Rules).

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/dharshan17sn/Mom.git
cd Mom
```

### 2. Set Up a Python Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

Install all core packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for PyTorch with CUDA (Optional for GPU Acceleration):**  
> If you have an NVIDIA graphics card, install CUDA-accelerated PyTorch:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
> Otherwise, the standard CPU version installed by `requirements.txt` runs smoothly.

---

### 4. Set Up the LLM Engine (3 Flexible Choices)

You have **three ways** to power the natural language reasoning:

#### Option A: Local Ollama (Recommended, 100% Free & Private)
1. Install Ollama from [ollama.com](https://ollama.com/download).
2. Start the Ollama service (it runs in your system tray or run `ollama serve`).
3. Pull the lightweight `llama3.2` model (2.0 GB):
   ```bash
   ollama pull llama3.2
   ```
   *(Note: You can also pull models directly from inside the Streamlit web dashboard!)*

#### Option B: Google Gemini API (Cloud, Zero Local RAM Required)
- If you don't want to install Ollama, simply select **"Google Gemini API"** in the app sidebar and paste your Gemini API key.

#### Option C: Deterministic Offline Rules (Zero Download, Instant)
- Select **"Deterministic Psychoacoustic Rules"** in the sidebar. This uses a closed-form expert rule engine derived from audio mastering frequency charts (Slide 11).

---

### 5. Launch the Web Application

#### Windows (One-Click):
Double-click `run_app.bat` or run:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

#### Linux / macOS:
```bash
streamlit run app.py
```

The application will launch automatically in your browser at:
```
http://localhost:8501
```

---

## 🎧 Step-by-Step Usage Guide

### Step 1: Select Audio Input
In the left sidebar under **1. Audio Input Track**:
- **Custom Upload**: Upload your own song (`.wav` or `.mp3`). Real-time Demucs will split it into `vocals`, `drums`, `bass`, and `other`.
- **High-Contrast Benchmark (Audible Preset)**: Built-in 8-second multi-stem track specifically designed with a deliberately boxy 300 Hz vocal formant, muddy low-mids, and harsh 3 kHz drum transients so EQ changes are immediately audible.
- **Stock Funk / Lo-Fi Groove**: A clean, balanced multi-stem production preset.

### Step 2: Choose LLM Inference Engine
- Under **2. AI Reasoning Engine**, choose your engine (`Ollama`, `Gemini`, or `Expert Rules`).
- If using Ollama, pick `llama3.2:latest` from the auto-detected model list.

### Step 3: Enter Your Audio Request
In the main interface, type any natural language audio mixing instruction or choose a quick prompt button:
- *"Make the drums punchier with more sub bass and cut harsh frequencies on vocals"*
- *"The vocal is boxy and nasal, clean it up and add top-end air"*
- *"Warm up the bass and remove low-end rumble from the vocals"*

Click **"Generate Parametric EQ Plan"**. Within 1–2 seconds:
- The LLM outputs strict, Pydantic-validated Biquad parameters ($f_0$, Gain, $Q$, filter topology).
- Theoretical transfer function plots $|H(e^{j\omega})|$ are displayed in real time.

### Step 4: Apply Filters & Audition Results
Click **"Execute DSP Equalization & Resynthesize Master"**:
1. **Interactive Audio Players**: Audition individual isolated stems (Before vs. After).
2. **Master A/B Audition**: Switch seamlessly between the raw mix and the polished master.
3. **The Delta / Null Test**: Audition the **difference signal** $[x_{\text{after}}(t) - x_{\text{before}}(t)]$. If no EQ was applied, this is absolute digital silence; anything you hear is *strictly* what the LLM sculpted!
4. **Spectrogram Comparison**: Inspect the side-by-side Before vs. After STFT spectrograms displaying frequency attenuation and resonance cuts.

---

## 🧪 Running Standalone Verification Tests

To verify that each mathematical phase passes unit tests independently:

```bash
# Test Phase 1: Audio synthesis, stem integrity, and SNR reconstruction check
python test_phase1.py

# Test Phase 2: LLM JSON schema validation, Pydantic boundary clamping
python test_phase2.py

# Test Phase 3: RBJ Biquad frequency math, STFT overlap-add resynthesis
python test_phase3.py
```

---

## 📁 Repository Structure

```
Mom/
├── app.py                      # Interactive Streamlit Web Application
├── run_app.bat                 # One-click Windows launch script
├── requirements.txt            # Python package dependencies
├── test_phase1.py              # Unit tests for Stem Separation
├── test_phase2.py              # Unit tests for LLM Reasoning & Schema
├── test_phase3.py              # Unit tests for Biquad DSP & Resynthesis
│
├── src/
│   ├── phase1_separation/      # Phase 1: Stem Separation & Preset Audio
│   │   ├── separator.py        # Real Demucs neural wrapper with chunk caching
│   │   ├── stock_generator.py  # Algorithmic multi-track audio synthesizer
│   │   └── benchmark_generator.py # High-contrast benchmark audio creator
│   │
│   ├── phase2_llm/             # Phase 2: LLM Reasoning Engine
│   │   ├── llm_engine.py       # Ollama SDK client & Gemini cloud integration
│   │   ├── schema.py           # Pydantic safety validators & JSON schema
│   │   └── psychoacoustics.py  # Audio engineering prior frequency maps
│   │
│   ├── phase3_dsp/             # Phase 3: Mathematical Signal Processing
│   │   ├── biquad.py           # RBJ Biquad filter mathematics & H(z)
│   │   ├── stft_processor.py   # Short-Time Fourier Transform & Overlap-Add
│   │   └── dsp_engine.py       # Multi-stem processor & soft-limiting master bus
│   │
│   └── visualization/          # Phase 4: Data Visualization
│       ├── biquad_plots.py     # Matplotlib frequency response curves
│       └── spectrogram.py      # Before / After STFT spectrogram plots
│
└── data/
    ├── presets/                # Pre-rendered 44.1 kHz WAV stems
    ├── uploads/                # User uploaded tracks
    └── output/                 # Processed master mixes, stems, and null tests
```

---

## 📐 Mathematical Formulation

### 1. Source Separation Conservation Law
For any time-domain audio mix $x(t)$, the separated stems must satisfy:
$$x(t) \approx \sum_{k \in \{\text{vocals, drums, bass, other}\}} s_k(t)$$
Verified via Signal-to-Noise Ratio (SNR):
$$\text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{\sum_t x(t)^2}{\sum_t \left( x(t) - \sum_k s_k(t) \right)^2} \right) > 70\text{ dB}$$

### 2. Robert Bristow-Johnson (RBJ) Biquad Transfer Function
Each parametric filter edit is realized as a digital second-order IIR biquad:
$$H(z) = \frac{Y(z)}{X(z)} = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}$$
Intermediate variables given sampling rate $f_s$, center frequency $f_0$, quality factor $Q$, and gain $G\text{ (dB)}$:
$$A = 10^{\frac{G}{40}}, \quad \omega_0 = \frac{2\pi f_0}{f_s}, \quad \alpha = \frac{\sin(\omega_0)}{2Q}$$

**Peaking EQ Coefficients:**
$$b_0 = 1 + \alpha A, \quad b_1 = -2\cos(\omega_0), \quad b_2 = 1 - \alpha A$$
$$a_0 = 1 + \frac{\alpha}{A}, \quad a_1 = -2\cos(\omega_0), \quad a_2 = 1 - \frac{\alpha}{A}$$

### 3. Overlap-Add (OLA) Resynthesis
Filtering in the spectral domain satisfies:
$$Y(m, k) = H(k) \cdot X(m, k)$$
$$y(t) = \sum_{m} w(t - mH) \cdot \text{iSTFT}\{Y(m, k)\}$$
with Hann window $w$, window length $N=2048$, and hop size $H = N/4 = 512$ ($75\%$ overlap), guaranteeing the **Constant Overlap-Add (COLA)** constraint.

---

## 🛠️ Troubleshooting

- **`ollama : The term 'ollama' is not recognized`**:
  - Close and reopen your terminal after installing Ollama, or run:
    ```powershell
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
    ```
- **Demucs takes long on long songs**:
  - By default, Demucs runs on CPU if no NVIDIA CUDA GPU is present. An 8–15 second clip processes in ~10–15 seconds; a full 4-minute track may take 1–2 minutes on CPU.
- **HuggingFace Warning**:
  - `Warning: You are sending unauthenticated requests to the HF Hub...` is completely normal and can be safely ignored. Model weights download and cache automatically.

---

## 📄 License & Academic Attribution
Developed as an academic submission for **MA375TGR | Mathematics of Music**, RV College of Engineering.  
DSP implementations are based on Robert Bristow-Johnson's *Audio EQ Cookbook*. Stem separation leverages Meta AI's *Demucs v4* architecture. Local language reasoning is powered by Meta's *Llama 3.2*.
