"""
Phase 1: Multi-Stem Stock Audio Generator
Generates a 4-stem multitrack preset (Vocals, Drums, Bass, Other) and mix.
Designed specifically with frequency characteristics that demonstrate
the project's psychoacoustic targets (300 Hz boxy vocal, 80 Hz punchy bass kick, etc.)
"""

import os
from typing import Dict, Tuple
import numpy as np
from scipy import signal
from .audio_utils import DEFAULT_SR, save_audio, sum_stems


def generate_kick(sr: int, duration: float = 0.35) -> np.ndarray:
    """Pitch-swept kick drum: starts at 130 Hz and sweeps down to 50 Hz."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq_env = 50.0 + 80.0 * np.exp(-t * 28.0)
    phase = 2.0 * np.pi * np.cumsum(freq_env) / sr
    amp_env = np.exp(-t * 12.0)
    kick = np.sin(phase) * amp_env
    # Add slight sub click
    click = np.random.normal(0, 0.2, len(t)) * np.exp(-t * 80.0)
    return (kick + click).astype(np.float32)


def generate_snare(sr: int, duration: float = 0.25) -> np.ndarray:
    """Snare drum: 200 Hz tone body + shaped white noise burst."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    body = np.sin(2 * np.pi * 195 * t) * np.exp(-t * 22.0)
    noise = np.random.normal(0, 1.0, len(t)) * np.exp(-t * 14.0)
    # Bandpass the noise between 1 kHz and 8 kHz
    sos = signal.butter(4, [1000, 8000], btype='bandpass', fs=sr, output='sos')
    noise_filtered = signal.sosfilt(sos, noise)
    return (0.45 * body + 0.55 * noise_filtered).astype(np.float32)


def generate_hihat(sr: int, duration: float = 0.08) -> np.ndarray:
    """Hi-hat: High-pass filtered noise transient."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.normal(0, 1.0, len(t)) * np.exp(-t * 45.0)
    sos = signal.butter(4, 7000, btype='highpass', fs=sr, output='sos')
    return (0.6 * signal.sosfilt(sos, noise)).astype(np.float32)


def generate_drums_stem(sr: int, total_duration: float = 8.0, bpm: float = 100.0) -> np.ndarray:
    """Generates stereo drum track (Kick on beats 1 & 3, Snare on 2 & 4, 8th-note Hi-Hats)."""
    n_samples = int(sr * total_duration)
    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)

    beat_dur = 60.0 / bpm
    kick = generate_kick(sr)
    snare = generate_snare(sr)
    hihat = generate_hihat(sr)

    current_t = 0.0
    beat_idx = 0

    while current_t + 0.4 < total_duration:
        sample_idx = int(current_t * sr)

        # Kick on beats 0, 2 (1 & 3 in musical terms)
        if beat_idx % 4 in (0, 2):
            k_len = min(len(kick), n_samples - sample_idx)
            left[sample_idx:sample_idx + k_len] += kick[:k_len] * 0.8
            right[sample_idx:sample_idx + k_len] += kick[:k_len] * 0.8

        # Snare on beats 1, 3 (2 & 4 in musical terms)
        if beat_idx % 4 in (1, 3):
            s_len = min(len(snare), n_samples - sample_idx)
            left[sample_idx:sample_idx + s_len] += snare[:s_len] * 0.7
            right[sample_idx:sample_idx + s_len] += snare[:s_len] * 0.7

        # 8th-note hi-hat (on beat and off-beat)
        h_len = min(len(hihat), n_samples - sample_idx)
        left[sample_idx:sample_idx + h_len] += hihat[:h_len] * 0.45
        right[sample_idx:sample_idx + h_len] += hihat[:h_len] * 0.40

        # Off-beat hi-hat
        off_idx = sample_idx + int(beat_dur * 0.5 * sr)
        if off_idx + len(hihat) < n_samples:
            h_len_off = min(len(hihat), n_samples - off_idx)
            left[off_idx:off_idx + h_len_off] += hihat[:h_len_off] * 0.35
            right[off_idx:off_idx + h_len_off] += hihat[:h_len_off] * 0.42

        beat_idx += 1
        current_t += beat_dur

    return np.stack([left, right], axis=0) * 0.7


def generate_bass_stem(sr: int, total_duration: float = 8.0, bpm: float = 100.0) -> np.ndarray:
    """
    Generates bass track with deep fundamentals centered around 60-100 Hz (e.g. E1 = 41.2 Hz, G1 = 49 Hz, A1 = 55 Hz, D2 = 73.4 Hz).
    Direct target for "punchy bass kick / sub boost at 80 Hz".
    """
    n_samples = int(sr * total_duration)
    left = np.zeros(n_samples, dtype=np.float32)
    beat_dur = 60.0 / bpm

    # Bassline notes (Hz)
    notes = [55.0, 55.0, 65.4, 73.4, 82.4, 73.4, 65.4, 49.0]
    note_duration = beat_dur

    for i, freq in enumerate(notes * int(np.ceil(total_duration / (len(notes) * note_duration)))):
        start_t = i * note_duration
        if start_t >= total_duration:
            break
        s_idx = int(start_t * sr)
        dur = min(note_duration * 0.85, total_duration - start_t)
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)

        # Fundamental + 2nd harmonic (warmth) + subtle 3rd harmonic
        env = (1.0 - np.exp(-t * 30.0)) * np.exp(-t * 2.5)
        wave = (
            0.7 * np.sin(2 * np.pi * freq * t) +
            0.25 * np.sin(2 * np.pi * 2 * freq * t) +
            0.05 * np.sin(2 * np.pi * 3 * freq * t)
        ) * env

        w_len = min(len(wave), n_samples - s_idx)
        left[s_idx:s_idx + w_len] += wave[:w_len] * 0.65

    # Bass is typically centered in stereo
    return np.stack([left, left], axis=0)


def generate_vocals_stem(sr: int, total_duration: float = 8.0, bpm: float = 100.0) -> np.ndarray:
    """
    Generates formant-rich vocal-like lead line.
    Has pronounced resonance around 300 Hz (boxy/muddy) and 1.8 kHz,
    exactly fulfilling the slide's prompt:
    "The vocal sounds too boxy and muddy, vocal: -4 dB at 300 Hz".
    """
    n_samples = int(sr * total_duration)
    left = np.zeros(n_samples, dtype=np.float32)
    beat_dur = 60.0 / bpm

    # Vocal melody notes: A3 (220 Hz), C4 (261.6 Hz), D4 (293.7 Hz), E4 (329.6 Hz)
    melody = [220.0, 261.63, 293.66, 329.63, 293.66, 261.63, 220.0, 196.0]
    note_dur = beat_dur * 2.0  # 2 beats per vocal phrase

    for i, f0 in enumerate(melody * int(np.ceil(total_duration / (len(melody) * note_dur)))):
        start_t = i * note_dur
        if start_t >= total_duration:
            break
        s_idx = int(start_t * sr)
        dur = min(note_dur * 0.9, total_duration - start_t)
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)

        # Add vibrato (5.5 Hz pitch modulation)
        vibrato = 1.0 + 0.015 * np.sin(2 * np.pi * 5.5 * t)
        inst_f0 = f0 * vibrato
        phase = 2 * np.pi * np.cumsum(inst_f0) / sr

        # Glottal pulse simulation (harmonic rich buzz)
        raw_vocal = np.zeros_like(t)
        for h in range(1, 12):
            raw_vocal += (1.0 / h) * np.sin(h * phase)

        # Vocal tract formants:
        # F1 around 300 Hz ("boxy / chest tone" - intentionally elevated to test EQ!)
        # F2 around 1800 Hz
        sos_f1 = signal.iirpeak(300.0, Q=2.5, fs=sr)
        sos_f2 = signal.iirpeak(1800.0, Q=4.0, fs=sr)
        formant1 = signal.lfilter(sos_f1[0], sos_f1[1], raw_vocal) * 1.8
        formant2 = signal.lfilter(sos_f2[0], sos_f2[1], raw_vocal) * 0.8

        vocal_sound = (formant1 + formant2)

        # Smooth attack and release envelope
        env = np.sin(np.pi * (t / dur)) ** 1.5
        vocal_sound = vocal_sound * env

        v_len = min(len(vocal_sound), n_samples - s_idx)
        left[s_idx:s_idx + v_len] += vocal_sound[:v_len] * 0.45

    # Gentle stereo widening
    right = np.roll(left, int(sr * 0.008))  # 8ms Haas effect delay
    return np.stack([left, right], axis=0)


def generate_other_stem(sr: int, total_duration: float = 8.0, bpm: float = 100.0) -> np.ndarray:
    """
    Generates harmonic keyboard/synth chord pad (Am7 -> Dm7 -> Em7).
    Fills mid-range and highs ('other' stem in Demucs).
    """
    n_samples = int(sr * total_duration)
    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)
    beat_dur = 60.0 / bpm

    # Chords: Am7 (A3, C4, E4, G4), Fmaj7 (F3, A3, C4, E4), G7 (G3, B3, D4, F4), Em7 (E3, G3, B3, D4)
    chords = [
        [220.0, 261.63, 329.63, 392.0],
        [174.61, 220.0, 261.63, 329.63],
        [196.0, 246.94, 293.66, 349.23],
        [164.81, 196.0, 246.94, 293.66]
    ]
    chord_dur = beat_dur * 4.0  # 1 measure per chord

    for i, chord in enumerate(chords * int(np.ceil(total_duration / (len(chords) * chord_dur)))):
        start_t = i * chord_dur
        if start_t >= total_duration:
            break
        s_idx = int(start_t * sr)
        dur = min(chord_dur * 0.95, total_duration - start_t)
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        env = (1.0 - np.exp(-t * 8.0)) * np.exp(-t * 0.9)

        pad_left = np.zeros_like(t)
        pad_right = np.zeros_like(t)

        for freq in chord:
            # Gentle detuned stereo oscillators
            pad_left += 0.25 * np.sin(2 * np.pi * (freq * 0.999) * t)
            pad_right += 0.25 * np.sin(2 * np.pi * (freq * 1.001) * t)

        pad_left *= env
        pad_right *= env

        p_len = min(len(pad_left), n_samples - s_idx)
        left[s_idx:s_idx + p_len] += pad_left[:p_len] * 0.4
        right[s_idx:s_idx + p_len] += pad_right[:p_len] * 0.4

    return np.stack([left, right], axis=0)


def create_stock_preset(
    output_dir: str,
    duration: float = 8.0,
    sr: int = DEFAULT_SR,
    bpm: float = 100.0
) -> Tuple[str, Dict[str, str]]:
    """
    Creates complete 4-stem stock preset in output_dir:
    - vocals.wav
    - drums.wav
    - bass.wav
    - other.wav
    - mixture.wav (normalized sum of all 4 stems)
    Returns mixture path and dictionary of stem paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Synthesizing 4-stem stock audio preset ({duration}s at {sr} Hz)...")
    drums = generate_drums_stem(sr, duration, bpm)
    bass = generate_bass_stem(sr, duration, bpm)
    vocals = generate_vocals_stem(sr, duration, bpm)
    other = generate_other_stem(sr, duration, bpm)

    stems = {
        "drums": drums,
        "bass": bass,
        "vocals": vocals,
        "other": other
    }

    # Sum to create the mixture: x(t) = vocals + drums + bass + other
    mixture = sum_stems(stems)

    # Master normalization to avoid digital clipping with comfortable headroom (-1 dB)
    peak = np.max(np.abs(mixture))
    scale = 0.89 / peak if peak > 0.89 else 1.0

    stem_paths = {}
    for name, data in stems.items():
        scaled_stem = data * scale
        path = os.path.join(output_dir, f"{name}.wav")
        save_audio(path, scaled_stem, sr=sr)
        stem_paths[name] = path

    mix_path = os.path.join(output_dir, "mixture.wav")
    save_audio(mix_path, mixture * scale, sr=sr)

    return mix_path, stem_paths


def create_high_contrast_preset(
    output_dir: str,
    duration: float = 8.0,
    sr: int = DEFAULT_SR,
    bpm: float = 100.0
) -> Tuple[str, Dict[str, str]]:
    """
    Creates an EXAGGERATED 'high-contrast' benchmark preset where acoustic defects
    are made prominent so anyone can immediately hear the EQ difference:
    - Vocals: Strongly exaggerated boxy/muddy resonance at 300 Hz (+10 dB peak).
    - Drums: Piercing 3.5 kHz cymbal harshness + light sub kick.
    - Bass: Deep 80 Hz fundamental.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Synthesizing High-Contrast Benchmark Preset ({duration}s at {sr} Hz)...")

    drums = generate_drums_stem(sr, duration, bpm)
    bass = generate_bass_stem(sr, duration, bpm)
    vocals = generate_vocals_stem(sr, duration, bpm)
    other = generate_other_stem(sr, duration, bpm)

    # Inject deliberate, heavy +10 dB boxiness into the vocal around 300 Hz
    # so the vocal sounds conspicuously muffled / cupped in hands
    sos_mud = signal.iirpeak(300.0, Q=2.0, fs=sr)
    for ch in range(vocals.shape[0]):
        mud_bump = signal.lfilter(sos_mud[0], sos_mud[1], vocals[ch]) * 2.8
        vocals[ch] = vocals[ch] * 0.7 + mud_bump * 0.9

    # Add sharp 3.5 kHz bite to drums
    sos_harsh = signal.iirpeak(3500.0, Q=3.0, fs=sr)
    for ch in range(drums.shape[0]):
        harsh_bump = signal.lfilter(sos_harsh[0], sos_harsh[1], drums[ch]) * 1.5
        drums[ch] = drums[ch] * 0.8 + harsh_bump * 0.6

    stems = {
        "drums": drums,
        "bass": bass,
        "vocals": vocals,
        "other": other
    }

    mixture = sum_stems(stems)
    peak = np.max(np.abs(mixture))
    scale = 0.88 / peak if peak > 0.88 else 1.0

    stem_paths = {}
    for name, data in stems.items():
        scaled_stem = data * scale
        path = os.path.join(output_dir, f"{name}.wav")
        save_audio(path, scaled_stem, sr=sr)
        stem_paths[name] = path

    mix_path = os.path.join(output_dir, "mixture.wav")
    save_audio(mix_path, mixture * scale, sr=sr)

    return mix_path, stem_paths

