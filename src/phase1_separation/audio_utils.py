"""
Phase 1: Multi-Stem Audio Utilities
Mathematics of Music (MA375TGR)
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import soundfile as sf
from scipy import signal

STEM_NAMES = ["vocals", "drums", "bass", "other"]
DEFAULT_SR = 44100


def load_audio(file_path: str, target_sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as float32 in [-1.0, 1.0].
    Converts stereo to 2-channel array shape (channels, samples).
    Resamples to target_sr if necessary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    data, sr = sf.read(file_path, dtype='float32')

    # Ensure shape is (channels, samples)
    if data.ndim == 1:
        # Mono -> shape (1, samples)
        audio = data[np.newaxis, :]
    else:
        # Stereo (samples, channels) -> (channels, samples)
        audio = data.T

    # Resample if sample rates do not match
    if sr != target_sr:
        num_target_samples = int(round(audio.shape[1] * float(target_sr) / sr))
        resampled_channels = []
        for ch in range(audio.shape[0]):
            resampled_ch = signal.resample(audio[ch], num_target_samples)
            resampled_channels.append(resampled_ch)
        audio = np.stack(resampled_channels, axis=0)
        sr = target_sr

    return audio.astype(np.float32), sr


def save_audio(file_path: str, audio: np.ndarray, sr: int = DEFAULT_SR, normalize: bool = False) -> None:
    """
    Save audio tensor (shape: (channels, samples) or (samples,)) to 16-bit or 24-bit WAV.
    Clips or normalizes to avoid digital distortion.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    out = np.copy(audio)
    if out.ndim == 2:
        # (channels, samples) -> (samples, channels) for soundfile
        out = out.T

    if normalize:
        peak = np.max(np.abs(out))
        if peak > 1e-6:
            out = out / peak * 0.95
    else:
        # Soft-clip or hard-clip to prevent wrap-around
        out = np.clip(out, -1.0, 1.0)

    sf.write(file_path, out, sr, subtype='PCM_16')


def sum_stems(stems: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Sum the 4 stems: x(t) = vocals(t) + drums(t) + bass(t) + other(t).
    Ensures matching lengths and channels.
    """
    if not stems:
        raise ValueError("No stems provided to sum.")

    # Determine common length and channel count
    min_len = min(s.shape[-1] for s in stems.values())
    channels = next(iter(stems.values())).shape[0] if next(iter(stems.values())).ndim > 1 else 1

    mix = np.zeros((channels, min_len), dtype=np.float32)
    for name in STEM_NAMES:
        if name in stems:
            stem = stems[name]
            if stem.ndim == 1:
                stem = stem[np.newaxis, :]
            mix += stem[:, :min_len]

    return mix


def compute_reconstruction_metrics(original_mix: np.ndarray, reconstructed_mix: np.ndarray) -> Dict[str, float]:
    """
    Compute Signal-to-Distortion / Signal-to-Noise Ratio (SNR) in dB and MSE.
    SNR = 10 * log10( ||x||^2 / ||x - x_rec||^2 )
    """
    min_len = min(original_mix.shape[-1], reconstructed_mix.shape[-1])
    orig = original_mix[..., :min_len]
    rec = reconstructed_mix[..., :min_len]

    noise = orig - rec
    p_signal = np.sum(orig ** 2)
    p_noise = np.sum(noise ** 2)

    mse = float(np.mean(noise ** 2))

    if p_noise == 0:
        snr_db = float("inf")
    elif p_signal == 0:
        snr_db = float("-inf")
    else:
        snr_db = float(10.0 * np.log10(p_signal / p_noise))

    return {
        "snr_db": round(snr_db, 2),
        "mse": float(f"{mse:.6e}"),
        "num_samples": min_len
    }

