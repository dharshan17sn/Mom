"""
Phase 3: STFT & Overlap-Add Spectral Processing
MA375TGR | Mathematics of Music
Implements STFT analysis (Hann window, N=2048, hop=512), bin gain multiplication,
and iSTFT resynthesis per Slide 12.
"""

from typing import List
import numpy as np
from scipy import signal
from .biquad import BiquadFilter

DEFAULT_N_FFT = 2048
DEFAULT_HOP = 512


def apply_spectral_eq(
    audio: np.ndarray,
    filters: List[BiquadFilter],
    fs: int = 44100,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    use_magnitude_only: bool = True
) -> np.ndarray:
    """
    Applies cascaded biquad filters in the STFT domain:
    1. STFT with Hann window (N=2048, hop=512)
    2. Bin gain: Yk = |H(wk)| * Xk (per Slide 12)
    3. iSTFT with overlap-add synthesis
    """
    if not filters:
        return np.copy(audio)

    # Ensure shape is 2D: (channels, samples)
    is_1d = audio.ndim == 1
    if is_1d:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio

    channels, n_samples = audio_2d.shape

    # STFT frequency bins: k * fs / N for k = 0 ... N//2
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # Calculate cascaded frequency response across all filters
    h_total = np.ones(len(freq_bins), dtype=np.complex128)
    for flt in filters:
        h_total *= flt.frequency_response(freq_bins)

    if use_magnitude_only:
        # Per slide 12: Bin gain Yk = |H(wk)| * Xk
        bin_gain = np.abs(h_total)[:, np.newaxis]  # shape (num_bins, 1)
    else:
        bin_gain = h_total[:, np.newaxis]

    processed_channels = []
    window = signal.windows.hann(n_fft, sym=False)

    for ch in range(channels):
        # 1. Forward STFT
        _, _, Zxx = signal.stft(
            audio_2d[ch],
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary='zeros',
            padded=True
        )

        # 2. Bin gain application
        Zxx_filtered = Zxx * bin_gain

        # 3. Inverse STFT with overlap-add
        _, x_resynth = signal.istft(
            Zxx_filtered,
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary='zeros'
        )

        # Match exact input length
        x_resynth = x_resynth[:n_samples]
        processed_channels.append(x_resynth.astype(np.float32))

    out = np.stack(processed_channels, axis=0)
    return out[0] if is_1d else out


def apply_time_domain_eq(
    audio: np.ndarray,
    filters: List[BiquadFilter]
) -> np.ndarray:
    """
    Applies cascaded biquad filters directly in time domain using SOS (Second-Order Sections).
    Provided as an alternative / comparative baseline to STFT domain processing.
    """
    if not filters:
        return np.copy(audio)

    # Combine all biquad SOS sections
    sos_list = [flt.get_sos()[0] for flt in filters]
    sos = np.array(sos_list, dtype=np.float64)

    is_1d = audio.ndim == 1
    if is_1d:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio

    channels, n_samples = audio_2d.shape
    processed = []
    for ch in range(channels):
        filtered_ch = signal.sosfilt(sos, audio_2d[ch])
        processed.append(filtered_ch.astype(np.float32))

    out = np.stack(processed, axis=0)
    return out[0] if is_1d else out

