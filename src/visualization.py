"""
Phase 4: Spectrogram & Filter Response Visualization
MA375TGR | Mathematics of Music
Generates side-by-side Before vs. After Spectrograms and Transfer Function curves.
"""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from src.phase3_dsp.biquad import BiquadFilter


def plot_filter_curves(filters: List[BiquadFilter], title: str = "Cascaded Biquad EQ Response") -> plt.Figure:
    """Plots combined transfer function H(w) in dB from 20 Hz to 20,000 Hz."""
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
    total_gain_db = np.zeros_like(freqs)

    # Plot individual filters
    colors = ["#38bdf8", "#ec4899", "#10b981", "#f59e0b"]
    for idx, flt in enumerate(filters):
        mag_db = flt.magnitude_response_db(freqs)
        total_gain_db += mag_db
        color = colors[idx % len(colors)]
        ax.plot(
            freqs, mag_db,
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.6,
            label=f"{flt.filter_type.capitalize()} ({flt.gain_db:+.1f}dB @ {flt.f0:.0f}Hz, Q={flt.q})"
        )

    # Plot combined response
    ax.plot(freqs, total_gain_db, linewidth=2.4, color="#00f2fe", label="Total Cascaded EQ H(z)")

    ax.axhline(0, color="#64748b", linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(20, 20000)
    ax.set_ylim(-14, 14)
    ax.set_xlabel("Frequency (Hz)", color="#e2e8f0", fontsize=10)
    ax.set_ylabel("Gain (dB)", color="#e2e8f0", fontsize=10)
    ax.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#cbd5e1", which="both")
    ax.grid(True, which="both", color="#334155", linestyle="--", alpha=0.5)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8, loc="upper right")

    plt.tight_layout()
    return fig


def plot_before_after_spectrogram(
    orig_audio: np.ndarray,
    proc_audio: np.ndarray,
    fs: int = 44100,
    stem_name: str = "Stem"
) -> plt.Figure:
    """
    Renders side-by-side Before and After STFT Spectrograms.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=120, sharey=True)
    fig.patch.set_facecolor("#0e1117")

    # Take mono mix if stereo
    sig1 = orig_audio[0] if orig_audio.ndim > 1 else orig_audio
    sig2 = proc_audio[0] if proc_audio.ndim > 1 else proc_audio

    # Compute STFT
    nperseg = 1024
    f1, t1, Sxx1 = signal.spectrogram(sig1, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    f2, t2, Sxx2 = signal.spectrogram(sig2, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    Sxx1_db = 10 * np.log10(Sxx1 + 1e-10)
    Sxx2_db = 10 * np.log10(Sxx2 + 1e-10)
    vmin = max(np.percentile(Sxx1_db, 20), -80)
    vmax = np.percentile(Sxx1_db, 99.5)

    # Subplot 1: Before
    im1 = ax1.pcolormesh(t1, f1, Sxx1_db, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
    ax1.set_facecolor("#161b22")
    ax1.set_title(f"BEFORE: {stem_name.capitalize()} (Original)", color="#f8fafc", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Time (s)", color="#e2e8f0", fontsize=9)
    ax1.set_ylabel("Frequency (Hz)", color="#e2e8f0", fontsize=9)
    ax1.set_yscale("symlog", linthresh=200)
    ax1.set_ylim(40, 18000)
    ax1.tick_params(colors="#cbd5e1")

    # Subplot 2: After
    im2 = ax2.pcolormesh(t2, f2, Sxx2_db, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
    ax2.set_facecolor("#161b22")
    ax2.set_title(f"AFTER: {stem_name.capitalize()} (Processed)", color="#00f2fe", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (s)", color="#e2e8f0", fontsize=9)
    ax2.set_yscale("symlog", linthresh=200)
    ax2.set_ylim(40, 18000)
    ax2.tick_params(colors="#cbd5e1")

    # Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Magnitude (dB)", color="#e2e8f0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#e2e8f0")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cbd5e1')

    return fig

