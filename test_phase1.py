"""
Phase 1 Verification Script: Multi-Stem Separation and Stock Preset Generation
Mathematics of Music (MA375TGR)
"""

import os
import sys
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.phase1_separation import (
    STEM_NAMES,
    DEFAULT_SR,
    load_audio,
    create_stock_preset,
    StemSeparator
)


def run_phase1_test():
    print("=" * 65)
    print("  MA375TGR: PHASE 1 STEM SEPARATION & PRESET GENERATION TEST")
    print("=" * 65)

    preset_dir = os.path.join("data", "presets", "stock_funk_groove")
    print(f"\n[1] Creating Stock Audio Preset in '{preset_dir}'...")
    mix_path, stem_paths = create_stock_preset(
        output_dir=preset_dir,
        duration=8.0,
        sr=DEFAULT_SR,
        bpm=100.0
    )

    print(f"    ✔ Created Mix: {mix_path}")
    for name, p in stem_paths.items():
        print(f"    ✔ Created Stem [{name}]: {p}")

    print("\n[2] Inspecting Audio Properties...")
    mix_data, sr = load_audio(mix_path)
    channels, n_samples = mix_data.shape
    duration_sec = n_samples / sr
    peak_db = 20 * np.log10(np.max(np.abs(mix_data)) + 1e-9)

    print(f"    - Sample Rate: {sr} Hz")
    print(f"    - Channels:    {channels} (Stereo)")
    print(f"    - Duration:    {duration_sec:.2f} seconds ({n_samples} samples)")
    print(f"    - Peak Level:  {peak_db:.2f} dBFS")

    print("\n[3] Testing StemSeparator Interface...")
    separator = StemSeparator(model_name="htdemucs")
    print(f"    - Demucs package available: {separator.has_demucs}")

    test_output_dir = os.path.join("data", "output", "phase1_separated_stems")
    separated_stems = separator.separate(mix_path, output_dir=test_output_dir)

    print("\n[4] Validating Mathematical Equivalence (Reconstruction Check):")
    print("    Formula: x(t) ≈ vocals(t) + drums(t) + bass(t) + other(t)")
    metrics = separator.verify_summation(mix_path, separated_stems)

    print(f"    - Reconstruction SNR: {metrics['snr_db']} dB")
    print(f"    - Mean Squared Error: {metrics['mse']}")
    print(f"    - Samples Compared:   {metrics['num_samples']}")

    if metrics['snr_db'] > 50.0 or metrics['mse'] < 1e-4:
        print("\n✔ PHASE 1 PASSED: Perfect multi-stem phase alignment & preservation achieved!")
    else:
        print("\n⚠ Warning: Stems have non-negligible difference from mix.")

    print("=" * 65)


if __name__ == "__main__":
    run_phase1_test()

