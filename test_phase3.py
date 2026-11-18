"""
Phase 3 Verification Script: RBJ Biquad Math, STFT Filtering & Master Resynthesis
Mathematics of Music (MA375TGR)
"""

import os
import sys
import json
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.phase1_separation import load_audio, STEM_NAMES
from src.phase2_llm import EQPlan, FilterEdit, FilterType, StemType
from src.phase3_dsp import BiquadFilter, DSPEngine


def compute_band_power(audio: np.ndarray, f_center: float, f_width: float, fs: int = 44100) -> float:
    """Calculates RMS power (dB) in a narrow frequency band around f_center."""
    freqs = np.fft.rfftfreq(audio.shape[-1], d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(audio, axis=-1))

    # Mask frequencies within [f_center - f_width/2, f_center + f_width/2]
    band_mask = (freqs >= (f_center - f_width / 2.0)) & (freqs <= (f_center + f_width / 2.0))
    if not np.any(band_mask):
        return -100.0

    band_energy = np.mean(fft_mag[:, band_mask] ** 2)
    return 10.0 * np.log10(band_energy + 1e-12)


def run_phase3_test():
    print("=" * 70)
    print("  MA375TGR: PHASE 3 DSP EXECUTION & RESYNTHESIS TEST")
    print("=" * 70)

    # 1. Inspect Exact Mathematical Biquad Transfer Function (Slide 12 RBJ Cookbook)
    print("\n[1] Verifying RBJ Peaking Biquad Math (Slide 12 Equations)...")
    flt_vocal = BiquadFilter(filter_type="peaking", f0_hz=300.0, gain_db=-4.0, q=1.4, fs=44100)
    flt_bass = BiquadFilter(filter_type="peaking", f0_hz=80.0, gain_db=3.0, q=1.0, fs=44100)

    test_freqs = np.array([50.0, 80.0, 150.0, 300.0, 1000.0, 5000.0])
    vocal_resp_db = flt_vocal.magnitude_response_db(test_freqs)
    bass_resp_db = flt_bass.magnitude_response_db(test_freqs)

    vocal_at_300 = flt_vocal.magnitude_response_db(np.array([300.0]))[0]
    bass_at_80 = flt_bass.magnitude_response_db(np.array([80.0]))[0]

    print(f"    - Vocal Filter H(z) at center 300.0 Hz: {vocal_at_300:+.2f} dB (Expected: -4.00 dB)")
    print(f"    - Bass Filter H(z)  at center 80.0 Hz:  {bass_at_80:+.2f} dB (Expected: +3.00 dB)")

    assert abs(vocal_at_300 - (-4.0)) < 0.05, f"Vocal filter math error: {vocal_at_300}"
    assert abs(bass_at_80 - 3.0) < 0.05, f"Bass filter math error: {bass_at_80}"
    print("    ✔ RBJ Biquad coefficients matched closed-form continuous & discrete theory!")

    # 2. Ingest Phase 1 Stems & Phase 2 EQ Plan
    preset_dir = os.path.join("data", "presets", "stock_funk_groove")
    stem_paths = {name: os.path.join(preset_dir, f"{name}.wav") for name in STEM_NAMES}
    for name, p in stem_paths.items():
        assert os.path.exists(p), f"Missing stem: {p}. Run Phase 1 test first!"

    plan_path = os.path.join("data", "output", "test_eq_plan.json")
    if os.path.exists(plan_path):
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_dict = json.load(f)
        eq_plan = EQPlan(**plan_dict)
    else:
        # Fallback inline plan matching slide anchor
        eq_plan = EQPlan(
            user_prompt="The vocal sounds too boxy and muddy, and I want the bass kick to have more punch.",
            edits=[
                FilterEdit(stem=StemType.VOCALS, filter=FilterType.PEAKING, f0_hz=300.0, gain_db=-4.0, q=1.4),
                FilterEdit(stem=StemType.DRUMS, filter=FilterType.PEAKING, f0_hz=80.0, gain_db=3.0, q=1.0)
            ],
            summary="Slide anchor test plan"
        )

    print(f"\n[2] Loaded Phase 2 EQ Plan:")
    print(f"    Summary: {eq_plan.summary}")
    for e in eq_plan.edits:
        print(f"    -> Edit: {e.stem.value:<6} | {e.filter.value:<8} | f0={e.f0_hz:.0f}Hz | gain={e.gain_db:+.1f}dB | Q={e.q:.1f}")

    # 3. Execute STFT-Domain Filtering & Overlap-Add Resynthesis
    print("\n[3] Executing STFT Parametric DSP Engine (Hann N=2048, hop=512)...")
    dsp = DSPEngine(mode="stft", fs=44100)
    output_dir = os.path.join("data", "output", "phase3_dsp_master")
    results = dsp.process_stems(stem_paths, eq_plan, output_dir=output_dir)

    print(f"    ✔ Resynthesized master mix to: {results['master_mix_path']}")
    print(f"    ✔ Master Peak Headroom: {results['master_peak_dbfs']} dBFS")

    # 4. Verify Acoustic Transformations (Before vs. After)
    print("\n[4] Acoustic Validation (Band Power Before vs. After):")
    for name, rep in results["stem_reports"].items():
        print(f"    - Stem [{name.upper():<6}]: {rep['filters_applied']} filters applied | Overall RMS Change: {rep['rms_change_db']:+.2f} dB")

    # Validate Vocal Cut at 300 Hz
    vocal_orig, _ = load_audio(stem_paths["vocals"])
    vocal_proc, _ = load_audio(results["stem_paths"]["vocals"])
    orig_300_pwr = compute_band_power(vocal_orig, f_center=300.0, f_width=100.0)
    proc_300_pwr = compute_band_power(vocal_proc, f_center=300.0, f_width=100.0)
    vocal_delta = proc_300_pwr - orig_300_pwr
    print(f"\n    [Spectral Check] Vocal 300 Hz Band Power:")
    print(f"      Before: {orig_300_pwr:.2f} dB | After: {proc_300_pwr:.2f} dB | Delta: {vocal_delta:+.2f} dB")
    assert vocal_delta < -2.0, f"Vocal 300 Hz cut failed to attenuate band energy: {vocal_delta:.2f} dB"

    # Validate Drum/Kick Boost at 80 Hz
    drum_orig, _ = load_audio(stem_paths["drums"])
    drum_proc, _ = load_audio(results["stem_paths"]["drums"])
    orig_80_pwr = compute_band_power(drum_orig, f_center=80.0, f_width=40.0)
    proc_80_pwr = compute_band_power(drum_proc, f_center=80.0, f_width=40.0)
    drum_delta = proc_80_pwr - orig_80_pwr
    print(f"\n    [Spectral Check] Drum/Kick 80 Hz Band Power:")
    print(f"      Before: {orig_80_pwr:.2f} dB | After: {proc_80_pwr:.2f} dB | Delta: {drum_delta:+.2f} dB")
    assert drum_delta > +1.5, f"Drum 80 Hz boost failed to increase band energy: {drum_delta:.2f} dB"

    # Validate Untouched Stem Preservation (Other)
    other_orig, _ = load_audio(stem_paths["other"])
    other_proc, _ = load_audio(results["stem_paths"]["other"])
    other_diff = np.max(np.abs(other_orig - other_proc))
    print(f"\n    [Integrity Check] Untouched 'Other' Stem Max Sample Difference: {other_diff:.2e}")
    assert other_diff < 1e-6, "Untouched stem was inadvertently altered!"

    print("\n✔ PHASE 3 PASSED: Exact RBJ biquad math and STFT resynthesis fully operational!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase3_test()

