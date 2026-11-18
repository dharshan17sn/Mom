"""
Phase 3: DSP Pipeline Engine & Resynthesis Coordinator
MA375TGR | Mathematics of Music
Ingests EQPlan from Phase 2 and Stems from Phase 1, executes biquad filtering
in STFT domain, resynthesizes each stem, and sums into polished master mix.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np

from src.phase1_separation.audio_utils import load_audio, save_audio, sum_stems, STEM_NAMES, DEFAULT_SR
from src.phase2_llm.schema import EQPlan, FilterEdit, StemType
from .biquad import BiquadFilter
from .stft_processor import apply_spectral_eq, apply_time_domain_eq


class DSPEngine:
    """
    Executes deterministic audio processing pipeline based on LLM-derived EQPlan.
    """

    def __init__(self, mode: str = "stft", fs: int = DEFAULT_SR):
        self.mode = mode.lower()  # "stft" or "sos"
        self.fs = fs

    def process_stems(
        self,
        stem_paths: Dict[str, str],
        eq_plan: EQPlan,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Executes EQ modifications per stem and resynthesizes the polished master mix.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load all stems
        loaded_stems = {}
        for name in STEM_NAMES:
            if name in stem_paths and os.path.exists(stem_paths[name]):
                audio, sr = load_audio(stem_paths[name], target_sr=self.fs)
                loaded_stems[name] = audio

        # 2. Group filters by stem
        stem_filters: Dict[str, List[BiquadFilter]] = {name: [] for name in STEM_NAMES}
        master_filters: List[BiquadFilter] = []

        for edit in eq_plan.edits:
            flt = BiquadFilter(
                filter_type=edit.filter.value,
                f0_hz=edit.f0_hz,
                gain_db=edit.gain_db,
                q=edit.q,
                fs=self.fs
            )
            target = edit.stem.value
            if target == "master":
                master_filters.append(flt)
            elif target in stem_filters:
                stem_filters[target].append(flt)

        # 3. Process each stem
        processed_stems = {}
        processed_paths = {}
        stem_reports = {}

        for name, audio in loaded_stems.items():
            flts = stem_filters[name]
            orig_rms = float(np.sqrt(np.mean(audio ** 2)))

            if flts:
                if self.mode == "stft":
                    proc_audio = apply_spectral_eq(audio, flts, fs=self.fs)
                else:
                    proc_audio = apply_time_domain_eq(audio, flts)
            else:
                proc_audio = np.copy(audio)

            new_rms = float(np.sqrt(np.mean(proc_audio ** 2)))
            rms_diff_db = 20.0 * np.log10((new_rms + 1e-12) / (orig_rms + 1e-12))

            # Save modified stem
            out_stem_path = os.path.join(output_dir, f"{name}_processed.wav")
            save_audio(out_stem_path, proc_audio, sr=self.fs)

            processed_stems[name] = proc_audio
            processed_paths[name] = out_stem_path
            stem_reports[name] = {
                "filters_applied": len(flts),
                "orig_rms_db": round(20 * np.log10(orig_rms + 1e-12), 2),
                "new_rms_db": round(20 * np.log10(new_rms + 1e-12), 2),
                "rms_change_db": round(rms_diff_db, 2)
            }

        # 4. Resynthesize Polished Master Mix
        polished_mix = sum_stems(processed_stems)

        # Apply any master bus filters if present
        if master_filters:
            if self.mode == "stft":
                polished_mix = apply_spectral_eq(polished_mix, master_filters, fs=self.fs)
            else:
                polished_mix = apply_time_domain_eq(polished_mix, master_filters)

        # 5. Check headroom and soft-normalize if peak exceeds 0 dBFS
        peak_val = float(np.max(np.abs(polished_mix)))
        peak_db = 20.0 * np.log10(peak_val + 1e-12)
        if peak_val > 0.98:
            print(f"[DSP Engine] Master peak at {peak_db:.2f} dBFS exceeds headroom. Applying transparent soft limiting.")
            polished_mix = polished_mix * (0.95 / peak_val)
            peak_db = 20.0 * np.log10(np.max(np.abs(polished_mix)))

        master_out_path = os.path.join(output_dir, "polished_master_mix.wav")
        save_audio(master_out_path, polished_mix, sr=self.fs)

        return {
            "master_mix_path": master_out_path,
            "stem_paths": processed_paths,
            "master_peak_dbfs": round(peak_db, 2),
            "stem_reports": stem_reports,
            "processing_mode": self.mode
        }

