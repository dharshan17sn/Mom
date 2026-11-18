"""
Phase 1: Stem Separation Module
Supports both:
1. Instant high-fidelity stock preset stems (Vocals, Drums, Bass, Other)
2. Real Demucs 4-stem neural separation (if PyTorch & demucs package are installed)
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np

from .audio_utils import (
    STEM_NAMES,
    DEFAULT_SR,
    load_audio,
    save_audio,
    sum_stems,
    compute_reconstruction_metrics
)
from .stock_generator import create_stock_preset


class StemSeparator:
    """
    Modular Stem Separation Manager.
    Separates a mix x(t) into: vocals(t), drums(t), bass(t), other(t).
    """

    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self._demucs_available = self._check_demucs_installed()

    @staticmethod
    def _check_demucs_installed() -> bool:
        try:
            import torch
            import demucs
            return True
        except ImportError:
            return False

    @property
    def has_demucs(self) -> bool:
        return self._demucs_available

    def separate(
        self,
        mix_audio_path: str,
        output_dir: str,
        force_preset_fallback: bool = False
    ) -> Dict[str, str]:
        """
        Separates mix_audio_path into 4 stems.
        If demucs is not installed or force_preset_fallback=True, uses the
        matching stock stems or generates a preset fallback.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Check if input directory itself already has stem files (e.g. stock preset dir)
        input_dir = os.path.dirname(os.path.abspath(mix_audio_path))
        preset_stems_found = True
        for name in STEM_NAMES:
            p = os.path.join(input_dir, f"{name}.wav")
            if not os.path.exists(p):
                preset_stems_found = False
                break

        if preset_stems_found:
            # Copy or return existing stem paths
            stem_paths = {}
            for name in STEM_NAMES:
                src_path = os.path.join(input_dir, f"{name}.wav")
                dst_path = os.path.join(output_dir, f"{name}.wav")
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    audio, sr = load_audio(src_path)
                    save_audio(dst_path, audio, sr=sr)
                    stem_paths[name] = dst_path
                else:
                    stem_paths[name] = src_path
            return stem_paths

        # Check if stems already exist in output directory
        existing_stems = {}
        all_exist = True
        for name in STEM_NAMES:
            stem_path = os.path.join(output_dir, f"{name}.wav")
            if os.path.exists(stem_path):
                existing_stems[name] = stem_path
            else:
                all_exist = False

        if all_exist:
            print(f"[Separator] Found existing stems in {output_dir}")
            return existing_stems

        # If real Demucs is available and not forcing preset
        if self._demucs_available and not force_preset_fallback:
            print(f"[Separator] Running Demucs ({self.model_name}) separation...")
            return self._run_demucs(mix_audio_path, output_dir)
        else:
            print("[Separator] Using preset audio generator mode (Demucs not loaded / preset mode active).")
            # Generate stock audio into the output directory
            _, stem_paths = create_stock_preset(output_dir)
            return stem_paths

    def _run_demucs(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Runs real Demucs neural network separation."""
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Demucs] Loading model '{self.model_name}' on {device}...")
        model = get_model(self.model_name)
        model.to(device)
        model.eval()

        audio, sr = load_audio(audio_path, target_sr=model.samplerate)
        wav = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

        print("[Demucs] Performing 4-stem separation...")
        with torch.no_grad():
            sources = apply_model(model, wav, device=device)[0]  # shape: (stems, channels, time)

        stem_paths = {}
        for idx, name in enumerate(model.sources):
            stem_audio = sources[idx].cpu().numpy()
            out_file = os.path.join(output_dir, f"{name}.wav")
            save_audio(out_file, stem_audio, sr=model.samplerate)
            stem_paths[name] = out_file

        return stem_paths

    def verify_summation(self, mixture_path: str, stem_paths: Dict[str, str]) -> Dict[str, float]:
        """
        Validates the fundamental relation:
        x(t) ≈ vocals(t) + drums(t) + bass(t) + other(t)
        Returns SNR (dB) and MSE.
        """
        mix_audio, sr = load_audio(mixture_path)
        stems = {}
        for name, path in stem_paths.items():
            stems[name], _ = load_audio(path, target_sr=sr)

        reconstructed = sum_stems(stems)
        metrics = compute_reconstruction_metrics(mix_audio, reconstructed)
        return metrics

