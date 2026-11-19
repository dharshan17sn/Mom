"""
Phase 1: Real-Time Neural Stem Separation Module
Powered by PyTorch + Hybrid Transformer Demucs (HT Demucs)
Splits mixed audio x(t) into 4 isolated stems: vocals(t), drums(t), bass(t), other(t).
"""

import os
import hashlib
from typing import Dict, Tuple, Optional, Callable
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
    Neural Stem Separation Manager using Meta's Demucs model.
    Dynamically separates any input audio file into 4 distinct musical stems.
    """

    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self._model = None
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
        output_dir: Optional[str] = None,
        force_recompute: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, str]:
        """
        Dynamically separates mix_audio_path into 4 stems.
        Uses real Demucs neural network for uploaded music.
        """
        if not os.path.exists(mix_audio_path):
            raise FileNotFoundError(f"Audio file not found: {mix_audio_path}")

        # Determine unique output directory based on track hash/name
        if output_dir is None:
            track_name = os.path.splitext(os.path.basename(mix_audio_path))[0]
            with open(mix_audio_path, "rb") as f:
                file_hash = hashlib.md5(f.read(1024 * 512)).hexdigest()[:8]
            output_dir = os.path.join("data", "output", f"separated_{track_name}_{file_hash}")

        os.makedirs(output_dir, exist_ok=True)

        # Check if this track is from a built-in stock preset that already has stems
        input_dir = os.path.dirname(os.path.abspath(mix_audio_path))
        is_preset_dir = "presets" in input_dir.lower()
        if is_preset_dir:
            preset_stems_found = True
            for name in STEM_NAMES:
                if not os.path.exists(os.path.join(input_dir, f"{name}.wav")):
                    preset_stems_found = False
                    break
            if preset_stems_found and not force_recompute:
                print(f"[Separator] Loading pre-rendered stock stems from {input_dir}")
                return {name: os.path.join(input_dir, f"{name}.wav") for name in STEM_NAMES}

        # Check if stems already exist in output directory for this exact track
        existing_stems = {}
        all_exist = True
        for name in STEM_NAMES:
            stem_path = os.path.join(output_dir, f"{name}.wav")
            if os.path.exists(stem_path):
                existing_stems[name] = stem_path
            else:
                all_exist = False

        if all_exist and not force_recompute:
            print(f"[Separator] Reusing previously separated stems from {output_dir}")
            return existing_stems

        # Run real Demucs Neural Network
        if not self._demucs_available:
            raise RuntimeError(
                "Demucs neural network is not installed in the environment. "
                "Run: pip install torch torchaudio demucs"
            )

        print(f"[Separator] Executing real Demucs ({self.model_name}) separation for: {mix_audio_path}")
        return self._run_demucs(mix_audio_path, output_dir, progress_callback=progress_callback)

    def _run_demucs(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, str]:
        """Runs real Demucs neural network separation on CPU or CUDA GPU."""
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if progress_callback:
            progress_callback(0.1, f"Loading Demucs neural model ({self.model_name}) on {device.upper()}...")

        if self._model is None:
            self._model = get_model(self.model_name)
        model = self._model
        model.to(device)
        model.eval()

        if progress_callback:
            progress_callback(0.25, f"Reading and resampling audio to {model.samplerate} Hz...")

        audio, sr = load_audio(audio_path, target_sr=model.samplerate)
        # Ensure stereo
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)

        wav = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

        if progress_callback:
            progress_callback(0.4, "Neural network is separating stems (Vocals, Drums, Bass, Other)...")

        def internal_callback(state):
            if progress_callback and isinstance(state, dict):
                pct = 0.4 + 0.45 * float(state.get("percent", 0.0)) / 100.0
                progress_callback(min(pct, 0.88), "Separating audio waveforms...")

        with torch.no_grad():
            sources = apply_model(
                model,
                wav,
                device=device,
                progress=False,
                callback=internal_callback
            )[0]  # shape: (stems, channels, time)

        if progress_callback:
            progress_callback(0.9, "Saving isolated stems to disk...")

        stem_paths = {}
        for idx, name in enumerate(model.sources):
            stem_audio = sources[idx].cpu().numpy()
            out_file = os.path.join(output_dir, f"{name}.wav")
            save_audio(out_file, stem_audio, sr=model.samplerate)
            stem_paths[name] = out_file

        if progress_callback:
            progress_callback(1.0, "Neural separation complete!")

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
