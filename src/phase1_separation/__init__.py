"""
Mathematics of Music: Phase 1 Separation Package
"""
from .audio_utils import (
    STEM_NAMES,
    DEFAULT_SR,
    load_audio,
    save_audio,
    sum_stems,
    compute_reconstruction_metrics
)
from .stock_generator import create_stock_preset, create_high_contrast_preset
from .separator import StemSeparator

__all__ = [
    "STEM_NAMES",
    "DEFAULT_SR",
    "load_audio",
    "save_audio",
    "sum_stems",
    "compute_reconstruction_metrics",
    "create_stock_preset",
    "create_high_contrast_preset",
    "StemSeparator"
]

