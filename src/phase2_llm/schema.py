"""
Phase 2: LLM Reasoning Schema & Validation
Mathematics of Music (MA375TGR)
"""

from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class StemType(str, Enum):
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    MASTER = "master"


class FilterType(str, Enum):
    PEAKING = "peaking"
    NOTCH = "notch"
    HIGHPASS = "highpass"
    LOWSHELF = "lowshelf"
    HIGHSHELF = "highshelf"


class FilterEdit(BaseModel):
    """
    Parametric Biquad Filter Specification for a specific stem.
    Safe ranges enforced per slide 10:
    - Gain: -12.0 dB to +12.0 dB
    - Frequency: 20 Hz to 20,000 Hz
    - Q factor: 0.1 to 10.0
    """
    stem: StemType = Field(description="Target stem: vocals, drums, bass, other, or master")
    filter: FilterType = Field(default=FilterType.PEAKING, description="Filter topology")
    f0_hz: float = Field(description="Center or cutoff frequency in Hertz (20 - 20000)")
    gain_db: float = Field(default=0.0, description="Boost or cut gain in decibels (-12 to +12)")
    q: float = Field(default=1.0, description="Quality factor Q (0.1 to 10.0)")
    description: Optional[str] = Field(default=None, description="Acoustic intent / explanation")

    @field_validator("stem", mode="before")
    @classmethod
    def normalize_stem(cls, v: Any) -> str:
        if hasattr(v, "value"):
            v = v.value
        s = str(v).lower().strip()
        if "." in s:
            s = s.split(".")[-1]
        if s in ("vocal", "voice", "singing", "lead", "vocals"):
            return "vocals"
        if s in ("drum", "percussion", "beat", "kick", "snare", "drums"):
            return "drums"
        if s in ("sub", "bassline", "bass"):
            return "bass"
        if s in ("synth", "piano", "guitar", "keys", "music", "harmony", "other"):
            return "other"
        if s in ("mix", "all", "master_track", "master"):
            return "master"
        return s

    @field_validator("filter", mode="before")
    @classmethod
    def normalize_filter(cls, v: Any) -> str:
        if hasattr(v, "value"):
            v = v.value
        f = str(v).lower().strip().replace("-", "").replace("_", "")
        if "." in f:
            f = f.split(".")[-1]
        if "peak" in f or "bell" in f:
            return "peaking"
        if "notch" in f:
            return "notch"
        if "highpass" in f or "lowcut" in f or "hp" in f:
            return "highpass"
        if "lowshelf" in f or "ls" in f:
            return "lowshelf"
        if "highshelf" in f or "hs" in f:
            return "highshelf"
        return f

    @field_validator("f0_hz")
    @classmethod
    def clamp_frequency(cls, v: float) -> float:
        # Clamp strictly between 20 Hz and 20,000 Hz
        clamped = max(20.0, min(20000.0, float(v)))
        return round(clamped, 1)

    @field_validator("gain_db")
    @classmethod
    def clamp_gain(cls, v: float) -> float:
        # Clamp strictly between -12 dB and +12 dB
        clamped = max(-12.0, min(12.0, float(v)))
        return round(clamped, 2)

    @field_validator("q")
    @classmethod
    def clamp_q(cls, v: float) -> float:
        # Clamp strictly between 0.1 and 10.0
        clamped = max(0.1, min(10.0, float(v)))
        return round(clamped, 2)


class EQPlan(BaseModel):
    """
    Complete JSON schema returned by the reasoning engine.
    """
    user_prompt: str
    edits: List[FilterEdit] = Field(default_factory=list)
    summary: str = Field(default="", description="High-level engineering summary of the edits")
    source: str = Field(default="llm", description="Origin of generation: 'llm' or 'expert_rules'")
