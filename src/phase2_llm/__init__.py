"""
Phase 2: LLM Reasoning Engine Package
Mathematics of Music (MA375TGR)
"""

from .schema import StemType, FilterType, FilterEdit, EQPlan
from .psychoacoustics import PSYCHOACOUSTIC_PRIORS, DESCRIPTOR_SPECS, parse_expert_rules
from .llm_engine import LLMReasoningEngine

__all__ = [
    "StemType",
    "FilterType",
    "FilterEdit",
    "EQPlan",
    "PSYCHOACOUSTIC_PRIORS",
    "DESCRIPTOR_SPECS",
    "parse_expert_rules",
    "LLMReasoningEngine"
]

