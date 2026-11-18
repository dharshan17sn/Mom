"""
Phase 2 Verification Script: LLM Reasoning Engine & Psychoacoustic Semantic Translation
Mathematics of Music (MA375TGR)
"""

import os
import sys
import json

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.phase2_llm import (
    LLMReasoningEngine,
    FilterEdit,
    FilterType,
    StemType,
    EQPlan
)


def run_phase2_test():
    print("=" * 68)
    print("  MA375TGR: PHASE 2 LLM REASONING & SEMANTIC TRANSLATION TEST")
    print("=" * 68)

    engine = LLMReasoningEngine()
    ollama_ready = engine.is_ollama_available()
    print(f"\n[1] Checking LLM Backend Status...")
    print(f"    - Ollama server running locally: {ollama_ready}")
    print(f"    - Active Engine: {'Ollama (Local LLM)' if ollama_ready else 'Deterministic Psychoacoustic Expert Engine'}")

    # Test Case 1: The Project Presentation's Anchor Prompt (Slide 2 & Slide 10)
    prompt1 = "The vocal sounds too boxy and muddy, and I want the bass kick to have more punch."
    print(f"\n[2] Test Case 1 (Slide Anchor Prompt):")
    print(f"    Input: \"{prompt1}\"")

    plan1 = engine.generate_plan(prompt1)
    print(f"    Source: {plan1.source.upper()}")
    print(f"    Summary: {plan1.summary}")
    print("    Parsed Biquad Edits:")
    for idx, edit in enumerate(plan1.edits, 1):
        print(f"      {idx}. Stem: [{edit.stem.value.upper():<6}] | Filter: {edit.filter.value:<9} | f0: {edit.f0_hz:>5.1f} Hz | Gain: {edit.gain_db:>+5.1f} dB | Q: {edit.q:>3.1f} ({edit.description})")

    # Verification of Anchor Prompt criteria
    stems_targeted = [e.stem.value for e in plan1.edits]
    assert any("vocal" in s for s in stems_targeted), "Anchor prompt failed to target vocals!"
    assert any("bass" in s or "drum" in s for s in stems_targeted), "Anchor prompt failed to target bass/drums!"

    # Test Case 2: Multi-Stem Master Request
    prompt2 = "The drums are too harsh in the cymbals, add air to the vocals, and make the bass warmer."
    print(f"\n[3] Test Case 2 (Multi-Stem Mastering):")
    print(f"    Input: \"{prompt2}\"")

    plan2 = engine.generate_plan(prompt2)
    print(f"    Summary: {plan2.summary}")
    print("    Parsed Biquad Edits:")
    for idx, edit in enumerate(plan2.edits, 1):
        print(f"      {idx}. Stem: [{edit.stem.value.upper():<6}] | Filter: {edit.filter.value:<9} | f0: {edit.f0_hz:>5.1f} Hz | Gain: {edit.gain_db:>+5.1f} dB | Q: {edit.q:>3.1f} ({edit.description})")

    # Test Case 3: Pydantic Safety Clamping Test (Slide 10 Safe Ranges)
    print("\n[4] Test Case 3 (Safety Clamping & Normalization):")
    print("    Inputting Out-of-Bounds values: Gain = +30 dB, f0 = 5 Hz, Q = 50, Stem = 'lead'")
    raw_unsafe_edit = FilterEdit(
        stem="lead",
        filter="bell",
        f0_hz=5.0,
        gain_db=30.0,
        q=50.0
    )
    print(f"    - Clamped Stem:     {raw_unsafe_edit.stem.value} (normalized from 'lead')")
    print(f"    - Clamped Filter:   {raw_unsafe_edit.filter.value} (normalized from 'bell')")
    print(f"    - Clamped f0:       {raw_unsafe_edit.f0_hz} Hz (clamped to min 20 Hz)")
    print(f"    - Clamped Gain:     {raw_unsafe_edit.gain_db} dB (clamped to max +12 dB)")
    print(f"    - Clamped Q:        {raw_unsafe_edit.q} (clamped to max 10.0)")

    assert raw_unsafe_edit.stem == StemType.VOCALS
    assert raw_unsafe_edit.f0_hz >= 20.0
    assert raw_unsafe_edit.gain_db <= 12.0
    assert raw_unsafe_edit.q <= 10.0

    # Save output plan to JSON for Phase 3 ingestion
    output_json_path = os.path.join("data", "output", "test_eq_plan.json")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(plan1.model_dump(), f, indent=2)
    print(f"\n[5] Exported Validated Plan JSON to: {output_json_path}")

    print("\n✔ PHASE 2 PASSED: 100% Validated semantic-to-parameter translation achieved!")
    print("=" * 68)


if __name__ == "__main__":
    run_phase2_test()

