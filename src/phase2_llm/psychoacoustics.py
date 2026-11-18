"""
Phase 2: Psychoacoustic Knowledge Base & Rule-Based Expert System
Encodes empirical mastering heuristics and the frequency maps from Slide 11.
Provides an offline deterministic engine and few-shot examples for the LLM.
"""

import re
from typing import List, Dict, Any, Optional
from .schema import FilterEdit, StemType, FilterType, EQPlan

# Semantic descriptor definitions with lexical variations and audio priors
DESCRIPTOR_SPECS: List[Dict[str, Any]] = [
    {
        "name": "punch",
        "aliases": ["punch", "punchy", "punchier", "punchiness", "thump", "slam", "weight", "knock"],
        "default_stem": StemType.BASS,
        "stem_affinity": [StemType.BASS, StemType.DRUMS],
        "filter": FilterType.PEAKING,
        "f0_hz": 80.0,
        "gain_db": 3.0,
        "q": 1.0,
        "description": "Boosting 60-100 Hz adds punch and low-end definition."
    },
    {
        "name": "boxy_muddy",
        "aliases": ["boxy and muddy", "muddy and boxy"],
        "default_stem": StemType.VOCALS,
        "stem_affinity": [StemType.VOCALS, StemType.DRUMS, StemType.OTHER],
        "filter": FilterType.PEAKING,
        "f0_hz": 300.0,
        "gain_db": -4.0,
        "q": 1.4,
        "description": "Cutting around 300 Hz clears vocal boxiness and low-mid mud."
    },
    {
        "name": "boxy",
        "aliases": ["boxy", "boxiness", "hollow"],
        "default_stem": StemType.VOCALS,
        "stem_affinity": [StemType.VOCALS, StemType.DRUMS],
        "filter": FilterType.PEAKING,
        "f0_hz": 400.0,
        "gain_db": -4.0,
        "q": 1.5,
        "description": "Cutting 300-600 Hz removes hollow, box-like acoustic resonance."
    },
    {
        "name": "muddy",
        "aliases": ["muddy", "muddier", "mud", "muddiness", "clutter"],
        "default_stem": StemType.VOCALS,
        "stem_affinity": [StemType.VOCALS, StemType.BASS, StemType.DRUMS, StemType.OTHER],
        "filter": FilterType.PEAKING,
        "f0_hz": 280.0,
        "gain_db": -4.0,
        "q": 1.4,
        "description": "Cutting 200-400 Hz clears up muddiness and acoustic clutter."
    },
    {
        "name": "warm",
        "aliases": ["warm", "warmer", "warmth", "thick", "thicker", "thickness", "body"],
        "default_stem": StemType.BASS,
        "stem_affinity": [StemType.BASS, StemType.VOCALS, StemType.OTHER],
        "filter": FilterType.PEAKING,
        "f0_hz": 180.0,
        "gain_db": 3.0,
        "q": 1.0,
        "description": "Boosting 100-250 Hz enhances low-mid warmth and musical body."
    },
    {
        "name": "nasal",
        "aliases": ["nasal", "honky", "honk"],
        "default_stem": StemType.VOCALS,
        "stem_affinity": [StemType.VOCALS],
        "filter": FilterType.PEAKING,
        "f0_hz": 1000.0,
        "gain_db": -3.5,
        "q": 2.0,
        "description": "Cutting 800-1200 Hz attenuates honky, nasal formant buildup."
    },
    {
        "name": "harsh",
        "aliases": ["harsh", "harsher", "harshness", "piercing", "shrill", "bite", "fatiguing"],
        "default_stem": StemType.DRUMS,
        "stem_affinity": [StemType.DRUMS, StemType.VOCALS, StemType.OTHER],
        "filter": FilterType.PEAKING,
        "f0_hz": 3200.0,
        "gain_db": -4.0,
        "q": 2.0,
        "description": "Cutting 2-5 kHz eliminates ear fatigue and piercing harshness."
    },
    {
        "name": "sibilant",
        "aliases": ["sibilant", "sibilance", "deess", "de-ess", "sharpness", "hiss"],
        "default_stem": StemType.VOCALS,
        "stem_affinity": [StemType.VOCALS],
        "filter": FilterType.PEAKING,
        "f0_hz": 6500.0,
        "gain_db": -5.0,
        "q": 2.8,
        "description": "Cutting 5-8 kHz controls sharp 's', 'sh' and 't' friction noise."
    },
    {
        "name": "airy",
        "aliases": ["airy", "airier", "air", "breath", "sheen", "open"],
        "default_stem": StemType.VOCALS,
        "stem_affinity": [StemType.VOCALS, StemType.OTHER],
        "filter": FilterType.HIGHSHELF,
        "f0_hz": 12000.0,
        "gain_db": 3.5,
        "q": 0.7,
        "description": "High-shelf boost above 10 kHz introduces breathiness and sheen."
    },
    {
        "name": "bright",
        "aliases": ["bright", "brighter", "brightness", "sparkle", "crisp", "presence"],
        "default_stem": StemType.OTHER,
        "stem_affinity": [StemType.OTHER, StemType.VOCALS, StemType.DRUMS],
        "filter": FilterType.HIGHSHELF,
        "f0_hz": 10000.0,
        "gain_db": 3.0,
        "q": 0.7,
        "description": "High-shelf boost for acoustic presence and sparkle."
    },
    {
        "name": "rumble",
        "aliases": ["rumble", "sub-rumble", "low-rumble", "boominess"],
        "default_stem": StemType.MASTER,
        "stem_affinity": [StemType.MASTER, StemType.BASS, StemType.VOCALS],
        "filter": FilterType.HIGHPASS,
        "f0_hz": 35.0,
        "gain_db": 0.0,
        "q": 0.707,
        "description": "High-pass cut below 35 Hz to eliminate sub-audible rumble."
    }
]

# Stem detection tokens
STEM_KEYWORDS = {
    StemType.VOCALS: ["vocal", "vocals", "voice", "singing", "lead", "singer"],
    StemType.DRUMS: ["drum", "drums", "kick", "snare", "hihat", "cymbals", "cymbal", "percussion", "beat"],
    StemType.BASS: ["bass", "sub", "bassline", "lowend", "808"],
    StemType.OTHER: ["synth", "piano", "keys", "keyboard", "guitar", "chords", "pad", "other"],
    StemType.MASTER: ["mix", "master", "overall", "everything", "track"]
}

PSYCHOACOUSTIC_PRIORS = {spec["name"]: spec for spec in DESCRIPTOR_SPECS}


def detect_stem_in_text(text: str) -> Optional[StemType]:
    """Finds which stem is referenced in a text snippet."""
    text_lower = text.lower()
    for stem, keywords in STEM_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return stem
    return None


def parse_expert_rules(prompt: str) -> EQPlan:
    """
    Deterministic rule-based expert parser.
    Extracts clauses, identifies target stems, matches psychoacoustic priors,
    and returns an EQPlan.
    """
    text = prompt.strip()

    # Split on sentence boundaries and coordinating conjunctions when they start new thoughts
    # e.g., "The vocal sounds too boxy and muddy, and I want the bass kick to have more punch."
    # -> ["The vocal sounds too boxy and muddy", "I want the bass kick to have more punch"]
    raw_clauses = re.split(r'[,;]|\band\s+(?:i\s+want|more|add|cut|make|the)\b|\bbut\b', text, flags=re.IGNORECASE)
    clauses = [c.strip() for c in raw_clauses if c.strip()]

    edits: List[FilterEdit] = []
    numeric_pattern = re.compile(
        r'([+-]?\d+(?:\.\d+)?)\s*db\s*(?:at|around)?\s*(\d+(?:\.\d+)?)\s*(?:hz|khz)?',
        re.IGNORECASE
    )

    current_stem_context = StemType.VOCALS

    for clause in clauses:
        c_lower = clause.lower()
        stem_found = detect_stem_in_text(clause)
        if stem_found:
            current_stem_context = stem_found

        # Check for explicit numbers (e.g. "-4 dB at 300 Hz on vocal")
        num_match = numeric_pattern.search(clause)
        if num_match:
            gain = float(num_match.group(1))
            f0 = float(num_match.group(2))
            if "khz" in clause[num_match.end(2):num_match.end(2)+4].lower():
                f0 *= 1000.0

            filt_type = FilterType.PEAKING
            if "highpass" in c_lower or "low cut" in c_lower:
                filt_type = FilterType.HIGHPASS
            elif "shelf" in c_lower:
                filt_type = FilterType.LOWSHELF if f0 < 500 else FilterType.HIGHSHELF

            edits.append(FilterEdit(
                stem=current_stem_context,
                filter=filt_type,
                f0_hz=f0,
                gain_db=gain,
                q=1.4,
                description=f"Explicit parameter edit for {current_stem_context.value}"
            ))
            continue

        # Check against DESCRIPTOR_SPECS
        matched_descriptor = False
        for spec in DESCRIPTOR_SPECS:
            # Check aliases
            for alias in spec["aliases"]:
                if re.search(r'\b' + re.escape(alias) + r'\b', c_lower):
                    target_stem = current_stem_context
                    if target_stem not in spec["stem_affinity"] and target_stem != StemType.MASTER:
                        target_stem = spec["default_stem"]

                    gain = spec["gain_db"]
                    is_cut_word = any(w in c_lower for w in ["cut", "remove", "tame", "reduce", "less", "too"])
                    is_boost_word = any(w in c_lower for w in ["boost", "add", "more", "punch", "bring up", "enhance"])

                    if is_cut_word and gain > 0:
                        gain = -gain
                    elif is_boost_word and gain < 0 and spec["name"] not in ("muddy", "boxy", "nasal", "harsh", "sibilant", "rumble", "boxy_muddy"):
                        gain = abs(gain)

                    # Avoid redundant duplicate filters on the exact same stem & frequency
                    if not any(e.stem == target_stem and abs(e.f0_hz - spec["f0_hz"]) < 20 for e in edits):
                        edits.append(FilterEdit(
                            stem=target_stem,
                            filter=spec["filter"],
                            f0_hz=spec["f0_hz"],
                            gain_db=gain,
                            q=spec["q"],
                            description=spec["description"]
                        ))
                    matched_descriptor = True
                    break
            if matched_descriptor:
                # If matched compound (e.g. boxy and muddy), don't re-match sub-aliases
                break

    # Fallback if nothing matched
    if not edits:
        edits.append(FilterEdit(
            stem=StemType.MASTER,
            filter=FilterType.PEAKING,
            f0_hz=1000.0,
            gain_db=0.0,
            q=1.0,
            description="Flat passthrough - no specific acoustic descriptor detected."
        ))

    summary = f"Applied {len(edits)} parametric EQ adjustment(s) across stems: " + ", ".join(
        f"{e.stem.value} ({e.gain_db:+.1f}dB @ {e.f0_hz:.0f}Hz)" for e in edits
    )

    return EQPlan(
        user_prompt=prompt,
        edits=edits,
        summary=summary,
        source="expert_rules"
    )
