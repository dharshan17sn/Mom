"""
Phase 2: LLM Reasoning Engine
Handles Natural Language to Biquad EQ JSON translation via Ollama (Llama-3.2 / Qwen2.5)
with automatic fallback to the psychoacoustic rule-based expert parser.
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

from .schema import EQPlan, FilterEdit, StemType, FilterType
from .psychoacoustics import parse_expert_rules

SYSTEM_PROMPT = """You are an expert mastering and mixing audio engineer.
Your task is to translate natural language user requests into exact parametric biquad equalizer parameters for a 4-stem multitrack mix.

Stems available:
- "vocals": Lead and backing vocals
- "drums": Kick, snare, hi-hats, cymbals, percussion
- "bass": Bass guitar, 808, sub-bass
- "other": Keys, synthesizers, guitars, brass, background harmonies
- "master": Master stereo bus

Supported filter topologies:
- "peaking": Bell curve (most common for cuts and boosts)
- "lowshelf": Shelf for low end (< 500 Hz)
- "highshelf": Shelf for top end air (> 5 kHz)
- "highpass": Low cut / rumble filter
- "notch": Narrow cut for sharp resonances

Psychoacoustic Guidelines:
- "punchy kick / punchy bass": boost 60 - 100 Hz (+3 to +6 dB, Q=1.0 to 1.4)
- "warm / muddy": cut 200 - 400 Hz (-3 to -6 dB, Q=1.2 to 1.5) to fix mud; boost 100 - 250 Hz for warmth.
- "boxy": cut 300 - 600 Hz (-3 to -5 dB, Q=1.4 to 1.8) on vocals/snare.
- "nasal": cut 800 - 1200 Hz (-3 to -5 dB, Q=1.8 to 2.5) on vocals.
- "harsh": cut 2000 - 4500 Hz (-3 to -6 dB, Q=1.5 to 2.5) on vocals or cymbals.
- "sibilant / de-ess": cut 5000 - 8000 Hz (-4 to -8 dB, Q=2.5 to 3.5) on vocals.
- "airy / bright / open": boost 10000 - 16000 Hz (+2 to +5 dB, Q=0.7) highshelf on vocals or other.

Safety boundaries:
- Gain must strictly be within -12.0 dB to +12.0 dB
- Frequency f0_hz must strictly be within 20 Hz to 20000 Hz
- Q must strictly be within 0.1 to 10.0

You MUST return ONLY a valid JSON object matching this schema, with no preamble and no markdown:
{
  "edits": [
    {
      "stem": "vocals",
      "filter": "peaking",
      "f0_hz": 300,
      "gain_db": -4.0,
      "q": 1.4,
      "description": "Cut boxy resonance around 300 Hz"
    }
  ],
  "summary": "1-sentence summary of mastering edits"
}
"""


class LLMReasoningEngine:
    """
    Translates natural language to EQPlan using local Ollama LLM,
    with zero-latency fallback to rule-based psychoacoustic expert system.
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        ollama_url: str = "http://localhost:11434/api/generate",
        timeout_seconds: float = 8.0
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout_seconds = timeout_seconds

    def is_ollama_available(self) -> bool:
        """Check if local Ollama server is responding."""
        try:
            req = urllib.request.Request("http://localhost:11434/", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate_plan(self, prompt: str, force_expert_rules: bool = False) -> EQPlan:
        """
        Translates prompt into an EQPlan.
        If force_expert_rules is True or Ollama is offline, uses the expert rules parser.
        """
        if force_expert_rules or not self.is_ollama_available():
            # Use deterministic psychoacoustic knowledge base
            return parse_expert_rules(prompt)

        # Query local Ollama instance
        try:
            payload = {
                "model": self.model_name,
                "prompt": f"{SYSTEM_PROMPT}\nUser Request: {prompt}\nJSON Output:",
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,  # Low temperature per slide 10
                    "top_p": 0.9
                }
            }

            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
                response_text = result.get("response", "{}")
                data = json.loads(response_text)

                raw_edits = data.get("edits", [])
                summary = data.get("summary", "LLM-generated parametric EQ plan")

                validated_edits = [FilterEdit(**e) for e in raw_edits]

                return EQPlan(
                    user_prompt=prompt,
                    edits=validated_edits,
                    summary=summary,
                    source="llm"
                )

        except Exception as e:
            print(f"[LLM Engine] Ollama query failed ({e}). Falling back to Psychoacoustic Expert Rules.")
            return parse_expert_rules(prompt)

