"""
Phase 2: LLM Reasoning Engine
Handles Natural Language to Biquad EQ JSON translation via:
1. Local Ollama Models (Llama-3.2, Qwen2.5, Mistral) via official ollama SDK
2. Cloud Frontier LLM API (Google Gemini / OpenAI compatible)
3. Deterministic Psychoacoustic Expert System (offline rule-based baseline)
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List

import ollama
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

Psychoacoustic Guidelines (Slide 11):
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
      "f0_hz": 300.0,
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
    Translates natural language requests into structured Biquad EQPlans.
    Supports official Ollama Python SDK, Cloud LLM APIs, and Offline Expert Rules.
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        ollama_host: str = "http://127.0.0.1:11434"
    ):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.client = ollama.Client(host=self.ollama_host)

    def is_ollama_available(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def list_installed_ollama_models(self) -> List[str]:
        """Returns list of all model names currently pulled in Ollama."""
        try:
            resp = self.client.list()
            # ollama.list() returns ListResponse with models attribute
            models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
            names = []
            for m in models:
                name = m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "")
                if name:
                    names.append(name)
            return names
        except Exception:
            return []

    def pull_ollama_model(self, model_name: str) -> None:
        """Pulls a model (e.g. llama3.2 or qwen2.5) via Ollama."""
        self.client.pull(model_name)

    def generate_with_ollama(self, prompt: str, model_name: Optional[str] = None) -> EQPlan:
        """
        Executes real local neural LLM inference via Ollama.
        Uses structured JSON output mode with low temperature.
        """
        target_model = model_name or self.model_name
        print(f"[Ollama] Querying model '{target_model}'...")

        response = self.client.chat(
            model=target_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User Request: {prompt}\nJSON Output:"}
            ],
            format="json",
            options={
                "temperature": 0.1,  # Low temperature per Slide 10
                "top_p": 0.9
            }
        )

        content = response["message"]["content"]
        data = json.loads(content)

        raw_edits = data.get("edits", [])
        summary = data.get("summary", f"Ollama ({target_model}) parametric EQ plan")
        validated_edits = [FilterEdit(**e) for e in raw_edits]

        return EQPlan(
            user_prompt=prompt,
            edits=validated_edits,
            summary=summary,
            source=f"ollama_{target_model}"
        )

    def generate_with_gemini_api(self, prompt: str, api_key: str, model: str = "gemini-1.5-flash") -> EQPlan:
        """
        Executes inference via Google Gemini API for real-time cloud LLM reasoning.
        """
        import urllib.request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\nUser Request: {prompt}\nJSON Output:"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=12.0) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            candidate = res_data["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(candidate)

            raw_edits = data.get("edits", [])
            summary = data.get("summary", f"Gemini ({model}) parametric EQ plan")
            validated_edits = [FilterEdit(**e) for e in raw_edits]

            return EQPlan(
                user_prompt=prompt,
                edits=validated_edits,
                summary=summary,
                source=f"gemini_{model}"
            )

    def generate_plan(
        self,
        prompt: str,
        backend_mode: str = "auto",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> EQPlan:
        """
        Unified router for plan generation:
        - "ollama": Force Ollama local LLM
        - "gemini": Force Gemini API
        - "rules": Force Deterministic Psychoacoustic rules
        - "auto": Try Ollama, if offline fallback to rules.
        """
        if backend_mode == "gemini" and api_key:
            return self.generate_with_gemini_api(prompt, api_key=api_key)

        if backend_mode == "ollama" or (backend_mode == "auto" and self.is_ollama_available()):
            try:
                return self.generate_with_ollama(prompt, model_name=model_name)
            except Exception as e:
                print(f"[LLM Engine] Ollama error ({e}). Falling back to Psychoacoustic Expert Rules.")
                return parse_expert_rules(prompt)

        # Offline deterministic psychoacoustic rules
        return parse_expert_rules(prompt)
