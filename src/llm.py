"""Provider clients with a deterministic on-disk cache.

Every call is keyed on sha256(model + prompt + params) so `make reproduce` can
replay recorded outputs with no network access and no API keys, and so
`make live` can report cached-vs-live agreement. Never touches secret values
in logs; only the hash and metadata are recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "artifacts" / "llm_cache.jsonl"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    cached: bool
    raw: dict[str, Any] | None = None


def _cache_key(model: str, prompt: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with CACHE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[record["key"]] = record
    return cache


def _append_cache(record: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class CachedLLM:
    """Wraps a provider call with the disk cache. `call_fn(prompt, params) -> str`."""

    def __init__(self, model: str, call_fn, *, allow_live: bool = True) -> None:
        self.model = model
        self._call_fn = call_fn
        self.allow_live = allow_live
        self._cache = _load_cache()

    def generate(self, prompt: str, params: dict[str, Any] | None = None) -> LLMResponse:
        params = params or {}
        key = _cache_key(self.model, prompt, params)
        if key in self._cache:
            return LLMResponse(text=self._cache[key]["text"], model=self.model, cached=True)
        if not self.allow_live:
            raise RuntimeError(
                f"No cached response for key {key[:12]}... and live calls are disabled "
                "(offline `make reproduce` mode). Run `make live` to populate the cache."
            )
        text = self._call_fn(prompt, params)
        record = {"key": key, "model": self.model, "prompt": prompt, "params": params, "text": text}
        _append_cache(record)
        self._cache[key] = record
        return LLMResponse(text=text, model=self.model, cached=False)


def groq_call_fn(model_id: str):
    """Build a call_fn against Groq's OpenAI-compatible chat completions API."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def call(prompt: str, params: dict[str, Any]) -> str:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 800),
        )
        return response.choices[0].message.content or ""

    return call


def gemini_call_fn(model_id: str):
    """Build a call_fn against the Gemini API."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def call(prompt: str, params: dict[str, Any]) -> str:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config={"temperature": params.get("temperature", 0.0)},
        )
        return response.text or ""

    return call
