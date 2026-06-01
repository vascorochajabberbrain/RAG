"""
Gemini chat-completion helper that mirrors the shape of
llms.openai_utils.openai_chat_completion so callers can swap models
without touching their flow.

Used as the LLM fallback in /api/execute/faq-extract when the
structured-accordion path doesn't catch the FAQ pairs (non-accordion
prose FAQ pages). gpt-4o-mini was too inconsistent at this task
(33 / 58 / 68 pair-count variance across the same content in 3 CEs);
Gemini 2.5 Flash with temperature=0 is materially more stable AND
cheaper than even gpt-4o-mini for this use case.

Matches the SDK choice in jBKB (api/src/routes/copilot-chat.ts uses
`@google/genai` with model='gemini-2.5-flash') — same Google account /
API key gets reused so there's no new billing surface to track.
"""
from __future__ import annotations

import os
import time
from typing import Optional

# google-genai is the unified Google Gen AI SDK (replaces the older
# google-generativeai package). pip install google-genai.
try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — caller surfaces a clear error
    genai = None
    genai_types = None


_gemini_client = None


def _get_gemini_client():
    """Lazy singleton so we don't initialise the client at import time
    (which would fail any test run without GEMINI_API_KEY set)."""
    global _gemini_client
    if _gemini_client is None:
        if genai is None:
            raise RuntimeError(
                "google-genai SDK not installed. Run: pip install google-genai"
            )
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) env var not set — required "
                "for Gemini chat completion. Same key jBKB uses for copilot-chat."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def gemini_chat_completion(
    prompt: str,
    text: str,
    model: str = "gemini-2.5-flash",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Drop-in shape match for openai_chat_completion.

    Args:
        prompt: System instruction (becomes Gemini's system_instruction).
        text:   User input text (becomes the single user turn content).
        model:  Gemini model id. Default 'gemini-2.5-flash' — the same
                tier jBKB uses for copilot-chat.
        max_tokens: Hard cap on output tokens. None leaves Gemini's
                per-model default. FAQ extraction passes ~8000 because
                ~70 Q+A pairs of JSON output requires it.
        temperature: Sampling temperature. None leaves Gemini's default
                (~1.0). Structured-output callers should pass 0.0 for
                near-deterministic output.

    Returns the text content of the model's response. Strips trailing
    code fences (``` / ```json) so JSON callers can json.loads()
    without an extra cleanup step — matching openai_utils.
    """
    start_time = time.time()
    client = _get_gemini_client()

    # Build the generation config. Only set fields the caller specified;
    # Gemini's per-model defaults are sensible.
    config_kwargs: dict = {"system_instruction": prompt}
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens
    if temperature is not None:
        config_kwargs["temperature"] = temperature

    resp = client.models.generate_content(
        model=model,
        contents=text,
        config=genai_types.GenerateContentConfig(**config_kwargs),
    )

    raw = (resp.text or "").strip()
    # Strip ```...``` fences the same way openai_utils does so JSON
    # callers don't have to special-case the response.
    if raw.startswith("```"):
        # Drop leading fence (and optional 'json' language tag).
        first_nl = raw.find("\n")
        if first_nl >= 0:
            raw = raw[first_nl + 1:]
    if raw.endswith("```"):
        raw = raw[: -3].rstrip()

    elapsed = time.time() - start_time
    print(f"[gemini_utils] {model} completed in {elapsed:.1f}s, {len(raw)} chars out")
    return raw
