"""
workflow/relevance.py — LLM-based per-page relevance check for RAG collections.

Uses GPT-4o-mini to decide whether each scraped page belongs in the intended
collection, based on the collection's routing metadata (description + keywords).

Public API:
  check_page_relevance(page_text, description, keywords) -> dict
  filter_scraped_items(scraped_items, routing, progress_cb=None) -> (relevant, mismatch, irrelevant)
"""

from __future__ import annotations

import json
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Per-page check
# ---------------------------------------------------------------------------

def check_page_relevance(
    page_text: str,
    collection_description: str,
    collection_keywords: list[str],
) -> dict:
    """
    Ask GPT-4o-mini whether this page's content belongs in the collection.

    Returns:
        {
          "relevant":   bool,                        # True = belongs here
          "reason":     str,                         # one-sentence explanation
          "confidence": "high" | "medium" | "low",  # how certain the model is
        }

    On any failure (API error, JSON parse error) → defaults to relevant=True,
    confidence="low" so that pages are never silently dropped.
    """
    from llms.openai_utils import openai_chat_completion

    prompt = (
        "You are a RAG collection auditor. Your job is to decide if a scraped web page "
        "belongs in a specific document collection.\n\n"
        "Collection description: {description}\n"
        "Collection keywords: {keywords}\n\n"
        "Analyse the page content and respond with a JSON object — no markdown:\n"
        '{{"relevant": true or false, "reason": "one sentence explanation", '
        '"confidence": "high" or "medium" or "low"}}\n\n'
        "Guidelines:\n"
        "- relevant=true if the page clearly relates to the collection topic\n"
        "- relevant=false if the page is clearly about a completely different topic\n"
        "- confidence=low if you are uncertain (e.g. generic landing page, navigation-only page)\n"
        "Return ONLY valid JSON, nothing else."
    ).format(
        description=collection_description or "(not provided)",
        keywords=", ".join(collection_keywords[:20]) if collection_keywords else "(none)",
    )

    # Keep input tokens low: 600 chars ≈ 150 tokens → total ~300 tokens per call → ~$0.00005
    content = f"Page content (first 600 chars):\n{page_text[:600]}"

    try:
        resp = openai_chat_completion(prompt, content, model="gpt-4o-mini")
        s = resp.strip()
        # Strip markdown code fences if the model added them
        if s.startswith("```"):
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
        result = json.loads(s.strip())
        return {
            "relevant":   bool(result.get("relevant", True)),
            "reason":     str(result.get("reason", "")),
            "confidence": str(result.get("confidence", "medium")),
        }
    except Exception as e:
        # On any failure: treat as relevant (conservative — never drop silently)
        return {
            "relevant":   True,
            "reason":     f"check_failed: {e}",
            "confidence": "low",
        }


# ---------------------------------------------------------------------------
# Batch filter
# ---------------------------------------------------------------------------

def filter_scraped_items(
    scraped_items: list[dict],
    routing: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Score each scraped item against the collection's routing metadata.

    Bucket definitions:
      relevant   — model says relevant=True  AND confidence is "high" or "medium"
      mismatch   — model says relevant=True  BUT confidence is "low"  (flagged for review)
      irrelevant — model says relevant=False (routed to 'not_relevant' collection)

    Mismatch items are still pushed to the intended collection — they're just flagged
    so a human can review them. Irrelevant items are NOT included in the returned
    scraped_items used for chunking; the caller is responsible for routing them.

    Returns:
        (relevant_items, mismatch_items, irrelevant_items)

    Early return (no check) if routing has neither description nor keywords.
    """
    description = routing.get("description", "") or ""
    keywords    = routing.get("keywords", []) or []

    if not description and not keywords:
        # No metadata to check against — skip entirely (safe for first runs)
        return scraped_items, [], []

    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    relevant:   list[dict] = []
    mismatch:   list[dict] = []
    irrelevant: list[dict] = []
    total = len(scraped_items)

    for i, item in enumerate(scraped_items):
        url = item.get("url", f"item[{i}]")
        _log(f"🔍 Relevance {i + 1}/{total}: {url}")

        result = check_page_relevance(item["text"], description, keywords)

        # Attach relevance result to item copy (don't mutate the original dict)
        enriched = dict(item)
        enriched["_relevance"] = result

        if result["relevant"]:
            if result["confidence"] == "low":
                mismatch.append(enriched)
                _log(f"   ⚠️  Flagged (low confidence): {result['reason']}")
            else:
                relevant.append(enriched)
        else:
            irrelevant.append(enriched)
            _log(f"   ❌ Irrelevant: {result['reason']}")

    return relevant, mismatch, irrelevant
