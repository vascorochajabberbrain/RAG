"""
Suggest chunking or scraper config changes using an LLM. Used by CLI/API for "suggest" actions.
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def suggest_chunking(text_preview: str, source_type: str = "unknown") -> dict:
    """
    Ask the LLM to recommend chunking settings. Returns dict with batch_size, overlap_size, use_proposition_chunking.
    """
    from llms.openai_utils import openai_chat_completion
    prompt = """You are helping configure RAG chunking. Given a short preview of the content and its source type, recommend:
- batch_size: int (e.g. 5000-15000 for raw text before LLM splitting)
- overlap_size: int (e.g. 100-200)
- use_proposition_chunking: bool (True for semantic/proposition-based splitting with an LLM, False for simple character/sentence split)

Reply with a JSON object only, no markdown, e.g. {"batch_size": 10000, "overlap_size": 100, "use_proposition_chunking": true}."""
    content = f"Source type: {source_type}\n\nPreview (first 1500 chars):\n{text_preview[:1500]}"
    resp = openai_chat_completion(prompt, content)
    try:
        # Strip possible markdown code block
        s = resp.strip()
        if s.startswith("```"):
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
        return json.loads(s)
    except json.JSONDecodeError:
        return {"batch_size": 10000, "overlap_size": 100, "use_proposition_chunking": True}


def suggest_collection_metadata(chunks: list, source_label: str = "document") -> dict:
    """
    Analyze a sample of chunks and generate collection-level metadata for RAG routing.
    Used by the HIRS pipeline to decide whether/which RAG collection to query.

    Returns a dict with:
      - topics: list of 3-7 topic tags
      - keywords: list of 10-20 specific terms a user might ask about
      - description: one sentence describing the collection
      - language: ISO 639-1 code of the main content language
      - doc_type: recipe_book | faq | manual | product_catalog | legal | general
    """
    from llms.openai_utils import openai_chat_completion

    if not chunks:
        return {}

    # Sample up to 20 evenly-spaced chunks
    n = len(chunks)
    if n <= 20:
        sampled = chunks
    else:
        step = n / 20
        sampled = [chunks[int(i * step)] for i in range(20)]

    # For hierarchical chunks (format: "Context:\n...\n\nPassage:\n..."),
    # extract only the Passage section to avoid redundancy in the sample
    def extract_text(chunk: str) -> str:
        if "\n\nPassage:\n" in chunk:
            return chunk.split("\n\nPassage:\n", 1)[1]
        return chunk

    sample_texts = [extract_text(c)[:400] for c in sampled]
    joined = "\n---\n".join(sample_texts)

    prompt = (
        "You are analyzing a document that has been split into chunks for a RAG retrieval system.\n"
        "Based on the sample chunks below, return a JSON object with these exact keys:\n"
        '- "topics": list of 3-7 short topic tags describing what this document is about\n'
        '- "keywords": list of 10-20 specific keywords or phrases a user might ask about\n'
        '- "description": one concise sentence describing what this collection contains\n'
        '- "language": ISO 639-1 language code of the main content (e.g. "pt", "en", "es")\n'
        '- "doc_type": one of: recipe_book, faq, manual, product_catalog, legal, general\n\n'
        "Return only valid JSON, no markdown, no explanation."
    )
    content = f'Document: "{source_label}"\n\nSample chunks ({len(sampled)} of {n}):\n\n{joined}'

    try:
        resp = openai_chat_completion(prompt, content, model="gpt-4o-mini")
        s = resp.strip()
        # Strip markdown code fences if present
        if s.startswith("```"):
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
        result = json.loads(s)
        # Validate expected keys are present
        for key in ("topics", "keywords", "description", "language", "doc_type"):
            if key not in result:
                result[key] = None
        return result
    except Exception as e:
        print(f"[suggest_collection_metadata] Failed: {e}")
        return {}


def suggest_scraper_fix(config_yaml: str, problem_description: str) -> str:
    """
    Ask the LLM to suggest YAML or code changes when a scrape failed or returned too little.
    Returns a string (suggested YAML diff or snippet).
    """
    from llms.openai_utils import openai_chat_completion
    prompt = """You are helping fix a web scraper config. The user has a scraper config (YAML) and a problem (e.g. "there's a Show more button", "content is behind a tab"). Suggest concrete changes: either an updated YAML snippet or a short Python/Selenium snippet to add (e.g. click a button, wait for selector). Be concise."""
    content = f"Current config or relevant part:\n{config_yaml[:2000]}\n\nProblem: {problem_description}"
    return openai_chat_completion(prompt, content)
