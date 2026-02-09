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


def suggest_scraper_fix(config_yaml: str, problem_description: str) -> str:
    """
    Ask the LLM to suggest YAML or code changes when a scrape failed or returned too little.
    Returns a string (suggested YAML diff or snippet).
    """
    from llms.openai_utils import openai_chat_completion
    prompt = """You are helping fix a web scraper config. The user has a scraper config (YAML) and a problem (e.g. "there's a Show more button", "content is behind a tab"). Suggest concrete changes: either an updated YAML snippet or a short Python/Selenium snippet to add (e.g. click a button, wait for selector). Be concise."""
    content = f"Current config or relevant part:\n{config_yaml[:2000]}\n\nProblem: {problem_description}"
    return openai_chat_completion(prompt, content)
