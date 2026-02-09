"""
Load solution specifications from solutions.yaml.
Later: replace this with DB reads keeping the same interface (get_solution, list_solutions, resolve_alias).
"""
import os
from typing import Optional

_SPECS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPECS_FILE = os.path.join(_SPECS_DIR, "solutions.yaml")

_cache = None


def _load():
    global _cache
    if _cache is not None:
        return _cache
    try:
        import yaml
        with open(_SPECS_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _cache = data.get("solutions") or []
        return _cache
    except Exception as e:
        raise RuntimeError(f"Failed to load solution specs from {_SPECS_FILE}: {e}") from e


def list_solutions() -> list[dict]:
    """Return all solutions (list of dicts)."""
    return _load()


def get_solution(solution_id: str) -> Optional[dict]:
    """Return the solution dict for the given id, or None."""
    for s in _load():
        if s.get("id") == solution_id:
            return s
    return None


def resolve_alias(alias: str) -> Optional[dict]:
    """Resolve a shortcut (e.g. '1', 'FAQ') to a solution dict, or None."""
    alias_lower = alias.strip().lower()
    for s in _load():
        if s.get("id") == alias or s.get("id", "").lower() == alias_lower:
            return s
        for a in s.get("aliases") or []:
            if str(a).strip().lower() == alias_lower:
                return s
    return None


def reload():
    """Clear cache so next access re-reads the file (e.g. after edit)."""
    global _cache
    _cache = None
