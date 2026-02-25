"""
Load solution specifications from solutions.yaml.
Later: replace this with DB reads keeping the same interface (get_solution, list_solutions, resolve_alias).

Schema: every solution has a 'collections' list. Each collection entry has:
  id, display_name, collection_name, collection_type, scraper_name (optional), routing (optional)
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


def get_collections(solution_id: str) -> list[dict]:
    """
    Return the list of collection dicts for a solution.
    Each entry has: id, display_name, collection_name, collection_type,
                    scraper_name (optional), routing (optional dict).
    Returns [] if solution not found.
    """
    sol = get_solution(solution_id)
    if not sol:
        return []
    return sol.get("collections") or []


def resolve_alias(alias: str) -> Optional[dict]:
    """
    Resolve a shortcut to a solution dict, or None.
    Matches: solution id, solution aliases, or collection id within any solution.
    When a collection id is matched, returns the parent solution dict
    (callers can then use get_collections() to find the specific collection).
    """
    alias_lower = alias.strip().lower()
    for s in _load():
        # Match solution id
        if s.get("id") == alias or s.get("id", "").lower() == alias_lower:
            return s
        # Match solution aliases
        for a in s.get("aliases") or []:
            if str(a).strip().lower() == alias_lower:
                return s
        # Match collection id within this solution
        for c in s.get("collections") or []:
            if c.get("id") == alias or c.get("id", "").lower() == alias_lower:
                return s
    return None


def get_collection_by_id(collection_id: str) -> Optional[dict]:
    """
    Find a specific collection dict by its id, searching across all solutions.
    Returns the collection dict (not the parent solution), or None.
    """
    for s in _load():
        for c in s.get("collections") or []:
            if c.get("id") == collection_id:
                return c
    return None


def reload():
    """Clear cache so next access re-reads the file (e.g. after edit)."""
    global _cache
    _cache = None
