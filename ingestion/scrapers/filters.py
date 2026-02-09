"""
Optional text filters for scraped content. Used by workflow CLEAN step when source_config.filter_name is set.
"""
import re


def apply_filter(text: str, filter_name: str) -> str:
    """
    Apply a named filter to scraped text. Returns cleaned text.
    filter_name can be: none, peixefresco (no-op for now), heyharper (product/influencer removal).
    """
    if not text:
        return ""
    if filter_name in ("none", ""):
        return text
    if filter_name == "peixefresco":
        return text
    if filter_name == "heyharper":
        try:
            from ingestion.url_ingestion import apply_filters
        except ImportError:
            from url_ingestion import apply_filters
        return apply_filters(text)
    return text
