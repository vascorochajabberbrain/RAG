"""
Optional text filters for scraped content. Used by workflow CLEAN step when source_config.filter_name is set.
All filter logic is self-contained here — no external scraper dependencies.
"""
import re


def apply_filter(text: str, filter_name: str) -> str:
    """
    Apply a named filter to scraped text. Returns cleaned text.
    filter_name can be: none, peixefresco (no-op), heyharper.
    """
    if not text:
        return ""
    if filter_name in ("none", ""):
        return text
    if filter_name == "peixefresco":
        return text
    if filter_name == "heyharper":
        return _apply_heyharper_filters(text)
    return text


# ── HeyHarper filters (inlined from url_ingestion_legacy.py) ─────────────────

def _apply_heyharper_filters(text: str) -> str:
    filtered = _remove_end_of_page_info(text)
    filtered = _remove_product_info_regex(filtered)
    filtered = _remove_influencers_tags(filtered)
    return filtered


def _remove_end_of_page_info(text: str) -> str:
    pattern = """Free gift with 1st order
Join our newsletter to claim it
Product
Brand
Resources
Support
Join us
Social
EUR €
English
Terms of service
·
Privacy Policy"""
    return re.sub(pattern, "", text)


def _remove_product_info_regex(text: str) -> str:
    def _optional(regex):
        return "(?:" + regex + ")?"

    status_line = r"(?:New|Bestseller|Save on the Set|\d\.\d|50% Off|Out of stock)"
    pos_status_line = r"(?:Save on the Set|Game Day Glow|Overtime Ready|All-Weather Ready)"
    promotion_line = r"(?:€\d+(?:[ \t]+with[ \t]+\d+%[ \t]+Off)?|Final Sale)"
    category_line = r"(?:Bracelet|Set|Set 2|Watch|Anklet|Subscription Box|Huggies|Earrings|Ring|Rings|Necklace|Jewelry Case|[\w \t\-]*Bikini|Shorts|Dress|Silver|Extenders|Pendant|Choker)"
    name_line = r"[\w \t\-\&]+"
    other_price_line = r"€\d+"
    price_line = r"€\d+"
    add_line = r"(?:Add|Notify me)"

    composed_regex = (
        "(?:"
        + _optional(status_line + r"\s*")
        + _optional(pos_status_line + r"\s*")
        + _optional(promotion_line + r"\s*")
        + name_line + r"\s*"
        + category_line + r"\s*"
        + _optional(other_price_line + r"\s*")
        + price_line + r"\s*"
        + add_line + r"\s*"
        + "|"
        + name_line + r"\s*"
        + price_line + r"\s*"
        + promotion_line + r"\s*"
        + add_line + r"\s*"
        + ")"
    )
    return re.sub(composed_regex, "", text, flags=re.DOTALL)


def _remove_influencers_tags(text: str) -> str:
    """Remove lines starting with @ (influencer handles). Be careful: any line starting with @ is removed."""
    pattern = r'^@.*\n'
    return re.sub(pattern, "", text, flags=re.MULTILINE)
