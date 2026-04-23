"""
Post-scrape text cleaning for generic (non-structured) web pages.

Strips common e-commerce noise: navigation, footer, star ratings,
sort controls, breadcrumbs, cookie notices, loading spinners.

Used between fetch and chunk steps. Does NOT apply to pages with
structured extraction (product pages, recipe pages) — those already
produce clean chunks via their chunk_template.

When a `lexicon` set is provided (tokenized collection keywords),
filtering becomes language-agnostic: short lines with no overlap
against the lexicon are considered chrome and dropped. The original
hardcoded PT/EN patterns are retained as a fallback for when the
lexicon is empty or unavailable.
"""

import re


# Always-safe universal rules that apply regardless of lexicon:
#   - whitespace / repeated newlines
#   - lone star-character strings (leftover rating markers)
#   - "breadcrumb" lines using » (short lines only)
# Everything language-specific (cart, cookie, copyright etc.) is
# handled by either the lexicon check or the legacy pattern fallback.

def clean_scraped_text(text: str, lexicon: set | None = None) -> str:
    """
    Clean raw scraped page text by removing common boilerplate.
    Returns cleaned text suitable for chunking and RAG retrieval.

    lexicon: optional set of lowercased content tokens from the
             collection's routing terms. When provided, short lines
             with zero lexicon overlap are dropped as likely chrome.
             When None/empty, falls back to the hardcoded PT/EN
             pattern list.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_universal_noise(stripped):
            continue
        if lexicon:
            # Lexicon mode: the single rule is "short line + zero
            # keyword overlap = drop". Long lines always pass.
            # Short lines with a keyword hit pass. Short lines with
            # no hit are dropped.
            if len(stripped) < 80 and not _has_lexicon_hit(stripped, lexicon):
                continue
        else:
            # Fallback: the historic PT/EN hardcoded list.
            if _is_noise_line(stripped):
                continue
        # Normalize star ratings in-line for any surviving line.
        stripped = _clean_star_ratings(stripped)
        if stripped:
            cleaned_lines.append(stripped)

    result = "\n".join(cleaned_lines)

    # Remove repeated whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"  +", " ", result)

    return result.strip()


def _is_universal_noise(line: str) -> bool:
    """Language-independent noise patterns that are always safe to drop."""
    # Pure star characters (no rating value)
    if re.match(r"^[☆★\s]+$", line):
        return True
    # Breadcrumb patterns: "A » B » C" on a short line
    if "»" in line and len(line.split("»")) >= 2 and len(line) < 100:
        return True
    return False


def _has_lexicon_hit(line: str, lexicon: set) -> bool:
    """True if the line contains at least one lexicon token. Case
    insensitive, Unicode-aware word split."""
    for tok in re.findall(r"[\w]+", line.lower(), re.UNICODE):
        if tok in lexicon:
            return True
    return False


def _is_noise_line(line: str) -> bool:
    """Check if a line is common web page noise that should be stripped."""
    lower = line.lower()

    # Navigation / breadcrumb patterns
    if lower in ("home", "menu", "search", "close"):
        return True
    if lower.startswith("pular para o conteúdo") or lower.startswith("skip to content"):
        return True
    if lower.startswith("home »") or lower.startswith("home >"):
        return True
    # Breadcrumb pattern: "Category » Subcategory » Page"
    if "»" in line and len(line.split("»")) >= 2 and len(line) < 100:
        return True

    # Sort / filter controls
    sort_patterns = [
        "ordenar por", "order by", "sort by",
        "relevância", "relevance",
        "preço: ascendente", "preço: descendente",
        "price: low to high", "price: high to low",
        "designação",
    ]
    if any(p in lower for p in sort_patterns) and len(line) < 80:
        return True

    # Loading / spinner text
    loading_patterns = [
        "carregando", "loading", "a carregar",
        "carregando produtos",
    ]
    if any(lower.startswith(p) or lower == p for p in loading_patterns):
        return True

    # Cookie / privacy notices
    cookie_patterns = [
        "aceitar cookies", "accept cookies",
        "política de privacidade", "privacy policy",
        "usamos cookies", "we use cookies",
    ]
    if any(p in lower for p in cookie_patterns) and len(line) < 150:
        return True

    # Footer patterns — common across e-commerce sites
    footer_patterns = [
        "empresa", "company",
        "apoios", "imprensa", "press",
        "contactos", "contacts",
        "recomenda e ganha", "recommend and earn",
        "© ", "copyright",
        "todos os direitos", "all rights reserved",
    ]
    # Only match if the line is short (likely a footer link, not content)
    if any(p in lower for p in footer_patterns) and len(line) < 60:
        return True

    # "Add to cart" / "Buy" buttons
    cart_patterns = [
        "adicionar", "add to cart", "comprar",
        "quantidade de", "quantity of",
    ]
    if any(lower.startswith(p) for p in cart_patterns) and len(line) < 80:
        return True

    # Pure star characters (no rating value)
    if re.match(r"^[☆★\s]+$", line):
        return True

    return False


def _clean_star_ratings(text: str) -> str:
    """
    Convert star rating characters to structured text.
    '☆ ☆ ☆ ☆ ☆ 4.58/5 (12)' -> 'Rating: 4.58/5 (12 reviews)'
    '☆ ☆ ☆ ☆ ☆ 0/5 (0)' -> '' (remove zero ratings)
    """
    # Pattern: stars followed by rating
    pattern = r"[☆★\s]+([\d.]+)/5\s*\((\d+)\)"
    def replace_rating(m):
        rating = m.group(1)
        count = m.group(2)
        if rating == "0" and count == "0":
            return ""  # Remove zero ratings entirely
        return f"Rating: {rating}/5 ({count} reviews)"

    result = re.sub(pattern, replace_rating, text)
    # Clean up leftover lone stars
    result = re.sub(r"[☆★]{2,}", "", result)
    return result.strip()
