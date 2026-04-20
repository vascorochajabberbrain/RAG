"""
Post-scrape text cleaning for generic (non-structured) web pages.

Strips common e-commerce noise: navigation, footer, star ratings,
sort controls, breadcrumbs, cookie notices, loading spinners.

Used between fetch and chunk steps. Does NOT apply to pages with
structured extraction (product pages, recipe pages) — those already
produce clean chunks via their chunk_template.
"""

import re


def clean_scraped_text(text: str) -> str:
    """
    Clean raw scraped page text by removing common boilerplate.
    Returns cleaned text suitable for chunking and RAG retrieval.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_noise_line(stripped):
            continue
        # Clean star ratings in-line
        stripped = _clean_star_ratings(stripped)
        if stripped:
            cleaned_lines.append(stripped)

    result = "\n".join(cleaned_lines)

    # Remove repeated whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"  +", " ", result)

    return result.strip()


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
