"""
Generic httpx + BeautifulSoup4 scraper.
Use for confirmed server-side rendered (SSR) sites — ~10x faster than Playwright, no browser needed.
Falls back gracefully to body text if selectors are not found.

YAML config keys used: same as playwright_scraper.py (engine field differs).
"""

import re
import time
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── Always-on strip list ──────────────────────────────────────────────────────
# Mirrors playwright_scraper.py — keep them in sync. Split into two
# groups so the CMP roots can be reused for cmp_text capture (used
# by jBKB to seed the per-CBVA "Cookies and Privacy" source).
_BASELINE_HTML_STRIP = ["script", "style", "noscript", "iframe"]
_BASELINE_CMP_STRIP = [
    # CookieYes — anchor to word boundary so we don't hit
    # Elementor's "elementor-sticky--*" classes (sti·cky-·-active).
    '[class^="cky-"]',
    '[class*=" cky-"]',
    "#onetrust-banner-sdk",             # OneTrust banner
    "#onetrust-consent-sdk",            # OneTrust prefs panel
    "#CybotCookiebotDialog",            # Cookiebot
    ".cookie-notice-container",         # WP Cookie Notice
    ".iubenda-cs-container",            # Iubenda
    ".klaro",                           # Klaro
    "#cmpwrapper",                      # Quantcast / TCFv2
    "#cookiescript_injected",           # CookieScript
    "#hs-eu-cookie-confirmation",       # HubSpot
]
_BASELINE_STRIP = _BASELINE_HTML_STRIP + _BASELINE_CMP_STRIP


# ── Public entry point ────────────────────────────────────────────────────────

def run_httpx_scraper(config: dict, cancel_check=None) -> tuple:
    """
    Run an httpx+BS4 scrape according to config.
    Returns (raw_text: str, scraped_items: list[dict])
    where each item is {"url": str, "text": str}.
    """
    mode = config.get("scrape_mode", "crawl")

    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=30) as client:
        if mode == "sitemap":
            result = _sitemap_scrape(client, config, cancel_check=cancel_check)
        elif mode == "url_list":
            urls = config.get("urls", [])
            if not urls:
                result = ("Error: urls list required for scrape_mode=url_list.", [])
            else:
                result = _scrape_url_list(client, urls, config, cancel_check=cancel_check)
        elif mode == "single_page":
            result = _single_page_scrape(client, config)
        elif mode == "crawl":
            result = _crawl_scrape(client, config, cancel_check=cancel_check)
        else:
            result = (f"Error: Unknown scrape_mode '{mode}'.", [])

    return result


# ── Scrape modes ──────────────────────────────────────────────────────────────

def _sitemap_scrape(client: httpx.Client, config: dict, cancel_check=None) -> tuple:
    sitemap_url = config.get("sitemap_url")
    if not sitemap_url:
        return "Error: sitemap_url required for scrape_mode=sitemap.", []

    urls = _fetch_sitemap_urls(client, sitemap_url)
    urls = _filter_urls(urls, config)

    if not urls:
        return "Error: No URLs matched after filtering the sitemap.", []

    print(f"[httpx_scraper] Sitemap: {len(urls)} URLs to scrape.")
    return _scrape_url_list(client, urls, config, cancel_check=cancel_check)


def _crawl_scrape(client: httpx.Client, config: dict, cancel_check=None) -> tuple:
    start_url = config.get("start_url")
    if not start_url:
        return "Error: start_url required for scrape_mode=crawl.", []

    max_pages = config.get("max_pages", 200)
    link_filter = config.get("link_filter", "")
    visited = set()
    stack = [start_url]
    results = []
    items = []

    while stack and len(visited) < max_pages:
        if cancel_check and cancel_check():
            print(f"[httpx_scraper] ⛔ Cancelled after {len(visited)} pages.")
            break
        url = stack.pop()
        if url in visited:
            continue
        visited.add(url)

        print(f"[httpx_scraper] Crawling ({len(visited)}/{max_pages}): {url}")
        try:
            resp = client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[httpx_scraper] WARNING: Failed {url}: {e}")
            continue

        text = _fetch_and_extract_from_soup(soup, url, config)
        if text:
            results.append(text)
            items.append({"url": url, "text": text})

        # Collect links
        if not config.get("structured_extraction"):
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and href not in visited and href not in stack:
                    if not link_filter or link_filter in href:
                        stack.append(href)

        time.sleep(config.get("page_delay", 0.3))

    print(f"[httpx_scraper] Crawl complete: {len(visited)} pages.")
    return "\n\n---\n\n".join(r for r in results if r), items


def _single_page_scrape(client: httpx.Client, config: dict) -> tuple:
    url = config.get("url")
    if not url:
        return "Error: url required for scrape_mode=single_page.", []
    try:
        resp = client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text, text_baseline, cmp_text, outgoing_links = _fetch_and_extract_from_soup_pair(soup, url, config)
        text = text or ""
        items = (
            [{
                "url": url,
                "text": text,
                "text_baseline": text_baseline or "",
                "cmp_text": cmp_text or "",
                "outgoing_links": outgoing_links or [],
            }]
            if text else []
        )
        return text, items
    except Exception as e:
        return f"Error fetching {url}: {e}", []


# ── Core fetch + extract ──────────────────────────────────────────────────────

def _fetch_and_extract_from_soup(soup: BeautifulSoup, url: str, config: dict) -> str:
    """Legacy single-return variant used by sitemap / crawl modes."""
    filtered, _baseline, _cmp, _links = _fetch_and_extract_from_soup_pair(soup, url, config)
    return filtered


def _extract_cmp_text_from_soup(soup: BeautifulSoup, selectors: list) -> str:
    """Joined inner text from any CMP-root element. Empty when none
    match. Filters to TOP-LEVEL matches only — children whose ancestor
    also matches the selector list would have their text already
    included in the parent's get_text(), producing 4-6× duplication
    of the same banner block. Must be called BEFORE _strip_selectors
    removes them."""
    if not selectors:
        return ""
    matched: list = []
    for sel in selectors:
        try:
            matched.extend(soup.select(sel))
        except Exception:
            continue
    if not matched:
        return ""
    # Keep only top-level matches (no matched ancestor).
    matched_set = set(id(m) for m in matched)
    top_level = []
    for el in matched:
        ancestor = el.parent
        skip = False
        while ancestor is not None:
            if id(ancestor) in matched_set:
                skip = True
                break
            ancestor = ancestor.parent
        if not skip:
            top_level.append(el)
    parts: list[str] = []
    for el in top_level:
        txt = el.get_text(separator="\n", strip=True)
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)


def _fetch_and_extract_from_soup_pair(soup: BeautifulSoup, url: str, config: dict) -> tuple:
    """Extract text twice from the same soup plus capture CMP text + links:
      - filtered        (baseline strip + user exclude_selectors)
      - baseline        (baseline strip only)
      - cmp_text        (CMP roots, captured BEFORE stripping)
      - outgoing_links  (same-host absolute URLs from <a href>, captured
                         BEFORE any stripping so the orphan-detection
                         link graph isn't biased by user exclude_selectors)

    Structured-extraction paths return the same text for both since they
    don't walk the DOM via text_selector. Downstream always needs SOMETHING
    to hash as content_raw."""
    if config.get("structured_extraction"):
        r = _extract_structured(soup, url, config) or ""
        return r, r, "", []

    selector = config.get("text_selector", "body")

    # Capture CMP text BEFORE stripping — those elements are about
    # to be decomposed in place.
    cmp_text = _extract_cmp_text_from_soup(soup, _BASELINE_CMP_STRIP)

    # Capture same-host outgoing <a href> URLs BEFORE the link-
    # annotation pass mutates the tree. Feeds the orphan-detection
    # BFS on the jBKB side.
    outgoing_links = _extract_outgoing_links(soup, url)

    # Annotate <a href> elements with markdown-style link syntax
    # ([text](abs_url)) BEFORE any stripping or text extraction so the
    # final chunk text carries inline URLs that the LLM can cite (e.g.
    # recipe pages that link to product pages — the appended per-
    # collection prompt can ask the model to include those URLs in
    # answers). Default ON; opt out by setting `preserve_links: false`
    # in scraper_config.
    if config.get("preserve_links", True):
        _annotate_links(soup, url)

    # Baseline strip + extract (soup is mutated in place).
    _strip_selectors(soup, _BASELINE_STRIP)
    baseline_text = _extract_via_selector(soup, selector)

    # User strip + extract again (soup mutated further).
    user_selectors = [s for s in (config.get("exclude_selectors") or []) if s]
    _strip_selectors(soup, user_selectors)
    filtered_text = _extract_via_selector(soup, selector)

    return filtered_text, baseline_text, cmp_text, outgoing_links


def _extract_outgoing_links(soup: BeautifulSoup, page_url: str) -> list:
    """Return a deduplicated list of same-host absolute URLs reachable
    via <a href> on this page. Used by jBKB's orphan-detection BFS to
    determine which sitemap URLs are clickable from the homepage.

    Filters:
      - Drops fragment-only / javascript: / mailto: / tel: / empty hrefs.
      - Resolves relative hrefs against the page URL.
      - Same-host only (cross-domain links can't make THIS site's URLs
        reachable).
      - Strips the fragment so /foo#bar and /foo are the same node in
        the graph.
    Idempotent — call before any soup-stripping passes."""
    from urllib.parse import urljoin, urlparse
    page_host = (urlparse(page_url).hostname or "").lower()
    seen: set = set()
    out: list = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        try:
            abs_url = urljoin(page_url, href)
            parsed = urlparse(abs_url)
        except Exception:
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        if (parsed.hostname or "").lower() != page_host:
            continue
        # Drop fragment so URL identity matches sitemap entries.
        normalized = parsed._replace(fragment="").geturl()
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _annotate_links(soup: BeautifulSoup, page_url: str) -> None:
    """Replace every <a href="..."> in soup with a text node carrying
    `[anchor text](absolute_url)` so subsequent get_text() preserves
    the link inline. Skips fragment / javascript: / mailto: / empty-
    text anchors. Idempotent within a soup since replace_with removes
    the <a> from the tree.

    Mutation is intentional — it runs once per page before either of
    the get_text passes, so both baseline_text and filtered_text
    inherit the same annotations."""
    from urllib.parse import urljoin
    for a in list(soup.find_all("a", href=True)):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        text = a.get_text(strip=True)
        if not text:
            # Image-only or empty anchor — drop the URL since there's
            # nothing to label it with.
            continue
        try:
            abs_url = urljoin(page_url, href)
        except Exception:
            abs_url = href
        a.replace_with(f"[{text}]({abs_url})")


def _strip_selectors(soup: BeautifulSoup, selectors: list) -> None:
    for sel in selectors:
        if not sel:
            continue
        try:
            for el in soup.select(sel):
                el.decompose()
        except Exception as e:
            print(f"[httpx_scraper] WARNING: bad selector {sel!r}: {e}")


def _strip_excluded(soup: BeautifulSoup, config: dict) -> None:
    """DEPRECATED backward-compat shim. New code should use
    _strip_selectors with an explicit list."""
    user_selectors = config.get("exclude_selectors") or []
    _strip_selectors(soup, _BASELINE_STRIP + list(user_selectors))


def _extract_via_selector(soup: BeautifulSoup, selector: str) -> str:
    el = soup.select_one(selector)
    if not el:
        el = soup.find("body")
    if not el:
        return ""
    return el.get_text(separator="\n", strip=True)


def _extract_text(soup: BeautifulSoup, config: dict) -> str:
    """DEPRECATED single-output. Use _fetch_and_extract_from_soup_pair."""
    _strip_excluded(soup, config)
    return _extract_via_selector(soup, config.get("text_selector", "body"))


def _extract_structured(soup: BeautifulSoup, url: str, config: dict) -> str:
    """Extract named fields and render via chunk_template."""
    fields: dict = {"url": url}
    extraction = config.get("structured_extraction", {})

    for field_name, selector in extraction.items():
        if selector == "url":
            fields[field_name] = url
        elif selector.startswith("li under "):
            heading_text = selector[len("li under "):]
            fields[field_name] = _extract_list_under_heading(soup, heading_text)
        elif selector.startswith("url of "):
            inner_selector = selector[len("url of "):]
            el = soup.select_one(inner_selector)
            fields[field_name] = el.get("href", "") if el else ""
        else:
            el = soup.select_one(selector)
            fields[field_name] = el.get_text(strip=True) if el else ""

    # Parse WooCommerce-style attribute table rows
    for row in config.get("attribute_rows", []):
        row_name = row.get("row_name", "")
        field = row.get("field", "")
        if not row_name or not field:
            continue
        th = soup.find("th", string=re.compile(re.escape(row_name), re.IGNORECASE))
        if th:
            td = th.find_next_sibling("td")
            if td:
                raw = td.get_text(separator=", ", strip=True)
                normalised = ", ".join(v.strip() for v in re.split(r"[\n,]+", raw) if v.strip())
                fields[field] = normalised
            else:
                fields[field] = ""
        else:
            fields[field] = ""

    template = config.get("chunk_template", "{url}")
    try:
        return template.format_map(fields).strip()
    except KeyError as e:
        print(f"[httpx_scraper] WARNING: chunk_template missing key {e}. Fields: {list(fields.keys())}")
        return str(fields)


def _extract_list_under_heading(soup: BeautifulSoup, heading_text: str) -> str:
    """
    Find h1/h2/h3 whose text matches heading_text,
    then collect all <li> items until the next heading.
    """
    heading_tags = ["h1", "h2", "h3", "h4"]
    target = None
    for tag in soup.find_all(heading_tags):
        if heading_text.lower() in tag.get_text(strip=True).lower():
            target = tag
            break

    if not target:
        return ""

    items = []
    node = target.find_next_sibling()
    while node:
        if node.name in heading_tags:
            break
        if node.name in ("ul", "ol"):
            for li in node.find_all("li"):
                text = li.get_text(strip=True)
                if text:
                    items.append(f"- {text}")
        elif node.name == "li":
            text = node.get_text(strip=True)
            if text:
                items.append(f"- {text}")
        node = node.find_next_sibling()

    return "\n".join(items)


# ── Sitemap helpers ───────────────────────────────────────────────────────────

def _fetch_sitemap_urls(client: httpx.Client, sitemap_url: str) -> list:
    """Fetch sitemap XML and return list of URLs."""
    try:
        resp = client.get(sitemap_url)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]
        if not urls:
            urls = [loc.text for loc in root.findall(".//loc") if loc.text]
        return urls
    except Exception as e:
        print(f"[httpx_scraper] ERROR fetching sitemap {sitemap_url}: {e}")
        return []


def _filter_urls(urls: list, config: dict) -> list:
    """Apply url_filter (substring) and url_allowlist."""
    url_filter = config.get("url_filter", "")
    url_allowlist = config.get("url_allowlist", [])
    if url_filter:
        urls = [u for u in urls if url_filter in u]
    if url_allowlist:
        urls = [u for u in urls if any(slug in u for slug in url_allowlist)]
    return urls


def _scrape_url_list(client: httpx.Client, urls: list, config: dict, cancel_check=None) -> tuple:
    """Fetch and extract from a list of URLs. Returns (joined_text, scraped_items)."""
    results = []
    items = []
    for i, url in enumerate(urls, 1):
        if cancel_check and cancel_check():
            print(f"[httpx_scraper] ⛔ Cancelled after {len(results)}/{len(urls)} pages.")
            break
        print(f"[httpx_scraper] Scraping {i}/{len(urls)}: {url}")
        try:
            resp = client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = _fetch_and_extract_from_soup(soup, url, config)
            if text:
                results.append(text)
                items.append({"url": url, "text": text})
            else:
                print(f"[httpx_scraper] WARNING: Empty result for {url}")
        except Exception as e:
            print(f"[httpx_scraper] WARNING: Failed {url}: {e}")
        if i < len(urls):
            time.sleep(config.get("page_delay", 0.3))

    print(f"[httpx_scraper] Done. {len(results)}/{len(urls)} pages extracted.")
    return "\n\n---\n\n".join(results), items
