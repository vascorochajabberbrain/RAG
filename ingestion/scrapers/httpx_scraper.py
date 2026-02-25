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


# ── Public entry point ────────────────────────────────────────────────────────

def run_httpx_scraper(config: dict) -> tuple:
    """
    Run an httpx+BS4 scrape according to config.
    Returns (raw_text: str, scraped_items: list[dict])
    where each item is {"url": str, "text": str}.
    """
    mode = config.get("scrape_mode", "crawl")

    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=30) as client:
        if mode == "sitemap":
            result = _sitemap_scrape(client, config)
        elif mode == "single_page":
            result = _single_page_scrape(client, config)
        elif mode == "crawl":
            result = _crawl_scrape(client, config)
        else:
            result = (f"Error: Unknown scrape_mode '{mode}'.", [])

    return result


# ── Scrape modes ──────────────────────────────────────────────────────────────

def _sitemap_scrape(client: httpx.Client, config: dict) -> tuple:
    sitemap_url = config.get("sitemap_url")
    if not sitemap_url:
        return "Error: sitemap_url required for scrape_mode=sitemap.", []

    urls = _fetch_sitemap_urls(client, sitemap_url)
    urls = _filter_urls(urls, config)

    if not urls:
        return "Error: No URLs matched after filtering the sitemap.", []

    print(f"[httpx_scraper] Sitemap: {len(urls)} URLs to scrape.")
    return _scrape_url_list(client, urls, config)


def _crawl_scrape(client: httpx.Client, config: dict) -> tuple:
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
        text = _fetch_and_extract_from_soup(soup, url, config) or ""
        items = [{"url": url, "text": text}] if text else []
        return text, items
    except Exception as e:
        return f"Error fetching {url}: {e}", []


# ── Core fetch + extract ──────────────────────────────────────────────────────

def _fetch_and_extract_from_soup(soup: BeautifulSoup, url: str, config: dict) -> str:
    if config.get("structured_extraction"):
        return _extract_structured(soup, url, config)
    else:
        return _extract_text(soup, config)


def _extract_text(soup: BeautifulSoup, config: dict) -> str:
    """Extract plain text using text_selector (CSS selector). Default: body."""
    selector = config.get("text_selector", "body")
    el = soup.select_one(selector)
    if not el:
        el = soup.find("body")
    if not el:
        return ""
    return el.get_text(separator="\n", strip=True)


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


def _scrape_url_list(client: httpx.Client, urls: list, config: dict) -> tuple:
    """Fetch and extract from a list of URLs. Returns (joined_text, scraped_items)."""
    results = []
    items = []
    for i, url in enumerate(urls, 1):
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
