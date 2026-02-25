"""
Generic Playwright-based scraper.
Supports scrape_mode: sitemap | crawl | single_page

YAML config keys used:
  engine: playwright
  scrape_mode: sitemap | crawl | single_page
  sitemap_url: str               (sitemap mode)
  url_filter: str                (optional substring filter on sitemap URLs)
  url_allowlist: list[str]       (optional: only URLs containing one of these substrings)
  start_url: str                 (crawl mode)
  link_filter: str               (optional substring filter for links to follow, crawl mode)
  max_pages: int                 (optional, default 200, crawl mode)
  url: str                       (single_page mode)
  text_selector: str             (optional CSS selector for text extraction, default "body")
  structured_extraction: dict    (optional field_name → CSS selector)
  attribute_rows: list[dict]     (optional WooCommerce attribute table rows)
  chunk_template: str            (required when structured_extraction is set)
  page_delay: float              (optional seconds between pages, default 0.5)
"""

import re
import time
import xml.etree.ElementTree as ET

import httpx
from playwright.sync_api import sync_playwright


# ── Public entry point ────────────────────────────────────────────────────────

def run_playwright_scraper(config: dict) -> tuple:
    """
    Run a Playwright scrape according to config.
    Returns (raw_text: str, scraped_items: list[dict])
    where each item is {"url": str, "text": str}.
    Callers that only need the text can do: text, _ = run_playwright_scraper(config)
    """
    mode = config.get("scrape_mode", "crawl")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            if mode == "sitemap":
                result = _sitemap_scrape(page, config)
            elif mode == "single_page":
                result = _single_page_scrape(page, config)
            elif mode == "crawl":
                result = _crawl_scrape(page, config)
            else:
                result = (f"Error: Unknown scrape_mode '{mode}'.", [])
        finally:
            browser.close()

    return result


# ── Scrape modes ──────────────────────────────────────────────────────────────

def _sitemap_scrape(page, config: dict) -> tuple:
    sitemap_url = config.get("sitemap_url")
    if not sitemap_url:
        return "Error: sitemap_url required for scrape_mode=sitemap.", []

    urls = _fetch_sitemap_urls(sitemap_url)
    urls = _filter_urls(urls, config)

    if not urls:
        return "Error: No URLs matched after filtering the sitemap.", []

    print(f"[playwright_scraper] Sitemap: {len(urls)} URLs to scrape.")
    return _scrape_url_list(page, urls, config)


def _crawl_scrape(page, config: dict) -> tuple:
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

        print(f"[playwright_scraper] Crawling ({len(visited)}/{max_pages}): {url}")
        text = _fetch_and_extract(page, url, config)
        if text:
            results.append(text)
            items.append({"url": url, "text": text})

        # Collect links from the page
        if not config.get("structured_extraction"):
            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
            for href in hrefs:
                if href and href not in visited and href not in stack:
                    if not link_filter or link_filter in href:
                        stack.append(href)

        time.sleep(config.get("page_delay", 0.5))

    print(f"[playwright_scraper] Crawl complete: {len(visited)} pages.")
    return "\n\n---\n\n".join(r for r in results if r), items


def _single_page_scrape(page, config: dict) -> tuple:
    url = config.get("url")
    if not url:
        return "Error: url required for scrape_mode=single_page.", []
    text = _fetch_and_extract(page, url, config) or ""
    items = [{"url": url, "text": text}] if text else []
    return text, items


# ── Core fetch + extract ──────────────────────────────────────────────────────

def _fetch_and_extract(page, url: str, config: dict) -> str:
    """Navigate to URL and extract text (structured or plain)."""
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
    except Exception as e:
        print(f"[playwright_scraper] WARNING: Failed to load {url}: {e}")
        return ""

    time.sleep(config.get("page_delay", 0.5))

    if config.get("custom_js_extraction"):
        return _extract_custom_js(page, url, config)
    elif config.get("structured_extraction"):
        return _extract_structured(page, url, config)
    else:
        return _extract_text(page, config)


def _extract_text(page, config: dict) -> str:
    """Extract plain text from the page using text_selector."""
    selector = config.get("text_selector", "body")
    try:
        el = page.locator(selector).first
        return el.inner_text(timeout=5000).strip()
    except Exception:
        try:
            return page.locator("body").first.inner_text(timeout=5000).strip()
        except Exception as e:
            print(f"[playwright_scraper] WARNING: Could not extract text: {e}")
            return ""


def _extract_custom_js(page, url: str, config: dict) -> str:
    """
    Run a custom JavaScript function in the browser context that returns a dict of fields.
    The JS must be a function expression: '() => { ... return { field1, field2, ... }; }'
    config keys:
      custom_js_extraction: str  — JS function expression returning a dict of fields
      chunk_template: str        — template rendered with the returned fields + {url}
    """
    js_fn = config.get("custom_js_extraction", "")
    if not js_fn:
        return ""

    try:
        fields = page.evaluate(js_fn)
        if not isinstance(fields, dict):
            print(f"[playwright_scraper] WARNING: custom_js_extraction did not return a dict for {url}")
            return ""
        fields["url"] = url  # always inject url
        template = config.get("chunk_template", "{url}")
        return template.format_map(fields).strip()
    except Exception as e:
        print(f"[playwright_scraper] WARNING: custom_js_extraction failed for {url}: {e}")
        return ""


def _extract_structured(page, url: str, config: dict) -> str:
    """
    Extract named fields using structured_extraction selectors,
    then render via chunk_template.
    """
    fields: dict = {"url": url}
    extraction = config.get("structured_extraction", {})

    for field_name, selector in extraction.items():
        if selector == "url":
            fields[field_name] = url
        elif selector.startswith("li under "):
            heading_text = selector[len("li under "):]
            fields[field_name] = _extract_list_under_heading(page, heading_text)
        elif selector.startswith("url of "):
            inner_selector = selector[len("url of "):]
            try:
                fields[field_name] = page.locator(inner_selector).first.get_attribute("href", timeout=3000) or ""
            except Exception:
                fields[field_name] = ""
        else:
            try:
                fields[field_name] = page.locator(selector).first.inner_text(timeout=3000).strip()
            except Exception:
                fields[field_name] = ""

    # Parse WooCommerce-style attribute table rows
    for row in config.get("attribute_rows", []):
        row_name = row.get("row_name", "")
        field = row.get("field", "")
        if not row_name or not field:
            continue
        try:
            th = page.locator(f"th:has-text('{row_name}')").first
            td = th.locator("xpath=following-sibling::td").first
            # Get all option text (attributes may be comma-separated or newline-separated)
            raw = td.inner_text(timeout=3000).strip()
            # Normalise: replace multiple newlines/whitespace with ", "
            normalised = ", ".join(v.strip() for v in re.split(r"[\n,]+", raw) if v.strip())
            fields[field] = normalised
        except Exception:
            fields[field] = ""

    template = config.get("chunk_template", "{url}")
    try:
        return template.format_map(fields).strip()
    except KeyError as e:
        print(f"[playwright_scraper] WARNING: chunk_template missing key {e}. Fields: {list(fields.keys())}")
        return str(fields)


def _extract_list_under_heading(page, heading_text: str) -> str:
    """
    Find an h2 (or h1/h3) whose text matches heading_text,
    then collect all <li> items until the next heading.
    Returns items as "- item\\n- item\\n..." string.
    """
    try:
        items = page.evaluate("""(headingText) => {
            const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4'));
            const target = headings.find(h => h.innerText.trim().toLowerCase().includes(headingText.toLowerCase()));
            if (!target) return [];

            const items = [];
            let node = target.nextElementSibling;
            while (node) {
                if (['H1','H2','H3','H4'].includes(node.tagName)) break;
                const lis = node.querySelectorAll('li');
                if (lis.length > 0) {
                    lis.forEach(li => items.push(li.innerText.trim()));
                } else if (node.tagName === 'LI') {
                    items.push(node.innerText.trim());
                }
                node = node.nextElementSibling;
            }
            return items;
        }""", heading_text)
        return "\n".join(f"- {item}" for item in items if item)
    except Exception as e:
        print(f"[playwright_scraper] WARNING: Could not extract list under '{heading_text}': {e}")
        return ""


# ── Sitemap helpers ───────────────────────────────────────────────────────────

def _fetch_sitemap_urls(sitemap_url: str) -> list:
    """Fetch sitemap XML via httpx (no browser needed for XML) and return list of URLs."""
    try:
        resp = httpx.get(sitemap_url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)"})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        # Handle both plain sitemaps and sitemap indexes
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]
        if not urls:
            # Try without namespace
            urls = [loc.text for loc in root.findall(".//loc") if loc.text]
        return urls
    except Exception as e:
        print(f"[playwright_scraper] ERROR fetching sitemap {sitemap_url}: {e}")
        return []


def _filter_urls(urls: list, config: dict) -> list:
    """Apply url_filter (substring) and url_allowlist to a list of URLs."""
    url_filter = config.get("url_filter", "")
    url_allowlist = config.get("url_allowlist", [])

    if url_filter:
        urls = [u for u in urls if url_filter in u]
    if url_allowlist:
        urls = [u for u in urls if any(slug in u for slug in url_allowlist)]
    return urls


def _scrape_url_list(page, urls: list, config: dict) -> tuple:
    """Scrape a list of URLs. Returns (joined_text, scraped_items)."""
    results = []
    items = []
    for i, url in enumerate(urls, 1):
        print(f"[playwright_scraper] Scraping {i}/{len(urls)}: {url}")
        text = _fetch_and_extract(page, url, config)
        if text:
            results.append(text)
            items.append({"url": url, "text": text})
        else:
            print(f"[playwright_scraper] WARNING: Empty result for {url}")
        if i < len(urls):
            time.sleep(config.get("page_delay", 0.5))

    print(f"[playwright_scraper] Done. {len(results)}/{len(urls)} pages extracted.")
    return "\n\n---\n\n".join(results), items
