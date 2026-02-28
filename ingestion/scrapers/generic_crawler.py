"""
Generic Playwright-based web crawler for the RAG pipeline.

Driven entirely by YAML config — no custom Python code needed per site.
Replaces the legacy Selenium peixefresco crawler.

Supported YAML config keys:
    start_url           (required) First URL to visit
    link_prefix         (required) Only follow links starting with this prefix
    max_pages           Max pages to crawl (default: 100)
    content_selector    CSS selector to extract text from (default: "body")
    delay_between_pages Seconds to wait between pages (default: 1.0)
    interactions        List of interaction steps to run before text extraction:
        - type: click_all       Click every element matching selector (e.g. accordions)
          selector: "button[aria-label='accordion']"
        - type: click_repeat    Click a button repeatedly until it disappears (Load More)
          selector: "button.load-more"
        - type: scroll_to_bottom  Scroll down N times (infinite scroll)
          times: 5

Usage:
    from ingestion.scrapers.generic_crawler import run_generic_crawl
    text = run_generic_crawl(config_dict)
"""
import time


def run_generic_crawl(config: dict) -> str:
    """
    Crawl a site using Playwright (sync), driven by a YAML config dict.
    Returns all extracted page text concatenated as a single string.
    """
    start_url = config.get("start_url")
    if not start_url:
        return "Error: config must have start_url."

    link_prefix = config.get("link_prefix", start_url)
    max_pages = int(config.get("max_pages", 100))
    selector = config.get("content_selector", "body")
    delay = float(config.get("delay_between_pages", 1.0))
    interactions = config.get("interactions") or []

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "Error: playwright is not installed. "
            "Run: pip install playwright && playwright install chromium"
        )

    collected_text = []
    visited = set()
    stack = [start_url]

    print(f"[crawler] Starting crawl: {start_url} (prefix={link_prefix}, max={max_pages})")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (compatible; RAGBuilder/1.0)"
        )
        page = context.new_page()

        while stack and len(visited) < max_pages:
            url = stack.pop()
            if url in visited:
                continue
            visited.add(url)

            print(f"[crawler] ({len(visited)}/{max_pages}) {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                if delay > 0:
                    time.sleep(delay)
            except Exception as e:
                print(f"[crawler] Failed to load {url}: {e}")
                continue

            # Run interactions before text extraction
            _run_interactions(page, interactions)

            # Extract text from content selector (fall back to body)
            try:
                el = page.query_selector(selector)
                if el:
                    text = el.inner_text()
                else:
                    text = page.inner_text("body")
                if text.strip():
                    collected_text.append(f"--- {url} ---\n{text.strip()}")
            except Exception as e:
                print(f"[crawler] Text extraction failed on {url}: {e}")

            # Collect links for DFS
            try:
                hrefs = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.href)"
                )
                for href in hrefs:
                    if (
                        href
                        and href.startswith(link_prefix)
                        and href not in visited
                        and href not in stack
                    ):
                        stack.append(href)
            except Exception as e:
                print(f"[crawler] Link collection failed on {url}: {e}")

        context.close()
        browser.close()

    print(f"[crawler] Done. Visited {len(visited)} pages, collected {len(collected_text)} text blocks.")
    return "\n\n".join(collected_text)


def _run_interactions(page, interactions: list):
    """Execute interaction steps on the current page before text extraction."""
    for interaction in interactions:
        itype = interaction.get("type", "")
        if itype == "click_all":
            _click_all(page, interaction.get("selector", ""))
        elif itype == "click_repeat":
            _click_repeat(page, interaction.get("selector", ""))
        elif itype == "scroll_to_bottom":
            _scroll_to_bottom(page, int(interaction.get("times", 3)))
        else:
            print(f"[crawler] Unknown interaction type: {itype}")


def _click_all(page, selector: str):
    """
    Click every element matching the selector once (e.g. accordion panels).
    Scrolls each into view first. Silently skips failures.
    """
    if not selector:
        return
    try:
        els = page.query_selector_all(selector)
        for el in els:
            try:
                el.scroll_into_view_if_needed()
                el.click()
                page.wait_for_timeout(200)  # small wait for content to reveal
            except Exception:
                pass  # accordion may already be open, or stale element
        if els:
            print(f"[crawler] click_all: clicked {len(els)} elements matching '{selector}'")
    except Exception as e:
        print(f"[crawler] click_all({selector}) error: {e}")


def _click_repeat(page, selector: str):
    """
    Click a button repeatedly until it disappears (e.g. Load More / Show More).
    Stops when the element is gone or not visible.
    """
    if not selector:
        return
    clicks = 0
    max_clicks = 50  # safety cap
    while clicks < max_clicks:
        try:
            btn = page.query_selector(selector)
            if not btn or not btn.is_visible():
                break
            btn.scroll_into_view_if_needed()
            btn.click()
            page.wait_for_timeout(1200)  # wait for new content to load
            clicks += 1
        except Exception:
            break
    if clicks:
        print(f"[crawler] click_repeat: clicked '{selector}' {clicks} time(s)")


def _scroll_to_bottom(page, times: int = 3):
    """
    Scroll to the bottom N times to trigger infinite scroll content loading.
    """
    for i in range(times):
        page.keyboard.press("End")
        page.wait_for_timeout(900)
    if times:
        print(f"[crawler] scroll_to_bottom: scrolled {times} time(s)")
