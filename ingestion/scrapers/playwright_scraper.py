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


# ── Always-on strip list ──────────────────────────────────────────────────────
# Applied to every page before text extraction, regardless of the
# user's scraper_config. Split into two groups so the CMP roots can
# also be reused for cmp_text capture (extracted BEFORE stripping
# and surfaced to jBKB for the synthetic "Cookies and Privacy"
# source).
_BASELINE_HTML_STRIP = ["script", "style", "noscript", "iframe"]
_BASELINE_CMP_STRIP = [
    # CookieYes — must anchor to word boundary, otherwise the
    # substring match also hits Elementor's "elementor-sticky--*"
    # classes (sti·cky-·-active etc.). Two-clause selector:
    #   [class^="cky-"]    → first class begins with cky-
    #   [class*=" cky-"]   → any subsequent class (space-prefixed)
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

def run_playwright_scraper(config: dict, cancel_check=None) -> tuple:
    """
    Run a Playwright scrape according to config.
    Returns (raw_text: str, scraped_items: list[dict])
    where each item is {"url": str, "text": str}.
    Callers that only need the text can do: text, _ = run_playwright_scraper(config)
    cancel_check: optional callable returning True when the user wants to stop.
    """
    mode = config.get("scrape_mode", "crawl")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            # Perform login before scraping if credentials are provided
            login_cfg = config.get("login_config")
            if login_cfg and login_cfg.get("username") and login_cfg.get("password"):
                _perform_login(page, login_cfg)

            if mode == "sitemap":
                result = _sitemap_scrape(page, config, cancel_check=cancel_check)
            elif mode == "url_list":
                urls = config.get("urls", [])
                if not urls:
                    result = ("Error: urls list required for scrape_mode=url_list.", [])
                else:
                    result = _scrape_url_list(page, urls, config, cancel_check=cancel_check)
            elif mode == "single_page":
                result = _single_page_scrape(page, config)
            elif mode == "crawl":
                result = _crawl_scrape(page, config, cancel_check=cancel_check)
            else:
                result = (f"Error: Unknown scrape_mode '{mode}'.", [])
        finally:
            browser.close()

    return result


def _perform_login(page, login_config: dict) -> bool:
    """
    Navigate to login URL and fill credentials before scraping gated content.
    Selectors are optional — falls back to common patterns if not provided.
    Returns True on success, False on failure (scraping continues either way).
    """
    login_url = login_config.get("url") or ""
    u_sel     = (login_config.get("username_selector")
                 or "input[type=email], input[name*=user], input[name*=email], input[name*=login]")
    p_sel     = login_config.get("password_selector") or "input[type=password]"
    sub_sel   = login_config.get("submit_selector") or "button[type=submit]"
    username  = login_config.get("username", "")
    password  = login_config.get("password", "")

    if not username or not password:
        return False
    try:
        if login_url:
            page.goto(login_url, wait_until="networkidle", timeout=30000)
        page.locator(u_sel).first.fill(username)
        page.locator(p_sel).first.fill(password)
        page.locator(sub_sel).first.click()
        page.wait_for_load_state("networkidle", timeout=15000)
        print(f"[playwright_scraper] Login completed for {login_url or 'current page'}")
        return True
    except Exception as e:
        print(f"[playwright_scraper] Login failed (continuing anyway): {e}")
        return False


# ── Scrape modes ──────────────────────────────────────────────────────────────

def _sitemap_scrape(page, config: dict, cancel_check=None) -> tuple:
    sitemap_url = config.get("sitemap_url")
    if not sitemap_url:
        return "Error: sitemap_url required for scrape_mode=sitemap.", []

    urls = _fetch_sitemap_urls(sitemap_url)
    urls = _filter_urls(urls, config)

    if not urls:
        return "Error: No URLs matched after filtering the sitemap.", []

    print(f"[playwright_scraper] Sitemap: {len(urls)} URLs to scrape.")
    return _scrape_url_list(page, urls, config, cancel_check=cancel_check)


def _crawl_scrape(page, config: dict, cancel_check=None) -> tuple:
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
            print(f"[playwright_scraper] ⛔ Cancelled after {len(visited)} pages.")
            break
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
    text, text_baseline, cmp_text, outgoing_links = _fetch_and_extract_pair(page, url, config)
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


# ── Core fetch + extract ──────────────────────────────────────────────────────

def _fetch_and_extract(page, url: str, config: dict) -> str:
    """Navigate to URL and extract text (structured or plain).

    Legacy single-return variant kept for the sitemap + crawl modes
    that don't need the baseline pair. New code should prefer
    _fetch_and_extract_pair which returns (filtered, baseline,
    cmp_text)."""
    filtered, _baseline, _cmp, _links = _fetch_and_extract_pair(page, url, config)
    return filtered


def _extract_cmp_text(page, selectors: list) -> str:
    """Inner text from any CMP-root element on the page, joined with
    blank-line separators. Empty string when nothing matches.

    Filters to TOP-LEVEL matches only — if a matched element has a
    matched ancestor we skip it, otherwise the parent's innerText
    already contains the descendant's text and we'd emit the same
    block 4-6× (cky-consent-container → cky-consent-bar →
    cky-notice → cky-notice-group → cky-notice-des all carry the
    same banner text via innerText).

    Must be called BEFORE _strip_selectors removes the elements.
    Used by jBKB to seed the per-CBVA "Cookies and Privacy"
    synthetic source — which keeps the cookie/privacy notice text
    searchable in Qdrant after the baseline strip removes it from
    every regular page."""
    if not selectors:
        return ""
    try:
        sel = ", ".join(selectors)
        # textContent (not innerText) — innerText respects CSS
        # visibility, so cky-modal (visibility:hidden by default
        # until "Customize" is clicked) returns "" and we'd lose
        # the ~1.5KB preferences-panel text. textContent ignores
        # CSS, so we get the full DOM text exactly the same way
        # the regular scraper's body.get_text() does.
        return page.evaluate(
            """(sel) => {
                const all = Array.from(document.querySelectorAll(sel));
                // keep only those whose ancestors don't include
                // another matched element
                const topLevel = all.filter(el => !all.some(other => other !== el && other.contains(el)));
                return topLevel
                    .map(el => (el.textContent || '').replace(/\\s+/g, ' ').trim())
                    .filter(Boolean)
                    .join('\\n\\n');
            }""",
            sel,
        ) or ""
    except Exception as e:
        print(f"[playwright_scraper] WARNING: cmp_text extraction failed: {e}")
        return ""


def _fetch_and_extract_pair(page, url: str, config: dict) -> tuple:
    """Navigate + extract text TWICE plus capture CMP text + links:
      - filtered text  (baseline strip + user exclude_selectors applied)
      - baseline text  (baseline strip only, no user exclude_selectors)
      - cmp_text       (joined inner text from any CMP root, captured
                        BEFORE stripping)
      - outgoing_links (same-host absolute URLs from <a href>, captured
                        BEFORE any stripping or annotation — feeds the
                        orphan-detection BFS on the jBKB side)

    Baseline text is the "truly raw" content that jBKB stores in
    content_raw for change-detection. cmp_text feeds the synthetic
    "Cookies and Privacy" source per CBVA."""
    # domcontentloaded is much faster and more reliable than
    # networkidle on sites loaded with trackers (GTM, analytics,
    # chat widgets) that keep the network busy indefinitely. We
    # then settle explicitly via page_delay so JS-injected content
    # has a chance to render. networkidle on a tracker-heavy site
    # times out → goto raises → scraper returns "" and the user
    # sees "Fetch returned no content" even though the page is fine.
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        # Fall back to plain goto (no wait) so we still capture
        # whatever HTML arrived before the timeout.
        try:
            page.goto(url, wait_until="commit", timeout=10000)
        except Exception as e2:
            print(f"[playwright_scraper] WARNING: Failed to load {url}: {e} (fallback also failed: {e2})")
            return "", "", "", []

    # Slightly longer default settle (was 0.5s) — Elementor + tracker
    # sites need ~1-2s to inject content after DCL.
    time.sleep(config.get("page_delay", 1.5))

    # Structured / custom-JS paths don't walk the DOM via text_selector,
    # so baseline vs filtered doesn't apply. Return the single result
    # for both, plus an empty cmp_text (these modes typically extract
    # specific structured data, not full-page boilerplate).
    if config.get("custom_js_extraction"):
        r = _extract_custom_js(page, url, config) or ""
        return r, r, "", []
    if config.get("structured_extraction"):
        r = _extract_structured(page, url, config) or ""
        return r, r, "", []

    # Expand FAQ accordions / <details> / Bootstrap collapsibles etc.
    # BEFORE text extraction so Pattern B/C sites (content hidden via
    # display:none or JS-loaded on click) contribute their answers too.
    if config.get("expand_collapsibles", True):
        _expand_collapsibles(page)

    # Capture CMP text BEFORE stripping — once _BASELINE_STRIP runs
    # those elements are gone. Cheap (one querySelectorAll per page).
    cmp_text = _extract_cmp_text(page, _BASELINE_CMP_STRIP)

    # Capture same-host outgoing <a href> URLs BEFORE the annotation
    # pass mutates the DOM. Feeds the orphan-detection BFS on jBKB.
    outgoing_links = _extract_outgoing_links(page, url)

    # Annotate <a href> elements with markdown-style link syntax
    # ([text](abs_url)) BEFORE any stripping or text extraction so
    # subsequent inner_text() picks up inline URLs the LLM can cite.
    # Default ON; opt out via `preserve_links: false` on scraper_config.
    if config.get("preserve_links", True):
        _annotate_links(page)

    # Generic text extraction — two passes with the text_selector.
    selector = config.get("text_selector", "body")

    _strip_selectors(page, _BASELINE_STRIP)
    baseline_text = _extract_with_selector(page, selector)

    user_selectors = [s for s in (config.get("exclude_selectors") or []) if s]
    _strip_selectors(page, user_selectors)
    filtered_text = _extract_with_selector(page, selector)

    return filtered_text, baseline_text, cmp_text, outgoing_links


def _extract_outgoing_links(page, page_url: str) -> list:
    """Return a deduplicated list of same-host absolute URLs reachable
    via <a href> on this page. Used by jBKB's orphan-detection BFS to
    determine which sitemap URLs are clickable from the homepage.
    Mirror of httpx_scraper._extract_outgoing_links but reads from the
    live Playwright DOM via page.evaluate. Filters fragment-only /
    javascript: / mailto: / tel: / cross-host hrefs; strips URL
    fragments so /foo#bar matches sitemap entry /foo."""
    try:
        from urllib.parse import urlparse
        page_host = (urlparse(page_url).hostname or "").lower()
        if not page_host:
            return []
        hrefs = page.evaluate(
            """() => {
                const out = new Set();
                for (const a of document.querySelectorAll('a[href]')) {
                    const raw = (a.getAttribute('href') || '').trim();
                    if (!raw) continue;
                    if (raw.startsWith('#') || raw.startsWith('javascript:')
                        || raw.startsWith('mailto:') || raw.startsWith('tel:')) continue;
                    try { out.add(a.href); } catch (e) { /* ignore */ }
                }
                return Array.from(out);
            }"""
        ) or []
        seen: set = set()
        cleaned: list = []
        for href in hrefs:
            try:
                parsed = urlparse(href)
            except Exception:
                continue
            if parsed.scheme not in ("http", "https"):
                continue
            if (parsed.hostname or "").lower() != page_host:
                continue
            normalized = parsed._replace(fragment="").geturl()
            if normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned
    except Exception as e:
        print(f"[playwright_scraper] WARNING: outgoing-links extraction failed for {page_url}: {e}")
        return []


def _annotate_links(page) -> None:
    """Replace every anchor in the live DOM with a text node
    `[anchor text](absolute_url)` so the next inner_text() call
    embeds URLs inline. Mirror of httpx_scraper._annotate_links;
    runs once per page before either of the strip passes.

    Uses node.href (not getAttribute) so the URL comes back already
    resolved against the page's base URL — Playwright's DOM does
    that automatically. Skips fragment / javascript: / mailto: / tel:
    / empty-text anchors."""
    try:
        page.evaluate(
            """() => {
                const anchors = Array.from(document.querySelectorAll('a[href]'));
                for (const a of anchors) {
                    const raw = (a.getAttribute('href') || '').trim();
                    if (!raw) continue;
                    if (raw.startsWith('#') || raw.startsWith('javascript:')
                        || raw.startsWith('mailto:') || raw.startsWith('tel:')) continue;
                    const text = (a.textContent || '').trim();
                    if (!text) continue;
                    const url = a.href; // already absolute
                    try {
                        a.replaceWith(document.createTextNode(`[${text}](${url})`));
                    } catch (e) { /* ignore one-off failures */ }
                }
            }"""
        )
    except Exception as e:
        print(f"[playwright_scraper] WARNING: Could not annotate links: {e}")


# Common accordion / collapsible triggers across the platforms we've
# seen in operator pages. ARIA + class patterns only — nothing
# page-specific. Clicking an already-expanded accordion is a no-op, so
# running these unconditionally is safe.
#
# Conservative-by-design: only click button-ish elements (or native
# <details>). We deliberately do NOT match anchors (`a`) or broad
# `.collapsed` class selectors — those would cause accidental
# navigation / mis-clicks on nav menus.
_EXPAND_SELECTORS = (
    # ARIA contract — used by accessible accordions on most modern stacks.
    '[aria-expanded="false"]',
    # Native HTML <details>.
    'details:not([open])',
    # Elementor (WordPress page builder).
    '.elementor-tab-title:not(.elementor-active)',
    '.elementor-toggle-title:not(.elementor-active)',
    '.elementor-accordion-title:not(.elementor-active)',
    # Bootstrap collapse.
    '[data-toggle="collapse"]',
    '[data-bs-toggle="collapse"]',
    # Shopify Dawn / Liquid theme primitives. Dawn ships an accessible
    # <summary>-based accordion but several merchant themes use a
    # button-class hybrid.
    '.collapsible-trigger:not(.collapsible-trigger--inactive)',
    'summary.h4',
    'summary.accordion__summary',
    'button.accordion-header',
    'button.accordion-toggle',
    'button.accordion__button',
    'button[class*="accordion-button"]',
    'button[class*="accordion__header"]',
    # FAQ-specific class patterns that surface on custom React /
    # Vue / Webflow builds. Restricted to <button> so we don't click
    # decorative span / h3 elements; the role="button" fallback below
    # catches the heading-as-trigger pattern.
    'button.faq-question',
    'button.faq-toggle',
    'button.faq__question',
    'button[class*="faq-question"]',
    'button[class*="faq__question"]',
    # Heading-as-trigger pattern (h2/h3/h4 with role=button, common in
    # custom builds that put click handlers on the heading itself).
    # Scoped to h2–h5 to avoid clicking arbitrary headings.
    'h2[role="button"][aria-expanded]',
    'h3[role="button"][aria-expanded]',
    'h4[role="button"][aria-expanded]',
    'h5[role="button"][aria-expanded]',
)


def _expand_collapsibles(page) -> None:
    """Click every common accordion / details / collapse trigger so the
    subsequent inner_text() call sees expanded content.

    Runs THREE rounds with a small wait between each — some widgets
    reveal nested collapsibles only after the parent opens, and some
    JS-driven FAQs hydrate the answer DOM on first click (so a second
    pass picks up the newly-visible elements). Empirically:
    - Round 1 opens top-level FAQ groups.
    - Round 2 opens individual Q items inside those groups.
    - Round 3 catches any sub-collapsibles inside long answers.

    A final post-round wait lets any JS-loaded content settle into the
    DOM before the inner_text() call that follows.
    """
    try:
        for _round in range(3):
            page.evaluate(
                """(sels) => {
                    for (const s of sels) {
                        try {
                            document.querySelectorAll(s).forEach(el => {
                                try {
                                    if (el.tagName === 'DETAILS') {
                                        el.open = true;
                                    } else {
                                        el.click();
                                    }
                                } catch (e) { /* single click failure ignored */ }
                            });
                        } catch (e) { /* bad selector, move on */ }
                    }
                }""",
                list(_EXPAND_SELECTORS),
            )
            page.wait_for_timeout(400)
        # Post-round settle — JS-loaded FAQ answers sometimes lag the
        # final click by a few hundred ms (network roundtrip for content
        # that wasn't in the initial HTML).
        page.wait_for_timeout(600)
    except Exception as e:
        print(f"[playwright_scraper] WARNING: expand_collapsibles failed: {e}")


def _strip_selectors(page, selectors: list) -> None:
    """Remove every element matching any selector from the live DOM.
    Silent on bad selectors so one typo can't break the whole scrape."""
    if not selectors:
        return
    try:
        page.evaluate(
            "(sels) => { for (const s of sels) { try { document.querySelectorAll(s).forEach(el => el.remove()); } catch (e) { console.warn('bad selector', s, e); } } }",
            selectors,
        )
    except Exception as e:
        print(f"[playwright_scraper] WARNING: strip_selectors failed: {e}")


def _strip_excluded(page, config: dict) -> None:
    """DEPRECATED backward-compat shim. Strips baseline + user in one
    call; left so third-party callers don't break. New code should use
    _strip_selectors with an explicit list."""
    user_selectors = [s for s in (config.get("exclude_selectors") or []) if s]
    _strip_selectors(page, _BASELINE_STRIP + user_selectors)


def _extract_with_selector(page, selector: str) -> str:
    try:
        el = page.locator(selector).first
        return el.inner_text(timeout=5000).strip()
    except Exception:
        try:
            return page.locator("body").first.inner_text(timeout=5000).strip()
        except Exception as e:
            print(f"[playwright_scraper] WARNING: Could not extract via {selector!r}: {e}")
            return ""


def _extract_text(page, config: dict) -> str:
    """DEPRECATED — returns only the filtered text. Use
    _fetch_and_extract_pair for new code."""
    _strip_excluded(page, config)
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


def _extract_custom_js(page, url: str, config: dict) -> str | list[str]:
    """
    Run a custom JavaScript function in the browser context.

    The JS function can return:
      - A dict of fields → rendered as a single chunk via chunk_template
      - An array of dicts → each rendered as a separate chunk (one per item)

    config keys:
      custom_js_extraction: str  — JS function expression returning a dict or array of dicts
      chunk_template: str        — template rendered with the returned fields + {url}
    """
    js_fn = config.get("custom_js_extraction", "")
    if not js_fn:
        return ""

    try:
        result = page.evaluate(js_fn)
        template = config.get("chunk_template", "{url}")

        # Array of dicts → multiple chunks from one page
        if isinstance(result, list):
            chunks = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                item["url"] = url
                try:
                    chunks.append(template.format_map(item).strip())
                except KeyError as e:
                    print(f"[playwright_scraper] WARNING: chunk_template missing key {e} in array item")
            print(f"[playwright_scraper] custom_js returned {len(chunks)} items for {url}")
            return chunks if chunks else ""

        # Single dict → one chunk
        if isinstance(result, dict):
            result["url"] = url
            return template.format_map(result).strip()

        print(f"[playwright_scraper] WARNING: custom_js_extraction returned unexpected type {type(result).__name__} for {url}")
        return ""
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


def _scrape_url_list(page, urls: list, config: dict, cancel_check=None) -> tuple:
    """Scrape a list of URLs. Returns (joined_text, scraped_items).

    When custom_js_extraction returns an array, each element becomes a separate item.
    """
    results = []
    items = []
    for i, url in enumerate(urls, 1):
        if cancel_check and cancel_check():
            print(f"[playwright_scraper] ⛔ Cancelled after {len(results)}/{len(urls)} pages.")
            break
        print(f"[playwright_scraper] Scraping {i}/{len(urls)}: {url}")
        text = _fetch_and_extract(page, url, config)
        if isinstance(text, list):
            # custom_js returned multiple chunks from one page
            for chunk in text:
                if chunk:
                    results.append(chunk)
                    items.append({"url": url, "text": chunk})
        elif text:
            results.append(text)
            items.append({"url": url, "text": text})
        else:
            print(f"[playwright_scraper] WARNING: Empty result for {url}")
        if i < len(urls):
            time.sleep(config.get("page_delay", 0.5))

    print(f"[playwright_scraper] Done. {len(items)} items from {len(urls)} pages.")
    return "\n\n---\n\n".join(results), items
