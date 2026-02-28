"""
Site Analysis Wizard — backend utilities.

Fetches sitemap structure, samples pages, suggests RAG collection groupings,
and generates scraper YAML configs.

Used by POST /api/wizard/analyse and POST /api/wizard/confirm in web/app.py.
"""

import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx
import yaml
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_SITEMAP_TIMEOUT = 15
_PAGE_TIMEOUT = 10
_SAMPLE_PAGES = 3
_PREVIEW_CHARS = 500

_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sitemap_structure(domain_url: str, login_config: dict | None = None) -> list:
    """
    Discover content categories on a website via its sitemap(s).

    Returns a list of category dicts, sorted by url_count descending.
    Each dict:
      id, display_name, sitemap_url, url_filter, url_count,
      sample_urls, preview, source

    login_config (optional): dict with keys username, password, url (login page URL).
      When provided, performs an httpx form-POST login before sampling pages so that
      previews work on gated sites.
    """
    domain_url = _normalise_url(domain_url)

    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=_SITEMAP_TIMEOUT) as client:
        # Detect Shopify early
        if _detect_shopify(client, domain_url):
            return [_shopify_sentinel(domain_url)]

        # Try to find the sitemap
        xml_text, sitemap_url_used = _find_sitemap(client, domain_url)
        if not xml_text:
            return [_no_sitemap_sentinel()]

        # Parse
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return [_no_sitemap_sentinel()]

        tag = root.tag.lower()

        if "sitemapindex" in tag:
            categories = _parse_sitemap_index(client, root, domain_url)
        elif "urlset" in tag:
            urls = _extract_locs(root)
            categories = _group_urls_by_path(sitemap_url_used, urls)
        else:
            return [_no_sitemap_sentinel()]

        if not categories:
            return [_no_sitemap_sentinel()]

        # If login credentials provided, attempt httpx login before page sampling
        if login_config and login_config.get("username"):
            _httpx_login(client, domain_url, login_config)

        # Sample pages for each category
        for cat in categories:
            previews = []
            for url in (cat.get("sample_urls") or [])[:_SAMPLE_PAGES]:
                text = _sample_page(client, url)
                if text:
                    previews.append(text)
            cat["preview"] = " … ".join(previews)[:_PREVIEW_CHARS]

        # Sort by url_count descending
        categories.sort(key=lambda c: c.get("url_count", 0), reverse=True)
        return categories


def suggest_collections(categories: list) -> list:
    """
    Ask GPT-4o-mini to group categories into suggested RAG collections.
    Falls back to rule-based grouping on any failure.
    """
    from llms.openai_utils import openai_chat_completion

    # Filter out sentinel categories for the LLM
    real_cats = [c for c in categories if not c["id"].startswith("_")]
    if not real_cats:
        return []

    # Build a compact summary for the LLM
    summary = [
        {
            "id": c["id"],
            "display_name": c["display_name"],
            "url_count": c["url_count"],
            "preview": (c.get("preview") or "")[:200],
        }
        for c in real_cats
    ]

    system_prompt = (
        "You are helping set up a RAG (Retrieval-Augmented Generation) chatbot for a website.\n"
        "Given the sitemap categories discovered on the site, suggest how to group them into "
        "RAG collections. Each collection should have a clear, distinct purpose for a chatbot.\n\n"
        "Return ONLY a valid JSON array. Each element:\n"
        "{\n"
        '  "collection_name": "snake_case (2-3 words max)",\n'
        '  "display_name": "Human readable name",\n'
        '  "doc_type": "product_catalog | recipe_book | faq | manual | legal | general",\n'
        '  "categories": ["list of category ids"],\n'
        '  "rationale": "One sentence why this grouping makes sense for a chatbot"\n'
        "}\n\n"
        "Rules:\n"
        "- Products must always be their own collection (doc_type: product_catalog)\n"
        "- FAQ, terms, returns, about, contact, legal pages → ALWAYS their own separate collection "
        "named 'customer_support' (doc_type: faq). Never merge with products, recipes, or other content.\n"
        "- If both FAQ/about AND legal/terms sections exist, keep them together in customer_support "
        "unless they are each very large (>50 pages), in which case split legal out separately.\n"
        "- Recipe/how-to content → recipe_book\n"
        "- If a category has <3 pages it can be merged into a related collection\n"
        "- Categories with unclear value for a chatbot should still be listed but with doc_type: general\n"
        "- No markdown, no explanation — ONLY the JSON array"
    )
    content = f"Site categories:\n{json.dumps(summary, ensure_ascii=False, indent=2)}"

    try:
        resp = openai_chat_completion(system_prompt, content, model="gpt-4o-mini")
        s = resp.strip()
        if s.startswith("```"):
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
            s = s.strip()
        result = json.loads(s)
        if isinstance(result, list) and result:
            return result
    except Exception as e:
        print(f"[sitemap_analyzer] LLM suggest_collections failed: {e} — using fallback")

    return _fallback_suggest(real_cats)


def fetch_all_pages(sitemap_url: str, url_filter: str = None) -> list:
    """
    Fetch all <loc> URLs from a single sitemap file (called lazily when user expands a sitemap row).
    If url_filter is given (path substring, e.g. '/produto/'), only matching URLs are returned.
    """
    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=_SITEMAP_TIMEOUT) as client:
        r = client.get(sitemap_url, timeout=_SITEMAP_TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        urls = _extract_locs(root)
    if url_filter:
        urls = [u for u in urls if url_filter in u]
    return urls


def generate_scraper_config(collection: dict, domain_url: str) -> dict:
    """
    Generate a scraper config dict for a confirmed collection.
    Used for inline storage in solutions.yaml (no YAML file needed).

    collection: {
      collection_name, display_name, doc_type,
      categories: [{id, sitemap_url, url_filter, url_count, excluded_urls}]
    }
    """
    doc_type = collection.get("doc_type", "general")
    coll_name = collection.get("collection_name", "collection")
    categories = collection.get("categories", [])

    high_js = {"product_catalog", "recipe_book"}
    engine = "playwright" if doc_type in high_js else "httpx"

    cfg = {
        "name": coll_name,
        "engine": engine,
        "scrape_mode": "sitemap",
    }

    if len(categories) == 1:
        cat = categories[0]
        if cat.get("sitemap_url"):
            cfg["sitemap_url"] = cat["sitemap_url"]
        if cat.get("url_filter"):
            cfg["url_filter"] = cat["url_filter"]
    elif len(categories) > 1:
        for cat in categories:
            if cat.get("sitemap_url"):
                cfg["sitemap_url"] = cat["sitemap_url"]
                break
        filters = [cat["url_filter"] for cat in categories if cat.get("url_filter")]
        if filters:
            cfg["url_allowlist"] = filters

    cfg["text_selector"] = "main, article, .entry-content, body"

    all_excluded = []
    for cat in categories:
        if cat.get("excluded_urls"):
            all_excluded.extend(cat["excluded_urls"])
    if all_excluded:
        cfg["excluded_urls"] = all_excluded

    if collection.get("extra_pages"):
        cfg["extra_urls"] = list(collection["extra_pages"])

    return cfg


def generate_scraper_yaml(collection: dict, domain_url: str) -> str:
    """
    Generate a scraper YAML config string for a confirmed collection.
    Thin wrapper around generate_scraper_config() for backward compatibility.
    """
    import yaml as _yaml
    cfg = generate_scraper_config(collection, domain_url)
    coll_name = collection.get("collection_name", "collection")
    doc_type = collection.get("doc_type", "general")
    comment = (
        f"# {collection.get('display_name', coll_name)}"
        f" — auto-generated by jB RAG Site Analysis Wizard\n"
        f"# Source: {domain_url}\n"
        f"# doc_type: {doc_type}\n"
        f"# Edit: add structured_extraction, chunk_template, custom_js_extraction as needed.\n"
    )
    return comment + _yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Sitemap fetching helpers
# ---------------------------------------------------------------------------

def _normalise_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url
    return url


def _find_sitemap(client: httpx.Client, domain_url: str):
    """Try common sitemap URLs and robots.txt. Returns (xml_text, sitemap_url) or (None, None)."""
    candidates = [
        f"{domain_url}/sitemap_index.xml",
        f"{domain_url}/sitemap.xml",
    ]

    # Check robots.txt for Sitemap: lines
    try:
        r = client.get(f"{domain_url}/robots.txt", timeout=8)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm_url = line.split(":", 1)[1].strip()
                    if sm_url not in candidates:
                        candidates.insert(0, sm_url)
    except Exception:
        pass

    for url in candidates:
        try:
            r = client.get(url, timeout=_SITEMAP_TIMEOUT)
            if r.status_code == 200 and r.text.strip().startswith("<"):
                return r.text, url
        except Exception:
            continue

    return None, None


def _extract_locs(root: ET.Element) -> list:
    """Extract all <loc> text values from an XML element tree."""
    locs = [el.text for el in root.findall(".//sm:loc", _NS) if el.text]
    if not locs:
        # Fallback: no-namespace
        locs = [el.text for el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc") if el.text]
    if not locs:
        locs = [el.text for el in root.iter() if el.tag.endswith("loc") and el.text]
    return locs


def _parse_sitemap_index(client: httpx.Client, root: ET.Element, domain_url: str) -> list:
    """Parse a sitemap index: each child sitemap becomes one category."""
    sub_sitemap_urls = _extract_locs(root)
    categories = []
    for sm_url in sub_sitemap_urls:
        cat_id = _derive_category_id(sm_url)
        cat = {
            "id": cat_id,
            "display_name": cat_id,
            "sitemap_url": sm_url,
            "url_filter": None,
            "url_count": 0,
            "sample_urls": [],
            "preview": "",
            "source": "sitemap_index",
        }
        # Fetch sub-sitemap to count + sample URLs
        try:
            r = client.get(sm_url, timeout=_SITEMAP_TIMEOUT)
            if r.status_code == 200:
                sub_root = ET.fromstring(r.text)
                urls = _extract_locs(sub_root)
                cat["url_count"] = len(urls)
                cat["sample_urls"] = urls[:_SAMPLE_PAGES]
        except Exception:
            pass
        categories.append(cat)
    return categories


def _group_urls_by_path(sitemap_url: str, urls: list) -> list:
    """Group flat sitemap URLs by first path segment."""
    groups: dict = {}
    for url in urls:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        seg = parts[0] if parts else "_root"
        groups.setdefault(seg, []).append(url)

    categories = []
    other_urls = []
    for seg, seg_urls in groups.items():
        if len(seg_urls) <= 1:
            other_urls.extend(seg_urls)
            continue
        categories.append({
            "id": seg,
            "display_name": seg,
            "sitemap_url": sitemap_url,
            "url_filter": f"/{seg}/",
            "url_count": len(seg_urls),
            "sample_urls": seg_urls[:_SAMPLE_PAGES],
            "preview": "",
            "source": "flat",
        })

    if other_urls:
        categories.append({
            "id": "_other",
            "display_name": "other pages",
            "sitemap_url": sitemap_url,
            "url_filter": None,
            "url_count": len(other_urls),
            "sample_urls": other_urls[:_SAMPLE_PAGES],
            "preview": "",
            "source": "flat",
        })

    return categories


def _derive_category_id(sitemap_url: str) -> str:
    """'https://example.com/product-sitemap.xml' → 'product-sitemap'"""
    path = urlparse(sitemap_url).path
    filename = path.rstrip("/").split("/")[-1]
    return re.sub(r"\.(xml|gz)$", "", filename, flags=re.IGNORECASE)


def _httpx_login(client: httpx.Client, domain_url: str, login_config: dict) -> bool:
    """
    Attempt a form-POST login using httpx so that subsequent page sampling
    works on gated sites. Returns True if the POST succeeded (2xx or redirect).
    The client's cookie jar is updated in-place.
    """
    login_url = (login_config.get("url") or "").strip()
    if not login_url:
        # Guess common login paths
        from urllib.parse import urlparse
        parsed = urlparse(domain_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        for path in ["/login", "/wp-login.php", "/account/login", "/signin"]:
            candidate = base + path
            try:
                r = client.get(candidate, timeout=8)
                if r.status_code == 200 and "password" in r.text.lower():
                    login_url = candidate
                    break
            except Exception:
                pass
    if not login_url:
        print("[sitemap_analyzer] Could not determine login URL — skipping login.")
        return False
    try:
        data = {
            "username": login_config.get("username", ""),
            "email": login_config.get("username", ""),
            "log": login_config.get("username", ""),  # WordPress field name
            "pwd": login_config.get("password", ""),  # WordPress field name
            "password": login_config.get("password", ""),
            "wp-submit": "Log In",
            "redirect_to": domain_url,
        }
        r = client.post(login_url, data=data, timeout=15)
        print(f"[sitemap_analyzer] Login attempt → {r.status_code} from {login_url}")
        return r.status_code in (200, 302)
    except Exception as e:
        print(f"[sitemap_analyzer] Login error: {e}")
        return False


def _sample_page(client: httpx.Client, url: str) -> str:
    """Fetch a page and extract a short text preview. Returns '' on any error."""
    try:
        r = client.get(url, timeout=_PAGE_TIMEOUT)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        # Try progressively broader selectors
        for selector in ["main", "article", ".entry-content", ".woocommerce", "body"]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator=" ")
                text = re.sub(r"\s+", " ", text).strip()
                return text[:_PREVIEW_CHARS]
        return ""
    except Exception:
        return ""


def _detect_shopify(client: httpx.Client, domain_url: str) -> bool:
    """Return True if this looks like a Shopify store."""
    try:
        r = client.get(f"{domain_url}/products.json?limit=1", timeout=8)
        if r.status_code == 200:
            data = r.json()
            return isinstance(data, dict) and "products" in data
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Sentinel categories
# ---------------------------------------------------------------------------

def _no_sitemap_sentinel() -> dict:
    return {
        "id": "_no_sitemap",
        "display_name": "No sitemap found",
        "sitemap_url": None,
        "url_filter": None,
        "url_count": 0,
        "sample_urls": [],
        "preview": "No sitemap.xml or sitemap_index.xml was found. You can still configure a scraper manually in the Build RAG tab.",
        "source": "none",
    }


def _shopify_sentinel(domain_url: str) -> dict:
    return {
        "id": "_shopify",
        "display_name": "Shopify store detected",
        "sitemap_url": None,
        "url_filter": None,
        "url_count": -1,
        "sample_urls": [],
        "preview": f"This looks like a Shopify store ({domain_url}). Use the Shopify Stores tab to connect via the Admin API for best results.",
        "source": "shopify",
        "shopify_url": domain_url,
    }


# ---------------------------------------------------------------------------
# Fallback collection suggestion (no LLM)
# ---------------------------------------------------------------------------

def _fallback_suggest(categories: list) -> list:
    """Rule-based collection suggestions when LLM is unavailable."""
    PRODUCT_KEYS = {"product", "produto", "produit", "produkt", "tienda", "catalog"}
    RECIPE_KEYS = {"recipe", "receita", "recette", "rezept", "ricetta"}
    FAQ_KEYS = {"page", "faq", "term", "legal", "sobre", "about", "support", "help", "contact"}

    name_to_cats: dict = {}

    for cat in categories:
        cid = cat["id"].lower()
        if any(k in cid for k in PRODUCT_KEYS):
            name = "products"
            doc_type = "product_catalog"
        elif any(k in cid for k in RECIPE_KEYS):
            name = "recipes"
            doc_type = "recipe_book"
        elif any(k in cid for k in FAQ_KEYS):
            name = "customer_support"
            doc_type = "faq"
        else:
            name = re.sub(r"[-_]sitemap$", "", cid).replace("-", "_")
            doc_type = "general"

        if name not in name_to_cats:
            name_to_cats[name] = {"doc_type": doc_type, "cats": []}
        name_to_cats[name]["cats"].append(cat["id"])

    result = []
    for name, info in name_to_cats.items():
        result.append({
            "collection_name": name,
            "display_name": name.replace("_", " ").title(),
            "doc_type": info["doc_type"],
            "categories": info["cats"],
            "rationale": f"Auto-suggested from {len(info['cats'])} category(s).",
        })
    return result
