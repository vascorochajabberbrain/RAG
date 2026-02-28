"""
WooCommerce product ingestion for the RAG pipeline.

Fetches all products from a WooCommerce store via the REST API v3 endpoint.
Requires a Consumer Key + Consumer Secret (Read access only).

Generate keys in: WP Admin → WooCommerce → Settings → Advanced → REST API

NOTE: Prices and stock levels will go stale as products change. Use the
"Re-sync" button in the RAG Builder UI to refresh the collection periodically.
For real-time price/stock queries, call the WooCommerce API live from the chatbot instead.

NOTE on security: Consumer keys are passed in source_config in-memory only.
They are stripped before saving state to disk to avoid credential leakage.
"""
import html
import re
import requests
from requests.auth import HTTPBasicAuth


def fetch_woocommerce_products(
    store_url: str,
    consumer_key: str,
    consumer_secret: str,
    options: dict = None
) -> list:
    """
    Fetch all products from a WooCommerce store via REST API.
    Returns a list of formatted text chunks (one per product).

    store_url:       e.g. 'https://store.mysite.com'
    consumer_key:    e.g. 'ck_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    consumer_secret: e.g. 'cs_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    options:         optional overrides (timeout, max_products, per_page)
    """
    options = options or {}
    base = store_url.strip().rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base

    timeout = options.get("timeout", 20)
    max_products = options.get("max_products", 10_000)
    per_page = min(int(options.get("per_page", 100)), 100)  # WooCommerce max is 100

    auth = HTTPBasicAuth(consumer_key, consumer_secret)
    api_base = f"{base}/wp-json/wc/v3/products"

    chunks = []
    page = 1

    while len(chunks) < max_products:
        url = f"{api_base}?per_page={per_page}&page={page}&status=publish"
        try:
            resp = requests.get(url, auth=auth, timeout=timeout, headers={"User-Agent": "RAGBuilder/1.0"})
        except requests.RequestException as e:
            print(f"[woocommerce] Request failed on page {page}: {e}")
            break

        if resp.status_code == 401:
            print(f"[woocommerce] 401 Unauthorized — check consumer key and secret.")
            break
        if resp.status_code == 404:
            print(f"[woocommerce] 404 — WooCommerce REST API not found at {api_base}. "
                  f"Check the store URL and ensure WooCommerce is installed.")
            break
        if not resp.ok:
            print(f"[woocommerce] HTTP {resp.status_code} on page {page}: {resp.text[:200]}")
            break

        try:
            products = resp.json()
        except Exception as e:
            print(f"[woocommerce] Failed to parse JSON on page {page}: {e}")
            break

        if not products:
            break  # No more products — we've reached the last page

        for product in products:
            chunk = _format_product(product, base)
            if chunk:
                chunks.append(chunk)

        # Check if there are more pages via X-WP-TotalPages header
        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        if page >= total_pages:
            break
        page += 1

    print(f"[woocommerce] Fetched {len(chunks)} products from {base}")
    return chunks


def _format_product(p: dict, base_url: str) -> str:
    """Format a single WooCommerce product dict as a RAG-friendly text chunk."""
    title = (p.get("name") or "").strip()
    if not title:
        return ""

    # Categories: list of {id, name, slug}
    categories = p.get("categories") or []
    category = ", ".join(c.get("name", "") for c in categories if c.get("name"))

    # Tags: list of {id, name, slug}
    tags_list = p.get("tags") or []
    tags = ", ".join(t.get("name", "") for t in tags_list if t.get("name"))

    # Price: prefer regular_price, fall back to price
    raw_price = p.get("regular_price") or p.get("price") or ""
    try:
        price = f"{float(raw_price):.2f}" if raw_price else ""
    except (ValueError, TypeError):
        price = str(raw_price)

    # Stock
    stock_status = p.get("stock_status", "")  # "instock" | "outofstock" | "onbackorder"
    in_stock = stock_status == "instock"

    # Description: prefer short_description, fall back to description
    short_desc_html = p.get("short_description") or ""
    full_desc_html = p.get("description") or ""
    description = _strip_html(short_desc_html) or _strip_html(full_desc_html)
    if len(description) > 600:
        description = description[:600].rsplit(" ", 1)[0] + "…"

    permalink = p.get("permalink") or base_url

    lines = [f"Product: {title}"]
    if category:
        lines.append(f"Category: {category}")
    if price:
        lines.append(f"Price: {price}")
    if description:
        lines.append(f"Description: {description}")
    if tags:
        lines.append(f"Tags: {tags}")
    lines.append(f"URL: {permalink}")
    lines.append(f"In stock: {'Yes' if in_stock else 'No'}")

    return "\n".join(lines)


def _strip_html(raw: str) -> str:
    """Remove HTML tags and normalise whitespace."""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
