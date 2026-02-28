"""
Shopify product ingestion for the RAG pipeline.

Fetches all products from a public Shopify store via the /products.json endpoint
(no API key needed for public stores). Returns one formatted text chunk per product,
ready to be pushed directly to Qdrant (no CHUNK step required).

NOTE: Prices and stock levels will go stale as products change. Use the
"Re-sync" button in the RAG Builder UI to refresh the collection periodically.
For real-time price/stock queries, call the Shopify API live from the chatbot instead.
"""
import html
import re
import requests


def fetch_shopify_products(store_domain: str, options: dict = None) -> list:
    """
    Fetch all products from a public Shopify store.
    Returns a list of formatted text chunks (one per product).

    store_domain: e.g. 'mystore.myshopify.com' or 'https://mystore.com'
    options: optional overrides (timeout, max_products, etc.)
    """
    options = options or {}
    base = store_domain.strip().rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base

    timeout = options.get("timeout", 20)
    max_products = options.get("max_products", 10_000)

    chunks = []
    url = f"{base}/products.json?limit=250"

    while url and len(chunks) < max_products:
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "RAGBuilder/1.0"})
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[shopify] Request failed: {e}")
            break

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type or not resp.text.strip():
            # Store is password-protected or returned unexpected response
            print(
                f"[shopify] Expected JSON but got HTML — store may be password-protected. "
                f"Status: {resp.status_code}, URL: {url}"
            )
            break

        try:
            data = resp.json()
        except Exception as e:
            print(f"[shopify] Failed to parse JSON response: {e}")
            break

        products = data.get("products", [])
        if not products:
            break

        for product in products:
            chunk = _format_product(product, base)
            if chunk:
                chunks.append(chunk)

        # Cursor pagination via Shopify Link header
        url = _next_page_url(resp.headers.get("Link", ""))

    print(f"[shopify] Fetched {len(chunks)} products from {base}")
    return chunks


def _format_product(p: dict, base_url: str) -> str:
    """Format a single Shopify product dict as a RAG-friendly text chunk."""
    title = (p.get("title") or "").strip()
    if not title:
        return ""

    vendor = (p.get("vendor") or "").strip()
    product_type = (p.get("product_type") or "").strip()
    handle = (p.get("handle") or "").strip()
    tags_raw = p.get("tags") or []
    # Tags can be a list or a comma-separated string depending on API version
    if isinstance(tags_raw, str):
        tags = tags_raw
    else:
        tags = ", ".join(t.strip() for t in tags_raw if t.strip())

    product_url = f"{base_url}/products/{handle}" if handle else base_url

    # Price from first (cheapest) variant
    variants = p.get("variants") or []
    price = ""
    in_stock = None
    if variants:
        v = variants[0]
        raw_price = v.get("price") or ""
        try:
            price = f"{float(raw_price):.2f}" if raw_price else ""
        except (ValueError, TypeError):
            price = str(raw_price)
        available = v.get("available")
        if available is not None:
            in_stock = bool(available)

    # Description: strip HTML tags from body_html
    body_html = p.get("body_html") or ""
    description = _strip_html(body_html)
    if len(description) > 600:
        description = description[:600].rsplit(" ", 1)[0] + "…"

    lines = [f"Product: {title}"]
    if product_type:
        lines.append(f"Category: {product_type}")
    if vendor:
        lines.append(f"Brand: {vendor}")
    if price:
        lines.append(f"Price: {price}")
    if description:
        lines.append(f"Description: {description}")
    if tags:
        lines.append(f"Tags: {tags}")
    lines.append(f"URL: {product_url}")
    if in_stock is not None:
        lines.append(f"In stock: {'Yes' if in_stock else 'No'}")

    return "\n".join(lines)


def _strip_html(raw: str) -> str:
    """Remove HTML tags and normalise whitespace."""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _next_page_url(link_header: str) -> str:
    """
    Parse Shopify Link header for next-page cursor URL.
    Format: <https://store.myshopify.com/products.json?page_info=xxx&limit=250>; rel="next"
    Returns the URL string or empty string if no next page.
    """
    if not link_header:
        return ""
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else ""
