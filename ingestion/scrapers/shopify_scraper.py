"""
Shopify JSON API scraper.
Uses the public /products.json endpoint — no HTML scraping needed.

YAML config keys used:
  engine: shopify
  shop_url: str          e.g. "https://mystore.myshopify.com"
  chunk_template: str    optional override for the default product text format
"""

import httpx
from bs4 import BeautifulSoup


_DEFAULT_TEMPLATE = (
    "Product: {title}\n"
    "Vendor: {vendor}\n"
    "Type: {product_type}\n"
    "Price: {price}\n"
    "Description: {description}\n"
    "URL: {url}"
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)"
}


# ── Public entry point ────────────────────────────────────────────────────────

def run_shopify_scraper(config: dict) -> str:
    """
    Fetch all products from a Shopify store via /products.json.
    Returns one structured text chunk per product, joined with '\\n\\n---\\n\\n'.
    """
    shop_url = config.get("shop_url", "").rstrip("/")
    if not shop_url:
        return "Error: shop_url required for Shopify scraper."

    template = config.get("chunk_template", _DEFAULT_TEMPLATE)

    all_products = _fetch_all_products(shop_url)
    if not all_products:
        return f"Error: No products found at {shop_url}/products.json."

    print(f"[shopify_scraper] Fetched {len(all_products)} products from {shop_url}.")

    chunks = []
    for product in all_products:
        chunk = _render_product(product, shop_url, template)
        if chunk:
            chunks.append(chunk)

    return "\n\n---\n\n".join(chunks)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_all_products(shop_url: str) -> list:
    """Paginate through /products.json and return all products."""
    all_products = []
    page = 1

    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=30) as client:
        while True:
            url = f"{shop_url}/products.json?limit=250&page={page}"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"[shopify_scraper] ERROR fetching page {page}: {e}")
                break

            products = data.get("products", [])
            if not products:
                break

            all_products.extend(products)
            print(f"[shopify_scraper] Page {page}: {len(products)} products (total so far: {len(all_products)})")
            page += 1

    return all_products


def _render_product(product: dict, shop_url: str, template: str) -> str:
    """Render a single Shopify product dict into a text chunk."""
    title = product.get("title", "")
    vendor = product.get("vendor", "")
    product_type = product.get("product_type", "")
    handle = product.get("handle", "")
    body_html = product.get("body_html", "") or ""
    url = f"{shop_url}/products/{handle}"

    # Extract price from first variant
    variants = product.get("variants", [])
    price = variants[0].get("price", "N/A") if variants else "N/A"

    # Strip HTML from description
    description = BeautifulSoup(body_html, "html.parser").get_text(separator=" ", strip=True)

    fields = {
        "title": title,
        "vendor": vendor,
        "product_type": product_type,
        "price": price,
        "description": description,
        "url": url,
        "handle": handle,
    }

    try:
        return template.format_map(fields).strip()
    except KeyError as e:
        print(f"[shopify_scraper] WARNING: chunk_template missing key {e}.")
        return f"Product: {title}\nURL: {url}"
