"""
Shopify JSON API scraper.

Two modes depending on whether an access token is available:

  Public (no auth):
    Uses the public /products.json endpoint — no credentials needed.
    Works on any Shopify store with public product listings.

  Admin API (with access token):
    Uses the Shopify Admin REST API (version 2024-01) to fetch:
      - Products (+ optional metafields per product)
      - Pages (FAQ, About, policies — not available publicly)
      - Blog articles (guides, recipes, advice content)
    Excluded: customers, orders, inventory (private/sensitive or stale).

YAML config keys:
  engine: shopify
  shop_url: str            e.g. "https://mystore.myshopify.com"
  access_token: str        optional — Admin API token (shpat_...)
                           OR set env var SHOPIFY_ACCESS_TOKEN_{SLUG}
                           where SLUG = hostname stripped of non-alphanumeric, uppercased
                           e.g. mystore.myshopify.com → SHOPIFY_ACCESS_TOKEN_MYSTOREMYSHOPIFYCOM
  include: list            optional — which resource types to fetch (default: all three)
                           valid values: products, pages, articles
  metafields: bool         optional — fetch metafields per product (default: false)
                           adds one extra API call per product; can be slow on large stores
  chunk_template: str      optional — override default product text format
                           available fields: {title} {vendor} {product_type} {price}
                                            {description} {metafields} {url} {handle}
  page_template: str       optional — override default page text format
                           available fields: {title} {content} {url} {handle}
  article_template: str    optional — override default article text format
                           available fields: {blog_title} {title} {content} {url} {handle}
"""

import os
import re
import time

import httpx
from bs4 import BeautifulSoup


# ── Constants ─────────────────────────────────────────────────────────────────

_API_VERSION = "2024-01"

_DEFAULT_TEMPLATE = (
    "Product: {title}\n"
    "Vendor: {vendor}\n"
    "Type: {product_type}\n"
    "Price: {price}\n"
    "Description: {description}\n"
    "URL: {url}"
)

_DEFAULT_PAGE_TEMPLATE = (
    "Title: {title}\n"
    "Content: {content}\n"
    "URL: {url}"
)

_DEFAULT_ARTICLE_TEMPLATE = (
    "Blog: {blog_title}\n"
    "Title: {title}\n"
    "Content: {content}\n"
    "URL: {url}"
)

_PUBLIC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)"
}


# ── Public entry point ────────────────────────────────────────────────────────

def run_shopify_scraper(config: dict) -> tuple:
    """
    Fetch content from a Shopify store.

    Returns (raw_text: str, scraped_items: list[dict])
    where each item is {"url": str, "text": str} — one entry per product/page/article.

    Dispatches to Admin API path if an access token is found, otherwise
    falls back to the public /products.json endpoint.
    """
    shop_url = config.get("shop_url", "").rstrip("/")
    if not shop_url:
        return "Error: shop_url required for Shopify scraper.", []

    access_token = _resolve_access_token(config)

    if access_token:
        return _run_admin_scraper(shop_url, access_token, config)
    else:
        return _run_public_scraper(shop_url, config)


# ── Token resolution ──────────────────────────────────────────────────────────

def _resolve_access_token(config: dict) -> str:
    """
    Resolve the Shopify Admin API access token.

    Priority:
      1. 'access_token' key in YAML config
      2. Env var SHOPIFY_ACCESS_TOKEN_{SLUG}
         SLUG = hostname with non-alphanumeric characters stripped, uppercased
         e.g. "mystore.myshopify.com" → SHOPIFY_ACCESS_TOKEN_MYSTOREMYSHOPIFYCOM

    Returns empty string if neither is present (→ public path).
    """
    token = config.get("access_token", "")
    if token:
        return token

    shop_url = config.get("shop_url", "")
    if shop_url:
        hostname = re.sub(r"https?://", "", shop_url).rstrip("/").split("/")[0]
        slug = re.sub(r"[^a-z0-9]", "", hostname.lower()).upper()
        env_key = f"SHOPIFY_ACCESS_TOKEN_{slug}"
        return os.getenv(env_key, "")

    return ""


# ── Public path (no auth) ─────────────────────────────────────────────────────

def _run_public_scraper(shop_url: str, config: dict) -> tuple:
    """
    Fetch all products via the public /products.json endpoint (no credentials).
    Returns (raw_text, scraped_items) — 1 item per product.
    """
    include = config.get("include", ["products", "pages", "articles"])
    skipped = [r for r in include if r != "products"]
    if skipped:
        print(
            f"[shopify_scraper] Public mode: skipping {', '.join(skipped)} "
            f"(requires Admin API access token — only products are available publicly)."
        )

    template = config.get("chunk_template", _DEFAULT_TEMPLATE)
    all_products = _fetch_all_products_public(shop_url)

    if not all_products:
        return f"Error: No products found at {shop_url}/products.json.", []

    print(f"[shopify_scraper] Public mode: {len(all_products)} products fetched from {shop_url}.")

    items = []
    for product in all_products:
        text = _render_product(product, shop_url, template)
        if text:
            handle = product.get("handle", "")
            url = f"{shop_url}/products/{handle}"
            items.append({"url": url, "text": text})

    raw_text = "\n\n---\n\n".join(item["text"] for item in items)
    return raw_text, items


def _fetch_all_products_public(shop_url: str) -> list:
    """Paginate through /products.json using page numbers. Returns all products."""
    all_products = []
    page = 1

    with httpx.Client(headers=_PUBLIC_HEADERS, follow_redirects=True, timeout=30) as client:
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
            print(
                f"[shopify_scraper] Page {page}: {len(products)} products "
                f"(total so far: {len(all_products)})"
            )
            page += 1

    return all_products


# ── Admin API path ────────────────────────────────────────────────────────────

def _run_admin_scraper(shop_url: str, access_token: str, config: dict) -> tuple:
    """
    Fetch content via the Shopify Admin REST API.
    Returns (raw_text, scraped_items) — 1 item per product/page/article.
    """
    include = config.get("include", ["products", "pages", "articles"])
    fetch_metafields = config.get("metafields", False)
    template_product = config.get("chunk_template", _DEFAULT_TEMPLATE)
    template_page = config.get("page_template", _DEFAULT_PAGE_TEMPLATE)
    template_article = config.get("article_template", _DEFAULT_ARTICLE_TEMPLATE)

    headers = {
        "X-Shopify-Access-Token": access_token,
        "User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)",
    }

    items = []

    with httpx.Client(headers=headers, follow_redirects=True, timeout=30) as client:
        if "products" in include:
            product_items = _fetch_admin_products(
                client, shop_url, template_product, fetch_metafields
            )
            items.extend(product_items)

        if "pages" in include:
            page_items = _fetch_admin_pages(client, shop_url, template_page)
            items.extend(page_items)

        if "articles" in include:
            article_items = _fetch_admin_articles(client, shop_url, template_article)
            items.extend(article_items)

    raw_text = "\n\n---\n\n".join(item["text"] for item in items)
    print(f"[shopify_scraper] Admin mode: {len(items)} total items from {shop_url}.")
    return raw_text, items


def _admin_paginate(client: httpx.Client, url: str) -> list:
    """
    Fetch all pages of a Shopify Admin REST endpoint using cursor-based pagination.

    Shopify Admin API returns a Link header with rel="next" for the next page cursor.
    Sleeps 0.5s between requests to respect the leaky-bucket rate limit (2 req/s).

    Returns a flat list of all items across all pages.
    """
    results = []
    next_url = url

    while next_url:
        try:
            resp = client.get(next_url)
            resp.raise_for_status()
        except Exception as e:
            print(f"[shopify_scraper] ERROR fetching {next_url}: {e}")
            break

        data = resp.json()
        # The top-level key varies by resource type ("products", "pages", "articles", etc.)
        for key, val in data.items():
            if isinstance(val, list):
                results.extend(val)
                break

        next_url = _parse_next_link(resp.headers.get("Link", ""))
        time.sleep(0.5)

    return results


def _parse_next_link(link_header: str) -> str:
    """
    Parse Shopify's Link response header and return the 'next' URL, or ''.

    Format: '<https://...?page_info=XYZ>; rel="next", <...>; rel="previous"'
    """
    if not link_header:
        return ""
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part:
            match = re.search(r"<([^>]+)>", part)
            if match:
                return match.group(1)
    return ""


def _fetch_admin_products(
    client: httpx.Client,
    shop_url: str,
    template: str,
    fetch_metafields: bool,
) -> list:
    """
    Fetch all products via Admin API.
    Optionally fetch metafields per product (1 extra API call each).
    Returns list of {"url": str, "text": str}.
    """
    base_url = f"{shop_url}/admin/api/{_API_VERSION}/products.json?limit=250"
    all_products = _admin_paginate(client, base_url)

    print(f"[shopify_scraper] Admin products: {len(all_products)} fetched.")

    items = []
    for product in all_products:
        metafields_text = ""
        if fetch_metafields:
            product_id = product.get("id")
            mf_url = f"{shop_url}/admin/api/{_API_VERSION}/products/{product_id}/metafields.json"
            try:
                resp = client.get(mf_url)
                resp.raise_for_status()
                mfs = resp.json().get("metafields", [])
                if mfs:
                    lines = [
                        f"{mf.get('namespace')}.{mf.get('key')}: {mf.get('value')}"
                        for mf in mfs
                    ]
                    metafields_text = "\n".join(lines)
            except Exception as e:
                print(
                    f"[shopify_scraper] WARNING: metafields fetch failed "
                    f"for product {product_id}: {e}"
                )
            time.sleep(0.5)  # rate limit between metafield calls

        handle = product.get("handle", "")
        url = f"{shop_url}/products/{handle}"
        text = _render_product(product, shop_url, template, metafields_text)
        if text:
            items.append({"url": url, "text": text})

    return items


def _fetch_admin_pages(
    client: httpx.Client,
    shop_url: str,
    template: str,
) -> list:
    """
    Fetch all static pages via Admin API (FAQ, About, policies, etc.).
    Returns list of {"url": str, "text": str}.
    """
    base_url = f"{shop_url}/admin/api/{_API_VERSION}/pages.json?limit=250"
    all_pages = _admin_paginate(client, base_url)

    print(f"[shopify_scraper] Admin pages: {len(all_pages)} fetched.")

    items = []
    for page in all_pages:
        title = page.get("title", "")
        handle = page.get("handle", "")
        body_html = page.get("body_html", "") or ""
        content = BeautifulSoup(body_html, "html.parser").get_text(separator=" ", strip=True)
        url = f"{shop_url}/pages/{handle}"

        fields = {"title": title, "content": content, "url": url, "handle": handle}
        try:
            text = template.format_map(fields).strip()
        except KeyError as e:
            print(f"[shopify_scraper] WARNING: page_template missing key {e}.")
            text = f"Title: {title}\nURL: {url}"

        if text:
            items.append({"url": url, "text": text})

    return items


def _fetch_admin_articles(
    client: httpx.Client,
    shop_url: str,
    template: str,
) -> list:
    """
    Fetch all blog articles via Admin API.
    First fetches the list of blogs, then fetches articles for each blog.
    Returns list of {"url": str, "text": str}.
    """
    blogs_url = f"{shop_url}/admin/api/{_API_VERSION}/blogs.json"
    try:
        resp = client.get(blogs_url)
        resp.raise_for_status()
        blogs = resp.json().get("blogs", [])
    except Exception as e:
        print(f"[shopify_scraper] ERROR fetching blogs list: {e}")
        return []

    print(f"[shopify_scraper] Admin blogs: {len(blogs)} found.")
    items = []

    for blog in blogs:
        blog_id = blog.get("id")
        blog_handle = blog.get("handle", str(blog_id))
        blog_title = blog.get("title", "")
        articles_url = (
            f"{shop_url}/admin/api/{_API_VERSION}/blogs/{blog_id}/articles.json?limit=250"
        )
        all_articles = _admin_paginate(client, articles_url)
        print(f"[shopify_scraper]   Blog '{blog_title}': {len(all_articles)} articles.")

        for article in all_articles:
            title = article.get("title", "")
            handle = article.get("handle", "")
            body_html = article.get("body_html", "") or ""
            content = BeautifulSoup(body_html, "html.parser").get_text(separator=" ", strip=True)
            url = f"{shop_url}/blogs/{blog_handle}/{handle}"

            fields = {
                "blog_title": blog_title,
                "title": title,
                "content": content,
                "url": url,
                "handle": handle,
            }
            try:
                text = template.format_map(fields).strip()
            except KeyError as e:
                print(f"[shopify_scraper] WARNING: article_template missing key {e}.")
                text = f"Blog: {blog_title}\nTitle: {title}\nURL: {url}"

            if text:
                items.append({"url": url, "text": text})

    return items


# ── Shared rendering helper ───────────────────────────────────────────────────

def _render_product(
    product: dict,
    shop_url: str,
    template: str,
    metafields_text: str = "",
) -> str:
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
        "metafields": metafields_text,  # empty string if not fetched; templates can ignore it
    }

    try:
        return template.format_map(fields).strip()
    except KeyError as e:
        print(f"[shopify_scraper] WARNING: chunk_template missing key {e}.")
        return f"Product: {title}\nURL: {url}"
