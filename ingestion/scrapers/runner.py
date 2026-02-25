"""
Run a scraper by name. Loads YAML config and dispatches to the correct engine.

Engines:
  playwright  — default. Handles JS, SPAs, dynamic content.
  httpx       — fast, for confirmed SSR-only sites.
  shopify     — uses Shopify /products.json API (no HTML scraping).

Legacy scrapers (legacy: true in YAML) → url_ingestion_legacy.py (archived Selenium code).
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_config(scraper_name: str) -> dict:
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    yaml_path = os.path.join(config_dir, f"{scraper_name}.yaml")
    if not os.path.isfile(yaml_path):
        return {"name": scraper_name}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {"name": scraper_name}


def run_scraper(scraper_name: str, options: dict = None) -> tuple:
    """
    Run a scraper by name. Returns (raw_text: str, scraped_items: list[dict]).
    scraped_items is a list of {"url": str, "text": str} — one entry per scraped page.
    options: optional overrides (engine, start_url, etc.) from the workflow/UI.
             options take priority over YAML config values.
    """
    options = options or {}
    config = _load_config(scraper_name)
    config.update(options)  # UI/workflow overrides win over YAML defaults

    # Legacy branch — archived Selenium scrapers (no per-URL metadata)
    if config.get("legacy"):
        if scraper_name == "peixefresco":
            from ingestion.url_ingestion_legacy import run_peixefresco_crawl
            return run_peixefresco_crawl(config), []
        return f"Error: Unknown legacy scraper '{scraper_name}'.", []

    # Engine dispatch — Playwright is the default
    engine = config.get("engine", "playwright")

    if engine == "playwright":
        from ingestion.scrapers.playwright_scraper import run_playwright_scraper
        return run_playwright_scraper(config)

    if engine == "httpx":
        from ingestion.scrapers.httpx_scraper import run_httpx_scraper
        return run_httpx_scraper(config)

    if engine == "shopify":
        from ingestion.scrapers.shopify_scraper import run_shopify_scraper
        # Shopify scraper returns a plain string; no per-URL metadata yet
        return run_shopify_scraper(config), []

    return f"Error: Unknown engine '{engine}' in config for scraper '{scraper_name}'.", []
