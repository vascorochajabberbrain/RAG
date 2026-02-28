"""
Run a scraper by name. Loads YAML config and dispatches to the correct engine.

Engines:
  playwright  — default. Handles JS, SPAs, dynamic content.
  httpx       — fast, for confirmed SSR-only sites.
  shopify     — Shopify JSON API. Public /products.json (no auth) or Admin REST API (with access token).

Legacy scrapers (legacy: true in YAML) → url_ingestion_legacy.py (archived Selenium code).
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_config(scraper_name: str, inline_config: dict = None) -> dict:
    """
    Load scraper config. Priority:
    1. Named YAML file in configs/ (for customized scrapers)
    2. inline_config dict (stored in solutions.yaml scraper_config field)
    3. Bare dict with just the name
    """
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    yaml_path = os.path.join(config_dir, f"{scraper_name}.yaml")
    if os.path.isfile(yaml_path):
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    if inline_config:
        return dict(inline_config)
    return {"name": scraper_name}


def run_scraper(scraper_name: str, options: dict = None, inline_config: dict = None) -> tuple:
    """
    Run a scraper by name. Returns (raw_text: str, scraped_items: list[dict]).
    scraped_items is a list of {"url": str, "text": str} — one entry per scraped page.

    options: optional overrides (engine, start_url, etc.) from the workflow/UI.
             options take priority over YAML config values.
    inline_config: fallback config dict from solutions.yaml scraper_config field.
                   Used when no named YAML file exists. Named YAML takes precedence.
    """
    options = options or {}
    # Extract inline_config from options if passed that way (from workflow/runner.py)
    _inline = inline_config or options.pop("inline_config", None)
    config = _load_config(scraper_name, inline_config=_inline)
    config.update(options)  # UI/workflow overrides win over YAML/inline defaults

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
        return run_shopify_scraper(config)

    return f"Error: Unknown engine '{engine}' in config for scraper '{scraper_name}'.", []
