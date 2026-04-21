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


def run_scraper(scraper_name: str, options: dict = None, inline_config: dict = None, cancel_check=None) -> tuple:
    """
    Run a scraper by name. Returns (raw_text: str, scraped_items: list[dict]).
    scraped_items is a list of {"url": str, "text": str} — one entry per scraped page.

    options: optional overrides (engine, start_url, etc.) from the workflow/UI.
             options take priority over YAML config values.
    inline_config: fallback config dict from solutions.yaml scraper_config field.
                   Used when no named YAML file exists. Named YAML takes precedence.
    cancel_check: optional callable returning True when the user wants to stop.
    """
    options = options or {}
    # Extract inline_config from options if passed that way (from workflow/runner.py)
    _inline = inline_config or options.pop("inline_config", None)
    config = _load_config(scraper_name, inline_config=_inline)

    # Library extractor merge: if config references a library extractor, merge them
    library_name = config.get("library_extractor")
    if library_name:
        lib_config = _load_library_extractor(library_name)
        if lib_config:
            config = _merge_library_config(config, lib_config)

    config.update(options)  # UI/workflow overrides win over YAML/inline defaults

    # Legacy branch — archived Selenium scrapers (no per-URL metadata)
    if config.get("legacy"):
        if scraper_name == "peixefresco":
            from ingestion.url_ingestion_legacy import run_peixefresco_crawl
            return run_peixefresco_crawl(config), []
        return f"Error: Unknown legacy scraper '{scraper_name}'.", []

    # Engine dispatch — Playwright is the default
    engine = config.get("engine", "playwright")
    mode = detect_extraction_mode(config)
    lib_tag = f" (library: {library_name})" if library_name else ""
    print(f"[scraper] Engine: {engine} | Extraction: {mode}{lib_tag} | Config: {scraper_name}")

    if engine == "playwright":
        from ingestion.scrapers.playwright_scraper import run_playwright_scraper
        return run_playwright_scraper(config, cancel_check=cancel_check)

    if engine == "httpx":
        from ingestion.scrapers.httpx_scraper import run_httpx_scraper
        return run_httpx_scraper(config, cancel_check=cancel_check)

    if engine == "shopify":
        from ingestion.scrapers.shopify_scraper import run_shopify_scraper
        return run_shopify_scraper(config)

    return f"Error: Unknown engine '{engine}' in config for scraper '{scraper_name}'.", []


def detect_extraction_mode(config: dict) -> str:
    """Determine extraction mode from config fields.

    Returns one of: 'generic', 'structured', 'custom_js', 'shopify'.
    """
    engine = config.get("engine", "playwright")
    if engine == "shopify":
        return "shopify"
    if config.get("custom_js_extraction"):
        return "custom_js"
    if config.get("structured_extraction"):
        return "structured"
    return "generic"


def _load_library_extractor(name: str) -> dict | None:
    """Load a library extractor YAML from configs/library/{name}.yaml."""
    lib_dir = os.path.join(os.path.dirname(__file__), "configs", "library")
    yaml_path = os.path.join(lib_dir, f"{name}.yaml")
    if not os.path.isfile(yaml_path):
        return None
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _merge_library_config(site_config: dict, library_config: dict) -> dict:
    """Merge library extractor (HOW) with site config (WHERE).

    Library provides: custom_js_extraction, chunk_template, description, engine (default).
    Site overrides: engine, sitemap_url, url_filter, url_allowlist, scrape_mode, name, etc.
    """
    merged = dict(library_config)
    # Remove library-only markers from merged result
    merged.pop("library", None)
    # Site config fields override library defaults
    for key, value in site_config.items():
        if key == "library_extractor":
            continue  # keep as metadata, don't overwrite
        if value is not None and value != "":
            merged[key] = value
    # Preserve the library_extractor reference
    merged["library_extractor"] = site_config.get("library_extractor", "")
    return merged


def resolve_config(scraper_name: str, inline_config: dict = None) -> dict:
    """Resolve the effective scraper config and return it with metadata.

    Returns:
        {
            "config": { ...full resolved config... },
            "source": "yaml_file" | "inline" | "default",
            "yaml_file_exists": bool,
            "extraction_mode": "generic" | "structured" | "custom_js" | "shopify",
            "description": str,
            "library_extractor": str | None  -- name of linked library extractor
        }
    """
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    yaml_path = os.path.join(config_dir, f"{scraper_name}.yaml")
    yaml_exists = os.path.isfile(yaml_path)

    config = _load_config(scraper_name, inline_config=inline_config)
    source = "yaml_file" if yaml_exists else ("inline" if inline_config else "default")

    # Library extractor merge: if config references a library extractor, merge them
    library_name = config.get("library_extractor")
    if library_name:
        lib_config = _load_library_extractor(library_name)
        if lib_config:
            config = _merge_library_config(config, lib_config)

    mode = detect_extraction_mode(config)

    # Description: from config 'description' field, or built-in default per mode
    description = config.get("description", "")
    if not description:
        description = _BUILTIN_MODE_DESCRIPTIONS.get(mode, "")

    return {
        "config": config,
        "source": source,
        "yaml_file_exists": yaml_exists,
        "extraction_mode": mode,
        "description": description,
        "library_extractor": library_name,
    }


# Built-in descriptions for modes that don't have a YAML config file
_BUILTIN_MODE_DESCRIPTIONS = {
    "generic": (
        "Grabs all visible text from the page. Good default for most sites. "
        "Content goes through the normal chunking step for splitting. "
        "Works with any engine (Playwright, httpx). "
        "No field extraction — just raw text from a CSS container or full page."
    ),
    "structured": (
        "Uses CSS selectors to extract specific named fields (e.g. name, price, description) "
        "from each page into a chunk template. Good for e-commerce product pages and other "
        "sites with consistent HTML structure. Selectors are site-specific."
    ),
    "custom_js": (
        "Runs a JavaScript function in the browser that returns a structured object with "
        "named fields. Needed for complex layouts like Elementor where CSS selectors alone "
        "can't navigate between widget containers. Requires Playwright engine."
    ),
    "shopify": (
        "Fetches product data via Shopify's JSON API (/products.json). No HTML scraping "
        "needed — structured product data comes directly from the API. "
        "Only works for Shopify stores."
    ),
}


def list_library_extractors() -> list[dict]:
    """List all library extractors from configs/library/.

    Returns list of {name, description, extraction_mode} dicts.
    """
    import yaml
    lib_dir = os.path.join(os.path.dirname(__file__), "configs", "library")
    if not os.path.isdir(lib_dir):
        return []
    extractors = []
    for fname in sorted(os.listdir(lib_dir)):
        if not fname.endswith(".yaml"):
            continue
        try:
            with open(os.path.join(lib_dir, fname), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if not cfg.get("library"):
                continue
            name = fname.removesuffix(".yaml")
            extractors.append({
                "name": name,
                "description": cfg.get("description", ""),
                "extraction_mode": detect_extraction_mode(cfg),
            })
        except Exception:
            continue
    return extractors
