"""
Run a scraper by name. Supports legacy scrapers (e.g. peixefresco) and future YAML-driven scrapers.
"""
import os
import sys

# Project root for imports
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


def run_scraper(scraper_name: str, options: dict = None) -> str:
    """
    Run a scraper by name. Returns raw text.
    options: optional overrides (start_url, output_file, etc.) and source_config from workflow.
    """
    options = options or {}
    config = _load_config(scraper_name)
    config.update(options)

    if config.get("legacy"):
        if scraper_name == "peixefresco":
            from ingestion.url_ingestion import run_peixefresco_crawl  # noqa: E402
            return run_peixefresco_crawl(config)
        return f"Error: Unknown legacy scraper '{scraper_name}'."

    # Future: generic YAML-driven Selenium crawl
    return _run_generic_crawl(scraper_name, config)


def _run_generic_crawl(scraper_name: str, config: dict) -> str:
    """Generic Selenium crawl from config. Not fully implemented yet."""
    start_url = config.get("start_url")
    if not start_url:
        return "Error: config must have start_url for generic crawl."
    # Placeholder: could implement Selenium from config (link_selectors, click_before_scrape, etc.)
    return "Error: Generic YAML crawl not implemented yet. Use legacy scrapers (e.g. peixefresco)."
