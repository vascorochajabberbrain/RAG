# Scraper configs and runner

## Adding a new site

1. Copy an existing config: `configs/peixefresco.yaml` → `configs/mysite.yaml`.
2. Edit `mysite.yaml`:
   - `name`: short id (e.g. `mysite`)
   - `start_url`: first page to crawl
   - `link_prefix`: only follow links starting with this (optional)
   - `click_before_scrape`: `accordion` (click elements with `aria-label="accordion"`) or `show_more` (click "Show more" buttons), or leave unset
   - `legacy: true`: use a custom Python scraper (see below) instead of the generic crawler

3. If the site needs custom logic (e.g. specific selectors, login, anti-bot), set `legacy: true` and add a branch in `ingestion/scrapers/runner.py` that calls a function in `ingestion/url_ingestion.py` (e.g. `run_mysite_crawl(options)`).

## When a scrape fails or content is missing

- **New button or “click to display”:** Describe what you see (e.g. “There’s a ‘Show more’ button that loads recipes”). You can paste the current config and this description to an AI assistant; it can suggest:
  - Changes to the YAML (e.g. a new `click_before_scrape` step), or
  - A short Selenium snippet (e.g. find and click the button, then get body text).
- After applying the suggestion, run the scraper again.

## Filters

The workflow CLEAN step can apply a named filter to scraped text (e.g. remove product listings, footer). Filters are in `ingestion/scrapers/filters.py`. To add one, implement a function that takes `(text: str) -> str` and register it in `apply_filter()` by name.
