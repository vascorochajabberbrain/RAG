# Step-by-Step Implementation Plan: RAG Workflow & UX

This plan builds on the existing codebase. Each step is designed to be implementable and testable before moving on.

---

## Phase 1: Workflow spine (no UI change yet)

**Goal:** Define the RAG pipeline as explicit steps and call existing code from a single workflow runner.

### Step 1.1 – Define workflow steps and data model

- Create a `workflow/` package (or module) with:
  - **Steps enum or constants:** e.g. `CREATE_COLLECTION`, `ADD_SOURCE`, `FETCH`, `CLEAN`, `CHUNK`, `GROUP` (optional), `PUSH_TO_QDRANT`, `TEST_QA`.
  - **Workflow state:** A small dataclass or dict that holds: `collection_name`, `collection_type`, `source_type` (pdf | url | txt | csv), `source_config` (path, URL, or scraper name), `raw_text` (after fetch), `cleaned_text` (optional), `chunks` (list of strings), `grouped` (optional), and any flags (e.g. `grouping_enabled`).
- **Deliverable:** `workflow/models.py` (or `workflow/state.py`) with step list and state structure. No execution yet.

### Step 1.2 – Implement workflow runner that calls existing code

- Create `workflow/runner.py` (or `workflow/run.py`):
  - **Run from state:** Given current state and “next step”, execute one step:
    - **CREATE_COLLECTION:** Use `QdrantTracker.new(collection_name)` (and collection type); store collection name and type in state.
    - **ADD_SOURCE:** Only set in state: `source_type`, `source_config` (e.g. path, URL, or scraper id). No fetching yet.
    - **FETCH:** According to `source_type`, call existing ingestion:
      - `pdf` → `pdf_ingestion.read_data_from_pdf(path)` (you’ll need to pass path from config; see Step 2.2 for making PDF path configurable).
      - `url` → call a single function that runs the scraper (for now, can still call current `url_ingestion.main()` or a thin wrapper that takes start URL + scraper name; refactor scraping in Phase 3).
      - `txt` / `csv` → same idea, call existing ingestion and get raw text or list of chunks.
    - **CLEAN:** Optional. If you have a “clean” function (e.g. from url_ingestion filters), call it on `raw_text` and set `cleaned_text`; otherwise set `cleaned_text = raw_text`.
    - **CHUNK:** Call `vectorization.create_batches_of_text` + `get_text_chunks` (or a wrapper that takes state + chunking options). Set `state.chunks`.
    - **GROUP:** If enabled, call `grouping.grouping(scs_collection, group_collection)`; you’ll need to build SCS/group collections from state or from Qdrant (this may mean “push chunks to Qdrant as SCS, then run grouping into a new collection” — align with how you currently do SCS → Group).
    - **PUSH_TO_QDRANT:** Use existing `qdrant_utils` + collection logic to create points from `state.chunks` (and optional grouping) and upsert into the collection.
    - **TEST_QA:** Call `chatbot.get_retrieved_info` + `chatbot.get_answer` (or a single “ask one question” helper) for the current collection; optional: read one test question from state or CLI.
  - **Run full pipeline:** A function that runs steps in order (CREATE_COLLECTION → ADD_SOURCE → FETCH → CLEAN → CHUNK → [GROUP] → PUSH_TO_QDRANT → [TEST_QA]), with state passed between steps.
- **Deliverable:** Runner that executes each step using existing modules. Can be driven by a minimal CLI (e.g. “run step X” or “run all”) for testing.

### Step 1.3 – Make PDF (and txt) ingestion configurable

- In `ingestion/pdf_ingestion.py`: Replace hardcoded path with parameters (e.g. `pdf_path`, `pdf_source` or a single path). Keep `main()` as default for backward compatibility, but add e.g. `read_from_pdf(path) -> text` and `run_pdf_pipeline(path, batch_size, overlap) -> (chunks, source)` so the workflow can pass path and options.
- Do the same for `ingestion/txt_ingestion.py` if you use it (path/config as arguments).
- **Deliverable:** PDF (and optionally txt) ingestion callable with paths/options from the workflow state.

---

## Phase 2: Guided CLI (first user-facing UX)

**Goal:** A single entry point that walks the user through the workflow steps with prompts and defaults.

### Step 2.1 – Interactive workflow CLI

- Create `workflow/cli.py` (or `run_workflow.py` in project root):
  - **Menu:** “What do you want to do?” → (1) Create new RAG from scratch, (2) Add source to existing collection, (3) Run chunking/grouping only, (4) Test Q&A on collection, (5) Show workflow status / list collections.
  - For (1): Prompt for collection name, collection type (scs / group / …), then source type (pdf / url / txt / csv). Then prompt for path or URL or scraper name; store in state and run FETCH → CLEAN → CHUNK (with optional “chunking options” prompt: use defaults first), then “Run grouping? (y/n)”, then PUSH_TO_QDRANT, then optional “Ask a test question?”.
  - For (2): Select existing collection (from QdrantTracker), then same ADD_SOURCE → FETCH → … as above, and merge new chunks into existing collection (reuse existing “append” logic).
  - For (3): Select collection, optionally “re-chunk” with different options or run grouping only (load existing points, run grouping, save to same or new collection).
  - For (4): Select collection, input question, call chatbot and print answer.
  - Reuse `workflow/runner.py` for each step; state can be in-memory for the session.
- **Deliverable:** One script (e.g. `python -m workflow.cli` or `python run_workflow.py`) that implements this guided flow.

### Step 2.2 – Chunking options in the CLI

- Add to workflow state: `chunk_batch_size`, `chunk_overlap`, `use_proposition_chunking` (True = current LLM; False = simple CharacterTextSplitter or similar).
- In the “CHUNK” step, branch on `use_proposition_chunking` and pass batch_size/overlap to `create_batches_of_text` / chunking.
- In the CLI, after “FETCH” (or when “Run chunking?”), ask: “Use default chunking? (y/n)”. If n, prompt for batch size, overlap, and proposition vs simple (or “suggest” that calls an LLM with a short content summary and prints a recommendation).
- **Deliverable:** Chunking configurable from CLI with optional “suggest” for defaults.

---

## Phase 3: Configurable scraping (per-site, AI-friendly)

**Goal:** Move site-specific scraping out of one big file into per-site configs + a small runner, so new sites and “click to display” fixes are additive.

### Step 3.1 – Scraper config format and folder

- Create `ingestion/scrapers/` (or `scraper_configs/`):
  - **Config format:** One file per site (e.g. `peixefresco.yaml`, `heyharper.yaml`). Contents: `name`, `start_url`, `link_selectors` (e.g. only follow links matching a pattern), `click_before_scrape` (list of selectors or “accordion” / “show_more” with optional selector), `text_selector` (e.g. body or main), optional `filter_regex` or `filter_module` (to reuse existing filters from url_ingestion).
  - **Deliverable:** Config schema (YAML or JSON) and at least one example config that reproduces current Peixe Fresco or a subset of current behavior.

### Step 3.2 – Generic crawler that reads config

- Create `ingestion/scrapers/runner.py` (or `crawler.py`):
  - Load config by name (e.g. `peixefresco` → load `peixefresco.yaml`).
  - Use Selenium (or reuse current driver setup from url_ingestion):
    - Go to `start_url`.
    - For each `click_before_scrape`: find element (by selector or by “Show more” / “accordion” text), click, wait.
    - Get text from page (by `text_selector` or body).
    - Find links from `link_selectors`, push to stack; repeat until visited limit or stack empty.
  - Apply optional filters (from config: regex or call a small filter function keyed by config name).
  - Return raw text (and optionally list of per-page texts) for the workflow.
- **Deliverable:** One function `run_scraper(config_name, options?) -> str` (or `-> list[str]`) that the workflow FETCH step can call for `source_type=url`.

### Step 3.3 – Migrate one existing site into config

- Take current Peixe Fresco (or Hey Harper) logic from `url_ingestion.py`: extract start URL, “open all toggles”, link-follow rule, and filters into a config + minimal code (e.g. “accordion” → reuse `open_all_toggles`; “show_more” → reuse `click_show_more`). Wire `run_scraper("peixefresco")` to this.
- Keep `url_ingestion.main()` working by having it call `run_scraper("peixefresco")` (or the one you migrated) plus the same chunking as now, so existing flows don’t break.
- **Deliverable:** One site fully driven by config; `url_ingestion.py` delegates to scraper runner for that site.

### Step 3.4 – Document “add new site / fix scraping” with AI

- Add a short doc (e.g. `ingestion/scrapers/README.md`): “To add a new site: copy an existing YAML, change name/start_url/selectors/click_before_scrape. To handle a new button: add a new click step or selector. When stuck, paste this config and describe the page; the AI can suggest selector changes or a small code snippet.”
- **Deliverable:** README so you (and the AI) can add or fix sites by editing config + small snippets.

---

## Phase 4: Optional web UI

**Goal:** Same workflow, but driven from a browser (forms + run step + show result).

### Step 4.1 – API layer for the workflow

- Create a small FastAPI (or Flask) app, e.g. `web/app.py`:
  - **Endpoints:** `GET /collections`, `POST /workflow/step` (body: current state + step name), `GET /workflow/state` (if you persist state in session or DB), `POST /qa` (collection_name, question) for test Q&A.
  - `POST /workflow/step` calls the same `workflow/runner` functions as the CLI; returns updated state + success/error message.
- **Deliverable:** API that the CLI logic uses under the hood (or refactor CLI to call the same “run step” function the API calls).

### Step 4.2 – Minimal front end

- Single-page or few-page UI: “Create RAG” wizard (steps: name → source type → config → run fetch → run chunk → run group? → push → test question). Buttons: “Run step”, “Next”, “Back”. Display: current state (e.g. “Fetched 5000 chars”, “Chunked into 42 chunks”), and last error if any.
- **Deliverable:** Simple UI that drives the workflow via the API. No need for auth or multi-user in the first version.

---

## Phase 5: Polish and “suggest” integration

**Goal:** Make chunking/grouping and scraping decisions easier with AI.

### Step 5.1 – Chunking “suggest” in workflow

- In workflow, before CHUNK: optional step “suggest_chunking”: send a short summary of the content (e.g. first 500 chars + “pdf” or “scraped recipe site”) to an LLM with a fixed prompt: “Recommend batch_size, overlap, and whether to use proposition-based chunking for RAG. Reply in JSON: batch_size, overlap, use_proposition.” Parse and prefill state; user can override in CLI or web UI.
- **Deliverable:** One suggest endpoint or CLI option that sets chunking parameters from LLM output.

### Step 5.2 – Scraper “suggest” (optional)

- When a scrape fails or returns too little text: “Describe what you see (e.g. ‘there’s a Show more button’).” Send config + description to LLM; ask for a suggested YAML diff or a small Python snippet (e.g. new click step). You apply manually and re-run. Can be documented in scraper README and later turned into a “suggest” button that outputs suggested config changes.
- **Deliverable:** Clear process (and optionally one script) for “scrape failed → describe → get suggested config/code → apply and re-run”.

---

## Order summary

| Order | Step | Phase |
|-------|------|--------|
| 1 | Workflow steps + state model | 1.1 |
| 2 | Workflow runner calling existing code | 1.2 |
| 3 | Configurable PDF/txt ingestion | 1.3 |
| 4 | Guided CLI for full workflow | 2.1 |
| 5 | Chunking options + suggest in CLI | 2.2 |
| 6 | Scraper config format + folder | 3.1 |
| 7 | Generic crawler from config | 3.2 |
| 8 | Migrate one site to config | 3.3 |
| 9 | Scraper README for new sites / AI | 3.4 |
| 10 | Optional: API layer | 4.1 |
| 11 | Optional: Minimal web UI | 4.2 |
| 12 | Optional: Chunking suggest automation | 5.1 |
| 13 | Optional: Scraper suggest process | 5.2 |

Implement in this order; Phases 4 and 5 can be done later or in parallel once 1–3 are stable.
