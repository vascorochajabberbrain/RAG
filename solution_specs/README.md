# Solution specifications

This folder holds **solution-specific configuration** only: collection names, company/assistant names, scraper references, and optional defaults. Code stays generic and reads from here.

**Adding a new solution:** copy the block from `solutions_template.yaml` into `solutions.yaml`, then fill in the fields.  
**Running a process:** point the workflow or chatbot to one solution (by `id` or `alias`).

**Later:** the same structure can be loaded from a database instead of YAML; only the loader needs to change.

## File layout

- `solutions.yaml` – list of solutions (one document per RAG “solution” / bot / collection set).
- `solutions_template.yaml` – template to copy when adding a new solution.
- `loader.py` – loads specs and exposes `get_solution(id)`, `list_solutions()`, `resolve_alias(alias)`.

## Schema (per solution)

| Field | Description |
|-------|--------------|
| `id` | Unique key (e.g. `hey_harper_1`, `peixefresco`). |
| `display_name` | Human name (e.g. "Hey Harper", "Peixe Fresco Recipes"). |
| `collection_name` | Qdrant collection name. |
| `company_name` | Name used in chatbot system prompt ("virtual assistant of …"). |
| `collection_type` | `scs` or `group`. |
| `scraper_name` | Optional; default scraper for URL ingestion (e.g. `peixefresco`). |
| `chunking_defaults` | Optional `batch_size`, `overlap_size`, `use_proposition_chunking`. |
| `aliases` | Optional list of shortcuts (e.g. `["1", "FAQ"]`) for CLI/chatbot selection. |

Add or edit entries in `solutions.yaml`; no code change needed for new solutions.
