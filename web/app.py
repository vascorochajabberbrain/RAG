"""
Minimal FastAPI app for the RAG workflow. Run with: uvicorn web.app:app --reload
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Any
import os, queue, threading, asyncio, json

# App version (read from VERSION file in project root)
def _read_version() -> str:
    try:
        _vf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION")
        return open(_vf).read().strip()
    except Exception:
        return "dev"

APP_VERSION = _read_version()
IS_DEV_MODE = os.environ.get("RAG_DEV_MODE", "") == "1"

# In-memory state for single-user demo (one workflow at a time)
_current_state: Optional[Any] = None
_tracker: Optional[Any] = None

# Progress reporting (used by long-running steps like translate)
# thread-safe queue written to by background thread, read by async SSE endpoint
_progress_queue: queue.Queue = queue.Queue()
_loop: asyncio.AbstractEventLoop = None


def get_state():
    global _current_state, _tracker
    if _tracker is None:
        from QdrantTracker import QdrantTracker
        _tracker = QdrantTracker()
    if _current_state is None:
        from workflow.models import WorkflowState
        _current_state = WorkflowState(tracker=_tracker)
    return _current_state


def reset_state():
    global _current_state
    _current_state = None


app = FastAPI(title="RAG Workflow API")

# Serve PDF files so the browser can open them with #page=N fragment
_PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ingestion", "data_to_ingest", "pdfs")
if os.path.isdir(_PDF_DIR):
    app.mount("/pdfs", StaticFiles(directory=_PDF_DIR), name="pdfs")

# Serve assets (favicon, icons, etc.)
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
if os.path.isdir(_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")


@app.on_event("startup")
async def _capture_loop():
    global _loop
    _loop = asyncio.get_running_loop()


class StepRequest(BaseModel):
    step: str
    state_update: Optional[dict] = None
    login_config: Optional[dict] = None


class QARequest(BaseModel):
    collection_name: str  # single name, "__all__", or comma-separated list
    question: str
    company: str = "Assistant"
    solution_id: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"  # must match model used at ingestion time


@app.get("/")
def root():
    html = _INDEX_HTML.replace("__APP_VERSION__", APP_VERSION)
    html = html.replace("__DEV_MODE__", "1" if IS_DEV_MODE else "0")
    return HTMLResponse(html)


@app.post("/api/shutdown")
def api_shutdown():
    """Gracefully shut down the server process."""
    import threading, signal, os
    def _kill():
        import time
        time.sleep(0.3)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return {"message": "Server shutting down…"}


@app.get("/api/version")
def api_version():
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    return {"version": APP_VERSION, "commit": commit}


@app.get("/api/collections")
def list_collections():
    s = get_state()
    return {"open": s.tracker.open_collections(), "all": s.tracker.all_collections()}


@app.get("/api/solutions")
def list_solutions_api():
    try:
        from solution_specs import list_solutions
        return {"solutions": list_solutions()}
    except Exception as e:
        return {"solutions": [], "error": str(e)}


_DEFAULT_SETTINGS = {
    "doc_types": ["product_catalog", "recipe_book", "faq", "manual", "legal", "general"],
    "llm_chat_model": "gpt-4o",
    "llm_processing_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-ada-002",
    "qdrant_url": "",
}

@app.get("/api/settings")
def get_settings():
    """Return global settings from solutions.yaml (settings: block) or defaults."""
    import yaml
    try:
        with open(_specs_file(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        stored = data.get("settings", {})
        merged = {**_DEFAULT_SETTINGS, **stored}
        # Fill qdrant_url from env if not stored
        if not merged.get("qdrant_url"):
            import os
            merged["qdrant_url"] = os.environ.get("QDRANT_URL", "")
        return merged
    except Exception as e:
        return {**_DEFAULT_SETTINGS, "error": str(e)}


class SettingsRequest(BaseModel):
    doc_types: list = []
    llm_chat_model: str = "gpt-4o"
    llm_processing_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-ada-002"
    qdrant_url: str = ""

@app.post("/api/settings")
def save_settings(req: SettingsRequest):
    """Save global settings into solutions.yaml settings: block."""
    import yaml
    try:
        with open(_specs_file(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data["settings"] = {
            "doc_types": req.doc_types,
            "llm_chat_model": req.llm_chat_model,
            "llm_processing_model": req.llm_processing_model,
            "embedding_model": req.embedding_model,
            "qdrant_url": req.qdrant_url,
        }
        with open(_specs_file(), "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        return {"message": "Settings saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dev/copy-prod-data")
def copy_prod_data():
    """Copy PROD state files into .dev_data/ so DEV has a fresh snapshot."""
    if not IS_DEV_MODE:
        raise HTTPException(status_code=403, detail="Only available in DEV mode")
    import glob as _glob, shutil
    patterns = [".wizard_state_*.json", ".shopify_stores.json", ".rag_state.json"]
    copied = []
    for pat in patterns:
        for src in _glob.glob(os.path.join(_REPO_ROOT, pat)):
            dst = os.path.join(_DATA_ROOT, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(os.path.basename(src))
    return {"copied": copied, "dest": _DATA_ROOT}


@app.get("/api/solutions/{solution_id}/collections")
def solution_collections(solution_id: str):
    """Return all Qdrant collections registered under a solution."""
    try:
        from solution_specs import get_solution, get_collections
        from solution_specs.loader import _ensure_sources
        sol = get_solution(solution_id)
        if not sol:
            raise HTTPException(status_code=404, detail=f"Solution '{solution_id}' not found")
        coll_entries = get_collections(solution_id)

        # Try to get Qdrant status; gracefully degrade if unavailable
        all_qdrant = set()
        tracker = None
        try:
            tracker = get_state().tracker
            all_qdrant = set(tracker.all_collections())
        except Exception:
            pass  # qdrant_client not available — return YAML data without Qdrant status

        import json as _json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def _source_status(cname: str, source_id: str) -> dict:
            """Check state file for a source and return its pipeline status."""
            # Try source-scoped file first, then collection-level fallback
            candidates = [
                os.path.join(root, f".rag_state_{cname}_{source_id}.json"),
                os.path.join(root, f".rag_state_{cname}.json"),
            ]
            for path in candidates:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            d = _json.load(f)
                        # Only use collection-level file if its source_id matches (or has none)
                        file_source_id = d.get("source_id")
                        if file_source_id and file_source_id != source_id:
                            continue
                        steps = d.get("completed_steps", [])
                        chunks = len(d.get("chunks", []))
                        items = len(d.get("scraped_items", []))
                        if "push_to_qdrant" in steps:
                            return {"status": "pushed", "chunks": chunks, "items": items}
                        elif "chunk" in steps:
                            return {"status": "chunked", "chunks": chunks, "items": items}
                        elif "fetch" in steps:
                            return {"status": "fetched", "chunks": 0, "items": items}
                        else:
                            return {"status": "started", "chunks": 0, "items": 0}
                    except Exception:
                        continue
            return {"status": "not_started", "chunks": 0, "items": 0}

        result = []
        for c in coll_entries:
            if not c.get("collection_name"):
                continue
            cname = c["collection_name"]
            exists = cname in all_qdrant
            points_count = 0
            if exists and tracker:
                try:
                    info = tracker._connection.get_collection(cname)
                    points_count = info.points_count or 0
                except Exception:
                    points_count = 0
            sources = _ensure_sources(c)
            # Add per-source pipeline status
            for src in sources:
                src["pipeline_status"] = _source_status(cname, src.get("id", "default"))
            result.append({
                "name": cname,
                "id": c.get("id", cname),
                "display_name": c.get("display_name", cname),
                "scraper_name": c.get("scraper_name", ""),
                "scraper_config": c.get("scraper_config"),
                "sources": sources,
                "routing": c.get("routing", {}),
                "exists": exists,
                "points_count": points_count,
            })
        return {"solution_id": solution_id, "collections": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _specs_file() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "solution_specs", "solutions.yaml")


class AddCollectionRequest(BaseModel):
    solution_id: str
    collection_name: str
    display_name: Optional[str] = None
    scraper_name: Optional[str] = None

@app.post("/api/solutions/add-collection")
def add_collection_to_solution(req: AddCollectionRequest):
    """Register a new collection dict under an existing solution in solutions.yaml."""
    import yaml
    try:
        with open(_specs_file(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        solutions = data.get("solutions", [])
        sol = next((s for s in solutions if s["id"] == req.solution_id), None)
        if not sol:
            raise HTTPException(status_code=404, detail=f"Solution '{req.solution_id}' not found")
        if "collections" not in sol:
            sol["collections"] = []
        existing_names = [c["collection_name"] for c in sol["collections"] if isinstance(c, dict)]
        if req.collection_name not in existing_names:
            new_entry = {
                "id": req.collection_name,
                "display_name": req.display_name or req.collection_name,
                "collection_name": req.collection_name,
                "collection_type": "scs",
                "routing": {},
            }
            if req.scraper_name:
                new_entry["scraper_name"] = req.scraper_name
            sol["collections"].append(new_entry)
        with open(_specs_file(), "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        from solution_specs import reload
        reload()
        return {"message": f"Added '{req.collection_name}' to solution '{req.solution_id}'"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteCollectionRequest(BaseModel):
    solution_id: str
    collection_name: str

@app.post("/api/solutions/delete-collection")
def delete_collection_from_solution(req: DeleteCollectionRequest):
    """Delete a collection from Qdrant and remove it from solutions.yaml."""
    import yaml
    try:
        tracker = get_state().tracker
        if tracker._existing_collection_name(req.collection_name):
            tracker._delete_collection(req.collection_name)
        with open(_specs_file(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for sol in data.get("solutions", []):
            if sol["id"] == req.solution_id:
                sol["collections"] = [
                    c for c in sol.get("collections", [])
                    if (c.get("collection_name") if isinstance(c, dict) else c) != req.collection_name
                ]
                break
        with open(_specs_file(), "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        from solution_specs import reload
        reload()
        return {"message": f"Deleted collection '{req.collection_name}'"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateRoutingRequest(BaseModel):
    routing: dict

class UpdateLanguageRequest(BaseModel):
    language: str

@app.put("/api/solutions/{solution_id}/collections/{collection_id}/routing")
def update_collection_routing(solution_id: str, collection_id: str, req: UpdateRoutingRequest):
    """Update the routing metadata block for a specific collection in solutions.yaml."""
    from workflow.suggest import save_routing_metadata
    success = save_routing_metadata(solution_id, collection_id, req.routing)
    if not success:
        raise HTTPException(status_code=404,
                            detail=f"Collection '{collection_id}' in solution '{solution_id}' not found")
    return {"message": f"Routing metadata updated for {solution_id}/{collection_id}"}


@app.put("/api/solutions/{solution_id}/language")
def update_solution_language(solution_id: str, req: UpdateLanguageRequest):
    """Set the base language for a solution (stored in solutions.yaml)."""
    from solution_specs.loader import save_solution_language
    language = req.language.strip()
    if not language:
        raise HTTPException(status_code=400, detail="language is required")
    success = save_solution_language(solution_id, language)
    if not success:
        raise HTTPException(status_code=404, detail=f"Solution '{solution_id}' not found")
    return {"message": f"Base language for '{solution_id}' set to '{language}'"}


@app.post("/api/solutions/{solution_id}/collections/{collection_id}/routing/suggest")
def suggest_collection_routing(solution_id: str, collection_id: str):
    """
    Re-generate routing metadata for a collection by sampling its existing Qdrant points.
    Calls suggest_collection_metadata() with the stored chunk texts, saves result to solutions.yaml.
    """
    from solution_specs.loader import get_solution
    from qdrant_utils import get_points_from_collection
    from workflow.suggest import suggest_collection_metadata, save_routing_metadata

    # Resolve collection_name from solutions.yaml
    sol = get_solution(solution_id)
    if not sol:
        raise HTTPException(status_code=404, detail=f"Solution '{solution_id}' not found")

    coll = next((c for c in sol.get("collections", []) if c.get("id") == collection_id), None)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found in solution '{solution_id}'")

    collection_name = coll.get("collection_name", collection_id)

    # Scroll all points from Qdrant and extract text
    try:
        points = get_points_from_collection(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve points from Qdrant: {e}")

    if not points:
        raise HTTPException(status_code=404, detail=f"No points found in collection '{collection_name}'. Has it been pushed to Qdrant?")

    # Extract text — payload structure is payload["point"]["text"] (new) or payload["text"] (legacy)
    chunks = []
    for p in points:
        payload = p.payload or {}
        text = payload.get("point", {}).get("text") or payload.get("text") or ""
        if text:
            chunks.append(text)

    if not chunks:
        raise HTTPException(status_code=404, detail="Points found but no text content could be extracted.")

    # Use solution-level language if set (enforces consistent language across all collections)
    base_language = sol.get("language") or None
    if base_language:
        print(f"[suggest_routing] Using solution base language: {base_language}")

    print(f"[suggest_routing] Sampling {len(chunks)} chunks from '{collection_name}' for metadata generation...")

    # Generate new metadata — pass language to enforce consistent output
    metadata = suggest_collection_metadata(chunks, source_label=collection_name, language=base_language)
    if not metadata:
        raise HTTPException(status_code=500, detail="LLM metadata generation returned empty result.")

    # Strip topics, keep only routing-relevant fields
    # Always enforce solution base language in the saved routing block
    routing = {
        k: metadata[k]
        for k in ("description", "keywords", "typical_questions", "not_covered", "language", "doc_type")
        if metadata.get(k) is not None
    }
    if base_language:
        routing["language"] = base_language  # guarantee it's saved correctly even if LLM drifted

    # Save to solutions.yaml
    save_routing_metadata(solution_id, collection_id, routing)

    return {"routing": routing, "chunks_sampled": len(chunks)}


class AddSourceRequest(BaseModel):
    source_type: str  # url | pdf | txt | csv
    label: str
    scraper_name: Optional[str] = None
    file_path: Optional[str] = None  # for pdf | txt | csv sources

@app.post("/api/solutions/{solution_id}/collections/{collection_name}/sources")
def add_source(solution_id: str, collection_name: str, req: AddSourceRequest):
    """Add a new source to a collection."""
    from solution_specs.loader import get_sources, save_collection_sources, _ensure_sources, get_collections
    coll = next((c for c in get_collections(solution_id) if c.get("collection_name") == collection_name), None)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    sources = _ensure_sources(coll)
    # Generate a unique id
    import re
    base_id = re.sub(r'[^a-z0-9]+', '_', req.label.lower()).strip('_') or 'source'
    existing_ids = {s.get("id") for s in sources}
    src_id = base_id
    n = 2
    while src_id in existing_ids:
        src_id = f"{base_id}_{n}"
        n += 1
    new_src = {"id": src_id, "type": req.source_type, "label": req.label}
    if req.scraper_name:
        new_src["scraper_name"] = req.scraper_name
    if req.file_path:
        new_src["file_path"] = req.file_path
    sources.append(new_src)
    save_collection_sources(solution_id, collection_name, sources)
    return {"source": new_src, "sources": sources}

class UpdateSourceRequest(BaseModel):
    file_path: Optional[str] = None

@app.patch("/api/solutions/{solution_id}/collections/{collection_name}/sources/{source_id}")
def update_source(solution_id: str, collection_name: str, source_id: str, req: UpdateSourceRequest):
    """Update a source's fields (e.g. file_path after first browse)."""
    from solution_specs.loader import save_collection_sources, _ensure_sources, get_collections
    coll = next((c for c in get_collections(solution_id) if c.get("collection_name") == collection_name), None)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    sources = _ensure_sources(coll)
    src = next((s for s in sources if s.get("id") == source_id), None)
    if not src:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")
    if req.file_path is not None:
        src["file_path"] = req.file_path
    save_collection_sources(solution_id, collection_name, sources)
    return {"source": src, "sources": sources}

@app.delete("/api/solutions/{solution_id}/collections/{collection_name}/sources/{source_id}")
def remove_source(solution_id: str, collection_name: str, source_id: str):
    """Remove a source from a collection."""
    from solution_specs.loader import save_collection_sources, _ensure_sources, get_collections
    coll = next((c for c in get_collections(solution_id) if c.get("collection_name") == collection_name), None)
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    sources = _ensure_sources(coll)
    new_sources = [s for s in sources if s.get("id") != source_id]
    if len(new_sources) == len(sources):
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")
    save_collection_sources(solution_id, collection_name, new_sources)
    return {"sources": new_sources}


@app.delete("/api/solutions/{solution_id}/collections/{collection_name}/sources/{source_id}/chunks")
def delete_source_chunks(solution_id: str, collection_name: str, source_id: str):
    """Delete all Qdrant points belonging to a specific source, using source URLs from state file."""
    if IS_DEV_MODE:
        raise HTTPException(status_code=403, detail="Qdrant operations are disabled in DEV mode. Merge to main and use the production server.")
    import json as _json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Find the state file
    candidates = [
        os.path.join(root, f".rag_state_{collection_name}_{source_id}.json"),
        os.path.join(root, f".rag_state_{collection_name}.json"),
    ]
    state_data = None
    state_path = None
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = _json.load(f)
                file_source_id = d.get("source_id")
                if file_source_id and file_source_id != source_id:
                    continue
                state_data = d
                state_path = path
                break
            except Exception:
                continue

    if not state_data:
        raise HTTPException(status_code=404,
            detail=f"No state file found for source '{source_id}' in collection '{collection_name}'")

    # Collect URLs to delete
    urls_to_delete = set()
    for item in state_data.get("scraped_items", []):
        if isinstance(item, dict) and item.get("url"):
            urls_to_delete.add(item["url"])
    sc = state_data.get("source_config") or {}
    file_path = sc.get("path") or sc.get("pdf_path")
    if file_path:
        urls_to_delete.add(file_path)

    if not urls_to_delete:
        raise HTTPException(status_code=400,
            detail="Could not determine source URLs to delete from state file")

    # Delete from Qdrant
    from QdrantTracker import QdrantTracker
    tracker = QdrantTracker()
    deleted_count = 0
    errors = []
    for url in urls_to_delete:
        try:
            tracker._delete_points_by_url(collection_name, url)
            deleted_count += 1
        except Exception as e:
            errors.append({"url": url, "error": str(e)})

    # Reset pipeline status: remove 'push_to_qdrant' from completed_steps
    steps = state_data.get("completed_steps", [])
    if "push_to_qdrant" in steps:
        steps.remove("push_to_qdrant")
        state_data["completed_steps"] = steps
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                _json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # non-fatal

    return {"deleted_count": deleted_count, "urls_processed": len(urls_to_delete), "errors": errors}


# ── Chunk viewer / editor endpoints ─────────────────────────────────────
@app.get("/api/collections/{collection_name}/chunks")
def get_collection_chunks(collection_name: str, source_id: str = None):
    """Return all Qdrant points for a source, grouped by source_url."""
    import json as _json
    from QdrantTracker import QdrantTracker

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Find the state file to get source URLs and excluded URLs
    urls = set()
    excluded_urls = []
    if source_id:
        candidates = [
            os.path.join(root, f".rag_state_{collection_name}_{source_id}.json"),
            os.path.join(root, f".rag_state_{collection_name}.json"),
        ]
        for path in candidates:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        d = _json.load(f)
                    file_source_id = d.get("source_id")
                    if file_source_id and file_source_id != source_id:
                        continue
                    for item in d.get("scraped_items", []):
                        if isinstance(item, dict) and item.get("url"):
                            urls.add(item["url"])
                    sc = d.get("source_config") or {}
                    fp = sc.get("path") or sc.get("pdf_path")
                    if fp:
                        urls.add(fp)
                    excluded_urls = d.get("excluded_urls", [])
                    break
                except Exception:
                    continue

    tracker = QdrantTracker()
    if not tracker._existing_collection_name(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in Qdrant")

    if urls:
        records = tracker.scroll_points_by_urls(collection_name, list(urls))
    else:
        # No state file or no source_id: return all points
        records = []
        offset = None
        while True:
            result, off = tracker._connection.scroll(
                collection_name=collection_name,
                with_payload=True, with_vectors=False,
                offset=offset, limit=100,
            )
            records.extend(result)
            if off is None:
                break
            offset = off

    # Group by source_url
    by_url = {}
    for rec in records:
        p = rec.payload.get("point", {})
        url = p.get("source_url", "__no_url__")
        chunk = {
            "id": str(rec.id),
            "text": p.get("text", ""),
            "idx": p.get("idx"),
            "source": p.get("source", ""),
            "source_url": url,
            "content_hash": p.get("content_hash", ""),
            "scraped_at": p.get("scraped_at", ""),
            "manually_edited": p.get("manually_edited", False),
            "edited_at": p.get("edited_at"),
            "original_text": p.get("original_text"),
        }
        by_url.setdefault(url, []).append(chunk)

    # Sort chunks within each URL by idx
    url_groups = []
    for url in sorted(by_url.keys()):
        chunks = sorted(by_url[url], key=lambda c: (c["idx"] if c["idx"] is not None else 0))
        url_groups.append({"url": url, "chunks": chunks})

    return {"urls": url_groups, "total_chunks": len(records), "excluded_urls": excluded_urls}


class ChunkUpdateRequest(BaseModel):
    text: str


@app.put("/api/collections/{collection_name}/chunks/{point_id}")
def update_chunk(collection_name: str, point_id: str, req: ChunkUpdateRequest):
    """Update a single chunk's text in Qdrant, re-embed, and mark as manually edited."""
    if IS_DEV_MODE:
        raise HTTPException(status_code=403, detail="Qdrant operations are disabled in DEV mode.")

    from QdrantTracker import QdrantTracker
    from vectorization import get_embedding

    tracker = QdrantTracker()
    if not tracker._existing_collection_name(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    new_text = req.text.strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Chunk text cannot be empty")

    # Get current point to check for existing original_text
    try:
        points = tracker._connection.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Point '{point_id}' not found: {e}")

    if not points:
        raise HTTPException(status_code=404, detail=f"Point '{point_id}' not found")

    current_payload = points[0].payload.get("point", {})
    current_text = current_payload.get("text", "")

    # Only set original_text on first edit
    original_text = None
    if not current_payload.get("manually_edited"):
        original_text = current_text

    # Re-embed
    try:
        new_vector = get_embedding(new_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    from datetime import datetime, timezone
    edited_at = datetime.now(timezone.utc).isoformat()

    success = tracker.update_point(collection_name, point_id, new_text, new_vector,
                                    original_text=original_text)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update point in Qdrant")

    return {
        "id": point_id,
        "text": new_text,
        "manually_edited": True,
        "edited_at": edited_at,
        "original_text": original_text or current_payload.get("original_text"),
    }


def _find_state_file(root: str, collection_name: str, source_id: str):
    """Find the state file for a source, trying source_id-suffixed path first, then plain."""
    candidates = [
        os.path.join(root, f".rag_state_{collection_name}_{source_id}.json"),
        os.path.join(root, f".rag_state_{collection_name}.json"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return None


class ChunkDeleteRequest(BaseModel):
    point_ids: list
    source_id: str = None
    url: str = None  # set when deleting an entire URL group


@app.post("/api/collections/{collection_name}/chunks/delete")
def delete_chunks(collection_name: str, req: ChunkDeleteRequest):
    """Delete specific chunks from Qdrant. If all chunks for a URL are deleted, auto-exclude that URL."""
    if IS_DEV_MODE:
        raise HTTPException(status_code=403, detail="Qdrant operations are disabled in DEV mode.")

    import json as _json
    from QdrantTracker import QdrantTracker

    tracker = QdrantTracker()
    if not tracker._existing_collection_name(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    if not req.point_ids:
        raise HTTPException(status_code=400, detail="point_ids is required")

    # --- Determine the URL of the deleted chunks (for auto-exclude check) ---
    chunk_url = req.url
    if not chunk_url and req.source_id:
        # Try to figure out the URL from the first point's payload
        try:
            points = tracker._connection.retrieve(
                collection_name=collection_name,
                ids=[req.point_ids[0]],
                with_payload=True,
                with_vectors=False,
            )
            if points:
                chunk_url = points[0].payload.get("point", {}).get("source_url")
        except Exception:
            pass

    # --- Delete points ---
    deleted = tracker.delete_points_by_ids(collection_name, req.point_ids)

    # --- Check if URL should be excluded ---
    excluded_url = None
    auto_excluded = False

    if req.url:
        # Explicit URL group delete → always exclude
        excluded_url = req.url
    elif chunk_url and req.source_id:
        # Individual chunk delete → check if URL still has chunks
        remaining = tracker.scroll_points_by_urls(collection_name, [chunk_url])
        if not remaining:
            excluded_url = chunk_url
            auto_excluded = True

    # --- Persist exclusion to state file ---
    if excluded_url and req.source_id:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        state_path = _find_state_file(root, collection_name, req.source_id)
        if state_path:
            try:
                from workflow.models import WorkflowState
                state = WorkflowState.load_from_disk(state_path)
                if excluded_url not in (state.excluded_urls or []):
                    state.excluded_urls.append(excluded_url)
                    state.save_to_disk()
            except Exception as e:
                print(f"[delete_chunks] Failed to update excluded_urls in state: {e}")

    return {
        "deleted_count": deleted,
        "excluded_url": excluded_url,
        "auto_excluded": auto_excluded,
    }


class ChunkRestoreUrlRequest(BaseModel):
    source_id: str
    url: str


@app.post("/api/collections/{collection_name}/chunks/restore-url")
def restore_url(collection_name: str, req: ChunkRestoreUrlRequest):
    """Remove a URL from excluded_urls so it will be included in future pushes."""
    if IS_DEV_MODE:
        raise HTTPException(status_code=403, detail="Qdrant operations are disabled in DEV mode.")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state_path = _find_state_file(root, collection_name, req.source_id)
    if not state_path:
        raise HTTPException(status_code=404, detail=f"State file not found for source '{req.source_id}'")

    try:
        from workflow.models import WorkflowState
        state = WorkflowState.load_from_disk(state_path)
        if req.url in (state.excluded_urls or []):
            state.excluded_urls.remove(req.url)
            state.save_to_disk()
            return {"restored": True}
        return {"restored": False, "detail": "URL was not in excluded list"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore URL: {e}")


@app.get("/api/collections/{collection_name}/edited-chunks")
def get_edited_chunks(collection_name: str, source_id: str = None):
    """Check which URLs have manually edited chunks (for push guard)."""
    import json as _json
    from QdrantTracker import QdrantTracker
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracker = QdrantTracker()

    if not tracker._existing_collection_name(collection_name):
        return {"edited_urls": [], "total_edited": 0}

    # Get source URLs from state file
    source_urls = set()
    if source_id:
        candidates = [
            os.path.join(root, f".rag_state_{collection_name}_{source_id}.json"),
            os.path.join(root, f".rag_state_{collection_name}.json"),
        ]
        for path in candidates:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        d = _json.load(f)
                    file_source_id = d.get("source_id")
                    if file_source_id and file_source_id != source_id:
                        continue
                    for item in d.get("scraped_items", []):
                        if isinstance(item, dict) and item.get("url"):
                            source_urls.add(item["url"])
                    sc = d.get("source_config") or {}
                    fp = sc.get("path") or sc.get("pdf_path")
                    if fp:
                        source_urls.add(fp)
                    break
                except Exception:
                    continue

    # Query for manually_edited points
    # Try server-side filter first; fall back to client-side if payload index missing
    edited_records = []
    try:
        must_filters = [FieldCondition(key="point.manually_edited", match=MatchValue(value=True))]
        if source_urls:
            must_filters.append(FieldCondition(key="point.source_url", match=MatchAny(any=list(source_urls))))
        scroll_filter = Filter(must=must_filters)
        offset = None
        while True:
            result, offset = tracker._connection.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                limit=100,
            )
            edited_records.extend(result)
            if offset is None:
                break
    except Exception:
        # Fallback: scroll all points, filter client-side
        edited_records = []
        offset = None
        while True:
            result, offset = tracker._connection.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                limit=100,
            )
            for r in result:
                p = r.payload.get("point", {})
                if not p.get("manually_edited"):
                    continue
                if source_urls and p.get("source_url") not in source_urls:
                    continue
                edited_records.append(r)
            if offset is None:
                break

    # Group by URL and count
    by_url = {}
    for rec in edited_records:
        p = rec.payload.get("point", {})
        url = p.get("source_url", "__no_url__")
        by_url.setdefault(url, 0)
        by_url[url] += 1

    # Also get total counts per URL (for display)
    edited_urls = []
    for url, edited_count in sorted(by_url.items()):
        edited_urls.append({
            "url": url,
            "edited_count": edited_count,
        })

    return {"edited_urls": edited_urls, "total_edited": len(edited_records)}


@app.post("/api/workflow/step")
def run_workflow_step(req: StepRequest):
    if IS_DEV_MODE and req.step == "push_to_qdrant":
        raise HTTPException(status_code=403, detail="Push to Qdrant is disabled in DEV mode. Merge to main and push from the production server.")
    from workflow.models import WorkflowState, Step, ChunkingConfig
    from workflow.runner import run_step
    try:
        state = get_state()
        # Ensure chunking_config is always a proper object, never a plain dict
        if isinstance(state.chunking_config, dict):
            cfg = state.chunking_config
            state.chunking_config = ChunkingConfig(
                batch_size=cfg.get("batch_size", 10000),
                overlap_size=cfg.get("overlap_size", 100),
                use_proposition_chunking=cfg.get("use_proposition_chunking", False),
                simple_chunk_size=cfg.get("simple_chunk_size", 1000),
                simple_chunk_overlap=cfg.get("simple_chunk_overlap", 200),
                use_hierarchical_chunking=cfg.get("use_hierarchical_chunking", False),
                hierarchical_parent_size=cfg.get("hierarchical_parent_size", 2000),
                hierarchical_parent_overlap=cfg.get("hierarchical_parent_overlap", 200),
                hierarchical_child_size=cfg.get("hierarchical_child_size", 400),
                hierarchical_child_overlap=cfg.get("hierarchical_child_overlap", 50),
            )
        if req.state_update:
            # Clear save_path when source_id changes so state file gets scoped correctly
            new_source_id = req.state_update.get("source_id")
            if new_source_id and new_source_id != state.source_id:
                state.save_path = None
            for k, v in req.state_update.items():
                if k == "source_config" and isinstance(v, dict):
                    state.source_config = v
                elif k == "chunking_config" and isinstance(v, dict):
                    if state.chunking_config is None:
                        state.chunking_config = ChunkingConfig()
                    if "batch_size" in v:
                        state.chunking_config.batch_size = v["batch_size"]
                    if "overlap_size" in v:
                        state.chunking_config.overlap_size = v["overlap_size"]
                    if "use_proposition_chunking" in v:
                        state.chunking_config.use_proposition_chunking = v["use_proposition_chunking"]
                    if "use_hierarchical_chunking" in v:
                        state.chunking_config.use_hierarchical_chunking = v["use_hierarchical_chunking"]
                    if "hierarchical_parent_size" in v:
                        state.chunking_config.hierarchical_parent_size = v["hierarchical_parent_size"]
                    if "hierarchical_child_size" in v:
                        state.chunking_config.hierarchical_child_size = v["hierarchical_child_size"]
                elif k != "chunking_config" and hasattr(state, k):
                    setattr(state, k, v)
        try:
            step = Step(req.step)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown step: {req.step}")
        msg = run_step(state, step)
        return {"message": msg, "state": state.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] /api/workflow/step crashed:\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qa")
def qa(req: QARequest):
    from chatbot import get_retrieved_info, get_answer
    # Resolve collection name(s)
    if req.collection_name == "__all__" and req.solution_id:
        from solution_specs import get_collections
        colls = get_collections(req.solution_id)
        collection_names = [c["collection_name"] for c in colls if c.get("collection_name")]
    elif "," in req.collection_name:
        collection_names = [s.strip() for s in req.collection_name.split(",") if s.strip()]
    else:
        collection_names = req.collection_name
    history = []
    retrieved = get_retrieved_info(req.question, history, collection_names,
                                   embedding_model=req.embedding_model)
    answer = get_answer(history, retrieved, req.question, req.company)
    # retrieved is now a dict {text, sources}; pass sources through to the UI
    sources = retrieved.get("sources", []) if isinstance(retrieved, dict) else []
    return {"question": req.question, "answer": answer, "sources": sources}


@app.post("/api/workflow/reset")
def workflow_reset():
    reset_state()
    return {"message": "State reset."}


@app.get("/api/state/check")
def state_check(path: str):
    """Check if a .rag_state.json exists for the given source file path."""
    import os
    base = os.path.splitext(path)[0]
    save_path = base + ".rag_state.json"
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        try:
            import json
            with open(save_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {
                "found": True,
                "save_path": save_path,
                "completed_steps": d.get("completed_steps", []),
                "collection_name": d.get("collection_name"),
                "chunks_count": len(d.get("chunks", [])),
                "has_cleaned_text": bool(d.get("cleaned_text")),
                "has_raw_text": bool(d.get("raw_text")),
            }
        except Exception as e:
            return {"found": False, "error": str(e)}
    return {"found": False}


class StateLoadRequest(BaseModel):
    save_path: Optional[str] = None
    path: Optional[str] = None

@app.post("/api/state/load")
def state_load(req: StateLoadRequest):
    """Load state from a .rag_state.json file and set it as current state."""
    global _current_state
    import os
    save_path = req.save_path or req.path
    if not save_path or not os.path.exists(save_path):
        raise HTTPException(status_code=404, detail=f"State file not found: {save_path}")
    try:
        from workflow.models import WorkflowState
        tracker = get_state().tracker  # reuse existing tracker
        state = WorkflowState.load_from_disk(save_path, tracker=tracker)

        # Recreate the collection object if a collection name exists in Qdrant
        if state.collection_name and tracker._existing_collection_name(state.collection_name):
            try:
                coll_type = "group" if state.grouping_enabled else "scs"
                state.collection_object = tracker.new(state.collection_name, coll_type)
            except Exception as ce:
                print(f"[state/load] Could not reopen collection: {ce}")

        _current_state = state
        steps_done = ', '.join(state.completed_steps) or 'none'
        chunks_info = f" {len(state.chunks)} chunks ready." if state.chunks else ""
        return {
            "message": f"✅ Resumed! Steps done: {steps_done}.{chunks_info} You can continue from where you left off.",
            "state": state.to_dict()
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/state/check-by-collection")
def state_check_by_collection(collection_name: str, source_id: Optional[str] = None):
    """Check if a .rag_state_{collection_name}[_{source_id}].json exists in project root.
    When source_id is given, checks source-scoped file first, falls back to collection-level."""
    import os, json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    if source_id:
        candidates.append(os.path.join(root, f".rag_state_{collection_name}_{source_id}.json"))
    candidates.append(os.path.join(root, f".rag_state_{collection_name}.json"))
    for save_path in candidates:
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                return {
                    "found": True,
                    "save_path": save_path,
                    "completed_steps": d.get("completed_steps", []),
                    "collection_name": d.get("collection_name"),
                    "source_id": d.get("source_id"),
                    "chunks_count": len(d.get("chunks", [])),
                    "scraped_items_count": len(d.get("scraped_items", [])),
                }
            except Exception as e:
                return {"found": False, "error": str(e)}
    return {"found": False}


def _clear_progress_queue():
    while not _progress_queue.empty():
        try:
            _progress_queue.get_nowait()
        except Exception:
            break


def _apply_state_update(state, state_update: dict):
    from workflow.models import ChunkingConfig
    if not state_update:
        return
    # Clear save_path when source_id changes so state file gets scoped correctly
    new_source_id = state_update.get("source_id")
    if new_source_id and new_source_id != state.source_id:
        state.save_path = None
    for k, v in state_update.items():
        if k == "source_config" and isinstance(v, dict):
            state.source_config = v
        elif k == "chunking_config" and isinstance(v, dict):
            pass  # handled elsewhere
        elif hasattr(state, k):
            setattr(state, k, v)


class FaqTableRequest(BaseModel):
    collection_name: str
    company_name: str = "the company"
    language: str = "en"
    max_items: int = 100


@app.post("/api/faq/generate-table")
def faq_generate_table(req: FaqTableRequest):
    """
    Extract Q&A pairs as a tab-separated table for copy-paste into jBKE.
    Source priority:
      1. Qdrant collection (if it exists and has chunks)
      2. Saved workflow state file (.rag_state_{collection_name}.json) — pre-Qdrant fallback
      3. Neither → helpful error message
    """
    import os
    from llms.openai_utils import openai_chat_completion
    tracker = get_state().tracker
    texts = []
    source_label = None

    # Source 1: Qdrant (preferred)
    if tracker._existing_collection_name(req.collection_name):
        points = tracker.scroll_all(req.collection_name, limit=req.max_items)
        texts = [p.get("text", "") for p in (points or []) if p.get("text")]
        if texts:
            source_label = "Qdrant"

    # Source 2: Saved workflow state file (pre-Qdrant fallback)
    if not texts:
        from workflow.models import WorkflowState
        state_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f".rag_state_{req.collection_name}.json"
        )
        if os.path.exists(state_path):
            try:
                saved = WorkflowState.load_from_disk(state_path)
                if saved and saved.chunks:
                    texts = [t for t in saved.chunks[:req.max_items] if t]
                    source_label = "saved state (chunks)"
                elif saved and saved.scraped_items:
                    texts = [item.get("text", "") for item in saved.scraped_items[:req.max_items] if item.get("text")]
                    source_label = "saved state (scraped pages)"
            except Exception:
                pass

    if not texts:
        return {"table": "", "count": 0, "error": "No content found. Open 'Work with RAG', run the Fetch step for this collection, then try again."}

    joined = "\n---\n".join(t[:400] for t in texts)
    system_prompt = (
        "Extract FAQ Q&A pairs from the provided text chunks.\n"
        "Format each pair as: Question<TAB>Answer\n"
        "One pair per line. No header. No markdown. No numbering.\n"
        "Skip chunks that are not FAQ content.\n"
        f"Language: {req.language}. Company: {req.company_name}.\n"
        "Return ONLY the tab-separated lines."
    )
    try:
        result = openai_chat_completion(system_prompt, f"Chunks:\n{joined}", model="gpt-4o-mini")
        lines = [ln for ln in result.strip().splitlines() if "\t" in ln]
        return {"table": "\n".join(lines), "count": len(lines), "source": source_label}
    except Exception as e:
        return {"table": "", "count": 0, "error": str(e)}


@app.get("/api/workflow/state")
def get_workflow_state():
    """Return the current WorkflowState as a dict (for polling after streaming steps)."""
    return get_state().to_dict()


@app.post("/api/workflow/fetch")
def run_fetch_streaming(req: StepRequest):
    """Run FETCH in a background thread, streaming stdout lines via SSE progress queue."""
    from workflow.models import Step, WorkflowState, ChunkingConfig
    from workflow.runner import run_step
    import io, sys

    state = get_state()
    _apply_state_update(state, req.state_update)
    # Pass login credentials (session-only, never persisted to disk)
    if req.login_config and req.login_config.get("username"):
        state.source_config = state.source_config or {}
        state.source_config["login_config"] = req.login_config
    _clear_progress_queue()

    class _QueueWriter(io.TextIOBase):
        """Intercepts print() calls and forwards them to the SSE progress queue."""
        def write(self, s):
            s = s.strip()
            if s:
                _progress_queue.put(f"LOG:{s}")
            return len(s)
        def flush(self): pass

    def _run():
        old_stdout = sys.stdout
        sys.stdout = _QueueWriter()
        try:
            msg = run_step(state, Step.FETCH)
            _progress_queue.put(f"DONE:{msg}")
        except Exception as e:
            _progress_queue.put(f"ERROR:{e}")
        finally:
            sys.stdout = old_stdout

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Fetch started in background. Watch /api/progress for updates."}


@app.post("/api/workflow/push")
def run_push_streaming(req: StepRequest):
    """Run PUSH in a background thread, streaming stdout lines via SSE progress queue."""
    if IS_DEV_MODE:
        raise HTTPException(status_code=403, detail="Push to Qdrant is disabled in DEV mode. Merge to main and push from the production server.")
    from workflow.models import Step
    from workflow.runner import run_step
    import io, sys

    state = get_state()
    _apply_state_update(state, req.state_update)
    _clear_progress_queue()

    class _QueueWriter(io.TextIOBase):
        def write(self, s):
            s = s.strip()
            if s:
                _progress_queue.put(f"LOG:{s}")
            return len(s)
        def flush(self): pass

    def _run():
        old_stdout = sys.stdout
        sys.stdout = _QueueWriter()
        try:
            msg = run_step(state, Step.PUSH_TO_QDRANT)
            _progress_queue.put(f"DONE:{msg}")
        except Exception as e:
            _progress_queue.put(f"ERROR:{e}")
        finally:
            sys.stdout = old_stdout

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Push started in background. Watch /api/progress for updates."}


@app.post("/api/workflow/translate")
def run_translate(req: StepRequest):
    """Run translate_and_clean in a background thread so SSE can stream progress."""
    from workflow.models import Step
    from workflow.runner import run_step
    state = get_state()
    _apply_state_update(state, req.state_update)
    _clear_progress_queue()

    def _run():
        run_step(state, Step.TRANSLATE_AND_CLEAN)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"message": "Translate started in background. Watch /api/progress for updates."}


@app.get("/api/progress")
async def progress_stream():
    """Async SSE endpoint — polls the thread-safe queue without blocking the event loop."""
    async def event_gen():
        while True:
            # Poll the thread queue every 0.2s without blocking the event loop
            try:
                msg = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, _progress_queue.get),
                    timeout=300  # 5 min max total wait
                )
                yield f"data: {msg}\n\n"
                if msg.startswith("DONE") or msg.startswith("ERROR"):
                    break
            except asyncio.TimeoutError:
                yield "data: TIMEOUT\n\n"
                break
    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class SuggestChunkingRequest(BaseModel):
    text_preview: str
    source_type: str = "unknown"


@app.post("/api/suggest/chunking")
def api_suggest_chunking(req: SuggestChunkingRequest):
    from workflow.suggest import suggest_chunking
    return suggest_chunking(req.text_preview, req.source_type)


@app.get("/api/pick-file")
def pick_file(source_type: str = "pdf"):
    """Open a native macOS file picker via osascript and return the selected path."""
    import subprocess

    ext_map = {
        "pdf": '{"pdf", "PDF"}',
        "txt": '{"txt", "TXT"}',
        "csv": '{"csv", "CSV"}',
    }
    ext = ext_map.get(source_type, '{"*"}')

    script = f"""
    tell application "Finder"
        activate
    end tell
    try
        set f to choose file with prompt "Select a {source_type.upper()} file" of type {ext}
        POSIX path of f
    on error
        ""
    end try
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=120
        )
        path = result.stdout.strip()
        return {"path": path}
    except Exception as e:
        return {"path": "", "error": str(e)}


class SyncRequest(BaseModel):
    scraper_name: str
    collection_name: str
    scraper_options: dict = {}


@app.post("/api/sync")
def sync_collection_api(req: SyncRequest):
    """
    Incremental sync: re-scrape all URLs, compare content_hash, re-embed only changed pages.
    Returns a diff summary: {added, updated, deleted, unchanged, errors, message}.
    """
    tracker = get_state().tracker
    if tracker is None:
        from QdrantTracker import QdrantTracker
        tracker = QdrantTracker()
        get_state().tracker = tracker

    diff = tracker.sync_collection(req.scraper_name, req.collection_name, req.scraper_options or {})
    msg = (
        f"Sync complete: "
        f"+{diff['added']} added, "
        f"~{diff['updated']} updated, "
        f"-{diff['deleted']} deleted, "
        f"{diff['unchanged']} unchanged"
    )
    if diff["errors"]:
        msg += f" | {len(diff['errors'])} error(s): " + "; ".join(diff["errors"][:3])
    return {"message": msg, "diff": diff}


# ── Shopify Stores — local config (gitignored) ────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Resolve the main repo root (not a worktree) for state/data files.
import subprocess as _sp
try:
    _REPO_ROOT = _sp.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=_PROJECT_ROOT, text=True
    ).strip().removesuffix("/.git")
except Exception:
    _REPO_ROOT = _PROJECT_ROOT

# DEV mode: isolate state files in .dev_data/ so PROD data is never touched.
# PROD mode: state files live in the repo root as before.
if IS_DEV_MODE:
    _DATA_ROOT = os.path.join(_REPO_ROOT, ".dev_data")
    os.makedirs(_DATA_ROOT, exist_ok=True)
else:
    _DATA_ROOT = _REPO_ROOT


def _stores_path() -> str:
    return os.path.join(_DATA_ROOT, ".shopify_stores.json")


def _load_stores() -> dict:
    import json
    p = _stores_path()
    if not os.path.exists(p):
        return {"stores": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_stores(data: dict):
    import json
    with open(_stores_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _store_id_from_url(shop_url: str) -> str:
    import re
    hostname = re.sub(r"https?://", "", shop_url).rstrip("/").split("/")[0]
    return re.sub(r"[^a-z0-9]", "", hostname.lower())


class ShopifyStoreRequest(BaseModel):
    display_name: str
    shop_url: str
    access_token: Optional[str] = ""
    include: Optional[list] = None  # default set in endpoint
    metafields: Optional[bool] = False


@app.get("/api/shopify/stores")
def shopify_list_stores():
    """List all locally configured Shopify stores. Access tokens are masked."""
    data = _load_stores()
    result = []
    for s in data.get("stores", []):
        token = s.get("access_token", "")
        result.append({
            "id": s["id"],
            "display_name": s.get("display_name", s["id"]),
            "shop_url": s.get("shop_url", ""),
            "include": s.get("include", ["products", "pages", "articles"]),
            "metafields": s.get("metafields", False),
            "last_fetched": s.get("last_fetched"),
            "has_token": bool(token),
            "token_hint": token[-4:] if token else "",
        })
    return {"stores": result}


@app.post("/api/shopify/stores")
def shopify_create_store(req: ShopifyStoreRequest):
    """Register a new Shopify store config."""
    data = _load_stores()
    store_id = _store_id_from_url(req.shop_url)
    if not store_id:
        raise HTTPException(status_code=400, detail="Invalid shop_url — could not derive store ID")
    if any(s["id"] == store_id for s in data["stores"]):
        raise HTTPException(status_code=409, detail=f"Store '{store_id}' already exists")
    new_store = {
        "id": store_id,
        "display_name": req.display_name,
        "shop_url": req.shop_url.rstrip("/"),
        "access_token": req.access_token or "",
        "include": req.include if req.include is not None else ["products", "pages", "articles"],
        "metafields": req.metafields or False,
        "last_fetched": None,
    }
    data["stores"].append(new_store)
    _save_stores(data)
    return {"ok": True, "id": store_id}


@app.put("/api/shopify/stores/{store_id}")
def shopify_update_store(store_id: str, req: ShopifyStoreRequest):
    """Update an existing Shopify store config. Blank access_token keeps the existing value."""
    data = _load_stores()
    store = next((s for s in data["stores"] if s["id"] == store_id), None)
    if not store:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found")
    store["display_name"] = req.display_name
    store["shop_url"] = req.shop_url.rstrip("/")
    if req.access_token:  # blank = keep existing
        store["access_token"] = req.access_token
    store["include"] = req.include if req.include is not None else ["products", "pages", "articles"]
    store["metafields"] = req.metafields or False
    _save_stores(data)
    return {"ok": True}


@app.delete("/api/shopify/stores/{store_id}")
def shopify_delete_store(store_id: str):
    """Delete a Shopify store config."""
    data = _load_stores()
    before = len(data["stores"])
    data["stores"] = [s for s in data["stores"] if s["id"] != store_id]
    if len(data["stores"]) == before:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found")
    _save_stores(data)
    return {"ok": True}


@app.post("/api/shopify/stores/{store_id}/test")
def shopify_test_store(store_id: str):
    """Test connection to a Shopify store. Returns shop info or error."""
    import httpx as _httpx
    data = _load_stores()
    store = next((s for s in data["stores"] if s["id"] == store_id), None)
    if not store:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found")

    shop_url = store["shop_url"].rstrip("/")
    token = store.get("access_token", "")

    try:
        if token:
            headers = {"X-Shopify-Access-Token": token, "User-Agent": "RAG-bot/1.0"}
            with _httpx.Client(headers=headers, follow_redirects=True, timeout=10) as client:
                # Shop info
                r = client.get(f"{shop_url}/admin/api/2024-01/shop.json")
                r.raise_for_status()
                shop_name = r.json().get("shop", {}).get("name", shop_url)
                # Product count
                r2 = client.get(f"{shop_url}/admin/api/2024-01/products/count.json")
                r2.raise_for_status()
                products = r2.json().get("count", 0)
                # Page count
                r3 = client.get(f"{shop_url}/admin/api/2024-01/pages/count.json")
                r3.raise_for_status()
                pages = r3.json().get("count", 0)
            return {"ok": True, "mode": "admin", "shop_name": shop_name,
                    "products": products, "pages": pages}
        else:
            r = _httpx.get(f"{shop_url}/products.json?limit=1",
                           headers={"User-Agent": "RAG-bot/1.0"},
                           follow_redirects=True, timeout=10)
            r.raise_for_status()
            count = len(r.json().get("products", []))
            return {"ok": True, "mode": "public",
                    "products": f"available ({count} on first page)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/shopify/stores/{store_id}/fetch")
def shopify_fetch_store(store_id: str):
    """Fetch/scrape a Shopify store in the background. Watch /api/progress for updates."""
    import io, sys
    from ingestion.scrapers.runner import run_scraper

    data = _load_stores()
    store = next((s for s in data["stores"] if s["id"] == store_id), None)
    if not store:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found")

    _clear_progress_queue()

    # Capture store values now (closure-safe snapshot)
    _shop_url = store["shop_url"]
    _token = store.get("access_token", "")
    _include = store.get("include", ["products", "pages", "articles"])
    _metafields = store.get("metafields", False)
    _display_name = store.get("display_name", store_id)

    class _QueueWriter(io.TextIOBase):
        def write(self, s):
            s = s.strip()
            if s:
                _progress_queue.put(f"LOG:{s}")
            return len(s)
        def flush(self): pass

    def _run():
        from datetime import datetime, timezone
        old_stdout = sys.stdout
        sys.stdout = _QueueWriter()
        try:
            config = {
                "engine": "shopify",
                "shop_url": _shop_url,
                "access_token": _token,
                "include": _include,
                "metafields": _metafields,
            }
            _raw_text, items = run_scraper(store_id, config)
            # Update last_fetched
            d = _load_stores()
            for s in d["stores"]:
                if s["id"] == store_id:
                    s["last_fetched"] = datetime.now(timezone.utc).isoformat()
                    break
            _save_stores(d)
            _progress_queue.put(f"DONE:Fetched {len(items)} items from {_display_name}")
        except Exception as e:
            _progress_queue.put(f"ERROR:{e}")
        finally:
            sys.stdout = old_stdout

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Fetch started in background. Watch /api/progress for updates."}


# ── Site Analysis Wizard ──────────────────────────────────────────────────────

class WizardAnalyseRequest(BaseModel):
    url: str
    solution_name: str
    language: str = "en"
    login_config: Optional[dict] = None


class WizardConfirmRequest(BaseModel):
    solution_id: str
    solution_name: str
    language: str
    domain: str
    collections: list  # [{collection_name, display_name, doc_type, categories:[{id,sitemap_url,url_filter}]}]


@app.post("/api/wizard/analyse")
def wizard_analyse(req: WizardAnalyseRequest):
    """
    Start background sitemap analysis + LLM collection suggestion.
    Streams LOG: messages then DONE:{json} via /api/progress SSE.
    """
    import io, sys, json as _json
    from ingestion.scrapers.sitemap_analyzer import fetch_sitemap_structure, suggest_collections

    _clear_progress_queue()

    _url = req.url.strip()
    _lang = req.language
    _login_config = req.login_config or {}

    class _QueueWriter(io.TextIOBase):
        def write(self, s):
            s = s.strip()
            if s:
                _progress_queue.put(f"LOG:{s}")
            return len(s)
        def flush(self): pass

    def _run():
        old_stdout = sys.stdout
        sys.stdout = _QueueWriter()
        try:
            _progress_queue.put("LOG:Fetching sitemap structure…")
            if _login_config:
                _progress_queue.put("LOG:Login credentials provided — will use for page preview sampling.")
            categories = fetch_sitemap_structure(_url, login_config=_login_config or None)
            _progress_queue.put(f"LOG:Found {len(categories)} category(s). Asking LLM for collection suggestions…")
            suggestions = suggest_collections(categories)
            payload = _json.dumps({"categories": categories, "suggested_collections": suggestions},
                                  ensure_ascii=False)
            _progress_queue.put(f"DONE:{payload}")
        except Exception as e:
            _progress_queue.put(f"ERROR:{e}")
        finally:
            sys.stdout = old_stdout

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Analysis started. Watch /api/progress for updates."}


@app.post("/api/wizard/confirm")
def wizard_confirm(req: WizardConfirmRequest):
    """
    Store inline scraper configs in solutions.yaml for all confirmed collections.
    No YAML files are written — inline config is stored directly in solutions.yaml.
    If a named YAML file already exists for a collection, it takes precedence at scrape time.
    Runs synchronously (filesystem writes only).
    """
    import yaml, re as _re
    from ingestion.scrapers.sitemap_analyzer import generate_scraper_config

    # Derive solution_id from name
    solution_id = _re.sub(r"[^a-z0-9]+", "_", req.solution_name.lower()).strip("_")

    created = []
    for coll in req.collections:
        coll_name_raw = coll.get("collection_name") or coll.get("display_name", "collection")
        coll_name = _re.sub(r"[^a-z0-9]+", "_", coll_name_raw.lower()).strip("_")
        full_name = f"{solution_id}_{coll_name}"

        cat_dicts = coll.get("categories", [])

        # Generate inline config dict (no file written)
        scraper_cfg = generate_scraper_config({
            "collection_name": full_name,
            "display_name": coll.get("display_name", full_name),
            "doc_type": coll.get("doc_type", "general"),
            "categories": cat_dicts,
            "extra_pages": coll.get("extra_pages"),
        }, req.domain)

        created.append({
            "collection_name": full_name,
            "display_name": coll.get("display_name", full_name),
            "doc_type": coll.get("doc_type", "general"),
            "scraper_config": scraper_cfg,
            "page_count": sum(cat.get("url_count", 0) for cat in cat_dicts),
        })

    # Update solutions.yaml
    specs_path = _specs_file()
    with open(specs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {"solutions": []}

    solutions = data.setdefault("solutions", [])
    sol = next((s for s in solutions if s["id"] == solution_id), None)
    if sol is None:
        sol = {
            "id": solution_id,
            "display_name": req.solution_name,
            "company_name": req.solution_name,
            "language": req.language,
            "aliases": [solution_id[:4]],
            "collections": [],
        }
        solutions.append(sol)
    else:
        if req.language and not sol.get("language"):
            sol["language"] = req.language

    existing_coll_names = {c["collection_name"] for c in sol.get("collections", []) if isinstance(c, dict)}
    for c in created:
        if c["collection_name"] not in existing_coll_names:
            sol.setdefault("collections", []).append({
                "id": c["collection_name"],
                "display_name": c["display_name"],
                "collection_name": c["collection_name"],
                "collection_type": "scs",
                "scraper_name": c["collection_name"],
                "scraper_config": c["scraper_config"],
                "routing": {},
            })
        else:
            # Update scraper_config on existing entry (re-confirm after edits)
            for existing in sol.get("collections", []):
                if isinstance(existing, dict) and existing.get("collection_name") == c["collection_name"]:
                    existing["scraper_config"] = c["scraper_config"]
                    break

    with open(specs_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    from solution_specs import reload
    reload()

    return {
        "solution_id": solution_id,
        "collections_created": len(created),
        "collections": created,
        "message": f"Created {len(created)} collection(s) for solution '{solution_id}'.",
    }


@app.get("/api/wizard/sitemap-pages")
def wizard_sitemap_pages(sitemap_url: str, url_filter: str = None):
    """Fetch all URLs from a single sitemap file on demand (lazy expansion)."""
    try:
        from ingestion.scrapers.sitemap_analyzer import fetch_all_pages
        urls = fetch_all_pages(sitemap_url, url_filter=url_filter)
        return {"urls": urls, "count": len(urls)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/wizard/page-preview")
def wizard_page_preview(url: str):
    """Fetch a 500-char content preview for a single page URL (lazy, on demand)."""
    try:
        import httpx as _httpx
        from ingestion.scrapers.sitemap_analyzer import _sample_page, _HEADERS
        with _httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=10) as client:
            text = _sample_page(client, url)
        return {"url": url, "preview": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WizardSaveRequest(BaseModel):
    solution_id: str
    state: dict  # full JS wizard state serialised


class WizardChatRequest(BaseModel):
    question: str
    categories: list = []
    collections: list = []
    domain: str = ""


@app.post("/api/wizard/chat")
def wizard_chat(req: WizardChatRequest):
    """Ask a contextual question about the current wizard sitemap analysis."""
    from llms.openai_utils import openai_chat_completion

    cat_lines = [
        f"- {c.get('display_name','?')} ({c.get('url_count',0)} pages, id={c.get('id','?')}): "
        f"{(c.get('preview') or '')[:150]}"
        for c in (req.categories or [])
    ]
    coll_lines = [
        f"- {c.get('display_name','?')} (doc_type={c.get('doc_type','?')}): "
        f"covers [{', '.join(str(x) for x in c.get('sitemapIds', []))}]"
        for c in (req.collections or [])
    ]
    system_prompt = (
        "You are a RAG collection design assistant helping configure a chatbot for a website.\n"
        "You have access to the sitemap analysis of the website and the current collection setup.\n"
        "Answer the user's question concisely. If relevant, suggest a specific action.\n"
        "Format: short answer paragraph, then optionally 'Suggestions:' with 1-3 bullet points.\n"
        "Keep total response under 120 words."
    )
    context = (
        f"Website: {req.domain}\n\n"
        f"Sitemaps:\n" + "\n".join(cat_lines or ["(none)"]) + "\n\n"
        f"Collections:\n" + "\n".join(coll_lines or ["(none)"]) + "\n\n"
        f"Question: {req.question}"
    )
    try:
        answer = openai_chat_completion(system_prompt, context, model="gpt-4o-mini")
        suggestions = []
        if "Suggestions:" in answer:
            parts = answer.split("Suggestions:", 1)
            answer_text = parts[0].strip()
            for line in parts[1].strip().splitlines():
                line = line.strip().lstrip("•-–*123456789. ").strip()
                if line:
                    suggestions.append(line)
        else:
            answer_text = answer.strip()
        return {"answer": answer_text, "suggestions": suggestions}
    except Exception as e:
        return {"answer": f"Error: {e}", "suggestions": []}


@app.post("/api/wizard/save")
def wizard_save(req: WizardSaveRequest):
    """Persist the wizard UI state to .wizard_state_{solution_id}.json in project root."""
    import json as _json
    sid = req.solution_id.strip().lower().replace(" ", "_")
    if not sid:
        raise HTTPException(status_code=400, detail="solution_id required")
    path = os.path.join(_DATA_ROOT, f".wizard_state_{sid}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(req.state, f, ensure_ascii=False, indent=2)
        return {"saved": True, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/wizard/load")
def wizard_load(solution_id: str):
    """Load a previously saved wizard state."""
    import json as _json
    sid = solution_id.strip().lower().replace(" ", "_")
    path = os.path.join(_DATA_ROOT, f".wizard_state_{sid}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No saved wizard state for '{sid}'")
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = _json.load(f)
        return {"found": True, "solution_id": sid, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/wizard/list-saves")
def wizard_list_saves():
    """List all saved wizard sessions (files matching .wizard_state_*.json in project root)."""
    import glob as _glob
    pattern = os.path.join(_DATA_ROOT, ".wizard_state_*.json")
    saves = []
    for p in sorted(_glob.glob(pattern)):
        sid = os.path.basename(p)[len(".wizard_state_"):-len(".json")]
        saves.append({"solution_id": sid, "path": p,
                      "mtime": os.path.getmtime(p)})
    return {"saves": saves}


class WizardDiffRequest(BaseModel):
    url: str
    solution_id: str


@app.post("/api/wizard/diff")
def wizard_diff(req: WizardDiffRequest):
    """
    Compare current sitemap structure against a saved wizard session.
    Returns new/removed categories and per-sitemap new/removed URLs.
    Streams via /api/progress (LOG:/DONE:/ERROR:) because sitemap fetch can be slow.
    """
    import json as _json
    from ingestion.scrapers.sitemap_analyzer import fetch_sitemap_structure, fetch_all_pages

    _clear_progress_queue()

    sid = req.solution_id.strip().lower().replace(" ", "_")
    path = os.path.join(_DATA_ROOT, f".wizard_state_{sid}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No saved session for '{sid}'")

    def _run():
        try:
            _progress_queue.put("LOG:Loading saved session…")
            with open(path, "r", encoding="utf-8") as f:
                saved = _json.load(f)

            _progress_queue.put("LOG:Fetching current sitemap structure…")
            current_cats = fetch_sitemap_structure(req.url)

            saved_cats = {c["id"]: c for c in saved.get("categories", []) if not c["id"].startswith("_")}
            current_cats_map = {c["id"]: c for c in current_cats if not c.get("id", "").startswith("_")}
            saved_pages = saved.get("pages", {})  # catId → [url, ...]

            new_categories = [c for cid, c in current_cats_map.items() if cid not in saved_cats]
            removed_categories = [c for cid, c in saved_cats.items() if cid not in current_cats_map]

            # Per-sitemap URL diff — fetch all pages for sitemaps in both old and new
            changed = {}
            common_ids = set(saved_cats) & set(current_cats_map)
            for i, cid in enumerate(sorted(common_ids), 1):
                cat = current_cats_map[cid]
                _progress_queue.put(f"LOG:Checking {cat['display_name']} ({i}/{len(common_ids)})…")
                try:
                    current_urls = set(fetch_all_pages(cat["sitemap_url"], cat.get("url_filter")))
                except Exception:
                    current_urls = set()
                saved_urls = set(saved_pages.get(cid, []))
                new_urls = sorted(current_urls - saved_urls)
                removed_urls = sorted(saved_urls - current_urls)
                if new_urls or removed_urls:
                    changed[cid] = {"new_urls": new_urls, "removed_urls": removed_urls}

            total_new = sum(len(v["new_urls"]) for v in changed.values()) + len(new_categories)
            total_removed = sum(len(v["removed_urls"]) for v in changed.values()) + len(removed_categories)
            _progress_queue.put(f"LOG:Diff complete — {total_new} new, {total_removed} removed.")

            payload = _json.dumps({
                "new_categories": new_categories,
                "removed_categories": [c for c in saved_cats.values() if c["id"] not in current_cats_map],
                "changed": changed,
                "current_categories": current_cats,
                "saved_state": saved,
            }, ensure_ascii=False)
            _progress_queue.put(f"DONE:{payload}")
        except Exception as e:
            _progress_queue.put(f"ERROR:{e}")

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Diff started. Watch /api/progress for updates."}


class WizardDeletePagesRequest(BaseModel):
    collection_name: str   # full Qdrant collection name (e.g. "peixefresco_products")
    urls: list             # list of source_url strings to delete chunks for


@app.post("/api/wizard/delete-pages")
def wizard_delete_pages(req: WizardDeletePagesRequest):
    """Delete all Qdrant chunks whose source_url matches any of the given URLs."""
    from QdrantTracker import QdrantTracker
    tracker = QdrantTracker()
    deleted_urls = []
    errors = []
    for url in req.urls:
        try:
            tracker._delete_points_by_url(req.collection_name, url)
            deleted_urls.append(url)
        except Exception as e:
            errors.append({"url": url, "error": str(e)})
    return {
        "collection": req.collection_name,
        "deleted_for_urls": len(deleted_urls),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Site Analyzer endpoints
# ─────────────────────────────────────────────────────────────────────────────

import csv
import io as _io

# In-memory results store (reset each run)
_site_results: list[dict] = []


class SiteAnalyzeRequest(BaseModel):
    urls: list[str]


@app.post("/api/sites/analyze")
def sites_analyze(req: SiteAnalyzeRequest, background_tasks: BackgroundTasks):
    """Launch site analysis in background; stream progress via /api/progress SSE."""
    global _site_results
    _site_results = []

    urls = [u.strip() for u in req.urls if u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    def _run():
        import concurrent.futures
        from ingestion.scrapers.tech_detector import detect_site_sync

        total = len(urls)
        _progress_queue.put(f"LOG:Starting analysis of {total} URL(s)…\n")

        results = []

        def _analyze_one(url):
            try:
                _progress_queue.put(f"LOG:🔍 Analysing {url}…\n")
                report = detect_site_sync(url)
                _progress_queue.put(
                    f"LOG:✅ {url} → {report['platform']} | chatbot: {report['chatbot']}\n"
                )
                return report
            except Exception as e:
                _progress_queue.put(f"LOG:❌ {url} — error: {e}\n")
                return {"url": url, "error": str(e), "platform": "Error", "chatbot": ""}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_analyze_one, u): u for u in urls}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        global _site_results
        _site_results = results
        _progress_queue.put(f"DONE:{json.dumps({'count': len(results)})}")

    background_tasks.add_task(lambda: threading.Thread(target=_run, daemon=True).start())
    return {"status": "started", "count": len(urls)}


@app.get("/api/sites/results")
def sites_results():
    """Return current site analysis results as JSON."""
    return {"results": _site_results}


@app.get("/api/sites/export-csv")
def sites_export_csv():
    """Stream results as a CSV download."""
    if not _site_results:
        raise HTTPException(status_code=404, detail="No results yet. Run analysis first.")

    columns = [
        "url", "platform", "platform_confidence",
        "chatbot", "chatbot_signal", "chatbot_category",
        "cms", "ssl", "payments", "social_links", "has_blog",
        "contact_form", "contact_mailto", "contact_whatsapp", "contact_phone",
        "rank", "rank_source", "error",
    ]

    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in _site_results:
        writer.writerow({col: row.get(col, "") for col in columns})

    csv_bytes = buf.getvalue().encode("utf-8")

    from starlette.responses import Response
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=site_analysis.csv"},
    )


@app.get("/api/sites/domcop-status")
def sites_domcop_status():
    """Return DomCop local DB status (last updated, domain count, available)."""
    try:
        from ingestion.scrapers.domain_rank_db import status as _dc_status
        return _dc_status()
    except ImportError:
        return {"available": False, "domain_count": 0, "last_updated": None, "db_path": None}


@app.post("/api/sites/update-domcop")
def sites_update_domcop(background_tasks: BackgroundTasks):
    """
    Kick off a background download + ingest of the DomCop top-10M CSV.
    Progress is streamed via the shared /api/progress SSE endpoint.
    """
    def _run():
        try:
            from ingestion.scrapers.domain_rank_db import download_and_ingest
            download_and_ingest(progress_cb=lambda msg: _progress_queue.put(f"LOG:{msg}\n"))
            _progress_queue.put('DONE:{"domcop": "updated"}')
        except Exception as e:
            _progress_queue.put(f"ERROR:DomCop update failed: {e}")

    background_tasks.add_task(lambda: threading.Thread(target=_run, daemon=True).start())
    return {"status": "started"}


# ─────────────────────────────────────────────────────────────────────────────

_INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>jB RAG Builder</title>
  <script>if("__DEV_MODE__"==="1"){document.title="jB RAG Builder [DEV]"}</script>
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon_32.png" id="favicon32">
  <link rel="icon" type="image/png" sizes="16x16" href="/assets/favicon_16.png" id="favicon16">
  <link rel="apple-touch-icon" sizes="192x192" href="/assets/favicon_192.png">
  <script>if("__DEV_MODE__"==="1"){document.getElementById("favicon32").href="/assets/favicon_dev.png";document.getElementById("favicon16").href="/assets/favicon_dev.png"}</script>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 1.5rem; background: #f8f9fa; color: #1a1a1a; }
    .container { max-width: 1100px; margin: 0 auto; }
    h1 { font-size: 1.5rem; font-weight: 600; margin: 0 0 0.5rem 0; }
    .subtitle { color: #555; margin-bottom: 1.5rem; }
    .card { background: #fff; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.08); padding: 1.25rem; margin-bottom: 1.25rem; }
    .card h2 { font-size: 1.1rem; margin: 0 0 1rem 0; font-weight: 600; }
    label { display: block; font-size: 0.9rem; font-weight: 500; margin-bottom: 0.35rem; color: #333; }
    input, select, textarea { width: 100%; padding: 0.5rem 0.6rem; border: 1px solid #ccc; border-radius: 6px; font-size: 0.95rem; margin-bottom: 0.75rem; }
    input:focus, select:focus, textarea:focus { outline: none; border-color: #0066cc; }
    button { padding: 0.5rem 1rem; border: none; border-radius: 6px; font-size: 0.95rem; font-weight: 500; cursor: pointer; }
    .btn-primary { background: #0066cc; color: #fff; }
    .btn-primary:hover { background: #0052a3; }
    .btn-secondary { background: #e9ecef; color: #333; margin-left: 0.5rem; }
    .btn-secondary:hover { background: #dee2e6; }
    .btn-translate { background: #888; color: #fff; margin-left: 0.5rem; }
    .btn-translate:hover { background: #666; }
    .btn-shutdown { font-size:0.72rem; padding:0.15rem 0.55rem; background:transparent; border:1px solid #ccc; border-radius:10px; color:#aaa; cursor:pointer; transition:all 0.15s; margin-left:0; }
    .btn-shutdown:hover { background:#d32f2f !important; border-color:#d32f2f !important; color:#fff !important; }
    .btn-sync { background: #6a1b9a; color: #fff; margin-left: 0.5rem; }
    .btn-sync:hover { background: #4a148c; }
    .progress-wrap { margin-top: 0.75rem; display: none; }
    .progress-bar-bg { background: #e9ecef; border-radius: 6px; height: 10px; overflow: hidden; }
    .progress-bar-fill { background: #2e7d32; height: 10px; width: 0%; transition: width 0.3s ease; border-radius: 6px; }
    .progress-label { font-size: 0.8rem; color: #555; margin-top: 0.3rem; text-align: center; }
    .log { white-space: pre-wrap; font-size: 0.85rem; background: #f1f3f5; padding: 0.75rem; border-radius: 6px; margin-top: 0.75rem; min-height: 3rem; max-height: 12rem; overflow-y: auto; }
    .log.error { background: #ffe0e0; }
    .log.success { background: #e0f0e0; }
    .status { font-size: 0.85rem; color: #666; margin-top: 0.5rem; }
    .row { display: flex; gap: 1rem; flex-wrap: wrap; }
    .row > * { flex: 1 1 200px; }
    .tabs { display: flex; gap: 0.25rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center; }
    .tab { padding: 0.5rem 1rem; border-radius: 6px; background: #e9ecef; border: none; cursor: pointer; font-size: 0.9rem; }
    .tab.active { background: #0066cc; color: #fff; }
    .tab:hover:not(.active) { background: #dee2e6; }
    .global-sol-wrap { display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; padding: 0.5rem 0; }
    .global-sol-wrap label { color: #555; font-weight: 500; white-space: nowrap; margin: 0; }
    .global-sol-wrap select { margin: 0; padding: 0.35rem 0.5rem; font-size: 0.85rem; max-width: 200px; }
    .global-sol-wrap input[type="text"] { margin: 0; padding: 0.35rem 0.5rem; font-size: 0.85rem; min-width: 140px; }
    .global-sol-wrap .btn-sm { padding: 0.3rem 0.6rem; font-size: 0.8rem; border: none; border-radius: 5px; cursor: pointer; }
    .global-sol-lang { font-size: 0.78rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 12px; background: #e8f0fe; color: #1a56a0; white-space: nowrap; cursor: pointer; border: 1px solid #b8d0f8; }
    .hidden { display: none; }
    /* ── Shopify Stores tab ── */
    .store-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; background: #fff; }
    .badge { display: inline-block; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 10px; font-weight: 600; }
    .badge-admin { background: #d4f5ee; color: #007a5e; }
    .badge-public { background: #f0f0f0; color: #666; }
    .store-meta { color: #666; font-size: 0.85rem; margin: 0.25rem 0 0.35rem 0; }
    .inline-result { margin-top: 0.6rem; font-size: 0.875rem; padding: 0.5rem 0.75rem; border-radius: 6px; }
    .inline-result.ok { background: #d4f5ee; color: #005a44; }
    .inline-result.err { background: #fde8e8; color: #c0392b; }
    /* ── Site Analysis Wizard ── */
    .btn-wizard { background: #1a7a3a; color: #fff; }
    .btn-wizard:hover { background: #145e2c; }
    .btn-wizard-add { background: #e8f0fe; color: #1a56a0; border: 1px dashed #9ab2e0; font-size: 0.85rem; padding: 0.3rem 0.7rem; }
    .btn-wizard-add:hover { background: #d0e0ff; }
    .btn-wizard-rm { background: none; color: #c0392b; border: none; font-size: 0.8rem; padding: 0.1rem 0.3rem; cursor: pointer; opacity: 0.7; }
    .btn-wizard-rm:hover { opacity: 1; }
    .wizard-layout { display: flex; gap: 1rem; align-items: flex-start; }
    .wizard-left { flex: 1; min-width: 0; }
    .wizard-right { width: 540px; flex-shrink: 0; position: sticky; top: 1rem; align-self: flex-start; }
    .wiz-sm-row { display: flex; align-items: center; gap: 0.4rem; padding: 0.45rem 0.3rem; border-bottom: 1px solid #f0f0f0; user-select: none; border-radius: 4px; }
    .wiz-sm-row:hover { background: #f8f9fa; }
    .wiz-sm-row[draggable=true] { cursor: grab; }
    .wiz-sm-toggle { width: 1.1rem; text-align: center; font-size: 0.72rem; color: #999; flex-shrink: 0; cursor: pointer; }
    .wiz-sm-name { font-weight: 600; font-size: 0.9rem; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
    .wiz-sm-count { font-size: 0.75rem; color: #888; background: #f0f0f0; padding: 0.1rem 0.45rem; border-radius: 8px; flex-shrink: 0; }
    .wiz-sm-coll-badge { font-size: 0.75rem; padding: 0.1rem 0.5rem; border-radius: 8px; cursor: pointer; flex-shrink: 0; white-space: nowrap; border: none; background: none; }
    .wiz-sm-coll-badge.assigned { background: #dbe8ff; color: #1a4a90; }
    .wiz-sm-coll-badge.unassigned { background: #f0f0f0; color: #999; border: 1px dashed #ccc; }
    .wiz-sm-sentinel { padding: 0.7rem 0.8rem; background: #fffde7; border: 1px solid #ffe082; border-radius: 8px; color: #5d4037; font-size: 0.88rem; margin-bottom: 0.5rem; }
    .wiz-page-list { border-left: 2px solid #e0ebff; margin: 0 0 0.3rem 1.2rem; padding-left: 0.5rem; }
    .wiz-page-row { display: flex; align-items: center; gap: 0.35rem; padding: 0.22rem 0.3rem; border-radius: 4px; font-size: 0.78rem; }
    .wiz-page-row:hover { background: #f5f8ff; }
    .wiz-page-row[draggable=true] { cursor: grab; }
    .wiz-page-url { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #444; font-family: monospace; font-size: 0.75rem; text-decoration: none; }
    .wiz-page-url:hover { text-decoration: underline; color: #0066cc; }
    .wiz-page-badge { font-size: 0.7rem; padding: 0.05rem 0.38rem; border-radius: 8px; cursor: pointer; flex-shrink: 0; white-space: nowrap; border: none; }
    .wiz-page-badge.inherited { background: #e8f0fe; color: #1a4a90; }
    .wiz-page-badge.overridden { background: #e8f8e8; color: #1a7a3a; font-weight: 600; }
    .wiz-page-badge.excluded { background: #fde8e8; color: #c0392b; }
    .wiz-page-badge.review { background: #fff8e1; color: #7a5a00; }
    .wiz-page-badge.unassigned { background: #f0f0f0; color: #999; }
    .wiz-page-excl-btn { font-size: 0.7rem; cursor: pointer; padding: 0.05rem 0.3rem; background: none; border: 1px solid #e88; border-radius: 4px; color: #c0392b; flex-shrink: 0; line-height: 1.4; margin-left: 2px; }
    .wiz-page-excl-btn:hover { background: #fde8e8; }
    .wiz-sm-skip-btn { font-size: 0.72rem; cursor: pointer; padding: 0.05rem 0.35rem; background: none; border: 1px solid #e88; border-radius: 4px; color: #c0392b; flex-shrink: 0; line-height: 1.4; }
    .wiz-sm-skip-btn:hover { background: #fde8e8; }
    .wiz-sm-showall-btn { font-size: 0.75rem; cursor: pointer; padding: 0.05rem 0.35rem; background: none; border: 1px solid #cce; border-radius: 4px; color: #446; flex-shrink: 0; line-height: 1.4; }
    .wiz-sm-showall-btn:hover { background: #eef0ff; }
    .wiz-chat-bubble { padding: 0.3rem 0.55rem; border-radius: 8px; margin-bottom: 0.3rem; font-size: 0.81rem; line-height: 1.4; max-width: 92%; }
    .wiz-chat-bubble.user { background: #e8f0fe; color: #1a4a90; margin-left: auto; text-align: right; }
    .wiz-chat-bubble.bot { background: #f5f5f5; color: #333; }
    .wiz-chat-suggestion { display: inline-block; margin: 0.15rem 0.2rem 0 0; padding: 0.15rem 0.5rem; background: #fff8e1; border: 1px solid #ffe082; border-radius: 12px; font-size: 0.76rem; color: #7a5a00; cursor: pointer; }
    .wiz-chat-suggestion:hover { background: #fff3cd; }
    .wiz-eye { color: #ccc; font-size: 0.8rem; cursor: pointer; flex-shrink: 0; padding: 0 0.1rem; line-height: 1; background: none; border: none; }
    .wiz-eye:hover { color: #0066cc; }
    .wiz-load-more { font-size: 0.78rem; color: #0066cc; cursor: pointer; padding: 0.3rem 0.3rem; display: block; }
    .wiz-load-more:hover { text-decoration: underline; }
    .wiz-coll-block { padding: 0.65rem 0.75rem; margin-bottom: 0.55rem; border: 2px dashed #c8d8f0; border-radius: 8px; background: #f5f8ff; transition: border-color 0.15s, background 0.15s; }
    .wiz-coll-block.dragover { border-color: #0066cc; background: #e8f3ff; }
    .wiz-coll-top { display: flex; gap: 0.4rem; align-items: center; margin-bottom: 0.4rem; }
    .wiz-coll-top input { margin: 0; flex: 1; font-size: 0.85rem; padding: 0.28rem 0.45rem; font-weight: 600; }
    .wiz-coll-top select { margin: 0; flex: 0 0 110px; font-size: 0.75rem; padding: 0.28rem 0.3rem; color: #555; }
    .wiz-coll-body { font-size: 0.78rem; color: #555; }
    .wiz-coll-sm-entry { display: flex; align-items: center; gap: 0.25rem; margin-bottom: 0.12rem; flex-wrap: nowrap; }
    .wiz-coll-pages-panel { max-height: 260px; overflow-y: auto; border-top: 1px solid #e8eaf0; margin-top: 0.25rem; padding: 0.2rem 0; background: #fafbfd; border-radius: 0 0 6px 6px; }
    .wiz-coll-sm-dot { color: #1a4a90; font-size: 0.65rem; }
    .wiz-coll-sm-name { color: #1a4a90; font-weight: 500; }
    .wiz-coll-sm-count { color: #888; }
    .wiz-coll-sm-excl { color: #c0392b; font-size: 0.71rem; }
    .wiz-coll-extras { font-size: 0.73rem; color: #1a7a3a; margin-top: 0.2rem; padding-left: 0.6rem; }
    .wiz-coll-empty { color: #bbb; font-style: italic; font-size: 0.78rem; }
    .wiz-preview-box { font-size: 0.78rem; color: #444; background: #f8f9fa; border-radius: 6px; padding: 0.45rem 0.55rem; margin-top: 0.25rem; white-space: pre-wrap; line-height: 1.4; max-height: 90px; overflow-y: auto; border: 1px solid #e8e8e8; }
    .wiz-spinner { display: inline-block; width: 0.75rem; height: 0.75rem; border: 2px solid #ddd; border-top-color: #0066cc; border-radius: 50%; animation: wiz-spin 0.6s linear infinite; vertical-align: middle; }
    @keyframes wiz-spin { to { transform: rotate(360deg); } }
    .wiz-load-item { display: block; width: 100%; text-align: left; background: none; border: none; padding: 0.45rem 0.9rem; font-size: 0.85rem; cursor: pointer; color: #1a56a0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .wiz-load-item:hover { background: #f0f6ff; }
    .wiz-load-empty { padding: 0.5rem 0.9rem; font-size: 0.82rem; color: #aaa; font-style: italic; }
    /* ── Diff / Update mode ── */
    .wiz-diff-banner { background: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 0.5rem 0.8rem; margin-bottom: 0.65rem; font-size: 0.85rem; color: #5d4037; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
    .wiz-page-badge.new-page { background: #e8f5e9; color: #1b5e20; font-weight: 600; }
    .wiz-page-badge.removed-page { background: #fde8e8; color: #b71c1c; font-weight: 600; }
    .wiz-page-row.wiz-removed { opacity: 0.55; }
    .wiz-page-row.wiz-removed .wiz-page-url { text-decoration: line-through; color: #999; }
    .wiz-page-row.wiz-new { background: #f0fdf4; border-radius: 4px; }
    .wiz-del-btn { font-size: 0.72rem; cursor: pointer; flex-shrink: 0; padding: 0.05rem 0.3rem; background: none; border: 1px solid #e88; border-radius: 4px; color: #c0392b; line-height: 1.4; }
    .wiz-del-btn:hover { background: #fde8e8; }
    .wiz-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 500; }
    .wiz-modal-box { background: #fff; border-radius: 12px; padding: 1.5rem 2rem; max-width: 460px; width: 90%; box-shadow: 0 8px 24px rgba(0,0,0,0.2); }
    .wiz-modal-title { font-size: 1.05rem; font-weight: 600; margin: 0 0 0.4rem; }
    .wiz-modal-sub { font-size: 0.85rem; color: #666; margin: 0 0 1.2rem; }
    .wiz-modal-btns { display: flex; gap: 0.75rem; flex-wrap: wrap; }
    .wiz-modal-btn { flex: 1; padding: 0.65rem 1rem; border-radius: 8px; border: 1px solid #c8d8f0; background: #f5f8ff; font-size: 0.9rem; font-weight: 500; cursor: pointer; text-align: left; transition: background 0.12s; }
    .wiz-modal-btn:hover { background: #e8f0ff; }
    .wiz-modal-btn.primary { background: #0066cc; color: #fff; border-color: #0066cc; }
    .wiz-modal-btn.primary:hover { background: #0052a3; }
    .wiz-modal-btn-icon { font-size: 1.3rem; display: block; margin-bottom: 0.25rem; }
    .wiz-modal-btn-label { font-weight: 600; display: block; }
    .wiz-modal-btn-desc { font-size: 0.75rem; color: inherit; opacity: 0.8; display: block; margin-top: 0.1rem; }
    .wiz-modal-cancel { display: block; text-align: center; margin-top: 0.9rem; font-size: 0.85rem; color: #888; cursor: pointer; background: none; border: none; }
    .wiz-modal-cancel:hover { color: #333; text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.1rem">
      <h1 style="margin:0">RAG – Build &amp; Chat</h1>
      <script>if("__DEV_MODE__"==="1")document.currentScript.previousElementSibling.insertAdjacentHTML('beforeend',' <span style="font-size:0.55em;padding:0.15rem 0.5rem;border-radius:4px;background:#ff6b00;color:#fff;vertical-align:middle;letter-spacing:0.05em;">DEV</span>')</script>
      <span id="serverStatusBadge" style="font-size:0.8rem;font-weight:600;padding:0.2rem 0.65rem;border-radius:12px;background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7;" title="Server status">&#9679; Online</span>
    </div>
    <div class="global-sol-wrap" style="margin:0.3rem 0 0.8rem 0;">
      <label>Solution:</label>
      <select id="globalSolution" onchange="onGlobalSolutionChange()">
        <option value="">— Select a solution —</option>
      </select>
      <span id="globalSolLang" class="global-sol-lang" style="display:none;" title="Click to change base language" onclick="showLangEditor()"></span>
      <div style="position:relative;display:inline-block;">
        <button type="button" id="btnWizardLoad" class="btn-sm" onclick="_wizardShowLoadDropdown(this)" title="Load a previously saved wizard session" style="background:#f0f4fa;color:#555;border:1px solid #c8d8f0;font-size:0.78rem;cursor:pointer;">📂 Load session</button>
        <div id="wizardLoadDropdown" style="display:none;position:absolute;top:100%;left:0;z-index:200;background:#fff;border:1px solid #c8d8f0;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.12);min-width:220px;padding:0.4rem 0;margin-top:2px;"></div>
      </div>
      <input id="globalSolNewName" type="text" placeholder="New solution name…" style="display:none;">
      <button id="globalSolNewBtn" class="btn-sm" style="display:none;background:#0066cc;color:#fff;" onclick="_globalCreateNewSolution()">Create</button>
      <button id="globalSolNewCancel" class="btn-sm" style="display:none;background:#e9ecef;color:#333;" onclick="_globalCancelNewSolution()">Cancel</button>
    </div>

    <div class="tabs">
      <button type="button" class="tab active" data-tab="wizard">🔍 Analyse Site</button>
      <button type="button" class="tab" data-tab="build">🛠 Work with RAG</button>
      <button type="button" class="tab" data-tab="chat">Chat / Test Q&A</button>
      <button type="button" class="tab" data-tab="shopify">🛍 Shopify Stores</button>
      <button type="button" class="tab" data-tab="sites">🕵️ Prospect Sites</button>
      <button type="button" class="tab" data-tab="settings" onclick="showTab('settings')">⚙️ Settings</button>
    </div>

    <div id="panel-build" class="panel hidden">
      <div class="card">
        <h2>1. Collection</h2>

        <!-- Inline language editor (hidden by default) -->
        <div id="solLangEditor" style="display:none;background:#f8faff;border:1px solid #c0d4f0;border-radius:8px;padding:0.6rem 0.75rem;margin-bottom:0.5rem;font-size:0.875rem;">
          <strong style="font-size:0.8rem;color:#444;">Base language</strong>
          <span style="color:#888;font-size:0.78rem;"> — All routing metadata will be generated in this language.</span>
          <div style="display:flex;gap:0.5rem;align-items:center;margin-top:0.4rem;">
            <select id="solLangInput" style="margin:0;font-size:0.9rem;padding:0.35rem 0.5rem;flex:1;">
              <option value="en">English (en)</option>
              <option value="pt">Portuguese (pt)</option>
              <option value="es">Spanish (es)</option>
              <option value="bg">Bulgarian (bg)</option>
              <option value="hr">Croatian (hr)</option>
              <option value="cs">Czech (cs)</option>
              <option value="da">Danish (da)</option>
              <option value="nl">Dutch (nl)</option>
              <option value="et">Estonian (et)</option>
              <option value="fi">Finnish (fi)</option>
              <option value="fr">French (fr)</option>
              <option value="de">German (de)</option>
              <option value="el">Greek (el)</option>
              <option value="hu">Hungarian (hu)</option>
              <option value="ga">Irish (ga)</option>
              <option value="it">Italian (it)</option>
              <option value="lv">Latvian (lv)</option>
              <option value="lt">Lithuanian (lt)</option>
              <option value="mt">Maltese (mt)</option>
              <option value="no">Norwegian (no)</option>
              <option value="pl">Polish (pl)</option>
              <option value="ro">Romanian (ro)</option>
              <option value="sk">Slovak (sk)</option>
              <option value="sl">Slovenian (sl)</option>
              <option value="sv">Swedish (sv)</option>
            </select>
            <button type="button" class="btn-primary" style="padding:0.35rem 0.75rem;font-size:0.85rem;" onclick="saveSolLanguage()">Save</button>
            <button type="button" class="btn-secondary" style="padding:0.35rem 0.6rem;font-size:0.85rem;" onclick="hideLangEditor()">Cancel</button>
          </div>
        </div>

        <!-- Shown when a solution IS selected: pick which collection within it -->
        <div id="collectionSection" style="display:none;margin-top:0.75rem;">
          <label>Collection <span style="font-weight:400;color:#888;font-size:0.85rem;">— which index to build or update</span></label>
          <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.4rem;">
            <select id="collectionSelect" onchange="onCollectionSelect()" style="margin-bottom:0;flex:1;"></select>
            <button type="button" id="btnDeleteCollection" onclick="deleteCollection(this)" title="Delete this collection from Qdrant"
              style="padding:0.4rem 0.6rem;background:#d32f2f;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:0.9rem;display:none;">🗑</button>
          </div>
          <div id="newCollectionRow" style="display:none;margin-top:0.4rem;">
            <input type="text" id="collectionName" placeholder="e.g. peixefresco_v2" style="margin-bottom:0.25rem;">
            <button type="button" class="btn-secondary" style="font-size:0.85rem;padding:0.3rem 0.75rem;" onclick="registerCollection()">Register in solution</button>
          </div>
          <div id="existingCollectionInfo" style="font-size:0.82rem;color:#555;margin-top:0.25rem;display:none;"></div>
        </div>

        <!-- Sub-collection picker: shown when solution has multiple collections -->
        <div id="subCollectionPicker" style="margin-top:0.5rem;"></div>
        <!-- Routing metadata panel: shown when a specific sub-collection is selected -->
        <div id="routingMetadataPanel" style="display:none;"></div>

        <!-- Shown when NO solution is selected -->
        <div id="noSolutionRow" style="margin-top:0.5rem;">
          <p class="status" style="margin:0;color:#888;">Select a solution from the top bar to see its collections.</p>
        </div>
      </div>
      <!-- 2. Sources list -->
      <div class="card" id="sourcesCard">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
          <h2 style="margin:0;">2. Sources</h2>
          <button type="button" class="btn-secondary" id="btnAddSource" onclick="showAddSourceForm()" style="font-size:0.82rem;padding:0.25rem 0.65rem;">+ Add source</button>
        </div>
        <p style="font-size:0.82rem;color:#888;margin:0 0 0.5rem;">Each source is fetched and chunked independently, then pushed to the same Qdrant collection.</p>
        <div id="sourcesList"></div>
        <!-- Inline add-source form (hidden by default) -->
        <div id="addSourceForm" style="display:none;margin-top:0.75rem;padding:0.75rem;background:#f8f9fa;border:1px solid #e0e0e0;border-radius:8px;">
          <div style="font-weight:600;font-size:0.88rem;margin-bottom:0.5rem;">New source</div>
          <div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">
            <select id="newSourceType" style="padding:0.3rem 0.5rem;font-size:0.85rem;border-radius:5px;border:1px solid #ccc;">
              <option value="url">Website (URL)</option>
              <option value="pdf">PDF file</option>
              <option value="txt">Text file</option>
              <option value="csv">CSV file</option>
            </select>
            <input type="text" id="newSourceLabel" placeholder="Label (e.g. Product sitemap)" style="flex:1;min-width:180px;padding:0.3rem 0.5rem;font-size:0.85rem;border-radius:5px;border:1px solid #ccc;">
            <input type="text" id="newSourceScraper" placeholder="Scraper name (for URL)" style="min-width:150px;padding:0.3rem 0.5rem;font-size:0.85rem;border-radius:5px;border:1px solid #ccc;">
            <button type="button" class="btn-primary" onclick="addSource()" style="font-size:0.82rem;padding:0.3rem 0.7rem;">Add</button>
            <button type="button" class="btn-secondary" onclick="hideAddSourceForm()" style="font-size:0.82rem;padding:0.3rem 0.5rem;">Cancel</button>
          </div>
        </div>
      </div>

      <!-- 3. Source config (shown when a source is selected) -->
      <div class="card" id="sourceConfigCard" style="display:none;">
        <h2>3. Source config <span id="sourceConfigLabel" style="font-weight:400;color:#888;font-size:0.85rem;"></span></h2>
        <label>Source type</label>
        <select id="sourceType" onchange="onSourceTypeChange()">
          <option value="pdf">PDF file</option>
          <option value="txt">Text file</option>
          <option value="csv">CSV file</option>
          <option value="url">Website (scraper)</option>
        </select>
        <div id="filePickerRow">
          <label>File</label>
          <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.2rem">
            <div id="sourcePathDisplay" onclick="browseFile()" title="Click to browse"
              style="flex:1;padding:0.5rem 0.75rem;border:1px solid #ccc;border-radius:6px;font-size:0.95rem;background:#fff;cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#888;min-width:0;user-select:none;"
            >Click to select a file…</div>
            <input type="hidden" id="sourcePath">
            <button type="button" class="btn-secondary" id="btnBrowse" onclick="browseFile()" style="margin-left:0;white-space:nowrap;flex-shrink:0;">📂 Browse…</button>
          </div>
          <div id="sourcePathFull" style="font-size:0.75rem;color:#aaa;margin-bottom:0.3rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:none;" title=""></div>
          <div id="recentFiles" style="display:none;margin-bottom:0.6rem;"></div>
          <div id="resumeBanner" style="display:none;background:#e8f5e9;border:1px solid #a5d6a7;border-radius:6px;padding:0.6rem 0.9rem;margin-bottom:0.75rem;font-size:0.9rem;">
            <strong>💾 Saved state found!</strong> <span id="resumeInfo"></span>
            <button type="button" onclick="resumeState()" style="margin-left:0.75rem;padding:0.25rem 0.75rem;background:#2e7d32;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:0.85rem;">Resume →</button>
            <button type="button" onclick="dismissResume()" style="margin-left:0.4rem;padding:0.25rem 0.5rem;background:#e9ecef;color:#333;border:none;border-radius:4px;cursor:pointer;font-size:0.85rem;">Start fresh</button>
          </div>
        </div>
        <div id="scraperRow" class="hidden">
          <label>Scraper name</label>
          <input type="text" id="scraperName" placeholder="e.g. peixefresco_products">
            <div style="margin-top:0.4rem;">
              <button type="button" id="loginToggleBtn" onclick="toggleLoginSection()" style="font-size:0.79rem;background:none;border:1px dashed #bbb;border-radius:5px;padding:0.18rem 0.5rem;color:#666;cursor:pointer;">🔒 Site requires login</button>
            </div>
            <div id="loginSection" style="display:none;margin-top:0.45rem;background:#fff8f0;border:1px solid #ffe0b2;border-radius:6px;padding:0.6rem 0.75rem;font-size:0.82rem;">
              <div style="font-weight:600;margin-bottom:0.35rem;color:#7a4a00;">Login credentials <span style="font-weight:400;font-size:0.76rem;color:#999;">(session only — never saved to disk)</span></div>
              <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.3rem;">
                <input id="loginUsername" type="text" placeholder="Username / email" style="flex:1;min-width:140px;padding:0.28rem 0.45rem;border:1px solid #e0b080;border-radius:5px;" />
                <input id="loginPassword" type="password" placeholder="Password" style="flex:1;min-width:140px;padding:0.28rem 0.45rem;border:1px solid #e0b080;border-radius:5px;" />
              </div>
              <div style="color:#aaa;font-size:0.75rem;margin-bottom:0.2rem;">Optional — CSS selectors (leave blank for auto-detect):</div>
              <div style="display:flex;gap:0.4rem;flex-wrap:wrap;">
                <input id="loginUrl" type="text" placeholder="Login page URL" style="flex:1.5;min-width:180px;padding:0.22rem 0.4rem;border:1px solid #ddd;border-radius:5px;font-size:0.77rem;" />
                <input id="loginUserSel" type="text" placeholder="Username selector" style="flex:1;min-width:140px;padding:0.22rem 0.4rem;border:1px solid #ddd;border-radius:5px;font-size:0.77rem;" />
                <input id="loginPassSel" type="text" placeholder="Password selector" style="flex:1;min-width:140px;padding:0.22rem 0.4rem;border:1px solid #ddd;border-radius:5px;font-size:0.77rem;" />
                <input id="loginSubmitSel" type="text" placeholder="Submit selector" style="flex:1;min-width:140px;padding:0.22rem 0.4rem;border:1px solid #ddd;border-radius:5px;font-size:0.77rem;" />
              </div>
            </div>
          <label style="margin-top:0.6rem;">Scraping engine</label>
          <div style="display:flex;flex-direction:column;gap:0.3rem;font-size:0.9rem;margin-top:0.2rem;">
            <label style="margin:0;font-weight:400;cursor:pointer;display:flex;align-items:flex-start;gap:0.5rem;">
              <input type="radio" name="scraperEngine" value="playwright" checked onchange="onScraperEngineChange()" style="width:auto;margin-top:2px;">
              <span><strong>Playwright</strong> <span style="color:#888;font-size:0.82rem;">— default. Handles JS, dynamic content, SPAs.</span></span>
            </label>
            <label style="margin:0;font-weight:400;cursor:pointer;display:flex;align-items:flex-start;gap:0.5rem;">
              <input type="radio" name="scraperEngine" value="httpx" onchange="onScraperEngineChange()" style="width:auto;margin-top:2px;">
              <span><strong>httpx (fast)</strong> <span style="color:#888;font-size:0.82rem;">— SSR-only sites. ~10x faster, no browser needed.</span></span>
            </label>
            <label style="margin:0;font-weight:400;cursor:pointer;display:flex;align-items:flex-start;gap:0.5rem;">
              <input type="radio" name="scraperEngine" value="shopify" onchange="onScraperEngineChange()" style="width:auto;margin-top:2px;">
              <span><strong>Shopify API</strong> <span style="color:#888;font-size:0.82rem;">— Shopify stores. Uses /products.json directly.</span></span>
            </label>
          </div>
          <div id="shopifyUrlRow" class="hidden" style="margin-top:0.5rem;">
            <label>Shop URL</label>
            <input type="text" id="shopUrl" placeholder="https://mystore.myshopify.com">
          </div>
          <div id="urlResumeBanner" style="display:none;background:#e8f5e9;border:1px solid #a5d6a7;border-radius:6px;padding:0.6rem 0.9rem;margin-top:0.75rem;font-size:0.9rem;">
            <strong>💾 Saved state found!</strong> <span id="urlResumeInfo"></span>
            <button type="button" onclick="resumeUrlState()" style="margin-left:0.75rem;padding:0.25rem 0.75rem;background:#2e7d32;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:0.85rem;">Resume →</button>
            <button type="button" onclick="dismissUrlResume()" style="margin-left:0.4rem;padding:0.25rem 0.5rem;background:#e9ecef;color:#333;border:none;border-radius:4px;cursor:pointer;font-size:0.85rem;">Start fresh</button>
          </div>
        </div>
      </div>

      <!-- 4. Run pipeline (shown when a source is selected) -->
      <div class="card" id="pipelineCard" style="display:none;">
        <h2>4. Run pipeline</h2>
        <p class="status">Create collection → Fetch → (Translate &amp; Clean) → Chunk → Push to Qdrant</p>
        <div style="margin-bottom:0.85rem;">
          <label title="The OpenAI embedding model used to vectorize text chunks. The vector dimension is fixed at collection creation and cannot be changed afterwards. Default: text-embedding-ada-002 (1536 dims).">
            Embedding model
          </label>
          <select id="embeddingModel"
            title="text-embedding-ada-002: original model, 1536 dims — safest default, widest compatibility. text-embedding-3-small: newer, cheaper, same 1536 dims — good drop-in upgrade. text-embedding-3-large: highest quality, 3072 dims — NOT compatible with ada-002 or 3-small collections.">
            <option value="text-embedding-ada-002" selected>text-embedding-ada-002 — default · 1536 dims</option>
            <option value="text-embedding-3-small">text-embedding-3-small — newer/cheaper · 1536 dims</option>
            <option value="text-embedding-3-large">text-embedding-3-large — best quality · 3072 dims</option>
          </select>
          <p style="font-size:0.78rem;color:#888;margin:0.2rem 0 0 0;">
            ⚠️ Locked at collection creation — all sources pushed to the same collection must use the same model.
          </p>
        </div>
        <button type="button" class="btn-primary" id="runCreate">1. Create collection</button>
        <button type="button" class="btn-primary" id="runFetch">2. Fetch</button>
        <button type="button" class="btn-translate" id="runTranslate">2b. Translate &amp; Clean (PT) 🇵🇹</button>
        <div class="progress-wrap" id="translateProgress">
          <div class="progress-bar-bg"><div class="progress-bar-fill" id="translateBar"></div></div>
          <div class="progress-label" id="translateLabel">Starting…</div>
        </div>
        <div style="margin-top:0.75rem;margin-bottom:0.5rem;">
          <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:0.5rem;">
            <button type="button" class="btn-primary" id="runChunk">3. Chunk</button>
            <span style="font-size:0.9rem;font-weight:500;color:#333;">Mode:</span>
          </div>
          <div style="display:flex;flex-direction:column;gap:0.35rem;font-size:0.9rem;padding-left:0.25rem;">
            <label style="margin:0;font-weight:400;cursor:pointer;display:flex;align-items:flex-start;gap:0.5rem;">
              <input type="radio" name="chunkMode" value="simple" checked style="margin-top:0.2rem;width:auto;">
              <span><strong>Simple</strong> <span style="color:#888;font-size:0.82rem;">— fast, free. Splits by character count. Best for most cases.</span></span>
            </label>
            <label style="margin:0;font-weight:400;cursor:pointer;display:flex;align-items:flex-start;gap:0.5rem;">
              <input type="radio" name="chunkMode" value="hierarchical" style="margin-top:0.2rem;width:auto;">
              <span><strong>Hierarchical</strong> <span style="color:#888;font-size:0.82rem;">— free, best quality. Large parent context + small child passages. Great for structured docs (recipes, manuals).</span></span>
            </label>
            <label style="margin:0;font-weight:400;cursor:pointer;display:flex;align-items:flex-start;gap:0.5rem;">
              <input type="radio" name="chunkMode" value="proposition" style="margin-top:0.2rem;width:auto;">
              <span><strong>Proposition</strong> <span style="color:#888;font-size:0.82rem;">— slow, costs $. LLM rewrites each chunk into atomic facts. Best for dense academic text.</span></span>
            </label>
          </div>
        </div>
        <button type="button" class="btn-primary" id="runPush">4. Push to Qdrant</button>
        <button type="button" class="btn-sync hidden" id="runSync">5. Sync (check for changes)</button>
        <button type="button" class="btn-secondary" id="runReset">Reset state</button>
        <div id="syncResult" class="hidden" style="margin-top:0.6rem;padding:0.6rem 0.9rem;border-radius:8px;font-size:0.88rem;font-family:monospace;"></div>
        <div id="buildLog" class="log"></div>
      </div>
    </div>

    <div id="panel-chat" class="panel hidden">
      <div class="card">
        <h2>Chat with a solution</h2>
        <div id="chatNoSolutionMsg" style="margin-bottom:0.6rem;">
          <p class="status" style="margin:0;color:#888;">Select a solution from the top bar to start chatting.</p>
        </div>
        <div id="chatCollectionRow" style="display:none;margin-top:0.6rem;">
          <label>Collection</label>
          <select id="chatCollectionSelect" style="margin-bottom:0.75rem;"></select>
        </div>
        <label title="Must match the embedding model used when the collection was built. Default: text-embedding-ada-002.">
          Embedding model
        </label>
        <select id="chatEmbeddingModel"
          title="Select the embedding model that was used to ingest this collection. Using a different model will return incorrect results. text-embedding-ada-002 is the default (1536 dims). text-embedding-3-small is 1536 dims. text-embedding-3-large is 3072 dims.">
          <option value="text-embedding-ada-002" selected>text-embedding-ada-002 — default · 1536 dims</option>
          <option value="text-embedding-3-small">text-embedding-3-small — newer/cheaper · 1536 dims</option>
          <option value="text-embedding-3-large">text-embedding-3-large — best quality · 3072 dims</option>
        </select>
        <label>Question</label>
        <textarea id="qaQuestion" rows="2" placeholder="Ask something about the content…"></textarea>
        <button type="button" class="btn-primary" id="runQA">Ask</button>
        <div id="qaResult" class="log"></div>
        <!-- FAQ Table Generation -->
        <div id="faqTableSection" style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid #eee;">
          <div style="font-weight:600;font-size:0.9rem;margin-bottom:0.6rem;">📋 FAQ Table Generator</div>
          <p style="font-size:0.82rem;color:#666;margin:0 0 0.75rem;">Generate a tab-separated Q&amp;A table from any FAQ collection — paste directly into jBKE Knowledge Editor.</p>
          <div id="faqCollList" style="font-size:0.83rem;color:#888;font-style:italic;">Loading FAQ collections…</div>
        </div>

      </div>
    </div>

    <div id="panel-shopify" class="panel hidden">
      <!-- Header -->
      <div class="card" style="display:flex;justify-content:space-between;align-items:center;padding:1rem 1.25rem">
        <div>
          <h2 style="margin:0">Shopify Stores</h2>
          <span class="status">Configs stored locally in <code>.shopify_stores.json</code> (not committed to git)</span>
        </div>
        <button class="btn-primary" onclick="showAddStoreForm()">+ Add Store</button>
      </div>

      <!-- Add/Edit form (hidden by default) -->
      <div class="card hidden" id="shopify-form-card">
        <h2 id="shopify-form-title">Add Store</h2>
        <label>Display name</label>
        <input id="shopify-form-display-name" type="text" placeholder="My Shopify Store">
        <label>Shop URL</label>
        <input id="shopify-form-url" type="text" placeholder="https://mystore.myshopify.com">
        <label title="Admin API access token (shpat_...). Leave blank to use the public /products.json API — no token required. Token is stored only in .shopify_stores.json on this machine and is never committed to git.">
          Access token <span class="status">(optional — enables Admin API: pages, articles, metafields)</span>
        </label>
        <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.75rem">
          <input id="shopify-form-token" type="password" placeholder="shpat_... (leave blank for public /products.json API)" style="margin:0;flex:1">
          <button type="button" class="btn-secondary" style="white-space:nowrap;margin:0"
            onclick="const f=document.getElementById('shopify-form-token');f.type=f.type==='password'?'text':'password'">👁</button>
        </div>
        <label title="Select which Shopify resource types to include when fetching. Requires Admin API token for pages and articles.">Include</label>
        <div style="display:flex;gap:1.25rem;margin-bottom:0.75rem">
          <label style="font-weight:400;display:flex;align-items:center;gap:0.35rem">
            <input type="checkbox" id="shopify-include-products" checked> Products
          </label>
          <label style="font-weight:400;display:flex;align-items:center;gap:0.35rem">
            <input type="checkbox" id="shopify-include-pages" checked> Pages
          </label>
          <label style="font-weight:400;display:flex;align-items:center;gap:0.35rem">
            <input type="checkbox" id="shopify-include-articles" checked> Articles
          </label>
        </div>
        <label style="display:flex;align-items:center;gap:0.5rem;font-weight:400;margin-bottom:0.75rem"
          title="Fetch metafields per product via an extra API call each. Requires Admin API token. Useful for stores with rich product metadata (ingredients, specs, certifications, etc.). Can be slow on large catalogs.">
          <input type="checkbox" id="shopify-metafields">
          Fetch metafields per product <span class="status">(requires Admin API · adds 1 API call per product)</span>
        </label>
        <div style="display:flex;gap:0.5rem">
          <button class="btn-primary" onclick="saveStore()">Save</button>
          <button class="btn-secondary" onclick="cancelStoreForm()">Cancel</button>
        </div>
      </div>

      <!-- Store list (populated by JS) -->
      <div id="shopify-store-list">
        <p class="status">Loading…</p>
      </div>
    </div>

    <!-- ── Wizard panel ── -->
    <div id="panel-wizard" class="panel">

      <!-- Input card -->
      <div class="card">
        <h2>🔍 Analyse Site</h2>
        <p class="status" style="margin-top:0">Enter a URL to discover the site&apos;s sitemap structure, then drag sitemaps or individual pages into collections.</p>
        <label>Website URL</label>
        <input id="wizardUrl" type="text" placeholder="https://example.com" style="margin-bottom:0.35rem;">
        <div style="margin-bottom:0.5rem;">
          <button type="button" onclick="toggleWizardLogin()" style="font-size:0.8rem;background:none;border:1px dashed #bbb;border-radius:5px;padding:0.2rem 0.55rem;color:#666;cursor:pointer;">🔒 Site requires login</button>
        </div>
        <div id="wizardLoginSection" style="display:none;margin-bottom:0.5rem;background:#fff8f0;border:1px solid #ffe0b2;border-radius:6px;padding:0.6rem 0.75rem;font-size:0.83rem;">
          <div style="font-weight:600;margin-bottom:0.4rem;color:#7a4a00;">Login credentials (session only — never saved to disk)</div>
          <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
            <input id="wizardLoginUsername" type="text" placeholder="Username / email" style="flex:1;min-width:140px;padding:0.3rem 0.5rem;border:1px solid #e0b080;border-radius:5px;">
            <input id="wizardLoginPassword" type="password" placeholder="Password" style="flex:1;min-width:140px;padding:0.3rem 0.5rem;border:1px solid #e0b080;border-radius:5px;">
          </div>
          <div style="margin-top:0.3rem;color:#999;font-size:0.76rem;">Login URL (if different from site URL):</div>
          <input id="wizardLoginUrl" type="text" placeholder="https://example.com/login" style="width:100%;box-sizing:border-box;margin-top:0.2rem;padding:0.25rem 0.45rem;border:1px solid #ddd;border-radius:5px;font-size:0.78rem;">
        </div>
        <!-- Hidden: synced from global solution selector -->
        <input id="wizardSolName" type="hidden">
        <datalist id="wizardSolNameList"></datalist>
        <div id="wizardNoSolutionMsg" style="margin-top:0.5rem;margin-bottom:0.5rem;display:none;">
          <p class="status" style="margin:0;color:#d47200;">Select a solution from the top bar (or create a new one) before analysing.</p>
        </div>
        <label>Language</label>
        <select id="wizardLang" style="margin-bottom:0.75rem;">
          <option value="en">English (en)</option>
          <option value="pt">Portuguese (pt)</option>
          <option value="es">Spanish (es)</option>
          <option value="bg">Bulgarian (bg)</option>
          <option value="hr">Croatian (hr)</option>
          <option value="cs">Czech (cs)</option>
          <option value="da">Danish (da)</option>
          <option value="nl">Dutch (nl)</option>
          <option value="et">Estonian (et)</option>
          <option value="fi">Finnish (fi)</option>
          <option value="fr">French (fr)</option>
          <option value="de">German (de)</option>
          <option value="el">Greek (el)</option>
          <option value="hu">Hungarian (hu)</option>
          <option value="ga">Irish (ga)</option>
          <option value="it">Italian (it)</option>
          <option value="lv">Latvian (lv)</option>
          <option value="lt">Lithuanian (lt)</option>
          <option value="mt">Maltese (mt)</option>
          <option value="pl">Polish (pl)</option>
          <option value="ro">Romanian (ro)</option>
          <option value="sk">Slovak (sk)</option>
          <option value="sl">Slovenian (sl)</option>
          <option value="sv">Swedish (sv)</option>
        </select>
        <div id="wizardModeBar" style="display:none;align-items:center;gap:0.5rem;margin-bottom:0.5rem;background:#f0f6ff;border:1px solid #c8d8f0;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.88rem;">
          <span id="wizardModeLabel" style="flex:1;font-weight:500;"></span>
          <button type="button" class="btn-wizard-add" onclick="runWizardAnalyse()" style="font-size:0.78rem;padding:0.2rem 0.5rem;">Change</button>
          <button type="button" class="btn-wizard" onclick="_wizardLaunch()" style="padding:0.35rem 0.9rem;">🚀 Launch Analyse Site</button>
        </div>
        <div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">
          <button type="button" class="btn-wizard" onclick="runWizardAnalyse()">🔍 Select Analyse Mode</button>
          <button type="button" id="btnWizardSave" class="btn-wizard-add" onclick="wizardSaveSession()" style="display:none;" title="Save current wizard session to disk">💾 Save session</button>
        </div>
        <div id="wizardLog" class="log hidden" style="margin-top:0.75rem;max-height:8rem;"></div>
      </div>

      <!-- Two-column results (shown after analyse) -->
      <div id="wizardResults" style="display:none;" class="wizard-layout">

        <!-- LEFT: Sitemaps + pages -->
        <div class="wizard-left">
          <div class="card" style="padding:1rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
              <h2 style="margin:0;">Sitemaps</h2>
              <input id="wizardSearch" type="search" placeholder="Search pages…" style="width:160px;margin:0;padding:0.28rem 0.5rem;font-size:0.82rem;" oninput="_wizardOnSearch(this.value)">
            </div>
            <p class="status" style="margin:0 0 0.6rem;">Drag a sitemap or page to a collection on the right. Click ▶ to expand. Click 👁 to preview page content.</p>
            <div id="wizardDiffBanner" style="display:none;" class="wiz-diff-banner"></div>
            <div id="wizardSitemapList"></div>
          </div>
        </div>

        <!-- RIGHT: Collections (sticky) -->
        <div class="wizard-right">
          <div class="card" style="padding:1rem;margin-bottom:0.75rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
              <h2 style="margin:0;font-size:1rem;">Collections</h2>
              <button class="btn-wizard-add" onclick="wizardAddCollection()">+ New</button>
            </div>
            <div id="wizardCollList"></div>
          </div>

          <!-- Card: Chat -->
          <div class="card" style="padding:1rem;margin-bottom:0.75rem;">
            <div id="wizardChatHint" style="font-size:0.78rem;color:#bbb;font-style:italic;">💬 Chat with the wizard — run analysis above to activate</div>
            <div id="wizardChatSection" style="display:none;">
              <div style="font-weight:600;font-size:0.85rem;color:#444;margin-bottom:0.4rem;">💬 Ask about this site</div>
              <div id="wizardChatHistory" style="max-height:200px;overflow-y:auto;margin-bottom:0.5rem;"></div>
              <div style="display:flex;gap:0.4rem;align-items:center;">
                <input id="wizardChatInput" type="text" placeholder="e.g. Are product-cat pages covered in the page sitemap?"
                  style="flex:1;font-size:0.83rem;padding:0.35rem 0.55rem;border:1px solid #c8d8f0;border-radius:6px;"
                  onkeydown="if(event.key==='Enter'){wizardAskChat();}" />
                <button onclick="wizardAskChat()" class="btn-wizard" style="font-size:0.82rem;padding:0.35rem 0.7rem;white-space:nowrap;">Ask</button>
              </div>
            </div>
          </div>

          <!-- Card: Confirm -->
          <div class="card" style="padding:1rem;">
            <button type="button" class="btn-wizard" style="width:100%;" onclick="runWizardConfirm()">✅ Confirm &amp; Create</button>
            <div id="wizardConfirmLog" class="log hidden" style="margin-top:0.6rem;max-height:8rem;overflow-y:auto;"></div>
            <div id="wizardConfirmResult" style="margin-top:0.5rem;"></div>
            <div id="wizardAutoSaveStatus" style="display:none;font-size:0.75rem;color:#888;margin-top:0.5rem;text-align:right;"></div>
          </div>
        </div>

      </div>

    </div>
    <!-- ── end Wizard panel ── -->

    <!-- ── Prospect Sites panel ── -->
    <div id="panel-sites" class="panel hidden">

      <!-- DomCop local DB status card -->
      <div class="card" id="domcopCard" style="border-left:3px solid #0066cc;">
        <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
          <div style="flex:1;min-width:220px;">
            <span style="font-weight:600;font-size:0.9rem;">🗄 Domain Rank Database</span>
            &nbsp;
            <span id="domcopBadge" style="font-size:0.78rem;padding:2px 7px;border-radius:10px;
              background:#eee;color:#666;">checking…</span>
          </div>
          <div style="font-size:0.82rem;color:#555;" id="domcopInfo">—</div>
          <button type="button" class="btn-secondary" id="btnDomcopUpdate"
            onclick="updateDomcop()" style="font-size:0.82rem;padding:0.3rem 0.75rem;">
            ⬇ Update now
          </button>
        </div>
        <p style="font-size:0.78rem;color:#999;margin:0.5rem 0 0 0;">
          <a href="https://www.domcop.com/top-10-million-websites" target="_blank" rel="noopener">DomCop top 10M</a>
          is used as a fallback when a site is too small for
          <a href="https://tranco-list.eu" target="_blank" rel="noopener">Tranco</a> (top ~1M).
          The database is stored locally and must be downloaded once (~100 MB).
          Updates take 1–3 minutes.
        </p>
        <div id="domcopLog" class="log hidden" style="margin-top:0.5rem;max-height:120px;font-size:0.78rem;"></div>
      </div>

      <!-- Input card -->
      <div class="card">
        <h2 style="margin-top:0">🕵️ Prospect Sites Analyzer</h2>
        <p style="color:#666;font-size:0.9rem;margin-top:0;">
          Paste one URL per line. Detects ecommerce platform, chatbot provider, CMS, SSL,
          payment signals, social links and global rank (Tranco → DomCop fallback).
        </p>
        <label style="font-weight:600;">URLs to analyze</label>
        <textarea id="sitesUrlInput" rows="8"
          style="width:100%;font-family:monospace;font-size:0.85rem;box-sizing:border-box;"
          placeholder="https://store1.com&#10;https://store2.com&#10;https://store3.com"></textarea>

        <div style="display:flex;gap:0.6rem;margin-top:0.6rem;align-items:center;flex-wrap:wrap;">
          <button type="button" class="btn-primary" onclick="runSiteAnalyze()">🔍 Analyze</button>
          <button type="button" class="btn-secondary" onclick="exportSitesCsv()"
            id="btnSitesCsv" style="display:none;">⬇ Export CSV</button>
          <span id="sitesStatus" style="font-size:0.85rem;color:#666;margin-left:auto;"></span>
        </div>

        <p style="font-size:0.8rem;color:#999;margin-top:0.5rem;">
          Traffic column: <strong>Tranco</strong> rank (live, top ~1M) falls back to <strong>DomCop</strong>
          (local, top 10M) for smaller sites. <strong>SW↗</strong> / <strong>AH↗</strong> open
          SimilarWeb &amp; Ahrefs for deeper research.
        </p>
      </div>

      <!-- Progress log -->
      <div id="sitesLog" class="log hidden" style="margin-top:0.75rem;"></div>

      <!-- Results table -->
      <div id="sitesTableWrap" style="display:none;margin-top:1rem;overflow-x:auto;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
          <span id="sitesTableCount" style="font-size:0.85rem;color:#666;"></span>
          <input type="text" id="sitesFilter" placeholder="Filter results…"
            oninput="_filterSitesTable()"
            style="width:220px;font-size:0.85rem;padding:0.3rem 0.6rem;">
        </div>
        <table id="sitesTable"
          style="width:100%;border-collapse:collapse;font-size:0.82rem;min-width:900px;">
          <thead>
            <tr style="background:#f0f4f8;text-align:left;position:sticky;top:0;">
              <th style="padding:7px 10px;border-bottom:2px solid #dde;">URL</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;">Platform</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;">Chatbot</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;">CMS</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;">SSL</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;" title="Global rank: Tranco (top ~1M, live) → DomCop (top 10M, local) + SW/AH research links">Traffic</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;">Payments</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;">Social</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;">Blog</th>
              <th style="padding:7px 8px;border-bottom:2px solid #dde;" title="Contact channels: embedded form, mailto, WhatsApp, phone">Contact</th>
            </tr>
          </thead>
          <tbody id="sitesTableBody"></tbody>
        </table>
      </div>
    </div>
    <!-- ── end Prospect Sites panel ── -->

    <!-- ── Settings panel ── -->
    <div id="panel-settings" class="panel hidden">
      <div style="max-width:720px;margin:0 auto;">

        <div class="card" style="margin-bottom:1rem;">
          <h2 style="margin:0 0 0.75rem;">Collection Types (doc_types)</h2>
          <p class="status" style="margin:0 0 0.75rem;">These types appear in the collection type dropdown in the Analyse Site wizard and are used for routing. The <code>faq</code> type also enables the "Build FAQ Table" button.</p>
          <div id="settingsDocTypesList" style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.6rem;"></div>
          <div style="display:flex;gap:0.4rem;align-items:center;">
            <input id="settingsNewDocType" type="text" placeholder="e.g. press_release" style="flex:1;max-width:220px;padding:0.3rem 0.55rem;font-size:0.85rem;border:1px solid #c8d8f0;border-radius:6px;">
            <button type="button" class="btn-secondary" onclick="settingsAddDocType()">+ Add</button>
          </div>
        </div>

        <div class="card" style="margin-bottom:1rem;">
          <h2 style="margin:0 0 0.75rem;">LLM Models</h2>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;">
            <div>
              <label style="font-size:0.82rem;font-weight:600;display:block;margin-bottom:0.25rem;">Chat / Answer model</label>
              <input id="settingsLlmChat" type="text" placeholder="gpt-4o" style="width:100%;box-sizing:border-box;padding:0.35rem 0.55rem;font-size:0.85rem;border:1px solid #c8d8f0;border-radius:6px;">
              <p class="status" style="margin:0.2rem 0 0;font-size:0.74rem;">Used for final RAG answer generation</p>
            </div>
            <div>
              <label style="font-size:0.82rem;font-weight:600;display:block;margin-bottom:0.25rem;">Processing model</label>
              <input id="settingsLlmProcessing" type="text" placeholder="gpt-4o-mini" style="width:100%;box-sizing:border-box;padding:0.35rem 0.55rem;font-size:0.85rem;border:1px solid #c8d8f0;border-radius:6px;">
              <p class="status" style="margin:0.2rem 0 0;font-size:0.74rem;">Used for chunking, query rewriting, metadata generation</p>
            </div>
          </div>
        </div>

        <div class="card" style="margin-bottom:1rem;">
          <h2 style="margin:0 0 0.75rem;">Embedding Model</h2>
          <input id="settingsEmbeddingModel" type="text" placeholder="text-embedding-ada-002" style="width:100%;max-width:340px;box-sizing:border-box;padding:0.35rem 0.55rem;font-size:0.85rem;border:1px solid #c8d8f0;border-radius:6px;">
          <p class="status" style="margin:0.3rem 0 0;font-size:0.74rem;">OpenAI embedding model — must match what was used when collections were created. Changing requires re-embedding all collections.</p>
        </div>

        <div class="card" style="margin-bottom:1rem;">
          <h2 style="margin:0 0 0.75rem;">Qdrant Connection</h2>
          <input id="settingsQdrantUrl" type="text" placeholder="https://xxx.qdrant.io" style="width:100%;box-sizing:border-box;padding:0.35rem 0.55rem;font-size:0.85rem;border:1px solid #c8d8f0;border-radius:6px;">
          <p class="status" style="margin:0.3rem 0 0;font-size:0.74rem;">⚠️ Changing the Qdrant URL requires a server restart to take effect.</p>
        </div>

        <div style="display:flex;align-items:center;gap:1rem;">
          <button type="button" class="btn-wizard" onclick="settingsSave()" style="padding:0.5rem 1.4rem;">💾 Save Settings</button>
          <span id="settingsSaveStatus" style="font-size:0.82rem;color:#555;"></span>
        </div>

      </div>

      <!-- DEV-only section (hidden in PROD) -->
      <div id="devDataSection" style="display:none;margin-top:1.5rem;padding-top:1rem;border-top:1px dashed #e0e0e0;">
        <h3 style="margin:0 0 0.5rem 0;font-size:0.95rem;">🔧 DEV Data</h3>
        <p style="font-size:0.82rem;color:#666;margin:0 0 0.6rem 0;">
          DEV mode uses isolated state files. Copy production data to start with a fresh snapshot.
        </p>
        <div style="display:flex;align-items:center;gap:1rem;">
          <button type="button" class="btn-wizard-add" onclick="_copyProdData()" style="padding:0.4rem 1rem;">📋 Copy PROD data to DEV</button>
          <span id="devCopyStatus" style="font-size:0.82rem;color:#555;"></span>
        </div>
      </div>
      <script>if("__DEV_MODE__"==="1")document.getElementById("devDataSection").style.display="";</script>

    </div>
    <!-- ── end Settings panel ── -->

  </div>

  <script>
    const api = (path, body) => fetch(path, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined
    }).then(async r => {
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
      return data;
    });

    const buildLog = document.getElementById('buildLog');
    const qaResult = document.getElementById('qaResult');
    const setLog = (el, msg, isError) => {
      el.textContent = msg;
      el.className = 'log' + (isError ? ' error' : ' success');
    };

    // ── Inline confirmation strip ───────────────────────────────────────────
    // Shows a confirmation bar below the trigger button with a descriptive
    // message and separate [Confirm] / [✕] buttons. Click outside = dismiss.
    function _inlineConfirm(btn, { message, onConfirm, confirmLabel }) {
      if (btn._confirmStrip) return;                       // prevent double-open
      const strip = document.createElement('div');
      strip.style.cssText = 'display:flex;align-items:center;gap:0.4rem;padding:0.3rem 0.55rem;background:#fff3e0;border:1px solid #ffe0b2;border-radius:6px;font-size:0.78rem;margin-top:0.25rem;';
      const msg = document.createElement('span');
      msg.style.cssText = 'color:#e65100;';
      msg.textContent = message || 'Are you sure?';
      strip.appendChild(msg);
      const yes = document.createElement('button');
      yes.type = 'button';
      yes.style.cssText = 'font-size:0.74rem;padding:0.15rem 0.55rem;background:#c0392b;color:#fff;border:none;border-radius:4px;cursor:pointer;white-space:nowrap;';
      yes.textContent = confirmLabel || 'Confirm';
      strip.appendChild(yes);
      const no = document.createElement('button');
      no.type = 'button';
      no.style.cssText = 'font-size:0.74rem;padding:0.15rem 0.35rem;background:none;border:1px solid #ccc;border-radius:4px;cursor:pointer;color:#666;';
      no.textContent = '✕';
      strip.appendChild(no);
      // Insert strip after the button's parent row
      const anchor = btn.parentElement;
      anchor.insertAdjacentElement('afterend', strip);
      btn._confirmStrip = strip;
      const cleanup = () => { strip.remove(); btn._confirmStrip = null; document.removeEventListener('click', outsideHandler, true); };
      yes.onclick = (e) => { e.stopPropagation(); cleanup(); onConfirm(btn); };
      no.onclick = (e) => { e.stopPropagation(); cleanup(); };
      const outsideHandler = (e) => { if (!strip.contains(e.target) && !btn.contains(e.target)) cleanup(); };
      setTimeout(() => document.addEventListener('click', outsideHandler, true), 10);
    }

    // Flash a temporary warning banner at top of wizard collections list
    function _wizardFlashWarning(msg) {
      const target = document.getElementById('wizardCollList') || document.getElementById('wizardResults');
      if (!target) return;
      const banner = document.createElement('div');
      banner.style.cssText = 'padding:0.4rem 0.7rem;background:#fff9c4;color:#f57f17;border:1px solid #ffe082;border-radius:6px;font-size:0.78rem;margin-bottom:0.5rem;';
      banner.textContent = '⚠ ' + msg;
      target.prepend(banner);
      setTimeout(() => banner.remove(), 5000);
    }

    // ── Chunk Viewer / Editor ──────────────────────────────────────────────
    let _chunkViewerCache = {};  // keyed by src.id

    function _toggleChunkViewer(src, row) {
      // If panel already open for this source, close it
      const existingPanel = row.nextElementSibling;
      if (existingPanel && existingPanel.dataset.chunkViewer === src.id) {
        existingPanel.remove();
        delete _chunkViewerCache[src.id];
        return;
      }
      // Close any other open viewer
      document.querySelectorAll('[data-chunk-viewer]').forEach(p => p.remove());
      // Create and insert panel
      const panel = document.createElement('div');
      panel.dataset.chunkViewer = src.id;
      panel.style.cssText = 'background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:0.7rem;margin-bottom:0.5rem;max-height:500px;overflow-y:auto;';
      panel.innerHTML = '<div style="text-align:center;color:#888;font-size:0.82rem;padding:1rem 0;">Loading chunks…</div>';
      row.insertAdjacentElement('afterend', panel);
      _loadChunkViewer(src, panel);
    }

    async function _loadChunkViewer(src, panel) {
      const collName = document.getElementById('collectionSelect').value;
      const cacheKey = src.id;
      try {
        const data = await api('/api/collections/' + encodeURIComponent(collName) + '/chunks?source_id=' + encodeURIComponent(src.id));
        _chunkViewerCache[cacheKey] = data;
        _renderChunkPanel(src, data, panel, collName);
      } catch(e) {
        panel.innerHTML = '<div style="color:#c0392b;font-size:0.82rem;padding:0.5rem;">Failed to load chunks: ' + e.message + '</div>';
      }
    }

    function _getShortUrl(fullUrl) {
      try {
        const u = new URL(fullUrl);
        return u.pathname.replace(/\/$/, '').split('/').slice(-2).join('/') || u.hostname;
      } catch(_) { return fullUrl.split('/').pop() || fullUrl; }
    }

    function _renderChunkPanel(src, data, panel, collName) {
      panel.innerHTML = '';
      let totalChunks = data.total_chunks;

      // Header
      const header = document.createElement('div');
      header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;padding-bottom:0.4rem;border-bottom:1px solid #dee2e6;';
      const title = document.createElement('span');
      title.style.cssText = 'font-size:0.85rem;font-weight:600;color:#333;';
      title.textContent = '📄 ' + (src.label || src.id) + ' (' + totalChunks + ' chunks)';
      header.appendChild(title);
      const closeBtn = document.createElement('button');
      closeBtn.type = 'button';
      closeBtn.style.cssText = 'font-size:0.78rem;padding:0.1rem 0.4rem;background:none;border:1px solid #ccc;border-radius:4px;color:#666;cursor:pointer;';
      closeBtn.textContent = '\u2715';
      closeBtn.onclick = () => { panel.remove(); delete _chunkViewerCache[src.id]; };
      header.appendChild(closeBtn);
      panel.appendChild(header);

      const updateTotalCount = (delta) => {
        totalChunks += delta;
        title.textContent = '📄 ' + (src.label || src.id) + ' (' + totalChunks + ' chunks)';
      };

      if ((!data.urls || !data.urls.length) && !(data.excluded_urls && data.excluded_urls.length)) {
        const empty = document.createElement('div');
        empty.style.cssText = 'color:#888;font-size:0.82rem;padding:0.5rem 0;';
        empty.textContent = 'No chunks found in Qdrant for this source.';
        panel.appendChild(empty);
        return;
      }

      // Excluded URLs section (rendered at bottom, but we need a ref now)
      const excludedSection = document.createElement('div');
      excludedSection.style.cssText = 'display:none;margin-top:0.5rem;padding-top:0.4rem;border-top:1px solid #dee2e6;';
      let excludedUrls = (data.excluded_urls || []).slice();  // mutable copy

      function _renderExcludedSection() {
        excludedSection.innerHTML = '';
        if (!excludedUrls.length) { excludedSection.style.display = 'none'; return; }
        excludedSection.style.display = 'block';
        const exHeader = document.createElement('div');
        exHeader.style.cssText = 'font-size:0.78rem;font-weight:600;color:#888;margin-bottom:0.3rem;';
        exHeader.textContent = '🚫 Excluded URLs (' + excludedUrls.length + ') — will not be re-pushed';
        excludedSection.appendChild(exHeader);
        excludedUrls.forEach(exUrl => {
          const row = document.createElement('div');
          row.style.cssText = 'display:flex;align-items:center;gap:0.4rem;padding:0.2rem 0.5rem;font-size:0.76rem;color:#999;';
          const label = document.createElement('span');
          label.style.cssText = 'flex:1;text-decoration:line-through;';
          label.textContent = _getShortUrl(exUrl);
          label.title = exUrl;
          row.appendChild(label);
          const restoreBtn = document.createElement('button');
          restoreBtn.type = 'button';
          restoreBtn.style.cssText = 'font-size:0.68rem;padding:0.1rem 0.35rem;background:none;border:1px solid #27ae60;border-radius:3px;color:#27ae60;cursor:pointer;';
          restoreBtn.textContent = 'Restore';
          restoreBtn.onclick = async (e) => {
            e.stopPropagation();
            restoreBtn.disabled = true;
            restoreBtn.textContent = '...';
            try {
              await api('/api/collections/' + encodeURIComponent(collName) + '/chunks/restore-url', {
                source_id: src.id, url: exUrl
              });
              excludedUrls = excludedUrls.filter(u => u !== exUrl);
              _renderExcludedSection();
            } catch(err) {
              restoreBtn.textContent = 'Error';
              restoreBtn.style.color = '#c0392b';
              setTimeout(() => { restoreBtn.textContent = 'Restore'; restoreBtn.style.color = '#27ae60'; restoreBtn.disabled = false; }, 2000);
            }
          };
          row.appendChild(restoreBtn);
          excludedSection.appendChild(row);
        });
      }

      function _addToExcluded(url) {
        if (!excludedUrls.includes(url)) excludedUrls.push(url);
        _renderExcludedSection();
      }

      // URL groups
      (data.urls || []).forEach((urlGroup, gi) => {
        const urlDiv = document.createElement('div');
        urlDiv.style.cssText = 'margin-bottom:0.3rem;';
        let urlChunkCount = urlGroup.chunks.length;

        const urlHeader = document.createElement('div');
        urlHeader.style.cssText = 'display:flex;align-items:center;gap:0.4rem;padding:0.3rem 0.5rem;background:#e9ecef;border-radius:5px;cursor:pointer;user-select:none;';
        const arrow = document.createElement('span');
        arrow.textContent = '\u25b8';
        arrow.style.cssText = 'font-size:0.7rem;color:#666;transition:transform 0.15s;';
        urlHeader.appendChild(arrow);
        const urlLabel = document.createElement('span');
        urlLabel.style.cssText = 'font-size:0.8rem;color:#444;flex:1;';
        urlLabel.textContent = _getShortUrl(urlGroup.url);
        urlHeader.appendChild(urlLabel);
        const countBadge = document.createElement('span');
        countBadge.style.cssText = 'font-size:0.7rem;color:#888;';
        countBadge.textContent = urlChunkCount + ' chunk' + (urlChunkCount > 1 ? 's' : '');
        urlHeader.appendChild(countBadge);

        // URL-group delete button
        const urlDelBtn = document.createElement('button');
        urlDelBtn.type = 'button';
        urlDelBtn.style.cssText = 'font-size:0.68rem;padding:0.1rem 0.3rem;background:none;border:1px solid #c0392b;border-radius:3px;color:#c0392b;cursor:pointer;';
        urlDelBtn.textContent = '🗑';
        urlDelBtn.title = 'Delete all chunks for this page';
        urlDelBtn.onclick = (e) => {
          e.stopPropagation();
          _inlineConfirm(urlDelBtn, {
            message: 'Delete all ' + urlChunkCount + ' chunk(s) & exclude page?',
            confirmLabel: 'Delete + Exclude',
            onConfirm: async () => {
              urlDelBtn.disabled = true;
              urlDelBtn.textContent = '...';
              try {
                const ids = urlGroup.chunks.map(c => c.id);
                await api('/api/collections/' + encodeURIComponent(collName) + '/chunks/delete', {
                  point_ids: ids, source_id: src.id, url: urlGroup.url
                });
                updateTotalCount(-urlChunkCount);
                urlDiv.remove();
                _addToExcluded(urlGroup.url);
              } catch(err) {
                urlDelBtn.textContent = '🗑';
                urlDelBtn.disabled = false;
                alert('Delete failed: ' + (err.message || err));
              }
            }
          });
        };
        urlHeader.appendChild(urlDelBtn);

        urlDiv.appendChild(urlHeader);

        const chunksContainer = document.createElement('div');
        chunksContainer.style.cssText = 'display:none;padding:0.3rem 0 0.3rem 1rem;';

        urlGroup.chunks.forEach((chunk) => {
          const chunkRow = document.createElement('div');
          const isEdited = chunk.manually_edited;
          chunkRow.style.cssText = 'margin-bottom:0.4rem;border:1px solid ' + (isEdited ? '#f0ad4e' : '#dee2e6') + ';border-radius:5px;padding:0.4rem 0.5rem;background:#fff;' + (isEdited ? 'border-left:3px solid #f0ad4e;' : '');

          // Chunk header with index and badges
          const chunkHeader = document.createElement('div');
          chunkHeader.style.cssText = 'display:flex;align-items:center;gap:0.4rem;margin-bottom:0.25rem;';
          const idxBadge = document.createElement('span');
          idxBadge.style.cssText = 'font-size:0.68rem;color:#888;';
          idxBadge.textContent = '#' + (chunk.idx != null ? chunk.idx : '?');
          chunkHeader.appendChild(idxBadge);
          if (isEdited) {
            const editBadge = document.createElement('span');
            editBadge.style.cssText = 'font-size:0.68rem;color:#e67e22;font-weight:500;';
            editBadge.textContent = '\u270f\ufe0f edited';
            if (chunk.edited_at) editBadge.title = 'Edited: ' + chunk.edited_at;
            chunkHeader.appendChild(editBadge);
          }
          const spacer = document.createElement('span');
          spacer.style.cssText = 'flex:1;';
          chunkHeader.appendChild(spacer);
          // Edit button
          const editBtn = document.createElement('button');
          editBtn.type = 'button';
          editBtn.style.cssText = 'font-size:0.72rem;padding:0.1rem 0.4rem;background:none;border:1px solid #2980b9;border-radius:3px;color:#2980b9;cursor:pointer;';
          editBtn.textContent = '\u270f\ufe0f Edit';
          chunkHeader.appendChild(editBtn);
          // Per-chunk delete button
          const chunkDelBtn = document.createElement('button');
          chunkDelBtn.type = 'button';
          chunkDelBtn.style.cssText = 'font-size:0.68rem;padding:0.1rem 0.3rem;background:none;border:1px solid #c0392b;border-radius:3px;color:#c0392b;cursor:pointer;';
          chunkDelBtn.textContent = '🗑';
          chunkDelBtn.title = 'Delete this chunk';
          chunkDelBtn.onclick = (e) => {
            e.stopPropagation();
            _inlineConfirm(chunkDelBtn, {
              message: 'Delete this chunk?',
              confirmLabel: 'Delete',
              onConfirm: async () => {
                chunkDelBtn.disabled = true;
                chunkDelBtn.textContent = '...';
                try {
                  const res = await api('/api/collections/' + encodeURIComponent(collName) + '/chunks/delete', {
                    point_ids: [chunk.id], source_id: src.id
                  });
                  chunkRow.remove();
                  urlChunkCount--;
                  updateTotalCount(-1);
                  countBadge.textContent = urlChunkCount + ' chunk' + (urlChunkCount > 1 ? 's' : '');
                  if (res.auto_excluded) {
                    urlDiv.remove();
                    _addToExcluded(res.excluded_url);
                  } else if (urlChunkCount <= 0) {
                    urlDiv.remove();
                  }
                } catch(err) {
                  chunkDelBtn.textContent = '🗑';
                  chunkDelBtn.disabled = false;
                  alert('Delete failed: ' + (err.message || err));
                }
              }
            });
          };
          chunkHeader.appendChild(chunkDelBtn);
          chunkRow.appendChild(chunkHeader);

          // Chunk text (readonly view)
          const textBox = document.createElement('div');
          textBox.style.cssText = 'font-size:0.78rem;font-family:monospace;color:#333;white-space:pre-wrap;word-break:break-word;max-height:150px;overflow-y:auto;background:#fafafa;padding:0.35rem 0.5rem;border-radius:4px;border:1px solid #eee;line-height:1.45;';
          textBox.textContent = chunk.text;
          chunkRow.appendChild(textBox);

          // Edit mode handler
          editBtn.onclick = (e) => {
            e.stopPropagation();
            _enterChunkEditMode(chunk, chunkRow, textBox, editBtn, collName, src);
          };

          chunksContainer.appendChild(chunkRow);
        });
        urlDiv.appendChild(chunksContainer);

        // Toggle expand/collapse
        let expanded = false;
        urlHeader.onclick = (e) => {
          if (e.target.tagName === 'BUTTON') return;  // don't toggle on delete button click
          expanded = !expanded;
          chunksContainer.style.display = expanded ? 'block' : 'none';
          arrow.style.transform = expanded ? 'rotate(90deg)' : '';
        };

        panel.appendChild(urlDiv);
      });

      // Excluded URLs section at bottom
      _renderExcludedSection();
      panel.appendChild(excludedSection);
    }

    function _enterChunkEditMode(chunk, chunkRow, textBox, editBtn, collName, src) {
      // Replace textBox with textarea
      const textarea = document.createElement('textarea');
      textarea.style.cssText = 'width:100%;min-height:100px;font-size:0.78rem;font-family:monospace;color:#333;background:#fff;padding:0.35rem 0.5rem;border-radius:4px;border:1px solid #2980b9;line-height:1.45;box-sizing:border-box;resize:vertical;';
      textarea.value = chunk.text;
      textBox.replaceWith(textarea);
      editBtn.style.display = 'none';

      // Add save/cancel buttons
      const btnRow = document.createElement('div');
      btnRow.style.cssText = 'display:flex;gap:0.4rem;margin-top:0.3rem;';
      const saveBtn = document.createElement('button');
      saveBtn.type = 'button';
      saveBtn.style.cssText = 'font-size:0.74rem;padding:0.2rem 0.6rem;background:#27ae60;color:#fff;border:none;border-radius:4px;cursor:pointer;';
      saveBtn.textContent = '💾 Save';
      const cancelBtn = document.createElement('button');
      cancelBtn.type = 'button';
      cancelBtn.style.cssText = 'font-size:0.74rem;padding:0.2rem 0.5rem;background:none;border:1px solid #ccc;border-radius:4px;color:#666;cursor:pointer;';
      cancelBtn.textContent = '✕ Cancel';
      btnRow.appendChild(saveBtn);
      btnRow.appendChild(cancelBtn);
      chunkRow.appendChild(btnRow);

      cancelBtn.onclick = (e) => {
        e.stopPropagation();
        textarea.replaceWith(textBox);
        btnRow.remove();
        editBtn.style.display = '';
      };

      saveBtn.onclick = async (e) => {
        e.stopPropagation();
        const newText = textarea.value.trim();
        if (!newText) { alert('Chunk text cannot be empty.'); return; }
        if (newText === chunk.text) { cancelBtn.click(); return; }
        saveBtn.textContent = '⏳ Saving…'; saveBtn.disabled = true;
        try {
          const res = await fetch('/api/collections/' + encodeURIComponent(collName) + '/chunks/' + encodeURIComponent(chunk.id), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: newText })
          });
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || 'Save failed');
          // Update chunk data in cache
          chunk.text = newText;
          chunk.manually_edited = true;
          chunk.edited_at = data.edited_at || new Date().toISOString();
          if (data.original_text) chunk.original_text = data.original_text;
          // Re-render this chunk
          textBox.textContent = newText;
          textarea.replaceWith(textBox);
          btnRow.remove();
          editBtn.style.display = '';
          // Update chunk row border to show edited state
          chunkRow.style.borderColor = '#f0ad4e';
          chunkRow.style.borderLeft = '3px solid #f0ad4e';
          // Add edited badge if not already present
          const header = chunkRow.querySelector('div');
          if (!header.querySelector('[data-edit-badge]')) {
            const badge = document.createElement('span');
            badge.style.cssText = 'font-size:0.68rem;color:#e67e22;font-weight:500;';
            badge.textContent = '✏️ edited';
            badge.dataset.editBadge = '1';
            header.insertBefore(badge, header.children[1]);
          }
          setLog(buildLog, 'Chunk updated and re-embedded in Qdrant.', false);
        } catch(err) {
          saveBtn.textContent = '💾 Save'; saveBtn.disabled = false;
          alert('Failed to save chunk: ' + err.message);
        }
      };
    }

    // ── Push Guard Modal ──────────────────────────────────────────────────
    function _showPushGuardModal(data, src, callback) {
      // Overlay
      const overlay = document.createElement('div');
      overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);z-index:10000;display:flex;align-items:center;justify-content:center;';

      const modal = document.createElement('div');
      modal.style.cssText = 'background:#fff;border-radius:10px;padding:1.2rem;max-width:550px;width:90%;max-height:80vh;overflow-y:auto;box-shadow:0 8px 30px rgba(0,0,0,0.3);';

      // Title
      const title = document.createElement('div');
      title.style.cssText = 'font-size:1rem;font-weight:600;color:#e67e22;margin-bottom:0.7rem;';
      title.textContent = '⚠️ Manually Edited Chunks Detected';
      modal.appendChild(title);

      const desc = document.createElement('p');
      desc.style.cssText = 'font-size:0.85rem;color:#555;margin:0 0 0.7rem 0;line-height:1.5;';
      desc.textContent = 'The following pages have manually edited chunks that will be overwritten if re-pushed:';
      modal.appendChild(desc);

      // Checkbox list
      const list = document.createElement('div');
      list.style.cssText = 'margin-bottom:0.8rem;';
      const checkboxes = [];
      data.edited_urls.forEach(eu => {
        const row = document.createElement('label');
        row.style.cssText = 'display:flex;align-items:center;gap:0.4rem;padding:0.3rem 0.5rem;margin-bottom:0.2rem;border-radius:5px;cursor:pointer;font-size:0.82rem;color:#333;';
        row.onmouseenter = () => { row.style.background = '#f5f5f5'; };
        row.onmouseleave = () => { row.style.background = ''; };
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true; // default: skip (preserve edits)
        cb.dataset.url = eu.url;
        checkboxes.push(cb);
        row.appendChild(cb);
        let shortUrl = eu.url;
        try { shortUrl = new URL(eu.url).pathname; } catch(_) {}
        const label = document.createElement('span');
        label.textContent = shortUrl + ' (' + eu.edited_count + ' edited chunk' + (eu.edited_count > 1 ? 's' : '') + ')';
        row.appendChild(label);
        list.appendChild(row);
      });
      modal.appendChild(list);

      const hint = document.createElement('p');
      hint.style.cssText = 'font-size:0.78rem;color:#888;margin:0 0 0.8rem 0;line-height:1.4;';
      hint.textContent = 'Checked pages will be SKIPPED (edits preserved). Uncheck to overwrite with fresh pipeline data.';
      modal.appendChild(hint);

      // Buttons
      const btnRow = document.createElement('div');
      btnRow.style.cssText = 'display:flex;gap:0.5rem;justify-content:flex-end;';

      const cancelBtn = document.createElement('button');
      cancelBtn.type = 'button';
      cancelBtn.style.cssText = 'padding:0.35rem 0.8rem;background:none;border:1px solid #ccc;border-radius:5px;color:#666;cursor:pointer;font-size:0.82rem;';
      cancelBtn.textContent = '✕ Cancel';
      cancelBtn.onclick = () => overlay.remove();

      const pushAllBtn = document.createElement('button');
      pushAllBtn.type = 'button';
      pushAllBtn.style.cssText = 'padding:0.35rem 0.8rem;background:#c0392b;border:none;border-radius:5px;color:#fff;cursor:pointer;font-size:0.82rem;';
      pushAllBtn.textContent = 'Push all (overwrite)';
      pushAllBtn.onclick = () => { overlay.remove(); callback([]); };

      const pushSkipBtn = document.createElement('button');
      pushSkipBtn.type = 'button';
      pushSkipBtn.style.cssText = 'padding:0.35rem 0.8rem;background:#27ae60;border:none;border-radius:5px;color:#fff;cursor:pointer;font-size:0.82rem;font-weight:500;';
      pushSkipBtn.textContent = 'Push with skips';
      pushSkipBtn.onclick = () => {
        const skipUrls = checkboxes.filter(cb => cb.checked).map(cb => cb.dataset.url);
        overlay.remove();
        callback(skipUrls);
      };

      btnRow.appendChild(cancelBtn);
      btnRow.appendChild(pushAllBtn);
      btnRow.appendChild(pushSkipBtn);
      modal.appendChild(btnRow);

      overlay.appendChild(modal);
      overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
      document.body.appendChild(overlay);
    }

    // Load settings (doc_types, models, etc.) on startup
    let _docTypes = ['product_catalog','recipe_book','faq','manual','legal','general']; // defaults, overwritten by API
    let _settingsCache = {};
    (async () => {
      try {
        const s = await api('/api/settings');
        _settingsCache = s;
        if (s.doc_types && s.doc_types.length) _docTypes = s.doc_types;
      } catch(_) {}
    })();

    // ── Global solution selector ─────────────────────────────────────────────
    let _allSolutions = [];
    let _currentSolutionId = null;
    let _pendingNewSolutionName = null;  // for "+ Create new solution" flow

    function _populateGlobalSolutionDropdown(selectId) {
      const sel = document.getElementById('globalSolution');
      sel.innerHTML = '';
      // Placeholder
      const ph = document.createElement('option');
      ph.value = ''; ph.textContent = '— Select a solution —';
      sel.appendChild(ph);
      // Solutions
      _allSolutions.forEach(s => {
        const o = document.createElement('option');
        o.value = s.id;
        o.textContent = s.display_name || s.id;
        o.dataset.company = s.company_name || '';
        sel.appendChild(o);
      });
      // Create new option
      const newOpt = document.createElement('option');
      newOpt.value = '__new__';
      newOpt.textContent = '+ Create new solution…';
      sel.appendChild(newOpt);
      // Auto-select if requested
      if (selectId) {
        sel.value = selectId;
        onGlobalSolutionChange();
      }
    }

    (async () => {
      const { solutions } = await api('/api/solutions');
      _allSolutions = solutions || [];
      _populateGlobalSolutionDropdown();
      _wizardPopulateSolNameList();
    })();

    function _getActiveTab() {
      const active = document.querySelector('.tab.active');
      return active ? active.dataset.tab : null;
    }

    function _getSolutionDisplayName(solId) {
      const sol = _allSolutions.find(s => s.id === solId);
      return sol ? (sol.display_name || sol.id) : solId;
    }

    async function onGlobalSolutionChange() {
      const sel = document.getElementById('globalSolution');
      const val = sel.value;

      // Handle "+ Create new solution" option
      if (val === '__new__') {
        document.getElementById('globalSolNewName').style.display = '';
        document.getElementById('globalSolNewBtn').style.display = '';
        document.getElementById('globalSolNewCancel').style.display = '';
        document.getElementById('globalSolLang').style.display = 'none';
        document.getElementById('globalSolNewName').focus();
        _currentSolutionId = null;
        _pendingNewSolutionName = null;
        _applyGlobalSolution();
        return;
      }

      // Hide new-solution inputs
      document.getElementById('globalSolNewName').style.display = 'none';
      document.getElementById('globalSolNewBtn').style.display = 'none';
      document.getElementById('globalSolNewCancel').style.display = 'none';
      _pendingNewSolutionName = null;

      _currentSolutionId = val || null;

      // Update language badge
      const langBadge = document.getElementById('globalSolLang');
      if (_currentSolutionId) {
        const sol = _allSolutions.find(s => s.id === _currentSolutionId);
        const lang = sol && sol.language ? sol.language : null;
        langBadge.style.display = 'inline-block';
        langBadge.textContent = lang ? `🌐 ${lang}` : '🌐 set language';
        langBadge.title = lang
          ? `Base language: ${lang} — click to change`
          : 'No base language set — click to set';
      } else {
        langBadge.style.display = 'none';
      }

      await _applyGlobalSolution();
    }

    async function _applyGlobalSolution() {
      const solId = _currentSolutionId;
      const activeTab = _getActiveTab();

      // ── Build tab ──
      const noSolRow = document.getElementById('noSolutionRow');
      const collSection = document.getElementById('collectionSection');
      const langEditor = document.getElementById('solLangEditor');
      if (!solId) {
        if (noSolRow) noSolRow.style.display = 'block';
        if (collSection) collSection.style.display = 'none';
        if (langEditor) langEditor.style.display = 'none';
        renderSubCollectionPicker([]);
      } else {
        if (noSolRow) noSolRow.style.display = 'none';
        if (collSection) collSection.style.display = 'block';
        hideLangEditor();
        if (activeTab === 'build') {
          await loadSolutionCollections(solId);
          renderRecentFiles();
        }
      }

      // ── Chat tab ──
      const chatNoSol = document.getElementById('chatNoSolutionMsg');
      const chatCollRow = document.getElementById('chatCollectionRow');
      const chatCollSel = document.getElementById('chatCollectionSelect');
      if (!solId) {
        if (chatNoSol) chatNoSol.style.display = 'block';
        if (chatCollRow) chatCollRow.style.display = 'none';
        if (chatCollSel) chatCollSel.innerHTML = '';
      } else {
        if (chatNoSol) chatNoSol.style.display = 'none';
        if (activeTab === 'chat') await _loadChatCollections(solId);
      }

      // ── Wizard tab: sync hidden wizardSolName input ──
      const wizSolName = document.getElementById('wizardSolName');
      const wizNoSolMsg = document.getElementById('wizardNoSolutionMsg');
      if (solId) {
        if (wizSolName) wizSolName.value = _getSolutionDisplayName(solId);
        if (wizNoSolMsg) wizNoSolMsg.style.display = 'none';
      } else if (_pendingNewSolutionName) {
        if (wizSolName) wizSolName.value = _pendingNewSolutionName;
        if (wizNoSolMsg) wizNoSolMsg.style.display = 'none';
      } else {
        if (wizSolName) wizSolName.value = '';
        if (wizNoSolMsg) wizNoSolMsg.style.display = 'block';
      }
    }

    // Extracted chat collection loading for reuse
    async function _loadChatCollections(solId) {
      const collRow = document.getElementById('chatCollectionRow');
      const collSel = document.getElementById('chatCollectionSelect');
      const chatNoSol = document.getElementById('chatNoSolutionMsg');
      if (!solId) {
        if (collRow) collRow.style.display = 'none';
        if (collSel) collSel.innerHTML = '';
        return;
      }
      if (chatNoSol) chatNoSol.style.display = 'none';
      if (collRow) collRow.style.display = 'block';
      collSel.innerHTML = '<option value="">Loading…</option>';
      try {
        const res = await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections').then(r => r.json());
        collSel.innerHTML = '';
        const existing = (res.collections || []).filter(c => c.exists);
        if (existing.length > 1) {
          const allOpt = document.createElement('option');
          allOpt.value = '__all__';
          allOpt.textContent = '⚡ All collections (recommended)';
          collSel.appendChild(allOpt);
        }
        existing.forEach(c => {
          const o = document.createElement('option');
          o.value = c.name;
          o.textContent = c.display_name || c.name;
          collSel.appendChild(o);
        });
        if (!existing.length) {
          collSel.innerHTML = '<option value="">No collections found in Qdrant yet</option>';
        }
      } catch(e) {
        collSel.innerHTML = '<option value="">Error loading collections</option>';
      }
    }

    // Backward compat: onSolutionChange still called by some internal functions
    async function onSolutionChange() {
      await _applyGlobalSolution();
      if (_currentSolutionId) {
        await loadSolutionCollections(_currentSolutionId);
        renderRecentFiles();
      }
    }

    // ── New solution creation from global dropdown ────────────────────────────
    function _globalCreateNewSolution() {
      const nameInput = document.getElementById('globalSolNewName');
      const name = nameInput.value.trim();
      if (!name) { nameInput.focus(); return; }
      _pendingNewSolutionName = name;
      // Set the hidden wizardSolName so wizard functions can pick it up
      const wizSolName = document.getElementById('wizardSolName');
      if (wizSolName) wizSolName.value = name;
      // Reset dropdown to placeholder (the new solution doesn't exist yet)
      document.getElementById('globalSolution').value = '';
      // Hide create inputs, show a pending badge instead
      nameInput.style.display = 'none';
      document.getElementById('globalSolNewBtn').style.display = 'none';
      document.getElementById('globalSolNewCancel').style.display = 'none';
      const badge = document.getElementById('globalSolLang');
      badge.style.display = 'inline-block';
      badge.textContent = '🆕 ' + name;
      badge.title = 'New solution (will be created when you confirm in Analyse Site)';
      // Update wizard no-solution message
      const wizNoSolMsg = document.getElementById('wizardNoSolutionMsg');
      if (wizNoSolMsg) wizNoSolMsg.style.display = 'none';
      _currentSolutionId = null;
      _applyGlobalSolution();
    }

    function _globalCancelNewSolution() {
      _pendingNewSolutionName = null;
      document.getElementById('globalSolNewName').style.display = 'none';
      document.getElementById('globalSolNewBtn').style.display = 'none';
      document.getElementById('globalSolNewCancel').style.display = 'none';
      document.getElementById('globalSolution').value = '';
      document.getElementById('globalSolLang').style.display = 'none';
      _currentSolutionId = null;
      _applyGlobalSolution();
    }

    function showLangEditor() {
      if (!_currentSolutionId) return;
      const sol = _allSolutions.find(s => s.id === _currentSolutionId);
      document.getElementById('solLangInput').value = (sol && sol.language) ? sol.language : 'en';
      document.getElementById('solLangEditor').style.display = 'block';
    }

    function hideLangEditor() {
      document.getElementById('solLangEditor').style.display = 'none';
    }

    async function saveSolLanguage() {
      const solId = _currentSolutionId;
      if (!solId) return;
      const lang = document.getElementById('solLangInput').value;
      try {
        const res = await fetch(`/api/solutions/${encodeURIComponent(solId)}/language`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ language: lang })
        });
        const data = await res.json();
        // Update local cache
        const sol = _allSolutions.find(s => s.id === solId);
        if (sol) sol.language = lang;
        // Update global badge
        const badge = document.getElementById('globalSolLang');
        badge.textContent = `🌐 ${lang}`;
        badge.title = `Base language: ${lang} — click to change`;
        hideLangEditor();
        setLog(buildLog, `✅ Base language set to '${lang}' for ${solId}`, false);
      } catch(e) {
        alert('Failed to save language: ' + e);
      }
    }

    // _currentCollections: full collection objects from the API, keyed by collection_name
    let _currentCollections = {};
    let _selectedSourceId = null; // currently selected source within the active collection

    async function loadSolutionCollections(solId) {
      const collSelect = document.getElementById('collectionSelect');
      collSelect.innerHTML = '<option value="">Loading…</option>';
      try {
        const resp = await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections');
        if (!resp.ok) throw new Error('API returned ' + resp.status);
        const res = await resp.json();
        _currentCollections = {};
        res.collections.forEach(c => { _currentCollections[c.name] = c; });

        collSelect.innerHTML = '';
        res.collections.forEach(c => {
          const o = document.createElement('option');
          o.value = c.name;
          o.textContent = (c.display_name || c.name) + (c.exists ? ' ✓' : ' (not in Qdrant)');
          o.dataset.exists = c.exists ? '1' : '0';
          o.dataset.scraper = c.scraper_name || '';
          o.dataset.collId = c.id || c.name;
          collSelect.appendChild(o);
        });
        // Add "create new" at end
        const newOpt = document.createElement('option');
        newOpt.value = '__new__';
        newOpt.textContent = '＋ Create new collection…';
        collSelect.appendChild(newOpt);

        // Render sub-collection pills
        renderSubCollectionPicker(res.collections, solId);

        // Auto-select first existing, or first option
        const first = res.collections.find(c => c.exists) || res.collections[0];
        if (first) {
          collSelect.value = first.name;
          onCollectionSelect();
        } else {
          onCollectionSelect();
        }
      } catch(e) {
        collSelect.innerHTML = '<option value="__new__">＋ Create new collection…</option>';
        renderSubCollectionPicker([], solId);
        onCollectionSelect();
      }
    }

    function renderSubCollectionPicker(collections, solId) {
      const container = document.getElementById('subCollectionPicker');
      if (!container) return;
      container.innerHTML = '';
      if (!collections || collections.length === 0) {
        document.getElementById('routingMetadataPanel').style.display = 'none';
        return;
      }
      if (collections.length === 1) {
        // Single collection — no pills needed, but still show routing panel
        const c = collections[0];
        if (c.scraper_config) window._activeScraperConfig = c.scraper_config;
        else window._activeScraperConfig = null;
        // Auto-select first source if available
        const sources = c.sources || [];
        if (sources.length === 1) selectSource(sources[0].id);
        renderRoutingMetadataPanel(c, solId);
        return;
      }
      // Multiple collections — render pills
      const label = document.createElement('p');
      label.style.cssText = 'font-size:0.82rem;color:#888;margin:0 0 0.4rem 0;';
      label.textContent = 'Select collection to build:';
      container.appendChild(label);
      const pillRow = document.createElement('div');
      pillRow.style.cssText = 'display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.6rem;';

      function selectPill(btn, c) {
        pillRow.querySelectorAll('button').forEach(b => {
          b.style.background = '#222'; b.style.color = '#ccc'; b.style.borderColor = '#555';
        });
        btn.style.background = '#2a5caa'; btn.style.color = '#fff'; btn.style.borderColor = '#2a5caa';
        if (c.scraper_config) window._activeScraperConfig = c.scraper_config;
        else window._activeScraperConfig = null;
        const collSelect = document.getElementById('collectionSelect');
        if (collSelect) { collSelect.value = c.name; onCollectionSelect(); }
        // Auto-select first source if only one
        const sources = c.sources || [];
        if (sources.length === 1) selectSource(sources[0].id);
        renderRoutingMetadataPanel(c, solId);
      }

      collections.forEach((c, idx) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = c.display_name || c.name;
        btn.style.cssText = 'padding:0.25rem 0.7rem;border-radius:999px;border:1px solid #555;background:#222;color:#ccc;cursor:pointer;font-size:0.85rem;';
        btn.onclick = () => selectPill(btn, c);
        pillRow.appendChild(btn);
      });
      container.appendChild(pillRow);

      // Auto-select the first pill on load
      const firstBtn = pillRow.querySelector('button');
      if (firstBtn) selectPill(firstBtn, collections[0]);
    }

    function renderRoutingMetadataPanel(coll, solId) {
      const panel = document.getElementById('routingMetadataPanel');
      if (!panel) return;
      const routing = coll.routing || {};
      const isEmpty = !routing.description;
      panel.style.display = 'block';
      panel.innerHTML = `
        <div style="background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:0.8rem;margin-top:0.6rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
            <span style="font-size:0.82rem;color:#888;">Routing Metadata — <em>${coll.display_name || coll.name}</em></span>
            <div style="display:flex;gap:0.4rem;">
              <button type="button" id="btnRegenRouting" title="Re-generate from Qdrant content (uses LLM)" style="font-size:0.78rem;padding:0.15rem 0.5rem;background:#1a3a1a;border:1px solid #3a6a3a;color:#8bc88b;border-radius:4px;cursor:pointer;">↺ Regenerate</button>
              <button type="button" id="btnEditRouting" style="font-size:0.78rem;padding:0.15rem 0.5rem;background:#333;border:1px solid #555;color:#ccc;border-radius:4px;cursor:pointer;">Edit</button>
            </div>
          </div>
          ${isEmpty
            ? '<p style="font-size:0.82rem;color:#666;margin:0;">No routing metadata yet. Will be auto-generated after chunking.</p>'
            : `<div style="font-size:0.82rem;color:#aaa;line-height:1.6;">
                <b>Description:</b> ${routing.description || '—'}<br>
                <b>Keywords:</b> ${(routing.keywords || []).join(', ') || '—'}<br>
                <b>Typical questions:</b> ${(routing.typical_questions || []).join(' | ') || '—'}<br>
                <b>Not covered:</b> ${(routing.not_covered || []).join(', ') || '—'}<br>
                <b>Language:</b> ${routing.language || '—'} &nbsp; <b>Type:</b> ${routing.doc_type || '—'}
               </div>`
          }
          <div id="routingEditArea" style="display:none;margin-top:0.6rem;">
            <textarea id="routingJsonInput" rows="10" style="width:100%;background:#111;border:1px solid #444;color:#ccc;font-family:monospace;font-size:0.78rem;padding:0.4rem;border-radius:4px;box-sizing:border-box;">${isEmpty ? JSON.stringify({description:'',keywords:[],typical_questions:[],not_covered:[],language:'',doc_type:''}, null, 2) : JSON.stringify(routing, null, 2)}</textarea>
            <div style="display:flex;gap:0.5rem;margin-top:0.4rem;">
              <button type="button" onclick="saveRoutingMetadata('${solId}','${coll.id || coll.name}')" style="padding:0.3rem 0.8rem;background:#2a5caa;border:none;color:#fff;border-radius:4px;cursor:pointer;font-size:0.82rem;">Save</button>
              <button type="button" onclick="document.getElementById('routingEditArea').style.display='none'" style="padding:0.3rem 0.8rem;background:#333;border:1px solid #555;color:#ccc;border-radius:4px;cursor:pointer;font-size:0.82rem;">Cancel</button>
            </div>
          </div>
        </div>`;
      document.getElementById('btnEditRouting').onclick = () => {
        const area = document.getElementById('routingEditArea');
        area.style.display = area.style.display === 'none' ? 'block' : 'none';
      };
      document.getElementById('btnRegenRouting').onclick = () => {
        regenerateRoutingMetadata(solId, coll.id || coll.name, coll.display_name || coll.name);
      };
    }

    async function regenerateRoutingMetadata(solId, collId, collName) {
      const btn = document.getElementById('btnRegenRouting');
      if (btn) { btn.textContent = '⏳ Generating…'; btn.disabled = true; }
      try {
        const res = await fetch(
          `/api/solutions/${encodeURIComponent(solId)}/collections/${encodeURIComponent(collId)}/routing/suggest`,
          { method: 'POST' }
        );
        const data = await res.json();
        if (!res.ok) {
          alert('Regenerate failed: ' + (data.detail || 'Unknown error'));
          return;
        }
        setLog(buildLog, `Routing metadata regenerated for "${collName}" (sampled ${data.chunks_sampled} chunks). Review and edit if needed.`, false);
        // Reload collections to refresh the panel with new data
        await loadSolutionCollections(solId);
      } catch(e) {
        alert('Regenerate error: ' + e.message);
      } finally {
        if (btn) { btn.textContent = '↺ Regenerate'; btn.disabled = false; }
      }
    }

    async function saveRoutingMetadata(solId, collId) {
      try {
        const raw = document.getElementById('routingJsonInput').value;
        const routing = JSON.parse(raw);
        const res = await fetch(`/api/solutions/${encodeURIComponent(solId)}/collections/${encodeURIComponent(collId)}/routing`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ routing })
        }).then(r => r.json());
        setLog(buildLog, res.message || res.detail, !!res.detail);
        // Reload collections to refresh routing display
        await loadSolutionCollections(solId);
      } catch(e) {
        setLog(buildLog, 'Failed to save routing metadata: ' + e.message, true);
      }
    }

    async function autoSaveRoutingMetadata(solId, collId, meta) {
      // Strip topics (not used for routing) and save the routing-relevant fields
      const routing = {};
      for (const k of ['description', 'keywords', 'typical_questions', 'not_covered', 'language', 'doc_type']) {
        if (meta[k] != null) routing[k] = meta[k];
      }
      if (!routing.description) return;  // nothing meaningful to save
      try {
        const res = await fetch(`/api/solutions/${encodeURIComponent(solId)}/collections/${encodeURIComponent(collId)}/routing`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ routing })
        }).then(r => r.json());
        if (!res.detail) {
          setLog(buildLog, (res.message || 'Routing metadata saved.') + ' Edit it in the panel above if needed.', false);
          await loadSolutionCollections(solId);
        }
      } catch(e) { /* silent — auto-save failure is non-critical */ }
    }

    function onCollectionSelect() {
      const val = document.getElementById('collectionSelect').value;
      const newRow = document.getElementById('newCollectionRow');
      const info = document.getElementById('existingCollectionInfo');
      const delBtn = document.getElementById('btnDeleteCollection');
      _selectedSourceId = null;
      if (val === '__new__') {
        newRow.style.display = 'block';
        info.style.display = 'none';
        delBtn.style.display = 'none';
        renderSourcesList([]);
      } else {
        newRow.style.display = 'none';
        info.style.display = 'block';
        const opt = document.getElementById('collectionSelect').options[document.getElementById('collectionSelect').selectedIndex];
        const exists = opt.dataset.exists === '1';
        const coll = _currentCollections[val];
        const sources = (coll && coll.sources) || [];
        // Build aggregate status from sources
        info.innerHTML = '';
        if (exists) {
          const base = document.createElement('span');
          base.textContent = '✓ In Qdrant';
          base.style.cssText = 'color:#2e7d32;font-weight:500;';
          if (coll && coll.points_count) base.textContent += ' · ' + coll.points_count + ' points';
          info.appendChild(base);
        } else {
          const base = document.createElement('span');
          base.textContent = '⚠ Not yet pushed to Qdrant';
          base.style.cssText = 'color:#e65100;';
          info.appendChild(base);
        }
        // Aggregate source status ovals
        if (sources.length > 0) {
          const groups = {};
          sources.forEach(s => {
            const st = (s.pipeline_status || {}).status || 'not_started';
            if (!groups[st]) groups[st] = { count: 0, chunks: 0 };
            groups[st].count++;
            groups[st].chunks += (s.pipeline_status || {}).chunks || 0;
          });
          const order = ['pushed', 'chunked', 'fetched', 'started', 'not_started'];
          order.forEach(st => {
            if (!groups[st] || st === 'not_started') return;
            const g = groups[st];
            const s = _statusStyles[st];
            const oval = document.createElement('span');
            oval.style.cssText = 'font-size:0.72rem;padding:0.12rem 0.5rem;border-radius:10px;background:' + s.bg + ';color:' + s.color + ';margin-left:0.4rem;font-weight:500;';
            let text = s.label;
            if (g.chunks) text += ' · ' + g.chunks + ' chunks';
            if (sources.length > 1) text += ' (' + g.count + ')';
            oval.textContent = text;
            info.appendChild(oval);
          });
        }
        delBtn.style.display = exists ? 'block' : 'none';
        renderSourcesList(sources);
      }
      // Hide source config and pipeline until a source is selected
      document.getElementById('sourceConfigCard').style.display = 'none';
      document.getElementById('pipelineCard').style.display = 'none';
    }

    // ── Sources list rendering & management ──
    const _sourceTypeIcons = { url: '🌐', pdf: '📄', txt: '📝', csv: '📊' };
    const _statusStyles = {
      pushed:      { bg: '#e8f5e9', color: '#2e7d32', label: 'Pushed' },
      chunked:     { bg: '#fff3e0', color: '#e65100', label: 'Chunked' },
      fetched:     { bg: '#fff9c4', color: '#f57f17', label: 'Fetched' },
      started:     { bg: '#f3e5f5', color: '#7b1fa2', label: 'Started' },
      not_started: { bg: '#f5f5f5', color: '#999',    label: '' },
    };

    function _statusBadge(ps) {
      if (!ps || ps.status === 'not_started') return null;
      const s = _statusStyles[ps.status] || _statusStyles.not_started;
      const chunks = ps.chunks ? ' · ' + ps.chunks + ' chunks' : '';
      const items = (ps.status === 'fetched' && ps.items) ? ' · ' + ps.items + ' items' : '';
      const el = document.createElement('span');
      el.style.cssText = 'font-size:0.7rem;padding:0.1rem 0.45rem;border-radius:10px;background:' + s.bg + ';color:' + s.color + ';flex-shrink:0;font-weight:500;';
      el.textContent = s.label + chunks + items;
      return el;
    }

    function renderSourcesList(sources) {
      const container = document.getElementById('sourcesList');
      container.innerHTML = '';
      if (!sources || !sources.length) {
        container.innerHTML = '<p style="font-size:0.83rem;color:#aaa;font-style:italic;margin:0;">No sources yet. Click "+ Add source" to get started.</p>';
        return;
      }
      sources.forEach(src => {
        const row = document.createElement('div');
        const isSelected = _selectedSourceId === src.id;
        row.style.cssText = 'display:flex;align-items:center;gap:0.5rem;padding:0.45rem 0.65rem;border:1px solid ' +
          (isSelected ? '#1a5276' : '#e0e0e0') + ';border-radius:8px;margin-bottom:0.35rem;cursor:pointer;background:' +
          (isSelected ? '#e8f4fd' : '#fff') + ';transition:all 0.15s;';
        row.onmouseenter = () => { if (!isSelected) row.style.background = '#f5f8fa'; };
        row.onmouseleave = () => { if (!isSelected) row.style.background = '#fff'; };
        row.onclick = () => selectSource(src.id);

        const icon = document.createElement('span');
        icon.style.cssText = 'font-size:1.1rem;flex-shrink:0;';
        icon.textContent = _sourceTypeIcons[src.type] || '📁';
        row.appendChild(icon);

        const labelEl = document.createElement('span');
        labelEl.style.cssText = 'flex:1;font-size:0.88rem;font-weight:' + (isSelected ? '600' : '400') + ';';
        labelEl.textContent = src.label || src.id;
        row.appendChild(labelEl);

        // Pipeline status badge
        const badge = _statusBadge(src.pipeline_status);
        if (badge) row.appendChild(badge);

        const typeBadge = document.createElement('span');
        typeBadge.style.cssText = 'font-size:0.72rem;padding:0.1rem 0.4rem;border-radius:10px;background:#f0f0f0;color:#666;flex-shrink:0;';
        typeBadge.textContent = src.type;
        row.appendChild(typeBadge);

        if (src.scraper_name) {
          const scraperBadge = document.createElement('span');
          scraperBadge.style.cssText = 'font-size:0.72rem;padding:0.1rem 0.4rem;border-radius:10px;background:#e3f0fd;color:#1a5276;flex-shrink:0;';
          scraperBadge.textContent = src.scraper_name;
          row.appendChild(scraperBadge);
        }

        // Per-source delete-chunks button — only when pushed to Qdrant
        const ps = src.pipeline_status || {};
        if (ps.status === 'pushed') {
          const chunkDelBtn = document.createElement('button');
          chunkDelBtn.type = 'button';
          chunkDelBtn.style.cssText = 'font-size:0.75rem;padding:0.1rem 0.45rem;background:none;border:1px solid #c0392b;border-radius:4px;color:#c0392b;cursor:pointer;flex-shrink:0;';
          chunkDelBtn.textContent = '🗑';
          chunkDelBtn.title = 'Delete this source\\'s chunks from Qdrant';
          chunkDelBtn.onclick = (e) => {
            e.stopPropagation();
            _inlineConfirm(chunkDelBtn, {
              message: 'Delete chunks for "' + (src.label || src.id) + '" from Qdrant?',
              confirmLabel: 'Delete',
              onConfirm: async (btn) => {
                btn.textContent = '…'; btn.disabled = true;
                try {
                  const solId = _currentSolutionId;
                  const collName = document.getElementById('collectionSelect').value;
                  const res = await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections/' + encodeURIComponent(collName) + '/sources/' + encodeURIComponent(src.id) + '/chunks', { method: 'DELETE' });
                  const data = await res.json();
                  if (!res.ok) throw new Error(data.detail || 'Delete failed');
                  setLog(buildLog, 'Deleted chunks for source "' + (src.label || src.id) + '" (' + data.urls_processed + ' URLs processed).', false);
                  await loadSolutionCollections(solId);
                } catch(err) {
                  btn.textContent = '🗑'; btn.disabled = false;
                  setLog(buildLog, 'Delete chunks failed: ' + err.message, true);
                }
              }
            });
          };
          row.appendChild(chunkDelBtn);

          // View chunks button
          const viewBtn = document.createElement('button');
          viewBtn.type = 'button';
          viewBtn.style.cssText = 'font-size:0.75rem;padding:0.1rem 0.45rem;background:none;border:1px solid #2980b9;border-radius:4px;color:#2980b9;cursor:pointer;flex-shrink:0;';
          viewBtn.textContent = '👁';
          viewBtn.title = 'View chunks in Qdrant';
          viewBtn.onclick = (e) => {
            e.stopPropagation();
            _toggleChunkViewer(src, row);
          };
          row.appendChild(viewBtn);
        }

        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.style.cssText = 'font-size:0.75rem;padding:0.1rem 0.35rem;background:none;border:1px solid #ddd;border-radius:4px;color:#c0392b;cursor:pointer;flex-shrink:0;';
        delBtn.textContent = '✕';
        delBtn.title = 'Remove source';
        delBtn.onclick = (e) => { e.stopPropagation(); _removeSourceWithConfirm(src, delBtn); };
        row.appendChild(delBtn);

        container.appendChild(row);
      });
    }

    function selectSource(sourceId) {
      const collName = document.getElementById('collectionSelect').value;
      if (!collName || collName === '__new__') return;
      const coll = _currentCollections[collName];
      const sources = (coll && coll.sources) || [];
      const src = sources.find(s => s.id === sourceId);
      if (!src) return;

      _selectedSourceId = sourceId;
      renderSourcesList(sources); // re-render to highlight

      // Reset pipeline state when switching sources
      _clearDone();
      setLog(buildLog, '', false);

      // Populate source config
      const configCard = document.getElementById('sourceConfigCard');
      const pipelineCard = document.getElementById('pipelineCard');
      configCard.style.display = 'block';
      pipelineCard.style.display = 'block';

      // Set the label
      document.getElementById('sourceConfigLabel').textContent = '— ' + (src.label || src.id);

      // Set source type
      const srcType = document.getElementById('sourceType');
      srcType.value = src.type;
      onSourceTypeChange();

      // Set type-specific fields
      if (src.type === 'url') {
        document.getElementById('scraperName').value = src.scraper_name || '';
        // Check for saved state
        checkUrlSavedState(collName, sourceId);
      } else {
        // Auto-fill file path if stored in source definition
        if (src.file_path) {
          setSelectedPath(src.file_path);
        } else {
          document.getElementById('sourcePath').value = '';
          document.getElementById('sourcePathDisplay').textContent = 'Click to select a file…';
          document.getElementById('sourcePathFull').style.display = 'none';
        }
      }
    }

    function showAddSourceForm() {
      document.getElementById('addSourceForm').style.display = 'block';
      document.getElementById('newSourceLabel').value = '';
      document.getElementById('newSourceScraper').value = '';
      document.getElementById('newSourceLabel').focus();
    }

    function hideAddSourceForm() {
      document.getElementById('addSourceForm').style.display = 'none';
    }

    async function addSource() {
      const solId = _currentSolutionId;
      const collName = document.getElementById('collectionSelect').value;
      if (!solId || !collName || collName === '__new__') return;
      const sourceType = document.getElementById('newSourceType').value;
      const label = document.getElementById('newSourceLabel').value.trim();
      if (!label) { alert('Please enter a label for the source.'); return; }
      const scraperName = document.getElementById('newSourceScraper').value.trim() || null;
      try {
        const res = await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections/' + encodeURIComponent(collName) + '/sources', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source_type: sourceType, label, scraper_name: scraperName })
        });
        const data = await res.json();
        if (!res.ok) { alert(data.detail || 'Failed to add source'); return; }
        // Update cached collection
        if (_currentCollections[collName]) _currentCollections[collName].sources = data.sources;
        renderSourcesList(data.sources);
        hideAddSourceForm();
      } catch(e) { alert('Error adding source: ' + e.message); }
    }

    // Gate source removal on Qdrant state + inline confirmation
    function _removeSourceWithConfirm(src, btn) {
      const ps = src.pipeline_status || {};
      if (ps.status === 'pushed') {
        alert('This source has chunks in Qdrant. Delete them first using the 🗑 button.');
        return;
      }
      _inlineConfirm(btn, {
        message: 'Remove source "' + (src.label || src.id) + '"?',
        confirmLabel: 'Remove',
        onConfirm: async () => {
          const solId = _currentSolutionId;
          const collName = document.getElementById('collectionSelect').value;
          if (!solId || !collName) return;
          try {
            const res = await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections/' + encodeURIComponent(collName) + '/sources/' + encodeURIComponent(src.id), {
              method: 'DELETE'
            });
            const data = await res.json();
            if (!res.ok) { alert(data.detail || 'Failed to remove source'); return; }
            if (_currentCollections[collName]) _currentCollections[collName].sources = data.sources;
            if (_selectedSourceId === src.id) {
              _selectedSourceId = null;
              document.getElementById('sourceConfigCard').style.display = 'none';
              document.getElementById('pipelineCard').style.display = 'none';
            }
            renderSourcesList(data.sources);
          } catch(e) { alert('Error removing source: ' + e.message); }
        }
      });
    }

    function deleteCollection(btn) {
      const solId = _currentSolutionId;
      const name = document.getElementById('collectionSelect').value;
      if (!name || name === '__new__') return;
      _inlineConfirm(btn, {
        message: 'Delete collection "' + name + '" from Qdrant? This cannot be undone.',
        confirmLabel: 'Delete',
        onConfirm: async () => {
          try {
            const res = await fetch('/api/solutions/delete-collection', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ solution_id: solId, collection_name: name })
            }).then(r => r.json());
            setLog(buildLog, res.message || res.detail, !!res.detail);
            await loadSolutionCollections(solId);
          } catch(e) {
            setLog(buildLog, 'Delete failed: ' + e.message, true);
          }
        }
      });
    }

    async function registerCollection() {
      const solId = _currentSolutionId;
      const name = document.getElementById('collectionName').value.trim();
      if (!solId || !name) return;
      try {
        const res = await fetch('/api/solutions/add-collection', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ solution_id: solId, collection_name: name })
        }).then(r => r.json());
        setLog(buildLog, res.message || res.detail, !!res.detail);
        await loadSolutionCollections(solId);
        document.getElementById('collectionSelect').value = name;
        onCollectionSelect();
      } catch(e) {
        setLog(buildLog, 'Failed: ' + e.message, true);
      }
    }

    function getCollectionName() {
      if (_currentSolutionId) {
        const val = document.getElementById('collectionSelect').value;
        if (val && val !== '__new__') return val;
        return document.getElementById('collectionName').value.trim();
      }
      return '';
    }

    // Tab switching helper — can be called programmatically
    function showTab(name) {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(x => x.classList.add('hidden'));
      const tab = document.querySelector('[data-tab="' + name + '"]');
      if (tab) tab.classList.add('active');
      const panel = document.getElementById('panel-' + name);
      if (panel) panel.classList.remove('hidden');

      // Solution-aware tab initialization
      const solId = _currentSolutionId;
      if (name === 'build' && solId) {
        loadSolutionCollections(solId);
        renderRecentFiles();
      }
      if (name === 'chat') {
        _loadFaqCollections();
        if (solId) _loadChatCollections(solId);
      }
      if (name === 'wizard') _wizardPopulateSolNameList();
      if (name === 'shopify') { if (typeof loadShopifyStores === 'function') loadShopifyStores(); }
      if (name === 'sites') { if (typeof _loadDomcopStatus === 'function') _loadDomcopStatus(); }
      if (name === 'settings') _settingsLoad();
    }

    // ── Settings tab functions ────────────────────────────────────────────────

    let _settingsDocTypes = []; // working copy of doc_types being edited

    async function _settingsLoad() {
      try {
        const s = await api('/api/settings');
        _settingsCache = s;
        _settingsDocTypes = [...(s.doc_types || _docTypes)];
        _settingsRenderDocTypes();
        const lc = v => document.getElementById(v);
        if (lc('settingsLlmChat')) lc('settingsLlmChat').value = s.llm_chat_model || '';
        if (lc('settingsLlmProcessing')) lc('settingsLlmProcessing').value = s.llm_processing_model || '';
        if (lc('settingsEmbeddingModel')) lc('settingsEmbeddingModel').value = s.embedding_model || '';
        if (lc('settingsQdrantUrl')) lc('settingsQdrantUrl').value = s.qdrant_url || '';
        document.getElementById('settingsSaveStatus').textContent = '';
      } catch(e) { console.warn('[settings load]', e); }
    }

    function _settingsRenderDocTypes() {
      const container = document.getElementById('settingsDocTypesList');
      if (!container) return;
      container.innerHTML = '';
      _settingsDocTypes.forEach((dt, i) => {
        const pill = document.createElement('span');
        pill.style.cssText = 'display:inline-flex;align-items:center;gap:0.25rem;background:#e8eef8;border:1px solid #c8d8f0;border-radius:14px;padding:0.18rem 0.55rem;font-size:0.82rem;';
        pill.innerHTML = '<span>' + dt + '</span>';
        const rm = document.createElement('button');
        rm.type = 'button';
        rm.textContent = '✕';
        rm.style.cssText = 'background:none;border:none;cursor:pointer;color:#888;font-size:0.8rem;padding:0 0 0 0.15rem;line-height:1;';
        rm.title = 'Remove ' + dt;
        rm.onclick = () => { _settingsDocTypes.splice(i, 1); _settingsRenderDocTypes(); };
        pill.appendChild(rm);
        container.appendChild(pill);
      });
    }

    function settingsAddDocType() {
      const inp = document.getElementById('settingsNewDocType');
      const val = (inp.value || '').trim().toLowerCase().replace(/[^a-z0-9_]+/g,'_').replace(/^_|_$/g,'');
      if (!val || _settingsDocTypes.includes(val)) { inp.value = ''; return; }
      _settingsDocTypes.push(val);
      _settingsRenderDocTypes();
      inp.value = '';
    }

    async function settingsSave() {
      const statusEl = document.getElementById('settingsSaveStatus');
      statusEl.textContent = 'Saving…';
      try {
        const body = {
          doc_types: _settingsDocTypes,
          llm_chat_model: document.getElementById('settingsLlmChat').value.trim(),
          llm_processing_model: document.getElementById('settingsLlmProcessing').value.trim(),
          embedding_model: document.getElementById('settingsEmbeddingModel').value.trim(),
          qdrant_url: document.getElementById('settingsQdrantUrl').value.trim(),
        };
        await api('/api/settings', body);
        _docTypes = [..._settingsDocTypes]; // update global so wizard dropdowns reflect new types
        statusEl.textContent = '✅ Saved';
        setTimeout(() => { statusEl.textContent = ''; }, 3000);
      } catch(e) {
        statusEl.textContent = '❌ ' + (e.message || e);
      }
    }

    async function _copyProdData() {
      const st = document.getElementById('devCopyStatus');
      st.textContent = 'Copying…';
      try {
        const res = await fetch('/api/dev/copy-prod-data', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Copy failed');
        const n = (data.copied || []).length;
        st.textContent = n ? '✅ Copied ' + n + ' file(s): ' + data.copied.join(', ') : '✅ No PROD files to copy.';
        setTimeout(() => { st.textContent = ''; }, 6000);
      } catch(e) {
        st.textContent = '❌ ' + (e.message || e);
      }
    }

    // Tabs
    document.querySelectorAll('.tab').forEach(t => {
      t.onclick = () => showTab(t.dataset.tab);
    });

    // Show/hide file picker vs scraper input based on source type
    function onSourceTypeChange() {
      const st = document.getElementById('sourceType').value;
      document.getElementById('filePickerRow').classList.toggle('hidden', st === 'url');
      document.getElementById('scraperRow').classList.toggle('hidden', st !== 'url');
      // Show Sync button only for URL/scraper sources
      document.getElementById('runSync').classList.toggle('hidden', st !== 'url');
      // Check for saved state when switching to URL source type
      if (st === 'url') {
        const collName = document.getElementById('collectionSelect').value;
        if (collName && collName !== '__new__') checkUrlSavedState(collName, _selectedSourceId);
      } else {
        document.getElementById('urlResumeBanner').style.display = 'none';
      }
    }

    let _urlSavedStatePath = null;

    async function checkUrlSavedState(collectionName, sourceId) {
      if (!collectionName || collectionName === '__new__') {
        document.getElementById('urlResumeBanner').style.display = 'none';
        return;
      }
      try {
        let url = '/api/state/check-by-collection?collection_name=' + encodeURIComponent(collectionName);
        if (sourceId) url += '&source_id=' + encodeURIComponent(sourceId);
        const res = await fetch(url).then(r => r.json());
        if (res.found) {
          _urlSavedStatePath = res.save_path;
          const steps = res.completed_steps.join(', ') || 'none';
          const chunks = res.chunks_count ? ` ${res.chunks_count} chunks ready.` : '';
          const items = res.scraped_items_count ? ` ${res.scraped_items_count} scraped pages.` : '';
          document.getElementById('urlResumeInfo').textContent = `Steps done: ${steps}.${items}${chunks}`;
          document.getElementById('urlResumeBanner').style.display = 'block';
        } else {
          _urlSavedStatePath = null;
          document.getElementById('urlResumeBanner').style.display = 'none';
        }
      } catch(e) {
        document.getElementById('urlResumeBanner').style.display = 'none';
      }
    }

    async function resumeUrlState() {
      if (!_urlSavedStatePath) return;
      setLog(buildLog, 'Resuming saved state…', false);
      try {
        const res = await fetch('/api/state/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ save_path: _urlSavedStatePath })
        }).then(r => r.json());
        setLog(buildLog, res.message || res.detail, !!res.detail);
        document.getElementById('urlResumeBanner').style.display = 'none';
      } catch(e) {
        setLog(buildLog, 'Resume failed: ' + e.message, true);
      }
    }

    function dismissUrlResume() {
      document.getElementById('urlResumeBanner').style.display = 'none';
      _urlSavedStatePath = null;
    }

    // Show/hide Shopify URL field based on engine selection
    function onScraperEngineChange() {
      const engine = document.querySelector('input[name="scraperEngine"]:checked').value;
      document.getElementById('shopifyUrlRow').classList.toggle('hidden', engine !== 'shopify');
    }

    // ── Recent files (persisted in localStorage) ──────────────────────────
    // Stored as [{path, solution_id, solution_name}], most recent first.
    const RECENT_KEY = 'rag_recent_files_v2';
    const RECENT_MAX = 8;

    function getRecentFiles() {
      try {
        const raw = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
        // Migrate old format (plain strings) to objects
        return raw.map(f => typeof f === 'string' ? { path: f, solution_id: null, solution_name: null } : f);
      } catch { return []; }
    }

    function addRecentFile(path) {
      if (!path) return;
      const solId = _currentSolutionId || null;
      const solName = solId ? _getSolutionDisplayName(solId) : null;
      let files = getRecentFiles().filter(f => f.path !== path);
      files.unshift({ path, solution_id: solId, solution_name: solName });
      if (files.length > RECENT_MAX) files = files.slice(0, RECENT_MAX);
      localStorage.setItem(RECENT_KEY, JSON.stringify(files));
      renderRecentFiles();
    }

    function removeRecentFile(path) {
      const files = getRecentFiles().filter(f => f.path !== path);
      localStorage.setItem(RECENT_KEY, JSON.stringify(files));
      renderRecentFiles();
    }

    function renderRecentFiles() {
      const container = document.getElementById('recentFiles');
      const all = getRecentFiles();
      if (!all.length) { container.style.display = 'none'; return; }

      const activeSolId = _currentSolutionId || null;

      // Sort: active solution first, then others
      const mine   = all.filter(f => f.solution_id && f.solution_id === activeSolId);
      const others = all.filter(f => !f.solution_id || f.solution_id !== activeSolId);

      container.style.display = 'block';
      container.innerHTML = '<div style="font-size:0.78rem;color:#888;margin-bottom:0.3rem;font-weight:500;">Recent files</div>';

      const makeRow = (f, dimmed) => {
        const name = f.path.split('/').pop();
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:0.4rem;margin-bottom:0.2rem;';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.title = f.path;
        btn.style.cssText = `flex:1;text-align:left;padding:0.3rem 0.6rem;background:${dimmed ? '#f8f9fa' : '#f1f3f5'};border:1px solid #dee2e6;border-radius:5px;cursor:pointer;font-size:0.84rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:${dimmed ? '0.7' : '1'};`;
        btn.onmouseover = () => btn.style.background = '#e2e6ea';
        btn.onmouseout  = () => btn.style.background = dimmed ? '#f8f9fa' : '#f1f3f5';

        // Label: filename + solution badge if from a different solution
        let label = '📄 ' + name;
        if (dimmed && f.solution_name) label += '  ·  ' + f.solution_name;
        btn.textContent = label;

        btn.onclick = async () => {
          // If file belongs to a different solution, switch the global solution
          if (f.solution_id && f.solution_id !== _currentSolutionId) {
            document.getElementById('globalSolution').value = f.solution_id;
            await onGlobalSolutionChange();
          }
          setSelectedPath(f.path);
          addRecentFile(f.path);  // bump to top + update solution tag
          await checkForSavedState(f.path);
        };

        const del = document.createElement('button');
        del.type = 'button';
        del.textContent = '✕';
        del.title = 'Remove from recent';
        del.style.cssText = 'padding:0.2rem 0.45rem;background:none;border:none;color:#aaa;cursor:pointer;font-size:0.8rem;border-radius:4px;';
        del.onmouseover = () => del.style.color = '#d32f2f';
        del.onmouseout  = () => del.style.color = '#aaa';
        del.onclick = (e) => { e.stopPropagation(); removeRecentFile(f.path); };

        row.appendChild(btn);
        row.appendChild(del);
        container.appendChild(row);
      };

      mine.forEach(f => makeRow(f, false));

      if (mine.length && others.length) {
        const div = document.createElement('div');
        div.style.cssText = 'border-top:1px solid #e9ecef;margin:0.35rem 0 0.35rem 0;';
        container.appendChild(div);
      }

      others.forEach(f => makeRow(f, true));
    }

    // Render on page load
    renderRecentFiles();

    // Call the backend to open a native file dialog
    async function browseFile() {
      const st = document.getElementById('sourceType').value;
      const btn = document.getElementById('btnBrowse');
      btn.textContent = '…';
      btn.disabled = true;
      try {
        const res = await api('/api/pick-file?source_type=' + st);
        if (res.path) {
          setSelectedPath(res.path);
          addRecentFile(res.path);
          await checkForSavedState(res.path);
          // Persist file_path to source definition
          await _persistSourceFilePath(res.path);
        }
      } finally {
        btn.textContent = '📂 Browse…';
        btn.disabled = false;
      }
    }

    async function _persistSourceFilePath(filePath) {
      const solId = _currentSolutionId;
      const collName = document.getElementById('collectionSelect').value;
      const srcId = _selectedSourceId;
      if (!solId || !collName || collName === '__new__' || !srcId) return;
      try {
        await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections/' + encodeURIComponent(collName) + '/sources/' + encodeURIComponent(srcId), {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_path: filePath })
        });
        // Update local cache
        const coll = _currentCollections[collName];
        if (coll && coll.sources) {
          const src = coll.sources.find(s => s.id === srcId);
          if (src) src.file_path = filePath;
        }
      } catch(e) { /* silent — not critical */ }
    }

    let _savedStatePath = null;
    let _savedStateInfo = null;

    function setSelectedPath(path) {
      document.getElementById('sourcePath').value = path;
      const display = document.getElementById('sourcePathDisplay');
      const fullHint = document.getElementById('sourcePathFull');
      if (path) {
        const name = path.split('/').pop();
        display.textContent = '📄 ' + name;
        display.style.color = '#1a1a1a';
        fullHint.textContent = path;
        fullHint.title = path;
        fullHint.style.display = 'block';
      } else {
        display.textContent = 'Click to select a file…';
        display.style.color = '#888';
        fullHint.style.display = 'none';
      }
    }

    async function onPathChange() {
      const path = document.getElementById('sourcePath').value.trim();
      if (path.length > 5) {
        await checkForSavedState(path);
        if (path.includes('.') && !path.endsWith('/')) addRecentFile(path);
      } else {
        hideBanner();
      }
    }

    async function checkForSavedState(path) {
      try {
        const res = await fetch('/api/state/check?path=' + encodeURIComponent(path)).then(r => r.json());
        if (res.found) {
          _savedStatePath = res.save_path;
          _savedStateInfo = res;
          const steps = res.completed_steps || [];
          const info = steps.length
            ? `Steps done: <strong>${steps.join(', ')}</strong>. ${res.chunks_count ? res.chunks_count + ' chunks ready.' : ''}`
            : 'Empty state file found.';
          document.getElementById('resumeInfo').innerHTML = info;
          document.getElementById('resumeBanner').style.display = 'block';
          // Pre-fill collection name if available
          if (res.collection_name && !document.getElementById('collectionName').value.trim()) {
            document.getElementById('collectionName').value = res.collection_name;
          }
        } else {
          hideBanner();
        }
      } catch(e) { hideBanner(); }
    }

    function hideBanner() {
      _savedStatePath = null;
      document.getElementById('resumeBanner').style.display = 'none';
    }

    function dismissResume() {
      hideBanner();
    }

    async function resumeState() {
      if (!_savedStatePath) return;
      setLog(buildLog, 'Resuming saved state…', false);
      try {
        const res = await fetch('/api/state/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ save_path: _savedStatePath })
        }).then(r => r.json());
        setLog(buildLog, res.message, false);
        if (res.state && res.state.collection_name) {
          const cn = document.getElementById('collectionName');
          if (cn && !cn.value.trim()) cn.value = res.state.collection_name;
        }
        hideBanner();
      } catch(e) {
        setLog(buildLog, 'Failed to resume: ' + e.message, true);
      }
    }

    const getStateUpdate = () => {
      const st = document.getElementById('sourceType').value;
      const path = document.getElementById('sourcePath').value.trim();
      const scraper = document.getElementById('scraperName').value.trim();
      const chunkMode = document.querySelector('input[name="chunkMode"]:checked').value;
      let source_config = {};
      if (st === 'url') {
        const engine = (document.querySelector('input[name="scraperEngine"]:checked') || {}).value || 'playwright';
        source_config = { scraper_name: scraper || 'peixefresco', source_label: scraper || scraper, engine };
        if (engine === 'shopify') {
          source_config.shop_url = (document.getElementById('shopUrl') || {}).value?.trim() || '';
        }
        // Attach inline scraper_config if one was set by the collection picker (wizard-created collections)
        if (window._activeScraperConfig) source_config.scraper_config = window._activeScraperConfig;
      } else if (path) source_config = { path, source_label: path.split('/').pop() };
      const embeddingModel = (document.getElementById('embeddingModel') || {}).value || 'text-embedding-ada-002';
      return {
        collection_name: getCollectionName(),
        source_type: st,
        source_id: _selectedSourceId || null,
        source_config,
        embedding_model: embeddingModel,
        chunking_config: {
          use_proposition_chunking: chunkMode === 'proposition',
          use_hierarchical_chunking: chunkMode === 'hierarchical'
        }
      };
    };

    const _pipelineBtns = () => ['runCreate','runFetch','runTranslate','runChunk','runPush','runSync'].map(id => document.getElementById(id)).filter(Boolean);
    const _clearDone    = () => _pipelineBtns().forEach(b => { if (b.dataset.done) { b.style.background = ''; b.style.color = ''; delete b.dataset.done; } });
    const _btnRunning   = (btn) => { _clearDone(); btn.disabled = true; btn.style.background = '#e65c00'; btn.style.color = '#fff'; };
    const _btnDone      = (btn) => { btn.disabled = false; btn.style.background = ''; btn.style.color = ''; };
    const _btnSuccess   = (btn) => { btn.disabled = false; btn.style.background = '#2e7d32'; btn.style.color = '#fff'; btn.dataset.done = '1'; };

    const runStep = async (step) => {
      const stepBtnMap = { chunk: 'runChunk', push_to_qdrant: 'runPush', create_collection: 'runCreate' };
      const stepLabelMap = { chunk: 'Chunking…', push_to_qdrant: 'Pushing to Qdrant…', create_collection: 'Creating collection…' };
      const stepBtn = stepBtnMap[step] ? document.getElementById(stepBtnMap[step]) : null;
      if (stepBtn) _btnRunning(stepBtn);
      setLog(buildLog, stepLabelMap[step] || 'Running…', false);
      try {
        const res = await api('/api/workflow/step', { step, state_update: getStateUpdate() });
        const msg = res.message || res.detail || JSON.stringify(res);
        const isError = msg.includes('Error') || !!res.detail;
        if (stepBtn) { isError ? _btnDone(stepBtn) : _btnSuccess(stepBtn); }
        setLog(buildLog, msg, isError);
        // After chunk step: render collection metadata card if available
        if (step === 'chunk' && res.state && res.state.collection_metadata) {
          renderMetadataCard(res.state.collection_metadata);
        }
        // After chunk: auto-save routing metadata if a solution + collection is selected
        if (step === 'chunk' && res.state && res.state.collection_metadata) {
          const solId = _currentSolutionId;
          const collSelect = document.getElementById('collectionSelect');
          if (solId && collSelect && collSelect.value && collSelect.value !== '__new__') {
            const collId = (collSelect.options[collSelect.selectedIndex] || {}).dataset?.collId || collSelect.value;
            await autoSaveRoutingMetadata(solId, collId, res.state.collection_metadata);
          }
        }
      } catch (e) {
        if (stepBtn) _btnDone(stepBtn);
        setLog(buildLog, e.message || String(e), true);
      }
    };

    async function _loadFaqCollections() {
      const container = document.getElementById('faqCollList');
      if (!container) return;
      try {
        const sols = await api('/api/solutions');
        const faqColls = [];
        for (const sol of (sols.solutions || [])) {
          for (const coll of (sol.collections || [])) {
            if ((coll.routing || {}).doc_type === 'faq') {
              faqColls.push({collection_name: coll.collection_name, display_name: coll.display_name, solution: sol.display_name, language: sol.language || 'en', company: sol.company_name || sol.display_name});
            }
          }
        }
        if (!faqColls.length) {
          container.innerHTML = '<em>No FAQ collections found. Create a collection with doc_type: faq to use this feature.</em>';
          return;
        }
        container.innerHTML = '';
        for (const fc of faqColls) {
          const row = document.createElement('div');
          row.style.cssText = 'display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem;padding:0.4rem 0.55rem;background:#f8f9fa;border-radius:6px;border:1px solid #eee;';
          row.innerHTML = `<span style="flex:1;font-size:0.83rem;"><strong>${fc.display_name}</strong> <span style="color:#888;font-size:0.76rem;">(${fc.solution})</span></span>
            <button onclick="generateFaqTable(${JSON.stringify(fc.collection_name)},${JSON.stringify(fc.company)},${JSON.stringify(fc.language)})" style="font-size:0.8rem;padding:0.22rem 0.6rem;">Generate Table</button>`;
          container.appendChild(row);
        }
      } catch(e) {
        container.innerHTML = '<em style="color:#c00;">Error loading solutions: ' + e.message + '</em>';
      }
    }

    async function generateFaqTable(collName, company, lang) {
      const modal = document.getElementById('faqTableModal');
      const status = document.getElementById('faqTableStatus');
      const content = document.getElementById('faqTableContent');
      const copyBtn = document.getElementById('btnCopyFaq');
      document.getElementById('faqTableCollName').textContent = collName;
      status.style.color = '#666';  // reset from any previous error
      status.textContent = 'Generating… (this may take 10-20 seconds)';
      content.style.display = 'none';
      copyBtn.style.display = 'none';
      modal.style.display = 'flex';
      try {
        const res = await fetch('/api/faq/generate-table', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({collection_name: collName, company_name: company || 'the company', language: lang || 'en', max_items: 100}),
        });
        const data = await res.json();
        if (data.error) { status.style.color = '#c0392b'; status.textContent = '❌ ' + data.error; return; }
        if (!data.count) { status.style.color = '#e67e22'; status.textContent = '⚠️ No FAQ pairs found in this collection.'; return; }
        content.value = data.table;
        content.style.display = '';
        copyBtn.style.display = '';
        const sourceNote = data.source ? ` (from ${data.source})` : '';
        status.style.color = '#2e7d32';
        status.textContent = `✅ ${data.count} Q&A pair${data.count !== 1 ? 's' : ''} extracted${sourceNote}`;
      } catch(e) {
        status.style.color = '#c0392b';
        status.textContent = '❌ Error: ' + (e.message || e);
      }
    }

    function copyFaqTable() {
      const ta = document.getElementById('faqTableContent');
      if (!ta) return;
      navigator.clipboard.writeText(ta.value).then(() => {
        const btn = document.getElementById('btnCopyFaq');
        if (btn) { const orig = btn.textContent; btn.textContent = '✓ Copied!'; setTimeout(() => btn.textContent = orig, 1500); }
      }).catch(() => { ta.select(); document.execCommand('copy'); });
    }

    function renderRelevanceCard(report) {
      const existing = document.getElementById('relevanceCard');
      if (existing) existing.remove();
      if (!report) return;
      const { relevant_count = 0, mismatch_count = 0, irrelevant_count = 0,
              mismatch_urls = [], irrelevant_urls = [] } = report;
      if (relevant_count === 0 && mismatch_count === 0 && irrelevant_count === 0) return;
      const card = document.createElement('div');
      card.id = 'relevanceCard';
      card.style.cssText = 'margin-top:0.6rem;background:#f5fff5;border:1px solid #a5d6a7;border-radius:8px;padding:0.75rem 1rem;font-size:0.85rem;';
      const mkLinks = (urls) => urls.length
        ? urls.map(u => `<a href="${u}" target="_blank" style="color:#0066cc;">${u}</a>`).join('<br>')
        : '';
      const mismatchSection = mismatch_urls.length
        ? `<details style="margin-top:0.4rem;"><summary style="cursor:pointer;color:#7a5a00;">⚠️ ${mismatch_count} flagged for review</summary><div style="margin-top:0.3rem;padding-left:0.75rem;">${mkLinks(mismatch_urls)}</div></details>`
        : '';
      const irrelevantSection = irrelevant_urls.length
        ? `<details style="margin-top:0.3rem;"><summary style="cursor:pointer;color:#b71c1c;">❌ ${irrelevant_count} pushed to 'not_relevant'</summary><div style="margin-top:0.3rem;padding-left:0.75rem;">${mkLinks(irrelevant_urls)}</div></details>`
        : '';
      card.innerHTML = `
        <div style="font-weight:600;margin-bottom:0.4rem;color:#2e7d32;">🔍 Relevance Check</div>
        <div>✅ ${relevant_count} relevant &nbsp;·&nbsp; ⚠️ ${mismatch_count} flagged &nbsp;·&nbsp; ❌ ${irrelevant_count} irrelevant</div>
        ${mismatchSection}${irrelevantSection}
      `;
      buildLog.parentNode.insertBefore(card, buildLog.nextSibling);
    }

    function renderMetadataCard(meta) {
      const existing = document.getElementById('metadataCard');
      if (existing) existing.remove();
      if (!meta || !Object.keys(meta).length) return;
      const card = document.createElement('div');
      card.id = 'metadataCard';
      card.style.cssText = 'margin-top:0.75rem;background:#f0f7ff;border:1px solid #b3d4f5;border-radius:8px;padding:0.85rem 1rem;font-size:0.86rem;';
      const topics = (meta.topics || []).map(t => `<span style="display:inline-block;background:#d0e8ff;border-radius:4px;padding:0.1rem 0.45rem;margin:0.1rem 0.2rem 0.1rem 0;">${t}</span>`).join('');
      const keywords = (meta.keywords || []).join(', ');
      card.innerHTML = `
        <div style="font-weight:600;margin-bottom:0.5rem;color:#1565c0;">📋 Collection Metadata</div>
        <div style="margin-bottom:0.35rem;"><strong>Type:</strong> ${meta.doc_type || '—'} &nbsp;·&nbsp; <strong>Language:</strong> ${meta.language || '—'}</div>
        <div style="margin-bottom:0.4rem;"><em>${meta.description || ''}</em></div>
        <div style="margin-bottom:0.35rem;"><strong>Topics:</strong> ${topics || '—'}</div>
        <div style="color:#555;"><strong>Keywords:</strong> ${keywords || '—'}</div>
      `;
      buildLog.parentNode.insertBefore(card, buildLog.nextSibling);
    }

    function toggleWizardLogin() {
      const sec = document.getElementById('wizardLoginSection');
      if (sec) sec.style.display = sec.style.display !== 'none' ? 'none' : '';
    }

    function _buildWizardLoginConfig() {
      const username = (document.getElementById('wizardLoginUsername') || {}).value || '';
      if (!username.trim()) return null;
      return {
        username: username.trim(),
        password: (document.getElementById('wizardLoginPassword') || {}).value || '',
        url: (document.getElementById('wizardLoginUrl') || {}).value || '',
      };
    }

    function toggleLoginSection() {
      const sec = document.getElementById('loginSection');
      const btn = document.getElementById('loginToggleBtn');
      if (sec) {
        const visible = sec.style.display !== 'none';
        sec.style.display = visible ? 'none' : '';
        if (btn) btn.style.borderStyle = visible ? 'dashed' : 'solid';
        if (btn) btn.style.color = visible ? '#666' : '#7a4a00';
      }
    }

    function _buildLoginConfig() {
      const username = (document.getElementById('loginUsername') || {}).value || '';
      if (!username.trim()) return null;
      return {
        username: username.trim(),
        password: (document.getElementById('loginPassword') || {}).value || '',
        url: (document.getElementById('loginUrl') || {}).value || '',
        username_selector: (document.getElementById('loginUserSel') || {}).value || '',
        password_selector: (document.getElementById('loginPassSel') || {}).value || '',
        submit_selector: (document.getElementById('loginSubmitSel') || {}).value || '',
      };
    }

    document.getElementById('runCreate').onclick = () => runStep('create_collection');
    document.getElementById('runFetch').onclick = () => runFetchWithProgress();
    document.getElementById('runTranslate').onclick = () => runTranslateWithProgress();
    document.getElementById('runChunk').onclick = () => runStep('chunk');
    document.getElementById('runPush').onclick = () => runPushWithProgress();
    document.getElementById('runSync').onclick = () => runSync();

    // Disable Push + Sync in DEV mode (don't set .disabled — it swallows click events)
    if ("__DEV_MODE__" === "1") {
      ['runPush', 'runSync'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
          btn.style.opacity = '0.45';
          btn.style.cursor = 'not-allowed';
          btn.title = 'Push to Qdrant is disabled in DEV mode';
          btn.onclick = (e) => { e.preventDefault(); e.stopPropagation(); setLog(buildLog, '⚠ Push to Qdrant is disabled in DEV mode. Merge to main and push from the production server.', true); };
        }
      });
    }

    async function runFetchWithProgress() {
      const btn = document.getElementById('runFetch');
      _btnRunning(btn);
      btn.textContent = '2. Fetching…';
      setLog(buildLog, 'Fetching… (this may take a few minutes for large sites)', false);
      buildLog.style.maxHeight = '20rem';

      const es = new EventSource('/api/progress');
      es.onmessage = (e) => {
        const data = e.data;
        if (data.startsWith('LOG:')) {
          const line = data.replace('LOG:', '');
          buildLog.textContent += (buildLog.textContent ? '\\n' : '') + line;
          buildLog.scrollTop = buildLog.scrollHeight;
        } else if (data.startsWith('DONE:')) {
          const msg = data.replace('DONE:', '');
          buildLog.textContent += '\\n✅ ' + msg;
          buildLog.scrollTop = buildLog.scrollHeight;
          buildLog.className = 'log success';
          es.close();
          _btnSuccess(btn); btn.textContent = '2. Fetch';
          // Fetch updated state to render relevance report card (if check ran)
          api('/api/workflow/state').then(st => {
            if (st && st.relevance_report) renderRelevanceCard(st.relevance_report);
          }).catch(() => {});
        } else if (data.startsWith('ERROR') || data === 'TIMEOUT') {
          buildLog.textContent += '\\n❌ ' + data;
          buildLog.className = 'log error';
          es.close();
          _btnDone(btn); btn.textContent = '2. Fetch';
        }
      };
      es.onerror = () => { es.close(); _btnDone(btn); btn.textContent = '2. Fetch'; };

      try {
        const _lc = _buildLoginConfig();
        await api('/api/workflow/fetch', { step: 'fetch', state_update: getStateUpdate(), ...(_lc ? {login_config: _lc} : {}) });
      } catch (e) {
        setLog(buildLog, e.message || String(e), true);
        es.close();
        btn.disabled = false;
        btn.textContent = '2. Fetch';
      }
    }

    async function runPushWithProgress(skipUrls) {
      const btn = document.getElementById('runPush');

      // Push guard: check for manually edited chunks before pushing
      if (!skipUrls) {
        const collName = document.getElementById('collectionSelect').value;
        const sourceId = _selectedSourceId;
        if (collName && sourceId) {
          try {
            const data = await api('/api/collections/' + encodeURIComponent(collName) + '/edited-chunks?source_id=' + encodeURIComponent(sourceId));
            if (data.edited_urls && data.edited_urls.length > 0) {
              // Build a temporary src object for the modal
              const sources = (_currentCollections[collName] || {}).sources || [];
              const src = sources.find(s => s.id === sourceId) || { id: sourceId, label: sourceId };
              _showPushGuardModal(data, src, (urls) => runPushWithProgress(urls));
              return;
            }
          } catch(_) { /* proceed without guard */ }
        }
        skipUrls = [];
      }

      _btnRunning(btn);
      setLog(buildLog, 'Pushing to Qdrant… (embedding each chunk with OpenAI)', false);
      buildLog.style.maxHeight = '20rem';

      const es = new EventSource('/api/progress');
      es.onmessage = (e) => {
        const data = e.data;
        if (data.startsWith('LOG:')) {
          const line = data.replace('LOG:', '');
          buildLog.textContent += (buildLog.textContent ? '\\n' : '') + line;
          buildLog.scrollTop = buildLog.scrollHeight;
        } else if (data.startsWith('DONE:')) {
          const msg = data.replace('DONE:', '');
          buildLog.textContent += '\\n✅ ' + msg;
          buildLog.scrollTop = buildLog.scrollHeight;
          buildLog.className = 'log success';
          es.close();
          _btnSuccess(btn);
          // Refresh collection view to update status badges
          if (_currentSolutionId) loadSolutionCollections(_currentSolutionId);
        } else if (data.startsWith('ERROR') || data === 'TIMEOUT') {
          buildLog.textContent += '\\n❌ ' + data;
          buildLog.className = 'log error';
          es.close();
          _btnDone(btn);
        }
      };
      es.onerror = () => { es.close(); _btnDone(btn); };

      try {
        const stateUpdate = getStateUpdate();
        if (skipUrls && skipUrls.length > 0) stateUpdate.skip_urls = skipUrls;
        await api('/api/workflow/push', { step: 'push_to_qdrant', state_update: stateUpdate });
      } catch (e) {
        setLog(buildLog, e.message || String(e), true);
        es.close();
        _btnDone(btn);
      }
    }

    async function runTranslateWithProgress() {
      const btn = document.getElementById('runTranslate');
      const wrap = document.getElementById('translateProgress');
      const bar = document.getElementById('translateBar');
      const label = document.getElementById('translateLabel');

      _btnRunning(btn);
      wrap.style.display = 'block';
      bar.style.width = '0%';
      label.textContent = 'Starting…';
      setLog(buildLog, 'Translating & cleaning… (this takes 1–3 minutes)', false);

      // Start SSE listener first
      const es = new EventSource('/api/progress');
      es.onmessage = (e) => {
        const data = e.data;
        if (data.startsWith('PROGRESS:')) {
          const [cur, tot] = data.replace('PROGRESS:', '').split('/').map(Number);
          const pct = Math.round((cur / tot) * 100);
          bar.style.width = pct + '%';
          label.textContent = `Batch ${cur} of ${tot} (${pct}%)`;
        } else if (data.startsWith('DONE:')) {
          const msg = data.replace('DONE:', '');
          bar.style.width = '100%';
          label.textContent = 'Done!';
          setLog(buildLog, msg, false);
          es.close();
          _btnSuccess(btn);
        } else if (data.startsWith('ERROR') || data === 'TIMEOUT') {
          es.close();
          _btnDone(btn);
        }
      };
      es.onerror = () => { es.close(); _btnDone(btn); };

      // Kick off translate in background thread (returns immediately)
      try {
        await api('/api/workflow/translate', { step: 'translate_and_clean', state_update: getStateUpdate() });
      } catch (e) {
        setLog(buildLog, e.message || String(e), true);
        es.close();
        btn.disabled = false;
      }
    }
    async function runSync() {
      const btn = document.getElementById('runSync');
      const resultDiv = document.getElementById('syncResult');
      const st = getStateUpdate();
      const scraperName = (st.source_config || {}).scraper_name || document.getElementById('scraperName')?.value || '';
      const collName = getCollectionName();
      if (!scraperName) { resultDiv.textContent = '❌ No scraper selected.'; resultDiv.className = ''; resultDiv.style.background = '#fdecea'; return; }
      if (!collName) { resultDiv.textContent = '❌ No collection selected.'; resultDiv.className = ''; resultDiv.style.background = '#fdecea'; return; }

      _btnRunning(btn);
      resultDiv.className = '';
      resultDiv.style.background = '#fffbe6';
      resultDiv.textContent = '⏳ Syncing — re-scraping all URLs and comparing hashes…';

      try {
        const res = await api('/api/sync', { scraper_name: scraperName, collection_name: collName, scraper_options: st.source_config || {} });
        const diff = res.diff || {};
        const added   = diff.added   || 0;
        const updated = diff.updated || 0;
        const deleted = diff.deleted || 0;
        const unchanged = diff.unchanged || 0;
        const errors  = diff.errors  || [];

        const parts = [];
        if (added)     parts.push(`<span style="color:#2e7d32;font-weight:600;">+${added} added</span>`);
        if (updated)   parts.push(`<span style="color:#e65c00;font-weight:600;">~${updated} updated</span>`);
        if (deleted)   parts.push(`<span style="color:#c62828;font-weight:600;">-${deleted} deleted</span>`);
        if (unchanged) parts.push(`<span style="color:#555;">${unchanged} unchanged</span>`);
        if (errors.length) parts.push(`<span style="color:#c62828;">${errors.length} error(s)</span>`);

        resultDiv.innerHTML = '✅ Sync complete: ' + (parts.join(' &nbsp;·&nbsp; ') || 'no changes');
        if (errors.length) {
          resultDiv.innerHTML += '<br><small>' + errors.slice(0, 3).map(e => '⚠ ' + e).join('<br>') + '</small>';
          resultDiv.style.background = '#fff3e0';
        } else {
          resultDiv.style.background = '#e8f5e9';
        }
        _btnSuccess(btn);
      } catch (e) {
        resultDiv.textContent = '❌ ' + (e.message || String(e));
        resultDiv.style.background = '#fdecea';
        _btnDone(btn);
      }
    }

    document.getElementById('runReset').onclick = async () => {
      await api('/api/workflow/reset', {});
      setLog(buildLog, 'State reset.', false);
    };

    // onChatSolutionChange kept as backward-compat alias
    async function onChatSolutionChange() {
      await _loadChatCollections(_currentSolutionId);
    }

    document.getElementById('runQA').onclick = async () => {
      const solId = _currentSolutionId || null;
      const sol = solId ? _allSolutions.find(s => s.id === solId) : null;
      const company = (sol && sol.company_name) ? sol.company_name : 'Assistant';
      const collection = document.getElementById('chatCollectionSelect').value.trim();
      const question = document.getElementById('qaQuestion').value.trim();
      if (!collection || !question) {
        setLog(qaResult, 'Pick a solution + collection and enter a question.', true);
        return;
      }
      setLog(qaResult, '…', false);
      try {
        const embeddingModel = (document.getElementById('chatEmbeddingModel') || {}).value || 'text-embedding-ada-002';
        const body = { collection_name: collection, question, company, embedding_model: embeddingModel };
        if (collection === '__all__' && solId) body.solution_id = solId;
        const res = await api('/api/qa', body);

        // Render answer text
        qaResult.textContent = 'Q: ' + res.question + '\\n\\nA: ' + res.answer;
        qaResult.classList.remove('error');
        qaResult.classList.add('success');

        // Render source attribution chips below the answer
        const existingSources = document.getElementById('qaSources');
        if (existingSources) existingSources.remove();
        const sources = res.sources || [];
        if (sources.length > 0) {
          const sourcesDiv = document.createElement('div');
          sourcesDiv.id = 'qaSources';
          sourcesDiv.style.cssText = 'margin-top:0.75rem;display:flex;flex-wrap:wrap;gap:0.4rem;align-items:center;';
          const label = document.createElement('span');
          label.textContent = 'Sources:';
          label.style.cssText = 'font-size:0.8rem;color:#888;font-weight:600;';
          sourcesDiv.appendChild(label);
          sources.forEach(src => {
            const a = document.createElement('a');
            a.href = src.url;
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            a.textContent = (src.type === 'pdf' ? '📄 ' : '🔗 ') + src.label + ' ↗';
            a.style.cssText = 'display:inline-block;padding:0.2rem 0.55rem;border-radius:12px;font-size:0.78rem;text-decoration:none;background:' + (src.type === 'pdf' ? '#fff3e0' : '#e8f4fd') + ';color:' + (src.type === 'pdf' ? '#e65100' : '#1565c0') + ';border:1px solid ' + (src.type === 'pdf' ? '#ffcc80' : '#90caf9') + ';';
            sourcesDiv.appendChild(a);
          });
          qaResult.insertAdjacentElement('afterend', sourcesDiv);
        }
      } catch (e) {
        setLog(qaResult, e.message || String(e), true);
      }
    };

    // ── Shopify Stores tab ────────────────────────────────────────────────────

    let _editingStoreId = null;

    async function loadShopifyStores() {
      const data = await fetch('/api/shopify/stores').then(r => r.json());
      renderStoreList(data.stores || []);
    }

    function renderStoreList(stores) {
      const el = document.getElementById('shopify-store-list');
      if (!stores.length) {
        el.innerHTML = '<p class="status">No stores yet. Click "+ Add Store" to get started.</p>';
        return;
      }
      el.innerHTML = stores.map(renderStoreCard).join('');
    }

    function renderStoreCard(s) {
      const mode = s.has_token
        ? '<span class="badge badge-admin">🟢 Admin API</span>'
        : '<span class="badge badge-public">⚪ Public</span>';
      const lastFetched = s.last_fetched
        ? 'Last fetched: ' + _relativeTime(s.last_fetched)
        : 'Never fetched';
      const includes = (s.include || []).join(', ');
      return `
        <div class="store-card" id="store-card-${s.id}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <strong>${s.display_name}</strong>&nbsp;${mode}
              <div class="store-meta">${s.shop_url}</div>
              <div class="store-meta">Include: ${includes}&nbsp;·&nbsp;${lastFetched}</div>
            </div>
            <div style="display:flex;gap:0.4rem;flex-shrink:0">
              <button class="btn-secondary" onclick="startEditStore('${s.id}')">Edit</button>
              <button class="btn-secondary" style="color:#c0392b" onclick="deleteStore('${s.id}','${s.display_name.replace(/'/g,"\\'")}',this)">Delete</button>
            </div>
          </div>
          <div style="margin-top:0.6rem;display:flex;gap:0.5rem">
            <button class="btn-secondary" onclick="testStore('${s.id}')">Test Connection</button>
            <button class="btn-primary" onclick="fetchStore('${s.id}')">Fetch Now</button>
          </div>
          <div id="store-result-${s.id}"></div>
        </div>`;
    }

    function _relativeTime(iso) {
      const diff = Date.now() - new Date(iso).getTime();
      const mins = Math.floor(diff / 60000);
      if (mins < 2) return 'just now';
      if (mins < 60) return mins + 'm ago';
      const hrs = Math.floor(mins / 60);
      if (hrs < 24) return hrs + 'h ago';
      return Math.floor(hrs / 24) + 'd ago';
    }

    function showAddStoreForm() {
      _editingStoreId = null;
      document.getElementById('shopify-form-title').textContent = 'Add Store';
      document.getElementById('shopify-form-display-name').value = '';
      document.getElementById('shopify-form-url').value = '';
      document.getElementById('shopify-form-token').value = '';
      document.getElementById('shopify-form-token').placeholder = 'shpat_... (leave blank for public /products.json API)';
      ['products','pages','articles'].forEach(k => {
        document.getElementById('shopify-include-' + k).checked = true;
      });
      document.getElementById('shopify-metafields').checked = false;
      document.getElementById('shopify-form-card').classList.remove('hidden');
      document.getElementById('shopify-form-display-name').focus();
    }

    function startEditStore(storeId) {
      fetch('/api/shopify/stores').then(r => r.json()).then(data => {
        const s = data.stores.find(x => x.id === storeId);
        if (!s) return;
        _editingStoreId = storeId;
        document.getElementById('shopify-form-title').textContent = 'Edit Store';
        document.getElementById('shopify-form-display-name').value = s.display_name;
        document.getElementById('shopify-form-url').value = s.shop_url;
        document.getElementById('shopify-form-token').value = '';
        document.getElementById('shopify-form-token').placeholder = s.has_token
          ? 'Keep existing token (ends in \u2026' + s.token_hint + ') — or type new token to replace'
          : 'shpat_... (leave blank for public API)';
        ['products','pages','articles'].forEach(k => {
          document.getElementById('shopify-include-' + k).checked = (s.include || []).includes(k);
        });
        document.getElementById('shopify-metafields').checked = !!s.metafields;
        document.getElementById('shopify-form-card').classList.remove('hidden');
        document.getElementById('shopify-form-card').scrollIntoView({ behavior: 'smooth' });
      });
    }

    async function saveStore() {
      const body = {
        display_name: document.getElementById('shopify-form-display-name').value.trim(),
        shop_url: document.getElementById('shopify-form-url').value.trim(),
        access_token: document.getElementById('shopify-form-token').value,
        include: ['products','pages','articles'].filter(k => document.getElementById('shopify-include-' + k).checked),
        metafields: document.getElementById('shopify-metafields').checked,
      };
      if (!body.display_name || !body.shop_url) { alert('Display name and Shop URL are required.'); return; }
      const url = _editingStoreId ? '/api/shopify/stores/' + _editingStoreId : '/api/shopify/stores';
      const method = _editingStoreId ? 'PUT' : 'POST';
      const res = await fetch(url, { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const data = await res.json();
      if (!res.ok) { alert(data.detail || 'Error saving store'); return; }
      cancelStoreForm();
      loadShopifyStores();
    }

    function cancelStoreForm() {
      document.getElementById('shopify-form-card').classList.add('hidden');
      _editingStoreId = null;
    }

    async function deleteStore(storeId, name, btn) {
      _inlineConfirm(btn, {
        message: 'Delete store "' + name + '"? This cannot be undone.',
        confirmLabel: 'Delete',
        onConfirm: async () => {
          await fetch('/api/shopify/stores/' + storeId, { method: 'DELETE' });
          loadShopifyStores();
        }
      });
    }

    async function testStore(storeId) {
      const el = document.getElementById('store-result-' + storeId);
      el.innerHTML = '<span class="status">Testing connection…</span>';
      try {
        const d = await fetch('/api/shopify/stores/' + storeId + '/test', { method: 'POST' }).then(r => r.json());
        if (d.ok) {
          const info = d.mode === 'admin'
            ? '✅ Connected &nbsp;·&nbsp; ' + d.shop_name + ' &nbsp;·&nbsp; ' + d.products + ' products, ' + d.pages + ' pages'
            : '✅ Connected (public API) &nbsp;·&nbsp; ' + d.products;
          el.innerHTML = '<div class="inline-result ok">' + info + '</div>';
        } else {
          el.innerHTML = '<div class="inline-result err">❌ ' + d.error + '</div>';
        }
      } catch(e) {
        el.innerHTML = '<div class="inline-result err">❌ ' + e + '</div>';
      }
    }

    async function fetchStore(storeId) {
      const el = document.getElementById('store-result-' + storeId);
      el.innerHTML = '<div class="log" id="shopify-fetch-log-' + storeId + '" style="max-height:160px;overflow-y:auto;font-size:0.8rem;padding:0.4rem"></div>';
      const log = document.getElementById('shopify-fetch-log-' + storeId);
      try {
        const res = await fetch('/api/shopify/stores/' + storeId + '/fetch', { method: 'POST' });
        if (!res.ok) { log.textContent = 'Failed to start fetch.'; return; }
        const sse = new EventSource('/api/progress');
        sse.onmessage = e => {
          const msg = e.data;
          if (msg.startsWith('LOG:')) {
            log.textContent += msg.slice(4) + '\\n';
            log.scrollTop = log.scrollHeight;
          } else if (msg.startsWith('DONE:')) {
            log.textContent += '✅ ' + msg.slice(5) + '\\n';
            sse.close();
            loadShopifyStores();
          } else if (msg.startsWith('ERROR:')) {
            log.textContent += '❌ ' + msg.slice(6) + '\\n';
            sse.close();
          }
        };
        sse.onerror = () => { log.textContent += 'Connection lost.\\n'; sse.close(); };
      } catch(e) {
        log.textContent = 'Error: ' + e;
      }
    }

    // Note: shopify, wizard, and sites tab side-effects are now handled in showTab()

    function _wizardPopulateSolNameList() {
      const dl = document.getElementById('wizardSolNameList');
      if (!dl) return;
      dl.innerHTML = '';
      (_allSolutions || []).forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.display_name || s.id;
        opt.setAttribute('data-id', s.id);
        dl.appendChild(opt);
      });
    }

    function _wizardOnSolNameInput(val) {
      // Keep cached solId in sync with what's typed
      _wizardSolId = val.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/,'');
      // If the typed value matches an existing solution, auto-set its language
      const match = (_allSolutions || []).find(s => (s.display_name || s.id) === val);
      if (match && match.language) {
        const langSel = document.getElementById('wizardLang');
        if (langSel) langSel.value = match.language;
      }
    }

    function _dismissShutdownOverlay() {
      const el = document.getElementById('shutdownOverlay');
      if (el) el.remove();
    }

    function _showShutdownOverlay(title, body) {
      if (document.getElementById('shutdownOverlay')) return;
      const overlay = document.createElement('div');
      overlay.id = 'shutdownOverlay';
      overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.75);display:flex;align-items:center;justify-content:center;z-index:9999';
      const box = document.createElement('div');
      box.style.cssText = 'background:#fff;border-radius:12px;padding:2rem 2.5rem;text-align:center;max-width:380px;position:relative';
      const xBtn = document.createElement('button');
      xBtn.textContent = '\u2715';
      xBtn.title = 'Dismiss';
      xBtn.style.cssText = 'position:absolute;top:0.6rem;right:0.75rem;background:none;border:none;font-size:1.2rem;color:#aaa;cursor:pointer;line-height:1';
      xBtn.onclick = _dismissShutdownOverlay;
      const icon = document.createElement('div');
      icon.textContent = '\u23f9';
      icon.style.cssText = 'font-size:2rem;margin-bottom:0.5rem';
      const h = document.createElement('h2');
      h.style.cssText = 'margin:0 0 0.5rem';
      h.textContent = title;
      const p1 = document.createElement('p');
      p1.style.cssText = 'color:#555;margin:0 0 0.75rem';
      p1.innerHTML = body;
      const p2 = document.createElement('p');
      p2.style.cssText = 'color:#aaa;font-size:0.8rem;margin:0';
      p2.textContent = 'You can close this browser window.';
      box.append(xBtn, icon, h, p1, p2);
      overlay.appendChild(box);
      document.body.prepend(overlay);
    }

    async function shutdownServer() {
      try {
        await fetch('/api/version', { signal: AbortSignal.timeout(4000) });
      } catch(_) {
        _showShutdownOverlay('Server already stopped', 'Run <code>python3 run_app.py</code> in the terminal to restart.');
        return;
      }
      if (!confirm('Stop the jB RAG Builder server? It will be unreachable until you restart it from the terminal.')) return;
      _showShutdownOverlay('Server stopped', 'Run <code>python3 run_app.py</code> in the terminal to restart.');
      fetch('/api/shutdown', { method: 'POST' }).catch(() => {});
    }

    // Server status badge — polls every 5s
    (function _pollServerStatus() {
      const badge = document.getElementById('serverStatusBadge');
      let _failCount = 0;
      async function check() {
        try {
          await fetch('/api/version', { signal: AbortSignal.timeout(4000) });
          _failCount = 0;
          badge.style.background = '#e8f5e9';
          badge.style.color = '#2e7d32';
          badge.style.borderColor = '#a5d6a7';
          badge.innerHTML = '\u25cf Online';
          badge.title = 'Server is running';
          // Auto-dismiss the overlay if the server has recovered
          const overlay = document.getElementById('shutdownOverlay');
          if (overlay) overlay.remove();
        } catch(_) {
          _failCount++;
          badge.style.background = '#fde8e8';
          badge.style.color = '#c0392b';
          badge.style.borderColor = '#f5c6cb';
          badge.innerHTML = '\u25cf Offline';
          badge.title = 'Server is not responding';
          // Only show the overlay after 2 consecutive failures (avoids false alarms on transient hiccups)
          if (_failCount >= 2) {
            _showShutdownOverlay('Server is not running', 'Run <code>python3 run_app.py</code> in the terminal to start it.');
          }
        }
      }
      check();
      setInterval(check, 5000);
    })();

    // ── Site Analysis Wizard ────────────────────────────────────────────────

    // State
    let _wizardCategories = [];   // category dicts from analyse
    let _wizardCollections = [];  // [{_id, display_name, doc_type, sitemapIds: Set, excludedUrls: Map<catId,Set<url>>, extraPages: Set<url>, fileSources: [{path, label}]}]
    let _wizardNextCollId = 0;
    let _wizardDomain = '';
    let _wizardExpanded = {};     // catId → bool
    let _wizardPages = {};        // catId → string[] or null (not yet loaded)
    let _wizardPageOverrides = {}; // url → collId  (page sent to a different coll than its sitemap)
    let _wizardExcluded = {};     // url → catId  (pages excluded from their sitemap's coll)
    let _wizardSearchQ = '';
    let _wizardPreviews = {};     // url → string  (cached page content previews)
    let _wizardShowAll = {};      // catId → bool  (show all pages, bypass PAGE_LIMIT)
    let _wizardDiffMode = false;  // true when in update/diff mode
    let _wizardNewUrls = {};      // url → true  (pages new in current sitemap vs saved session)
    let _wizardRemovedUrls = {};  // url → true  (pages gone from sitemap since last save)
    let _wizardPendingMode = null; // {type:'fresh'|'update', url, solId, solName, lang} — mode chosen but not yet launched
    let _wizardConfirmedColls = {}; // fullCollName → {points_count, exists} — populated from API
    let _wizardSolId = '';          // cached solution id — avoids DOM read timing issues
    let _wizardCollPagesOpen = new Set(); // keys: "{collId}::{catId}" — which sitemap page panels are expanded

    async function _wizardLoadConfirmedColls(solId) {
      if (!solId) return;
      _wizardSolId = solId;
      try {
        // Try the given solId; if not found or empty, search all solutions for a close match
        // (handles sessions saved as "peixe_fresco" when solution is stored as "peixefresco")
        let apiColls = [];
        const primaryResp = await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections');
        const primary = primaryResp.ok ? await primaryResp.json() : {};
        if (primary.collections && primary.collections.length) {
          apiColls = primary.collections;
        } else {
          // Strip underscores and compare — e.g. peixe_fresco == peixefresco
          const stripped = id => id.replace(/_/g,'');
          const allData = await fetch('/api/solutions').then(r => r.json());
          for (const sol of (allData.solutions || [])) {
            if (stripped(sol.id) !== stripped(solId)) continue;
            const innerResp = await fetch('/api/solutions/' + encodeURIComponent(sol.id) + '/collections');
            const res = innerResp.ok ? await innerResp.json() : {};
            if (res.collections && res.collections.length) {
              apiColls = res.collections;
              _wizardSolId = sol.id; // update cache to canonical id
              break;
            }
          }
        }
        _wizardConfirmedColls = {};
        apiColls.forEach(c => {
          _wizardConfirmedColls[c.name] = { points_count: c.points_count || 0, exists: c.exists || false };
        });
        // Back-fill confirmed_collection_name by best-effort matching
        _wizardCollections.forEach(wc => {
          if (wc.confirmed_collection_name) return;
          const norm = s => (s || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
          const wizNorm = norm(wc.display_name);
          // 1) Suffix match: peixefresco_products ends with _products
          let match = apiColls.find(ac => ac.name.endsWith('_' + wizNorm));
          // 2) API display_name match: "Products" == "Products" or "Product Catalog" contains "Product"
          if (!match) match = apiColls.find(ac => norm(ac.display_name) === wizNorm);
          // 3) Loose: wizard "product_catalog" starts with API suffix "product"
          if (!match) match = apiColls.find(ac => {
            const apiSuffix = norm(ac.display_name);
            return wizNorm.startsWith(apiSuffix) || apiSuffix.startsWith(wizNorm);
          });
          if (match) wc.confirmed_collection_name = match.name;
        });
        _wizardRenderCollections();
      } catch(e) { console.warn('[_wizardLoadConfirmedColls]', e); }
    }

    function _collQdrantName(c) {
      // Returns the Qdrant collection name for a wizard collection entry
      if (c.confirmed_collection_name) return '🗄 ' + c.confirmed_collection_name;
      const solId = _wizardSolId || _wizardCurrentSolId();
      if (!solId) return '';
      const suffix = (c.display_name || '').toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'');
      return '🗄 ' + solId + '_' + suffix + ' (pending confirm)';
    }

    function _escHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    // Which collection owns a sitemap?
    function _smCollId(catId) {
      const c = _wizardCollections.find(x => x.sitemapIds.has(catId));
      return c ? c._id : null;
    }

    // Effective state of a page url within catId
    function _pageState(url, catId) {
      if (_wizardExcluded[url] === 'review') return 'review';
      if (_wizardExcluded[url]) return 'excluded';
      if (_wizardPageOverrides[url] !== undefined) return 'overridden';
      const smColl = _smCollId(catId);
      if (smColl !== null) return 'inherited';
      return 'unassigned';
    }

    // ── Analyse ──────────────────────────────────────────────────────────────

    async function runWizardAnalyse() {
      const url = document.getElementById('wizardUrl').value.trim();
      const solName = document.getElementById('wizardSolName').value.trim();
      const lang = document.getElementById('wizardLang').value;
      if (!url) { alert('Please enter a website URL.'); return; }
      if (!solName) { alert('Please select a solution from the top bar first.'); return; }
      _wizardDomain = url;

      // Check if a saved session exists for this solution → offer Fresh vs Update
      const solId = solName.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/, '');
      try {
        const checkRes = await fetch('/api/wizard/load?solution_id=' + encodeURIComponent(solId));
        if (checkRes.ok) {
          const choice = await _wizardAskUpdateOrFresh(solName);
          if (choice === 'cancel') return;
          // Store choice — do NOT launch yet. Show mode bar with Launch button.
          _wizardPendingMode = { type: choice, url, solId, solName, lang };
          _wizardShowPendingLaunch();
          return;
        }
      } catch(_) { /* no saved session or network error — proceed fresh */ }

      // No saved session → go straight to fresh analysis (no modal needed)
      _runWizardFreshAnalyse(url, solName, lang);
    }

    async function _runWizardFreshAnalyse(url, solName, lang) {
      _wizardHideModeBar();
      const log = document.getElementById('wizardLog');
      log.textContent = 'Starting analysis…\\n';
      log.classList.remove('hidden', 'error', 'success');
      document.getElementById('wizardResults').style.display = 'none';
      try {
        const wizLoginCfg = _buildWizardLoginConfig();
        const res = await fetch('/api/wizard/analyse', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({url, solution_name: solName, language: lang,
                                ...(wizLoginCfg ? {login_config: wizLoginCfg} : {})})
        });
        if (!res.ok) { log.textContent += 'Failed to start.\\n'; return; }
        const sse = new EventSource('/api/progress');
        sse.onmessage = e => {
          const msg = e.data;
          if (msg.startsWith('LOG:')) {
            log.textContent += msg.slice(4) + '\\n'; log.scrollTop = log.scrollHeight;
          } else if (msg.startsWith('DONE:')) {
            sse.close();
            try {
              const p = JSON.parse(msg.slice(5));
              log.textContent += '✅ Analysis complete.\\n'; log.scrollTop = log.scrollHeight;
              _wizardInitState(p.categories, p.suggested_collections);
              _wizardRenderAll();
              document.getElementById('wizardResults').style.display = 'flex';
              const saveBtn = document.getElementById('btnWizardSave');
              if (saveBtn) saveBtn.style.display = '';
              // Fetch confirmed collection statuses (for returning users with existing collections)
              _wizardLoadConfirmedColls(_wizardCurrentSolId());
            } catch(err) { log.textContent += '❌ Parse error: ' + err + '\\n'; }
          } else if (msg.startsWith('ERROR:')) {
            log.textContent += '❌ ' + msg.slice(6) + '\\n'; log.classList.add('error'); sse.close();
          }
        };
        sse.onerror = () => { log.textContent += 'Connection lost.\\n'; sse.close(); };
      } catch(err) { log.textContent += 'Error: ' + err + '\\n'; }
    }

    // ── Diff / Update mode helpers ────────────────────────────────────────────

    // Show the mode bar with chosen mode label + Launch button
    function _wizardShowPendingLaunch() {
      const m = _wizardPendingMode;
      if (!m) return;
      const label = m.type === 'update' ? '🔄 Check for updates' : '🆕 Fresh start';
      const bar = document.getElementById('wizardModeBar');
      if (bar) {
        document.getElementById('wizardModeLabel').textContent = 'Mode: ' + label;
        bar.style.display = 'flex';
      }
    }

    // Launch the pending analysis (called by 🚀 Launch button)
    function _wizardLaunch() {
      const m = _wizardPendingMode;
      if (!m) return;
      _wizardPendingMode = null;
      _wizardHideModeBar();
      if (m.type === 'update') _runWizardDiff(m.url, m.solId, m.lang);
      else _runWizardFreshAnalyse(m.url, m.solName, m.lang);
    }

    // Hide the mode bar and clear pending mode
    function _wizardHideModeBar() {
      const bar = document.getElementById('wizardModeBar');
      if (bar) bar.style.display = 'none';
      _wizardPendingMode = null;
    }

    // Show modal asking user: Update (diff) or Fresh start?
    // Returns a Promise resolving to 'update' | 'fresh' | 'cancel'
    function _wizardAskUpdateOrFresh(solName) {
      return new Promise(resolve => {
        const overlay = document.createElement('div');
        overlay.className = 'wiz-modal-overlay';

        const box = document.createElement('div');
        box.className = 'wiz-modal-box';

        const title = document.createElement('div');
        title.className = 'wiz-modal-title';
        title.textContent = '💾 Saved session found';

        const sub = document.createElement('div');
        sub.className = 'wiz-modal-sub';
        sub.textContent = 'A saved wizard session exists for "' + solName + '". What would you like to do?';

        const btns = document.createElement('div');
        btns.className = 'wiz-modal-btns';

        // Update button
        const updateBtn = document.createElement('button');
        updateBtn.className = 'wiz-modal-btn primary';
        updateBtn.innerHTML = '<span class="wiz-modal-btn-icon">🔄</span>'
          + '<span class="wiz-modal-btn-label">Check for updates</span>'
          + '<span class="wiz-modal-btn-desc">Keep your collection assignments — only show new and removed pages</span>';

        // Fresh button
        const freshBtn = document.createElement('button');
        freshBtn.className = 'wiz-modal-btn';
        freshBtn.innerHTML = '<span class="wiz-modal-btn-icon">🆕</span>'
          + '<span class="wiz-modal-btn-label">Fresh start</span>'
          + '<span class="wiz-modal-btn-desc">Discard saved session and re-analyse from scratch</span>';

        // Launch button (disabled until a choice is selected)
        const launchBtn = document.createElement('button');
        launchBtn.className = 'wiz-modal-btn primary';
        launchBtn.style.cssText = 'margin-top:0.75rem;width:100%;justify-content:center;opacity:0.4;pointer-events:none;';
        launchBtn.innerHTML = '▶ Select';

        let selectedChoice = null;
        const selectChoice = (choice, activeBtn, otherBtn) => {
          selectedChoice = choice;
          activeBtn.style.outline = '3px solid #1a4a90';
          otherBtn.style.outline = '';
          launchBtn.style.opacity = '1';
          launchBtn.style.pointerEvents = '';
        };

        updateBtn.onclick = () => selectChoice('update', updateBtn, freshBtn);
        freshBtn.onclick = () => selectChoice('fresh', freshBtn, updateBtn);
        launchBtn.onclick = () => { if (selectedChoice) { overlay.remove(); resolve(selectedChoice); } };

        btns.appendChild(updateBtn);
        btns.appendChild(freshBtn);

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'wiz-modal-cancel';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.onclick = () => { overlay.remove(); resolve('cancel'); };

        box.appendChild(title);
        box.appendChild(sub);
        box.appendChild(btns);
        box.appendChild(launchBtn);
        box.appendChild(cancelBtn);
        overlay.appendChild(box);
        document.body.appendChild(overlay);
      });
    }

    // Run diff against saved session and restore state with markers
    async function _runWizardDiff(url, solId, lang) {
      _wizardHideModeBar();
      const log = document.getElementById('wizardLog');
      log.textContent = 'Loading saved session…\\n';
      log.classList.remove('hidden', 'error', 'success');
      document.getElementById('wizardResults').style.display = 'none';

      try {
        const res = await fetch('/api/wizard/diff', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({url, solution_id: solId})
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          log.textContent += '❌ ' + (err.detail || 'Diff failed') + '\\n';
          log.classList.add('error');
          return;
        }

        const sse = new EventSource('/api/progress');
        sse.onmessage = e => {
          const msg = e.data;
          if (msg.startsWith('LOG:')) {
            log.textContent += msg.slice(4) + '\\n'; log.scrollTop = log.scrollHeight;
          } else if (msg.startsWith('DONE:')) {
            sse.close();
            try {
              const data = JSON.parse(msg.slice(5));
              _applyDiffResult(data, lang);
              log.textContent += '✅ Done.\\n'; log.scrollTop = log.scrollHeight;
            } catch(err) { log.textContent += '❌ Parse error: ' + err + '\\n'; }
          } else if (msg.startsWith('ERROR:')) {
            log.textContent += '❌ ' + msg.slice(6) + '\\n'; log.classList.add('error'); sse.close();
          }
        };
        sse.onerror = () => { log.textContent += 'Connection lost.\\n'; sse.close(); };
      } catch(err) {
        log.textContent += 'Error: ' + err + '\\n'; log.classList.add('error');
      }
    }

    // Apply the diff result: restore saved state, then inject new/removed markers
    function _applyDiffResult(data, lang) {
      // 1. Restore saved state (collections, assignments, exclusions, etc.)
      _wizardRestoreState(data.saved_state);

      // 2. Switch to diff mode
      _wizardDiffMode = true;

      // 3. Build new/removed URL sets
      _wizardNewUrls = {};
      _wizardRemovedUrls = {};

      // New whole categories → add to _wizardCategories as unassigned, mark all their URLs new
      for (const cat of (data.new_categories || [])) {
        if (!_wizardCategories.find(c => c.id === cat.id)) {
          cat._diff = 'new';
          _wizardCategories.push(cat);
          _wizardPages[cat.id] = null; // not loaded yet
        }
      }

      // Per-sitemap URL changes
      const changed = data.changed || {};
      for (const [catId, delta] of Object.entries(changed)) {
        // Merge new URLs into the loaded page list for this sitemap
        if (!_wizardPages[catId]) _wizardPages[catId] = [];
        for (const u of (delta.new_urls || [])) {
          if (!_wizardPages[catId].includes(u)) _wizardPages[catId].push(u);
          _wizardNewUrls[u] = true;
        }
        for (const u of (delta.removed_urls || [])) {
          // Keep in list so they show with DEL marker
          if (!_wizardPages[catId].includes(u)) _wizardPages[catId].push(u);
          _wizardRemovedUrls[u] = true;
        }
        // Auto-expand sitemaps that have changes
        if ((delta.new_urls || []).length || (delta.removed_urls || []).length) {
          _wizardExpanded[catId] = true;
        }
      }

      // 4. Render
      _wizardRenderDiffBanner(data);
      _wizardRenderAll();
      document.getElementById('wizardResults').style.display = 'flex';
      const saveBtn = document.getElementById('btnWizardSave');
      if (saveBtn) saveBtn.style.display = '';
    }

    // Render the yellow diff summary banner above the sitemap list
    function _wizardRenderDiffBanner(data) {
      const banner = document.getElementById('wizardDiffBanner');
      if (!banner) return;

      const changed = data.changed || {};
      const totalNew = (data.new_categories || []).length
        + Object.values(changed).reduce((s, d) => s + (d.new_urls || []).length, 0);
      const totalRemoved = (data.removed_categories || []).length
        + Object.values(changed).reduce((s, d) => s + (d.removed_urls || []).length, 0);
      const totalUnchanged = (data.current_categories || [])
        .filter(c => !c.id.startsWith('_') && !changed[c.id] && !(data.new_categories||[]).find(x=>x.id===c.id)).length;

      let html = '🔄 <strong>Update check complete</strong> &nbsp;·&nbsp; ';
      if (!totalNew && !totalRemoved) {
        html += '✅ No changes — everything is up to date.';
      } else {
        const parts = [];
        if (totalNew) parts.push('<span style="color:#1b5e20;font-weight:600;">🟢 ' + totalNew + ' new</span>');
        if (totalRemoved) parts.push('<span style="color:#b71c1c;font-weight:600;">🔴 ' + totalRemoved + ' removed</span>');
        if (totalUnchanged) parts.push(totalUnchanged + ' unchanged');
        html += parts.join(' &nbsp;·&nbsp; ');
      }
      banner.innerHTML = html;
      banner.style.display = 'flex';
    }

    function _wizardInitState(cats, suggestions) {
      _wizardCategories = cats || [];
      _wizardNextCollId = 0;
      _wizardExpanded = {};
      _wizardPages = {};
      _wizardPageOverrides = {};
      _wizardExcluded = {};
      _wizardSearchQ = '';
      _wizardPreviews = {};
      _wizardShowAll = {};
      _wizardDiffMode = false;
      _wizardNewUrls = {};
      _wizardRemovedUrls = {};
      document.getElementById('wizardDiffBanner').style.display = 'none';
      // Pre-populate pages from sample_urls so expand shows something immediately
      for (const cat of _wizardCategories) {
        if (!cat.id.startsWith('_') && cat.sample_urls && cat.sample_urls.length) {
          _wizardPages[cat.id] = null; // null = not fully loaded yet
        }
      }
      _wizardCollections = (suggestions || []).map(s => ({
        _id: _wizardNextCollId++,
        display_name: s.display_name || s.collection_name,
        doc_type: s.doc_type || 'general',
        sitemapIds: new Set(s.categories || []),
        excludedUrls: new Map(),  // catId → Set<url>
        extraPages: new Set(),    // individual page URLs from other sitemaps
        fileSources: [],          // [{path, label}] non-web sources (PDFs, CSVs, etc.)
      }));
      // Show chat section now that we have analysis context; hide the pre-analysis hint
      const chatHint = document.getElementById('wizardChatHint');
      if (chatHint) chatHint.style.display = 'none';
      const chatSec = document.getElementById('wizardChatSection');
      if (chatSec) { chatSec.style.display = ''; document.getElementById('wizardChatHistory').innerHTML = ''; }
    }

    // Confirm and execute Qdrant chunk deletion for a removed URL
    function _wizardConfirmDeletePage(url, catId, delBtn, row) {
      _inlineConfirm(delBtn, {
        message: 'Delete chunks for this page from Qdrant?',
        confirmLabel: 'Delete',
        onConfirm: async (btn) => {
          btn.textContent = 'Deleting…'; btn.disabled = true;

          // Find Qdrant collection name for this catId
          const collId = _smCollId(catId)
            || (_wizardPageOverrides[url] !== undefined ? _wizardPageOverrides[url] : null);
          const coll = _wizardCollections.find(c => c._id === collId);
          const solId = document.getElementById('wizardSolName').value.trim()
            .toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/,'');
          const collSuffix = coll
            ? coll.display_name.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/,'')
            : null;
          const qdrantName = coll ? solId + '_' + collSuffix : null;

          if (!qdrantName) {
            btn.textContent = '⚠ No collection'; btn.disabled = false;
            return;
          }

          try {
            const res = await fetch('/api/wizard/delete-pages', {
              method: 'POST', headers: {'Content-Type':'application/json'},
              body: JSON.stringify({collection_name: qdrantName, urls: [url]})
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Delete failed');
            delete _wizardRemovedUrls[url];
            row.style.opacity = '1'; row.style.textDecoration = '';
            btn.textContent = '✅ deleted'; btn.disabled = true;
            btn.style.background = '#e8f5e9'; btn.style.borderColor = '#a5d6a7'; btn.style.color = '#1b5e20';
          } catch(err) {
            btn.textContent = '❌ ' + err.message; btn.disabled = false;
          }
        }
      });
    }

    // ── Render ────────────────────────────────────────────────────────────────

    function _wizardRenderAll() {
      _wizardRenderSitemapList();
      _wizardRenderCollections();
      _wizardAutoSave();
    }

    function _wizardRenderSitemapList() {
      const list = document.getElementById('wizardSitemapList');
      list.innerHTML = '';
      const q = _wizardSearchQ.toLowerCase();

      for (const cat of _wizardCategories) {
        if (cat.id.startsWith('_')) {
          const d = document.createElement('div');
          d.className = 'wiz-sm-sentinel';
          d.innerHTML = '<strong>' + _escHtml(cat.display_name) + '</strong> — ' + _escHtml(cat.preview || '');
          list.appendChild(d);
          continue;
        }

        const collId = _smCollId(cat.id);
        const coll = _wizardCollections.find(x => x._id === collId);
        const collName = coll ? coll.display_name : null;

        // Sitemap row
        const smRow = document.createElement('div');
        smRow.className = 'wiz-sm-row';
        smRow.draggable = true;
        smRow.dataset.catId = cat.id;
        smRow.addEventListener('dragstart', e => {
          e.dataTransfer.setData('text/plain', JSON.stringify({type:'sitemap', catId: cat.id}));
          e.dataTransfer.effectAllowed = 'move';
        });

        const tog = document.createElement('span');
        tog.className = 'wiz-sm-toggle';
        tog.textContent = _wizardExpanded[cat.id] ? '▼' : '▶';
        tog.onclick = () => _wizardToggleSitemap(cat.id);

        const nameSpan = document.createElement('span');
        nameSpan.className = 'wiz-sm-name';
        nameSpan.title = cat.sitemap_url || cat.id;
        nameSpan.textContent = cat.display_name;
        nameSpan.onclick = () => _wizardToggleSitemap(cat.id);

        const countSpan = document.createElement('span');
        countSpan.className = 'wiz-sm-count';
        countSpan.textContent = (cat.url_count || 0) + ' pages';

        const badge = document.createElement('button');
        badge.className = 'wiz-sm-coll-badge ' + (collName ? 'assigned' : 'unassigned');
        badge.textContent = collName ? ('→ ' + collName) : '+ assign';
        badge.onclick = e => { e.stopPropagation(); _wizardShowSitemapDropdown(cat.id, badge); };

        const isSkipped = collId === null;
        const skipBtn = document.createElement('button');
        skipBtn.className = 'wiz-sm-skip-btn';
        skipBtn.title = isSkipped ? 'Un-skip this sitemap' : 'Skip this sitemap';
        skipBtn.textContent = isSkipped ? '↩' : '✕';
        skipBtn.onclick = e => {
          e.stopPropagation();
          wizardAssignSitemap(cat.id, isSkipped ? (_wizardCollections[0] ? _wizardCollections[0]._id : null) : null);
        };

        const showAllBtn = document.createElement('button');
        showAllBtn.className = 'wiz-sm-showall-btn';
        showAllBtn.title = 'Expand and show ALL pages in this sitemap';
        showAllBtn.textContent = '📄 all';
        showAllBtn.onclick = async e => {
          e.stopPropagation();
          _wizardShowAll[cat.id] = true;
          if (!_wizardExpanded[cat.id]) {
            _wizardExpanded[cat.id] = true;
            if (_wizardPages[cat.id] === null || _wizardPages[cat.id] === undefined) {
              await _wizardLoadPages(cat.id);
            }
          }
          _wizardRenderAll();
        };

        smRow.appendChild(tog);
        smRow.appendChild(nameSpan);
        smRow.appendChild(countSpan);
        smRow.appendChild(skipBtn);
        smRow.appendChild(showAllBtn);
        smRow.appendChild(badge);
        list.appendChild(smRow);

        // Expanded pages
        if (_wizardExpanded[cat.id]) {
          const pageContainer = document.createElement('div');
          pageContainer.className = 'wiz-page-list';
          pageContainer.id = 'wiz-pages-' + cat.id;
          list.appendChild(pageContainer);
          _wizardRenderPageList(cat.id, q);
        }
      }
    }

    function _wizardRenderPageList(catId, q) {
      const container = document.getElementById('wiz-pages-' + catId);
      if (!container) return;
      container.innerHTML = '';
      const pages = _wizardPages[catId];
      if (pages === null) {
        // Lazy load
        container.innerHTML = '<span class="wiz-spinner"></span> Loading pages…';
        _wizardLoadPages(catId);
        return;
      }
      if (!pages || !pages.length) {
        container.textContent = 'No pages found.';
        return;
      }

      const filtered = q ? pages.filter(u => u.toLowerCase().includes(q)) : pages;
      const PAGE_LIMIT = 60;
      const showAll = !!_wizardShowAll[catId];
      const shown = (showAll || filtered.length <= PAGE_LIMIT) ? filtered : filtered.slice(0, PAGE_LIMIT);

      for (const url of shown) {
        const state = _pageState(url, catId);
        const smColl = _wizardCollections.find(x => x._id === _smCollId(catId));
        const ovColl = _wizardCollections.find(x => x._id === _wizardPageOverrides[url]);
        let badgeText, badgeCls;
        if (state === 'excluded') { badgeText = '✕ excluded'; badgeCls = 'excluded'; }
        else if (state === 'review') { badgeText = '📌 review'; badgeCls = 'review'; }
        else if (state === 'overridden') { badgeText = '→ ' + (ovColl ? ovColl.display_name : '?'); badgeCls = 'overridden'; }
        else if (state === 'inherited') { badgeText = smColl ? smColl.display_name : ''; badgeCls = 'inherited'; }
        else { badgeText = '— unassigned'; badgeCls = 'unassigned'; }

        // Diff mode markers
        const isNew = _wizardDiffMode && !!_wizardNewUrls[url];
        const isRemoved = _wizardDiffMode && !!_wizardRemovedUrls[url];

        const row = document.createElement('div');
        let rowCls = 'wiz-page-row';
        if (isNew) rowCls += ' wiz-new';
        if (isRemoved) rowCls += ' wiz-removed';
        row.className = rowCls;
        row.draggable = !isRemoved;
        if (!isRemoved) {
          row.addEventListener('dragstart', e => {
            e.dataTransfer.setData('text/plain', JSON.stringify({type:'page', catId, url}));
            e.dataTransfer.effectAllowed = 'move';
            e.stopPropagation();
          });
        }

        // Show just the path portion
        let displayUrl = url;
        try { displayUrl = new URL(url).pathname; } catch(_) {}

        const urlSpan = document.createElement('a');
        urlSpan.className = 'wiz-page-url';
        urlSpan.href = url;
        urlSpan.target = '_blank';
        urlSpan.rel = 'noopener noreferrer';
        urlSpan.title = url;
        urlSpan.textContent = displayUrl;

        row.appendChild(urlSpan);

        // Diff badges (before assignment badge)
        if (isNew) {
          const nb = document.createElement('span');
          nb.className = 'wiz-page-badge new-page';
          nb.textContent = '🟢 NEW';
          row.appendChild(nb);
        }
        if (isRemoved) {
          const rb = document.createElement('span');
          rb.className = 'wiz-page-badge removed-page';
          rb.textContent = '🔴 DEL';
          row.appendChild(rb);
        }

        if (!isRemoved) {
          const pageBadge = document.createElement('button');
          pageBadge.className = 'wiz-page-badge ' + badgeCls;
          pageBadge.textContent = badgeText;
          pageBadge.onclick = e => { e.stopPropagation(); _wizardShowPageDropdown(url, catId, pageBadge); };
          row.appendChild(pageBadge);
        }

        if (isRemoved) {
          // Delete from Qdrant button
          const delBtn = document.createElement('button');
          delBtn.className = 'wiz-del-btn';
          delBtn.title = 'Delete chunks for this URL from Qdrant';
          delBtn.textContent = '🗑 delete';
          delBtn.onclick = e => { e.stopPropagation(); _wizardConfirmDeletePage(url, catId, delBtn, row); };
          row.appendChild(delBtn);
        }

        const eyeBtn = document.createElement('button');
        eyeBtn.className = 'wiz-eye';
        eyeBtn.title = 'Preview content';
        eyeBtn.textContent = '👁';
        eyeBtn.onclick = e => { e.stopPropagation(); _wizardTogglePreview(url, catId, row, eyeBtn); };
        row.appendChild(eyeBtn);

        if (!isRemoved) {
          const exclBtn = document.createElement('button');
          exclBtn.className = 'wiz-page-excl-btn';
          const st = _pageState(url, catId);
          if (st === 'excluded') {
            exclBtn.title = 'Revert (un-exclude)';
            exclBtn.textContent = '↩';
            exclBtn.onclick = e => { e.stopPropagation(); _wizardRevertPage(url); _wizardRenderAll(); };
          } else {
            exclBtn.title = 'Exclude this page';
            exclBtn.textContent = '✕';
            exclBtn.onclick = e => { e.stopPropagation(); _wizardExcludePageFn(url); _wizardRenderAll(); };
          }
          row.appendChild(exclBtn);
        }

        container.appendChild(row);
      }

      if (filtered.length > PAGE_LIMIT) {
        const more = document.createElement('span');
        more.className = 'wiz-load-more';
        if (showAll) {
          more.textContent = '▲ Collapse (showing all ' + filtered.length + ' pages)';
          more.onclick = () => { _wizardShowAll[catId] = false; _wizardRenderPageList(catId, q); };
        } else {
          more.textContent = '… ' + (filtered.length - PAGE_LIMIT) + ' more — click to show all';
          more.onclick = () => { _wizardShowAll[catId] = true; _wizardRenderPageList(catId, q); };
        }
        container.appendChild(more);
      }
      if (!filtered.length && q) {
        container.textContent = 'No pages match "' + q + '".';
      }
    }

    async function _wizardLoadPages(catId) {
      const cat = _wizardCategories.find(x => x.id === catId);
      if (!cat) return;
      try {
        let apiUrl = '/api/wizard/sitemap-pages?sitemap_url=' + encodeURIComponent(cat.sitemap_url || '');
        if (cat.url_filter) apiUrl += '&url_filter=' + encodeURIComponent(cat.url_filter);
        const d = await fetch(apiUrl).then(r => r.json());
        _wizardPages[catId] = d.urls || [];
      } catch(e) {
        _wizardPages[catId] = [];
      }
      _wizardRenderPageList(catId, _wizardSearchQ.toLowerCase());
      // If this catId's page panel is open in any collection, refresh it
      for (const key of _wizardCollPagesOpen) {
        const [collIdStr, openCatId] = key.split('::');
        if (openCatId === catId) {
          const collId = parseInt(collIdStr);
          const coll = _wizardCollections.find(x => x._id === collId);
          const panelId = 'wiz-pages-panel-' + collId + '-' + catId.replace(/[^a-z0-9]/gi,'_');
          const existing = document.getElementById(panelId);
          if (existing && coll) _wizardBuildSitemapPagePanel(existing, coll, catId);
        }
      }
    }

    // Returns pages belonging to collection `c` from sitemap `catId`, filtered by exclusions/overrides
    function _collectionPagesForCat(c, catId) {
      const pages = _wizardPages[catId];
      if (!Array.isArray(pages)) return null; // not loaded yet
      return pages.filter(url => {
        if (_wizardExcluded[url]) return false; // excluded globally
        const override = _wizardPageOverrides[url];
        if (override !== undefined && override !== c._id) return false; // moved to another coll
        return true;
      });
    }

    // Builds the content of a sitemap page panel into `container`
    function _wizardBuildSitemapPagePanel(container, c, catId) {
      container.innerHTML = '';
      const pages = _collectionPagesForCat(c, catId);
      if (pages === null) {
        // Not loaded yet — show spinner and trigger load
        container.innerHTML = '<span style="font-size:0.8rem;color:#999;padding:0.3rem 0.5rem;">⏳ Loading pages…</span>';
        _wizardLoadPages(catId); // will call back into here via the hook above
        return;
      }
      if (!pages.length) {
        container.innerHTML = '<span style="font-size:0.8rem;color:#aaa;padding:0.3rem 0.5rem;font-style:italic;">No pages in this sitemap.</span>';
        return;
      }
      pages.forEach(url => {
        // Page row
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:0.35rem;padding:0.18rem 0.4rem;border-radius:4px;font-size:0.78rem;';
        row.onmouseenter = () => row.style.background = '#f5f8ff';
        row.onmouseleave = () => row.style.background = '';

        // URL link — show pathname only, full URL as title
        const link = document.createElement('a');
        try { link.textContent = new URL(url).pathname; } catch(_) { link.textContent = url; }
        link.href = url; link.target = '_blank';
        link.title = url;
        link.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#1a5276;text-decoration:none;font-size:0.78rem;';
        link.onmouseenter = () => link.style.textDecoration = 'underline';
        link.onmouseleave = () => link.style.textDecoration = 'none';
        row.appendChild(link);

        // State badge (only show if overridden/extra)
        const state = _pageState(url, catId);
        if (state === 'overridden') {
          const badge = document.createElement('span');
          badge.style.cssText = 'font-size:0.68rem;padding:0.05rem 0.35rem;border-radius:8px;background:#fff3e0;color:#e65100;border:1px solid #ffcc80;white-space:nowrap;flex-shrink:0;';
          badge.textContent = '→ extra';
          row.appendChild(badge);
        }

        // 👁 Preview button
        const eyeBtn = document.createElement('button');
        eyeBtn.type = 'button'; eyeBtn.textContent = '👁';
        eyeBtn.title = 'Preview content';
        eyeBtn.style.cssText = 'background:none;border:none;cursor:pointer;font-size:0.8rem;padding:0.1rem 0.2rem;opacity:0.5;flex-shrink:0;';
        eyeBtn.onclick = () => _wizardTogglePreview(url, catId, row, eyeBtn);
        row.appendChild(eyeBtn);

        // ✕ / action button — opens the existing dropdown
        const actBtn = document.createElement('button');
        actBtn.type = 'button'; actBtn.textContent = '✕';
        actBtn.title = 'Exclude or move page';
        actBtn.style.cssText = 'background:none;border:none;cursor:pointer;font-size:0.75rem;padding:0.1rem 0.2rem;color:#c0392b;opacity:0.6;flex-shrink:0;';
        actBtn.onmouseenter = () => actBtn.style.opacity = '1';
        actBtn.onmouseleave = () => actBtn.style.opacity = '0.6';
        actBtn.onclick = (e) => {
          e.stopPropagation();
          const dd = _wizardMakeDropdown(catId, url);
          if (!dd) return;
          dd.style.position = 'fixed';
          dd.style.removeProperty('top');
          document.body.appendChild(dd);
          const r = actBtn.getBoundingClientRect();
          // Position below button, flip up if too close to bottom
          let top = r.bottom + 4;
          if (top + 200 > window.innerHeight) top = r.top - 204;
          dd.style.top = top + 'px';
          dd.style.left = Math.min(r.left, window.innerWidth - 180) + 'px';
        };
        row.appendChild(actBtn);

        container.appendChild(row);
      });
    }

    // Toggle a sitemap page panel open/closed within a collection block
    function _wizardToggleSitemapPagePanel(c, catId, toggleBtn, insertAfter) {
      const key = c._id + '::' + catId;
      const panelId = 'wiz-pages-panel-' + c._id + '-' + catId.replace(/[^a-z0-9]/gi,'_');
      const existing = document.getElementById(panelId);
      if (existing) {
        // Toggle visibility
        const hidden = existing.style.display === 'none';
        existing.style.display = hidden ? '' : 'none';
        if (hidden) { _wizardCollPagesOpen.add(key); toggleBtn.textContent = '▴'; }
        else { _wizardCollPagesOpen.delete(key); toggleBtn.textContent = '▾'; }
        return;
      }
      // Create panel
      _wizardCollPagesOpen.add(key);
      toggleBtn.textContent = '▴';
      const panel = document.createElement('div');
      panel.id = panelId;
      panel.style.cssText = 'max-height:260px;overflow-y:auto;border-top:1px solid #e8eaf0;margin-top:0.25rem;padding:0.2rem 0;background:#fafbfd;border-radius:0 0 6px 6px;';
      _wizardBuildSitemapPagePanel(panel, c, catId);
      // Insert after the sitemap row
      if (insertAfter.nextSibling) insertAfter.parentNode.insertBefore(panel, insertAfter.nextSibling);
      else insertAfter.parentNode.appendChild(panel);
    }

    async function _wizardTogglePreview(url, catId, row, eyeBtn) {
      // If a preview box already exists as next sibling, toggle its visibility
      const existing = row.nextSibling && row.nextSibling.classList && row.nextSibling.classList.contains('wiz-preview-box')
        ? row.nextSibling : null;
      if (existing) {
        const isHidden = existing.style.display === 'none';
        existing.style.display = isHidden ? '' : 'none';
        if (eyeBtn) eyeBtn.style.opacity = isHidden ? '1' : '0.4';
        return;
      }
      // Create box
      const box = document.createElement('div');
      box.className = 'wiz-preview-box';
      box.style.cssText = 'margin-top:0.3rem;';
      // Insert after the row immediately (before fetching, to hold position)
      row.parentNode.insertBefore(box, row.nextSibling);
      if (eyeBtn) eyeBtn.style.opacity = '1';
      // Use cached content if available
      if (_wizardPreviews[url] !== undefined) {
        box.textContent = _wizardPreviews[url] || '(no content extracted)';
        return;
      }
      box.textContent = 'Loading…';
      try {
        const d = await fetch('/api/wizard/page-preview?url=' + encodeURIComponent(url)).then(r => r.json());
        const text = d.preview || '(no content extracted)';
        _wizardPreviews[url] = text;   // cache for future toggles
        box.textContent = text;
      } catch(e) {
        _wizardPreviews[url] = '';
        box.textContent = 'Error loading preview.';
      }
    }

    function _wizardToggleSitemap(catId) {
      _wizardExpanded[catId] = !_wizardExpanded[catId];
      if (_wizardExpanded[catId] && _wizardPages[catId] === null) {
        _wizardPages[catId] = null; // will trigger load
      }
      _wizardRenderSitemapList();
    }

    // ── Dropdowns ─────────────────────────────────────────────────────────────

    function _wizardShowSitemapDropdown(catId, anchor) {
      _wizardRemoveDropdown();
      const menu = _wizardMakeDropdown(catId, null);
      document.body.appendChild(menu);
      const r = anchor.getBoundingClientRect();
      menu.style.top = (r.bottom + window.scrollY + 4) + 'px';
      menu.style.left = (r.left + window.scrollX) + 'px';
    }

    function _wizardShowPageDropdown(url, catId, anchor) {
      _wizardRemoveDropdown();
      const menu = _wizardMakeDropdown(catId, url);
      document.body.appendChild(menu);
      const r = anchor.getBoundingClientRect();
      menu.style.top = (r.bottom + window.scrollY + 4) + 'px';
      menu.style.left = (r.left + window.scrollX) + 'px';
    }

    function _wizardMakeDropdown(catId, url) {
      const isPage = url !== null;
      const menu = document.createElement('div');
      menu.id = '_wizDropdown';
      menu.style.cssText = 'position:absolute;z-index:500;background:#fff;border:1px solid #ddd;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,.15);min-width:160px;overflow:hidden;';

      const items = [];
      if (isPage) {
        const state = _pageState(url, catId);
        if (state !== 'excluded') {
          items.push({label: '✕ Exclude this page', fn: () => { _wizardExcludePageFn(url); _wizardRenderAll(); }});
        }
        if (state !== 'review') {
          items.push({label: '📌 Flag for review', fn: () => { _wizardReviewPageFn(url); _wizardRenderAll(); }});
        }
        if (state === 'excluded' || state === 'review' || state === 'overridden') {
          items.push({label: '↩ Revert to sitemap', fn: () => { _wizardRevertPage(url); _wizardRenderAll(); }});
        }
        items.push({label: '— unassign —', fn: () => { _wizardRevertPage(url); _wizardExcludePageFn(url); _wizardRenderAll(); }});
        items.push({type:'sep'});
      }
      // Collection options
      for (const c of _wizardCollections) {
        const isCurrent = isPage ? _wizardPageOverrides[url] === c._id : _smCollId(catId) === c._id;
        items.push({label: (isCurrent ? '✓ ' : '') + c.display_name, fn: () => {
          if (isPage) wizardAssignPage(url, catId, c._id);
          else wizardAssignSitemap(catId, c._id);
        }});
      }
      if (!isPage) {
        const isSkipped = _smCollId(catId) === null;
        items.push({label: (isSkipped ? '✓ ' : '') + '— skip sitemap —', fn: () => wizardAssignSitemap(catId, null)});
      }

      for (const it of items) {
        if (it.type === 'sep') {
          const sep = document.createElement('div');
          sep.style.cssText = 'border-top:1px solid #eee;margin:2px 0;';
          menu.appendChild(sep);
          continue;
        }
        const btn = document.createElement('button');
        btn.style.cssText = 'width:100%;text-align:left;padding:0.45rem 0.85rem;border:none;background:none;cursor:pointer;font-size:0.85rem;';
        btn.textContent = it.label;
        btn.onmouseover = () => btn.style.background = '#f0f4ff';
        btn.onmouseout = () => btn.style.background = 'none';
        btn.onclick = () => { it.fn(); _wizardRemoveDropdown(); };
        menu.appendChild(btn);
      }

      // Close on outside click
      setTimeout(() => {
        document.addEventListener('click', _wizardRemoveDropdown, {once: true});
      }, 0);
      return menu;
    }

    function _wizardRemoveDropdown() {
      const el = document.getElementById('_wizDropdown');
      if (el) el.remove();
    }

    // ── Assignment functions ──────────────────────────────────────────────────

    function wizardAssignSitemap(catId, collId) {
      // Remove from all collections
      for (const c of _wizardCollections) c.sitemapIds.delete(catId);
      if (collId !== null) {
        const c = _wizardCollections.find(x => x._id === collId);
        if (c) c.sitemapIds.add(catId);
      }
      _wizardRenderAll();
    }

    function wizardAssignPage(url, catId, collId) {
      delete _wizardExcluded[url];
      // Remove from other collections' extraPages
      for (const c of _wizardCollections) c.extraPages.delete(url);
      // If same as sitemap collection, just revert
      if (_smCollId(catId) === collId) {
        delete _wizardPageOverrides[url];
      } else {
        _wizardPageOverrides[url] = collId;
        const c = _wizardCollections.find(x => x._id === collId);
        if (c) c.extraPages.add(url);
      }
      _wizardRenderAll();
    }

    function _wizardExcludePageFn(url) {
      delete _wizardPageOverrides[url];
      for (const c of _wizardCollections) c.extraPages.delete(url);
      _wizardExcluded[url] = 'excluded';
    }

    async function wizardAskChat() {
      const input = document.getElementById('wizardChatInput');
      const history = document.getElementById('wizardChatHistory');
      const q = (input.value || '').trim();
      if (!q) return;
      input.value = '';

      // User bubble
      const uBubble = document.createElement('div');
      uBubble.className = 'wiz-chat-bubble user';
      uBubble.textContent = q;
      history.appendChild(uBubble);

      // Thinking indicator
      const thinking = document.createElement('div');
      thinking.className = 'wiz-chat-bubble bot';
      thinking.textContent = '…';
      history.appendChild(thinking);
      history.scrollTop = history.scrollHeight;

      // Serialize collections (Set/Map → plain)
      const collsSer = _wizardCollections.map(c => ({
        display_name: c.display_name,
        doc_type: c.doc_type || '',
        sitemapIds: [...(c.sitemapIds || [])],
      }));

      try {
        const res = await fetch('/api/wizard/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            question: q,
            categories: _wizardCategories,
            collections: collsSer,
            domain: _wizardDomain || '',
          }),
        });
        const data = await res.json();
        thinking.remove();

        const botBubble = document.createElement('div');
        botBubble.className = 'wiz-chat-bubble bot';
        botBubble.textContent = data.answer || '(no answer)';
        history.appendChild(botBubble);

        // Suggestion pills — click loads into input; user presses Ask to run
        if (data.suggestions && data.suggestions.length) {
          const pillWrap = document.createElement('div');
          pillWrap.style.marginBottom = '0.3rem';
          for (const s of data.suggestions) {
            const pill = document.createElement('span');
            pill.className = 'wiz-chat-suggestion';
            pill.textContent = s;
            pill.title = 'Click to load into the input field, then press Ask';
            pill.onclick = () => {
              const inp = document.getElementById('wizardChatInput');
              if (inp) { inp.value = s; inp.focus(); }
            };
            pillWrap.appendChild(pill);
          }
          history.appendChild(pillWrap);
        }
      } catch(e) {
        thinking.textContent = '❌ Error: ' + (e.message || e);
      }
      history.scrollTop = history.scrollHeight;
    }

    function _wizardRevertPage(url) {
      delete _wizardExcluded[url];
      delete _wizardPageOverrides[url];
      for (const c of _wizardCollections) c.extraPages.delete(url);
    }

    function _wizardReviewPageFn(url) {
      delete _wizardPageOverrides[url];
      for (const c of _wizardCollections) c.extraPages.delete(url);
      _wizardExcluded[url] = 'review';
    }

    // ── Collections panel ─────────────────────────────────────────────────────

    // Merge local wizard collections with API (Qdrant) collections into a unified list
    function _wizardMergeCollections(apiColls) {
      const used = new Set();
      const result = [];
      const solId = _wizardSolId || _wizardCurrentSolId();
      for (const wc of _wizardCollections) {
        // Try to match by confirmed name, then by display_name, then by suffix
        const suffix = (wc.display_name || '').toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'');
        const apiMatch = apiColls.find(ac =>
          ac.name === wc.confirmed_collection_name ||
          (ac.display_name || '').toLowerCase() === (wc.display_name || '').toLowerCase() ||
          ac.name === solId + '_' + suffix
        );
        if (apiMatch) used.add(apiMatch.name);
        result.push({ source: apiMatch ? 'both' : 'local', wizColl: wc, apiColl: apiMatch || null });
      }
      // Orphans — exist in Qdrant but not matched to any wizard collection
      for (const ac of apiColls) {
        if (!used.has(ac.name)) {
          result.push({ source: 'qdrant', wizColl: null, apiColl: ac });
        }
      }
      return result;
    }

    function _wizardRenderCollections() {
      const list = document.getElementById('wizardCollList');
      list.innerHTML = '';
      const solId = _wizardSolId || _wizardCurrentSolId();
      // Build API colls array from _wizardConfirmedColls dict
      const apiColls = Object.entries(_wizardConfirmedColls).map(([name, info]) => ({
        name, ...info,
        // display_name from solutions cache if available
        display_name: ((_allSolutions.find(s => s.id === solId) || {}).collections || []).find(c => c.collection_name === name)?.display_name || name,
        doc_type: ((_allSolutions.find(s => s.id === solId) || {}).collections || []).find(c => c.collection_name === name)?.routing?.doc_type || 'general',
      }));
      const merged = _wizardMergeCollections(apiColls);

      if (!merged.length) {
        list.innerHTML = '<p class="wiz-coll-empty">No collections yet. Drag a sitemap here or click + New.</p>';
        return;
      }

      for (const item of merged) {
        const { source, wizColl: c, apiColl } = item;
        const block = document.createElement('div');
        block.className = 'wiz-coll-block';
        if (source === 'local') {
          block.style.cssText = 'border:1px dashed #b0bec5;background:#f8f9fa;';
        } else if (source === 'qdrant') {
          block.style.cssText = 'background:#fffde7;border:1px solid #ffe082;';
        }
        if (c) block.dataset.collId = c._id;

        // ── Drag-and-drop (local / both only) ──
        if (source !== 'qdrant') {
          block.addEventListener('dragover', e => { e.preventDefault(); block.classList.add('dragover'); });
          block.addEventListener('dragleave', () => block.classList.remove('dragover'));
          block.addEventListener('drop', e => {
            e.preventDefault(); block.classList.remove('dragover');
            try {
              const data = JSON.parse(e.dataTransfer.getData('text/plain'));
              if (data.type === 'sitemap') wizardAssignSitemap(data.catId, c._id);
              else if (data.type === 'page') wizardAssignPage(data.url, data.catId, c._id);
            } catch(_) {}
          });
        }

        // ── Top row: name + doc_type + remove ──
        const top = document.createElement('div');
        top.className = 'wiz-coll-top';
        const nameWrap = document.createElement('div');
        nameWrap.style.cssText = 'flex:1;min-width:0;';

        if (source === 'qdrant') {
          // Read-only name for orphan Qdrant collections
          const nameSpan = document.createElement('div');
          nameSpan.style.cssText = 'font-weight:600;font-size:0.88rem;color:#555;';
          nameSpan.textContent = apiColl.display_name || apiColl.name;
          const orphanBadge = document.createElement('div');
          orphanBadge.style.cssText = 'font-size:0.68rem;color:#f57f17;margin-top:1px;';
          orphanBadge.textContent = '⚠️ In Qdrant but not in current plan';
          nameWrap.appendChild(nameSpan);
          nameWrap.appendChild(orphanBadge);
        } else {
          const nameIn = document.createElement('input');
          nameIn.type = 'text'; nameIn.value = c.display_name; nameIn.placeholder = 'Collection name';
          nameIn.style.cssText = 'width:100%;box-sizing:border-box;';
          nameIn.title = 'Display name — Qdrant collection named: {solution_id}_{slugified_name}';
          const qdrantHint = document.createElement('div');
          qdrantHint.style.cssText = 'font-size:0.68rem;color:#aaa;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
          qdrantHint.textContent = _collQdrantName(c);
          nameIn.oninput = () => { c.display_name = nameIn.value; qdrantHint.textContent = _collQdrantName(c); };
          nameWrap.appendChild(nameIn);
          nameWrap.appendChild(qdrantHint);
        }

        top.appendChild(nameWrap);

        // doc_type selector (editable for local/both, readonly display for qdrant)
        const docTypes = _docTypes;
        const effectiveDocType = (source === 'qdrant') ? (apiColl.doc_type || 'general') : (c.doc_type || 'general');
        const dtSel = document.createElement('select');
        dtSel.disabled = (source === 'qdrant');
        const dtOpts = docTypes.map(dt => '<option value="' + dt + '"' + (effectiveDocType===dt?' selected':'') + '>' + dt + '</option>').join('');
        dtSel.innerHTML = dtOpts;
        if (source !== 'qdrant') dtSel.onchange = () => { c.doc_type = dtSel.value; _wizardRenderCollections(); };
        top.appendChild(dtSel);

        // Remove button (local/both only)
        if (source !== 'qdrant') {
          const rmBtn = document.createElement('button');
          rmBtn.className = 'btn-wizard-rm'; rmBtn.title = 'Remove collection'; rmBtn.textContent = '✕';
          rmBtn.onclick = () => {
            if (source === 'both') {
              alert('This collection exists in Qdrant. Delete it from Work with RAG first.');
              return;
            }
            _inlineConfirm(rmBtn, {
              message: 'Remove "' + (c.label || c.display_name) + '" from plan?',
              confirmLabel: 'Remove',
              onConfirm: () => { _wizardCollections = _wizardCollections.filter(x=>x._id!==c._id); _wizardRenderAll(); }
            });
          };
          top.appendChild(rmBtn);
        }
        block.appendChild(top);

        // ── Body: sitemap assignments (local / both only) ──
        if (source !== 'qdrant') {
          const body = document.createElement('div');
          body.className = 'wiz-coll-body';
          if (!c.sitemapIds.size && !c.extraPages.size) {
            body.innerHTML = '<span class="wiz-coll-empty">Drop a sitemap or page here</span>';
          } else {
            for (const catId of c.sitemapIds) {
              const cat = _wizardCategories.find(x => x.id === catId);
              const totalPages = cat ? (cat.url_count || 0) : 0;
              const exclCount = Object.entries(_wizardExcluded).filter(([url]) => {
                const pages = _wizardPages[catId];
                return pages && pages.includes(url);
              }).length;
              const row = document.createElement('div');
              row.className = 'wiz-coll-sm-entry';

              // Dot
              const dot = document.createElement('span');
              dot.className = 'wiz-coll-sm-dot'; dot.textContent = '●';
              row.appendChild(dot);

              // Sitemap name
              const nameSpan = document.createElement('span');
              nameSpan.className = 'wiz-coll-sm-name'; nameSpan.textContent = catId;
              row.appendChild(nameSpan);

              // Clickable page count toggle
              const key = c._id + '::' + catId;
              const isOpen = _wizardCollPagesOpen.has(key);
              const pgBtn = document.createElement('button');
              pgBtn.type = 'button';
              pgBtn.style.cssText = 'background:none;border:none;cursor:pointer;font-size:0.75rem;color:#1a5276;padding:0 0.2rem;text-decoration:underline;flex-shrink:0;';
              pgBtn.textContent = '📄 ' + totalPages + ' pages ' + (isOpen ? '▴' : '▾');
              pgBtn.title = 'Show / hide pages in this collection';
              pgBtn.onclick = () => _wizardToggleSitemapPagePanel(c, catId, pgBtn, row);
              row.appendChild(pgBtn);

              // Excl count
              if (exclCount) {
                const exclSpan = document.createElement('span');
                exclSpan.className = 'wiz-coll-sm-excl'; exclSpan.textContent = '−' + exclCount + ' excl.';
                row.appendChild(exclSpan);
              }

              // Unassign button
              const rmBtn = document.createElement('button');
              rmBtn.className = 'btn-wizard-rm'; rmBtn.title = 'Unassign sitemap'; rmBtn.textContent = '✕';
              rmBtn.style.marginLeft = 'auto';
              rmBtn.onclick = () => {
                _inlineConfirm(rmBtn, {
                  message: 'Unassign "' + catId + '" from this collection?',
                  confirmLabel: 'Unassign',
                  onConfirm: () => {
                    wizardAssignSitemap(catId, null);
                    if (source === 'both') {
                      _wizardFlashWarning('Chunks from this sitemap are still in Qdrant. Use Work with RAG to remove them.');
                    }
                  }
                });
              };
              row.appendChild(rmBtn);

              body.appendChild(row);

              // If panel was open before re-render, re-attach it
              if (isOpen) {
                const panelId = 'wiz-pages-panel-' + c._id + '-' + catId.replace(/[^a-z0-9]/gi,'_');
                const panel = document.createElement('div');
                panel.id = panelId;
                panel.style.cssText = 'max-height:260px;overflow-y:auto;border-top:1px solid #e8eaf0;margin-top:0.25rem;padding:0.2rem 0;background:#fafbfd;border-radius:0 0 6px 6px;';
                _wizardBuildSitemapPagePanel(panel, c, catId);
                body.appendChild(panel);
              }
            }
            if (c.extraPages.size) {
              const ep = document.createElement('div');
              ep.className = 'wiz-coll-extras';
              ep.textContent = '+ ' + c.extraPages.size + ' individual page(s)';
              body.appendChild(ep);
            }

            // File sources (PDFs, CSVs, etc.)
            for (const fs of (c.fileSources || [])) {
              const fsRow = document.createElement('div');
              fsRow.className = 'wiz-coll-sm-entry';
              fsRow.style.cssText = 'color:#555;';

              const icon = document.createElement('span');
              icon.style.cssText = 'font-size:0.8rem;flex-shrink:0;';
              icon.textContent = fs.path.endsWith('.pdf') ? '📄' : fs.path.endsWith('.csv') ? '📊' : '📃';
              fsRow.appendChild(icon);

              const lbl = document.createElement('span');
              lbl.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:0.78rem;';
              lbl.textContent = fs.label || fs.path.split('/').pop();
              lbl.title = fs.path;
              fsRow.appendChild(lbl);

              const rmFs = document.createElement('button');
              rmFs.className = 'btn-wizard-rm'; rmFs.title = 'Remove file source'; rmFs.textContent = '✕';
              rmFs.style.marginLeft = 'auto';
              rmFs.onclick = () => {
                _inlineConfirm(rmFs, {
                  message: 'Remove "' + (fs.label || fs.path.split('/').pop()) + '"?',
                  confirmLabel: 'Remove',
                  onConfirm: () => {
                    c.fileSources = (c.fileSources || []).filter(f => f.path !== fs.path);
                    _wizardRenderCollections(); _wizardAutoSave();
                  }
                });
              };
              fsRow.appendChild(rmFs);
              body.appendChild(fsRow);
            }

            // "＋ Add file source" button
            const addFileBtn = document.createElement('button');
            addFileBtn.type = 'button';
            addFileBtn.style.cssText = 'font-size:0.73rem;color:#1a5276;background:none;border:1px dashed #aed6f1;border-radius:5px;padding:0.1rem 0.45rem;cursor:pointer;margin-top:0.2rem;width:100%;text-align:left;';
            addFileBtn.textContent = '＋ Add file source (PDF, CSV…)';
            addFileBtn.onclick = async () => {
              try {
                const res = await api('/api/pick-file?source_type=pdf');
                if (res && res.path) {
                  if (!c.fileSources) c.fileSources = [];
                  const label = res.path.split('/').pop();
                  if (!c.fileSources.find(f => f.path === res.path)) {
                    c.fileSources.push({ path: res.path, label });
                    _wizardRenderCollections(); _wizardAutoSave();
                  }
                }
              } catch(e) { /* user cancelled picker */ }
            };
            body.appendChild(addFileBtn);
          }
          block.appendChild(body);
        }

        // ── Footer: status + action buttons ──
        const footer = document.createElement('div');
        footer.style.cssText = 'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.4rem;margin-top:0.4rem;padding-top:0.4rem;border-top:1px solid #e8eaf0;';

        // Status badge
        const badgeWrap = document.createElement('div');
        badgeWrap.style.cssText = 'display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;';
        if (source === 'local') {
          const badge = document.createElement('span');
          badge.style.cssText = 'font-size:0.72rem;padding:0.1rem 0.45rem;border-radius:10px;background:#f5f5f5;color:#888;border:1px solid #ddd;';
          badge.textContent = '⚪ Not yet in Qdrant';
          badgeWrap.appendChild(badge);
        } else {
          const pts = apiColl ? (apiColl.points_count || 0) : 0;
          const badge = document.createElement('span');
          if (pts > 0) {
            badge.style.cssText = 'font-size:0.72rem;padding:0.1rem 0.45rem;border-radius:10px;background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7;';
            const nSrc = (c.sitemapIds ? c.sitemapIds.size || 0 : 0) + (c.fileSources ? c.fileSources.length : 0);
            badge.textContent = '✅ Live — ' + pts + ' chunks' + (nSrc > 1 ? ' · ' + nSrc + ' sources' : '');
          } else {
            badge.style.cssText = 'font-size:0.72rem;padding:0.1rem 0.45rem;border-radius:10px;background:#e3f0fd;color:#1a5276;border:1px solid #aed6f1;';
            badge.textContent = '📥 Fetched (0 chunks)';
          }
          badgeWrap.appendChild(badge);
        }
        footer.appendChild(badgeWrap);

        // Action buttons — shown for ALL sources
        const btnWrap = document.createElement('div');
        btnWrap.style.cssText = 'display:flex;gap:0.35rem;align-items:center;flex-shrink:0;';
        // Compute raw collection name (no display decorations)
        // For local: derive expected Qdrant name directly from solId + slugified display_name
        let collName = null;
        if (apiColl) {
          collName = apiColl.name;
        } else if (c && c.confirmed_collection_name) {
          collName = c.confirmed_collection_name;
        } else if (c) {
          const suffix = (c.display_name || '').toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'');
          collName = (solId || '') + '_' + suffix;
        }
        if (collName) {
          const buildBtn = document.createElement('button');
          buildBtn.type = 'button';
          buildBtn.style.cssText = 'font-size:0.75rem;padding:0.18rem 0.55rem;white-space:nowrap;background:#1a5276;color:#fff;border:none;border-radius:5px;cursor:pointer;';
          buildBtn.textContent = '🛠 Work with RAG';
          buildBtn.onclick = () => _openCollectionInBuildRag(solId, collName);
          btnWrap.appendChild(buildBtn);

          // FAQ button — only for faq doc_type (any source)
          if (effectiveDocType === 'faq') {
            const faqBtn = document.createElement('button');
            faqBtn.type = 'button';
            faqBtn.style.cssText = 'font-size:0.75rem;padding:0.18rem 0.55rem;white-space:nowrap;background:#6a1b9a;color:#fff;border:none;border-radius:5px;cursor:pointer;';
            faqBtn.textContent = '📋 FAQ Table';
            faqBtn.onclick = () => {
              const sol = (_allSolutions || []).find(s => s.id === solId);
              const company = (sol && (sol.company_name || sol.display_name)) || 'the company';
              const lang = (sol && sol.language) || document.getElementById('wizardLang').value || 'en';
              generateFaqTable(collName, company, lang);
            };
            btnWrap.appendChild(faqBtn);
          }

          // Qdrant delete removed — all Qdrant mutations happen in Work with RAG
        }

        // "Add to plan" button — for orphan Qdrant-only collections
        if (source === 'qdrant' && apiColl) {
          const addBtn = document.createElement('button');
          addBtn.type = 'button';
          addBtn.style.cssText = 'font-size:0.75rem;padding:0.18rem 0.55rem;white-space:nowrap;background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7;border-radius:5px;cursor:pointer;';
          addBtn.textContent = '➕ Add to plan';
          addBtn.title = 'Create a new plan entry for this Qdrant collection';
          addBtn.onclick = () => {
            const displayName = (apiColl.display_name || apiColl.name)
              .replace(new RegExp('^' + solId + '_'), '');
            if (!confirm('Add "' + displayName + '" to your plan?\\nThis will create a new plan entry linked to the existing Qdrant collection "' + apiColl.name + '".')) return;
            _wizardCollections.push({
              _id: _wizardNextCollId++,
              display_name: displayName,
              doc_type: apiColl.doc_type || effectiveDocType || 'general',
              confirmed_collection_name: apiColl.name,
              sitemapIds: new Set(),
              excludedUrls: new Map(),
              extraPages: new Set(),
              fileSources: [],
            });
            _wizardAutoSave();
            _wizardLoadConfirmedColls(solId);
          };
          btnWrap.appendChild(addBtn);

          // Merge button — link this orphan to an existing local plan entry
          const localColls = _wizardCollections.filter(wc =>
            !_wizardConfirmedColls[wc.confirmed_collection_name] &&
            !Object.keys(_wizardConfirmedColls).some(n => n === wc.confirmed_collection_name)
          );
          if (localColls.length) {
            const mergeBtn = document.createElement('button');
            mergeBtn.type = 'button';
            mergeBtn.style.cssText = 'font-size:0.75rem;padding:0.18rem 0.55rem;white-space:nowrap;background:#fff8e1;color:#e65100;border:1px solid #ffcc02;border-radius:5px;cursor:pointer;';
            mergeBtn.textContent = '🔗 Merge';
            mergeBtn.title = 'Link this Qdrant collection to an existing plan entry';
            mergeBtn.onclick = () => {
              // Build options list from unmatched local collections
              const opts = _wizardCollections
                .filter(wc => !wc.confirmed_collection_name || !_wizardConfirmedColls[wc.confirmed_collection_name])
                .map(wc => wc.display_name + ' (→ ' + _collQdrantName(wc).replace(/^[^\s]+ /,'') + ')');
              if (!opts.length) { alert('No unmatched plan entries to merge with.'); return; }
              const choice = prompt(
                'Merge "' + apiColl.name + '" with which plan entry?\\n\\n' +
                opts.map((o,i) => (i+1) + '. ' + o).join('\\n') +
                '\\n\\nEnter number:'
              );
              if (!choice) return;
              const idx = parseInt(choice) - 1;
              const candidates = _wizardCollections.filter(wc => !wc.confirmed_collection_name || !_wizardConfirmedColls[wc.confirmed_collection_name]);
              if (idx < 0 || idx >= candidates.length) { alert('Invalid selection.'); return; }
              const target = candidates[idx];
              if (!confirm('Link plan entry "' + target.display_name + '" to Qdrant collection "' + apiColl.name + '"?\\nThe plan entry keeps its sitemaps/pages/doc_type. The Qdrant name will be used.')) return;
              target.confirmed_collection_name = apiColl.name;
              _wizardAutoSave();
              _wizardLoadConfirmedColls(solId);
            };
            btnWrap.appendChild(mergeBtn);
          }
        }

        footer.appendChild(btnWrap);
        block.appendChild(footer);

        list.appendChild(block);
      }
    }

    function wizardAddCollection() {
      _wizardCollections.push({
        _id: _wizardNextCollId++,
        display_name: 'New collection',
        doc_type: 'general',
        sitemapIds: new Set(),
        excludedUrls: new Map(),
        extraPages: new Set(),
        fileSources: [],
      });
      _wizardRenderCollections();
      _wizardAutoSave();
    }

    // ── Search ────────────────────────────────────────────────────────────────

    function _wizardOnSearch(val) {
      _wizardSearchQ = val;
      // Auto-expand all sitemaps when searching
      if (val) {
        for (const cat of _wizardCategories) {
          if (!cat.id.startsWith('_')) {
            _wizardExpanded[cat.id] = true;
            if (_wizardPages[cat.id] === undefined) _wizardPages[cat.id] = null;
          }
        }
      }
      _wizardRenderSitemapList();
    }

    // ── Confirm ───────────────────────────────────────────────────────────────

    async function runWizardConfirm() {
      const solName = document.getElementById('wizardSolName').value.trim();
      const lang = document.getElementById('wizardLang').value;
      const log = document.getElementById('wizardConfirmLog');
      document.getElementById('wizardConfirmResult').innerHTML = '';
      log.textContent = 'Creating collections…\\n';
      log.classList.remove('hidden', 'error', 'success');
      if (!solName) { alert('Please select a solution from the top bar first.'); return; }
      const activeColls = _wizardCollections.filter(c => c.sitemapIds.size > 0 || c.extraPages.size > 0);
      if (!activeColls.length) { alert('Assign at least one sitemap or page to a collection.'); return; }

      const collections = activeColls.map(c => {
        const categories = Array.from(c.sitemapIds).map(catId => {
          const cat = _wizardCategories.find(x => x.id === catId);
          // Collect excluded URLs that belong to this sitemap
          const pages = _wizardPages[catId] || [];
          const excluded = pages.filter(u => _wizardExcluded[u]);
          return {
            id: catId,
            sitemap_url: cat ? cat.sitemap_url : null,
            url_filter: cat ? cat.url_filter : null,
            excluded_urls: excluded.length ? excluded : undefined,
          };
        });
        return {
          display_name: c.display_name,
          collection_name: c.display_name.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,''),
          doc_type: c.doc_type,
          categories,
          extra_pages: c.extraPages.size ? Array.from(c.extraPages) : undefined,
        };
      });

      const solId = solName.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'');
      try {
        const res = await fetch('/api/wizard/confirm', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({solution_id: solId, solution_name: solName, language: lang, domain: _wizardDomain, collections})
        });
        const data = await res.json();
        if (!res.ok) { log.textContent += '❌ ' + (data.detail||'Error') + '\\n'; log.classList.add('error'); return; }
        log.classList.add('hidden');
        // Store the actual Qdrant collection name on each _wizardCollections entry
        // so status lookup is exact (not derived from display_name which may differ)
        (data.collections || []).forEach(confirmed => {
          const match = _wizardCollections.find(c =>
            (c.display_name||'').toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'') ===
            confirmed.collection_name.replace(/^[^_]+_/, '')  // strip solId prefix
          );
          if (match) match.confirmed_collection_name = confirmed.collection_name;
        });
        _wizardRenderConfirmResult(data, solId);
        _reloadSolutions(solId);
        _wizardLoadConfirmedColls(solId);
      } catch(err) { log.textContent += 'Error: ' + err + '\\n'; log.classList.add('error'); }
    }

    function _wizardRenderConfirmResult(data, solId) {
      const container = document.getElementById('wizardConfirmResult');
      if (!container) return;
      container.innerHTML = '';

      const header = document.createElement('div');
      header.style.cssText = 'font-weight:600;color:#2e7d32;margin-bottom:0.6rem;font-size:0.92rem;';
      header.textContent = '✅ ' + data.message;
      container.appendChild(header);

      (data.collections || []).forEach(c => {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:0.5rem;padding:0.45rem 0.65rem;margin-bottom:0.4rem;background:#f5f8ff;border:1px solid #c8d8f0;border-radius:7px;';

        const left = document.createElement('div');
        left.style.cssText = 'display:flex;align-items:center;gap:0.5rem;min-width:0;';

        const badge = document.createElement('span');
        badge.style.cssText = 'font-size:0.72rem;padding:0.1rem 0.45rem;border-radius:10px;background:#e3f0fd;color:#1a5276;border:1px solid #aed6f1;white-space:nowrap;flex-shrink:0;';
        badge.textContent = '📥 Fetched';

        const name = document.createElement('span');
        name.style.cssText = 'font-size:0.85rem;font-weight:500;color:#333;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
        name.textContent = c.display_name || c.collection_name;
        if (c.page_count) {
          const pg = document.createElement('span');
          pg.style.cssText = 'font-size:0.75rem;color:#888;margin-left:0.35rem;';
          pg.textContent = '(' + c.page_count + ' pages)';
          name.appendChild(pg);
        }

        left.appendChild(badge);
        left.appendChild(name);

        const openBtn = document.createElement('button');
        openBtn.type = 'button';
        openBtn.style.cssText = 'font-size:0.78rem;padding:0.2rem 0.55rem;white-space:nowrap;background:#1a5276;color:#fff;border:none;border-radius:5px;cursor:pointer;flex-shrink:0;';
        openBtn.textContent = '🛠 Work with RAG';
        openBtn.onclick = () => _openCollectionInBuildRag(solId, c.collection_name);

        row.appendChild(left);
        row.appendChild(openBtn);
        container.appendChild(row);
      });
    }

    async function _openCollectionInBuildRag(solutionId, collectionName) {
      // Step 1: resolve canonical solution ID from the global dropdown.
      const solSel = document.getElementById('globalSolution');
      let canonicalSolId = null;
      const opts = Array.from(solSel.options);
      const normalize = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
      if (opts.some(o => o.value === solutionId)) {
        canonicalSolId = solutionId;
      } else {
        const normTarget = normalize(solutionId);
        const match = opts.find(o => normalize(o.value) === normTarget);
        if (match) canonicalSolId = match.value;
      }

      // Step 2: set global solution and switch to build tab
      if (canonicalSolId) {
        solSel.value = canonicalSolId;
        _currentSolutionId = canonicalSolId;
        // Update global badge
        const sol = _allSolutions.find(s => s.id === canonicalSolId);
        const lang = sol && sol.language ? sol.language : null;
        const badge = document.getElementById('globalSolLang');
        badge.style.display = 'inline-block';
        badge.textContent = lang ? `🌐 ${lang}` : '🌐 set language';
      }

      showTab('build');

      if (canonicalSolId) {
        try {
          await loadSolutionCollections(canonicalSolId);
          const collSel = document.getElementById('collectionSelect');
          if (collSel && collectionName) {
            const opts = Array.from(collSel.options).filter(o => o.value && o.value !== '__new__');
            // Try exact match first
            let target = opts.find(o => o.value === collectionName);
            // Fuzzy: normalize both sides and compare
            if (!target) {
              const norm = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
              const normTarget = norm(collectionName);
              target = opts.find(o => norm(o.value) === normTarget);
            }
            // Partial: check if one contains the other
            if (!target) {
              const normTarget = collectionName.toLowerCase();
              target = opts.find(o => o.value.toLowerCase().includes(normTarget) || normTarget.includes(o.value.toLowerCase()));
            }
            if (target) {
              collSel.value = target.value;
              onCollectionSelect();
              // Auto-select first source so pipeline is visible
              const coll = _currentCollections[target.value];
              const sources = (coll && coll.sources) || [];
              if (sources.length) selectSource(sources[0].id);
            }
          }
        } catch(e) { /* ignore */ }
      }
    }

    function _reloadSolutions(selectSolutionId) {
      fetch('/api/solutions').then(r => r.json()).then(data => {
        _allSolutions = data.solutions || [];
        _populateGlobalSolutionDropdown(selectSolutionId || _currentSolutionId);
        _wizardPopulateSolNameList();
      }).catch(() => {});
    }

    // ── Wizard state persistence ──────────────────────────────────────────────

    // Serialise JS state (Set/Map → plain objects) for JSON storage
    function _wizardSerialiseState() {
      return {
        domain: _wizardDomain,
        sol_name: document.getElementById('wizardSolName').value.trim(),
        sol_lang: document.getElementById('wizardLang').value,
        saved_at: new Date().toISOString(),
        categories: _wizardCategories,
        collections: _wizardCollections.map(c => ({
          _id: c._id,
          display_name: c.display_name,
          doc_type: c.doc_type,
          confirmed_collection_name: c.confirmed_collection_name || null,
          sitemapIds: Array.from(c.sitemapIds),
          // excludedUrls: Map<catId, Set<url>> → {catId: [url, ...]}
          excludedUrls: Object.fromEntries(
            Array.from(c.excludedUrls.entries()).map(([k, v]) => [k, Array.from(v)])
          ),
          extraPages: Array.from(c.extraPages),
          fileSources: c.fileSources || [],
        })),
        next_coll_id: _wizardNextCollId,
        page_overrides: _wizardPageOverrides,  // url → collId (plain obj, already serialisable)
        excluded: _wizardExcluded,              // url → catId  (plain obj)
        // pages: catId → string[] — only save loaded pages (non-null)
        pages: Object.fromEntries(
          Object.entries(_wizardPages).filter(([, v]) => Array.isArray(v))
        ),
        // previews: url → text — cached page content (skip empty strings to save space)
        previews: Object.fromEntries(
          Object.entries(_wizardPreviews).filter(([, v]) => v)
        ),
      };
    }

    // Restore JS state from a saved JSON object and re-render
    function _wizardRestoreState(saved) {
      _wizardDomain = saved.domain || '';
      _wizardCategories = saved.categories || [];
      _wizardNextCollId = saved.next_coll_id || 0;
      _wizardPageOverrides = saved.page_overrides || {};
      _wizardExcluded = saved.excluded || {};
      _wizardPreviews = saved.previews || {};
      _wizardSearchQ = '';
      _wizardExpanded = {};

      // Restore pages (null for sitemaps that had pages but were not loaded)
      _wizardPages = {};
      for (const cat of _wizardCategories) {
        if (cat.id.startsWith('_')) continue;
        if (saved.pages && saved.pages[cat.id]) {
          _wizardPages[cat.id] = saved.pages[cat.id];
          _wizardExpanded[cat.id] = false; // not expanded by default
        } else if (cat.sample_urls && cat.sample_urls.length) {
          _wizardPages[cat.id] = null; // not loaded yet
        }
      }

      // Restore collections (convert plain objects back to Set/Map)
      _wizardCollections = (saved.collections || []).map(c => ({
        _id: c._id,
        display_name: c.display_name,
        doc_type: c.doc_type,
        confirmed_collection_name: c.confirmed_collection_name || null,
        sitemapIds: new Set(c.sitemapIds || []),
        excludedUrls: new Map(
          Object.entries(c.excludedUrls || {}).map(([k, v]) => [k, new Set(v)])
        ),
        extraPages: new Set(c.extraPages || []),
        fileSources: c.fileSources || [],
      }));

      // Restore input fields
      const urlInput = document.getElementById('wizardUrl');
      const solInput = document.getElementById('wizardSolName');
      const langSel  = document.getElementById('wizardLang');
      if (urlInput && saved.domain) urlInput.value = saved.domain;
      if (solInput && saved.sol_name) {
        solInput.value = saved.sol_name;
        // Cache the solId immediately so _wizardRenderCollections() can use it synchronously
        _wizardSolId = saved.sol_name.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/,'');
        // Sync global solution dropdown if this solution exists
        const globalSel = document.getElementById('globalSolution');
        const matchOpt = Array.from(globalSel.options).find(o => o.value === _wizardSolId);
        if (matchOpt) {
          globalSel.value = _wizardSolId;
          onGlobalSolutionChange();
        }
      }
      if (langSel  && saved.sol_lang) langSel.value = saved.sol_lang;

      // Show results and re-render
      const saveBtn = document.getElementById('btnWizardSave');
      if (saveBtn) saveBtn.style.display = '';
      document.getElementById('wizardResults').style.display = 'flex';
      document.getElementById('wizardSearch').value = '';
      // Activate wizard chat (session has analysis data)
      const chatHint = document.getElementById('wizardChatHint');
      if (chatHint) chatHint.style.display = 'none';
      const chatSec = document.getElementById('wizardChatSection');
      if (chatSec) chatSec.style.display = '';
      _wizardRenderAll();
      // Load confirmed collection statuses from the API
      _wizardLoadConfirmedColls(_wizardCurrentSolId());
    }

    // Derive solution_id from current sol name input
    function _wizardCurrentSolId() {
      const name = document.getElementById('wizardSolName').value.trim();
      return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/, '');
    }

    // Save session to disk
    async function wizardSaveSession() {
      const solId = _wizardCurrentSolId();
      if (!solId) { alert('Select a solution from the top bar before saving.'); return; }
      const state = _wizardSerialiseState();
      const statusEl = document.getElementById('wizardAutoSaveStatus');
      try {
        const res = await fetch('/api/wizard/save', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({solution_id: solId, state})
        });
        if (!res.ok) throw new Error((await res.json()).detail || 'Save failed');
        if (statusEl) {
          statusEl.style.display = '';
          statusEl.textContent = '💾 Saved at ' + new Date().toLocaleTimeString();
        }
      } catch(err) {
        if (statusEl) {
          statusEl.style.display = '';
          statusEl.textContent = '⚠️ Save failed: ' + err.message;
        }
        console.error('Wizard save failed:', err);
      }
    }

    // Auto-save wrapper (called after state changes)
    let _wizardAutoSaveTimer = null;
    function _wizardAutoSave() {
      const solId = _wizardCurrentSolId();
      if (!solId || !_wizardCategories.length) return;
      clearTimeout(_wizardAutoSaveTimer);
      _wizardAutoSaveTimer = setTimeout(() => {
        wizardSaveSession();
      }, 1500); // debounce: save 1.5s after last change
    }

    // Load session — restores state from disk
    async function wizardLoadSession(solId) {
      // Close dropdown
      const dd = document.getElementById('wizardLoadDropdown');
      if (dd) dd.style.display = 'none';
      try {
        const res = await fetch('/api/wizard/load?solution_id=' + encodeURIComponent(solId));
        if (!res.ok) { alert('Could not load session for "' + solId + '".'); return; }
        const data = await res.json();
        _wizardRestoreState(data.state);
        const log = document.getElementById('wizardLog');
        log.textContent = '📂 Session "' + solId + '" loaded from disk.\\n';
        log.classList.remove('hidden', 'error', 'success');
        const saveBtn = document.getElementById('btnWizardSave');
        if (saveBtn) saveBtn.style.display = '';
      } catch(err) {
        alert('Load failed: ' + err.message);
      }
    }

    // Show the load-session dropdown with list of saved sessions
    async function _wizardShowLoadDropdown(btn) {
      const dd = document.getElementById('wizardLoadDropdown');
      if (!dd) return;
      // Toggle
      if (dd.style.display !== 'none') { dd.style.display = 'none'; return; }
      dd.innerHTML = '<span class="wiz-load-empty">Loading…</span>';
      dd.style.display = 'block';
      // Close on outside click
      const closeDD = (e) => {
        if (!dd.contains(e.target) && e.target !== btn) {
          dd.style.display = 'none';
          document.removeEventListener('click', closeDD);
        }
      };
      setTimeout(() => document.addEventListener('click', closeDD), 10);

      try {
        const res = await fetch('/api/wizard/list-saves');
        const data = await res.json();
        dd.innerHTML = '';
        const saves = data.saves || [];
        if (!saves.length) {
          const em = document.createElement('span');
          em.className = 'wiz-load-empty';
          em.textContent = 'No saved sessions yet.';
          dd.appendChild(em);
          return;
        }
        saves.sort((a, b) => b.mtime - a.mtime); // newest first
        for (const s of saves) {
          const btn2 = document.createElement('button');
          btn2.className = 'wiz-load-item';
          const date = new Date(s.mtime * 1000).toLocaleString();
          btn2.title = date;
          btn2.textContent = s.solution_id + '  (' + date + ')';
          btn2.onclick = () => wizardLoadSession(s.solution_id);
          dd.appendChild(btn2);
        }
      } catch(err) {
        dd.innerHTML = '<span class="wiz-load-empty">Error: ' + _escHtml(err.message) + '</span>';
      }
    }

    // ── DomCop status & update ────────────────────────────────────────────────

    async function _loadDomcopStatus() {
      const badge   = document.getElementById('domcopBadge');
      const info    = document.getElementById('domcopInfo');
      const btnUpd  = document.getElementById('btnDomcopUpdate');
      try {
        const r = await fetch('/api/sites/domcop-status');
        const d = await r.json();
        if (d.available) {
          const count = (d.domain_count || 0).toLocaleString();
          // Format last_updated as a readable date
          let dateLabel = d.last_updated || '';
          if (dateLabel) {
            try { dateLabel = new Date(dateLabel).toLocaleDateString(undefined, {year:'numeric',month:'short',day:'numeric'}); }
            catch(_) {}
          }
          badge.textContent = '✅ Ready';
          badge.style.background = '#e6f4ec';
          badge.style.color = '#1a7b3a';
          info.textContent = `${count} domains · Last updated: ${dateLabel || 'unknown'}`;
        } else {
          badge.textContent = '⚠ Not downloaded';
          badge.style.background = '#fff3cd';
          badge.style.color = '#7a5a00';
          info.textContent = 'Database not found — click Update now to download (~100 MB).';
        }
        btnUpd.disabled = false;
      } catch(e) {
        badge.textContent = 'Error';
        badge.style.color = '#c00';
        info.textContent = 'Could not reach /api/sites/domcop-status';
      }
    }

    async function updateDomcop() {
      const btnUpd  = document.getElementById('btnDomcopUpdate');
      const dcLog   = document.getElementById('domcopLog');
      const badge   = document.getElementById('domcopBadge');

      btnUpd.disabled = true;
      dcLog.textContent = '';
      dcLog.classList.remove('hidden', 'error');
      badge.textContent = '⏳ Updating…';
      badge.style.background = '#e8f0fe';
      badge.style.color = '#1a56a0';

      // Kick off background download
      const startRes = await fetch('/api/sites/update-domcop', {method:'POST'});
      if (!startRes.ok) {
        dcLog.textContent = 'Failed to start DomCop update.';
        dcLog.classList.add('error');
        btnUpd.disabled = false;
        return;
      }

      // Stream progress via shared SSE
      const sse = new EventSource('/api/progress');
      sse.onmessage = async e => {
        const msg = e.data;
        if (msg.startsWith('LOG:')) {
          dcLog.textContent += msg.slice(4);
          dcLog.scrollTop = dcLog.scrollHeight;
        } else if (msg.startsWith('DONE:')) {
          sse.close();
          btnUpd.disabled = false;
          await _loadDomcopStatus();   // refresh status card
        } else if (msg.startsWith('ERROR:')) {
          sse.close();
          dcLog.textContent += '❌ ' + msg.slice(6) + '\\n';
          dcLog.classList.add('error');
          badge.textContent = '❌ Failed';
          badge.style.background = '#fde8e8';
          badge.style.color = '#c00';
          btnUpd.disabled = false;
        }
      };
      sse.onerror = () => { sse.close(); btnUpd.disabled = false; };
    }

    // ── Prospect Sites Analyzer ──────────────────────────────────────────────

    let _sitesAllResults = [];

    async function runSiteAnalyze() {
      const raw = document.getElementById('sitesUrlInput').value;
      const urls = raw.split('\\n').map(s => s.trim()).filter(Boolean);
      if (!urls.length) { alert('Please enter at least one URL.'); return; }

      const log = document.getElementById('sitesLog');
      const status = document.getElementById('sitesStatus');
      const tableWrap = document.getElementById('sitesTableWrap');

      log.textContent = '';
      log.classList.remove('hidden', 'error');
      tableWrap.style.display = 'none';
      document.getElementById('btnSitesCsv').style.display = 'none';
      status.textContent = '';
      _sitesAllResults = [];

      // Start analysis
      const startRes = await fetch('/api/sites/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({urls})
      });
      if (!startRes.ok) {
        log.textContent = 'Failed to start analysis.';
        log.classList.add('error');
        return;
      }

      // Stream progress via SSE
      const sse = new EventSource('/api/progress');
      sse.onmessage = async e => {
        const msg = e.data;
        if (msg.startsWith('LOG:')) {
          log.textContent += msg.slice(4);
          log.scrollTop = log.scrollHeight;

          // Fetch partial results after each completed URL
          if (msg.includes('✅')) {
            try {
              const r = await fetch('/api/sites/results');
              const data = await r.json();
              _sitesAllResults = data.results || [];
              _renderSitesTable(_sitesAllResults);
              tableWrap.style.display = 'block';
            } catch(_) {}
          }
        } else if (msg.startsWith('DONE:')) {
          sse.close();
          try {
            const r = await fetch('/api/sites/results');
            const data = await r.json();
            _sitesAllResults = data.results || [];
            _renderSitesTable(_sitesAllResults);
            tableWrap.style.display = 'block';
            document.getElementById('btnSitesCsv').style.display = '';
            status.textContent = `✅ Done — ${_sitesAllResults.length} site(s) analysed.`;
          } catch(e) {
            log.textContent += 'Error fetching results: ' + e.message + '\\n';
          }
        } else if (msg.startsWith('ERROR:')) {
          sse.close();
          log.textContent += '❌ ' + msg.slice(6) + '\\n';
          log.classList.add('error');
        }
      };
      sse.onerror = () => sse.close();
    }

    // Extract root hostname from a URL string (e.g. "https://store.foo.com/x" → "store.foo.com")
    function _siteDomain(url) {
      try { return new URL(url).hostname; } catch(_) { return url; }
    }

    function _renderSitesTable(results) {
      const tbody = document.getElementById('sitesTableBody');
      tbody.innerHTML = '';
      const filter = (document.getElementById('sitesFilter').value || '').toLowerCase();
      let shown = 0;

      for (const r of results) {
        const rowStr = JSON.stringify(r).toLowerCase();
        if (filter && !rowStr.includes(filter)) continue;
        shown++;

        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid #eee';
        tr.dataset.json = JSON.stringify(r);

        const platformColor = r.platform === 'Unknown' || r.platform === 'Error'
          ? '#999' : (r.platform_confidence === 'high' ? '#1a7b3a' : '#7a5a00');
        const chatbotColor = r.chatbot === 'None detected' ? '#bbb' : '#1a56a0';
        const domain = _siteDomain(r.url);
        const swUrl  = 'https://www.similarweb.com/website/' + encodeURIComponent(domain) + '/';
        const ahUrl  = 'https://ahrefs.com/traffic-checker/?input=' + encodeURIComponent(domain);
        const rankLabel = r.rank || 'N/A';
        const rankSource = r.rank_source || '';
        const rankColor = rankLabel.startsWith('#') ? '#1a5f3a' : '#999';

        // Chatbot category badge
        const _catStyles = {
          'Bot':       'color:#7a3aaa;border-color:#c7a8e8;',
          'Live Chat': 'color:#1a5f8a;border-color:#a8c8e8;',
          'Hybrid':    'color:#7a5a00;border-color:#e8d8a0;',
        };
        const catStyle = _catStyles[r.chatbot_category] || '';
        const catBadge = catStyle
          ? ` <span style="font-size:0.65rem;border:1px solid;border-radius:9px;
               padding:0 4px;vertical-align:middle;${catStyle}">${_escHtml(r.chatbot_category)}</span>`
          : '';

        // Contact cell: icons + tooltips
        const contactParts = [];
        if (r.contact_form)
          contactParts.push('<span title="Embedded contact form" style="cursor:default;">📧 Form</span>');
        if (r.contact_mailto) {
          const firstMail = r.contact_mailto.split(',')[0].trim();
          contactParts.push(`<a href="mailto:${_escHtml(firstMail)}" title="${_escHtml(r.contact_mailto)}"
            style="color:#0066cc;text-decoration:none;">✉ Mail</a>`);
        }
        if (r.contact_whatsapp)
          contactParts.push(`<span title="${_escHtml(r.contact_whatsapp)}"
            style="color:#25d366;cursor:default;">💬 WA</span>`);
        if (r.contact_phone)
          contactParts.push(`<span title="${_escHtml(r.contact_phone)}"
            style="color:#555;cursor:default;">📞</span>`);
        const contactCell = contactParts.length
          ? contactParts.join(' <span style="color:#ccc;">·</span> ')
          : '<span style="color:#ddd;">—</span>';

        tr.innerHTML = [
          `<td style="padding:6px 10px;max-width:220px;word-break:break-all;">
            <a href="${_escHtml(r.url)}" target="_blank" rel="noopener"
              style="color:#0066cc;font-size:0.8rem;">${_escHtml(r.url)}</a>
            ${r.error ? '<br><span style="color:#c00;font-size:0.75rem;">⚠ ' + _escHtml(r.error) + '</span>' : ''}
          </td>`,
          `<td style="padding:6px 8px;color:${platformColor};font-weight:500;">
            ${_escHtml(r.platform || '—')}
            ${r.platform_confidence && r.platform_confidence !== 'none'
              ? '<span style="font-size:0.72rem;color:#999;font-weight:400;"> (' + r.platform_confidence + ')</span>' : ''}
          </td>`,
          `<td style="padding:6px 8px;color:${chatbotColor};">${_escHtml(r.chatbot || '—')}${catBadge}</td>`,
          `<td style="padding:6px 8px;color:#555;">${_escHtml(r.cms || '—')}</td>`,
          `<td style="padding:6px 8px;text-align:center;">${r.ssl ? '🔒' : '⚠️'}</td>`,
          `<td style="padding:6px 8px;white-space:nowrap;">
            <span style="font-size:0.8rem;font-weight:500;color:${rankColor};"
              title="${rankSource ? rankSource + ' rank' : 'Global rank'}">${_escHtml(rankLabel)}</span>
            ${rankSource ? '<span style="font-size:0.68rem;color:#aaa;margin-left:2px;">(' + _escHtml(rankSource) + ')</span>' : ''}
            &nbsp;
            <a href="${_escHtml(swUrl)}" target="_blank" rel="noopener"
              title="Open SimilarWeb for ${_escHtml(domain)}"
              style="font-size:0.72rem;color:#0066cc;text-decoration:none;border:1px solid #c8d8f0;border-radius:3px;padding:1px 4px;">SW↗</a>
            <a href="${_escHtml(ahUrl)}" target="_blank" rel="noopener"
              title="Open Ahrefs Traffic Checker for ${_escHtml(domain)}"
              style="font-size:0.72rem;color:#0066cc;text-decoration:none;border:1px solid #c8d8f0;border-radius:3px;padding:1px 4px;margin-left:2px;">AH↗</a>
          </td>`,
          `<td style="padding:6px 8px;font-size:0.78rem;color:#555;">${_escHtml(r.payments || '—')}</td>`,
          `<td style="padding:6px 8px;font-size:0.78rem;color:#555;">${_escHtml(r.social_links || '—')}</td>`,
          `<td style="padding:6px 8px;text-align:center;">${r.has_blog ? '✅' : '—'}</td>`,
          `<td style="padding:6px 8px;font-size:0.8rem;white-space:nowrap;">${contactCell}</td>`,
        ].join('');

        tbody.appendChild(tr);
      }

      document.getElementById('sitesTableCount').textContent =
        filter ? `Showing ${shown} of ${results.length}` : `${results.length} site(s)`;
    }

    function _filterSitesTable() {
      _renderSitesTable(_sitesAllResults);
    }

    async function exportSitesCsv() {
      const a = document.createElement('a');
      a.href = '/api/sites/export-csv';
      a.download = 'site_analysis.csv';
      a.click();
    }

  </script>
  <footer style="text-align:center;padding:1.2rem 0 0.8rem;font-size:0.78rem;color:#aaa;">
    jabberbrain RAG builder &nbsp;·&nbsp; v__APP_VERSION__
    &nbsp;&nbsp;
    <button type="button" id="btnShutdown" class="btn-shutdown" onclick="shutdownServer()"
      title="Stop the server (same as Ctrl+C in terminal)">⏹ Stop server</button>
  </footer>

  <!-- FAQ Table Modal — lives outside all panels so it works from any tab -->
  <div id="faqTableModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:600;align-items:center;justify-content:center;" onclick="if(event.target===this)this.style.display='none';">
    <div style="background:#fff;border-radius:12px;padding:1.5rem;max-width:700px;width:92%;box-shadow:0 8px 24px rgba(0,0,0,0.2);max-height:85vh;display:flex;flex-direction:column;">
      <div style="font-weight:600;font-size:1rem;margin-bottom:0.5rem;">📋 FAQ Table — <span id="faqTableCollName"></span></div>
      <div id="faqTableStatus" style="font-size:0.82rem;color:#666;margin-bottom:0.4rem;">Generating…</div>
      <textarea id="faqTableContent" style="flex:1;width:100%;min-height:260px;font-family:monospace;font-size:0.77rem;border:1px solid #ddd;border-radius:6px;padding:0.5rem;box-sizing:border-box;display:none;resize:vertical;" readonly></textarea>
      <div style="margin-top:0.75rem;display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;">
        <button id="btnCopyFaq" onclick="copyFaqTable()" style="display:none;">📋 Copy all</button>
        <button onclick="document.getElementById('faqTableModal').style.display='none'">Close</button>
        <span style="font-size:0.75rem;color:#999;margin-left:auto;">Tab-separated · Paste into jBKE Knowledge Editor</span>
      </div>
    </div>
  </div>
</body>
</html>
"""
