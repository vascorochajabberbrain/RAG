"""
Minimal FastAPI app for the RAG workflow. Run with: uvicorn web.app:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Any
import os, queue, threading, asyncio

# App version (read from VERSION file in project root)
def _read_version() -> str:
    try:
        _vf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION")
        return open(_vf).read().strip()
    except Exception:
        return "dev"

APP_VERSION = _read_version()

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

@app.on_event("startup")
async def _capture_loop():
    global _loop
    _loop = asyncio.get_running_loop()


class StepRequest(BaseModel):
    step: str
    state_update: Optional[dict] = None


class QARequest(BaseModel):
    collection_name: str  # single name, "__all__", or comma-separated list
    question: str
    company: str = "Assistant"
    solution_id: Optional[str] = None


@app.get("/")
def root():
    html = _INDEX_HTML.replace("__APP_VERSION__", APP_VERSION)
    return HTMLResponse(html)


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


@app.get("/api/solutions/{solution_id}/collections")
def solution_collections(solution_id: str):
    """Return all Qdrant collections registered under a solution."""
    try:
        from solution_specs import get_solution, get_collections
        sol = get_solution(solution_id)
        if not sol:
            raise HTTPException(status_code=404, detail=f"Solution '{solution_id}' not found")
        tracker = get_state().tracker
        all_qdrant = tracker.all_collections()
        # collections is always a list of dicts in the new schema
        coll_entries = get_collections(solution_id)
        result = [
            {
                "name": c["collection_name"],
                "id": c.get("id", c["collection_name"]),
                "display_name": c.get("display_name", c["collection_name"]),
                "scraper_name": c.get("scraper_name", ""),
                "routing": c.get("routing", {}),
                "exists": c["collection_name"] in all_qdrant,
            }
            for c in coll_entries if c.get("collection_name")
        ]
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

@app.put("/api/solutions/{solution_id}/collections/{collection_id}/routing")
def update_collection_routing(solution_id: str, collection_id: str, req: UpdateRoutingRequest):
    """Update the routing metadata block for a specific collection in solutions.yaml."""
    from workflow.suggest import save_routing_metadata
    success = save_routing_metadata(solution_id, collection_id, req.routing)
    if not success:
        raise HTTPException(status_code=404,
                            detail=f"Collection '{collection_id}' in solution '{solution_id}' not found")
    return {"message": f"Routing metadata updated for {solution_id}/{collection_id}"}


@app.post("/api/workflow/step")
def run_workflow_step(req: StepRequest):
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
    retrieved = get_retrieved_info(req.question, history, collection_names)
    answer = get_answer(history, retrieved, req.question, req.company)
    return {"question": req.question, "answer": answer}


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
def state_check_by_collection(collection_name: str):
    """Check if a .rag_state_{collection_name}.json exists in project root."""
    import os, json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root, f".rag_state_{collection_name}.json")
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {
                "found": True,
                "save_path": save_path,
                "completed_steps": d.get("completed_steps", []),
                "collection_name": d.get("collection_name"),
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
    for k, v in state_update.items():
        if k == "source_config" and isinstance(v, dict):
            state.source_config = v
        elif k == "chunking_config" and isinstance(v, dict):
            pass  # handled elsewhere
        elif hasattr(state, k):
            setattr(state, k, v)


@app.post("/api/workflow/fetch")
def run_fetch_streaming(req: StepRequest):
    """Run FETCH in a background thread, streaming stdout lines via SSE progress queue."""
    from workflow.models import Step, WorkflowState, ChunkingConfig
    from workflow.runner import run_step
    import io, sys

    state = get_state()
    _apply_state_update(state, req.state_update)
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


_INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG – Build &amp; Chat</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 1.5rem; background: #f8f9fa; color: #1a1a1a; }
    .container { max-width: 720px; margin: 0 auto; }
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
    .tabs { display: flex; gap: 0.25rem; margin-bottom: 1rem; }
    .tab { padding: 0.5rem 1rem; border-radius: 6px; background: #e9ecef; border: none; cursor: pointer; font-size: 0.9rem; }
    .tab.active { background: #0066cc; color: #fff; }
    .tab:hover:not(.active) { background: #dee2e6; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>RAG – Build &amp; Chat</h1>
    <p class="subtitle">Build a RAG pipeline or chat with an existing solution. No terminal needed.</p>

    <div class="tabs">
      <button type="button" class="tab active" data-tab="build">Build RAG</button>
      <button type="button" class="tab" data-tab="chat">Chat / Test Q&A</button>
    </div>

    <div id="panel-build" class="panel">
      <div class="card">
        <h2>1. Solution &amp; Collection</h2>

        <!-- Step 1: pick a solution -->
        <label>Solution</label>
        <select id="solutionBuild" onchange="onSolutionChange()">
          <option value="">— Select a solution —</option>
        </select>

        <!-- Shown when a solution IS selected: pick which collection within it -->
        <div id="collectionSection" style="display:none;margin-top:0.75rem;">
          <label>Collection <span style="font-weight:400;color:#888;font-size:0.85rem;">— which index to build or update</span></label>
          <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.4rem;">
            <select id="collectionSelect" onchange="onCollectionSelect()" style="margin-bottom:0;flex:1;"></select>
            <button type="button" id="btnDeleteCollection" onclick="deleteCollection()" title="Delete this collection from Qdrant"
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

        <!-- Shown when NO solution is selected: type collection name directly -->
        <div id="noSolutionRow" style="margin-top:0.75rem;display:block;">
          <label>Collection name <span style="font-weight:400;color:#888;font-size:0.85rem;">— or select a solution above</span></label>
          <input type="text" id="collectionNameDirect" placeholder="e.g. my_rag">
        </div>
      </div>
      <div class="card">
        <h2>2. Source</h2>
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
      <div class="card">
        <h2>3. Run pipeline</h2>
        <p class="status">Create collection → Fetch → (Translate &amp; Clean) → Chunk → Push to Qdrant</p>
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
        <label>Solution</label>
        <select id="solutionChat" onchange="onChatSolutionChange()">
          <option value="">— Pick a solution —</option>
        </select>
        <div id="chatCollectionRow" style="display:none;margin-top:0.6rem;">
          <label>Collection</label>
          <select id="chatCollectionSelect" style="margin-bottom:0.75rem;"></select>
        </div>
        <label>Question</label>
        <textarea id="qaQuestion" rows="2" placeholder="Ask something about the content…"></textarea>
        <button type="button" class="btn-primary" id="runQA">Ask</button>
        <div id="qaResult" class="log"></div>
      </div>
    </div>
  </div>

  <script>
    const api = (path, body) => fetch(path, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined
    }).then(r => r.json());

    const buildLog = document.getElementById('buildLog');
    const qaResult = document.getElementById('qaResult');
    const setLog = (el, msg, isError) => {
      el.textContent = msg;
      el.className = 'log' + (isError ? ' error' : ' success');
    };

    // Load solutions into dropdowns
    let _allSolutions = [];
    (async () => {
      const { solutions } = await api('/api/solutions');
      _allSolutions = solutions || [];
      const selBuild = document.getElementById('solutionBuild');
      const selChat = document.getElementById('solutionChat');
      _allSolutions.forEach(s => {
        const makeOpt = (parent) => {
          const o = document.createElement('option');
          o.value = s.id;
          o.textContent = s.display_name || s.id;
          o.dataset.company = s.company_name || '';
          parent.appendChild(o);
        };
        makeOpt(selBuild);
        makeOpt(selChat);
      });
    })();

    async function onSolutionChange() {
      const sel = document.getElementById('solutionBuild');
      const solId = sel.value;
      const noSolRow = document.getElementById('noSolutionRow');
      const collSection = document.getElementById('collectionSection');

      if (!solId) {
        noSolRow.style.display = 'block';
        collSection.style.display = 'none';
        renderSubCollectionPicker([]);
        return;
      }
      noSolRow.style.display = 'none';
      collSection.style.display = 'block';

      // Load collections for this solution — renders pills + populates collectionSelect
      await loadSolutionCollections(solId);
      renderRecentFiles();
    }

    // _currentCollections: full collection objects from the API, keyed by collection_name
    let _currentCollections = {};

    async function loadSolutionCollections(solId) {
      const collSelect = document.getElementById('collectionSelect');
      collSelect.innerHTML = '<option value="">Loading…</option>';
      try {
        const res = await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections').then(r => r.json());
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
        // Single collection — auto-set scraper if available, no pills needed, but still show routing panel
        const c = collections[0];
        if (c.scraper_name) {
          document.getElementById('sourceType').value = 'url';
          document.getElementById('scraperName').value = c.scraper_name;
          onSourceTypeChange();
          checkUrlSavedState(c.name);
        }
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
        if (c.scraper_name) {
          document.getElementById('sourceType').value = 'url';
          document.getElementById('scraperName').value = c.scraper_name;
          onSourceTypeChange();
        }
        const collSelect = document.getElementById('collectionSelect');
        if (collSelect) { collSelect.value = c.name; onCollectionSelect(); }
        if (c.scraper_name) checkUrlSavedState(c.name);
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
            <button type="button" id="btnEditRouting" style="font-size:0.78rem;padding:0.15rem 0.5rem;background:#333;border:1px solid #555;color:#ccc;border-radius:4px;cursor:pointer;">Edit</button>
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
      if (val === '__new__') {
        newRow.style.display = 'block';
        info.style.display = 'none';
        delBtn.style.display = 'none';
      } else {
        newRow.style.display = 'none';
        info.style.display = 'block';
        const opt = document.getElementById('collectionSelect').options[document.getElementById('collectionSelect').selectedIndex];
        const exists = opt.dataset.exists === '1';
        info.textContent = exists ? '✓ Collection exists in Qdrant' : '⚠ Not yet pushed to Qdrant';
        delBtn.style.display = exists ? 'block' : 'none';
        // Check for URL-source saved state when collection changes
        const st = document.getElementById('sourceType').value;
        if (st === 'url') checkUrlSavedState(val);
      }
    }

    async function deleteCollection() {
      const solId = document.getElementById('solutionBuild').value;
      const name = document.getElementById('collectionSelect').value;
      if (!name || name === '__new__') return;
      if (!confirm(`Delete collection "${name}" from Qdrant and this solution?\n\nThis cannot be undone.`)) return;
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

    async function registerCollection() {
      const solId = document.getElementById('solutionBuild').value;
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
      const solId = document.getElementById('solutionBuild').value;
      if (solId) {
        const val = document.getElementById('collectionSelect').value;
        if (val && val !== '__new__') return val;
        return document.getElementById('collectionName').value.trim();
      }
      return document.getElementById('collectionNameDirect').value.trim();
    }

    // Tabs
    document.querySelectorAll('.tab').forEach(t => {
      t.onclick = () => {
        document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(x => x.classList.add('hidden'));
        t.classList.add('active');
        document.getElementById('panel-' + t.dataset.tab).classList.remove('hidden');
      };
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
        if (collName && collName !== '__new__') checkUrlSavedState(collName);
      } else {
        document.getElementById('urlResumeBanner').style.display = 'none';
      }
    }

    let _urlSavedStatePath = null;

    async function checkUrlSavedState(collectionName) {
      if (!collectionName || collectionName === '__new__') {
        document.getElementById('urlResumeBanner').style.display = 'none';
        return;
      }
      try {
        const res = await fetch('/api/state/check-by-collection?collection_name=' + encodeURIComponent(collectionName)).then(r => r.json());
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
      const sel = document.getElementById('solutionBuild');
      const solId = sel.value || null;
      const solName = solId ? (sel.options[sel.selectedIndex].textContent || solId) : null;
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

      const activeSolId = document.getElementById('solutionBuild').value || null;

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
          // If file belongs to a different solution, switch the solution dropdown
          if (f.solution_id && f.solution_id !== document.getElementById('solutionBuild').value) {
            document.getElementById('solutionBuild').value = f.solution_id;
            await onSolutionChange();
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
        }
      } finally {
        btn.textContent = '📂 Browse…';
        btn.disabled = false;
      }
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
      } else if (path) source_config = { path, source_label: path.split('/').pop() };
      return {
        collection_name: getCollectionName(),
        source_type: st,
        source_config,
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
          const solId = document.getElementById('solutionBuild').value;
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

    document.getElementById('runCreate').onclick = () => runStep('create_collection');
    document.getElementById('runFetch').onclick = () => runFetchWithProgress();
    document.getElementById('runTranslate').onclick = () => runTranslateWithProgress();
    document.getElementById('runChunk').onclick = () => runStep('chunk');
    document.getElementById('runPush').onclick = () => runPushWithProgress();
    document.getElementById('runSync').onclick = () => runSync();

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
        } else if (data.startsWith('ERROR') || data === 'TIMEOUT') {
          buildLog.textContent += '\\n❌ ' + data;
          buildLog.className = 'log error';
          es.close();
          _btnDone(btn); btn.textContent = '2. Fetch';
        }
      };
      es.onerror = () => { es.close(); _btnDone(btn); btn.textContent = '2. Fetch'; };

      try {
        await api('/api/workflow/fetch', { step: 'fetch', state_update: getStateUpdate() });
      } catch (e) {
        setLog(buildLog, e.message || String(e), true);
        es.close();
        btn.disabled = false;
        btn.textContent = '2. Fetch';
      }
    }

    async function runPushWithProgress() {
      const btn = document.getElementById('runPush');
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
        } else if (data.startsWith('ERROR') || data === 'TIMEOUT') {
          buildLog.textContent += '\\n❌ ' + data;
          buildLog.className = 'log error';
          es.close();
          _btnDone(btn);
        }
      };
      es.onerror = () => { es.close(); _btnDone(btn); };

      try {
        await api('/api/workflow/push', { step: 'push_to_qdrant', state_update: getStateUpdate() });
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

    async function onChatSolutionChange() {
      const sel = document.getElementById('solutionChat');
      const solId = sel.value;
      const collRow = document.getElementById('chatCollectionRow');
      const collSel = document.getElementById('chatCollectionSelect');
      if (!solId) {
        collRow.style.display = 'none';
        collSel.innerHTML = '';
        return;
      }
      collRow.style.display = 'block';
      collSel.innerHTML = '<option value="">Loading…</option>';
      try {
        const res = await fetch('/api/solutions/' + encodeURIComponent(solId) + '/collections').then(r => r.json());
        collSel.innerHTML = '';
        const existing = (res.collections || []).filter(c => c.exists);
        if (existing.length > 1) {
          // "All collections" option for multi-collection solutions
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

    document.getElementById('runQA').onclick = async () => {
      const sel = document.getElementById('solutionChat');
      const o = sel.options[sel.selectedIndex];
      const company = (o && o.value ? o.dataset.company : null) || 'Assistant';
      const solId = sel.value || null;
      const collection = document.getElementById('chatCollectionSelect').value.trim();
      const question = document.getElementById('qaQuestion').value.trim();
      if (!collection || !question) {
        setLog(qaResult, 'Pick a solution + collection and enter a question.', true);
        return;
      }
      setLog(qaResult, '…', false);
      try {
        const body = { collection_name: collection, question, company };
        if (collection === '__all__' && solId) body.solution_id = solId;
        const res = await api('/api/qa', body);
        qaResult.textContent = 'Q: ' + res.question + '\\n\\nA: ' + res.answer;
        qaResult.classList.remove('error');
        qaResult.classList.add('success');
      } catch (e) {
        setLog(qaResult, e.message || String(e), true);
      }
    };
  </script>
  <footer style="text-align:center;padding:1.2rem 0 0.8rem;font-size:0.78rem;color:#aaa;">
    jabberbrain RAG builder &nbsp;·&nbsp; v__APP_VERSION__
  </footer>
</body>
</html>
"""
