"""
Minimal FastAPI app for the RAG workflow. Run with: uvicorn web.app:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Any

# In-memory state for single-user demo (one workflow at a time)
_current_state: Optional[Any] = None
_tracker: Optional[Any] = None


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


class StepRequest(BaseModel):
    step: str
    state_update: Optional[dict] = None


class QARequest(BaseModel):
    collection_name: str
    question: str
    company: str = "Assistant"


@app.get("/")
def root():
    return HTMLResponse(_INDEX_HTML)


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


@app.post("/api/workflow/step")
def run_workflow_step(req: StepRequest):
    from workflow.models import WorkflowState, Step
    from workflow.runner import run_step
    state = get_state()
    if req.state_update:
        for k, v in req.state_update.items():
            if k == "source_config" and isinstance(v, dict):
                state.source_config = v
            elif k == "chunking_config" and isinstance(v, dict):
                state.chunking_config.batch_size = v.get("batch_size", state.chunking_config.batch_size)
                state.chunking_config.overlap_size = v.get("overlap_size", state.chunking_config.overlap_size)
                state.chunking_config.use_proposition_chunking = v.get("use_proposition_chunking", state.chunking_config.use_proposition_chunking)
            elif hasattr(state, k):
                setattr(state, k, v)
    try:
        step = Step(req.step)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown step: {req.step}")
    msg = run_step(state, step)
    return {"message": msg, "state": state.to_dict()}


@app.post("/api/qa")
def qa(req: QARequest):
    from chatbot import get_retrieved_info, get_answer
    history = []
    retrieved = get_retrieved_info(req.question, history, req.collection_name)
    answer = get_answer(history, retrieved, req.question, req.company)
    return {"question": req.question, "answer": answer}


@app.post("/api/workflow/reset")
def workflow_reset():
    reset_state()
    return {"message": "State reset."}


class SuggestChunkingRequest(BaseModel):
    text_preview: str
    source_type: str = "unknown"


@app.post("/api/suggest/chunking")
def api_suggest_chunking(req: SuggestChunkingRequest):
    from workflow.suggest import suggest_chunking
    return suggest_chunking(req.text_preview, req.source_type)


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
        <h2>1. Choose or create a solution</h2>
        <label>Solution (optional – pick to prefill)</label>
        <select id="solutionBuild">
          <option value="">— New (type name below) —</option>
        </select>
        <label>Collection name</label>
        <input type="text" id="collectionName" placeholder="e.g. my_rag or use solution above">
      </div>
      <div class="card">
        <h2>2. Source</h2>
        <label>Source type</label>
        <select id="sourceType">
          <option value="pdf">PDF file</option>
          <option value="txt">Text file</option>
          <option value="url">Website (scraper)</option>
          <option value="csv">CSV file</option>
        </select>
        <label>Path or scraper name</label>
        <input type="text" id="sourcePath" placeholder="e.g. ingestion/data_to_ingest/pdfs/doc.pdf or peixefresco">
      </div>
      <div class="card">
        <h2>3. Run pipeline</h2>
        <p class="status">Create collection → Fetch → Chunk → Push to Qdrant</p>
        <button type="button" class="btn-primary" id="runCreate">1. Create collection</button>
        <button type="button" class="btn-primary" id="runFetch">2. Fetch</button>
        <button type="button" class="btn-primary" id="runChunk">3. Chunk</button>
        <button type="button" class="btn-primary" id="runPush">4. Push to Qdrant</button>
        <button type="button" class="btn-secondary" id="runReset">Reset state</button>
        <div id="buildLog" class="log"></div>
      </div>
    </div>

    <div id="panel-chat" class="panel hidden">
      <div class="card">
        <h2>Chat with a solution</h2>
        <label>Solution</label>
        <select id="solutionChat">
          <option value="">— Pick a solution —</option>
        </select>
        <label>Or type collection name (if not using a solution)</label>
        <input type="text" id="qaCollectionOverride" placeholder="e.g. my_collection">
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
    (async () => {
      const { solutions } = await api('/api/solutions');
      const selBuild = document.getElementById('solutionBuild');
      const selChat = document.getElementById('solutionChat');
      solutions.forEach(s => {
        const opt = (parent) => {
          const o = document.createElement('option');
          o.value = s.id;
          o.textContent = s.display_name || s.id;
          o.dataset.collection = s.collection_name || '';
          o.dataset.company = s.company_name || '';
          o.dataset.scraper = s.scraper_name || '';
          parent.appendChild(o);
        };
        opt(selBuild);
        opt(selChat);
      });
    })();

    // Tabs
    document.querySelectorAll('.tab').forEach(t => {
      t.onclick = () => {
        document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(x => x.classList.add('hidden'));
        t.classList.add('active');
        document.getElementById('panel-' + t.dataset.tab).classList.remove('hidden');
      };
    });

    // Build: solution selected → prefill collection (and scraper for url)
    document.getElementById('solutionBuild').onchange = function() {
      const o = this.options[this.selectedIndex];
      if (o && o.value) {
        document.getElementById('collectionName').value = o.dataset.collection || '';
        if (o.dataset.scraper) {
          document.getElementById('sourceType').value = 'url';
          document.getElementById('sourcePath').value = o.dataset.scraper;
        }
      }
    };

    const getStateUpdate = () => {
      const sol = document.getElementById('solutionBuild').options[document.getElementById('solutionBuild').selectedIndex];
      const path = document.getElementById('sourcePath').value.trim();
      const st = document.getElementById('sourceType').value;
      let source_config = {};
      if (st === 'url') source_config = { scraper_name: path || 'peixefresco', source_label: path || 'peixefresco' };
      else if (path) source_config = { path, source_label: path.split('/').pop() };
      return {
        collection_name: document.getElementById('collectionName').value.trim() || (sol && sol.value ? sol.dataset.collection : undefined),
        source_type: st,
        source_config
      };
    };

    const runStep = async (step) => {
      setLog(buildLog, 'Running…', false);
      try {
        const res = await api('/api/workflow/step', { step, state_update: getStateUpdate() });
        setLog(buildLog, res.message, res.message.indexOf('Error') !== -1);
      } catch (e) {
        setLog(buildLog, e.message || String(e), true);
      }
    };

    document.getElementById('runCreate').onclick = () => runStep('create_collection');
    document.getElementById('runFetch').onclick = () => runStep('fetch');
    document.getElementById('runChunk').onclick = () => runStep('chunk');
    document.getElementById('runPush').onclick = () => runStep('push_to_qdrant');
    document.getElementById('runReset').onclick = async () => {
      await api('/api/workflow/reset', {});
      setLog(buildLog, 'State reset.', false);
    };

    document.getElementById('runQA').onclick = async () => {
      const sel = document.getElementById('solutionChat');
      const o = sel.options[sel.selectedIndex];
      const override = document.getElementById('qaCollectionOverride').value.trim();
      const collection = override || (o && o.value ? o.dataset.collection : '');
      const company = (o && o.value ? o.dataset.company : null) || 'Assistant';
      const question = document.getElementById('qaQuestion').value.trim();
      if (!collection || !question) {
        setLog(qaResult, 'Pick a solution and enter a question.', true);
        return;
      }
      setLog(qaResult, '…', false);
      try {
        const res = await api('/api/qa', { collection_name: collection, question, company });
        qaResult.textContent = 'Q: ' + res.question + '\\n\\nA: ' + res.answer;
        qaResult.classList.remove('error');
        qaResult.classList.add('success');
      } catch (e) {
        setLog(qaResult, e.message || String(e), true);
      }
    };
  </script>
</body>
</html>
"""
