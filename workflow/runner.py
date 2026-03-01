"""
Workflow runner: executes one step at a time using existing ingestion, vectorization, and Qdrant logic.
"""
from workflow.models import Step, WorkflowState, ChunkingConfig


def run_step(state: WorkflowState, step: Step) -> str:
    """
    Execute a single workflow step. Mutates state. Returns a short status message.
    Auto-saves state to disk after steps that produce significant output.
    """
    if state.tracker is None and step != Step.ADD_SOURCE:
        return "Error: No Qdrant tracker set on state. Create tracker and set state.tracker first."

    if step == Step.CREATE_COLLECTION:
        msg = _run_create_collection(state)
    elif step == Step.ADD_SOURCE:
        return "Source config set in state (no side effects)."
    elif step == Step.FETCH:
        msg = _run_fetch(state)
    elif step == Step.CLEAN:
        msg = _run_clean(state)
    elif step == Step.TRANSLATE_AND_CLEAN:
        msg = _run_translate_and_clean(state)
    elif step == Step.CHUNK:
        msg = _run_chunk(state)
    elif step == Step.GROUP:
        msg = _run_group(state)
    elif step == Step.PUSH_TO_QDRANT:
        msg = _run_push(state)
    elif step == Step.TEST_QA:
        return _run_test_qa(state)
    else:
        return f"Unknown step: {step}"

    # Track completion and auto-save for steps that produce persistent output
    _SAVEABLE_STEPS = {Step.FETCH, Step.TRANSLATE_AND_CLEAN, Step.CHUNK, Step.PUSH_TO_QDRANT}
    if not msg.startswith("Error") and step in _SAVEABLE_STEPS:
        if step.value not in state.completed_steps:
            state.completed_steps.append(step.value)

        # After chunking: generate collection-level topic/keyword metadata for RAG routing
        if step == Step.CHUNK and state.chunks:
            try:
                from workflow.suggest import suggest_collection_metadata
                label = state.source_label or (state.source_config or {}).get("source_label", "document")
                meta = suggest_collection_metadata(state.chunks, source_label=label)
                if meta:
                    state.collection_metadata = meta
                    topics = ", ".join(meta.get("topics") or [])
                    msg += f"\n📋 Metadata: {meta.get('doc_type', '?')} · {meta.get('language', '?')} · Topics: {topics}"
            except Exception as e:
                print(f"[runner] Metadata generation failed (non-fatal): {e}")

        saved = state.save_to_disk()
        if saved:
            msg += f"\n💾 State saved to: {saved}"

    return msg


def _run_create_collection(state: WorkflowState) -> str:
    if not state.collection_name:
        return "Error: collection_name is required."
    from QdrantTracker import QdrantTracker
    from workflow.models import EMBEDDING_DIMS
    tracker = state.tracker or QdrantTracker()
    state.tracker = tracker
    coll_type = "group" if state.grouping_enabled else "scs"
    vector_size = EMBEDDING_DIMS.get(state.embedding_model, 1536)
    # Delete existing collection silently so we never hit interactive input() prompts
    if tracker._existing_collection_name(state.collection_name):
        tracker._delete_collection(state.collection_name)
    # Create with correct vector size for the chosen embedding model
    tracker._create_collection(state.collection_name, vector_size=vector_size)
    state.collection_object = tracker.new(state.collection_name, coll_type)
    return (
        f"Created and opened collection '{state.collection_name}' "
        f"(type={coll_type}, model={state.embedding_model}, dims={vector_size})."
    )


def _run_fetch(state: WorkflowState) -> str:
    stype = state.source_type or ""
    config = state.source_config or {}

    if stype == "pdf":
        from ingestion.pdf_ingestion import read_from_pdf, read_from_pdf_pages
        path = config.get("path") or config.get("pdf_path")
        if not path:
            return "Error: source_config must contain 'path' or 'pdf_path' for PDF."
        state.raw_text = read_from_pdf(path)
        state.pdf_pages = read_from_pdf_pages(path)  # per-page for source attribution
        state.source_label = config.get("source_label") or path.split("/")[-1]
        return f"Fetched PDF: {len(state.raw_text)} characters ({len(state.pdf_pages)} pages)."

    if stype == "txt":
        from ingestion.txt_ingestion import read_txt_as_text
        path = config.get("path") or config.get("file_path")
        if not path:
            return "Error: source_config must contain 'path' or 'file_path' for txt."
        state.raw_text = read_txt_as_text(path)
        state.source_label = config.get("source_label") or path.split("/")[-1]
        return f"Fetched TXT: {len(state.raw_text)} characters."

    if stype == "url":
        from ingestion.scrapers.runner import run_scraper
        scraper_name = config.get("scraper_name") or config.get("scraper")
        if not scraper_name:
            return "Error: source_config must contain 'scraper_name' or 'scraper' for url."
        inline_cfg = config.get("scraper_config")  # inline config from solutions.yaml
        raw_text, scraped_items = run_scraper(scraper_name, config, inline_config=inline_cfg)
        state.raw_text = raw_text
        state.scraped_items = scraped_items
        state.source_label = config.get("source_label") or scraper_name

        # ── Relevance check (runs if collection has routing metadata in solutions.yaml) ──
        routing = _get_collection_routing(state)
        if routing and scraped_items:
            # Reuse the SSE progress queue from the web app if running in that context
            try:
                from web.app import _progress_queue as _pq
            except Exception:
                _pq = None

            def _log(msg: str) -> None:
                # When _pq is available, put directly (avoids double-logging via _QueueWriter)
                # Otherwise, fall through to print() which _QueueWriter intercepts
                if _pq:
                    _pq.put(f"LOG:{msg}")
                else:
                    print(f"[relevance] {msg}")

            _log(f"🔍 Running relevance check on {len(scraped_items)} pages…")

            from workflow.relevance import filter_scraped_items
            relevant, mismatch, irrelevant = filter_scraped_items(
                scraped_items, routing, progress_cb=_log
            )

            if irrelevant:
                _push_to_not_relevant(state, irrelevant, _log)

            state.relevance_report = {
                "relevant_count":   len(relevant),
                "mismatch_count":   len(mismatch),
                "irrelevant_count": len(irrelevant),
                "mismatch_urls":    [it["url"] for it in mismatch],
                "irrelevant_urls":  [it["url"] for it in irrelevant],
            }
            # Flagged (mismatch) pages still go into the intended collection
            state.scraped_items = relevant + mismatch
            # Rebuild raw_text to match filtered items
            state.raw_text = "\n\n".join(it["text"] for it in state.scraped_items)

            summary = (
                f"✅ {len(relevant)} relevant · "
                f"⚠️ {len(mismatch)} flagged · "
                f"❌ {len(irrelevant)} irrelevant (→ 'not_relevant')"
            )
            _log(summary)
            items_info = f", {len(state.scraped_items)} pages after relevance check"
        else:
            items_info = f", {len(scraped_items)} pages with URL metadata" if scraped_items else ""

        return f"Fetched URL (scraper={scraper_name}): {len(state.raw_text)} characters{items_info}."

    if stype == "csv":
        from ingestion.csv_ingestion import read_csv_to_chunks
        path = config.get("path") or config.get("csv_path")
        if not path:
            return "Error: source_config must contain 'path' or 'csv_path' for csv."
        chunks = read_csv_to_chunks(path, config)
        state.chunks = chunks
        state.source_label = config.get("source_label") or path.split("/")[-1]
        return f"Fetched CSV: {len(chunks)} chunks (no CHUNK step needed)."

    return f"Error: Unknown or missing source_type: {stype}"


def _run_clean(state: WorkflowState) -> str:
    if not state.raw_text:
        state.cleaned_text = ""
        return "No raw text to clean."
    filter_name = (state.source_config or {}).get("filter_name")
    if filter_name:
        from ingestion.scrapers.filters import apply_filter
        state.cleaned_text = apply_filter(state.raw_text, filter_name)
    else:
        state.cleaned_text = state.raw_text
    return f"Cleaned text: {len(state.cleaned_text)} characters."


def _run_translate_and_clean(state: WorkflowState) -> str:
    """
    Takes raw_text (bilingual PT/ES from PDF extraction), strips the Spanish,
    and reconstructs each recipe cleanly in Portuguese.
    Uses GPT-4o-mini in batches of ~4000 chars to keep costs low.
    Result is stored in state.cleaned_text.
    """
    text = state.raw_text or ""
    if not text:
        return "Error: No raw text found. Run FETCH first."

    from llms.openai_utils import openai_chat_completion

    BATCH_SIZE = 4000
    OVERLAP = 200

    system_prompt = (
        "És um assistente especializado em culinária portuguesa. "
        "Vais receber texto extraído de um PDF bilingue (português e espanhol) de receitas de peixe. "
        "O texto está misturado — português e espanhol aparecem juntos porque o PDF tinha duas colunas. "
        "A tua tarefa é: "
        "1. Remover todo o texto em espanhol. "
        "2. Manter apenas o texto em português. "
        "3. Reconstruir cada receita com a estrutura: Nome da receita → Ingredientes → Confeção. "
        "4. Não traduzir nada — apenas filtrar e organizar o português já existente. "
        "5. Devolver apenas o texto limpo em português, sem comentários ou explicações."
    )

    # Split into overlapping batches
    batches = []
    start = 0
    while start < len(text):
        end = min(start + BATCH_SIZE, len(text))
        batches.append(text[start:end])
        if end == len(text):
            break
        start += BATCH_SIZE - OVERLAP

    # Try to get the progress queue from the web app (optional – works without it too)
    try:
        from web.app import _progress_queue as _pq
    except Exception:
        _pq = None

    def _report(msg: str):
        if _pq:
            _pq.put(msg)

    cleaned_parts = []
    total = len(batches)
    for i, batch in enumerate(batches):
        result = openai_chat_completion(
            prompt=system_prompt,
            text=batch,
            model="gpt-4o-mini"
        )
        cleaned_parts.append(result.strip())
        _report(f"PROGRESS:{i+1}/{total}")

    state.cleaned_text = "\n\n".join(cleaned_parts)
    final_msg = (
        f"Translated & cleaned: {total} batch(es) processed. "
        f"Result: {len(state.cleaned_text)} characters of Portuguese text."
    )
    _report(f"DONE:{final_msg}")
    return final_msg


def _run_chunk(state: WorkflowState) -> str:
    from workflow.models import ChunkingConfig
    # Ensure chunking_config is a ChunkingConfig object, not a plain dict
    if isinstance(state.chunking_config, dict):
        cfg_dict = state.chunking_config
        state.chunking_config = ChunkingConfig(
            batch_size=cfg_dict.get("batch_size", 10000),
            overlap_size=cfg_dict.get("overlap_size", 100),
            use_proposition_chunking=cfg_dict.get("use_proposition_chunking", False),
            simple_chunk_size=cfg_dict.get("simple_chunk_size", 1000),
            simple_chunk_overlap=cfg_dict.get("simple_chunk_overlap", 200),
            use_hierarchical_chunking=cfg_dict.get("use_hierarchical_chunking", False),
            hierarchical_parent_size=cfg_dict.get("hierarchical_parent_size", 2000),
            hierarchical_parent_overlap=cfg_dict.get("hierarchical_parent_overlap", 200),
            hierarchical_child_size=cfg_dict.get("hierarchical_child_size", 400),
            hierarchical_child_overlap=cfg_dict.get("hierarchical_child_overlap", 50),
        )
    cfg = state.chunking_config
    text = state.get_text_for_chunking()

    if state.chunks and state.source_type == "csv":
        return f"Chunks already set from CSV: {len(state.chunks)} (skipping CHUNK)."

    if not text:
        return "Error: No text to chunk. Run FETCH (and optionally CLEAN) first."

    # Fast path for URL/scraper sources: scraped_items already has exactly 1 entry per page,
    # each pre-rendered by the chunk_template. Use them directly — no splitting needed.
    if state.scraped_items:
        state.chunks = [item["text"] for item in state.scraped_items]
        return f"Chunked into {len(state.chunks)} chunks (1 per scraped page — no splitting needed)."

    if cfg.use_hierarchical_chunking:
        msg = _run_chunk_hierarchical(state, cfg, text)
    elif cfg.use_proposition_chunking:
        from vectorization import get_text_chunks, create_batches_of_text
        batches = create_batches_of_text(text, cfg.batch_size, cfg.overlap_size)
        chunks = []
        for batch in batches:
            chunks.extend(get_text_chunks(batch))
        state.chunks = chunks
        msg = f"Chunked into {len(state.chunks)} chunks."
    else:
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=cfg.simple_chunk_size,
            chunk_overlap=cfg.simple_chunk_overlap,
            length_function=len,
        )
        state.chunks = splitter.split_text(text)
        msg = f"Chunked into {len(state.chunks)} chunks."

    # For PDF sources: map each chunk to its source page and populate scraped_items
    # with pdf:// URLs for source attribution in the chat UI.
    if state.source_type == "pdf" and state.chunks:
        _attach_pdf_page_urls(state)
        msg += f" (page attribution: {len(state.scraped_items)} items with pdf:// URLs)"

    return msg


def _find_page_for_chunk(chunk: str, pdf_pages: list) -> int:
    """
    Find which PDF page (1-based) a chunk most likely came from.
    Strategy: check if the first 200 chars of the chunk appear verbatim in a page's text.
    Fallback: count common words between the sample and each page text, pick best match.
    Returns 1 if no match found.
    """
    sample = chunk[:200]
    best_page, best_score = 1, 0
    for entry in pdf_pages:
        page_text = entry.get("text", "")
        if sample in page_text:
            return entry["page"]
        # Word-overlap fallback
        score = sum(1 for w in sample.split() if w in page_text)
        if score > best_score:
            best_score = score
            best_page = entry["page"]
    return best_page


def _attach_pdf_page_urls(state: WorkflowState) -> None:
    """
    After chunking a PDF source, map each chunk to its source page and populate
    state.scraped_items with {text, url} entries using the pdf:// URI scheme.
    This is called after any PDF chunking path (simple, hierarchical, proposition).
    """
    if not state.pdf_pages:
        return  # no per-page data available, skip

    source_label = state.source_label or "document"
    # Strip file extension for use in the pdf:// URI
    import os
    base_name = os.path.splitext(source_label)[0]

    total_chunks = len(state.chunks)
    total_pages = len(state.pdf_pages)

    items = []
    for idx, chunk in enumerate(state.chunks):
        # Try exact / word-overlap match first
        page_num = _find_page_for_chunk(chunk, state.pdf_pages)
        # If the chunk came from cleaned/translated text it won't match raw pages →
        # the score will be 0 (or very low). Detect that and fall back to proportional.
        sample = chunk[:200]
        best_raw_score = max(
            sum(1 for w in sample.split() if w in entry.get("text", ""))
            for entry in state.pdf_pages
        ) if state.pdf_pages else 0

        if best_raw_score < 3 and total_pages > 0:
            # Proportional fallback for translated text
            page_num = max(1, round((idx / max(total_chunks - 1, 1)) * (total_pages - 1)) + 1)
            page_num = min(page_num, total_pages)

        url = f"pdf://{base_name}#page={page_num}"
        items.append({"text": chunk, "url": url})

    state.scraped_items = items


def _run_chunk_hierarchical(state: WorkflowState, cfg, text: str) -> str:
    """
    Hierarchical chunking: split into large parent chunks, then split each parent
    into small child chunks. Each stored chunk = 'Context: {parent}\\n\\nPassage: {child}'
    so the embedding captures the fine-grained child passage but retrieval context
    includes the full parent section. This gives much better recall on structured
    documents (e.g. recipe books, technical docs) — completely free, no LLM needed.
    """
    from langchain_text_splitters import CharacterTextSplitter

    parent_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=cfg.hierarchical_parent_size,
        chunk_overlap=cfg.hierarchical_parent_overlap,
        length_function=len,
    )
    child_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=cfg.hierarchical_child_size,
        chunk_overlap=cfg.hierarchical_child_overlap,
        length_function=len,
    )

    parents = parent_splitter.split_text(text)
    chunks = []
    for parent in parents:
        children = child_splitter.split_text(parent)
        for child in children:
            # Embed parent context alongside the child passage
            chunk = f"Context:\n{parent}\n\nPassage:\n{child}"
            chunks.append(chunk)

    state.chunks = chunks
    n_parents = len(parents)
    n_children = len(chunks)
    return (
        f"Hierarchical chunking: {n_parents} parent sections → "
        f"{n_children} child chunks (parent_size={cfg.hierarchical_parent_size}, "
        f"child_size={cfg.hierarchical_child_size})."
    )


def _run_group(state: WorkflowState) -> str:
    if not state.grouping_enabled or not state.chunks:
        return "Grouping skipped (disabled or no chunks)."
    from my_collections.SCS_Collection import SCS_Collection
    from my_collections.groupCollection import GroupCollection
    from grouping import grouping as run_grouping

    scs = SCS_Collection(state.collection_name + "_scs_temp")
    scs.append_sentences(state.chunks, state.source_label)
    gc = state.collection_object  # already a GroupCollection from CREATE_COLLECTION
    run_grouping(scs, gc)
    return "Grouping completed."


def _run_push(state: WorkflowState) -> str:
    tracker = state.tracker
    name = state.collection_name
    coll = state.collection_object

    if not name:
        return "Error: collection_name is required."

    if state.grouping_enabled:
        # collection_object not persisted to disk — recreate if needed
        if coll is None:
            coll = tracker.new(name, "group")
            state.collection_object = coll
        tracker.save_collection(name)
        return f"Saved group collection '{name}' to Qdrant."

    # collection_object is a live Python object — NOT persisted to .rag_state.json.
    # When loaded from disk, coll is None. Two cases:
    #   A) Collection already exists in Qdrant → append mode (don't wipe existing points)
    #   B) Collection doesn't exist yet → fresh push (original behaviour)
    embedding_model = state.embedding_model or "text-embedding-ada-002"

    # ── Push guard: filter out skip_urls (manually edited chunks preserved) ──
    chunks_to_push = state.chunks
    items_to_push = state.scraped_items or []
    skip_urls = state.skip_urls or []
    skip_msg = ""
    if skip_urls and items_to_push:
        skip_set = set(skip_urls)
        before = len(items_to_push)
        items_to_push = [it for it in items_to_push if it.get("url") not in skip_set]
        chunks_to_push = [it["text"] for it in items_to_push]
        skipped = before - len(items_to_push)
        if skipped:
            skip_msg = f" (skipped {skipped} URL(s) with manually edited chunks)"
            print(f"[push] Skipping {skipped} URLs with manually edited chunks: {skip_urls}")

    if coll is None and tracker._existing_collection_name(name):
        # Append mode: build a temp SCS_Collection, embed, upsert without deleting.
        from my_collections.SCS_Collection import SCS_Collection
        temp_coll = SCS_Collection(name)
        temp_coll.append_sentences(chunks_to_push, state.source_label,
                                   scraped_items=items_to_push)
        points = temp_coll.points_to_save(model_id=embedding_model)
        tracker.append_points_to_collection(name, points)
        state.skip_urls = []  # clear after push
        return f"Appended {len(points)} chunks to existing '{name}' in Qdrant (model={embedding_model}).{skip_msg}"

    # Fresh push: collection doesn't exist yet — create + save all at once.
    if coll is None:
        coll = tracker.new(name, "scs")
        state.collection_object = coll

    coll.append_sentences(chunks_to_push, state.source_label,
                          scraped_items=items_to_push)
    # Use save_collection for the first push; it calls points_to_save() internally.
    # We need to pass model_id through — override the collection's embedding call.
    # save_collection → _upsert_points(coll.points_to_save()) so patch it here:
    points = coll.points_to_save(model_id=embedding_model)
    tracker._upsert_points(name, points)
    state.skip_urls = []  # clear after push
    return f"Pushed {len(points)} chunks to '{name}' in Qdrant (model={embedding_model}).{skip_msg}"


def _get_collection_routing(state: WorkflowState) -> dict | None:
    """
    Look up the routing block for the current collection in solutions.yaml.
    Returns the routing dict if it has a description or keywords, otherwise None.
    """
    collection_name = state.collection_name
    if not collection_name:
        return None
    try:
        from solution_specs.loader import get_all_solutions
        for sol in get_all_solutions():
            for coll in sol.get("collections", []):
                if (
                    coll.get("collection_name") == collection_name
                    or coll.get("id") == collection_name
                ):
                    routing = coll.get("routing") or {}
                    if routing.get("description") or routing.get("keywords"):
                        return routing
    except Exception as e:
        print(f"[runner] _get_collection_routing failed (non-fatal): {e}")
    return None


def _push_to_not_relevant(
    state: WorkflowState,
    items: list[dict],
    log_fn,
) -> None:
    """
    Push irrelevant pages to a dedicated 'not_relevant' Qdrant collection.
    Creates the collection if it doesn't exist. Non-fatal — errors are logged.
    """
    try:
        from QdrantTracker import QdrantTracker
        from my_collections.SCS_Collection import SCS_Collection
        from workflow.models import EMBEDDING_DIMS

        tracker = state.tracker or QdrantTracker()
        coll_name = "not_relevant"
        embedding_model = state.embedding_model or "text-embedding-ada-002"
        vector_size = EMBEDDING_DIMS.get(embedding_model, 1536)

        if not tracker._existing_collection_name(coll_name):
            tracker._create_collection(coll_name, vector_size=vector_size)
            log_fn(f"  Created '{coll_name}' Qdrant collection.")

        # Tag source so items are traceable back to their original collection
        source_label = f"not_relevant:{state.collection_name or 'unknown'}"
        temp_coll = SCS_Collection(coll_name)
        temp_coll.append_sentences(
            [it["text"] for it in items],
            source_label,
            scraped_items=items,
        )
        points = temp_coll.points_to_save(model_id=embedding_model)
        tracker.append_points_to_collection(coll_name, points)
        log_fn(f"  → Pushed {len(points)} irrelevant page(s) to '{coll_name}'.")
    except Exception as e:
        log_fn(f"  ⚠ Could not push to 'not_relevant' collection: {e}")


def _run_test_qa(state: WorkflowState) -> str:
    from chatbot import get_retrieved_info, get_answer
    question = (state.source_config or {}).get("test_question") or "What is this content about?"
    history = []
    collection_name = state.collection_name
    company = (state.source_config or {}).get("company_name") or "the assistant"
    retrieved = get_retrieved_info(question, history, collection_name)
    answer = get_answer(history, retrieved, question, company)
    return f"Q: {question}\nA: {answer}"
