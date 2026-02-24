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
    _SAVEABLE_STEPS = {Step.FETCH, Step.TRANSLATE_AND_CLEAN, Step.CHUNK}
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
    tracker = state.tracker or QdrantTracker()
    state.tracker = tracker
    coll_type = "group" if state.grouping_enabled else "scs"
    # Delete existing collection silently so we never hit interactive input() prompts
    if tracker._existing_collection_name(state.collection_name):
        tracker._delete_collection(state.collection_name)
    state.collection_object = tracker.new(state.collection_name, coll_type)
    return f"Created and opened collection '{state.collection_name}' (type={coll_type})."


def _run_fetch(state: WorkflowState) -> str:
    stype = state.source_type or ""
    config = state.source_config or {}

    if stype == "pdf":
        from ingestion.pdf_ingestion import read_from_pdf
        path = config.get("path") or config.get("pdf_path")
        if not path:
            return "Error: source_config must contain 'path' or 'pdf_path' for PDF."
        state.raw_text = read_from_pdf(path)
        state.source_label = config.get("source_label") or path.split("/")[-1]
        return f"Fetched PDF: {len(state.raw_text)} characters."

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
        state.raw_text = run_scraper(scraper_name, config)
        state.source_label = config.get("source_label") or scraper_name
        return f"Fetched URL (scraper={scraper_name}): {len(state.raw_text)} characters."

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

    if cfg.use_hierarchical_chunking:
        return _run_chunk_hierarchical(state, cfg, text)
    elif cfg.use_proposition_chunking:
        from vectorization import get_text_chunks, create_batches_of_text
        batches = create_batches_of_text(text, cfg.batch_size, cfg.overlap_size)
        chunks = []
        for batch in batches:
            chunks.extend(get_text_chunks(batch))
        state.chunks = chunks
    else:
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=cfg.simple_chunk_size,
            chunk_overlap=cfg.simple_chunk_overlap,
            length_function=len,
        )
        state.chunks = splitter.split_text(text)

    return f"Chunked into {len(state.chunks)} chunks."


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

    if state.grouping_enabled:
        tracker.save_collection(name)
        return f"Saved group collection '{name}' to Qdrant."
    # SCS path: append chunks to collection then save
    coll.append_sentences(state.chunks, state.source_label)
    tracker.save_collection(name)
    return f"Appended {len(state.chunks)} chunks and saved '{name}' to Qdrant."


def _run_test_qa(state: WorkflowState) -> str:
    from chatbot import get_retrieved_info, get_answer
    question = (state.source_config or {}).get("test_question") or "What is this content about?"
    history = []
    collection_name = state.collection_name
    company = (state.source_config or {}).get("company_name") or "the assistant"
    retrieved = get_retrieved_info(question, history, collection_name)
    answer = get_answer(history, retrieved, question, company)
    return f"Q: {question}\nA: {answer}"
