"""
Workflow runner: executes one step at a time using existing ingestion, vectorization, and Qdrant logic.
"""
from workflow.models import Step, WorkflowState, ChunkingConfig


def run_step(state: WorkflowState, step: Step) -> str:
    """
    Execute a single workflow step. Mutates state. Returns a short status message.
    """
    if state.tracker is None and step != Step.ADD_SOURCE:
        return "Error: No Qdrant tracker set on state. Create tracker and set state.tracker first."

    if step == Step.CREATE_COLLECTION:
        return _run_create_collection(state)
    if step == Step.ADD_SOURCE:
        return "Source config set in state (no side effects)."
    if step == Step.FETCH:
        return _run_fetch(state)
    if step == Step.CLEAN:
        return _run_clean(state)
    if step == Step.CHUNK:
        return _run_chunk(state)
    if step == Step.GROUP:
        return _run_group(state)
    if step == Step.PUSH_TO_QDRANT:
        return _run_push(state)
    if step == Step.TEST_QA:
        return _run_test_qa(state)
    return f"Unknown step: {step}"


def _run_create_collection(state: WorkflowState) -> str:
    if not state.collection_name:
        return "Error: collection_name is required."
    from QdrantTracker import QdrantTracker
    tracker = state.tracker or QdrantTracker()
    state.tracker = tracker
    coll_type = "group" if state.grouping_enabled else "scs"
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


def _run_chunk(state: WorkflowState) -> str:
    cfg = state.chunking_config
    text = state.get_text_for_chunking()

    if state.chunks and state.source_type == "csv":
        return f"Chunks already set from CSV: {len(state.chunks)} (skipping CHUNK)."

    if not text:
        return "Error: No text to chunk. Run FETCH (and optionally CLEAN) first."

    if cfg.use_proposition_chunking:
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
