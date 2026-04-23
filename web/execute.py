"""
Stateless execution endpoints for the RAG pipeline.

These endpoints receive all configuration via the request body,
execute one pipeline step, and return the result. No internal state
is read or written — the caller (jBKB Fastify) owns all state.

Mount on the main FastAPI app:
    from web.execute import router as execute_router
    app.include_router(execute_router, prefix="/api/execute")
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Any
import asyncio
import queue
import threading
import json

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════

class ScraperConfig(BaseModel):
    """Scraper configuration passed from jBKB."""
    name: str = ""
    engine: str = "playwright"  # playwright | httpx
    scrape_mode: str = "sitemap"  # sitemap | links | single
    sitemap_url: Optional[str] = None
    start_url: Optional[str] = None
    excluded_urls: list[str] = Field(default_factory=list)
    text_selector: str = "main, article, .entry-content, body"
    # CSS selectors for elements to REMOVE before extracting text. Per-sitemap
    # user config (cookie banners, newsletter popups, footers). A baseline of
    # script/style/noscript/iframe is always stripped in addition.
    exclude_selectors: list[str] = Field(default_factory=list)
    # Extra config (custom JS extractors, click selectors, etc.)
    extra: dict = Field(default_factory=dict)


class FetchRequest(BaseModel):
    """Request to fetch/scrape content from a source."""
    source_type: str  # url | pdf | txt | csv
    # For URL sources:
    scraper_config: Optional[ScraperConfig] = None
    source_url: Optional[str] = None  # alternative to scraper_config for simple single-URL fetch
    # For file sources:
    file_path: Optional[str] = None
    # Optional: routing metadata for relevance filtering
    routing: Optional[dict] = None
    # Collection context (for labeling)
    collection_name: Optional[str] = None
    source_label: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"


class FetchResponse(BaseModel):
    """Result of a fetch step."""
    raw_text: str
    scraped_items: list[dict] = Field(default_factory=list)  # [{url, text}, ...]
    pdf_pages: list[dict] = Field(default_factory=list)  # [{page, text}, ...]
    source_label: str = ""
    char_count: int = 0
    page_count: int = 0
    relevance_report: Optional[dict] = None


class ChunkingConfig(BaseModel):
    """Chunking parameters."""
    mode: str = "hierarchical"  # simple | hierarchical | proposition
    size: int = 2000  # chunk size (simple) or parent size (hierarchical)
    overlap: int = 200
    child_size: int = 400  # hierarchical only
    child_overlap: int = 50  # hierarchical only


class ChunkRequest(BaseModel):
    """Request to chunk text into pieces."""
    text: Optional[str] = None  # raw or cleaned text to chunk
    scraped_items: list[dict] = Field(default_factory=list)  # alternative: pre-scraped pages
    pdf_pages: list[dict] = Field(default_factory=list)  # for PDF page attribution
    source_type: str = "url"
    source_label: str = "document"
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)


class ChunkResponse(BaseModel):
    """Result of a chunk step."""
    chunks: list[str]
    scraped_items: list[dict] = Field(default_factory=list)  # with URL attribution
    chunk_count: int = 0
    metadata: Optional[dict] = None  # auto-generated collection metadata


class PushRequest(BaseModel):
    """Request to embed and push chunks to Qdrant."""
    collection_name: str
    chunks: list[str]
    scraped_items: list[dict] = Field(default_factory=list)  # [{url, text}, ...] for source attribution
    source_label: str = "document"
    embedding_model: str = "text-embedding-ada-002"
    # Push behavior:
    recreate_collection: bool = False  # True = delete + create fresh; False = append
    skip_urls: list[str] = Field(default_factory=list)  # URLs to skip (manually edited)
    excluded_urls: list[str] = Field(default_factory=list)  # URLs permanently excluded


class PushResponse(BaseModel):
    """Result of a push step."""
    points_pushed: int
    collection_name: str
    message: str = ""


class DeletePointsRequest(BaseModel):
    """Request to delete all points for specific source URLs from a Qdrant collection."""
    collection_name: str
    urls: list[str] = Field(default_factory=list)


class DeletePointsResponse(BaseModel):
    """Result of a delete-points call."""
    collection_name: str
    deleted_urls: list[str] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)  # [{url, error}]
    collection_existed: bool = True


class QARequest(BaseModel):
    """Ask a question against a Qdrant collection."""
    collection_name: str
    question: str
    company_name: str = "the assistant"
    embedding_model: str = "text-embedding-ada-002"


class QAResponse(BaseModel):
    """Answer from RAG Q&A."""
    question: str
    answer: str
    sources: list[dict] = Field(default_factory=list)


class CollectionInfoResponse(BaseModel):
    """Qdrant collection health info."""
    collection_name: str
    exists: bool
    points_count: int = 0
    vector_size: int = 0
    status: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════

@router.post("/fetch", response_model=FetchResponse)
def execute_fetch(req: FetchRequest):
    """Fetch/scrape content from a source. Returns raw text and metadata."""
    from workflow.models import WorkflowState

    # Build a minimal WorkflowState to reuse existing runner logic
    state = WorkflowState()
    state.collection_name = req.collection_name or "temp"
    state.source_type = req.source_type
    state.embedding_model = req.embedding_model

    # Build source_config from request
    source_config: dict[str, Any] = {}
    if req.source_type == "url":
        if req.scraper_config:
            sc = req.scraper_config
            # Use the scraper name if provided, otherwise generate one from the sitemap URL
            scraper_name = sc.name or f"sitemap_{hash(sc.sitemap_url or sc.start_url or 'default') & 0xFFFF:04x}"
            source_config["scraper_name"] = scraper_name
            source_config["scraper_config"] = {
                "name": scraper_name,
                "engine": sc.engine,
                "scrape_mode": sc.scrape_mode,
                "sitemap_url": sc.sitemap_url,
                "start_url": sc.start_url,
                "excluded_urls": sc.excluded_urls,
                "text_selector": sc.text_selector,
                "exclude_selectors": sc.exclude_selectors,
                **(sc.extra or {}),
            }
        elif req.source_url:
            source_config["scraper_name"] = "single_url"
            source_config["scraper_config"] = {
                "name": "single_url",
                "engine": "httpx",
                "scrape_mode": "single",
                "start_url": req.source_url,
            }
        else:
            raise HTTPException(400, "URL source requires scraper_config or source_url")
    elif req.source_type in ("pdf", "txt"):
        if not req.file_path:
            raise HTTPException(400, f"{req.source_type} source requires file_path")
        source_config["path"] = req.file_path
    elif req.source_type == "csv":
        if not req.file_path:
            raise HTTPException(400, "CSV source requires file_path")
        source_config["path"] = req.file_path
    else:
        raise HTTPException(400, f"Unknown source_type: {req.source_type}")

    if req.source_label:
        source_config["source_label"] = req.source_label

    state.source_config = source_config

    # Routing metadata supplied by jBKB — used by the fetch-time relevance filter.
    # When present, runner._get_collection_routing prefers this over solutions.yaml.
    state.routing = req.routing

    # Run fetch
    from workflow.runner import _run_fetch
    msg = _run_fetch(state)

    if msg.startswith("Error"):
        raise HTTPException(400, msg)

    return FetchResponse(
        raw_text=state.raw_text or "",
        scraped_items=state.scraped_items or [],
        pdf_pages=state.pdf_pages or [],
        source_label=state.source_label or "",
        char_count=len(state.raw_text or ""),
        page_count=len(state.scraped_items) or len(state.pdf_pages),
        relevance_report=state.relevance_report,
    )


@router.post("/chunk", response_model=ChunkResponse)
def execute_chunk(req: ChunkRequest):
    """Chunk text into pieces using the specified strategy."""
    from workflow.models import WorkflowState, ChunkingConfig as WfChunkingConfig

    state = WorkflowState()
    state.source_type = req.source_type
    state.source_label = req.source_label

    # Set text or scraped items
    if req.scraped_items:
        state.scraped_items = req.scraped_items
        state.raw_text = "\n\n".join(it.get("text", "") for it in req.scraped_items)
    elif req.text:
        state.raw_text = req.text
        state.cleaned_text = req.text  # use as-is
    else:
        raise HTTPException(400, "Either 'text' or 'scraped_items' is required")

    if req.pdf_pages:
        state.pdf_pages = req.pdf_pages

    # Map chunking config
    cfg = req.chunking_config
    state.chunking_config = WfChunkingConfig(
        use_hierarchical_chunking=(cfg.mode == "hierarchical"),
        use_proposition_chunking=(cfg.mode == "proposition"),
        simple_chunk_size=cfg.size if cfg.mode == "simple" else 1000,
        simple_chunk_overlap=cfg.overlap if cfg.mode == "simple" else 200,
        hierarchical_parent_size=cfg.size,
        hierarchical_parent_overlap=cfg.overlap,
        hierarchical_child_size=cfg.child_size,
        hierarchical_child_overlap=cfg.child_overlap,
    )

    # Run chunk
    from workflow.runner import _run_chunk
    msg = _run_chunk(state)

    if msg.startswith("Error"):
        raise HTTPException(400, msg)

    # Generate metadata from chunks (non-fatal)
    metadata = None
    if state.chunks:
        try:
            from workflow.suggest import suggest_collection_metadata
            metadata = suggest_collection_metadata(state.chunks, source_label=req.source_label)
        except Exception:
            pass

    return ChunkResponse(
        chunks=state.chunks,
        scraped_items=state.scraped_items or [],
        chunk_count=len(state.chunks),
        metadata=metadata,
    )


@router.post("/push", response_model=PushResponse)
def execute_push(req: PushRequest):
    """Embed chunks and push to Qdrant."""
    from QdrantTracker import QdrantTracker
    from my_collections.SCS_Collection import SCS_Collection
    from workflow.models import EMBEDDING_DIMS

    tracker = QdrantTracker()
    name = req.collection_name
    embedding_model = req.embedding_model
    vector_size = EMBEDDING_DIMS.get(embedding_model, 1536)

    # Filter out skip/excluded URLs
    chunks = req.chunks
    items = req.scraped_items or []
    if (req.skip_urls or req.excluded_urls) and items:
        skip_set = set(req.skip_urls + req.excluded_urls)
        filtered = [(it, ch) for it, ch in zip(items, chunks) if it.get("url") not in skip_set]
        if filtered:
            items, chunks = zip(*filtered)
            items, chunks = list(items), list(chunks)
        else:
            items, chunks = [], []

    if req.recreate_collection:
        # Delete and recreate
        if tracker._existing_collection_name(name):
            tracker._delete_collection(name)
        tracker._create_collection(name, vector_size=vector_size)

    elif not tracker._existing_collection_name(name):
        # Create if doesn't exist
        tracker._create_collection(name, vector_size=vector_size)

    # Build points and push
    temp_coll = SCS_Collection(name)
    temp_coll.append_sentences(chunks, req.source_label, scraped_items=items)
    points = temp_coll.points_to_save(model_id=embedding_model)
    tracker.append_points_to_collection(name, points)

    return PushResponse(
        points_pushed=len(points),
        collection_name=name,
        message=f"Pushed {len(points)} points to '{name}' (model={embedding_model})",
    )


@router.post("/delete-points", response_model=DeletePointsResponse)
def execute_delete_points(req: DeletePointsRequest):
    """Delete all Qdrant points whose source_url matches any of the given URLs."""
    from QdrantTracker import QdrantTracker

    tracker = QdrantTracker()
    if not tracker._existing_collection_name(req.collection_name):
        return DeletePointsResponse(
            collection_name=req.collection_name,
            deleted_urls=[],
            errors=[],
            collection_existed=False,
        )

    deleted: list[str] = []
    errors: list[dict] = []
    for url in req.urls:
        try:
            tracker._delete_points_by_url(req.collection_name, url)
            deleted.append(url)
        except Exception as e:
            errors.append({"url": url, "error": str(e)})

    return DeletePointsResponse(
        collection_name=req.collection_name,
        deleted_urls=deleted,
        errors=errors,
        collection_existed=True,
    )


@router.post("/qa", response_model=QAResponse)
def execute_qa(req: QARequest):
    """Ask a question against a Qdrant collection."""
    from chatbot import get_retrieved_info, get_answer

    history: list = []
    retrieved = get_retrieved_info(req.question, history, req.collection_name)
    answer = get_answer(history, retrieved, req.question, req.company_name)

    # Extract source URLs from retrieved context
    sources = []
    if hasattr(retrieved, "sources"):
        sources = retrieved.sources
    elif isinstance(retrieved, dict) and "sources" in retrieved:
        sources = retrieved["sources"]

    return QAResponse(
        question=req.question,
        answer=answer,
        sources=sources,
    )


@router.get("/collection-info/{collection_name}", response_model=CollectionInfoResponse)
def get_collection_info(collection_name: str):
    """Get Qdrant collection health info."""
    from QdrantTracker import QdrantTracker

    tracker = QdrantTracker()
    exists = bool(tracker._existing_collection_name(collection_name))

    if not exists:
        return CollectionInfoResponse(
            collection_name=collection_name,
            exists=False,
        )

    try:
        from qdrant_utils import get_qdrant_connection
        client = get_qdrant_connection()
        info = client.get_collection(collection_name)
        return CollectionInfoResponse(
            collection_name=collection_name,
            exists=True,
            points_count=info.points_count or 0,
            vector_size=info.config.params.vectors.size if info.config and info.config.params else 0,
            status=str(info.status) if info.status else "unknown",
        )
    except Exception as e:
        return CollectionInfoResponse(
            collection_name=collection_name,
            exists=True,
            status=f"error: {e}",
        )


@router.get("/scrapers")
def list_scrapers():
    """List available scraper configs from the scrapers directory."""
    import os
    import yaml

    scrapers_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ingestion", "scrapers", "configs"
    )

    result = []
    if os.path.isdir(scrapers_dir):
        for fname in sorted(os.listdir(scrapers_dir)):
            if fname.endswith((".yaml", ".yml", ".json")):
                path = os.path.join(scrapers_dir, fname)
                try:
                    with open(path) as f:
                        if fname.endswith(".json"):
                            cfg = json.load(f)
                        else:
                            cfg = yaml.safe_load(f)
                    result.append({
                        "name": cfg.get("name", fname.rsplit(".", 1)[0]),
                        "engine": cfg.get("engine", "playwright"),
                        "scrape_mode": cfg.get("scrape_mode", "sitemap"),
                        "file": fname,
                    })
                except Exception:
                    result.append({"name": fname.rsplit(".", 1)[0], "file": fname, "error": "parse_failed"})

    return {"scrapers": result}


@router.get("/scrapers/{name}")
def get_scraper(name: str):
    """Get a specific scraper config by name."""
    import os
    import yaml

    scrapers_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ingestion", "scrapers", "configs"
    )

    # Try multiple extensions
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(scrapers_dir, name + ext)
        if os.path.isfile(path):
            with open(path) as f:
                if ext == ".json":
                    return json.load(f)
                return yaml.safe_load(f)

    # Also check by 'name' field inside configs
    if os.path.isdir(scrapers_dir):
        for fname in os.listdir(scrapers_dir):
            if not fname.endswith((".yaml", ".yml", ".json")):
                continue
            path = os.path.join(scrapers_dir, fname)
            try:
                with open(path) as f:
                    if fname.endswith(".json"):
                        cfg = json.load(f)
                    else:
                        import yaml
                        cfg = yaml.safe_load(f)
                if cfg.get("name") == name:
                    return cfg
            except Exception:
                continue

    raise HTTPException(404, f"Scraper config '{name}' not found")


# ═══════════════════════════════════════════════════════════════════════
# FAQ extraction — ask an LLM for Q&A pairs from raw text
# ═══════════════════════════════════════════════════════════════════════

class FaqExtractRequest(BaseModel):
    """Extract Q&A pairs from a block of text (one rag_sources row's content)."""
    text: str
    language: str = "en"
    company_name: str = "the assistant"
    max_items: int = 50
    # Optional: source URL echoed back in each pair for traceability
    source_url: Optional[str] = None


class FaqPair(BaseModel):
    question: str
    answer: str


class FaqExtractResponse(BaseModel):
    pairs: list[FaqPair] = Field(default_factory=list)
    count: int = 0
    source_url: Optional[str] = None
    error: Optional[str] = None


@router.post("/faq-extract", response_model=FaqExtractResponse)
def execute_faq_extract(req: FaqExtractRequest):
    """
    Extract FAQ Q&A pairs from raw text. Stateless: no collection lookup,
    no disk I/O — just text in, pairs out. Used by jBKB's FAQ pipeline to
    turn one rag_sources row (destination='faq_table') into faq_items rows.
    """
    from llms.openai_utils import openai_chat_completion

    text = (req.text or "").strip()
    if not text:
        return FaqExtractResponse(pairs=[], count=0, source_url=req.source_url, error="empty text")

    # Cap input size to avoid expensive calls on very large pages
    truncated = text[:40_000]

    system_prompt = (
        "Extract FAQ-style question-and-answer pairs from the text.\n"
        "Rules:\n"
        "  - Output valid JSON: {\"pairs\": [{\"question\": \"...\", \"answer\": \"...\"}]}\n"
        "  - One pair per distinct topic.\n"
        "  - Only include pairs where both question and answer are clearly supported by the text.\n"
        "  - Skip promotional filler, navigation blocks, and duplicates.\n"
        f"  - Language: {req.language}. Speak as {req.company_name}.\n"
        f"  - At most {req.max_items} pairs.\n"
        "Return ONLY the JSON. No prose, no markdown fences."
    )

    try:
        raw = openai_chat_completion(system_prompt, f"Text:\n{truncated}", model="gpt-4o-mini")
    except Exception as e:
        return FaqExtractResponse(pairs=[], count=0, source_url=req.source_url, error=str(e))

    # Strip accidental code fences / surrounding prose
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    # Take the JSON object if there's leading/trailing noise
    lbrace = cleaned.find("{")
    rbrace = cleaned.rfind("}")
    if lbrace >= 0 and rbrace > lbrace:
        cleaned = cleaned[lbrace:rbrace + 1]

    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        return FaqExtractResponse(pairs=[], count=0, source_url=req.source_url, error=f"LLM did not return valid JSON: {e}")

    raw_pairs = parsed.get("pairs") if isinstance(parsed, dict) else None
    if not isinstance(raw_pairs, list):
        return FaqExtractResponse(pairs=[], count=0, source_url=req.source_url, error="no 'pairs' array in LLM output")

    pairs: list[FaqPair] = []
    for p in raw_pairs[:req.max_items]:
        if not isinstance(p, dict):
            continue
        q = (p.get("question") or "").strip()
        a = (p.get("answer") or "").strip()
        if q and a:
            pairs.append(FaqPair(question=q, answer=a))

    return FaqExtractResponse(pairs=pairs, count=len(pairs), source_url=req.source_url)
