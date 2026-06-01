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
    # Optional page filter for PDF sources. Format: "20-45" or
    # "1,5,8-10". Null/empty means the entire file.
    page_range: Optional[str] = None
    # Optional: routing metadata for relevance filtering
    routing: Optional[dict] = None
    # Collection context (for labeling)
    collection_name: Optional[str] = None
    source_label: Optional[str] = None
    # 3-small replaces ada-002 as the default — same 1536 dims (so still
    # drop-in compatible with existing ada-002 Qdrant collections at query
    # time), cheaper, higher quality. jBKB-driven flows always pass an
    # explicit `embedding_model` from the rag_collection's config so this
    # default only applies to direct API callers.
    embedding_model: str = "text-embedding-3-small"


class FetchResponse(BaseModel):
    """Result of a fetch step."""
    raw_text: str
    # Text with ONLY the baseline strip applied (script/style/noscript/
    # iframe). No user exclude_selectors. Same shape as raw_text. Used
    # by jBKB to populate rag_sources.content_raw and compute a stable
    # content hash for change detection. Falls back to raw_text when
    # the scraper doesn't produce a distinct baseline (structured mode,
    # sitemap / crawl modes that haven't been upgraded).
    baseline_text: str = ""
    # Cookie / consent / privacy text captured BEFORE the baseline
    # strip removes the CMP roots (CookieYes, OneTrust, Cookiebot, …).
    # jBKB uses this to seed the per-CBVA "Cookies and Privacy"
    # synthetic source on the first page where it appears, so the
    # cookie-policy text remains searchable in Qdrant after the
    # boilerplate is stripped from every regular page. Empty when
    # the page has no CMP roots.
    cmp_text: str = ""
    scraped_items: list[dict] = Field(default_factory=list)  # [{url, text, text_baseline, cmp_text, outgoing_links}]
    pdf_pages: list[dict] = Field(default_factory=list)  # [{page, text}, ...]
    source_label: str = ""
    char_count: int = 0
    page_count: int = 0
    relevance_report: Optional[dict] = None
    # Deduplicated, same-host absolute URLs reachable via <a href>
    # from every scraped page in this fetch. Joined from
    # scraped_items[].outgoing_links so jBKB doesn't have to walk
    # the per-item list. Empty when no scraper populated it
    # (sitemap / crawl modes that haven't been upgraded).
    outgoing_links: list[str] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    """Chunking parameters."""
    mode: str = "hierarchical"  # simple | hierarchical | proposition
    size: int = 2000  # chunk size (simple) or parent size (hierarchical)
    overlap: int = 200
    child_size: int = 400  # hierarchical only
    child_overlap: int = 50  # hierarchical only
    # Per-source override: when True, skip both the parent/child and
    # simple splitters and emit ONE chunk per page (full cleaned
    # text). Best for list/table content (postal codes, opening
    # hours, product price lists) where splitting would scatter
    # related rows across multiple chunks.
    single_chunk: bool = False


class ChunkRequest(BaseModel):
    """Request to chunk text into pieces."""
    text: Optional[str] = None  # raw or cleaned text to chunk
    scraped_items: list[dict] = Field(default_factory=list)  # alternative: pre-scraped pages
    pdf_pages: list[dict] = Field(default_factory=list)  # for PDF page attribution
    source_type: str = "url"
    source_label: str = "document"
    # Original source URL or file path (whatever was used at fetch time).
    # For PDF sources this drives _attach_pdf_page_urls — when it's an
    # http(s):// URL, chunks get the real URL with #page=N appended; when
    # it's a local path or empty, falls back to the legacy pdf://<basename>
    # format. /chunk doesn't refetch; the caller is the source of truth.
    source_path: Optional[str] = None
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    # Collection's content lexicon — raw strings (routing_keywords,
    # typical_questions, etc.). Tokenized into a stopword-filtered set
    # and passed to the text_cleaner so short lines without keyword
    # overlap are dropped. When empty, text_cleaner falls back to its
    # hardcoded PT/EN pattern list.
    keywords: list[str] = Field(default_factory=list)


class ChunkResponse(BaseModel):
    """Result of a chunk step."""
    chunks: list[str]
    scraped_items: list[dict] = Field(default_factory=list)  # with URL attribution
    chunk_count: int = 0
    metadata: Optional[dict] = None  # auto-generated collection metadata
    # Plain post-cleaner pre-chunk text — what jBKB writes to
    # rag_sources.content_clean for human inspection / hand editing.
    # Replaces the old behaviour where the orchestrator joined the
    # chunked-with-Context/Passage strings, which produced massive
    # duplication for hierarchical chunks.
    cleaned_text: str = ""


class PushRequest(BaseModel):
    """Request to embed and push chunks to Qdrant."""
    collection_name: str
    chunks: list[str]
    scraped_items: list[dict] = Field(default_factory=list)  # [{url, text}, ...] for source attribution
    source_label: str = "document"
    # 3-small replaces ada-002 as the default — same 1536 dims (so still
    # drop-in compatible with existing ada-002 Qdrant collections at query
    # time), cheaper, higher quality. jBKB-driven flows always pass an
    # explicit `embedding_model` from the rag_collection's config so this
    # default only applies to direct API callers.
    embedding_model: str = "text-embedding-3-small"
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
    # 3-small replaces ada-002 as the default — same 1536 dims (so still
    # drop-in compatible with existing ada-002 Qdrant collections at query
    # time), cheaper, higher quality. jBKB-driven flows always pass an
    # explicit `embedding_model` from the rag_collection's config so this
    # default only applies to direct API callers.
    embedding_model: str = "text-embedding-3-small"


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


class ProbePdfRequest(BaseModel):
    """Probe a PDF (local path or http(s) URL) for page structure.

    Used by the jBKB upload dialog to show a page outline and
    auto-suggest per-range splits before the source rows are created.
    """
    file_path: str


class ProbePdfPagePreview(BaseModel):
    page: int
    first_line: str


class ProbePdfResponse(BaseModel):
    page_count: int
    page_previews: list[ProbePdfPagePreview] = Field(default_factory=list)


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
        # Page filter only applies to PDFs; txt doesn't have pages.
        if req.source_type == "pdf" and req.page_range:
            source_config["page_range"] = req.page_range
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

    # baseline_text — joined from scraped_items[].text_baseline when the
    # scraper populated it (single_page mode does; sitemap/crawl/
    # structured modes don't, so baseline_text falls back to raw_text so
    # content_raw never ends up empty).
    baseline_pieces = [
        (it.get("text_baseline") or it.get("text") or "")
        for it in (state.scraped_items or [])
    ]
    baseline_text = "\n\n".join(p for p in baseline_pieces if p)
    if not baseline_text:
        baseline_text = state.raw_text or ""

    # cmp_text — joined from scraped_items[].cmp_text. First non-empty
    # wins for the response-level field; per-item values are still in
    # scraped_items so jBKB can attribute back to a URL if needed.
    cmp_pieces = [
        (it.get("cmp_text") or "")
        for it in (state.scraped_items or [])
    ]
    cmp_text = next((p for p in cmp_pieces if p and p.strip()), "")

    # outgoing_links — union of scraped_items[].outgoing_links across
    # every page in this fetch. Deduped while preserving first-seen
    # order so the BFS source list is stable across reruns. Same-host
    # filtering already happened in the scraper.
    seen_links: set = set()
    outgoing_links: list = []
    for it in (state.scraped_items or []):
        for link in (it.get("outgoing_links") or []):
            if not isinstance(link, str) or link in seen_links:
                continue
            seen_links.add(link)
            outgoing_links.append(link)

    return FetchResponse(
        raw_text=state.raw_text or "",
        baseline_text=baseline_text,
        cmp_text=cmp_text,
        scraped_items=state.scraped_items or [],
        pdf_pages=state.pdf_pages or [],
        source_label=state.source_label or "",
        char_count=len(state.raw_text or ""),
        page_count=len(state.scraped_items) or len(state.pdf_pages),
        relevance_report=state.relevance_report,
        outgoing_links=outgoing_links,
    )


@router.post("/chunk", response_model=ChunkResponse)
def execute_chunk(req: ChunkRequest):
    """Chunk text into pieces using the specified strategy."""
    from workflow.models import WorkflowState, ChunkingConfig as WfChunkingConfig

    state = WorkflowState()
    state.source_type = req.source_type
    state.source_label = req.source_label
    state.keywords = req.keywords or []
    # Carry the source's original URL/path through so _attach_pdf_page_urls
    # can build correct per-chunk source URLs. Without this, PDF sources
    # would always fall back to the legacy pdf://<basename> form even when
    # the source was fetched from an http(s) URL.
    if req.source_path:
        state.source_config = {"path": req.source_path}

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
        single_chunk=cfg.single_chunk,
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
        cleaned_text=state.cleaned_text or "",
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
# Suggest exclude_selectors — scan a few URLs, find boilerplate, return
# CSS selectors the caller can strip before text extraction.
#
# Three signals, ranked:
#   A. Landmark tags/roles (nav, footer, [role=banner], [role=contentinfo])
#      — always propose if present on any page.
#   B. class/id match against a small set of boilerplate patterns
#      (cookie, consent, gdpr, newsletter, subscribe, popup, banner,
#      modal, notice) — proposed once we've seen the selector on ≥ 1
#      page, with higher confidence if seen on more pages.
#   C. Cross-page text repetition — elements with class or id whose
#      text content hashes identically across ≥ 2 pages. Catches things
#      like a long footer blurb or a site-wide CTA that signals A/B miss.
#
# Output is deduplicated and sorted by "seen_on" count descending, so the
# UI can present the strongest signals first.
# ═══════════════════════════════════════════════════════════════════════

class SuggestExcludeRequest(BaseModel):
    """Sample N URLs and return CSS selectors of likely boilerplate."""
    urls: list[str]
    # How many pages to actually fetch. Truncates urls when larger.
    max_pages: int = 5
    # Per-request HTTP timeout in seconds.
    timeout: float = 10.0
    # Content lexicon — raw strings (e.g. routing_keywords,
    # typical_questions, collection name). Tokenized into a stopword-
    # filtered word set. Signal C refuses to promote any element whose
    # text contains at least one lexicon word: those elements almost
    # certainly hold on-topic content that shouldn't be stripped. When
    # empty, Signal C falls back to purely structural gates (tag +
    # text-length + deny list).
    keywords: list[str] = Field(default_factory=list)


class SuggestedSelector(BaseModel):
    selector: str
    reason: str
    seen_on: int  # count of pages where this selector was matched


class SuggestExcludeResponse(BaseModel):
    selectors: list[SuggestedSelector] = Field(default_factory=list)
    sampled: list[str] = Field(default_factory=list)   # URLs successfully fetched
    errors: list[str] = Field(default_factory=list)    # "url — error message"


_BOILERPLATE_PATTERNS = [
    "cookie", "consent", "gdpr", "newsletter", "subscribe",
    "popup", "banner", "modal", "notice",
]

_LANDMARK_RULES: list[tuple[str, str]] = [
    ("nav", "<nav> element"),
    ("footer", "<footer> element"),
    ("header", "<header> element"),
    ('[role="banner"]', "ARIA role=banner"),
    ('[role="contentinfo"]', "ARIA role=contentinfo"),
    ('[role="navigation"]', "ARIA role=navigation"),
    ('[role="complementary"]', "ARIA role=complementary"),
]

_GENERIC_CLASSES = {
    "container", "wrapper", "content", "inner", "outer", "row", "col",
    "column", "box", "block", "item", "section",
}

# Tags that are never useful as a user-facing exclude suggestion —
# baseline-stripped server-side or pure metadata.
_SKIP_TAGS = {"script", "style", "noscript", "iframe", "link", "meta"}

# Block-level container tags that are plausible wrappers for boilerplate
# (footers, cookie banners, sidebars). Signal C (text repetition) only
# promotes an element if its tag is in this set — filters out spans,
# paragraphs, list items, anchors, headings that repeat for structural
# reasons but aren't "chrome".
_BOILERPLATE_CONTAINER_TAGS = {
    "div", "section", "aside", "nav", "header", "footer",
    "main", "article", "template",
}


def _is_generic_class(name: str) -> bool:
    n = name.lower()
    return n in _GENERIC_CLASSES or len(n) <= 2


def _is_hashed_class(name: str) -> bool:
    """Elementor-style classes like `elementor-element-4a39687` — suffix is
    a random hash, so the selector is neither stable nor meaningful. Treat
    them as unstable so we don't suggest `div.elementor-element-a492b86`
    when the same widget shape repeats across pages."""
    import re as _re
    return bool(_re.search(r"-[0-9a-f]{6,}$", name.lower()))


# Known page-builder LAYOUT classes — never boilerplate, always content
# wrappers. Includes Elementor primitives (columns, sections, widget
# containers, template roots) and Jet Engine listing primitives. If any
# of these patterns match, _tag_selector refuses the class so Signal C
# can't promote it from "repeats across pages" to "strip me".
#
# We're deliberately conservative here: stripping a layout wrapper wipes
# the whole page, but false-negatives just mean the user types the
# selector by hand or keeps unnecessary chrome.
import re as _layout_re
_LAYOUT_CLASS_PATTERNS = [
    # Elementor — column / section / widget primitives
    _layout_re.compile(r"^elementor-column(?:-|$)", _layout_re.I),
    _layout_re.compile(r"^elementor-element-populated$", _layout_re.I),
    _layout_re.compile(r"^elementor-top-column$", _layout_re.I),
    _layout_re.compile(r"^elementor-inner-column$", _layout_re.I),
    _layout_re.compile(r"^elementor-widget-container$", _layout_re.I),
    _layout_re.compile(r"^elementor-section(?:-|$)", _layout_re.I),
    _layout_re.compile(r"^elementor-container$", _layout_re.I),
    _layout_re.compile(r"^elementor-row$", _layout_re.I),
    # Elementor widget wrappers that by themselves are content hosts
    _layout_re.compile(r"^elementor-widget-text-editor$", _layout_re.I),
    _layout_re.compile(r"^elementor-widget-shortcode$", _layout_re.I),
    _layout_re.compile(r"^elementor-widget-image$", _layout_re.I),
    _layout_re.compile(r"^elementor-widget-heading$", _layout_re.I),
    _layout_re.compile(r"^elementor-shortcode$", _layout_re.I),
    _layout_re.compile(r"^elementor-icon-list(?:--|$)", _layout_re.I),
    _layout_re.compile(r"^elementor-list-item-link", _layout_re.I),
    # Elementor template root wrappers (classname is just the numeric id
    # of the template — e.g. elementor-269). Not useful as a selector.
    _layout_re.compile(r"^elementor-\d+$", _layout_re.I),
    # Jet Engine / JetElements listing + dynamic-post primitives
    _layout_re.compile(r"^jet-listing-dynamic-post(?:-|$)", _layout_re.I),
    _layout_re.compile(r"^jet-listing-grid(?:$|_)", _layout_re.I),
    _layout_re.compile(r"^jet-equal-columns", _layout_re.I),
]


def _is_layout_class(name: str) -> bool:
    return any(p.match(name) for p in _LAYOUT_CLASS_PATTERNS)


# Stopwords for the content-lexicon filter. Multilingual because the
# RAG collections we see in jBKB span PT/EN/ES/FR. Purpose: words like
# "de", "the", "para" shouldn't count as meaningful content terms —
# they'd short-circuit the filter and let footers through.
_STOPWORDS: set[str] = {
    # English
    "the", "and", "or", "of", "for", "to", "in", "on", "at", "by", "with",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "but", "not", "no", "this", "that",
    "these", "those", "it", "its", "if", "so", "an", "any", "all", "each",
    "our", "your", "their", "his", "her", "we", "you", "they", "he", "she",
    "can", "will", "just", "which", "when", "where", "what", "how", "why",
    "about", "more", "also",
    # Portuguese
    "da", "do", "das", "dos", "para", "em", "por", "com", "sem", "mais",
    "ao", "às", "aos", "ou", "mas", "que", "se", "na", "no", "nas", "nos",
    "um", "uma", "uns", "umas", "é", "são", "foi", "seu", "sua", "seus",
    "suas", "este", "esta", "esse", "essa", "isso", "isto", "aquele",
    "aquela", "aquilo", "muito", "muita",
    # Spanish
    "el", "los", "las", "del", "al", "y", "u", "pero", "sí",
    # French
    "le", "les", "un", "une", "des", "du", "et", "aux",
    "pour", "par", "avec", "sans", "sur",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokens. Keeps accented characters (PT/ES/FR)
    and drops punctuation."""
    import re as _tre
    return _tre.findall(r"[\w]+", text.lower(), flags=_tre.UNICODE)


def _build_lexicon(raw_strings: list[str]) -> set[str]:
    """Tokenize every string, drop short/stopword/numeric tokens, return
    the deduplicated content-term set."""
    out: set[str] = set()
    for raw in raw_strings or []:
        if not raw:
            continue
        for tok in _tokenize(raw):
            if len(tok) <= 2:
                continue
            if tok.isdigit():
                continue
            if tok in _STOPWORDS:
                continue
            out.add(tok)
    return out


def _has_lexicon_hit(text: str, lexicon: set[str]) -> bool:
    """True if ANY token in text overlaps the lexicon. Zero-overlap is a
    strong "off-topic" signal — almost all real chrome has zero overlap
    with the collection's routing terms."""
    if not lexicon:
        return False
    return any(t in lexicon for t in _tokenize(text))


def _tag_selector(tag) -> Optional[str]:
    """Build a stable selector for this tag — prefer id, then class,
    then None. Used to key elements across pages.

    Skips tags that are baseline-stripped server-side so we don't waste
    suggestions on them (users would see #wp-emoji-styles-inline-css
    and friends otherwise). Also rejects hashed classes like
    `elementor-element-4a39687` — those repeat by accident, not because
    the element is boilerplate."""
    if tag.name in _SKIP_TAGS:
        return None
    tag_id = tag.get("id")
    if tag_id and " " not in tag_id and "\n" not in tag_id:
        # Elementor template-root IDs ("#elementor-269") aren't useful.
        if not _layout_re.match(r"^(?:elementor|jet)-?\d+$", tag_id, _layout_re.I):
            return f"#{tag_id}"
    classes = [
        c for c in (tag.get("class") or [])
        if not _is_generic_class(c)
        and not _is_hashed_class(c)
        and not _is_layout_class(c)
    ]
    classes.sort(key=len, reverse=True)
    if classes:
        return f"{tag.name}.{classes[0]}"
    return None


def _normalize_text(raw: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", raw).strip()


@router.post("/suggest-exclude-selectors", response_model=SuggestExcludeResponse)
def execute_suggest_exclude_selectors(req: SuggestExcludeRequest):
    """Fetch up to max_pages URLs, run three heuristics, return selectors."""
    import re as _re
    from collections import defaultdict

    try:
        import httpx
        from bs4 import BeautifulSoup
    except Exception as e:
        raise HTTPException(500, f"Missing deps: {e}")

    urls = (req.urls or [])[: max(1, min(req.max_pages, 10))]
    if not urls:
        return SuggestExcludeResponse()

    lexicon = _build_lexicon(req.keywords)

    # ── Fetch HTML ──────────────────────────────────────────────────
    htmls: list[tuple[str, str]] = []  # (url, html)
    errors: list[str] = []
    with httpx.Client(
        timeout=req.timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; jBKB-RAG/1.0)"},
    ) as client:
        for url in urls:
            try:
                resp = client.get(url)
                resp.raise_for_status()
                htmls.append((url, resp.text))
            except Exception as e:
                errors.append(f"{url} — {e}")

    if not htmls:
        return SuggestExcludeResponse(errors=errors)

    # ── Scan each page ──────────────────────────────────────────────
    boilerplate_re = _re.compile("|".join(_BOILERPLATE_PATTERNS), _re.I)

    # selector -> (reason, seen_on_count)
    seen_counts: dict[str, int] = defaultdict(int)
    reasons: dict[str, str] = {}
    # selector -> {text_hash: {count, sample, landmark}} — sample/landmark
    # are captured on first occurrence so the Signal-C reason can include
    # WHERE the element sits (nearest landmark ancestor) and WHAT it says
    # (first ~60 chars of the repeated text). Lets the operator recognise
    # "footer copyright" vs "newsletter CTA" without opening the page.
    text_hash_pages: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "sample": "", "landmark": None})
    )

    _LANDMARK_TAGS = {"header", "footer", "nav", "aside", "main"}
    _LANDMARK_ROLES = {"banner", "contentinfo", "navigation", "complementary", "main"}

    def _nearest_landmark(el) -> Optional[str]:
        """Walk up parents, return first landmark ancestor as e.g.
        '<footer>' or '<div role=navigation>'. None when nothing on the
        way up the tree qualifies."""
        cur = el.parent
        while cur is not None and getattr(cur, "name", None):
            name = cur.name
            if name in _LANDMARK_TAGS:
                return f"<{name}>"
            role_attr = cur.get("role") if hasattr(cur, "get") else None
            if role_attr and role_attr in _LANDMARK_ROLES:
                return f"<{name} role={role_attr}>"
            cur = cur.parent
        return None

    for (_url, html) in htmls:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            errors.append(f"{_url} — parse failed: {e}")
            continue

        page_hits: set[str] = set()

        # Signal A — landmarks (always propose if present)
        for sel, why in _LANDMARK_RULES:
            try:
                if soup.select_one(sel):
                    page_hits.add(sel)
                    reasons.setdefault(sel, why)
            except Exception:
                pass

        # Signal B — class/id contains boilerplate pattern.
        # Skip script/style/etc up front (baseline-stripped anyway, and on
        # WordPress sites the plugins inject dozens of them so they drown
        # out real suggestions).
        for el in soup.find_all(True):
            if el.name in _SKIP_TAGS:
                continue
            cls = el.get("class") or []
            id_val = el.get("id") or ""
            matched_via = None
            for c in cls:
                if boilerplate_re.search(c):
                    matched_via = f"class contains {_re.escape(c)!r}"
                    break
            if not matched_via and id_val and boilerplate_re.search(id_val):
                matched_via = f"id contains {id_val!r}"
            if matched_via:
                sel = _tag_selector(el)
                if sel:
                    page_hits.add(sel)
                    reasons.setdefault(sel, matched_via)

        # Signal C — cross-page text repetition. Catches block-level
        # wrappers that don't match A (not a landmark tag) or B (no
        # boilerplate keyword in class/id) but still repeat identically
        # across pages — e.g., div.elementor-location-footer on
        # Elementor sites which use a <div> for the footer.
        #
        # Gates to keep noise down:
        #   - tag must be a block-level container (div/section/aside/nav/
        #     header/footer/main/article/template). Filters span, p, li,
        #     a, headings, images etc.
        #   - text >= 100 chars (short labels like "Sort by: …" repeat
        #     for structural reasons, not because they're chrome).
        #   - _tag_selector already rejects script/style/etc and hashed
        #     classes (elementor-element-4a39687).
        #   - Content-lexicon veto: if the element's text contains any
        #     keyword from the collection's routing terms (typical
        #     questions, doc type, collection name, keywords), it's
        #     on-topic content and must NOT be promoted. Catches the
        #     Elementor-widget-container case where layout wrappers
        #     surround product descriptions with real keywords.
        import hashlib
        for el in soup.find_all(True):
            if el.name not in _BOILERPLATE_CONTAINER_TAGS:
                continue
            sel = _tag_selector(el)
            if not sel:
                continue
            text = _normalize_text(el.get_text(separator=" ", strip=True))
            if len(text) < 100:
                continue
            if _has_lexicon_hit(text, lexicon):
                continue
            h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            bucket = text_hash_pages[sel][h]
            bucket["count"] += 1
            if not bucket["sample"]:
                bucket["sample"] = text[:80]
                bucket["landmark"] = _nearest_landmark(el)

        for s in page_hits:
            seen_counts[s] += 1

    # Finalise Signal C: promote any selector whose text hash was seen
    # identically on ≥ 2 pages. When the selector was already flagged by
    # A or B, we keep the original reason; the text-repeat fact gets
    # appended as an upgrade note.
    if len(htmls) >= 2:
        for sel, hash_buckets in text_hash_pages.items():
            if not hash_buckets:
                continue
            # Dominant hash = the most-repeated text body; that's the one
            # whose sample + landmark we cite in the reason.
            top_hash, top_bucket = max(hash_buckets.items(), key=lambda kv: kv[1]["count"])
            top = top_bucket["count"]
            if top < 2:
                continue
            seen_counts[sel] = max(seen_counts[sel], top)
            bits = [f"identical text on {top}/{len(htmls)} pages"]
            landmark = top_bucket["landmark"]
            if landmark:
                bits.append(f"inside {landmark}")
            sample = top_bucket["sample"]
            if sample:
                snippet = sample[:60].rstrip()
                if len(sample) > 60:
                    snippet += "…"
                bits.append(f'starts: "{snippet}"')
            addendum = " · ".join(bits)
            existing = reasons.get(sel, "")
            reasons[sel] = (existing + " · " + addendum).strip(" ·") if existing else addendum

    # ── Build response ──────────────────────────────────────────────
    out = [
        SuggestedSelector(
            selector=sel,
            reason=reasons.get(sel, ""),
            seen_on=cnt,
        )
        for sel, cnt in seen_counts.items()
        if cnt > 0
    ]
    # Highest confidence first, then alphabetical for stable order.
    out.sort(key=lambda s: (-s.seen_on, s.selector))
    return SuggestExcludeResponse(
        selectors=out,
        sampled=[u for (u, _h) in htmls],
        errors=errors,
    )


# ═══════════════════════════════════════════════════════════════════════
# Preview exclusions — render a single page with matched elements
# outlined in red so the user can visually confirm the exclude_selectors
# list does the right thing before saving.
#
# Output: one big HTML string that the caller renders inside a sandboxed
# iframe. We:
#   - strip every <script> (safety + avoids any JS that fights our
#     highlight CSS)
#   - inject a <base href> so relative URLs for CSS/images/fonts still
#     resolve against the original origin
#   - inject a <style> with .jbkb-ex-hit and a floating legend
#   - tag each matched element with class="jbkb-ex-hit" and
#     data-jbkb-sel="<matched selector>"
# The baseline strip list (script/style/noscript/iframe) is ALSO included
# so users see what the scraper actually drops in production.
# ═══════════════════════════════════════════════════════════════════════

class PreviewExclusionsRequest(BaseModel):
    url: str
    exclude_selectors: list[str] = Field(default_factory=list)
    # Bumped from 10s → 30s to match the scraper's timeout. Complex
    # sites (WP + Elementor + trackers + chat widgets) routinely take
    # longer than 10s to reach networkidle. Default is generous; actual
    # goto uses domcontentloaded below so it usually returns much faster.
    timeout: float = 30.0
    # When true, the returned HTML includes a small inspector script
    # that intercepts clicks inside the iframe and posts candidate
    # CSS selectors back to the parent window via postMessage. The
    # parent (jBKB) renders a picker so the user can add any of the
    # suggested selectors to their exclude list. Requires the parent
    # to mount the iframe with sandbox="allow-same-origin allow-scripts".
    inspect_mode: bool = False
    # Fetch engine — "playwright" (default) runs the page in a real
    # headless browser so JS-rendered content (cookie banners, dynamic
    # menus, Elementor widgets) appears exactly as the actual scrape
    # sees it. "httpx" is faster (~10× cheaper) but only sees the
    # initial server-rendered HTML, so JS-injected boilerplate won't
    # show up — matching the scraper's view is what actually matters
    # for building exclude_selectors.
    engine: str = "playwright"


class PreviewExclusionsResponse(BaseModel):
    url: str
    html: str = ""                      # rewritten HTML to iframe
    matched_count: int = 0              # total elements tagged
    per_selector: dict[str, int] = Field(default_factory=dict)  # selector -> count
    error: Optional[str] = None


_PREVIEW_BASELINE = [
    # HTML elements that never carry meaningful page content
    "script", "style", "noscript", "iframe",
    # Universal cookie / consent / privacy banner roots. Always
    # boilerplate, never page content. Stripped by default so users
    # don't have to discover and configure them per-site. The phantom
    # "Cookies and Privacy" source captures this text once per CBVA
    # so it's still searchable in Qdrant.
    # CookieYes — anchor to word boundary so we don't also strip
    # Elementor's "elementor-sticky--*" classes (sti·cky-·-active).
    '[class^="cky-"]',
    '[class*=" cky-"]',
    "#onetrust-banner-sdk",             # OneTrust banner
    "#onetrust-consent-sdk",            # OneTrust prefs panel
    "#CybotCookiebotDialog",            # Cookiebot
    ".cookie-notice-container",         # WP Cookie Notice
    ".iubenda-cs-container",            # Iubenda
    ".klaro",                           # Klaro
    "#cmpwrapper",                      # Quantcast / TCFv2
    "#cookiescript_injected",           # CookieScript
    "#hs-eu-cookie-confirmation",       # HubSpot
]

# Inspector script injected when inspect_mode=True. Runs in the iframe
# with the parent's (jBKB's) origin since the iframe uses srcDoc +
# sandbox allow-same-origin allow-scripts.
#
# Intercepts all clicks, prevents navigation/form submission, computes
# a short list of candidate CSS selectors for the clicked element,
# and posts them to the parent. Also highlights the hovered element
# in blue so users can see what they're about to click.
_INSPECTOR_SCRIPT = r"""
(function () {
  // Candidate builder — prefers stable signals (id, role, landmark tag,
  // non-generic class names). Hash-suffix classes and 1-2 char classes
  // are dropped since they're unlikely to be useful across pages.
  function buildCandidates(el) {
    var out = [];
    function addUnique(s) { if (s && out.indexOf(s) === -1) out.push(s); }

    // Walk up to three levels; collect candidates for each.
    var node = el, depth = 0;
    while (node && node.nodeType === 1 && depth < 3) {
      var tag = (node.tagName || '').toLowerCase();
      if (!tag) break;

      // id (if id has no whitespace or newline)
      if (node.id && /^[^\s]+$/.test(node.id)) {
        addUnique('#' + node.id);
      }
      // role-based
      var role = node.getAttribute && node.getAttribute('role');
      if (role) addUnique('[role="' + role + '"]');
      // aria-label contains pattern (for cookie/newsletter dialogs)
      var aria = node.getAttribute && node.getAttribute('aria-label');
      if (aria) {
        var lower = aria.toLowerCase();
        if (/cookie|consent|newsletter|subscribe|banner/.test(lower)) {
          addUnique('[aria-label*="' + lower.slice(0, 40).replace(/"/g, '') + '" i]');
        }
      }
      // tag+class
      var classes = [].slice.call(node.classList || []);
      for (var i = 0; i < classes.length; i++) {
        var c = classes[i];
        if (!c || c.length <= 2) continue;
        if (/-[0-9a-f]{6,}$/.test(c)) continue;          // hashed
        addUnique(tag + '.' + c);
      }
      // bare landmark tag
      if (['nav','footer','header','aside','main','article'].indexOf(tag) !== -1) {
        addUnique(tag);
      }

      node = node.parentElement;
      depth++;
    }
    return out;
  }

  function outlineOn(el)  { if (!el || !el.style) return; el.__jbkbPrev = el.style.outline; el.style.outline = '2px solid #3b82f6'; el.style.outlineOffset = '-2px'; el.style.cursor = 'crosshair'; }
  function outlineOff(el) { if (!el || !el.style) return; el.style.outline = el.__jbkbPrev || ''; el.style.outlineOffset = ''; el.style.cursor = ''; }

  var hovered = null;
  document.addEventListener('mouseover', function (e) {
    if (hovered && hovered !== e.target) outlineOff(hovered);
    hovered = e.target;
    outlineOn(hovered);
  }, true);

  document.addEventListener('mouseout', function (e) {
    if (hovered === e.target) { outlineOff(hovered); hovered = null; }
  }, true);

  function intercept(e) {
    e.preventDefault();
    e.stopPropagation();
    var target = e.target;
    if (!target || target.nodeType !== 1) return;
    var candidates = buildCandidates(target);
    var rect = target.getBoundingClientRect();
    var preview = (target.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 160);
    try {
      window.parent.postMessage({
        type: 'jbkb-inspect',
        tag: (target.tagName || '').toLowerCase(),
        candidates: candidates,
        preview: preview,
        rect: { top: rect.top, left: rect.left, width: rect.width, height: rect.height }
      }, '*');
    } catch (err) { /* no-op */ }
  }
  document.addEventListener('click', intercept, true);
  document.addEventListener('submit', function (e) { e.preventDefault(); e.stopPropagation(); }, true);
})();
"""

_PREVIEW_STYLE = """
/* Force the iframe body + html to scroll their own content. Many
   real-world pages set `body { overflow: hidden; }` (often as part
   of cookie-banner / modal lockdown logic that we strip server-side
   but the inline style stays), which makes the rendered preview
   un-scrollable inside the iframe even when its content is taller
   than the viewport. !important wins over the page's own rules.
   height:auto + min-height:100% lets short pages still fill the
   iframe but tall ones overflow + scroll naturally. */
html, body {
  overflow: auto !important;
  height: auto !important;
  min-height: 100% !important;
  max-height: none !important;
  position: static !important;
}
/* Matched elements: red diagonal "caution tape" stripes over whatever
   background they had, plus a solid red outline so the bounds are
   unambiguous. !important overrides site CSS. */
.jbkb-ex-hit {
  outline: 3px solid #dc2626 !important;
  outline-offset: -3px !important;
  background-image: repeating-linear-gradient(
    45deg,
    rgba(220, 38, 38, 0.28),
    rgba(220, 38, 38, 0.28) 10px,
    rgba(220, 38, 38, 0.08) 10px,
    rgba(220, 38, 38, 0.08) 20px
  ) !important;
  background-color: rgba(220, 38, 38, 0.12) !important;
}
.jbkb-ex-hit::before {
  content: "stripped: " attr(data-jbkb-sel);
  display: inline-block;
  background: #dc2626;
  color: #fff;
  font: 10px/1.2 ui-monospace, Menlo, monospace;
  padding: 2px 6px;
  border-radius: 0 0 4px 0;
  margin: -2px 0 2px -2px;
  max-width: 100%;
  word-break: break-all;
  position: relative;
  z-index: 1;
}
/* Hide-mode: parent toggles `jbkb-hide-mode` on <body> to make
   matched elements disappear entirely instead of being highlighted.
   Lets the user see what the page looks like AFTER stripping. */
body.jbkb-hide-mode .jbkb-ex-hit {
  display: none !important;
}
#__jbkb_legend {
  position: fixed;
  top: 8px;
  right: 8px;
  z-index: 2147483647;
  background: rgba(17, 24, 39, 0.95);
  color: #fff;
  padding: 8px 10px;
  border-radius: 6px;
  font: 11px/1.35 ui-sans-serif, system-ui, sans-serif;
  max-width: 320px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}
#__jbkb_legend b { font-weight: 600; }
#__jbkb_legend ul { margin: 4px 0 0; padding-left: 18px; }
#__jbkb_legend li { font-family: ui-monospace, Menlo, monospace; font-size: 10px; }
"""


@router.post("/preview-exclusions", response_model=PreviewExclusionsResponse)
def execute_preview_exclusions(req: PreviewExclusionsRequest):
    """Fetch a URL, tag matched elements with a red outline, return HTML."""
    try:
        import httpx
        from bs4 import BeautifulSoup, Tag
    except Exception as e:
        raise HTTPException(500, f"Missing deps: {e}")

    if not req.url:
        return PreviewExclusionsResponse(url="", error="url is required")

    # ── Fetch ───────────────────────────────────────────────────────
    # Engine matches the actual scraper so the preview reflects what
    # the pipeline sees, including JS-rendered content like cookie
    # banners. User can override via engine="httpx" for a faster
    # preview on static pages.
    engine = (req.engine or "playwright").lower()
    try:
        if engine == "playwright":
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    ctx = browser.new_context(
                        user_agent="Mozilla/5.0 (compatible; jBKB-RAG/1.0)",
                        viewport={"width": 1280, "height": 900},
                    )
                    page = ctx.new_page()
                    # domcontentloaded is much faster and more reliable
                    # than networkidle on sites loaded with trackers /
                    # chat widgets that keep the network busy indefinitely.
                    # We then settle explicitly for content-rendering JS.
                    try:
                        page.goto(req.url, wait_until="domcontentloaded", timeout=int(req.timeout * 1000))
                    except Exception:
                        # Fall back to plain goto (no wait) if even DCL
                        # times out — we may still get usable HTML from
                        # partial load.
                        page.goto(req.url, wait_until="commit", timeout=int(req.timeout * 1000))
                    # Wait for cookie consent / CMP root elements to
                    # appear, with a max timeout. Many CMP libraries
                    # (CookieYes, OneTrust, Cookiebot, Iubenda etc.)
                    # are loaded via Google Tag Manager and take 2-4s
                    # post-DCL to render. The previous flat 1.5s wait
                    # missed them — preview showed a "clean" page that
                    # doesn't match what Build later sees.
                    cmp_root_selector = ', '.join([
                        '[class*="cky-"]',
                        '#onetrust-banner-sdk',
                        '#onetrust-consent-sdk',
                        '#CybotCookiebotDialog',
                        '.cookie-notice-container',
                        '.iubenda-cs-container',
                        '.klaro',
                        '#cmpwrapper',
                        '#cookiescript_injected',
                        '#hs-eu-cookie-confirmation',
                        '[aria-label*="cookie" i]',
                        '[aria-label*="consent" i]',
                    ])
                    try:
                        # First-element-to-appear wins, max 4s. If
                        # nothing matches, we don't crash — just fall
                        # through to the generic 1.5s settle wait.
                        page.wait_for_selector(cmp_root_selector, timeout=4000, state="attached")
                    except Exception:
                        pass
                    # Generic post-DCL settle. Catches any remaining
                    # async widgets that aren't on the CMP list above
                    # (Elementor lazy sections, analytics popups, etc.)
                    page.wait_for_timeout(1500)
                    # Auto-expand FAQ / <details> / Bootstrap collapsibles
                    # so the preview iframe visually matches what the
                    # scraper actually extracts. Two rounds with a short
                    # wait handle nested collapsibles. Same selector list
                    # the scraper uses when expand_collapsibles=true.
                    try:
                        expand_selectors = [
                            '[aria-expanded="false"]',
                            'details:not([open])',
                            '.elementor-tab-title:not(.elementor-active)',
                            '.elementor-toggle-title:not(.elementor-active)',
                            '.elementor-accordion-title:not(.elementor-active)',
                            '[data-toggle="collapse"]',
                            '[data-bs-toggle="collapse"]',
                        ]
                        for _ in range(2):
                            page.evaluate(
                                """(sels) => {
                                    for (const s of sels) {
                                        try {
                                            document.querySelectorAll(s).forEach(el => {
                                                try {
                                                    if (el.tagName === 'DETAILS') { el.open = true; }
                                                    else { el.click(); }
                                                } catch (_) {}
                                            });
                                        } catch (_) {}
                                    }
                                }""",
                                expand_selectors,
                            )
                            page.wait_for_timeout(250)
                    except Exception:
                        pass
                    # Force-show CSS-hidden DOM so the preview iframe
                    # matches what the scraper will actually extract.
                    # body.get_text() in BS4 ignores CSS visibility,
                    # so any hidden boilerplate (CookieYes preferences
                    # modal, accordion bodies, off-screen menus,
                    # newsletter pop-ups built into the page, etc.)
                    # silently leaks into chunks unless the user adds
                    # a selector for it. Showing them in the preview
                    # is the only way the user can SEE that something
                    # needs excluding. The page will look stylistically
                    # broken (modals stacking, dropdowns open) — that's
                    # the point: surface what the scraper sees.
                    # Selectors we never want to force-show. They're
                    # either rendering-hostile (script/style etc), kept
                    # hidden until interaction (nav/menus), or about to
                    # be stripped anyway by the baseline (CMP roots).
                    # CMP roots especially: cky-overlay is a fullscreen
                    # position:fixed element — force-showing it paints
                    # the red strip overlay across the WHOLE viewport,
                    # making it look like the entire page is being
                    # stripped when only 83 boilerplate elements are.
                    no_force_show_selector = ", ".join(
                        sel for sel in _PREVIEW_BASELINE
                        if not sel.lower() in ("script", "style", "noscript", "iframe")
                    )
                    try:
                        page.evaluate(
                            """(skipSelector) => {
                                // Tags the browser keeps hidden for a
                                // reason — never force-show them.
                                // <style>/<script> would render their
                                // source code as raw page text.
                                // <head>/<link>/<meta>/<template> never
                                // carry visible content. Plus their
                                // descendants.
                                const SKIP_TAGS = new Set([
                                  'HEAD', 'SCRIPT', 'STYLE', 'NOSCRIPT',
                                  'LINK', 'META', 'TEMPLATE', 'IFRAME',
                                ]);
                                function shouldSkip(el) {
                                  if (SKIP_TAGS.has(el.tagName)) return true;
                                  if (el.closest('script,style,noscript,link,meta,template,head,iframe')) return true;
                                  // Nav menus / dropdowns / off-screen
                                  // tooltips — SUPPOSED to be hidden
                                  // until interaction. Force-showing
                                  // them explodes the page layout.
                                  if (el.closest('nav,[role="menu"],[role="menubar"],[role="tooltip"],[role="navigation"]')) return true;
                                  // Baseline-stripped boilerplate (CMP
                                  // roots etc) — already going to be
                                  // tagged + counted by the legend.
                                  // Don't force-show them; cky-overlay
                                  // is fullscreen and would paint over
                                  // the entire page.
                                  if (skipSelector) {
                                    try {
                                      if (el.matches(skipSelector)) return true;
                                      if (el.closest(skipSelector)) return true;
                                    } catch (_) {}
                                  }
                                  return false;
                                }
                                document.querySelectorAll('*').forEach(el => {
                                    try {
                                        if (shouldSkip(el)) return;
                                        const cs = getComputedStyle(el);
                                        if (cs.display === 'none') {
                                            el.style.setProperty('display', 'block', 'important');
                                        }
                                        if (cs.visibility === 'hidden') {
                                            el.style.setProperty('visibility', 'visible', 'important');
                                        }
                                        if (cs.opacity === '0') {
                                            el.style.setProperty('opacity', '1', 'important');
                                        }
                                        if (el.hasAttribute('hidden')) {
                                            el.removeAttribute('hidden');
                                        }
                                    } catch (_) {}
                                });
                            }""",
                            no_force_show_selector,
                        )
                    except Exception:
                        pass
                    html = page.content()
                finally:
                    browser.close()
        else:
            with httpx.Client(
                timeout=req.timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; jBKB-RAG/1.0)"},
            ) as client:
                resp = client.get(req.url)
                resp.raise_for_status()
                html = resp.text
    except Exception as e:
        return PreviewExclusionsResponse(url=req.url, error=f"fetch failed: {e}")

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        return PreviewExclusionsResponse(url=req.url, error=f"parse failed: {e}")

    # ── Inject <base href> so CSS/images/fonts resolve ──────────────
    # If the page already has a <base>, leave it alone.
    try:
        from urllib.parse import urljoin, urlparse
        origin = "{0.scheme}://{0.netloc}".format(urlparse(req.url))
        if soup.head and not soup.head.find("base"):
            base = soup.new_tag("base", href=req.url if req.url.endswith("/") else urljoin(req.url, "./"))
            soup.head.insert(0, base)
        elif not soup.head:
            # No <head>; create one.
            head = soup.new_tag("head")
            head.append(soup.new_tag("base", href=origin + "/"))
            if soup.html:
                soup.html.insert(0, head)
    except Exception:
        pass

    # ── Strip everything that makes the sandboxed iframe complain ──
    # Chrome warns "Blocked script execution in 'about:srcdoc'" for
    # every <script>, <link rel=preload as=script>, and every inline
    # event-handler attribute in a sandbox without allow-scripts.
    # None of these would execute anyway — removing them just quiets
    # the console.
    for s in soup.find_all("script"):
        s.decompose()
    # <link rel=preload as=script>, rel=modulepreload, rel=prefetch
    # (for scripts) — these trigger fetches the sandbox then blocks.
    for link in soup.find_all("link"):
        rel_attr = link.get("rel") or []
        rel_vals = [r.lower() for r in (rel_attr if isinstance(rel_attr, list) else [rel_attr])]
        as_attr = (link.get("as") or "").lower()
        if "modulepreload" in rel_vals:
            link.decompose()
        elif ("preload" in rel_vals or "prefetch" in rel_vals) and as_attr == "script":
            link.decompose()
    # Inline event handlers (onload, onclick, onerror, on*=...) —
    # strip them from every element so the browser doesn't try to run
    # anything when hovering / clicking in the preview.
    for el in soup.find_all(True):
        # iterate a snapshot of attrs so we can mutate the dict
        for attr in list(el.attrs.keys()):
            if attr.lower().startswith("on"):
                del el.attrs[attr]
        # javascript: URLs in href/src
        for url_attr in ("href", "src", "action", "formaction"):
            val = el.attrs.get(url_attr)
            if isinstance(val, str) and val.strip().lower().startswith("javascript:"):
                del el.attrs[url_attr]

    # ── Tag matched elements ────────────────────────────────────────
    user_selectors = [s for s in (req.exclude_selectors or []) if s]
    all_selectors = _PREVIEW_BASELINE + user_selectors

    per_selector: dict[str, int] = {}
    total = 0
    for sel in all_selectors:
        try:
            matches = soup.select(sel)
        except Exception:
            per_selector[sel] = 0
            continue
        per_selector[sel] = len(matches)
        for el in matches:
            if not isinstance(el, Tag):
                continue
            # Don't double-count if another selector already hit this node
            cls = el.get("class") or []
            if "jbkb-ex-hit" not in cls:
                cls = list(cls) + ["jbkb-ex-hit"]
                el["class"] = cls
                total += 1
            # Keep the first selector that matched; if the element was
            # already hit, we still note both in data-jbkb-sel.
            existing = el.get("data-jbkb-sel", "")
            el["data-jbkb-sel"] = (existing + " " + sel).strip() if existing else sel

    # ── Inject <style> + legend ─────────────────────────────────────
    style_tag = soup.new_tag("style")
    style_tag.string = _PREVIEW_STYLE
    if soup.head:
        soup.head.append(style_tag)
    elif soup.html:
        soup.html.insert(0, style_tag)
    else:
        soup.insert(0, style_tag)

    # Legend — only if there's a <body>; otherwise skip to avoid
    # corrupting the document structure.
    if soup.body and total > 0:
        legend = soup.new_tag("div", id="__jbkb_legend")
        legend.append(BeautifulSoup(
            f"<b>{total} element(s) stripped</b>",
            "html.parser",
        ))
        ul = soup.new_tag("ul")
        # Sort by count desc, skip selectors with 0 matches
        for sel, cnt in sorted(per_selector.items(), key=lambda kv: -kv[1]):
            if cnt == 0:
                continue
            li = soup.new_tag("li")
            li.string = f"{sel} ({cnt})"
            ul.append(li)
        legend.append(ul)
        soup.body.append(legend)

    # Inspector mode — inject a <script> at the end of <body> that
    # intercepts clicks and posts candidate selectors to the parent.
    # Only emitted when inspect_mode=true; the parent mounts the iframe
    # with "allow-same-origin allow-scripts" only while inspecting.
    if req.inspect_mode:
        inspector = soup.new_tag("script")
        inspector.string = _INSPECTOR_SCRIPT
        target = soup.body if soup.body else soup
        target.append(inspector)

    return PreviewExclusionsResponse(
        url=req.url,
        html=str(soup),
        matched_count=total,
        per_selector=per_selector,
    )


# ═══════════════════════════════════════════════════════════════════════
# FAQ extraction — ask an LLM for Q&A pairs from raw text
# ═══════════════════════════════════════════════════════════════════════

class FaqExtractRequest(BaseModel):
    """Extract Q&A pairs from a block of text (one rag_sources row's content)."""
    text: str
    language: str = "en"
    company_name: str = "the assistant"
    # Default raised 50 → 200 (2026-05-27). A typical product / brand FAQ
    # page commonly has 50–80 entries; the previous default was silently
    # truncating the long tail. Callers can still override down per call
    # but the default no longer needs to.
    max_items: int = 200
    # Optional: source URL echoed back in each pair for traceability
    source_url: Optional[str] = None


class FaqPair(BaseModel):
    question: str
    answer: str
    # Optional category label derived from the source page's section /
    # heading the Q&A pair lived under. E.g. on a FAQ page organised
    # as "Promotions / Jewelry / Shipping & Returns", a question under
    # the Shipping heading comes back with category="Shipping & Returns".
    # When the source page has no visible grouping, category is None
    # and the downstream faq_items row stays at the operator's manual
    # category (if any). Free-form string — the FAQ Items panel
    # treats category as a label, not an enum.
    category: Optional[str] = None


class FaqExtractResponse(BaseModel):
    pairs: list[FaqPair] = Field(default_factory=list)
    count: int = 0
    source_url: Optional[str] = None
    error: Optional[str] = None


# Tuning constants for FAQ extraction (2026-05-27 — Hey Harper "70 FAQs
# but only 35 extracted" report).
#
# - INPUT_CHAR_CAP: hard cap on input text length, separate from the
#   per-chunk size below. gpt-4o-mini has a 128k input window, so the
#   previous 40k cap was conservative-by-default and dropped half of
#   long FAQ pages.
# - CHUNK_THRESHOLD_CHARS: input length above which we switch to the
#   chunk-and-merge path. Below this, one LLM call is faster and easier
#   to debug. Above this, a single call would have to produce too many
#   output tokens.
# - CHUNK_SIZE_CHARS: target per-chunk input size. Splits on paragraph
#   boundaries to keep each chunk semantically self-contained.
# - FAQ_EXTRACT_MAX_TOKENS: explicit output cap. Default
#   (no max_tokens) falls back to gpt-4o-mini's 4096, which truncated
#   ~35 pairs into the response and made the JSON un-parseable past the
#   cut-off. 8000 comfortably fits 80+ pairs of typical question-answer
#   lengths.
INPUT_CHAR_CAP = 100_000
CHUNK_THRESHOLD_CHARS = 25_000
CHUNK_SIZE_CHARS = 20_000
FAQ_EXTRACT_MAX_TOKENS = 8000


def _faq_extract_system_prompt(language: str, company_name: str, max_items: int) -> str:
    """System prompt used in both the single-shot and per-chunk paths.

    Output language must match the SOURCE TEXT, not `language` — the
    FAQ table downstream is grouped by CBVA and translated by jBSE at
    delivery time. Auto-translating here would (a) lose the operator's
    intended phrasing and (b) make duplicate detection brittle (existing
    PT pairs vs new EN pairs would never match). `language` stays as
    a hint for the rare case where the page mixes languages.
    """
    return (
        "Extract FAQ-style question-and-answer pairs from the text.\n"
        "Rules:\n"
        "  - Output valid JSON: {\"pairs\": [{\"question\": \"...\", \"answer\": \"...\", \"category\": \"...\"}]}\n"
        "  - One pair per distinct topic.\n"
        "  - Only include pairs where both question and answer are clearly supported by the text.\n"
        "  - Skip promotional filler, navigation blocks, and duplicates.\n"
        "  - Category: copy the section heading or group label the pair appears under in the source text (e.g. \"Promotions\", \"Jewelry Care\", \"Shipping & Returns\"). When the page has no visible grouping, omit the field or set it to an empty string.\n"
        "  - Output language: match the SOURCE TEXT exactly. Do NOT translate. This applies to category too.\n"
        f"    (Hint: the page is most likely in '{language}', but if the text is in another language, keep that language.)\n"
        f"  - Speak as {company_name}.\n"
        f"  - At most {max_items} pairs.\n"
        "Return ONLY the JSON. No prose, no markdown fences."
    )


def _parse_faq_json_response(raw: str) -> tuple[list[dict], Optional[str]]:
    """Parse the LLM's JSON response into raw pair dicts.

    Returns (pairs, error). On any failure pairs=[] and error explains.
    Tolerates code fences and leading/trailing prose.
    """
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    lbrace = cleaned.find("{")
    rbrace = cleaned.rfind("}")
    if lbrace >= 0 and rbrace > lbrace:
        cleaned = cleaned[lbrace:rbrace + 1]
    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        return [], f"LLM did not return valid JSON: {e}"
    raw_pairs = parsed.get("pairs") if isinstance(parsed, dict) else None
    if not isinstance(raw_pairs, list):
        return [], "no 'pairs' array in LLM output"
    return raw_pairs, None


def _split_into_chunks(text: str, target_size: int) -> list[str]:
    """Split text into segments of approximately `target_size` chars,
    cutting on paragraph boundaries so each chunk is self-contained.

    Algorithm: greedily accumulate paragraphs until adding the next one
    would exceed target_size, then start a new chunk. Single paragraphs
    longer than target_size go in their own chunk (no mid-paragraph
    splits — those tend to mangle Q/A pairs).
    """
    # Paragraph separator = one or more blank lines.
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for p in paragraphs:
        # +2 for the join separator we'll re-insert.
        if current and current_len + len(p) + 2 > target_size:
            chunks.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += len(p) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _normalize_question_for_dedup(q: str) -> str:
    """Lowercase + collapse internal whitespace — used to dedup pairs
    that span chunk boundaries (the same FAQ can appear in two
    consecutive chunks if it straddles the split)."""
    return " ".join(q.lower().split())


@router.post("/faq-extract", response_model=FaqExtractResponse)
def execute_faq_extract(req: FaqExtractRequest):
    """
    Extract FAQ Q&A pairs from raw text. Stateless: no collection lookup,
    no disk I/O — just text in, pairs out. Used by jBKB's FAQ pipeline to
    turn one rag_sources row (destination='faq_table') into faq_items rows.

    For inputs longer than CHUNK_THRESHOLD_CHARS we split into segments
    on paragraph boundaries and call the LLM per segment, then dedup the
    union by normalized question. This avoids both the input cap AND the
    output-token cap a single big call would hit on long pages.
    """
    from llms.openai_utils import openai_chat_completion

    text = (req.text or "").strip()
    if not text:
        return FaqExtractResponse(pairs=[], count=0, source_url=req.source_url, error="empty text")

    truncated = text[:INPUT_CHAR_CAP]
    system_prompt = _faq_extract_system_prompt(req.language, req.company_name, req.max_items)

    # Decide chunking. Below the threshold = single call (cheaper, easier
    # to debug). Above = chunked path.
    chunks = (
        _split_into_chunks(truncated, CHUNK_SIZE_CHARS)
        if len(truncated) > CHUNK_THRESHOLD_CHARS
        else [truncated]
    )

    # Per-chunk extraction. Dedup by normalized question — same FAQ can
    # legitimately appear in two adjacent chunks when the split happens
    # mid-section. First-seen wins (preserves the answer from the chunk
    # where the FAQ was extracted in full).
    seen: set[str] = set()
    pairs: list[FaqPair] = []
    last_error: Optional[str] = None
    for chunk in chunks:
        try:
            raw = openai_chat_completion(
                system_prompt,
                f"Text:\n{chunk}",
                model="gpt-4o-mini",
                max_tokens=FAQ_EXTRACT_MAX_TOKENS,
            )
        except Exception as e:
            last_error = str(e)
            continue
        raw_pairs, parse_err = _parse_faq_json_response(raw)
        if parse_err:
            last_error = parse_err
            continue
        for p in raw_pairs:
            if len(pairs) >= req.max_items:
                break
            if not isinstance(p, dict):
                continue
            q = (p.get("question") or "").strip()
            a = (p.get("answer") or "").strip()
            if not q or not a:
                continue
            key = _normalize_question_for_dedup(q)
            if key in seen:
                continue
            seen.add(key)
            cat_raw = p.get("category")
            category = cat_raw.strip() if isinstance(cat_raw, str) and cat_raw.strip() else None
            pairs.append(FaqPair(question=q, answer=a, category=category))
        if len(pairs) >= req.max_items:
            break

    # Only return an error when ZERO pairs came out. Partial success
    # (some chunks failed, some succeeded) still returns the successful
    # pairs — better than an all-or-nothing.
    err = last_error if not pairs and last_error else None
    return FaqExtractResponse(pairs=pairs, count=len(pairs), source_url=req.source_url, error=err)


@router.post("/probe-pdf", response_model=ProbePdfResponse)
def execute_probe_pdf(req: ProbePdfRequest):
    """Probe a PDF for its page structure.

    Used by the jBKB upload dialog to:
     1. confirm the PDF is readable before the user commits to a save,
     2. display a page outline (first non-empty line per page) so the
        user can spot chapter boundaries for range-split mode.

    Local paths and http(s) URLs are handled by the same underlying
    reader (pdf_ingestion._open_pdf), so this endpoint doesn't care
    which mode the caller is in.
    """
    from ingestion.pdf_ingestion import read_from_pdf_pages

    try:
        pages = read_from_pdf_pages(req.file_path)
    except FileNotFoundError as e:
        raise HTTPException(404, f"PDF not found: {e}")
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}")

    previews: list[ProbePdfPagePreview] = []
    for p in pages:
        text = p.get("text", "") or ""
        # First non-empty line, collapsed whitespace, clipped to 140
        # chars — enough to show a chapter heading or a first
        # sentence. Anything longer is noise in the outline view.
        first_line = ""
        for raw_line in text.splitlines():
            stripped = " ".join(raw_line.split())
            if stripped:
                first_line = stripped[:140]
                break
        previews.append(ProbePdfPagePreview(page=p["page"], first_line=first_line))

    return ProbePdfResponse(page_count=len(pages), page_previews=previews)


# ═══════════════════════════════════════════════════════════════════════
# Collection-list extractor — parses a "collection / category / theme"
# page (e.g. /collections/bracelets on a Shopify store) into a structured
# list of products. Emits ONE chunk per page shaped for retrieval against
# questions like "what bracelets do you sell?" — the chunk's embedded
# text reads like a tidy product list, and the chunk's payload carries
# the same data as JSON so downstream consumers (UI grid, faceted browse)
# can use it directly.
#
# Strategy order — first one that yields a non-empty product list wins:
#   1. JSON-LD ItemList of Product (Shopify, most stores with SEO theme)
#   2. JSON-LD bare Product entries on the page (still semantic)
#   3. Microdata itemtype=schema.org/Product
#   4. Heuristic DOM walk: elements matching common product-card classes
#      with both a link AND a price text
#
# Pagination: V1 reads page 1 only. Many collections have ≤30 products
# on the first page (Shopify default); long collections will trail off.
# Adding pagination is a sitemap-level concern (operator can already
# pre-scan all paginated URLs into the sitemap's url list).
# ═══════════════════════════════════════════════════════════════════════

class ExtractCollectionRequest(BaseModel):
    """Single-URL collection-page extraction."""
    url: str
    timeout: float = 15.0
    # exclude_selectors passed through for parity with /fetch — we
    # strip these elements before heuristic DOM walking so the same
    # "remove cookie banner" config that the operator tuned for
    # generic strip still applies. JSON-LD parsing ignores them.
    exclude_selectors: list[str] = Field(default_factory=list)


class ExtractedProduct(BaseModel):
    name: str
    price: str | None = None
    price_currency: str | None = None
    url: str | None = None       # absolute URL to the product detail page
    image: str | None = None


class ExtractCollectionResponse(BaseModel):
    collection_name: str
    products: list[ExtractedProduct] = Field(default_factory=list)
    # Embed-friendly summary text. The Node API stores this as the
    # chunk's text + as one of the payload fields; the structured
    # `products` array is also placed in the payload.
    summary_text: str = ""
    # Diagnostic string for the operator — which strategy produced
    # the list, or why it came back empty.
    strategy: str = ""
    error: str | None = None


def _absolute_url(href: str | None, base: str) -> str | None:
    if not href:
        return None
    import urllib.parse
    try:
        return urllib.parse.urljoin(base, href)
    except Exception:
        return None


def _extract_collection_jsonld(soup, base_url: str) -> tuple[list[ExtractedProduct], str | None]:
    """Returns (products, collection_name) parsed from JSON-LD blocks.
    Empty list on no-match. Collection name comes from a sibling
    @type=CollectionPage or BreadcrumbList tail when present."""
    import json
    products: list[ExtractedProduct] = []
    collection_name: str | None = None
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        # JSON-LD entries can be a single dict or a list of dicts;
        # @graph wraps nested entries on some sites.
        def walk(obj):
            nonlocal collection_name
            if isinstance(obj, list):
                for item in obj:
                    walk(item)
                return
            if not isinstance(obj, dict):
                return
            t = obj.get("@type")
            if isinstance(t, list):
                t_set = set(t)
            else:
                t_set = {t} if t else set()
            # ItemList → walk itemListElement
            if "ItemList" in t_set or "CollectionPage" in t_set:
                if not collection_name and obj.get("name"):
                    collection_name = str(obj["name"]).strip()
                items = obj.get("itemListElement") or []
                for el in items if isinstance(items, list) else []:
                    if isinstance(el, dict):
                        # ListItem wraps a Product in `item`
                        if el.get("@type") == "ListItem" and isinstance(el.get("item"), dict):
                            walk(el["item"])
                        else:
                            walk(el)
            # Product → extract
            if "Product" in t_set:
                name = obj.get("name")
                if not name:
                    return
                offers = obj.get("offers")
                price = None
                currency = None
                if isinstance(offers, dict):
                    price = offers.get("price") or offers.get("lowPrice")
                    currency = offers.get("priceCurrency")
                elif isinstance(offers, list) and offers:
                    first = offers[0]
                    if isinstance(first, dict):
                        price = first.get("price") or first.get("lowPrice")
                        currency = first.get("priceCurrency")
                url = obj.get("url") or obj.get("@id")
                image = obj.get("image")
                if isinstance(image, list):
                    image = image[0] if image else None
                if isinstance(image, dict):
                    image = image.get("url")
                products.append(ExtractedProduct(
                    name=str(name).strip(),
                    price=str(price).strip() if price is not None else None,
                    price_currency=str(currency).strip() if currency else None,
                    url=_absolute_url(url, base_url) if isinstance(url, str) else None,
                    image=_absolute_url(image, base_url) if isinstance(image, str) else None,
                ))
            # @graph wrapper
            graph = obj.get("@graph")
            if isinstance(graph, list):
                walk(graph)
        walk(data)
    # Dedupe products by (name, url) — JSON-LD can list the same
    # product under both ItemList and standalone Product.
    seen: set[tuple[str, str | None]] = set()
    deduped: list[ExtractedProduct] = []
    for p in products:
        key = (p.name, p.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped, collection_name


def _extract_collection_heuristic(soup, base_url: str) -> list[ExtractedProduct]:
    """Fallback: look for elements matching common product-card class
    patterns. Name resolution tries multiple paths to handle modern
    headless storefronts (Hydrogen / Svelte / Next-on-Shopify) that
    skip semantic heading tags for product card titles."""
    import re as _re
    import urllib.parse
    products: list[ExtractedProduct] = []
    seen_urls: set[str] = set()
    candidates = soup.select(
        '[class*="product-card"], [class*="product-item"], '
        '[class*="ProductCard"], [class*="ProductItem"], '
        '[class*="product-tile"], [class*="ProductTile"], '
        'li[class*="product"], article[class*="product"]'
    )
    price_re = _re.compile(
        r'(?:€|£|\$|USD|EUR|GBP|kr|CHF|SEK|NOK|DKK)\s*\d[\d.,\s]*'
        r'|\d[\d.,]*\s*(?:€|£|\$|USD|EUR|GBP|kr|CHF|SEK|NOK|DKK)'
    )
    for el in candidates:
        # Prefer the first link to /products/* — that's the canonical
        # product URL on every Shopify-shaped storefront. Falls back
        # to any href when no product-shaped link exists.
        prod_links = el.find_all("a", href=True)
        if not prod_links:
            continue
        canonical_link = None
        for a in prod_links:
            if "/products/" in (a.get("href") or ""):
                canonical_link = a
                break
        link = canonical_link or prod_links[0]
        url = _absolute_url(link.get("href"), base_url)
        if not url or url in seen_urls:
            continue

        # Name resolution, in priority order:
        # 1. Heading element (h1-h4) inside the card
        # 2. Any element matching common product-title class patterns
        # 3. The link with the LONGEST non-empty text among links
        #    pointing to the same product URL (image link is usually
        #    empty; the second/third link carries the visible name)
        # 4. The URL slug, prettified
        name = ""
        heading = el.find(["h1", "h2", "h3", "h4"])
        if heading:
            name = " ".join(heading.get_text(separator=" ", strip=True).split())
        if not name:
            title_el = el.select_one(
                '[class*="product-title"], [class*="ProductTitle"], '
                '[class*="product-name"], [class*="ProductName"], '
                '[class*="card-title"], [class*="CardTitle"]'
            )
            if title_el:
                name = " ".join(title_el.get_text(separator=" ", strip=True).split())
        if not name:
            same_url_links = [a for a in prod_links if _absolute_url(a.get("href"), base_url) == url]
            best_text = ""
            for a in same_url_links:
                txt = " ".join(a.get_text(separator=" ", strip=True).split())
                if len(txt) > len(best_text):
                    best_text = txt
            name = best_text
        if not name:
            # Slug fallback: /products/aurora-silver-bracelet → "Aurora Silver Bracelet"
            try:
                slug = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
                name = slug.replace("-", " ").replace("_", " ").strip().title()
            except Exception:
                name = ""
        if not name:
            continue
        name = name[:200]

        # Price = first price-shaped match anywhere in the card text
        card_text = " ".join(el.get_text(separator=" ", strip=True).split())
        m = price_re.search(card_text)
        price = m.group(0).strip() if m else None

        # Image
        img = el.find("img")
        image_src = img.get("src") if img else None
        if not image_src and img:
            image_src = img.get("data-src") or img.get("srcset", "").split()[0] if img.get("srcset") else None
        image = _absolute_url(image_src, base_url) if image_src else None

        products.append(ExtractedProduct(
            name=name,
            price=price,
            url=url,
            image=image,
        ))
        seen_urls.add(url)
    return products


def _collection_name_from_dom(soup, url: str) -> str:
    """Pick a sensible name for the collection from the page itself.
    Priority: <h1> → <title> minus site name → URL slug."""
    h1 = soup.find("h1")
    if h1:
        txt = " ".join(h1.get_text(separator=" ", strip=True).split())
        if txt:
            return txt[:200]
    title = soup.find("title")
    if title:
        txt = " ".join(title.get_text(separator=" ", strip=True).split())
        # Strip common "| Site Name" or "— Site Name" suffixes
        for sep in [" | ", " — ", " - ", " · "]:
            if sep in txt:
                txt = txt.split(sep, 1)[0]
                break
        if txt:
            return txt[:200]
    # Slug fallback
    import urllib.parse
    try:
        path = urllib.parse.urlparse(url).path.rstrip("/")
        slug = path.rsplit("/", 1)[-1]
        return slug.replace("-", " ").replace("_", " ").title() or url
    except Exception:
        return url


def _playwright_render_html(url: str, exclude_selectors: list[str], timeout_sec: float) -> str:
    """Fall back to Playwright when httpx + parse returns 0 products —
    almost always means the page is a JS-rendered storefront (Hydrogen,
    Next.js + Storefront API, etc.) whose product data only appears
    after the client-side framework hydrates. Reuses the operator's
    exclude_selectors for parity with the httpx path.

    Returns the rendered HTML after networkidle. Raises on failure;
    the caller catches and reports the strategy as 'playwright_failed'."""
    from playwright.sync_api import sync_playwright
    timeout_ms = int(timeout_sec * 1000)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            ctx = browser.new_context(user_agent="Mozilla/5.0 (compatible; jBKB-RAG/1.0)")
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            # Some storefronts lazy-load product grids on scroll. A
            # single scroll-to-bottom + brief wait is cheap and
            # catches most lazy-grid cases without per-site tuning.
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(800)
            except Exception:
                pass
            # Strip selectors via DOM removal so the returned HTML
            # matches what the operator told us to ignore. Mirrors
            # the soup.decompose loop on the httpx path.
            if exclude_selectors:
                page.evaluate(
                    """(sels) => {
                        for (const sel of sels) {
                            try {
                                for (const el of document.querySelectorAll(sel)) el.remove();
                            } catch { /* invalid selector — skip */ }
                        }
                    }""",
                    [s for s in exclude_selectors if isinstance(s, str) and s.strip()],
                )
            html = page.content()
            return html
        finally:
            browser.close()


@router.post("/extract-collection-list", response_model=ExtractCollectionResponse)
def execute_extract_collection_list(req: ExtractCollectionRequest):
    """Parse a collection page into a structured product list."""
    try:
        import httpx
        from bs4 import BeautifulSoup
    except Exception as e:
        raise HTTPException(500, f"Missing deps: {e}")

    def _parse(html: str) -> tuple[list[ExtractedProduct], str | None, str]:
        """Run exclude_selectors + JSON-LD + heuristic against the
        given HTML. Returns (products, jsonld_name, strategy_label)."""
        s = BeautifulSoup(html, "html.parser")
        for sel in req.exclude_selectors:
            if not isinstance(sel, str) or not sel.strip():
                continue
            try:
                for el in s.select(sel):
                    el.decompose()
            except Exception:
                pass
        prods, name = _extract_collection_jsonld(s, req.url)
        if prods:
            return prods, name, "jsonld"
        prods = _extract_collection_heuristic(s, req.url)
        if prods:
            return prods, name, "heuristic"
        return [], name, ""

    # ── Stage 1: httpx (cheap, fast, works for SSR / static stores) ──
    try:
        with httpx.Client(
            timeout=req.timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; jBKB-RAG/1.0)"},
        ) as client:
            resp = client.get(req.url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        return ExtractCollectionResponse(
            collection_name="",
            products=[],
            summary_text="",
            strategy="fetch_failed",
            error=f"{e}",
        )

    products, jsonld_name, strategy = _parse(html)

    # ── Stage 2: Playwright fallback (when httpx parse yields nothing) ──
    # Almost always a JS-rendered storefront. Pay the Playwright cost
    # once per page; if it ALSO yields nothing, the page genuinely
    # isn't a collection-shaped page (or uses non-standard markup
    # neither JSON-LD nor heuristic detect).
    if not products:
        try:
            rendered = _playwright_render_html(req.url, req.exclude_selectors, req.timeout * 2)
            products, jsonld_name_pw, strategy_pw = _parse(rendered)
            if products:
                jsonld_name = jsonld_name or jsonld_name_pw
                strategy = f"playwright_{strategy_pw}"
            else:
                # Capture the rendered HTML's collection name for the
                # error response even on no-products.
                s = BeautifulSoup(rendered, "html.parser")
                pw_name = _collection_name_from_dom(s, req.url)
                jsonld_name = jsonld_name or pw_name
        except Exception as e:
            return ExtractCollectionResponse(
                collection_name=_collection_name_from_dom(BeautifulSoup(html, "html.parser"), req.url),
                products=[],
                summary_text="",
                strategy="playwright_failed",
                error=f"httpx parse found 0 products; Playwright fallback failed: {e}",
            )

    collection_name = jsonld_name or _collection_name_from_dom(BeautifulSoup(html, "html.parser"), req.url)

    if not products:
        return ExtractCollectionResponse(
            collection_name=collection_name,
            products=[],
            summary_text="",
            strategy="no_products_found",
            error="Could not find product entries via JSON-LD or heuristic DOM walk, even after Playwright rendering. Page may not be a collection page, or its template uses non-standard markup.",
        )

    # Build the embedded summary text — collection name, product
    # count, then one line per product. Format optimised for the
    # LLM-reading-it-back case AND for embedding the right keywords.
    lines = [f"{collection_name} — {len(products)} product{'s' if len(products) != 1 else ''}:"]
    for p in products:
        bits = [p.name]
        if p.price:
            currency = p.price_currency or ""
            bits.append(f"{currency}{p.price}".strip())
        if p.url:
            bits.append(p.url)
        lines.append(" — ".join(bits))
    summary_text = "\n".join(lines)

    return ExtractCollectionResponse(
        collection_name=collection_name,
        products=products,
        summary_text=summary_text,
        strategy=strategy,
    )


# ═══════════════════════════════════════════════════════════════════════
# Product-detail extractor — parses a single product page (e.g.
# /products/aurora-silver-bracelet) into a structured payload. Sister
# to extract-collection-list; same strategy ladder (JSON-LD → heuristic
# → Playwright fallback). One chunk per page; embedded text is a clean
# prose summary, payload carries the structured fields so downstream
# consumers (Session Engine, faceted browse) can use them directly.
# ═══════════════════════════════════════════════════════════════════════

class ExtractProductRequest(BaseModel):
    url: str
    timeout: float = 15.0
    exclude_selectors: list[str] = Field(default_factory=list)


class ExtractedAttribute(BaseModel):
    """Free-form key/value pair extracted from a product page —
    materials, dimensions, country of origin, etc. Both fields kept as
    strings since heuristic extraction can't reliably type them."""
    key: str
    value: str


class ExtractProductResponse(BaseModel):
    name: str
    handle: str | None = None        # URL slug for cross-linking with collection chunks
    price: str | None = None
    price_currency: str | None = None
    description: str | None = None
    image: str | None = None
    sku: str | None = None
    brand: str | None = None
    in_stock: bool | None = None
    availability: str | None = None  # raw availability string (e.g. "InStock", "OutOfStock", "Few left")
    attributes: list[ExtractedAttribute] = Field(default_factory=list)
    url: str
    summary_text: str = ""
    strategy: str = ""
    error: str | None = None


def _extract_product_jsonld(soup, base_url: str) -> dict | None:
    """Pull the first Product JSON-LD block on the page. Returns a
    dict of extracted fields (or None when nothing usable found).
    Mirrors the @type / @graph traversal logic from the collection
    extractor; reuses the offer-walk for price/currency/availability."""
    import json
    found: dict = {}

    def walk(obj):
        if isinstance(obj, list):
            for it in obj:
                walk(it)
            return
        if not isinstance(obj, dict):
            return
        t = obj.get("@type")
        t_set = set(t) if isinstance(t, list) else ({t} if t else set())
        if "Product" in t_set and not found:
            name = obj.get("name")
            if not name:
                return
            found["name"] = str(name).strip()
            desc = obj.get("description")
            if isinstance(desc, str) and desc.strip():
                found["description"] = " ".join(desc.split())[:800]
            img = obj.get("image")
            if isinstance(img, list) and img:
                img = img[0]
            if isinstance(img, dict):
                img = img.get("url")
            if isinstance(img, str):
                found["image"] = _absolute_url(img, base_url)
            sku = obj.get("sku")
            if sku:
                found["sku"] = str(sku).strip()
            brand = obj.get("brand")
            if isinstance(brand, dict):
                brand = brand.get("name")
            if isinstance(brand, str):
                found["brand"] = brand.strip()
            offers = obj.get("offers")
            offer_one: dict | None = None
            if isinstance(offers, dict):
                offer_one = offers
            elif isinstance(offers, list) and offers and isinstance(offers[0], dict):
                offer_one = offers[0]
            if offer_one:
                price = offer_one.get("price") or offer_one.get("lowPrice")
                if price is not None:
                    found["price"] = str(price).strip()
                cur = offer_one.get("priceCurrency")
                if cur:
                    found["price_currency"] = str(cur).strip()
                avail = offer_one.get("availability")
                if avail:
                    avail_str = str(avail)
                    # schema.org URLs like "https://schema.org/InStock"
                    if "/" in avail_str:
                        avail_str = avail_str.rsplit("/", 1)[-1]
                    found["availability"] = avail_str
                    in_stock_map = {
                        "InStock": True, "instock": True, "in_stock": True,
                        "OutOfStock": False, "outofstock": False, "out_of_stock": False,
                        "SoldOut": False, "Discontinued": False,
                    }
                    if avail_str in in_stock_map:
                        found["in_stock"] = in_stock_map[avail_str]
        graph = obj.get("@graph") if isinstance(obj, dict) else None
        if isinstance(graph, list):
            walk(graph)

    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        walk(data)
        if found.get("name"):
            break
    return found if found.get("name") else None


def _extract_product_heuristic(soup, base_url: str) -> dict:
    """Fallback when JSON-LD missing. Walks the DOM for likely product
    fields by class / itemprop / position. Best-effort — won't get
    structured attributes, but should pull name + price + description."""
    import re as _re
    out: dict = {}
    # Name: h1 first, fall back to <title> minus suffix
    h1 = soup.find("h1")
    if h1:
        name = " ".join(h1.get_text(separator=" ", strip=True).split())
        if name:
            out["name"] = name[:200]
    if "name" not in out:
        title = soup.find("title")
        if title:
            t = " ".join(title.get_text(separator=" ", strip=True).split())
            for sep in [" | ", " — ", " - ", " · "]:
                if sep in t:
                    t = t.split(sep, 1)[0]
                    break
            if t:
                out["name"] = t[:200]

    # Price: itemprop first, then common class patterns
    price_el = soup.select_one(
        '[itemprop="price"], [class*="product-price"], [class*="ProductPrice"], '
        '[class*="price__"], [data-product-price]'
    )
    price_re = _re.compile(
        r'(?:€|£|\$|USD|EUR|GBP|kr|CHF|SEK|NOK|DKK)\s*\d[\d.,]*'
        r'|\d[\d.,]*\s*(?:€|£|\$|USD|EUR|GBP|kr|CHF|SEK|NOK|DKK)'
    )
    if price_el:
        # Prefer the content attribute (often the raw number) over the rendered text
        price = price_el.get("content") or price_el.get("data-price")
        if price:
            out["price"] = str(price).strip()
        else:
            txt = " ".join(price_el.get_text(separator=" ", strip=True).split())
            m = price_re.search(txt)
            if m:
                out["price"] = m.group(0).strip()
    # If no price element matched, scan the whole page body once
    if "price" not in out:
        body_text = " ".join(soup.get_text(separator=" ", strip=True).split())
        m = price_re.search(body_text)
        if m:
            out["price"] = m.group(0).strip()

    # Description
    desc_el = soup.select_one(
        '[itemprop="description"], [class*="product-description"], '
        '[class*="ProductDescription"], [class*="rte"], [class*="product-content"]'
    )
    if desc_el:
        desc = " ".join(desc_el.get_text(separator=" ", strip=True).split())
        if desc:
            out["description"] = desc[:800]

    # Image
    img_el = soup.select_one(
        '[class*="product-image"] img, [class*="ProductImage"] img, '
        'picture img, [itemprop="image"]'
    )
    if img_el:
        src = img_el.get("src") or img_el.get("data-src")
        if not src and img_el.get("srcset"):
            src = img_el["srcset"].split(",")[0].strip().split()[0]
        if src:
            out["image"] = _absolute_url(src, base_url)
    return out


def _handle_from_url(url: str) -> str | None:
    """Extract the URL slug — last non-empty path segment. Used for
    cross-linking product chunks with their parent collection chunks
    (collection chunks include /products/<handle> URLs)."""
    import urllib.parse
    try:
        path = urllib.parse.urlparse(url).path.rstrip("/")
        slug = path.rsplit("/", 1)[-1]
        return slug or None
    except Exception:
        return None


@router.post("/extract-product-details", response_model=ExtractProductResponse)
def execute_extract_product_details(req: ExtractProductRequest):
    """Parse a product page into a structured payload."""
    try:
        import httpx
        from bs4 import BeautifulSoup
    except Exception as e:
        raise HTTPException(500, f"Missing deps: {e}")

    def _parse(html: str) -> tuple[dict, str]:
        """Returns (fields, strategy_label). Empty fields when neither
        JSON-LD nor heuristic yields a product name."""
        s = BeautifulSoup(html, "html.parser")
        for sel in req.exclude_selectors:
            if not isinstance(sel, str) or not sel.strip():
                continue
            try:
                for el in s.select(sel):
                    el.decompose()
            except Exception:
                pass
        ld = _extract_product_jsonld(s, req.url)
        if ld and ld.get("name"):
            return ld, "jsonld"
        h = _extract_product_heuristic(s, req.url)
        if h.get("name"):
            return h, "heuristic"
        return {}, ""

    # Stage 1: httpx
    try:
        with httpx.Client(
            timeout=req.timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; jBKB-RAG/1.0)"},
        ) as client:
            resp = client.get(req.url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        return ExtractProductResponse(
            name="",
            url=req.url,
            summary_text="",
            strategy="fetch_failed",
            error=f"{e}",
        )

    fields, strategy = _parse(html)

    # Stage 2: Playwright fallback for JS-rendered storefronts
    if not fields.get("name"):
        try:
            rendered = _playwright_render_html(req.url, req.exclude_selectors, req.timeout * 2)
            fields, strategy_pw = _parse(rendered)
            if fields.get("name"):
                strategy = f"playwright_{strategy_pw}"
        except Exception as e:
            return ExtractProductResponse(
                name="",
                url=req.url,
                summary_text="",
                strategy="playwright_failed",
                error=f"httpx parse found no product; Playwright fallback failed: {e}",
            )

    if not fields.get("name"):
        return ExtractProductResponse(
            name="",
            url=req.url,
            summary_text="",
            strategy="no_product_found",
            error="Could not extract product name via JSON-LD or heuristic DOM walk, even after Playwright rendering. Page may not be a product page or its template uses non-standard markup.",
        )

    # Build embed-friendly summary text. Format: name + price line,
    # then description, then any attributes joined as "key: value" lines.
    # Optimised so a question like "what does Brazil Fan cost" matches
    # against the price line and "describe Aurora Silver" matches the
    # description block.
    lines: list[str] = []
    price_bit = ""
    if fields.get("price"):
        cur = fields.get("price_currency") or ""
        price_bit = f" — {cur}{fields['price']}".strip()
    lines.append(f"{fields['name']}{price_bit}")
    if fields.get("brand"):
        lines.append(f"Brand: {fields['brand']}")
    if fields.get("availability"):
        lines.append(f"Availability: {fields['availability']}")
    if fields.get("sku"):
        lines.append(f"SKU: {fields['sku']}")
    if fields.get("description"):
        lines.append("")
        lines.append(fields["description"])
    summary_text = "\n".join(lines)

    return ExtractProductResponse(
        name=fields["name"],
        handle=_handle_from_url(req.url),
        price=fields.get("price"),
        price_currency=fields.get("price_currency"),
        description=fields.get("description"),
        image=fields.get("image"),
        sku=fields.get("sku"),
        brand=fields.get("brand"),
        in_stock=fields.get("in_stock"),
        availability=fields.get("availability"),
        attributes=[],  # Future: parse schema.org additionalProperty or definition lists
        url=req.url,
        summary_text=summary_text,
        strategy=strategy,
    )
