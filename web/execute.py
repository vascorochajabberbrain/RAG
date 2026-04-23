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
    # Text with ONLY the baseline strip applied (script/style/noscript/
    # iframe). No user exclude_selectors. Same shape as raw_text. Used
    # by jBKB to populate rag_sources.content_raw and compute a stable
    # content hash for change detection. Falls back to raw_text when
    # the scraper doesn't produce a distinct baseline (structured mode,
    # sitemap / crawl modes that haven't been upgraded).
    baseline_text: str = ""
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

    return FetchResponse(
        raw_text=state.raw_text or "",
        baseline_text=baseline_text,
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
    state.keywords = req.keywords or []

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
    # selector -> {text_hash: pages_with_that_text}
    text_hash_pages: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

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
            text_hash_pages[sel][h] += 1

        for s in page_hits:
            seen_counts[s] += 1

    # Finalise Signal C: promote any selector whose text hash was seen
    # identically on ≥ 2 pages. When the selector was already flagged by
    # A or B, we keep the original reason; the text-repeat fact gets
    # appended as an upgrade note.
    if len(htmls) >= 2:
        for sel, hash_counts in text_hash_pages.items():
            top = max(hash_counts.values()) if hash_counts else 0
            if top < 2:
                continue
            # Promote (if not already) and upgrade the reason
            seen_counts[sel] = max(seen_counts[sel], top)
            existing = reasons.get(sel, "")
            addendum = f"identical text on {top}/{len(htmls)} pages"
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


_PREVIEW_BASELINE = ["script", "style", "noscript", "iframe"]

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
                    # Give async widgets (cookie banner, analytics popups,
                    # Elementor lazy sections) 1.5s to render. Long enough
                    # for most sites, not so long the preview crawls.
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
