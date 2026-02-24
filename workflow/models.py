"""
Workflow steps and state for the RAG pipeline.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import os


class Step(str, Enum):
    CREATE_COLLECTION = "create_collection"
    ADD_SOURCE = "add_source"
    FETCH = "fetch"
    CLEAN = "clean"
    TRANSLATE_AND_CLEAN = "translate_and_clean"
    CHUNK = "chunk"
    GROUP = "group"
    PUSH_TO_QDRANT = "push_to_qdrant"
    TEST_QA = "test_qa"


@dataclass
class ChunkingConfig:
    """Options for chunking step."""
    batch_size: int = 10_000
    overlap_size: int = 100
    use_proposition_chunking: bool = False  # True = LLM propositions, False = simple split
    simple_chunk_size: int = 1000
    simple_chunk_overlap: int = 200
    use_hierarchical_chunking: bool = False  # True = parent/child chunks (best quality, free)
    hierarchical_parent_size: int = 2000     # large parent context window
    hierarchical_parent_overlap: int = 200
    hierarchical_child_size: int = 400       # small child for retrieval
    hierarchical_child_overlap: int = 50


@dataclass
class WorkflowState:
    """State passed through the RAG workflow."""
    # Collection
    collection_name: Optional[str] = None
    collection_type: str = "scs"  # scs | group | group_sameSource
    collection_object: Any = None  # SCS_Collection or GroupCollection when open

    # Source
    source_type: Optional[str] = None  # pdf | url | txt | csv
    source_config: Optional[dict] = None  # path, url, scraper_name, etc.

    # Fetched / cleaned text
    raw_text: Optional[str] = None
    cleaned_text: Optional[str] = None

    # Chunks (list of strings)
    chunks: list = field(default_factory=list)
    source_label: Optional[str] = None  # e.g. filename or site name for payload

    # Persistence
    completed_steps: list = field(default_factory=list)  # steps already done
    save_path: Optional[str] = None  # path to .rag_state.json on disk

    # Chunking options
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Grouping (optional)
    grouping_enabled: bool = False
    group_collection_name: Optional[str] = None
    group_collection_object: Any = None

    # Collection-level metadata (generated after chunking, used by HIRS for RAG routing)
    collection_metadata: Optional[dict] = None  # topics, keywords, description, language, doc_type

    # Tracker reference (set by runner; not serialized)
    tracker: Any = None

    def to_dict(self) -> dict:
        """For API responses — truncates large fields for readability."""
        return {
            "collection_name": self.collection_name,
            "collection_type": self.collection_type,
            "source_type": self.source_type,
            "source_config": self.source_config,
            "raw_text": self.raw_text[:5000] + "..." if self.raw_text and len(self.raw_text) > 5000 else self.raw_text,
            "cleaned_text": self.cleaned_text[:5000] + "..." if self.cleaned_text and len(self.cleaned_text) > 5000 else self.cleaned_text,
            "chunks_count": len(self.chunks),
            "source_label": self.source_label,
            "chunking_config": {
                "batch_size": self.chunking_config.batch_size,
                "overlap_size": self.chunking_config.overlap_size,
                "use_proposition_chunking": self.chunking_config.use_proposition_chunking,
                "use_hierarchical_chunking": self.chunking_config.use_hierarchical_chunking,
                "hierarchical_parent_size": self.chunking_config.hierarchical_parent_size,
                "hierarchical_parent_overlap": self.chunking_config.hierarchical_parent_overlap,
                "hierarchical_child_size": self.chunking_config.hierarchical_child_size,
                "hierarchical_child_overlap": self.chunking_config.hierarchical_child_overlap,
            },
            "grouping_enabled": self.grouping_enabled,
            "group_collection_name": self.group_collection_name,
            "completed_steps": self.completed_steps,
            "save_path": self.save_path,
            "collection_metadata": self.collection_metadata,
        }

    def to_full_dict(self) -> dict:
        """Full serialization including all text and chunks — for saving to disk."""
        return {
            "collection_name": self.collection_name,
            "collection_type": self.collection_type,
            "source_type": self.source_type,
            "source_config": self.source_config,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "chunks": self.chunks,
            "source_label": self.source_label,
            "chunking_config": {
                "batch_size": self.chunking_config.batch_size,
                "overlap_size": self.chunking_config.overlap_size,
                "use_proposition_chunking": self.chunking_config.use_proposition_chunking,
                "simple_chunk_size": self.chunking_config.simple_chunk_size,
                "simple_chunk_overlap": self.chunking_config.simple_chunk_overlap,
                "use_hierarchical_chunking": self.chunking_config.use_hierarchical_chunking,
                "hierarchical_parent_size": self.chunking_config.hierarchical_parent_size,
                "hierarchical_parent_overlap": self.chunking_config.hierarchical_parent_overlap,
                "hierarchical_child_size": self.chunking_config.hierarchical_child_size,
                "hierarchical_child_overlap": self.chunking_config.hierarchical_child_overlap,
            },
            "grouping_enabled": self.grouping_enabled,
            "group_collection_name": self.group_collection_name,
            "completed_steps": self.completed_steps,
            "collection_metadata": self.collection_metadata,
        }

    def get_save_path(self) -> Optional[str]:
        """Derive .rag_state.json path from the source PDF/file path."""
        if self.save_path:
            return self.save_path
        path = (self.source_config or {}).get("path") or (self.source_config or {}).get("pdf_path")
        if path:
            base = os.path.splitext(path)[0]
            return base + ".rag_state.json"
        return None

    def save_to_disk(self) -> Optional[str]:
        """Save full state to disk atomically. Returns the path written, or None on failure."""
        p = self.get_save_path()
        if not p:
            return None
        tmp = p + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_full_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)  # atomic rename — never leaves a partial/empty file
            self.save_path = p
            return p
        except Exception as e:
            print(f"[state] Failed to save state to {p}: {e}")
            try:
                os.remove(tmp)
            except Exception:
                pass
            return None

    @classmethod
    def load_from_disk(cls, path: str, tracker: Any = None) -> "WorkflowState":
        """Load state from a .rag_state.json file."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        state = cls.from_dict(d)
        state.save_path = path
        state.tracker = tracker
        return state

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowState":
        """Build state from dict; tracker and collection_object left None."""
        cfg = d.get("chunking_config") or {}
        chunking_config = ChunkingConfig(
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
        return cls(
            collection_name=d.get("collection_name"),
            collection_type=d.get("collection_type", "scs"),
            source_type=d.get("source_type"),
            source_config=d.get("source_config"),
            raw_text=d.get("raw_text"),
            cleaned_text=d.get("cleaned_text"),
            chunks=d.get("chunks", []),
            source_label=d.get("source_label"),
            chunking_config=chunking_config,
            grouping_enabled=d.get("grouping_enabled", False),
            group_collection_name=d.get("group_collection_name"),
            completed_steps=d.get("completed_steps", []),
            collection_metadata=d.get("collection_metadata"),
        )

    def get_text_for_chunking(self) -> str:
        """Return the text to use for chunking (cleaned or raw)."""
        return self.cleaned_text if self.cleaned_text else (self.raw_text or "")
