"""
Workflow steps and state for the RAG pipeline.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Step(str, Enum):
    CREATE_COLLECTION = "create_collection"
    ADD_SOURCE = "add_source"
    FETCH = "fetch"
    CLEAN = "clean"
    CHUNK = "chunk"
    GROUP = "group"
    PUSH_TO_QDRANT = "push_to_qdrant"
    TEST_QA = "test_qa"


@dataclass
class ChunkingConfig:
    """Options for chunking step."""
    batch_size: int = 10_000
    overlap_size: int = 100
    use_proposition_chunking: bool = True  # True = LLM propositions, False = simple split
    simple_chunk_size: int = 1000
    simple_chunk_overlap: int = 200


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

    # Chunking options
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Grouping (optional)
    grouping_enabled: bool = False
    group_collection_name: Optional[str] = None
    group_collection_object: Any = None

    # Tracker reference (set by runner; not serialized)
    tracker: Any = None

    def to_dict(self) -> dict:
        """For API/serialization; excludes tracker and collection objects."""
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
            },
            "grouping_enabled": self.grouping_enabled,
            "group_collection_name": self.group_collection_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowState":
        """Build state from API payload; tracker and collection_object left None."""
        cfg = d.get("chunking_config") or {}
        chunking_config = ChunkingConfig(
            batch_size=cfg.get("batch_size", 10000),
            overlap_size=cfg.get("overlap_size", 100),
            use_proposition_chunking=cfg.get("use_proposition_chunking", True),
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
        )

    def get_text_for_chunking(self) -> str:
        """Return the text to use for chunking (cleaned or raw)."""
        return self.cleaned_text if self.cleaned_text else (self.raw_text or "")
