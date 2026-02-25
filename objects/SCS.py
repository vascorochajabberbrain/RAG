import hashlib
from datetime import datetime, timezone

from objects.Item import Item


class SCS(Item):

    """----------------------Constructors----------------------"""
    def __init__(self, scs, source=None, source_url=None, content_hash=None, scraped_at=None):
        super().__init__(source)
        self.scs = scs
        self.source_url = source_url
        self.content_hash = content_hash or hashlib.sha256(scs.encode("utf-8")).hexdigest()
        self.scraped_at = scraped_at or datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_payload(cls, payload):
        scs = payload["text"]
        source = payload.get("source", None)
        source_url = payload.get("source_url", None)
        content_hash = payload.get("content_hash", None)
        scraped_at = payload.get("scraped_at", None)
        return cls(scs, source, source_url=source_url, content_hash=content_hash, scraped_at=scraped_at)
    
    """------------------------Dunders------------------------"""

    def __str__(self):
        return f"{self.scs}\n   Source: {self.source}" if self.source else self.scs

    """--------------------Public Methods----------------------"""
    """-----------Sentence related methods-----------"""

    def get_sentence(self):
        return self.scs
    
    
    """---------Qdrant related methods-----------"""
    def to_payload(self, index=None):
        payload = {"text": self.scs}
        if self.source is not None:
            payload["source"] = self.source
        if index is not None:
            payload["idx"] = index
        if self.source_url is not None:
            payload["source_url"] = self.source_url
        if self.content_hash is not None:
            payload["content_hash"] = self.content_hash
        if self.scraped_at is not None:
            payload["scraped_at"] = self.scraped_at
        return payload
        
    def to_embed(self):
        return self.scs