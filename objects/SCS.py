

from objects.Item import Item


class SCS(Item):

    """----------------------Constructors----------------------"""
    def __init__(self, scs, source=None):
        super().__init__(source)
        self.scs = scs

    @classmethod
    def from_payload(cls, payload):
        scs = payload["text"]
        source = payload.get("source", None)
        return cls(scs, source)
    
    """------------------------Dunders------------------------"""

    def __str__(self):
        return f"{self.scs}\n   Source: {self.source}" if self.source else self.scs

    """--------------------Public Methods----------------------"""
    """-----------Sentence related methods-----------"""

    def get_sentence(self):
        return self.scs
    
    
    """---------Qdrant related methods-----------"""
    def to_payload(self, index=None):
        if index is not None and self.source is not None:
            return {
                "text": self.scs,
                "source": self.source,
                "idx": index
            }
        if self.source is not None:
            return {
                "text": self.scs,
                "source": self.source
            }
        if index is not None:
            return {
                "text": self.scs,
                "idx": index
            }
        else:
            return {
                "text": self.scs
            }
        
    def to_embed(self):
        return self.scs