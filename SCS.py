

class SCS:

    """----------------------Constructors----------------------"""
    def __init__(self, scs, source=None):
        self.scs = scs
        self.source = source

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
    
    """-----------Source related methods-----------"""
    
    def set_source(self, source):
        self.source = source
    
    def get_source(self):
        return self.source
    
    def delete_source(self):
        self.source = None
    
    """---------Qdrant related methods-----------"""
    def to_payload(self):
        if self.source:
            return {
                "text": self.scs,
                "source": self.source
            }
        else:
            return {
                "text": self.scs
            }
        
    def to_embed(self):
        return self.scs