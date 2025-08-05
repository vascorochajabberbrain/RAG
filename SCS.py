

class SCS:
    def __init__(self, scs, source=None):
        self.scs = scs
        self.source = source

    @classmethod
    def from_payload(cls, payload):
        scs = payload["text"]
        source = payload.get("source", None)
        return cls(scs, source)
    
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
    
    def add_source(self, source):
        self.source = source

    def get_sentence(self):
        return self.scs
    
    def get_source(self):
        return self.source
    
    def __str__(self):
        return f"{self.scs}\n   Source: {self.source}" if self.source else self.scs