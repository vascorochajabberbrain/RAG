'''Still supposing only one source per group.'''

class Group:
    def __init__(self, description=None, scss=None, source=None):
        self.description = description if description else ""
        self.scss = scss if scss else []
        self.source = source

    @classmethod
    def from_payload(cls, payload):
        description = payload["description"]
        source = payload.get("source", None)
        self = cls(description, source=source)
        self.add_scss(self._string_to_scss(payload["text"]))
        return self
    
    def add_scss(self, scss):
        self.scss.extend(scss)

    def _string_to_scss(self, string):
        return string.split("\n")

    def add_source(self, source):
        self.source = source

    def get_sentence(self):
        return self.scs
    
    def get_source(self):
        return self.source
    
    def to_payload(self):
        if self.source:
            return {
                "description": self.description,
                "text": "\n".join(self.scss),
                "source": self.source
            }
        else:
            return {
                "description": self.description,
                "text": "\n".join(self.scss)
            }
        
    def to_embed(self):
        """
        Convert the group to a string suitable for embedding.
        """
        return self.description + "\n" + "".join(self.scss)
    
    def __str__(self):
        text = f"Description: {self.description}\n"
        for i, prep in enumerate(self.scss):
            text += f"   ({i}) {prep}\n"
        if self.source:
            text += f"   Source: {self.source}\n"
        return text