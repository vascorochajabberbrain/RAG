'''Still supposing only one source per group.'''

class Group:
    LIMIT_OF_SCS = 8
    """-----------------------------Constructors-----------------------------"""
    
    def __init__(self, description=None, scss=None, source=None):
        self.description = description if description else ""
        self.scss = scss if scss else []
        self.source = source

    @classmethod
    def from_payload(cls, payload):
        description = payload["description"]
        source = payload.get("source", None)
        self = cls(description, source=source)
        self.append_scs(self._string_to_scss(payload["text"]))
        return self
    
    """---------------------------------Dunders---------------------------------"""

    def __str__(self):
        text = f"Description: {self.description}\n"
        for i, prep in enumerate(self.scss):
            text += f"   ({i}) {prep}\n"
        if self.source:
            text += f"   Source: {self.source}\n"
        return text
    
    def __repr__(self):
        if self.source:
            return f"Group(description={self.description}, scss={self.scss}, source={self.source})"
        else:
            return f"Group(description={self.description}, scss={self.scss})"
    
    """-----------------------------Public Methods-----------------------------"""
    """--------Status related methods---------"""

    def is_full(self):
        return len(self.scss) >= Group.LIMIT_OF_SCS
    
    """----------SCS related methods----------"""

    def append_scs(self, scss):# one or more
        """
        Add SCS's to the group.
        """
        self._check_full()
        if isinstance(scss, str):
            self.scss.append(scss)
        elif isinstance(scss, list):
            for i, scs in enumerate(scss):
                if not isinstance(scs, str):
                    raise TypeError(f"Expected a string, got {type(scs)} at index {i}.")
            self.scss.extend(scss)
        else:
            raise TypeError("Expected a string or a list of strings for SCS.")

    def get_scs(self, idx):#only one
        """
        Get a specific SCS by index.
        """
        self._check_index(idx)
        return self.scss[idx]
        
    def get_all_scs(self):
        """
        Get all SCS's in the group.
        """
        return self.scss
    
    def delete_scs(self, idxs):#one or more
        """
        Delete SCS's by index.
        """
        if isinstance(idxs, int):
            self._check_index(idxs)
            del self.scss[idxs]
        elif isinstance(idxs, range):
            idxs = list(idxs)
        elif isinstance(idxs, list):
            for i, idx in enumerate(idxs):
                if not isinstance(i, int):
                    raise TypeError(f"Expected an integer, got {type(idx)} at index {i}.")
                self._check_index(idx)
            idxs.sort(reverse=True)  # Sort in reverse order to avoid index shifting
            for idx in idxs:
                del self.scss[idx]
        else:
            raise TypeError("Expected an integer or a list of integers for SCS index.")

    """--------Source related methods--------"""

    def set_source(self, source):
        """
        Set a source to the group.
        """
        if not isinstance(source, str):
            raise TypeError(f"Expected a string for source, got {type(source)}.")
        if not source:
            raise ValueError("Source cannot be an empty string.")
        self.source = source

    def get_source(self):
        """
        Get the source of the group.
        """
        return self.source
    
    def delete_source(self):
        """
        Delete the source from the group.
        """
        self.source = None

    """--------Description related methods--------"""

    def set_description(self, new_description):
        """
        Set a new description for the group.
        """
        if not isinstance(new_description, str):
            raise TypeError(f"Expected a string for description, got {type(new_description)}.")
        if not new_description:
            raise ValueError("Description cannot be an empty string.")
        self.description = new_description

    def get_description(self):
        """
        Get the description of the group.
        """
        return self.description
    
    def delete_description(self):
        """
        Delete the description from the group.
        """
        self.description = None
    
    """---------Qdrant related methods-----------"""

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
    
    """-----------------------------Private Methods-----------------------------"""

    def _string_to_scss(self, string):
            if string == "":
                return []
            return string.split("\n")

    def _check_index(self, idx):
        """
        Check if the index is not valid for the current group.
        """
        if idx < 0 or idx >= len(self.scss):
            raise IndexError("Index out of bounds for this Group.")
        
    def _check_full(self):
        """
        Check if the group is full.
        """
        if self.is_full():
            raise ValueError("Group is full. Cannot add more SCS's.")