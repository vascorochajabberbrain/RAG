from abc import abstractmethod


class Item:

    @abstractmethod
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def from_payload(cls, payload):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def set_source(self, source):
        """
        Set a source to the item.
        """
        if not isinstance(source, str):
            raise TypeError(f"Expected a string for source, got {type(source)}.")
        if not source:
            raise ValueError("Source cannot be an empty string.")
        self.source = source

    def get_source(self):
        """
        Get the source of the item.
        """
        return self.source
    
    def delete_source(self):
        """
        Delete the source from the item.
        """
        self.source = None
    
    @abstractmethod
    def to_payload(self, index=None):
        pass

    @abstractmethod
    def to_embed(self):
        pass