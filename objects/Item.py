from abc import abstractmethod


class Item:

    def __init__(self, source=None):
        self.source = source
        

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
        if not isinstance(source, str) and source is not None:
            raise TypeError(f"Expected a string or None for source, got {type(source)}.")
        self.source = source

    def has_source(self):
        """
        Check if the item has a source.
        """
        return self.source is not None
    
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
    
    def same_source(self, source):
        return self.source == source
    
    @abstractmethod
    def to_payload(self, index=None):
        pass

    @abstractmethod
    def to_embed(self):
        pass