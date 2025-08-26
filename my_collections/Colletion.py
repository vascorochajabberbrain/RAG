from qdrant_client.http.models import PointStruct
from vectorization import get_embedding, get_point_id




from abc import abstractmethod


class Collection:
    TYPE = "base"

    def __init__(self, collection_name=None):
        self.items = []
        self.collection_name = collection_name

    @classmethod
    def init_from_qdrant(cls, collection_name, qdrant_points):
        """
        Download the SCS List from Qdrant.
        """

        #TO-DO: verify if _from_payload works on this collection type, might exist a missmatch
        self = cls(collection_name)
        print(f"Downloading collection: {collection_name}")
        for qdrant_point in qdrant_points:
            #print(qdrant_point)
            self.append_item(self.init_item_from_qdrant(self._get_only_point_data_from_payload(qdrant_point)))
        
        return self

    @abstractmethod
    def init_item_from_qdrant(self, point_data):
        pass
        #return SCS.from_payload(point_data)

    """--------------------------Dunders---------------------------"""

    def __str__(self):
        list_indexes = range(len(self.items))
        return self.print(list_indexes)
    
    def print(self, list_indexes=None):
        if list_indexes is None:
            list_indexes = range(len(self.items))
        if not (isinstance(list_indexes, list) or isinstance(list_indexes, range)) and len(self.items) != 0:
            raise TypeError("list_indexes must be a list of integers")
        for i, item in enumerate(self.items):
            if i not in list_indexes:
                continue
            print(f"[{i}] {item}")

    """----------------------Public Methods------------------------"""

    def get_collection_name(self):
        """
        Get the name of the collection.
        """
        return self.collection_name
    
    def points_to_save(self):
        """
        Save the Collection on Qdrant.
        """
        qdrant_points = []

        for item in self.items:
            print(item)
            qdrant_points.append(PointStruct(
                id=get_point_id(),
                vector=get_embedding(item.to_embed()),
                payload=self._add_collection_data_to_payload(item.to_payload())
            ))

        return qdrant_points
    
    def delete_item(self, idx):
        """
        Delete an item from the collection by index.
        """
        self._check_index(idx)
        del self.items[idx]

    def delete_all_items(self):
        """
        Delete all items from the collection.
        """
        self.items = []

    def append_item(self, item):
        """
        Add an item to the collection.
        """
        self.items.append(item)

    def insert_item(self, idx, item):
        """
        Insert an item at a specific index in the collection.
        """
        self._check_index(idx)
        self.items.insert(idx, item)

    def add_item(self, idx, item):
        """
        Alias for append_item to maintain consistency.
        """
        self.items[idx] = item

    @abstractmethod
    def menu(self):
        pass
    """-----------------------------Private Methods-----------------------------"""
    
    def _add_collection_data_to_payload(self, point_payload):
        """
        Add collection specific data to the payload.
        """
        return {
            "collection": {
                "type": self.TYPE
            },
            "point": point_payload
        }
    
    def _get_only_point_data_from_payload(self, point_payload):
        return point_payload["point"]
    
    def _check_index(self, idx):
        """
        Validate the index for accessing the Collection.
        """
        if idx < 0 or idx >= len(self.items):
            raise IndexError("Index out of bounds for this Collection.")