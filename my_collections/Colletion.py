import time
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
        
        if self._point_index(qdrant_points[0]) is not None:
            print("It is an ordered collection")
            return self.init_from_qdrant_ordered(qdrant_points)
        else:
            print("It is an unordered collection")
            return self.init_from_qdrant_unordered(qdrant_points)
    
    def init_from_qdrant_ordered(self, qdrant_points):
        self.items = [None] * len(qdrant_points)  # Preallocate list with None
        has_collection_data = self._check_collection_data_in_payload(qdrant_points[0])
        print(f"has_collection_data: {has_collection_data}")
        for qdrant_point in qdrant_points:
            if has_collection_data: # if it has collection data, the point data has its own key on the payload
                self.add_item(self._point_index(qdrant_point), self.init_item_from_qdrant(self._get_only_point_data_from_payload(qdrant_point)))
            else: # if it doesn't have collection data, the point data is in the payload itself
                self.add_item(self._point_index(qdrant_point), self.init_item_from_qdrant(qdrant_point))
        return self
    
    def init_from_qdrant_unordered(self, qdrant_points):
        self.items = []
        has_collection_data = self._check_collection_data_in_payload(qdrant_points[0])
        print(f"has_collection_data: {has_collection_data}")
        for qdrant_point in qdrant_points:
            if has_collection_data: # if it has collection data, the point data has its own key on the payload
                self.append_item(self.init_item_from_qdrant(self._get_only_point_data_from_payload(qdrant_point)))
            else: # if it doesn't have collection data, the point data is in the payload itself
                self.append_item(self.init_item_from_qdrant(qdrant_point))
        return self
    
    @abstractmethod
    def init_item_from_qdrant(self, point_data):
        pass

    """--------------------------Dunders---------------------------"""

    def __str__(self):
        list_indexes = range(len(self.items))
        return self.print(list_indexes)
    def print(self, list_indexes=None):
        print(self.to_string(list_indexes))

    def to_string(self, list_indexes=None):
        print(f"list_indexes: {list_indexes}")
        if list_indexes is None:
            list_indexes = range(len(self.items))
        if not (isinstance(list_indexes, list) or isinstance(list_indexes, range)) and len(self.items) != 0:
            raise TypeError("list_indexes must be a list of integers")
        text = ""
        for i, item in enumerate(self.items):
            if i not in list_indexes:
                continue
            text += f"[{i}] {item}\n"
        return text

    

    """----------------------Public Methods------------------------"""

    def get_collection_name(self):
        """
        Get the name of the collection.
        """
        return self.collection_name
    
    def points_to_save(self, model_id: str = "text-embedding-ada-002"):
        """
        Embed all items and return a list of Qdrant PointStructs ready to upsert.
        model_id: OpenAI embedding model to use (must match the collection's vector size).
        """
        qdrant_points = []

        total = len(self.items)
        for idx, item in enumerate(self.items):
            print(f"Embedding {idx + 1}/{total}…")
            qdrant_points.append(PointStruct(
                id=get_point_id(),
                vector=get_embedding(item.to_embed(), model_id=model_id),
                payload=self._add_collection_data_to_payload(item.to_payload(idx))
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

    def get_item(self, idx):
        """
        Get an item from the collection by index.
        """
        self._check_index(idx)
        return self.items[idx]
    
    def move_item(self, from_idx, to_idx):
        """
        Move an item from one index to another.
        """
        self._check_index(from_idx)
        self._check_index(to_idx)
        item = self.items.pop(from_idx)
        self.items.insert(to_idx, item)
        
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
    
    def _check_collection_data_in_payload(self, point_payload):
        #there was an update where collection data was added to the payload, prior to that all payload had point data
        #making it unecessary to call _get_only_point_data_from_payload
        if "collection" in point_payload:
            return True
        else:
            return False

    def _get_only_point_data_from_payload(self, point_payload):
        return point_payload["point"]
    
    def _point_index(self, point_payload):
        """
        Get the index of the point from the payload.
        """
        has_collection_data = self._check_collection_data_in_payload(point_payload)#same story of adapting to collections that don't have collection data at any point
        if has_collection_data:
            print(f"The _point_index output is {point_payload['point'].get('idx', None)}")
            return point_payload["point"].get("idx", None)
        else:
            print(f"The _point_index output is {point_payload.get('idx', None)}")
            return point_payload.get("idx", None)
    
    def _check_index(self, idx):
        """
        Validate the index for accessing the Collection.
        """
        if idx < 0 or idx >= len(self.items):
            raise IndexError("Index out of bounds for this Collection.")