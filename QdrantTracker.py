from qdrant_client import QdrantClient, models
import os

from my_collections.groupCollection import GroupCollection
from my_collections.SCS_Collection import SCS_Collection

COLLECTIONS_TYPES_MAP = {
    "group": GroupCollection,
    "scs": SCS_Collection
}


class QdrantTracker:
    """
    A class to track Qdrant connection and open collections.
    """

    def __init__(self):
        print("QdrantTracker: Initializing QdrantTracker...")
        try:
            self._connection = QdrantClient(
                url = os.getenv("QDRANT_URL"),
                api_key = os.getenv("QDRANT_API_KEY"),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Qdrant: {e}. "
                "Please check your QDRANT_URL and QDRANT_API_KEY environment variables."
            )
        self._open_collections = set()
        print("QdrantTracker: QdrantTracker initialized.")

    def open(self, collection_name):
        """
        Make it sure that exists a 1:1 relation between Qdrant and local collection.
        Will certify if the user wants to use an existing collection or create a new one.
        In case of an existing collection it can return the points qDrant points from it.
        It is of each collection to know how to handle the points.
        If the collection does not exist, it will create a new one.
        Returns the collection name and a list of points if the collection exists.
        """
        if collection_name is None:
            collection_name = input("Insert the name of the collection:")

        points = []
        # Loop until we have a valid collection name
        while True:

            # we assume that if the collection does not exist, the user wants to create a new collection
            if not self._existing_collection_name(collection_name):
                print(f"QdrantTracker: Collection {collection_name} does not exist. Going to execute the new method...")
                collection =self.new(collection_name)  # Create a new collection
                return collection

            # if the collection exists, we make sure the user wants to overwrite it
            else:

                #loop until the user gives a valid answer
                while True:
                    using = input(f"The collection {collection_name} already exists. Do you want to use the existing points? (y/n): ")
                    if using.lower() in ['y', 'n']:
                        break
                    else:
                        print("Please enter 'y' for yes or 'n' for no.")

                if using.lower() == 'n':
                    print(f"Qdrant: Deleting collection {collection_name}...")
                    self._delete_collection(collection_name)
                    print(f"QdrantTracker: Collection {collection_name} does not exist. Going to execute the new method...")
                    collection =self.new(collection_name)  # Create a new collection
                    return collection

                elif using.lower() == 'y':
                    print(f"Qdrant: Getting points from collection {collection_name}...")
                    points = self._get_all_points(collection_name, points)
                    
                    break
        
        collection_type = self.get_collection_type(points[0]) #any point will have collection information
        collection = COLLECTIONS_TYPES_MAP[collection_type].init_from_qdrant(collection_name, points)
        self._open_collections.add(collection)
        print(f"QdrantTracker: Collection: {collection_name} is open.")
        return collection
    
    def new(self, collection_name):
        """
        Create a new collection with the given name.
        """
        if collection_name is None:
            collection_name = input("Insert the name of the collection:")
        
        if self._existing_collection_name(collection_name):
            print(f"QdrantTracker: Collection {collection_name} already exists. Going to execute the open method...")
            return self.open(collection_name)
        
        while True:
            collection_type = input(f"""From the options: {', '.join(COLLECTIONS_TYPES_MAP.keys())}\nEnter the type of the collection you want: """).strip().lower()
        
            if collection_type not in COLLECTIONS_TYPES_MAP:
                print(f"QdrantTracker: Invalid collection type.")
                continue
            break
        
        print(f"QdrantTracker: Creating new collection {collection_name}...")
        self._create_collection(collection_name)

        collection = COLLECTIONS_TYPES_MAP[collection_type](collection_name)
        self._open_collections.add(collection)
        print(f"QdrantTracker: New collection {collection_name} created and opened.")
        return collection
    
    def save_collection(self, collection_name):
        """
        Save the collection to Qdrant.
        """

        print(f"QdrantTracker: Deleting collection {collection_name}...")
        self._delete_collection(collection_name)
        print(f"QdrantTracker: Creating collection {collection_name}...")
        self._create_collection(collection_name)

        collection = self.get_collection(collection_name)
        points = collection.points_to_save()

        for i in range(0, len(points), 5):
            batch = points[i:i + 5]
            self._connection.upsert(
                collection_name=collection_name,
                wait = True,
                points=batch
            )
    
    def disconnect(self, collection_name):
        """
        Disconnect from the Qdrant collection.
        """
        self._check_open_collection(collection_name)
        
        while True:
            save_collection = input("Do you want to save the collection? (y/n): ")
            if save_collection.lower() == 'y':
                self.save_collection(collection_name)
                break
            elif save_collection.lower() == 'n':
                break
        

        self._remove(collection_name)
        print(f"QdrantTracker: Disconnected from collection: {collection_name}")
    
    def get_collection(self, collection_name):
        """
        Returns the collection object.
        """
        for c in self._open_collections:
            if c.get_collection_name() == collection_name:
                return c
        else:
            raise ValueError(f"Collection {collection_name} is not open.")

    def all_collections(self):
        collections = self._connection.get_collections().collections
        return [c.name for c in collections]
    
    def open_collections(self):
        """
        Returns a list of currently open collections.
        """
        return [c.get_collection_name() for c in self._open_collections]
    
    def delete_collection(self, collection_name):
        if collection_name in [c.get_collection_name() for c in self._open_collections]:
            collection = self.get_collection(collection_name)
            self._open_collections.remove(collection)
        self._delete_collection(collection_name)

    """-----------------------------Private Methods-----------------------------"""
    def _remove(self, collection_name):
        self._check_open_collection(collection_name)
        collection = self.get_collection(collection_name)
        self._open_collections.remove(collection)

    def get_collection_type(self, point):
        """
        Get the type of collection from the point payload.
        """
        if "collection" in point:
            return point["collection"]["type"]
        else:
            raise ValueError("Point payload does not contain collection type information.")

    def _get_all_points(self, collection_name, points=[]):
        offset = None
        while True:
            result, offset = self._connection.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            points.extend([point.payload for point in result])
            if offset is None:
                break #all points taken
        print(f"Só para ver como estão os points: {points}")
        return points

    def _check_open_collection(self, collection_name):
        if collection_name not in [c.get_collection_name() for c in self._open_collections]:
            raise ValueError(f"Collection {collection_name} is not open.")
        
    def _existing_collection_name(self, collection_name):
        collections = self._connection.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def _delete_collection(self, collection_name):
        self._connection.delete_collection(collection_name)  

    def _create_collection(self, collection_name):
        # to create the collection if it does not exist
        self._connection.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )