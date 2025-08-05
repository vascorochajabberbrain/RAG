from qdrant_client import QdrantClient, models
import os


class QdrantTracker:
    """
    A class to track Qdrant connection and open collections.
    """

    def __init__(self):
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

    def connect(self, collection_name):
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
                print(f"Qdrant: Creating collection {collection_name}...")
                self._create_collection(collection_name)
                break

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
                    print(f"Qdrant: Creating collection {collection_name}...")
                    self._create_collection(collection_name)
                    break

                elif using.lower() == 'y':
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
                    
                    break
            
        self._open_collections.add(collection_name)
        print(f"QdrantTracker: Collection: {collection_name} is open.")
        return collection_name, points
    
    def disconnect(self, collection_name, points=None):
        """
        Disconnect from the Qdrant collection.
        """
        if collection_name in self._open_collections:
            
            print(f"QdrantTracker: Deleting collection {collection_name}...")
            self._delete_collection(collection_name)
            print(f"QdrantTracker: Creating collection {collection_name}...")
            self._create_collection(collection_name)

            for i in range(0, len(points), 5):
                batch = points[i:i + 5]
                self._connection.upsert(
                    collection_name=collection_name,
                    wait = True,
                    points=batch
                )

            self._open_collections.remove(collection_name)
            print(f"QdrantTracker: Disconnected from collection: {collection_name}")
        else:
            print(f"QdrantTracker: Collection {collection_name} is not open.")
    

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