from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient,models

# Load environment variables from .env file
load_dotenv()
# Global variable to store the connection
_connection = None  


def get_qdrant_connection():
    global _connection
    if _connection is None:  # Only initialize if not already created
        print(os.getenv("QDRANT_API_KEY"))
        _connection = QdrantClient(
            url = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY"),
        )
        print("New connection successful:") # for checking connection is successful or not
    else:
        print("Using previous connection")
    return _connection


def create_collection(collection_name):
    # to create the collection if it does not exist
    _connection.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

#info = _connection.get_collection(collection_name="fruit_example")
# until here the duplicated code

# I will use the main to create the collections when I need
def main():
    get_qdrant_connection()
    collection_name = input("Name of the collection:")
    create_collection(collection_name)

if __name__ == '__main__':
    main()