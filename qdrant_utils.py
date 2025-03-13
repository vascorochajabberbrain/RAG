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
    get_qdrant_connection()
    # to create the collection if it does not exist
    _connection.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

def insert_points(collection_name, points):
    get_qdrant_connection()
    _connection.upsert(
        collection_name = collection_name,
        wait = True,
        points = points
    )
    return

def delete_points(collection_name, point_ids):
    get_qdrant_connection()
    _connection.delete(
        collection_name = collection_name,
        wait = True,
        points_selector=models.PointIdsList(
            points=point_ids,
        )
    )
    return

def get_point_text(collection_name, point_id):
    get_qdrant_connection()
    return _connection.retrieve(collection_name=collection_name, ids=[point_id])[0].payload['text']

#info = _connection.get_collection(collection_name="fruit_example")
# until here the duplicated code


def main():
    get_qdrant_connection()
    #collection_name = input("Name of the collection:")
    #create_collection(collection_name)

if __name__ == '__main__':
    main()