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
        #print(os.getenv("QDRANT_API_KEY"))
        _connection = QdrantClient(
            url = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY"),
        )
        #print("New connection successful:") # for checking connection is successful or not
    #else:
        #print("Using previous connection")
    return _connection


def create_collection(collection_name):
    get_qdrant_connection()
    # to create the collection if it does not exist
    _connection.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

def duplicate_collection(collection_name, new_collection_name):
    get_qdrant_connection()
    _connection.create_collection(
    collection_name=new_collection_name,
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    init_from=models.InitFrom(collection=collection_name),
)
    
def get_points_from_collection(collection_name):
    connection = get_qdrant_connection()
    points = []
    offset = None

    while True:
        result, offset = connection.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        points.extend(result)
        if offset is None:
            break
    return points

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
    #print(_connection.retrieve(collection_name=collection_name, ids=[point_id]))
    return _connection.retrieve(collection_name=collection_name, ids=[point_id])[0].payload['text']

#info = _connection.get_collection(collection_name="fruit_example")
# until here the duplicated code

def add_source(collection_name, source):
    get_qdrant_connection()
    _connection.set_payload(
    collection_name=collection_name,
    payload={
        "source": source,
    },
    points=models.Filter(
    ),
)
    
def delete_collection(collection_name):
    get_qdrant_connection()
    _connection.delete_collection(collection_name)

def valid_collection_name(collection_name):
    get_qdrant_connection()
    collections = _connection.get_collections().collections
    return any(c.name == collection_name for c in collections)


def main():
    get_qdrant_connection()
    
    basic_menu_string = """Any action? 
                   -- "q" for exit
                   -- "d" to delete a collection
                   -- "r" to rename a collection
                   -- "a" to add a source
                   """
    action = input(basic_menu_string)
    
    while action != "q":
        match action:
            case "d":
                collection_name = input("Name of the collection:")
                delete_collection(collection_name)
            case "r":
                #not sure the payload is well transfered from the old to the new
                old_collection_name = input("Name of the old collection:")
                new_collection_name = input("Name of the new collection:")
                duplicate_collection(old_collection_name, new_collection_name)
                delete_collection(old_collection_name)
            case "a":
                collection_name = input("Name of the collection:")
                source = input("Source:")
                add_source(collection_name, source)
            case _:
                print("Invalid action")
        action = input(basic_menu_string)

if __name__ == '__main__':
    main()