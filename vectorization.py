from langchain.text_splitter import CharacterTextSplitter
from qdrant_client.http.models import PointStruct
import uuid

from openai_utils import get_openai_client
from qdrant_utils import get_qdrant_connection


def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(
    separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len)
  chunks = text_splitter.split_text(text)
  return chunks


def get_embedding(text_chunks, model_id="text-embedding-ada-002"):
    openai_client = get_openai_client()
    points = []
    for idx, chunk in enumerate(text_chunks):
        response = openai_client.embeddings.create(
            input=chunk,
            model=model_id
        )
        embeddings = response.data[0].embedding
        point_id = str(uuid.uuid4())  # Generate a unique ID for the point
        points.append(PointStruct(id=point_id, vector=embeddings, payload={"text": chunk}))

    return points


def insert_data(get_points):
    connection = get_qdrant_connection()
    while True:
        print("write 'q' if you want to quit the insertion")
        collection_name = input("Insert into which collection?")
        if collection_name == "q":
            return
        try:
            operation_info = connection.upsert(
            collection_name = collection_name,
            wait = True,
            points = get_points
            )
            return
        except Exception as e:
            print(e)
            print("That collection does not exist, try again")

    