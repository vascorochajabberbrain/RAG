from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient,models
from qdrant_client.http.models import PointStruct
import os

from openai_utils import get_openai_client
from qdrant_utils import get_qdrant_connection

    
def get_retrieved_info(query):
    openai_client = get_openai_client()
    connection = get_qdrant_connection()
    response = openai_client.embeddings.create(
        input=query,

        model="text-embedding-ada-002"
    )
    embeddings = response.data[0].embedding
    search_result = connection.query_points(
        collection_name="fruit_example",
        query=embeddings,
        limit=3
    )
    print(search_result)
    print("Question: " ,query,'\n')
    print("Searching.......\n")
    prompt=""
    for result in search_result.points:
        prompt += result.payload['text'] + "\n"
    return prompt
    
def get_answer(retrieved_info, query):
    openai_client = get_openai_client()
    concatenated_string = " ".join([retrieved_info,query])
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": concatenated_string}
        ]
        )
    return completion.choices[0].message.content


def main():

  while True:
    question= input("What is your question\n")
    if question == "q":
       break
    retrieved_info = get_retrieved_info(question)

    print("Retrieved chunks:\n", retrieved_info, "\n")

    answer = get_answer(retrieved_info, question)
    print("Answer : ",answer,"\n")
    print("searching completed")


if __name__ == '__main__':
    main()