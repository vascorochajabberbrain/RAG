from langchain.text_splitter import CharacterTextSplitter
from qdrant_client.http.models import PointStruct
import uuid

from langchain import hub

from openai_utils import get_openai_client
from qdrant_utils import create_collection, insert_points

import json

''' #initial code for chunking
def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(
    separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len)
  chunks = text_splitter.split_text(text)
  return chunks
'''

def get_text_chunks(text):
    openai_client = get_openai_client()

    prompt = '''Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context. 
    1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
    2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
    3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
    4. Present the results as a list of strings, formatted in JSON. 
    Example:
        Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. 
        Content:The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny." 
        Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America."]'''
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages= [{"role": "system", "content": prompt},
                   {"role": "user", "content": text}]
        )
    
    return json.loads(completion.choices[0].message.content)


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


def insert_data(get_points, collection_name = None):
    while True:
        if collection_name is not None:
            insert_points(collection_name, get_points)
            return
        print("write 'q' if you want to quit the insertion")
        collection_name = input("Insert into which collection?")
        if collection_name == "q":
            return
        try:
            insert_points(collection_name, get_points)
            return
        except Exception as e:
            print(e)
            print("That collection does not exist, do you want to create it? or try again? (c/a)")
            choice = input()
            if choice == "c":
                create_collection(collection_name)
                #print("Collection created, now you need to write again the name of the collection to actually insert the data")
            elif choice == "a":
                continue

