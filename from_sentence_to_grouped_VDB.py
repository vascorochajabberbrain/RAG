import json
from groupCollection import GroupCollection
from openai_utils import get_openai_client
from qdrant_utils import get_point_text, get_qdrant_connection


def making_description_of_groups(chunks):
    string_of_the_chunks = "\n".join(chunks)
    print(string_of_the_chunks)

    openai_client = get_openai_client()
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"""I need to divide these prepositions into groups. Return a list of descriptions of the different groups into these prepositions can be grouped by.
                   Here are the prepositions to group: {string_of_the_chunks}
                   The format of your response is:
                   [
                       "First group description",
                       "Second group description",
                       "Third group description",
                   ]
                   You can return as many groups as you think are necessary.
                   Note, just write the descriptions, not the names of the groups or any reference to the number od the group.Return in JSON but without writting json at beggining"""}]
    )

    print(completion.choices[0].message.content)
    return json.loads(completion.choices[0].message.content)

#outdated, not used anymore, was used when I had a list of the groups prepositions, where the prepositions were already only one concatenated string
def list_of_chunks_to_numbered_string(chunks):
    string = ""
    for chunk_ix, chunk in enumerate(chunks):
        #single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"""
        single_chunk_string = f"""Group ({chunk_ix}): {chunk}\n\n"""
        string += single_chunk_string
    return string

def disctionary_of_chunks_to_string(chunks):
    string = ""
    for group_ix, group in chunks.items():
        string += f"""Group ({group_ix}): {group["description"]}\n"""
        string += "\n".join(group["prepositions"])
        string += "\n\n"
    return string

def initalize_dictionary(descriptions):
    dictionary_of_groups = {}
    for ix, description in enumerate(descriptions):
        dictionary_of_groups[ix] = {
            "description": description,
            "prepositions": [],
        }
    return dictionary_of_groups

def grouping_chunks(descriptions, chunks, gc):
    openai_client = get_openai_client()
    #assuming there is at least one chunk
    groupedChunks = [chunks[0]]

    gc.add_groups(descriptions)
    gc.print()

    for chunk in chunks:
            
            # TO-DO add the response format to also include an explanation
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"""You are classifier which porpuse is to group prepositions together depending on their meaning.
                           You will receive a list of groups of prepositions and a new one.
                           Each group will be identified by a number.
                           Please answer onlywith the number of the group in which the new preposition fits well, or -1 if it doesn't fit well in any of them.
                           I rreally just want you to answer with a umber for me to be able to conver it to an integer on my code. Examples of answers:
                           2
                           -1
                           The groups are:
                           {gc.to_string()}
                           New preposition: {chunk}"""}
                          ]
            )
            response = completion.choices[0].message.content
            if response == "-1":
                gc.add_group("temporary description")
                gc.add_preposition(len(gc.groups)-1, chunk)
            else:
                gc.add_preposition(int(response), chunk)
            #path where I decide
            '''print("grouped chunks:\n", string_of_chunks)
            print("chunk: ", chunk)
            print("response: ", response)
            agreed = input("agree?")
            if agreed == "y":
                if response == "-1":
                    dictionary_of_groups[len(dictionary_of_groups)] = {
                        "description": "temporary description",
                        "prepositions": [chunk]
                    }
                else:
                    dictionary_of_groups[int(response)]["prepositions"].append(chunk)
            else:
                user_response = input("then what?")
                if user_response == "-1":
                    dictionary_of_groups[len(dictionary_of_groups)] = {
                        "description": "temporary description",
                        "prepositions": [chunk]
                    }
                else:
                    dictionary_of_groups[int(user_response)]["prepositions"].append(chunk)
            '''
    
    return gc

def get_list_of_chunks(collection_name):
    connection = get_qdrant_connection()

    all_ids = []
    offset = None

    while True:
        result, offset = connection.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        all_ids.extend([pt.id for pt in result])
        if offset is None:
            break
    return all_ids

def get_all_descriptions(dict):
    descriptions = []
    for group in dict:
        descriptions.append(dict[group]["description"])
    return descriptions

def rewrite_descriptions(gc):

    #get the descriptions of all the groups
    all_descriptions = gc.get_all_descriptions()
    for ix in range(len(gc.groups)):
        completion = get_openai_client().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"""I need you to possibly rewrite the description of this group of sentences. 
                        Use the sentences themselfs and compare with the other descriptions to better decide what to call this one.
                        You can return the same description if you think it is the best one.
                        Here are the descriptions of all the groups: {all_descriptions}
                        Here is the description of the group to rewrite: {gc.get_description(ix)}
                        Here are the prepositions of the group: {gc.get_prepositions(ix)}
                        Return only the new description, no explanation"""}]
        )
        gc.update_description(ix, completion.choices[0].message.content)
    return gc

def main():
    sentence_collection_name = "hh_ps_prepositions"
    #sentence_collection_name = input("Which sentence collection should we use:")
    chunksIds = get_list_of_chunks(sentence_collection_name)
    print("chunksIds: ", chunksIds)
    chunks = []
    for chunkId in chunksIds:
        chunks.append(get_point_text(sentence_collection_name, chunkId))
    print("chunks: ", chunks)
    #first_descriptions = making_description_of_groups(chunks)
    first_descriptions = ["Different payment month plans, pros and cons", "'Monthly' Plan details", "'6-Month' plan details", "'12-Month' plan details", "Style options of the pieces and selection", "heart box", "general brand positioning on the market and target audience", "shipping info", "included jewelry on the subscription", "policies and warranties", "pieces composition and materials", "others"]
    
    gc = GroupCollection()
    gc = grouping_chunks(first_descriptions, chunks, gc)  
    gc.print()
    gc = rewrite_descriptions(gc)
    gc.print()

if __name__ == '__main__':
    main()