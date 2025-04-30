import json
from groupCollection import GroupCollection
from openai_utils import get_openai_client
from qdrant_utils import valid_collection_name, create_collection, delete_collection, get_qdrant_connection, get_points_from_collection, insert_points


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

    chunks = []
    offset = None

    while True:
        result, offset = connection.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        chunks.extend([pt.payload["text"] for pt in result])
        if offset is None:
            break
    return chunks

def get_collection_from_(gc, collection_name):
    gc.from_save_points(get_points_from_collection(collection_name))
    return gc

def get_all_descriptions(dict):
    descriptions = []
    for group in dict:
        descriptions.append(dict[group]["description"])
    return descriptions


def ai_new_description(gc, group_index):
    #get the description of all the groups
    all_descriptions = gc.get_all_descriptions()
    completion = get_openai_client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"""I need you to possibly rewrite the description of this group of sentences. 
                    Use the sentences themselfs and compare with the other descriptions to better decide what to call this one.
                    You can return the same description if you think it is the best one.
                    Here are the descriptions of all the groups: {all_descriptions}
                    Here is the description of the group to rewrite: {gc.get_description(group_index)}
                    Here are the prepositions of the group: {gc.get_prepositions(group_index)}
                    Return only the new description, no explanation"""}]
    )
    return completion.choices[0].message.content

def ai_rewrite_all_descriptions(gc):

    for ix in range(len(gc.groups)):
        new_description = ai_new_description(gc, ix)
        gc.update_description(ix, new_description)
    return gc

def main():

    grouped_collection_name = None
    gc = GroupCollection()

# Initialization of the class collection, either from sentence or already created grouped vectorbase

    sentence_or_grouped_collection = input("Do you want to create one from a sentence collection or update a grouped one? (s/g)")
    if sentence_or_grouped_collection == "s":

        sentence_collection_name = "hh_ps_prepositions"

        #sentence_collection_name = input("Which sentence collection should we use:")
        #while not valid_collection_name(sentence_collection_name):
        #    print("Invalid collection name")
        #    sentence_collection_name = input("Which sentence collection should we use:")

        chunks = get_list_of_chunks(sentence_collection_name)
        print("chunks: ", chunks)

        #first_descriptions = making_description_of_groups(chunks)
        first_descriptions = ["Different payment month plans, pros and cons", "'Monthly' Plan details", "'6-Month' plan details", "'12-Month' plan details", "Style options of the pieces and selection", "heart box", "general brand positioning on the market and target audience", "shipping info", "included jewelry on the subscription", "policies and warranties", "pieces composition and materials", "others"]
        
        gc = grouping_chunks(first_descriptions, chunks, gc)  
        gc.print()

        gc = ai_rewrite_all_descriptions(gc)
        gc.print()

    elif sentence_or_grouped_collection == "g":
        grouped_collection_name = input("Which grouped collection should we use:")
        while not valid_collection_name(grouped_collection_name):
            print("Invalid collection name")
            grouped_collection_name = input("Which grouped collection should we use:")
        gc = get_collection_from_(gc, grouped_collection_name)
        gc.print()

# Normal edition runnig mode

    basic_menu_string = """Any action? 
                   -- "q" for exit
                   -- "r" to rewrite a description
                   -- "m" to move a preposition
                   -- "c" to copy a preposition
                   -- "d" to delete a group
                   -- "p" to delete a preposition
                   -- "s" to save collection
                   """
    action = input(basic_menu_string)

    while action != "q":
        match action:
            case "r":
                group_index = int(input("Which group do you want to rewrite? "))
                manual_or_ai = input("Do you want to do it manually or with AI? (m/a) ")
                if manual_or_ai == "a":
                    new_description = ai_new_description(gc, group_index)
                else:
                    #manual
                    new_description = input("What is the new description? ")
                gc.update_description(group_index, new_description)
            case "m":
                preposition_index = int(input("Which preposition do you want to move? "))
                from_group_index = int(input("From which group? "))
                to_group_index = int(input("To which group? "))
                gc.move_preposition(preposition_index, from_group_index, to_group_index)
            case "c":
                preposition_index = int(input("Which preposition do you want to copy? "))
                from_group_index = int(input("From which group? "))
                to_group_index = int(input("To which group? "))
                gc.copy_preposition(preposition_index, from_group_index, to_group_index)
            case "d":
                group_index = int(input("Which group do you want to delete? "))
                gc.delete_group(group_index)
            case "p":
                group_index = int(input("Which group do you want to delete a preposition from? "))
                preposition_index = int(input("Which preposition do you want to delete? "))
                gc.delete_preposition(group_index, preposition_index)
            case "s":
                if grouped_collection_name == None:
                    grouped_collection_name = input("What is the name of the new grouped collection? ")
                    while not valid_collection_name(grouped_collection_name):
                        print("Invalid collection name")
                        grouped_collection_name = input("What is the name of the new grouped collection? ")
                else:
                    # save can be dangerous because if something happens in between the process we might lose the VDB
                    delete_collection(grouped_collection_name)
                create_collection(grouped_collection_name)
                insert_points(grouped_collection_name, gc.to_save_points())
            #case "a":
                #insert_data(gc.to_save_points(), grouped_collection_name)
            case _:
                print("Invalid action")
        
        gc.print()
        action = input(basic_menu_string)

if __name__ == '__main__':
    main()