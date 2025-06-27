import json
from groupCollection import GroupCollection
from openai_utils import get_openai_client, openai_chat_completion, wait_for_run_completion
from qdrant_utils import existing_collection_name, create_collection, delete_collection, get_qdrant_connection, get_points_from_collection, insert_points


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

def clean_LLM_response(response):
    #if the response comes as a list with an integer, just return the string of the integer
    return response.strip("[]")

def is_valid_group_response(response, collection_is_full):
    if response == "-1" and not collection_is_full:
        return True
    if response == "-1" and collection_is_full:
        return False
    #check if the response is a number, if it is not, return False
    try:
        int(response)
        return True
    except ValueError:
        return False

def grouping_chunks(descriptions, chunks, gc):
    openai_client = get_openai_client()
    #assuming there is at least one chunk
    groupedChunks = [chunks[0]]

    gc.add_groups(descriptions)
    gc.print()

    assistant = openai_client.beta.assistants.create(
        name="Sentence Inserter",
        instructions="""You are a classifier whose task is to group self-contained sentences by meaning.

            You will receive:
            - A list of groups, where each group is a list of self-contained sentences. Each group is identified by a number.
            - A new sentence to insert.
            - A boolean flag indicating whether creating a new group is allowed.

            Your response must be:
            - The number of the group where the sentence fits best.
            - Or `-1` if it fits in none and should form a new group.

            Rules:
            - If no groups exist yet, the only valid answer is `-1`.
            - If the boolean is True, you must pick an existing group — do not return `-1`.

            Only respond with the number. Examples:
            `2`  
            `-1`""",
        model="gpt-4o"
    )
    
    

    for index, chunk in enumerate(chunks[:1000]):
            
            thread = openai_client.beta.threads.create()

            user_message = f"""Groups:
                {gc.to_string()}

                New sentence: {chunk}

                Allow new group creation (allowed '-1' response): {not gc.collection_is_full()}"""

            openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )

            run = openai_client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )

            wait_for_run_completion(thread.id, run.id)

            response = openai_client.beta.threads.messages.list(thread_id=thread.id).data[0].content[0].text.value

            response = clean_LLM_response(response)
            while not is_valid_group_response(response, gc.collection_is_full()):
                #repeat until the LLM ansers a valid response, this is dangerous though
                print("The LLM answer an invalid response, the response was: ", response)
                openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content="That response was invalid, please pay attention to the rules and try again, no explanations just the response"
            )
                run = openai_client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id
                )
                wait_for_run_completion(thread.id, run.id)
                response = openai_client.beta.threads.messages.list(thread_id=thread.id).data[0].content[0].text.value
                response = clean_LLM_response(response)

            if response == "-1":
                print(f"Sentence({index + 1}/{len(chunks)}), response: {response})")
            else:
                print(f"Sentence({index + 1}/{len(chunks)}), response: {response} that has {gc.number_of_prepositions(int(response))} SCS's")
            print(f"Groups({gc.number_of_full_groups()}/{gc.number_of_groups()})full, chunk: {chunk}")

            #path where I don't decide
            if response == "-1":
                gc.add_group("temporary description")
                gc.add_preposition(len(gc.groups)-1, chunk)
            else:
                gc.add_preposition(int(response), chunk)
            '''
            #path where I decide
            gc.print()
            print("chunk: ", chunk)
            print("response: ", response)
            agreed = input("agree?")
            if agreed == "y":
                if response == "-1":
                    gc.add_group("temporary description")
                    gc.add_preposition(len(gc.groups)-1, chunk)
                else:
                    gc.add_preposition(int(response), chunk)
            else:
                user_response = input("then what?")
                if user_response == "-1":
                    gc.add_group("temporary description")
                    gc.add_preposition(len(gc.groups)-1, chunk)
                else:
                    gc.add_preposition(int(user_response), chunk)
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


#get the description of all the groups
def ai_new_description(gc, group_index, with_other_descriptions=False):

    if with_other_descriptions:
        all_descriptions = gc.get_all_descriptions()
        prompt = f"""I need you to possibly rewrite the description of this group of sentences. 
                    Use the sentences themselfs and compare with the other descriptions to better decide what to call this one.
                    You can return the same description if you think it is the best one.
                    Here are the descriptions of all the groups: {all_descriptions}
                    Here is the description of the group to rewrite: {gc.get_description(group_index)}
                    Here are the prepositions of the group: {gc.get_prepositions(group_index)}
                    Return only the new description, no explanation"""
    else:
        prompt = f"""I need you to possibly rewrite the description of this group of sentences. 
                    Use the sentences themselfs to better decide what to call this one.
                    You can return the same description if you think it is the best one.
                    Here is the description of the group to rewrite: {gc.get_description(group_index)}
                    Here are the prepositions of the group: {gc.get_prepositions(group_index)}
                    Return only the new description, no explanation"""
    
    completion = get_openai_client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
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

    #sentence_or_grouped_collection = input("Do you want to create one from a sentence collection or update a grouped one? (s/g)")
    sentence_or_grouped_collection = "s"
    if sentence_or_grouped_collection == "s":


        sentence_collection_name = "autoderm_alpha"
        #sentence_collection_name = input("Which sentence collection should we use:")
        #while not existing_collection_name(sentence_collection_name):
        #    print("That collection does not exist, try again with another collection name")
        #    sentence_collection_name = input("Which sentence collection should we use:")

        chunks = get_list_of_chunks(sentence_collection_name)
        print("chunks: ", chunks)

        #first_descriptions = making_description_of_groups(chunks)
        #first_descriptions = ["Different payment month plans, pros and cons", "'Monthly' Plan details", "'6-Month' plan details", "'12-Month' plan details", "Style options of the pieces and selection", "heart box", "general brand positioning on the market and target audience", "shipping info", "included jewelry on the subscription", "policies and warranties", "pieces composition and materials", "others"]
        first_descriptions = []

        gc = grouping_chunks(first_descriptions, chunks, gc)  
        #gc.print()

        print("Now we have the groups, we can rewrite the descriptions with AI")
        gc = ai_rewrite_all_descriptions(gc)
        gc.print([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])

    elif sentence_or_grouped_collection == "g":
        grouped_collection_name = input("Which grouped collection should we use:")
        while not existing_collection_name(grouped_collection_name):
            print("That collection does not exist, try again with another collection name")
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
                    #means we came from a sentence collection and we need to create a new grouped collection
                    grouped_collection_name = input("What is the name of the new grouped collection? ")
                    while existing_collection_name(grouped_collection_name):
                        print("That collection already exists, try again with another collection name")
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