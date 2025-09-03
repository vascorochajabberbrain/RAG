from my_collections.SCS_Collection import SCS_Collection
from my_collections.groupCollection import GroupCollection
from llms.openai_utils import openai_chat_completion


def grouping(sc: SCS_Collection, gc: GroupCollection):
    """Groups SCSs from the SCS_Collection into the GroupCollection based on some criteria."""
    
    for scs in sc.get_all_scss():

        #get source of scs
        source = scs.get_source()

        #get possible groups for this source in gc
        possible_groups = gc.available_groups_for_source(source)

        #call llm to chose which group
        prompt = """You are a classifier whose task is to insert a sentence into a group by meaning.

                You will receive:
                - A list of groups, where each group is a list of sentences. Each group is identified by a number.
                - A new sentence to insert.
                - A boolean flag indicating whether creating a new group is allowed.

                Your response must be:
                - A list of one of these two options, for each new sentence:
                    - The number of the group where the sentence fits best.
                    - Or `-1` if it fits in none and should form a new group.

                Rules:
                - If no groups exist yet, the only valid answer is `-1`, for the first new sentence.
                - If the boolean is True, you must pick an existing group — do not return `-1`.

                Only respond with a number. Examples:
                `3`
                `-1`
                `34`
                `12`"""
        
        possible_answers = gc.unfull_groups_indexes(scs)
        if not gc.collection_is_full(scs):
            possible_answers[0].append(-1)
        
        text = f"""Groups:{gc.to_string(gc.unfull_groups_indexes(scs))}

                New sentence: {scs.get_sentence()}

                Allow new group creation (allowed '-1' response): {not gc.collection_is_full(scs)}

                So, possible answers are: {possible_answers}"""
        print(f"Text: {text}")
        llm_response = openai_chat_completion(prompt, text)
        print(f"LLM response: {llm_response}")
        
        #repeat until the LLM ansers a valid response, this is dangerous though
        while True:
            try:
                llm_response = check_valid_llm_response(llm_response)
                if llm_response == -1:
                    index = gc.append_description("", scs)
                else:
                    index = llm_response
                gc.append_scs(index, scs)
                break
            except Exception as e:
                print(f"You answered {llm_response}, but it is not a valid response because {e}. These are the possible answers: {possible_answers}. Please answer again.")
                text += f"You answered {llm_response}, but it is not a valid response because {e}. These are the possible answers: {possible_answers}. Please answer again."
                llm_response = openai_chat_completion(prompt, text)

        #append scs to the chosen group in gc

def check_valid_llm_response(response):
    try:
        response = int(response)
    except ValueError:
        response.replace("`", "")
        try:
            response = int(response)
        except ValueError:
            raise ValueError("Response is not an integer.")
        
    return response