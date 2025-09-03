from my_collections.SCS_Collection import SCS_Collection
from my_collections.groupCollection import GroupCollection
from llms.openai_utils import openai_chat_completion


def grouping(sc: SCS_Collection, gc: GroupCollection):
    """Groups SCSs from the SCS_Collection into the GroupCollection based on some criteria."""
    
    for scs in sc.get_all_scss():

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
        
        groups_part_of_text = f"""Groups:{gc.to_string(gc.unfull_groups_indexes(scs))}"""

        readable_text_to_print = f"""New sentence: {scs.get_sentence()}

                Allow new group creation (allowed '-1' response): {not gc.collection_is_full(scs)}

                So, possible answers are: {possible_answers}"""
        
        print(f"Grouping: sending: {text}")

        text = f"{groups_part_of_text}\n{readable_text_to_print}"
        llm_response = openai_chat_completion(prompt, text)
        
        print(f"Grpuping: LLM response: {llm_response}")
        
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
                error_message = f"You answered {llm_response}, but it is not a valid response because {e}. These are the possible answers: {possible_answers}. Please answer again."
                print(error_message)
                text += error_message
                llm_response = openai_chat_completion(prompt, text)

        
    print("Grouping: Creating descriptions for all groups...")
    gc.create_all_descriptions()

    print("Grouping: Done.")

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