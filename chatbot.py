from datetime import datetime

from llms.openai_utils import get_openai_client
from qdrant_utils import get_qdrant_connection

def make_conversation_file(conversation, standard_filename):
    # Generate a filename with the current date and minute
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")  # Example: 2025-02-19_14-30
    #filename = f"conversation_logs/conversation_file_{timestamp}.txt"
    filename = f"{standard_filename}_{timestamp}.txt"

    # Save the file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(conversation)

def improve_query(query, history):
    openai_client = get_openai_client()

    content = """Rewrite the next user input, instead of answering, rewritte what the user said so that it becomes a self-contained question.
    Replace vague references and pronouns with the appropriate details from the context.
    Do not add extra information that was not asked in the original query.
    Preserve the intent and structure of the question as much as possible."""
    messages = history.copy()
    messages.append({"role": "system", "content": content})
    messages.append({"role": "user", "content": query})
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
        )
    new_query = completion.choices[0].message.content
    print("----------------------------------------\nOld query: ", query, "\nNew query: ", new_query, "\n----------------------------------------\n")
    return new_query

def retrieve_from_vdb(query, collection_names):
    """
    Retrieve relevant chunks from one or more Qdrant collections.
    collection_names: str or list[str]
    When multiple collections are provided, all are queried and results are merged by score (top K overall).
    """
    if isinstance(collection_names, str):
        collection_names = [collection_names]

    openai_client = get_openai_client()
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    embeddings = response.data[0].embedding

    connection = get_qdrant_connection()
    all_points = []
    for collection_name in collection_names:
        try:
            search_result = connection.query_points(
                collection_name=collection_name,
                query=embeddings,
                limit=5
            )
            all_points.extend(search_result.points)
        except Exception as e:
            print(f"[retrieve_from_vdb] Warning: could not query '{collection_name}': {e}")

    # Sort by score descending and take top 5 across all collections
    all_points.sort(key=lambda p: p.score, reverse=True)
    top_points = all_points[:5]

    prompt = ""
    for result in top_points:
        print("--", result.payload['point']['text'])
        prompt += result.payload['point']['text'] + "\n"
    print()
    return prompt

def get_retrieved_info(query, history, collection_names):
    new_query = improve_query(query, history)
    return retrieve_from_vdb(new_query, collection_names)
    
def get_answer(history, retrieved_info, query, company):
    openai_client = get_openai_client()
    messages = history.copy()
    messages.append({"role": "system", "content": f"""You are the virtual assistant of `{company}`, an e-commerce store specializing in jewelry and clothing.  
You will receive context before answering user questions. However, this context may not always be relevant.  
Use it **only if it clearly provides helpful information**.  

The available information for this answer is: `{retrieved_info}`.  
Answer the user's query based on this data when applicable."""
})
    messages.append({"role": "user", "content": query})
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
        )
    return completion.choices[0].message.content


def main():
    user_input = input("Which solution should we use (name or alias, e.g. pf, hh, peixefresco): ").strip()
    try:
        from solution_specs import resolve_alias, get_solution, get_collections
        solution = resolve_alias(user_input) or get_solution(user_input)
        if solution:
            colls = get_collections(solution["id"])
            collection_names = [c["collection_name"] for c in colls if c.get("collection_name")]
            company = solution.get("company_name") or solution.get("display_name")
        else:
            # Fallback: treat input as a direct collection name
            collection_names = [user_input]
            company = None
    except Exception:
        collection_names = [user_input]
        company = None
    if company is None:
        company = "the assistant"
    if not collection_names:
        collection_names = [user_input]

    print(f"Querying collections: {', '.join(collection_names)}")

    conversation_file = f"""Conversation with bot retrieving from `{company}`\n
    Collections: {', '.join(collection_names)}\n
    Using gpt-4o for queries and text-embedding-ada-002 for embeddings.\nConversation starts second next line:\n\n"""
    
    history = []

    inital_greeting = f"Hi, I am your `{company}` virtual assistant, please make questions for me to answer"
    print("Bot: ", inital_greeting)
    conversation = "Bot: " + inital_greeting
    history.append({"role": "assistant", "content": inital_greeting})
    
    final_conversation_comments = "Here are the final comments from the conversation:\n"

    while True:
        question= input("User: ")
        if question == "q":
            final_conversation_comments += input("Any final comments?")
            break

        retrieved_info = get_retrieved_info(question, history, collection_names)
        #print("+++++++++++++++++++++++++++++++++++++++++++\nRetrieved chunks:\n", retrieved_info, "+++++++++++++++++++++++++++++++++++++++++++\n")
        answer = get_answer(history, retrieved_info, question, company)

        history.append({"role": "user", "content": question})
        print("Bot : ",answer,"\n")
        print(history)
        conversation += "\n\nUser: " + question + "\n\nBot: " + answer
        history.append({"role": "assistant", "content": answer})
        #print("searching completed")

    conversation_file += conversation + "\n\n" + final_conversation_comments
    make_conversation_file(conversation_file, f"conversation_{company.replace(' ','_')}")


if __name__ == '__main__':
    main()