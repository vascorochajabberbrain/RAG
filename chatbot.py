from datetime import datetime

from openai_utils import get_openai_client
from qdrant_utils import get_qdrant_connection

def make_conversation_file(conversation):
    # Generate a filename with the current date and minute
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")  # Example: 2025-02-19_14-30
    filename = f"conversation_logs/conversation_file_{timestamp}.txt"

    # Save the file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(conversation)

def get_retrieved_info(query, context, collection_name):
    openai_client = get_openai_client()

    content = "From this context: " + context + ", rewrite the query: " + query + """ so that it becomes a self-contained question.
    Replace vague references and pronouns with the appropriate details from the context.
    Do not add extra information that was not asked in the original query.
    Preserve the intent and structure of the question as much as possible."""
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": content}
        ]
        )
    new_query = completion.choices[0].message.content
    print("----------------------------------------\nOld query: ", query, "\nNew query: ", new_query, "\n----------------------------------------\n")

    response = openai_client.embeddings.create(
        input=new_query,
        model="text-embedding-ada-002"
    )
    embeddings = response.data[0].embedding
    
    connection = get_qdrant_connection()
    search_result = connection.query_points(
        collection_name=collection_name,
        query=embeddings,
        limit=3
    )
    #print(search_result)
    #print("Question: " ,query,'\n')
    #print("Searching.......\n")
    print("Retrieved chunks:+++++++++++++++++++++++++++++++++++++++++++\n")
    prompt=""
    for result in search_result.points:
        print("\n", result.payload['text'], "++++++++++++++++++++++++++++++++++++++++++\n")
        prompt += result.payload['text'] + "\n"
    return prompt
    
def get_answer(messages, retrieved_info, query):
    openai_client = get_openai_client()
    messages.append({"role": "system", "content": "The information you may use for this anwser is: " + retrieved_info + " and will try yo answer based on it to the user input"})
    messages.append({"role": "user", "content": query})
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
        )
    return completion.choices[0].message.content


def main():
    collection_name = input("Which collection should we use:")
    if collection_name == "1":
        collection_name = "hey_harper_1"
    elif collection_name == "FAQ":
        collection_name = "en_route_FAQ"

    conversation_file = """Conversation with bot retrieving from `${collection_name}`\n
    Using gpt-4o for queries and text-embedding-ada-002 for embeddings.\n
    retrieved information of all products individual page.\nConversation starts second next line:\n\n"""
    
    history = [{"role": "system", "content": """You are the virtual assitant ofHey Harper, a e-commerce store that sells jewelry and clothes.
                 You will receive some context anytime you have to anser a user question, notice that the context may not be helpfull, 
                use it only when clearly has the information you need to better help the user"""}]

    inital_greeting = "Hi, I am your `${collection_name}` virtual assistant, please make questions for me to answer"
    print("Bot: ", inital_greeting)
    conversation = "Bot: " + inital_greeting
    history.append({"role": "assistant", "content": inital_greeting})
    
    final_conversation_comments = "Here are the final comments from the conversation:\n"

    while True:
        question= input("User: ")
        if question == "q":
            final_conversation_comments += input("Any final comments?")
            break
        retrieved_info = get_retrieved_info(question, conversation, collection_name)

        #print("+++++++++++++++++++++++++++++++++++++++++++\nRetrieved chunks:\n", retrieved_info, "+++++++++++++++++++++++++++++++++++++++++++\n")

        answer = get_answer(history, retrieved_info, question)
        print("Bot : ",answer,"\n")
        conversation += "\nUser: " + question + "\nBot: " + answer
        history.append({"role": "assistant", "content": answer})
        #print("searching completed")

    conversation_file += conversation + "\n\n" + final_conversation_comments
    make_conversation_file(conversation_file)


if __name__ == '__main__':
    main()