


"""-------------Answers functions------------------"""
from chatbot import make_conversation_file, retrieve_from_vdb
from llms.openai_utils import openai_chat_completion


def question_prompt_generator(type):
    task_prompt = """ Use the provided text to answer the question, notice that the text was generated from a website scrapping process."""
    match type:
        case "yes_or_no":
            task_prompt += """ It is a yes or not question. Only possibly answers are: "Yes", "No" or "IDontKnow"."""
        case "list":
            task_prompt += """ It is a question to list something, just name the items."""
        case "process":
            task_prompt += """ It is a question on a process, explain all the steps."""
        case _:
            pass
    task_prompt += """ Overall try to be brieve.
    Use only the provided text to answer the question.
    If the provided text does not contain the answer, do not explain that the text does have the answer just answer "IDontKnow".
    The format of the user input will be:
    Question: <question>
    Text: <text>
    Answer: """
    return task_prompt

def ask_questions_from_scraped_text(text, questions, types):
    """ Receives a list of questions and returns a list of answers. 
    FOR NOW: we receive a list with the same size of the questions to specify the type of question. """
    answers = []
    for question, type in zip(questions, types):
        task_prompt = question_prompt_generator(type)

        response = openai_chat_completion(task_prompt, "Question: " + question + "\nContext: " + text)
        answers.append(response)
    return answers

def ask_questions_from_vdb(collection_name, questions, types):
    """ Receives a list of questions and returns a list of answers. 
    FOR NOW: we receive a list with the same size of the questions to specify the type of question. """
    answers = []
    for question, type in zip(questions, types):
        task_prompt = question_prompt_generator(type)

        retrieved_info = retrieve_from_vdb(question, collection_name)
        response = openai_chat_completion(task_prompt, "Question: " + question + "\nContext: " + retrieved_info)
        answers.append(response)
    return answers

def main():

        questions = [
    "What products do you sell?",
    "Do you sell bracelets?",
    "Do you sell rings?",
    "Do you sell dresses?",
    "Apart from rings, bracelets, and earrings, what other products do you offer?",
    "What payment methods do you offer?",
    "Is it possible to pay with VISA?",
    "Is it possible to pay with multiple credit cards?",
    "Apart from VISA, AMEX, and PayPal, what other payment methods do you offer?",
    "Explain the domestic return delivery process, including details of the time and cost.",
    "What is the fee for domestic return delivery?",
    "Explain the international return delivery process, including details of the time and cost.",
    "What is the fee for international return delivery?",
    "What is the process to return a product?",
    "Which products, if any, require a different return process?",
    "Explain where your offices are located.",
    "Do you have physical stores that customers can visit?",
    "Explain how to review and change the language preferences.",
    "Explain how to navigate the website.",
    "Explain what a customer should do if your website is unresponsive.",
    "Explain how to purchase only one product.",
    "Explain how to purchase bracelet products.",
    "Explain where to find the terms and conditions.",
    "Explain how to add a user account.",
    "Explain how to review and change the password.",
    "Do you offer personalized products?",
    "Do you offer digital gift vouchers?",
    "Enter a general description of your products.",
    "Describe the quality and reasons for the quality of your necklace products.",
    "Describe the range of prices of your products.",
    "Describe the sizes your products are available in.",
    "Explain how to select the correct size of your products.",
    "Explain the general traits of your products.",
    "Explain what is covered by the warranty that comes with your products.",
    "Is an extended warranty offered on your products?",
    "Enter a text about the products you recommend in general.",
    "Enter a text about the more modern style products you recommend.",
    "Enter a text about the more traditional jewelry products you recommend.",
    "Enter a text about the low-cost products you recommend.",
    "Enter a text about the low-cost products you recommend for a child.",
    "Enter a text about the low-cost products you recommend for a girlfriend.",
    "Explain which products are especially suitable for formal occasions.",
    "Explain which products are especially suitable for the beach.",
    "Explain what to do if a product is out of stock.",
    "Explain how to replace the chain of a product.",
    "Describe what the products come with.",
    "Explain if the product comes with a gift box.",
    "Explain how your products can be modified.",
    "Explain if/how to engrave your jewelry products.",
    "Enter a response to use if the customer has a general complaint about your products.",
    "Explain what to do if the product has not been received.",
    "Explain what to do if a product is losing quality.",
    "Explain what to do if a product keeps breaking.",
    "Explain what to do if a product is peeling off.",
    "Explain what to do if a product is too tight.",
    "For when customers specifically ask about delivery time, please summarize for your domestic shipping options.",
    "Provide a summary of all your domestic shipping options, including details of the delivery times and costs.",
    "Provide a summary of all your international shipping options, including details of the delivery times and costs.",
    "For when customers specifically ask about delivery costs, please summarize for your international standard delivery options.",
    "Explain if/how you ship to any state.",
    "Explain where to find the delivery status.",
    "Explain what to do if the delivery status has not changed for a while.",
    "How long does it take to receive shipping confirmation?",
    "Explain what to do if you entered the wrong shipping information before the order has been shipped.",
    "Explain what to do if you entered the wrong shipping information after the order has been shipped.",
    "Explain what to do if the order was returned to the company because the apartment number was not included in the delivery address.",
    "Explain what a customer should do if they receive the wrong product.",
    "Explain what a customer should do if they receive the wrong color product or if they say the color looks different from what was advertised.",
    "Explain what to do if you only received one earring.",
    "Explain what to do if a package arrived late.",
    "Explain what to do if a product was damaged.",
    "Explain how to track an order.",
    "Explain about your domestic return delivery, including details of the time and cost.",
    "Explain about your international return delivery, including details of the time and cost.",
    "How much time does international return delivery normally take?",
    "Explain the process for returning only one of a pair of articles.",
    "Explain how returns work for products.",
    "Explain where to find the product return status.",
    "Explain the main points of your product exchange policy.",
    "Explain how much time the product exchange process normally takes.",
    "Explain how to exchange jewelry products.",
    "Explain any situations (and the charges) where you charge for exchanges of a discounted jewelry product.",
    "Explain what to do if you have not received any instructions when making an exchange.",
    "Explain what to do if the return portal is closed when making an exchange.",
    "Explain what to do if the order is not found when making an exchange."
]
        types = [
"list",
"yes_or_no",
"yes_or_no",
"yes_or_no",
"list",
"list",
"yes_or_no",
"yes_or_no",
"list",
"process",
"process",
"process",
"process",
"process",
"list",
"process",
"yes_or_no",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"yes_or_no",
"yes_or_no",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"yes_or_no",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"yes_or_no",
"process",
"yes_or_no",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"yes_or_no",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process",
"process"
]
        #answers = ask_questions_from_scraped_text(text, questions, types)
        collection_name = "hh_when_onboarding_b_10000_w_groups"
        answers = ask_questions_from_vdb(collection_name, questions, types)


        formatted_lines = [
    f"Q: {question}\nA: {answer}\n" for question, answer in zip(questions, answers)
]
        formatted_output = "\n".join(formatted_lines)

        make_conversation_file(formatted_output, f"conversation_logs/onboarding_w_vdb_{collection_name}")
        print(formatted_output)


if __name__ == '__main__':
    main()