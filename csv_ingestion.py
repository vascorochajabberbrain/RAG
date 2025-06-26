import sys

import json
import pandas as pd

from openai_utils import openai_chat_completion
from url_ingestion import scrape_page
from vectorization import get_points, get_points_with_source, get_text_chunks, insert_data


def filter_non_condition_chunks(chunks, condition):
    # Filters out chunks that do not have to do with the condition
    prompt = f"""You are a helpful assistant that filters OUT prepositions that do not have to do with the condition: {condition}. 
    Some examples of prepositoins that do not have to do with the condition {condition} may be about:
        - Information about a professional
        - Information about a place
    So, return only the chunks that are related to the condition {condition}.
    Return them as you received them, as a valid array of strings
    Do not provide any additional text or explanation — just the array of strings.
    The array should follow this format:
    [
    "string 1",
    "string 2",
    "string 3"
    ]"""
    #print(f"chunks bedore {json.dumps(chunks)}")
    response = openai_chat_completion(prompt, json.dumps(chunks))
    #print(f"response: {response}")
    
    return json.loads(response)

def filter_boots_chunks(chunks, condition):
    chunks = filter_non_condition_chunks(chunks, condition)
    return chunks


def main():
    
    # Load the CSV file
    df = pd.read_csv('Autoderm Content jB - Sheet1.csv')

    chunks = []

    #Nameing the two columns that had no name
    df.columns.values[3] = "Small resume"
    df.columns.values[4] = "Resume"

    n_rows = len(df)

    #additional limits if the upload is causing problems, quick fix
    '''
    #n_rows_scraped = int(sys.argv[1])
    n_rows_scraped = 72
    n_rows_limit = n_rows_scraped +1
    '''

    # Iterate through each row and column
    for index, row in df.iterrows():

        #Limits for the quick fix
        '''
        if index < n_rows_scraped:
            print(f"Skipping row {index + 1}/{n_rows} as it has already been scraped.")
            continue
        if n_rows_scraped >= n_rows_limit:
            print(f"Reached the limit of {n_rows_limit} rows scraped, stopping.")
            break
        '''


        condition = row['Condition']
        condition = condition.replace("_", " ")  # Replace underscores with spaces for better readability
        print(f"Condition {condition} ({index + 1}/{n_rows})")

        for column in df.columns:
            value = row[column]

            match column:
                case "Condition" | "Tag":
                    continue
                #TO-DO: ask what exactly is the meaning of the NEVER ON KIDS column, so that I can write a proper sentence about it
                case "Never on kids":
                    continue # WARNING: this continue is only until we are finalizing the scraping
                    if value == "Never":
                        chunks.append(f"{condition} never occurs on kids")
                    else:
                        chunks.append(f"{condition} may occur on kids")
                case "Small resume" | "Resume":
                    continue # WARNING: this continue is only until we are finalizing the scraping
                    if pd.isna(value):
                        continue
                    chunks.append(value)
                case "Boots Links":
                    continue
                    try:
                        text = scrape_page(value)
                        print(f"Scraped {len(text)} characters from {condition} Boots Links")
                        additional_considerations_when_scraping = "Do not cut any name of the condition, as two different conditions may have two out of three names the same, but we really do not want to create wrong information."
                        link_chunks=get_text_chunks(text, additional_considerations_when_scraping)
                        print(f"Scraped {len(link_chunks)} chunks, going to filter them")
                        filtered_link_chunks = filter_boots_chunks(link_chunks, condition)
                        chunks.extend(filtered_link_chunks)
                        print(f"Scraped {len(filtered_link_chunks)} chunks after filetering, out of {len(link_chunks)}")
                        n_rows_scraped += 1
                        points=get_points_with_source(chunks, value)
                        insert_data(points, collection_name="autoderm_alpha")
                        print("Data inserted with no errors")
                        continue
                    except Exception as e:
                        print(f"""Something went wrong, uploading the new chunks anyway
                              Remember to update the n_rows_scraped to {n_rows_scraped}""")
                case "Generic Over the counter medication" | "Brand name Over the counter medication" | "Generic Prescription name" | "Brand Prescription name":
                    continue
                    if pd.isna(value) or value == "None" or "None" in value or "N/A" in value:
                        continue
                    if value in ["Depends", "Excision if suspicious", "Surgical excision if symptomatic", "Incision and drainage (if inflamed)", "Treat underlying cause", "Surgical excision, Immunotherapy", "Surgical removal if needed", "Electrocautery, Surgical excision", "Cryotherapy, surgical removal"]:
                        aux_chunks.append(openai_chat_completion(
                            "You are an assistant to make more reasonable sentences. They were formed on an automatic, going throught a table and the wording sometimes is incorrect but the content it is correct." \
                            "Please rephrase the following text to make it a proper sentence maintaining the content." \
                            "Answer only with the rephrased text, do not add any additional text or explanation." ,
                            f"For {condition} there are no really {column} because you should {value}"
                        ))
                    else:
                        aux_chunks.append(openai_chat_completion(
                            "You are an assistant to make more reasonable sentences. They were formed on an automatic, going throught a table and the wording sometimes is incorrect but the content it is correct." \
                            "Please rephrase the following text to make it a proper sentence maintaining the content." \
                            "Answer only with the rephrased text, do not add any additional text or explanation.",
                            f"The {column} for {condition} is {value}"))
                case "Selfcare":
                    chunks.append(openai_chat_completion(
                        "You are an assistant to make more reasonable sentences. They were formed on an automatic, going throught a table and the wording sometimes is incorrect but the content it is correct." \
                        "Please rephrase the following text to make it a proper sentence maintaining the content." \
                        "Answer only with the rephrased text, do not add any additional text or explanation.",
                        f"For {condition} the selfcare you must have is {value}"))
                    continue
                case _:
                    continue
        
        '''
        if aux_chunks != []:
            points=get_points(aux_chunks)
            insert_data(points, collection_name="autoderm_alpha")
        '''

    print(chunks)
    y_n = input("You like the chunks? (y/n) ")
    if y_n.lower() != "y":
        print("Exiting without inserting data.")
        return
                    
    points=get_points(chunks)
    insert_data(points, collection_name="autoderm_alpha")


if __name__ == '__main__':
    main()