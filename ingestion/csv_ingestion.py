import sys
import os

import json
import pandas as pd

from my_collections.SCS_Collection import SCS_Collection
from llms.openai_utils import openai_chat_completion
from qdrant_utils import create_collection
from ingestion.url_ingestion_legacy import scrape_page
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


def read_csv_to_chunks(path: str, config: dict = None) -> list:
    """
    Simple CSV → list of text chunks for the workflow.
    config can include: text_columns (list of column names to join), or default is to use all columns as string.
    Returns list of non-empty strings (one per row, or per cell if multiple text columns).
    """
    config = config or {}
    df = pd.read_csv(path)
    chunks = []
    text_columns = config.get("text_columns")
    if text_columns:
        for _, row in df.iterrows():
            parts = [str(row[c]) for c in text_columns if c in df.columns and pd.notna(row.get(c))]
            if parts:
                chunks.append(" ".join(parts))
    else:
        for _, row in df.iterrows():
            parts = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
            if parts:
                chunks.append(" ".join(parts))
    return chunks


def csv_ingestion(collection: SCS_Collection):
    
    # Load the CSV file
    df = pd.read_csv('ingestion/data_to_ingest/csvs/Autoderm Content jB - Sheet1.csv')

    chunks = []
    things_that_went_wrong = []

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

    #create_collection("autoderm_with_order")
    
    
    chunks_count = 0
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
                case "Condition":
                    continue
                case "Tag":
                    if value == "Never":
                        chunks.append(f"{condition} never appears on the genital areas.")
                    if value == "Can be":
                        chunks.append(f"{condition} may appear on the genital areas.")
                    if value == "Only":
                        chunks.append(f"{condition} only appears on the genital areas.")
                case "Never on kids":

                    if value == "Never":
                        chunks.append(f"{condition} never occurs on kids.")
                    else:
                        chunks.append(f"{condition} may occur on kids.")
                case "Small resume" | "Resume":

                    if pd.isna(value):
                        continue
                    chunks.append(value)
                case "Boots Links":

                    try:
                        if value == "":
                            print(f"Skipping scrapping for {condition} as there is no link provided.")
                            continue
                        text = scrape_page(value)
                        print(f"Scraped {len(text)} characters from {condition} Boots Links")

                        additional_considerations_when_scraping = "Do not cut any name of the condition, as two different conditions may have two out of three names the same, but we really do not want to create wrong information."
                        link_chunks=get_text_chunks(text, additional_considerations_when_scraping)
                        print(f"Scraped {len(link_chunks)} chunks, going to filter them")

                        filtered_link_chunks = filter_boots_chunks(link_chunks, condition)
                        #chunks.extend(filtered_link_chunks)
                        print(f"Scraped {len(filtered_link_chunks)} chunks after filetering, out of {len(link_chunks)}")
                        
                        #n_rows_scraped += 1
                        
                        #points=get_points_with_source(filtered_link_chunks, value, condition, chunks_count)
                        #insert_data(points, collection_name="autoderm_with_order")
                        collection.append_sentences(filtered_link_chunks, source=value)

                        chunks_count += len(filtered_link_chunks)
                        print(f"Scrapped data inserted with no errors: {len(filtered_link_chunks)} chunks")
                        continue
                    except Exception as e:
                        things_that_went_wrong.append(f"Error scraping {condition}: {e}")
                case "Generic Over the counter medication" | "Brand name Over the counter medication" | "Generic Prescription name" | "Brand Prescription name":
                    
                    if pd.isna(value) or value == "None" or "None" in value or "N/A" in value:
                        continue
                    if value in ["Depends", "Excision if suspicious", "Surgical excision if symptomatic", "Incision and drainage (if inflamed)", "Treat underlying cause", "Surgical excision, Immunotherapy", "Surgical removal if needed", "Electrocautery, Surgical excision", "Cryotherapy, surgical removal"]:
                        chunks.append(openai_chat_completion(
                            "You are an assistant to make more reasonable sentences. They were formed on an automatic, going throught a table and the wording sometimes is incorrect but the content it is correct." \
                            "Please rephrase the following text to make it a proper sentence maintaining the content." \
                            "Answer only with the rephrased text, do not add any additional text or explanation." ,
                            f"For {condition} there are no really {column} because you should {value}"
                        ))
                    else:
                        chunks.append(openai_chat_completion(
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
        

        
        if chunks != []:
            try:
                #points=get_points(chunks, condition, chunks_count)
                #insert_data(points, collection_name="autoderm_with_order")
                collection.append_sentences(chunks)
                print(f"Table data inserted with no errors: {len(chunks)} chunks")
                chunks_count += len(chunks)
            except Exception as e:
                things_that_went_wrong.append(f"Error inserting table data for {condition}: {e}")
        
            chunks = []
    '''
    print(chunks)
    y_n = input("You like the chunks? (y/n) ")
    if y_n.lower() != "y":
        print("Exiting without inserting data.")
        return
                    
    points=get_points(chunks)
    insert_data(points, collection_name="autoderm_alpha")
    '''
    print(things_that_went_wrong)

def main():
    print("running this file now does nothing")

if __name__ == '__main__':
    main()