from ingestion.csv_ingestion import csv_ingestion
from my_collections.SCS_Collection import SCS_Collection


def ingestion_menu(collection: SCS_Collection):

    menu = """Select an action:
    -- "q" to quit
    -- "csv" to ingest a csv file
    -- "pdf" to ingest a pdf file
    -- "txt" to ingest a txt file
    -- "url" to ingest from scrapping a website
"""

    action = input(menu)

    while action != "q":
        match action:
            case "csv":
                #file = input("Name of the file:")
                #for now the everything is done inside the function... even the name of the collection
                csv_ingestion(collection)
            case "pdf":
                print("To-do")
                #pdf_ingestion()
            case "txt":
                print("To-do")
                #txt_ingestion()
            case "url":
                print("To-do")
                #url_ingestion()
        
        action = input(menu)