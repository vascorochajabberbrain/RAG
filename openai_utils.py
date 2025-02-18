from dotenv import load_dotenv
import os
import openai

# Load environment variables from .env file
load_dotenv()

_openaiclient = None

def get_openai_client():
    global _openaiclient
    if _openaiclient is None:  # Only initialize if not already created
        # to-do put this on the .env variable
        _openaiclient = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    return _openaiclient
