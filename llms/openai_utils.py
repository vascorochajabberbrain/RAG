import time
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
        _openaiclient = openai.Client(api_key = os.getenv("OPENAI_API_KEY"))
    return _openaiclient

def clean_json_response(text):
    """Remove backticks and json tags from a GPT-typed json block."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()
    return text

def openai_chat_completion(prompt, text, model="gpt-4o-mini", max_tokens=None):
    """
    Function to get a chat completion from OpenAI's API.

    Args:
        prompt (str): The system prompt for the chat model.
        text (str): The user input text for the chat model.
        model (str): Model id. Default gpt-4o-mini.
        max_tokens (int|None): Hard cap on output tokens. Default None
            leaves OpenAI's per-model default (4096 for gpt-4o-mini),
            which is fine for short responses. Callers returning long
            structured JSON (e.g. FAQ extraction with N pairs) must
            set this explicitly — the default truncates around row
            ~35 in a 70-pair list, leaving the response un-parseable
            (and prompting confused operators).
    """
    start_time = time.time()
    openai_client = get_openai_client()

    # Build kwargs once so the retry branch uses the same args.
    create_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    }
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens

    try:
        completion = openai_client.chat.completions.create(**create_kwargs)
    except openai.RateLimitError as e:
        # Extract details from the error object
        error_data = e.response.json()["error"]

        print(error_data)

        # Check if it's a token-per-minute (TPM) limit
        if "tokens per min" in error_data.get("message", "").lower():
            print("Hit the TPM (tokens per minute) limit. Will wait a bit and try again")
            time.sleep(10)  # Wait for 10 seconds before retrying
            completion = openai_client.chat.completions.create(**create_kwargs)
    # Track token usage
    if hasattr(completion, 'usage') and completion.usage:
        from llms.token_tracker import record_usage
        record_usage(model, completion.usage.prompt_tokens, completion.usage.completion_tokens)
    end_time = time.time()
    print(f"Elapsed time measured locally: {end_time - start_time:.2f} seconds")
    return completion.choices[0].message.content.strip()

# Helper: Wait for run to complete
def wait_for_run_completion(thread_id, run_id, poll_interval=0.1, timeout=100):
    start_time = time.time()
    while True:
        run = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status == "completed":
            end_time = time.time()
            print(f"Elapsed time measured locally: {end_time - start_time:.2f} seconds")
            return run
        elif run.status in ["failed", "cancelled", "expired"]:
            raise Exception(f"Run failed with status: {run.status}")
        elif time.time() - start_time > timeout:
            raise TimeoutError("Run did not complete within timeout.")
        time.sleep(poll_interval)
        