import io
from typing import Union

import httpx
from PyPDF2 import PdfReader

from vectorization import get_text_chunks, get_embedding, insert_data, create_batches_of_text


def _open_pdf(path_or_url: str):
    """Return a PdfReader for either a local path or an http(s) URL.

    Remote PDFs are downloaded in-memory so every ingestion run picks
    up the latest version — jBKB's content_raw_hash change detection
    then decides whether to re-chunk. No local caching: a stale cache
    would defeat the whole point of URL-based auto-refresh.
    """
    if path_or_url.startswith(('http://', 'https://')):
        resp = httpx.get(path_or_url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        return PdfReader(io.BytesIO(resp.content))
    return PdfReader(open(path_or_url, 'rb'))


def read_data_from_pdf(path):
  pdf_path = path
  text = "" # for storing the extracted text
  with open(pdf_path, 'rb') as file:
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
      text += page.extract_text()
  print(text)
  return text


def read_from_pdf(path: str, silent: bool = False) -> str:
    """
    Extract raw text from a PDF file or remote URL.
    If silent=True, does not print the extracted text.
    """
    reader = _open_pdf(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    if not silent:
        print(text)
    return text


def read_from_pdf_pages(path: str) -> list:
    """
    Extract text per page from a PDF (local file path or http(s) URL).
    Returns a list of dicts: [{page: int (1-based), text: str}, ...]
    Used for page-level source attribution in RAG responses.
    Remote URLs are downloaded in-memory each call so Build always
    reflects the latest remote PDF.
    """
    reader = _open_pdf(path)
    result = []
    for i, page in enumerate(reader.pages, start=1):
        result.append({"page": i, "text": page.extract_text() or ""})
    return result


def run_pdf_pipeline(path: str, batch_size: int = 10000, overlap: int = 100):
    """Read PDF, chunk with LLM, return (chunks, source_label)."""
    text = read_from_pdf(path, silent=True)
    batches = create_batches_of_text(text, batch_size, overlap)
    chunks = []
    for batch in batches:
        chunks += get_text_chunks(batch)
    import os
    source_label = os.path.basename(path)
    return chunks, source_label


def main():
  pdf_path = 'ingestion/data_to_ingest/pdfs/'
  pdf_source = 'caderno_de_receitas_do_mar.pdf'
  text=read_data_from_pdf(pdf_path + pdf_source)
  batches = create_batches_of_text(text, 1000, 100)
  chunks = []
  for batch in batches:
      chunks += get_text_chunks(batch)
  return chunks, pdf_source
  #vectors=get_embedding(chunks)
  #insert_data(vectors)

  

if __name__ == '__main__':
    main()