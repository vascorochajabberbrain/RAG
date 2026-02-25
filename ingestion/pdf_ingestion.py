from PyPDF2 import PdfReader

from vectorization import get_text_chunks, get_embedding, insert_data, create_batches_of_text


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
    Extract raw text from a PDF file. Use this from the workflow.
    If silent=True, does not print the extracted text.
    """
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    if not silent:
        print(text)
    return text


def read_from_pdf_pages(path: str) -> list:
    """
    Extract text per page from a PDF file.
    Returns a list of dicts: [{page: int (1-based), text: str}, ...]
    Used for page-level source attribution in RAG responses.
    """
    result = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
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