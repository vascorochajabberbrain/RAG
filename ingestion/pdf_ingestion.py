from PyPDF2 import PdfReader

from vectorization import get_text_chunks, get_embedding, insert_data, create_batches_of_text


def read_data_from_pdf():
  pdf_path = 'ingestion/data_to_ingest/pdfs/caderno_de_receitas_do_mar.pdf'
  text = "" # for storing the extracted text
  with open(pdf_path, 'rb') as file:
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
      text += page.extract_text()
  print(text)
  return text


def main():
  text=read_data_from_pdf()
  batches = create_batches_of_text(text, 1000, 100)
  chunks = []
  for batch in batches:
      chunks += get_text_chunks(batch)
  return chunks
  #vectors=get_embedding(chunks)
  #insert_data(vectors)

  

if __name__ == '__main__':
    main()