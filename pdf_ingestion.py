from PyPDF2 import PdfReader

from vectorization import get_text_chunks, get_embedding, insert_data





def read_data_from_pdf():
  pdf_path = './cores_erradas_da_fruta_2.pdf'
  text = "" # for storing the extracted text
  with open(pdf_path, 'rb') as file:
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text


def main():
  get_raw_text=read_data_from_pdf()
  chunks=get_text_chunks(get_raw_text)
  vectors=get_embedding(chunks)
  insert_data(vectors)

  

if __name__ == '__main__':
    main()