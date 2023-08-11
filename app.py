from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit

from db_create import db_create
from doc_qa import doc_qa
from text_splitter import text_splitter


def main():
    load_dotenv()
    streamlit.set_page_config(page_title="CramAI")
    streamlit.header("CramAI 📚")
    
    pdf = streamlit.file_uploader("Upload a PDF", type="pdf")
    if pdf:
      pdf_extracted = PdfReader(pdf)
      
      # Convert PDF to a string
      text = ""
      for page in pdf_extracted.pages:
        text += page.extract_text()
      
      # Split string into chunks
      chunks = text_splitter.split_text(text)

      # Generate vector database of chunks
      db = db_create(chunks)

      question = streamlit.text_input("Ask me a question about the content in this PDF:")
      if question:
        # Perform similarity search over database with question
        similar_documents = db.similarity_search(question)

        # Pass documents to LLM to answer question
        streamlit.write(doc_qa(similar_documents, question))

if __name__ == '__main__':
    main()
