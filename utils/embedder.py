import os
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document  # Import Document class from langchain

def load_and_split_docs(directory):
    """
    This function loads all PDF files from the given directory, extracts text from them, 
    and splits them into chunks.
    """
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                # Split the text into chunks (you can customize how you split it)
                chunk_size = 1000  # Set chunk size (can be adjusted)
                for i in range(0, len(text), chunk_size):
                    # Wrap each chunk in a Document object
                    chunks.append(Document(page_content=text[i:i + chunk_size]))
    return chunks

def embed_docs():
    """
    This function loads documents, generates embeddings, and saves them in ChromaDB.
    """
    # Load and split documents from the 'data' folder
    chunks = load_and_split_docs("data")
    
    # Initialize the embedding function with Ollama model
    embedding_fn = OllamaEmbeddings(model="llama3")
    
    # Create a Chroma vector store and persist the embeddings
    Chroma.from_documents(documents=chunks, embedding=embedding_fn, persist_directory="embeddings/chroma")
    
    print("✅ Embeddings created and saved in ChromaDB")

if __name__ == "__main__":
    # Call the embed_docs function to generate embeddings
    embed_docs()
