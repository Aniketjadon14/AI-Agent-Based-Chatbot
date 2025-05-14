from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_docs(data_dir):
    all_chunks = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
    return all_chunks