"""
Phase 1: build the vector database from source PDFs.

Run once (or whenever the source PDFs change):
    python -m src.ingest
"""

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    PDF_FOLDER_PATH,
    CHROMA_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
)


def build_vector_db():
    print("Step 1: Loading Documents...")
    loader = PyPDFDirectoryLoader(PDF_FOLDER_PATH)
    raw_documents = loader.load()
    print(f"-> Loaded {len(raw_documents)} pages from the PDFs.")

    print("\nStep 2: Chunking Text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"-> Split documents into {len(chunks)} searchable chunks.")

    print("\nStep 3: Initializing Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("\nStep 4: Creating Vector Database (this may take a few minutes)...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    print(f"\nDone. Vector database saved to {CHROMA_DB_PATH}")
    return vector_db


if __name__ == "__main__":
    build_vector_db()