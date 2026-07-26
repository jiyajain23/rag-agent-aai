"""
Phase 1: build the vector database from source PDFs.

Run once (or whenever the source PDFs change):
    python -m src.ingest
"""

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import fitz  # PyMuPDF
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document-

from src.config import (
    PDF_FOLDER_PATH,
    CHROMA_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
)

ocr_engine = RapidOCR()

MIN_CHARS_THRESHOLD = 20  # below this, treat the page as scanned/image-only

def load_pdfs_with_ocr_fallback(pdf_folder_path: str) -> list[Document]:
    documents = []
    pdf_paths = sorted(Path(pdf_folder_path).glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDF files to process.\n")
    for pdf_path in pdf_paths:
        print(f" Processing: {pdf_path.name}...")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
        
            if len(text) < MIN_CHARS_THRESHOLD:
                # Likely a scanned page — rasterize and OCR it
                print(f"   └─ Page {page_num + 1}/{total_pages}: Running RapidOCR...")
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                result, _ = ocr_engine(img_bytes)
                text = "\n".join(line[1] for line in result) if result else ""
                source_type = "ocr"
            else:
                source_type = "digital"

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_num,
                        "extraction_method": source_type,
                    },
                )
            )
        print(f"   ✅ Completed {pdf_path.name} ({total_pages} pages)\n")
        doc.close()

    return documents

def build_vector_db():
    print("Step 1: Loading Documents...")
    raw_documents = load_pdfs_with_ocr_fallback(PDF_FOLDER_PATH)
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