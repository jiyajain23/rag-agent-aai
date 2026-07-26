
# --- Paths ---
CHROMA_DB_PATH = "data/chroma_db"
PDF_FOLDER_PATH="data/aim-docs"
# --- Chunking (settled on after comparing 500/1000/1500 in the exploration notebook) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval ---
BM25_K = 6           # widen the candidate pool before reranking
VECTOR_K = 6
HYBRID_WEIGHTS = [0.25, 0.75]   # [bm25, vector] — leans semantic after BM25 false-positive on phraseology chunk
RERANK_TOP_N = 3
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- LLM ---
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2