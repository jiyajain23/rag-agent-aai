"""
Retrieval layer: hybrid (BM25 + vector) search, then cross-encoder reranking.

BM25 catches exact-term / acronym queries (e.g. "ARC") that pure semantic
search can miss. Vector search catches paraphrased/semantic queries BM25
misses. The reranker then filters the merged pool down to the most relevant
chunks, which is what removes false positives like a BM25 hit on the word
"contact" in an unrelated phraseology glossary entry.
"""

from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
    BM25_K,
    VECTOR_K,
    HYBRID_WEIGHTS,
    RERANK_TOP_N,
    RERANK_MODEL_NAME,
)


class RerankRetriever(BaseRetriever):
    """Wraps a base retriever and reranks its results with a cross-encoder."""

    base_retriever: BaseRetriever
    reranker: CrossEncoder
    top_n: int = RERANK_TOP_N

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[: self.top_n]]


def load_vector_db() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)


def build_reranked_hybrid_retriever(vector_db: Chroma) -> RerankRetriever:
    # Pull chunks back out of the persisted store so BM25 doesn't depend on
    # having the original `chunks` list in memory (keeps this script standalone).
    raw = vector_db.get()
    chunk_docs = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(raw["documents"], raw["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(chunk_docs)
    bm25_retriever.k = BM25_K

    vector_retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_K})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=HYBRID_WEIGHTS,
    )

    reranker = CrossEncoder(RERANK_MODEL_NAME)

    return RerankRetriever(base_retriever=hybrid_retriever, reranker=reranker, top_n=RERANK_TOP_N)