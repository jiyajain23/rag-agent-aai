import sys
from pathlib import Path
import pytest

# Add root folder to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app import load_pipeline


@pytest.fixture(scope="module")
def pipeline():
    """Load RAG chain once for all test cases."""
    return load_pipeline()


def test_pipeline_initialization(pipeline):
    """Verify that the chain initializes properly."""
    assert pipeline is not None


def test_retrieval_returns_context(pipeline):
    """Verify that a standard query retrieves context chunks."""
    query = "What is a Clearance Delivery (CLD)?"
    result = pipeline.invoke({"input": query, "chat_history": []})
    assert "context" in result
    assert len(result["context"]) > 0


def test_retrieved_documents_have_metadata(pipeline):
    """Verify that retrieved context documents contain source and page metadata."""
    query = "What does ATS stand for?"
    result = pipeline.invoke({"input": query, "chat_history": []})
    for doc in result["context"]:
        assert hasattr(doc, "metadata")
        assert "page" in doc.metadata or "source" in doc.metadata


def test_arc_query_context_relevance(pipeline):
    """
    Verify that a runway-contact ARC query retrieves chunks containing
    relevant terms.

    Note: the bare acronym "ARC" is ambiguous in this corpus -- it's also
    used for "DME Arc" (a navigation/approach procedure, Chapter 5:
    Separation Methods and Minima), which is unrelated to "Abnormal Runway
    Contact". "What is ARC?" with no other context can legitimately retrieve
    either meaning, so the test query is written out in full to match how a
    real user would disambiguate it.
    """
    query = "What is an Abnormal Runway Contact?"
    result = pipeline.invoke({"input": query, "chat_history": []})
    all_retrieved_text = " ".join([d.page_content for d in result["context"]]).lower()

    # Verify that either 'runway' or 'landing' was fetched by the retriever
    assert "runway" in all_retrieved_text or "landing" in all_retrieved_text


def test_arc_query_excludes_phraseology_false_positive(pipeline):
    """
    Regression test for a BM25 false-positive bug found during development:
    the hybrid retriever used to return a phraseology glossary chunk (page 325)
    for the ARC query, matched only on the unrelated word 'CONTACT' (as in
    "establish communication with..."), not the actual ARC definition.
    Reranking fixed this -- this test guards against it coming back if the
    hybrid weights, k values, or reranker are changed later.
    """
    query = "What is an Abnormal Runway Contact (ARC)?"
    result = pipeline.invoke({"input": query, "chat_history": []})
    pages = [d.metadata.get("page") for d in result["context"]]

    assert 325 not in pages, (
        "Page 325 (phraseology glossary, false positive on the word 'CONTACT') "
        "reappeared in results -- check hybrid retriever weights / reranker top_n."
    )