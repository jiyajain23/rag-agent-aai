# ✈️ AIM Docs RAG Assistant

LIVE DEMO : https://rag-agent-jiya.streamlit.app/

A retrieval-augmented generation (RAG) chatbot that answers questions over the **Airports Authority of India (AAI) Manual of Air Traffic Services (MATS) Part 1 and other circulars (check out data/aim-docs to see)**, 

Ask it a question about the manual — ARC procedures, ATCO eligibility, SAR training, separation minima — and it retrieves the relevant pages, cites them, and answers strictly from that context (refusing to guess when the manual doesn't cover something).

> ⚠️ **Disclaimer:** Built for personal learning and demonstration purposes only. This is not an authoritative source — always defer to the official MATS document for operational decisions.

---

## Demo

```
Q: What is an Abnormal Runway Contact (ARC)?
A: An Abnormal Runway Contact (ARC) is any landing or takeoff involving abnormal
   runway or landing surface contact. This includes events such as:
   - Hard/heavy landings
   - Long/fast landings
   - Off-center landings
   - Crabbed landings
   - Nose wheel first touchdown
   - Tail strikes
   - Wingtip/nacelle strikes.

   Sources: Manual of Air Traffic Services Part-1 Edition 7.pdf (page 24)

Q: Does that include helicopters?
A: I don't know. The context provided does not mention helicopters.
```

*(Second answer is a deliberate feature, not a failure — the system refuses to hallucinate beyond what the retrieved context supports.)*

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────┐
│ History-aware retriever  │  reformulates follow-up questions using chat history
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│   Hybrid retrieval        │  BM25 (keyword) + vector (semantic) search, weighted 0.3 / 0.7
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│  Cross-encoder reranker   │  re-scores candidates for true relevance, keeps top 3
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│   LLM (Groq / Llama 3.1)  │  answers strictly from retrieved context, cites sources
└─────────────────────────┘
```

**Why hybrid + reranking, not just vector search?** Pure semantic search missed exact-term/acronym queries (e.g. "ARC"). Adding BM25 fixed that but introduced its own false positives (keyword overlap without meaning — see [Problem 4](#4-bm25-false-positive-on-a-glossary-page) below). Reranking with a cross-encoder was the fix that got both benefits without the noise.

---

## Tech Stack

| Component        | Choice                                            | Why |
|-------------------|---------------------------------------------------|-----|
| PDF loading       | PyPDFDirectoryLoader (pypdf)                      | Simple, works for digital-text PDFs |
| Chunking          | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) | Balances context retention vs. retrieval precision — see chunk-size experiment below |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2`          | Small, fast, free, runs locally — no API cost |
| Vector store      | Chroma                                            | Simple local persistence, good LangChain support |
| Keyword search    | BM25 (`rank_bm25`)                                | Catches exact-term/acronym queries semantic search misses |
| Reranking         | `cross-encoder/ms-marco-MiniLM-L-6-v2`            | Filters hybrid retrieval noise by scoring true query-chunk relevance |
| LLM               | Groq (`llama-3.1-8b-instant`)                     | Fast, free-tier friendly for prototyping |
| Orchestration     | LangChain (LCEL) + LangChain Classic              | Standard RAG chain patterns; migrated off deprecated `RetrievalQA` — see Problem 2 |
| UI                | Streamlit                                         | Fast to stand up a real chat interface |
| Testing           | pytest                                            | Regression tests for retrieval correctness |
| Evaluation        | Custom keyword-overlap scorer                     | Lightweight sanity check, not a substitute for human eval |

---

## Repository Structure

```
aim-rag-assistant/
├── app.py                     # Streamlit entrypoint (chat UI)
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml           # GROQ_API_KEY (gitignored)
│
├── src/
│   ├── config.py               # all tunable constants in one place
│   ├── ingest.py                # load PDFs → chunk → embed → persist Chroma
│   ├── retrieval.py              # hybrid retriever + reranker wrapper
│   └── chain.py                   # conversational RAG chain (history-aware + grounded QA)
│
├── data/
│   ├── raw_pdfs/               # source PDFs (gitignored)
│   └── chroma_db/               # persisted vector store (gitignored, rebuild via src/ingest.py)
│
├── notebooks/
│   └── rag-agent-notebook.ipynb  # original exploration notebook (chunking tests, debugging)
│
├── eval/
│   ├── eval_questions.json     # test set with expected keywords
│   └── run_eval.py               # runs eval set through the chain, scores answers
│
└── tests/
    └── test_retrieval.py        # pytest regression tests (retrieval correctness, no bare-LLM-only checks)
```

---

## Setup & Installation

**1. Clone and create a virtual environment**
```bash
git clone <your-repo-url>
cd aim-rag-assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt

streamlit run app.py 
