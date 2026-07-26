# AIM Docs RAG Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions over the **Airports Authority of India (AAI) Manual of Air Traffic Services (MATS) Part 1**, built as a personal project to apply concepts from IBM's *Fundamentals of RAG* and *AI Agents* courses in a domain relevant to my internship at AAI.

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

**Why hybrid + reranking, not just vector search?** Pure semantic search missed exact-term/acronym queries (e.g. "ARC"). Adding BM25 fixed that but introduced its own false positives (keyword overlap without meaning — see [Problem 3](#4-bm25-false-positive-on-a-glossary-page) below). Reranking with a cross-encoder was the fix that got both benefits without the noise.

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
│   ├── aim-docs/               # source PDFs (gitignored)
│   └── chroma_db/               # persisted vector store (gitignored, rebuild via src/ingest.py)
│
├── notebooks/
│   └── rag-agent.ipynb         # original exploration notebook (chunking tests, debugging)
│
├── eval/
│   ├── eval_questions.json     # test set with expected keywords
│   └── run_eval.py               # runs eval set through the chain, scores answers
│
└── test/
    └── test_retrieval.py        # pytest regression tests (retrieval correctness, no bare-LLM-only checks)
└── test-results/
    └── eval.csv                # csv format of eval results
    └── eval.txt                # text format of eval results
    └── results.xml             # xml format of test results
    └── test_log.txt            # log of test results
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
```

**3. Add your PDFs**
Place source PDFs in `data/raw_pdfs/`. (Alternatively, if you already have a built `chroma_db/`, copy it directly into `data/chroma_db/` and skip step 4.)

**4. Build the vector store**
```bash
python -m src.ingest
```
This loads the PDFs, chunks them, embeds them, and persists a Chroma DB to `data/chroma_db/`. Takes a few minutes depending on corpus size.

**5. Add your Groq API key**
    Create a .streamlit/secrets.toml file if not already exists.
Edit the new file:
```toml
GROQ_API_KEY = "your-real-key-here"
```

**6. Run the app**
```bash
streamlit run app.py
```
Opens at `localhost:8501`.

**7. (Optional) Run tests**
```bash
pytest tests/test_retrieval.py -v
```

**8. (Optional) Run evaluation**
```bash
python -m eval.run_eval 
```

---

## Development Journey: Problems Encountered & Solutions

This section documents the actual debugging process, kept here deliberately because the problems were often more instructive than the final working code.

### 1. `RetrievalQA` → LCEL migration
**Problem:** `RetrievalQA` hides the prompt template by default, offers no easy path to add conversational memory, and is being phased out.

**Solution:** Rebuilt the chain with `create_retrieval_chain(retriever, combine_docs_chain)`, using an explicit prompt template requiring the model to answer only from retrieved context and say "I don't know" otherwise. This also unblocked adding conversational memory later (Problem 4).

### 2. Chunk size selection
**Problem:** No principled reason to pick 1000/200 for `chunk_size`/`chunk_overlap` — it was a default carried over from a tutorial.

**Solution:** Ran a controlled comparison across 500/100, 1000/200, and 1500/300 configurations, comparing retrieved chunks for the same query. **Learning:** raw similarity *scores* aren't directly comparable across chunk sizes (smaller chunks naturally score "better" since there's less irrelevant text diluting the embedding) — the real comparison has to be on retrieved *content quality*, not the score alone. Also discovered a subtler bug during this: the comparison initially used disposable in-memory Chroma collections instead of the persisted production DB, producing misleadingly different (worse) results than the actual pipeline — a good reminder to always verify you're testing the same index you're shipping.

### 3. BM25 false positive on a glossary page
**Problem:** After adding hybrid (BM25 + vector) retrieval to catch exact-term/acronym queries, the query *"What is an Abnormal Runway Contact (ARC)?"* started returning a chunk from **Chapter 12: Phraseologies** (page 325) — a glossary entry defining the ATC radio term **"CONTACT: Establish communication with..."** — completely unrelated to runway contact. BM25 matched purely on the shared word "contact," with no understanding of meaning.

**Solution:** Added a cross-encoder reranking step on top of the hybrid retriever. The base retrievers now cast a wider net (`k=6` each), and the reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each `(query, chunk)` pair for genuine relevance before narrowing to the final top 3 — filtering out keyword-only matches like the phraseology page. Also rebalanced ensemble weights from 50/50 to 30/70 (BM25/vector), leaning on the more semantically reliable retriever as the primary signal.

**Follow-up:** This bug was turned into a permanent regression test (`tests/test_retrieval.py::test_arc_query_excludes_phraseology_false_positive`) asserting page 325 never reappears in ARC query results — so if retrieval weights or `top_n` are changed later and the bug resurfaces, the test suite catches it immediately.

### 4. Conversational memory
**Problem:** Original pipeline was stateless — every query was independent, so a natural follow-up like *"Does that include helicopters?"* had no way to resolve "that" without the prior turn.

**Solution:** Added `create_history_aware_retriever`, which reformulates follow-up questions into standalone queries using chat history before retrieval runs. Verified with a real two-turn exchange: the follow-up correctly resolved to "does Abnormal Runway Contact include helicopters" and the system appropriately answered "I don't know" rather than fabricating an answer, since the manual doesn't address it.

### 5. Ambiguous acronym collision ("ARC")
**Problem:** A test asserting that `"What is ARC?"` retrieves runway-contact content failed — the retriever instead returned content from **Chapter 5: Separation Methods and Minima**, about **DME Arc**, an unrelated navigation/approach procedure that also happens to be abbreviated "ARC" in the same manual.

**Solution:** This wasn't a retrieval bug — it's a genuine terminology collision in the source corpus (the same acronym legitimately means two different things in different chapters). The bare acronym alone is insufficiently specific for the retriever to disambiguate, same as it would be for a human skimming the manual. Fixed by writing the test query as a real user would to get the intended meaning ("What is an Abnormal Runway Contact?"), and documented the ambiguity explicitly in the test docstring as a known corpus characteristic rather than a defect.

### 6. Scanned PDF pages (OCR)
**Problem:** Some source PDFs contain scanned image pages rather than embedded digital text, which `pypdf`-based loaders silently return as empty or near-empty text for — meaning those pages would be invisible to retrieval with no error or warning.

**Solution:** Replaced the loader with a page-by-page PyMuPDF (`fitz`) pipeline that checks extracted text length per page; pages below a `MIN_CHARS_THRESHOLD` are rasterized and passed to RapidOCR as a fallback. Added an `extraction_method` field (`"digital"` / `"ocr"`) to each document's metadata, so any future answer-quality issue can be traced back to whether it came from an OCR'd page.

---

## Evaluation Results

Run via `eval/run_eval.py`, scoring each answer by keyword overlap against an expected-keyword set (crude but useful as a regression signal — not a substitute for careful human review).

| # | Question                                              | Score | Keywords Found |
|---|--------------------------------------------------------|-------|-----------------|
| 0 | What is an Abnormal Runway Contact (ARC)?              | 1.0   | runway, contact |
| 1 | What does ATS stand for?                                | 1.0   | air traffic service |
| 2 | What data does ATS units need to collect?                | 1.0   | traffic sample data, large height deviation |
| 3 | Eligibility criteria for ATCO                            | 1.0   | class III medical assessment, one year of experience |
| 4 | The SAR presentation in the Refresher Training…          | 1.0   | sar system components, sar cooperation |
| 5 | ATS units                                                 | 1.0   | flight information service, air traffic advisory |

**Average score: 100%**

Note: this is a small (n=6), self-authored eval set on straightforward factual questions — a strong signal that the core pipeline works, but not a rigorous benchmark. See [Future Improvements](#future-improvements) for how this could be extended.

---

## Testing

`tests/test_retrieval.py` runs through the real app pipeline (`app.load_pipeline()`), so it validates the actual production wiring — retrieval, reranking, and LLM response — not just the retriever in isolation. Because of this, tests make real Groq API calls; they're a correctness check, not a fast unit-test loop.

| Test | Verifies |
|------|----------|
| `test_pipeline_initialization` | Chain builds without error |
| `test_retrieval_returns_context` | A standard query returns non-empty context |
| `test_retrieved_documents_have_metadata` | Retrieved chunks carry `source`/`page` metadata for citation |
| `test_arc_query_context_relevance` | ARC (spelled out) query retrieves runway/landing-relevant content |
| `test_arc_query_excludes_phraseology_false_positive` | Regression guard for the Problem 3 bug (page 325 should never reappear) |

```bash
pytest tests/test_retrieval.py -v
```

---

## Known Limitations

- **Ambiguous acronyms:** terms like "ARC" have multiple valid meanings within the corpus (Abnormal Runway Contact vs. DME Arc); a bare acronym query can retrieve either, depending on context. Full-phrase queries disambiguate reliably.
- **Small eval set:** 6 questions is enough to catch obvious regressions, not enough to claim broad accuracy.
- **OCR quality on scanned pages is unverified at scale:** RapidOCR fallback is implemented but hasn't been benchmarked against a large sample of scanned pages — quality may vary on tables, stamps, or rotated scans.
- **BM25/vector weighting (0.25/0.75) is manually tuned, not optimized:** chosen based on observed behavior on a handful of queries, not a systematic sweep.
- **Not authoritative:** answers are generated by an LLM from retrieved excerpts and can still be incomplete or misphrased relative to the source manual — always verify against the original document for operational use.

---

## Future Improvements

- Expand the eval set to 20–30 questions covering more manual sections, plus multi-hop and out-of-scope questions to test refusal behavior more thoroughly.
- Benchmark OCR output quality on a sample of scanned pages against manual transcription.
- Systematically sweep hybrid retrieval weights and reranker `top_n` against the eval set rather than tuning by inspection.
- Add multi-query retrieval (LLM-generated query rephrasings merged before reranking) to improve recall on vaguely worded questions.

---