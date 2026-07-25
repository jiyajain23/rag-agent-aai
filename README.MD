# ✈️ AAI AIM Docs RAG Assistant

A production-grade Retrieval-Augmented Generation (RAG) application built to intelligently query and summarize the Airports Authority of India (AAI) Manual of Air Traffic Services (MATS) and other operational aviation documents. 

##  Features

* **Hybrid Retrieval System:** Combines dense semantic search (`ChromaDB`) with sparse keyword matching (`BM25`) to accurately fetch both broad concepts and specific aviation acronyms (e.g., ARC, NOTAM, ILS).
* **Two-Stage Reranking:** Implements a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to intelligently re-rank the retrieved chunks based on deep contextual relevance before feeding them to the LLM.
* **Conversational Memory:** Context-aware chat history allows users to ask natural, follow-up questions without losing the thread of the conversation.
* **Strict Source Grounding:** Prevents hallucinations by forcing the LLM to answer *only* from the provided context or admit it does not know.
* **Citation UI:** Transparently displays the exact document name and page number for every retrieved fact.

---

##  Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain (`langchain-classic`, `langchain-core`)
* **LLM:** Meta Llama 3.1 8B (via Groq API for ultra-fast inference)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Vector Database:** ChromaDB
* **Keyword Indexing:** Rank-BM25
* **Cross-Encoder:** Sentence-Transformers

---

##  Setup & Installation

### 1. Prerequisites
* Python 3.9+
* A free [Groq API Key](https://console.groq.com/)

### 2. Clone and Install
Clone this repository and install the required dependencies:
```bash
git clone <your-repo-url>
cd <your-repo-directory>
pip install -r requirements.txt

streamlit run app.py 