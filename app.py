
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.retrieval import load_vector_db, build_reranked_hybrid_retriever
from src.chain import build_llm, build_rag_chain

st.set_page_config(page_title="AIM Docs RAG Assistant", page_icon="✈️")


@st.cache_resource(show_spinner="Loading pipeline (first run only)...")
def load_pipeline():
    vector_db = load_vector_db()
    retriever = build_reranked_hybrid_retriever(vector_db)
    llm = build_llm(st.secrets["GROQ_API_KEY"])
    return build_rag_chain(llm, retriever)


rag_chain = load_pipeline()

st.title("✈️ AIM Docs RAG Assistant")
st.caption(
    "Ask questions about the Manual of Air Traffic Services. "
    "For internal/demo use only — not an authoritative source."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # display history: [{"role", "content", "sources"?}]
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []  # LangChain message objects for the chain

# Replay prior turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src, page in msg["sources"]:
                    st.write(f"- {src} (page {page})")

# New turn
if question := st.chat_input("Ask about the manual..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            result = rag_chain.invoke(
                {"input": question, "chat_history": st.session_state.lc_history}
            )
            answer = result["answer"]
            sources = [
                (doc.metadata.get("source", "?").split("/")[-1], doc.metadata.get("page", "?"))
                for doc in result["context"]
            ]
        st.markdown(answer)
        with st.expander("Sources"):
            for src, page in sources:
                st.write(f"- {src} (page {page})")

    st.session_state.lc_history.append(HumanMessage(content=question))
    st.session_state.lc_history.append(AIMessage(content=answer))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with st.sidebar:
    st.header("About")
    st.write(
        "Retrieval-augmented chat over the Manual of Air Traffic Services "
        "(hybrid BM25 + vector search, reranked with a cross-encoder)."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.rerun()