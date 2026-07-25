from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.config import GROQ_MODEL_NAME, LLM_TEMPERATURE

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, reformulate it into a standalone question. "
    "Do NOT answer the question, just reformulate it if needed, otherwise "
    "return it as is."
)

QA_SYSTEM_PROMPT = (
    "Answer the question based only on the following context. "
    "If the answer isn't in the context, say you don't know.\n\nContext:\n{context}"
)


def build_llm(groq_api_key: str) -> ChatGroq:
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )


def build_rag_chain(llm: ChatGroq, retriever: BaseRetriever):
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)