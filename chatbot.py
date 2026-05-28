import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser

import history as hist
import vector_store as vs
from config import get_settings
from utils import format_sources

logger = logging.getLogger(__name__)

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the follow-up question below, rewrite the "
     "follow-up into a standalone question that contains all necessary context. "
     "Return ONLY the rewritten question, nothing else."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are an intelligent assistant that answers questions strictly based \
on the provided document context. Rules:
1. Answer ONLY from the context. Do not hallucinate or use outside knowledge.
2. If the answer is not found, say: "I don't have enough information in the \
provided documents to answer that."
3. Be concise and accurate.

--- CONTEXT ---
{context}
---------------"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

_llm: Optional[ChatAnthropic] = None
_condense_llm: Optional[ChatAnthropic] = None


def init_llms() -> None:
    global _llm, _condense_llm
    settings = get_settings()
    _llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
    )
    _condense_llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.0,
        anthropic_api_key=settings.anthropic_api_key,
    )
    logger.info(f"LLMs initialized | model={settings.llm_model}")


def _get_llms() -> tuple[ChatAnthropic, ChatAnthropic]:
    if _llm is None or _condense_llm is None:
        raise RuntimeError("LLMs not initialized. Call init_llms() first.")
    return _llm, _condense_llm


def _format_docs(docs: list) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )


def _condense_question(inputs: dict) -> str:
    _, condense_llm = _get_llms()
    chat_history: list[BaseMessage] = inputs["chat_history"]
    if not chat_history:
        return inputs["question"]
    chain = _CONDENSE_PROMPT | condense_llm | StrOutputParser()
    return chain.invoke({"chat_history": chat_history, "question": inputs["question"]})


def chat(session_id: str, question: str, doc_id: Optional[str] = None) -> dict:
    settings = get_settings()
    llm, _ = _get_llms()
    chat_history = hist.get_history(session_id)

    logger.info(f"[session={session_id}] Q: '{question[:100]}'")

    standalone = _condense_question({"question": question, "chat_history": chat_history})
    logger.debug(f"[session={session_id}] Standalone: '{standalone[:100]}'")

    filter_dict = {"doc_id": doc_id} if doc_id else None
    retriever = vs.get_retriever(k=settings.top_k_retrieval, filter_dict=filter_dict)
    source_docs = retriever.invoke(standalone)
    context = _format_docs(source_docs)

    qa_chain = _QA_PROMPT | llm | StrOutputParser()
    answer = qa_chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": standalone,
    })

    hist.add_user_message(session_id, question)
    hist.add_ai_message(session_id, answer)

    sources = format_sources(source_docs)
    logger.info(f"[session={session_id}] A: '{answer[:100]}...' | sources={len(sources)}")

    return {"answer": answer, "sources": sources, "session_id": session_id}
