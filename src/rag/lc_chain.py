# src/rag/lc_chain.py

import os
import re
from typing import Optional, Dict, Any, List

from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableWithMessageHistory,
)
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# New split packages
from langchain_qdrant import Qdrant as QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Reranker (HF cross-encoder) + compressor wrapper
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever


import pathlib
from sqlalchemy.exc import OperationalError



# ---------------------- Config ----------------------

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = os.getenv("QDRANT_COLLECTION", "sec_filings_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "data/chat_history.sqlite3")


# ---------------------- Lightweight resolvers ----------------------

TICKER_TO_CIK = {
    "AAPL": 320193, "MSFT": 789019, "AMZN": 1018724, "GOOGL": 1652044, "GOOG": 1652044,
    "META": 1318605, "NVDA": 1045810, "TSLA": 1090872, "NFLX": 1326801, "CRM": 1329099,
    "ADBE": 796343, "INTC": 50863, "AMD": 2488,
}
NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "meta": "META",
    "nvidia": "NVDA", "tesla": "TSLA", "netflix": "NFLX",
    "salesforce": "CRM", "adobe": "ADBE", "intel": "INTC",
    "advanced micro devices": "AMD",
}

def infer_cik(q: str) -> Optional[int]:
    # Try ticker pattern in ALLCAPS 2–5 letters
    m = re.search(r"\b([A-Z]{2,5})\b", q)
    if m:
        t = m.group(1)
        if t in TICKER_TO_CIK:
            return TICKER_TO_CIK[t]
    # Try company name
    low = q.lower()
    for name, t in NAME_TO_TICKER.items():
        if name in low:
            return TICKER_TO_CIK[t]
    return None

def infer_form(q: str) -> str:
    s = q.lower()
    if "10-k" in s or "annual" in s:
        return "10-K"
    return "10-Q"


# ---------------------- Retriever (Qdrant + reranker) ----------------------

def build_retriever():
    # Vector store wrapper
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, grpc_port=6334, timeout=180)
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vs = QdrantVectorStore(client=client, collection_name=COLL, embeddings=emb)

    # Wide first-pass vector retrieval
    base = vs.as_retriever(search_kwargs={"k": 80})

    # Cross-encoder reranker (HF); downloads model on first use
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=6)

    # Compose: base retriever + compressor
    comp = ContextualCompressionRetriever(
        base_retriever=base,
        base_compressor=reranker,  # NOTE: argument name is base_compressor in recent LC
    )
    return comp


# LLM
llm = ChatOllama(model=LLM_MODEL)

# Instantiate retriever once
retriever = build_retriever()


# ---------------------- Prompt & helpers ----------------------

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a cautious financial analyst. Answer only from the provided context. "
     "Use short quotes in quotation marks where relevant, and include the section title "
     "and filing date when possible."),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

def mdna_bias(docs: List[Document], want_mdna: bool) -> List[Document]:
    if not docs:
        return docs
    mdna = [d for d in docs if str(d.metadata.get("section", "")).lower().startswith("item 2.")]
    others = [d for d in docs if d not in mdna]
    if want_mdna and mdna:
        return mdna[:50] + others[:20]
    return (mdna[:30] + others[:50]) if mdna else docs


# ---------------------- Chain builder ----------------------

def build_chain():
    def retrieve_with_filters(inputs: Dict[str, Any]) -> List[Document]:
        q = inputs["question"]

        # Vector search (+ cross-encoder via compression retriever)
        docs = retriever.get_relevant_documents(q)

        # Lightweight metadata post-filter based on inferred company/form
        cik = infer_cik(q)
        form = infer_form(q)
        if cik is not None:
            docs = [d for d in docs if int(d.metadata.get("cik", -1)) == int(cik)]
        if form:
            docs = [d for d in docs if d.metadata.get("form") == form]

        # Prefer MD&A for revenue-driver style questions
        want_mdna = any(k in q.lower() for k in [
            "md&a", "management’s discussion", "management's discussion", "revenue driver"
        ])
        docs = mdna_bias(docs, want_mdna)
        return docs

    def format_ctx(inputs: Dict[str, Any]) -> Dict[str, Any]:
        docs = inputs["docs"]
        context = "\n\n---\n\n".join(d.page_content for d in docs[:12])
        return {"question": inputs["question"], "context": context}

    chain = (
        RunnableParallel(
            question=lambda x: x["question"],
            docs=RunnableLambda(retrieve_with_filters),
        )
        | RunnableLambda(format_ctx)
        | PROMPT
        | llm
    )
    return chain


# ---------------------- Memory wrapper ----------------------

def get_history(session_id: str):
    # Ensure directory exists
    db_path = os.getenv("CHAT_DB_PATH", "data/chat_history.sqlite3")
    db_path = str(pathlib.Path(db_path))  # normalize
    parent = pathlib.Path(db_path).parent
    parent.mkdir(parents=True, exist_ok=True)

    try:
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{db_path}"
        )
    except OperationalError:
        # Fallback to in‑memory if SQLite can’t be opened
        return InMemoryChatMessageHistory()

def with_memory():
    chain = build_chain()

    def _get_session_history(session_id: str):
        return get_history(session_id)

    # Note: RunnableWithMessageHistory expects we pass session_id via config
    mem_chain = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return mem_chain
