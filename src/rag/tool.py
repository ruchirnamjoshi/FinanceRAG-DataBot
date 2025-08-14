# src/rag/tool.py
from langchain_core.tools import tool
from .lc_chain import with_memory

_chain = None
def _get_chain():
    global _chain
    if _chain is None:
        _chain = with_memory()
    return _chain

@tool("rag_query", return_direct=True)
def rag_query_tool(question: str, session_id: str = "default") -> str:
    """Answer questions from SEC filings using the RAG chain. Keeps chat memory by session_id."""
    chain = _get_chain()
    resp = chain.invoke({"question": question}, config={"configurable": {"session_id": session_id}})
    return resp.content
