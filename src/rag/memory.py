# src/rag/memory.py
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
import os

DB_PATH = os.getenv("CHAT_DB_PATH", "data/chat_history.sqlite3")

def get_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection_string=f"sqlite:///{DB_PATH}")
