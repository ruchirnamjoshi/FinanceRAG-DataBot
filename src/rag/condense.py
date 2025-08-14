# src/rag/condense.py
import ollama

def condense_question(history_text: str, user_question: str, model="llama3.1:8b"):
    prompt = f"""Rewrite the user question to be standalone, using the chat history for context.
History:
{history_text}

Question:
{user_question}

Standalone question:"""
    out = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
    return out["message"]["content"].strip()
