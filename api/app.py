# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.rag.pipeline import answer

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/ask")
def ask(q: Query):
    ans, ctx = answer(q.question, session_id=q.session_id)
    return {"answer": ans, "chunks": ctx}
