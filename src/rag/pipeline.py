import os, ollama
from .vectorstore import query
from .rerank import rerank
from .resolve import infer_form, infer_cik, infer_quarter, mdna_only

from .memory import get_history  # or get_messages/save_messages if using JSON
from .condense import condense_question



SYSTEM = "Answer only from the provided context. Quote short phrases, and name the section and filing date."

LLM = os.getenv("LLM_MODEL","llama3.1:8b")

def answer(question: str, top_k=6, cik: int | None = None, form: str | None = None, session_id: str = "default"):
    # 1) load history (SQLite)
    history = get_history(session_id)
    hist_msgs = history.messages  # list[BaseMessage]

    # Turn last N into a plain text for condensation
    hist_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in hist_msgs[-10:]])

    standalone_q = condense_question(hist_text, question)

    # auto-infer if not provided
    form = form or infer_form(standalone_q)
    cik = cik or infer_cik(standalone_q)
    want_mdna = mdna_only(standalone_q)

    # First pass search with whatever we know
    hits = query(standalone_q, top_k=80, cik=cik, form=form)
    payloads = [h.payload for h in hits]

    # If we didn't infer cik, try “majority CIK” from first pass then re-query
    if cik is None and payloads:
        counts = {}
        for p in payloads:
            ck = p.get("cik")
            if ck is not None:
                counts[ck] = counts.get(ck, 0) + 1
        if counts:
            picked = max(counts.items(), key=lambda kv: kv[1])[0]
            # re-run with chosen cik
            hits = query(standalone_q, top_k=80, cik=picked, form=form)
            payloads = [h.payload for h in hits]

    # MD&A filter/bias
    texts_mdna = [p["text"] for p in payloads if str(p.get("section","")).lower().startswith("item 2.")]
    texts_other = [p["text"] for p in payloads if not str(p.get("section","")).lower().startswith("item 2.")]
    candidates = (texts_mdna[:60] + texts_other[:20]) if want_mdna else (texts_mdna[:30] + texts_other[:50])
    if not candidates:
        answer_text, top = "No matching context found for your query in the current index.", []
        return answer_text, top

    # rerank + answer
    top = rerank(standalone_q, candidates, top_k=top_k)
    context = "\n\n---\n\n".join(top)
    msg = f"{SYSTEM}\n\nQuestion: {standalone_q}\n\nContext:\n{context}\n\nAnswer:"
    out = ollama.chat(model=LLM, messages=[{"role":"system","content":SYSTEM},
                                           {"role":"user","content":msg}])
    answer_text, top = out["message"]["content"], top

    history.add_user_message(standalone_q)
    history.add_ai_message(answer_text)

    return answer_text, top
