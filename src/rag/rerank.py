import os
from sentence_transformers import CrossEncoder
RERANKER=os.getenv("RERANKER","BAAI/bge-reranker-v2-m3")
_ce=None
def get_reranker():
    global _ce
    if _ce is None: _ce=CrossEncoder(RERANKER, trust_remote_code=True)
    return _ce
def rerank(query, candidates, top_k=6):
    ce=get_reranker()
    scores=ce.predict([(query,c) for c in candidates])
    ranked=sorted(zip(candidates, scores), key=lambda x:-x[1])[:top_k]
    return [c for c,_ in ranked]
