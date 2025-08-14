import os, hashlib, ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from qdrant_client.models import Filter, FieldCondition, MatchValue


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = os.getenv("QDRANT_COLLECTION", "sec_filings_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
client=QdrantClient(url=QDRANT_URL)

client = QdrantClient(
    url=QDRANT_URL,
    timeout=float(os.getenv("QDRANT_TIMEOUT", "180")),
    prefer_grpc=True,
    grpc_port=6334,
)

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLL not in existing:
        client.recreate_collection(
            collection_name=COLL,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

def _embed_one(text: str):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

def upsert(chunks, batch_size=128):
    ensure_collection()
    n = len(chunks)
    if n == 0:
        return
    for i in range(0, n, batch_size):
        batch = chunks[i:i+batch_size]
        vectors = [_embed_one(c["text"]) for c in batch]
        points = []
        for c, vec in zip(batch, vectors):
            pid = int(hashlib.md5(c["text"].encode()).hexdigest()[:16], 16)
            # c is your payload dict (JSON-serializable)
            points.append(PointStruct(id=pid, vector=vec, payload=c))
        client.upsert(collection_name=COLL, points=points, wait=True)
        print(f"Upserted {i + len(batch)}/{n}")



def query(query_text: str, top_k=80, cik: int | None = None, form: str | None = None, section_prefix: str | None = None):
    qvec = _embed_one(query_text)

    must = []
    if cik is not None:
        must.append(FieldCondition(key="cik", match=MatchValue(value=int(cik))))
    if form is not None:
        must.append(FieldCondition(key="form", match=MatchValue(value=form)))
    # NOTE: Qdrant doesn't do prefix on payload strings natively; we’ll handle section bias in pipeline.
    flt = Filter(must=must) if must else None

    return client.search(
        collection_name=COLL,
        query_vector=qvec,
        limit=top_k,
        query_filter=flt,
    )


