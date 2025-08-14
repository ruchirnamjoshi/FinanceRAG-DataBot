import os, json, glob, pathlib
from typing import List,Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter




from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from ingest.parse_html import html_to_sections

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = os.getenv("QDRANT_COLLECTION", "sec_filings_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

def _docs_from_file(html_path: str) -> List[Document]:
    meta_path = pathlib.Path(html_path).with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(open(meta_path, encoding="utf-8").read())

    html = open(html_path, encoding="utf-8", errors="ignore").read()
    sections = html_to_sections(html)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    docs: List[Document] = []
    for sec in sections:
        title = (sec.get("title") or "").strip()
        for ch in splitter.split_text(sec.get("text", "")):
            docs.append(Document(
                page_content=ch,
                metadata={
                    "title": title,
                    "section": title.lower().replace("\u00a0", " "),
                    **meta,
                    "source": os.path.basename(html_path),
                }
            ))
    return docs

def ensure_collection(client: QdrantClient, dim: int = 1024):
    names = [c.name for c in client.get_collections().collections]
    if COLL not in names:
        client.recreate_collection(
            collection_name=COLL,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def index_directory(raw_dir: str) -> int:
    files = [
        f for f in glob.glob(os.path.join(raw_dir, "*.html"))
        if pathlib.Path(f).with_suffix(".meta.json").exists()
    ]
    print("Primary filings:", len(files))

    client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, grpc_port=6334, timeout=180)
    ensure_collection(client, dim=1024)
    emb = OllamaEmbeddings(model=EMBED_MODEL)

    # NOTE: new API uses `embeddings=` (not embedding_function)
    vs = LCQdrant(client=client, collection_name=COLL, embeddings=emb)

    total = 0
    for f in files:
        docs = _docs_from_file(f)
        if not docs:
            continue
        for i in range(0, len(docs), 200):
            vs.add_documents(docs[i:i+200])
        total += len(docs)
        print(f"Indexed {len(docs)} chunks from {os.path.basename(f)}")
    print("Total chunks indexed:", total)
    return total

if __name__ == "__main__":
    raw_dir = os.getenv("DATA_DIR", "./data/raw")
    index_directory(raw_dir)
