# src/databot/csv_catalog.py
import json, os, numpy as np
import pandas as pd
from typing import List, Dict, Optional
from langchain_ollama import OllamaEmbeddings
import numpy as np

# in databot/csv_catalog.py

CATALOG_PATH = os.getenv("CSV_CATALOG_PATH", "data/csv/catalog.json")

class CSVCatalog:
    def __init__(self, embed_model: str = "mxbai-embed-large"):
        self.items = []   # [{key, path, columns, schema_text, vec, ...}]
        self.emb = OllamaEmbeddings(model=embed_model)
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            data = json.load(open(CATALOG_PATH, "r"))
        except FileNotFoundError:
            data = []
        self.items = []
        for row in data:
            schema_text = f"file:{os.path.basename(row['path'])}; cols:{', '.join(row['columns'])}; "\
                          f"table:{row.get('table_type','')}; ticker:{row.get('ticker')}; "\
                          f"form:{row.get('form')}; date:{row.get('date')}"
            vec = self.emb.embed_query(schema_text)
            self.items.append({**row, "schema_text": schema_text, "vec": vec})
        self._loaded = True

    # existing add() stays (for ad-hoc uploads)
    # best_match() should call self.load() first
    def best_match(self, query: str) -> str | None:
        self.load()
        if not self.items: return None
        qvec = self.emb.embed_query(query)
        def cos(a,b):
            a,b=np.array(a),np.array(b); return float(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
        ranked = sorted(((it["path"], cos(qvec, it["vec"])) for it in self.items),
                        key=lambda x: x[1], reverse=True)
        return ranked[0][0] if ranked else None


CATALOG = CSVCatalog()
