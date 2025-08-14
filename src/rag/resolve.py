# src/rag/resolve.py
import re
from datetime import datetime
from typing import Optional, Tuple
from ingest.tickers import ticker_to_cik, name_to_cik

QTR_RE = re.compile(r"\bq([1-4])\b(?:\s*(?:fy)?\s*(\d{2,4}))?", re.I)

def infer_form(question: str) -> str:
    s = question.lower()
    if "annual" in s or "10-k" in s:
        return "10-K"
    # MD&A typically lives in 10-Q and 10-K; for “this quarter” we prefer 10-Q
    if "quarter" in s or "10-q" in s or "md&a" in s or "management's discussion" in s:
        return "10-Q"
    return "10-Q"

def infer_cik(question: str) -> Optional[int]:
    # Look for ticker in ALLCAPS 2-5 chars (basic heuristic)
    m = re.search(r"\b([A-Z]{2,5})\b", question)
    if m:
        cik = ticker_to_cik(m.group(1))
        if cik: return cik
    # Look for common company names
    for name in ("Apple","Microsoft","Amazon","Alphabet","Google","Meta","NVIDIA","Tesla","Netflix","Salesforce","Adobe","Intel","Advanced Micro Devices"):
        if name.lower() in question.lower():
            cik = name_to_cik(name)
            if cik: return cik
    return None

def infer_quarter(question: str) -> Optional[Tuple[int, Optional[int]]]:
    # Returns (quarter, year) if found
    m = QTR_RE.search(question)
    if not m: return None
    q = int(m.group(1))
    year = m.group(2)
    if year:
        y = int(year)
        if y < 100:  # FY24 -> 2024-ish; you can refine to client's FY definition
            y += 2000
        return (q, y)
    return (q, None)

def mdna_only(question: str) -> bool:
    s = question.lower()
    return "md&a" in s or "management’s discussion" in s or "management's discussion" in s or "revenue driver" in s
