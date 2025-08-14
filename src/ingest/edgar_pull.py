# src/ingest/edgar_pull.py
import os, time, re, pathlib, random, json
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Optional
import httpx
from xml.etree import ElementTree as ET

BASE = "https://www.sec.gov"
DATA = "https://data.sec.gov"
IDX  = "https://www.sec.gov/Archives/edgar/daily-index"

UA = {
    "User-Agent": "Ruchir Namjoshi Finance-RAG/0.1 (ruchir.namjoshi@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
}
PAUSE_BASE = 0.7  # be polite
_HTM_RE = re.compile(r'href="([^"]+\.htm[l]?)"', re.I)

# ------------ HTTP helpers ------------
def _client():
    return httpx.Client(
        timeout=httpx.Timeout(30.0),
        headers=UA,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=5),
        follow_redirects=True,
        http2=True,
    )

def _get(url: str, tries: int = 4, base_sleep: float = PAUSE_BASE) -> httpx.Response:
    with _client() as c:
        for i in range(tries):
            r = c.get(url)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(base_sleep * (2 ** i) + random.uniform(0.0, 0.3))
                continue
            r.raise_for_status()
        r.raise_for_status()

# ------------ Daily master.idx (fallback) ------------
def list_recent_master_indexes(days: int = 2) -> list[str]:
    out = []
    today = datetime.utcnow()
    for d in range(days):
        day = today - timedelta(days=d)
        y, q = day.year, (day.month - 1) // 3 + 1
        out.append(f"{IDX}/{y}/QTR{q}/master.{day.strftime('%Y%m%d')}.idx")
    return out

def fetch_text(url: str) -> str:
    return _get(url).text

def parse_master_idx(text: str):
    rows, start = [], False
    for line in text.splitlines():
        if start:
            parts = line.split("|")
            if len(parts) == 5 and parts[2] in ("10-Q", "10-K"):
                rows.append({
                    "cik": parts[0],
                    "company": parts[1],
                    "form": parts[2],
                    "date": parts[3],
                    "path": parts[4],
                })
        if line.startswith("-----"):
            start = True
    return rows

def fetch_primary_html_from_index(index_url: str, out_dir: str) -> Optional[str]:
    """
    Given a filing *document index* URL (...-index.html), find the primary 10-Q/10-K doc and save it.
    We try to select the row whose Type contains '10-Q' or '10-K'; fallback to first .htm.
    """
    r = _get(index_url)
    # prefer rows that mention 10-Q / 10-K in same HTML table row
    # crude but effective: find all .htm links then bias ones containing 10q, 10-k
    links = _HTM_RE.findall(r.text)
    if not links:
        return None

    def score(href: str) -> int:
        h = href.lower()
        s = 0
        if "10-q" in h or "10q" in h: s += 3
        if "10-k" in h or "10k" in h: s += 2
        if "form" in h: s += 1
        return s

    links = sorted(links, key=score, reverse=True)
    target = links[0]
    doc_url = f"{BASE}{target}" if target.startswith("/Archives/") else target
    html = _get(doc_url).text

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_url.replace("https://", "").replace("http://",""))
    fn = pathlib.Path(out_dir) / f"{safe}.html"
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(html, encoding="utf-8")
    time.sleep(PAUSE_BASE + random.uniform(0.0, 0.4))
    return str(fn)

# ------------ Preferred: company submissions JSON ------------
def _strip_cik(cik: str|int) -> str:
    s = str(cik).lstrip("0")
    return s if s else "0"

def recent_filings_for_cik(cik: str|int, forms: Iterable[str] = ("10-Q","10-K"), limit: int = 5) -> List[Dict]:
    """
    Use the submissions JSON to get recent filings for a company (CIK).
    Returns a list of dicts with the fully qualified primary document URL.
    """
    cik_nz = _strip_cik(cik)
    url = f"{DATA}/submissions/CIK{int(cik_nz):010d}.json"  # CIK must be zero-padded to 10
    js = _get(url).json()

    recent = js.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    accnos = recent.get("accessionNumber", [])
    primaries = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])
    out = []
    for f, acc, prim, dt in zip(forms_list, accnos, primaries, dates):
        if f not in forms: 
            continue
        # build URL to the primary document HTML
        acc_nodash = acc.replace("-", "")
        doc_url = f"{BASE}/Archives/edgar/data/{int(cik_nz)}/{acc_nodash}/{prim}"
        out.append({
            "cik": int(cik_nz),
            "form": f,
            "date": dt,
            "doc_url": doc_url
        })
        if len(out) >= limit:
            break
    return out

def fetch_html_to_dir(url: str, out_dir: str, meta: dict | None = None) -> str | None:
    html = _get(url).text
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", url.replace("https://", "").replace("http://",""))
    fn = pathlib.Path(out_dir) / f"{safe}.html"
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(html, encoding="utf-8")
    if meta:
        (fn.with_suffix(".meta.json")).write_text(json.dumps(meta), encoding="utf-8")
    time.sleep(PAUSE_BASE + random.uniform(0.0, 0.3))
    return str(fn)


# ------------ Orchestrators ------------
DEFAULT_CIKS = [
    320193,    # Apple
    789019,    # Microsoft
    1018724,   # Amazon
    1652044,   # Alphabet
    1318605,   # Meta
]

def run_submissions_seed(out_dir: str = "./data/raw", ciks: Iterable[int] = DEFAULT_CIKS, per_cik: int = 3) -> int:
    """
    Seed local corpus by pulling recent 10-Q/10-K for a few known CIKs.
    Much more reliable than daily-index and respects SEC UA policy.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for cik in ciks:
        try:
            items = recent_filings_for_cik(cik, ("10-Q","10-K"), limit=per_cik)
            for it in items:
                p = fetch_html_to_dir(
                        it["doc_url"], out_dir,
                        meta={"cik": it["cik"], "form": it["form"], "date": it["date"]}
                    )

                if p:
                    saved += 1
                    print(f"Saved (Submissions) CIK {cik} {it['form']} {it['date']} -> {p}")
        except Exception as e:
            print("submissions error for CIK", cik, e)
    print(f"Total saved via submissions: {saved}")
    return saved

def run(days: int = 3, out_dir: str = "./data/raw", max_per_day: int | None = None) -> int:
    # Only seed via submissions; do NOT fall back to daily index (to avoid exhibits)
    saved = run_submissions_seed(out_dir=out_dir, ciks=DEFAULT_CIKS, per_cik=3)
    print(f"Total saved HTML filings (submissions only): {saved}")
    return saved


if __name__ == "__main__":
    run()
