# src/extract/tables_to_csv.py
# -*- coding: utf-8 -*-
import os, re, json, pathlib
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# ---------------- Config ----------------
RAW_DIR_DEFAULT = "data/raw"
OUT_DIR_DEFAULT = "data/csv_clean"          # plot-ready CSVs here
CATALOG_PATH_DEFAULT = "data/csv_clean/catalog.json"

# If you know more mappings, extend these:
TICKER_TO_CIK = {
    "AAPL": 320193, "MSFT": 789019, "AMZN": 1018724, "GOOGL": 1652044, "GOOG": 1652044,
    "META": 1318605, "NVDA": 1045810, "TSLA": 1090872, "NFLX": 1326801, "CRM": 1329099,
    "ADBE": 796343, "INTC": 50863, "AMD": 2488,
}
CIK_TO_TICKER = {v: k for k, v in TICKER_TO_CIK.items()}
NBSP = "\u00a0"

# ---------------- Helpers ----------------

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("’", "'")
    s = s.replace("†", "").replace("‡", "").replace("®", "").replace("™", "")
    s = re.sub(rf"[ \t{NBSP}]+", " ", s)
    return s.strip()

def _flatten_columns(df: pd.DataFrame) -> List[str]:
    """Return a flat list of column names from possibly MultiIndex columns."""
    if isinstance(df.columns, pd.MultiIndex):
        cols = [
            " | ".join([_norm(x) for x in tup if _norm(x)])
            for tup in df.columns
        ]
    else:
        cols = [_norm(c) for c in df.columns]
    # Replace empty names with placeholders
    cols = [c if c else "Column" for c in cols]
    # Deduplicate
    seen = {}
    out = []
    for c in cols:
        base = c
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

def _looks_unnamed(cols: List[str]) -> bool:
    """Heuristic: a header row with mostly Unnamed/empty/nan."""
    c = [str(x) for x in cols]
    nm = sum([1 for x in c if x.lower().startswith("unnamed") or x.strip()=="" or x.strip().lower()=="nan"])
    return nm >= max(3, int(0.6 * max(1, len(c))))

def _score_header_row(row: List[str]) -> float:
    """Higher is better: more real words, fewer numeric-only tokens."""
    vals = [(_norm(v) or "") for v in row]
    nonempty = sum(1 for v in vals if v)
    alphaish = sum(1 for v in vals if re.search(r"[A-Za-z]", v))
    numericish = sum(1 for v in vals if re.fullmatch(r"[-+]?[\d,.$()%]+", v or ""))
    # Prefer alpha > numeric; normalize by row length
    L = max(1, len(vals))
    return (0.7 * alphaish + 0.3 * nonempty - 0.6 * numericish) / L

def _promote_best_header(df: pd.DataFrame) -> pd.DataFrame:
    """If current header looks bad, scan first N rows to pick a better header row."""
    if df.empty:
        return df
    if not _looks_unnamed([str(c) for c in df.columns]):
        return df  # header seems fine

    # Try up to first 6 rows to find a headerish row
    best_idx, best_score = None, -1e9
    top = min(6, len(df))
    for i in range(top):
        row = df.iloc[i].tolist()
        score = _score_header_row(row)
        if score > best_score:
            best_idx, best_score = i, score

    if best_idx is None:
        return df

    # Promote that row to header
    new_cols = [(_norm(v) or f"Column{j+1}") for j, v in enumerate(df.iloc[best_idx])]
    tmp = df.iloc[best_idx+1:].reset_index(drop=True).copy()
    tmp.columns = _dedupe_cols(new_cols)
    return tmp

def _dedupe_cols(cols: List[str]) -> List[str]:
    seen, out = {}, []
    for c in cols:
        base = _norm(c) or "Column"
        k = base.lower()
        if k not in seen:
            seen[k] = 0; out.append(base)
        else:
            seen[k] += 1; out.append(f"{base}_{seen[k]}")
    return out

def _percent_column(series: pd.Series) -> bool:
    vals = series.dropna().astype(str).str.strip()
    if len(vals) == 0:
        return False
    return (vals.str.endswith("%")).mean() >= 0.60

def _to_number(val):
    """Convert common financial strings to numeric, handling () negatives, % and em‑dashes."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    x = str(val).strip()
    # blanks/dashes
    if x in {"—", "–", "-", "--", "— —", ""}:
        return np.nan
    # remove currency and thousands
    x = x.replace("$", "").replace(",", "")
    # percentage
    if x.endswith("%"):
        try:
            return float(x[:-1]) / 100.0
        except:
            return np.nan
    # parentheses negative
    m = re.match(r"^\(([-+]?\d*\.?\d+)\)$", x)
    if m:
        try:
            return -float(m.group(1))
        except:
            return np.nan
    # plain float
    try:
        return float(x)
    except:
        return np.nan

def _read_html_tables(path: str) -> List[pd.DataFrame]:
    """Read tables robustly: pandas.read_html, fallback per-table with bs4."""
    try:
        tables = pd.read_html(path, flavor="lxml", header=0)
        if tables:
            return tables
    except Exception:
        pass

    html = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    out = []
    for tbl in soup.find_all("table"):
        try:
            df = pd.read_html(str(tbl), flavor="lxml", header=[0, 1])[0]
        except Exception:
            try:
                df = pd.read_html(str(tbl), flavor="lxml", header=0)[0]
            except Exception:
                continue
        out.append(df)
    return out

def _classify_table(df: pd.DataFrame) -> str:
    head = " ".join(map(str, df.columns)).lower()
    left = " ".join(map(str, df.head(12).iloc[:, 0].tolist())).lower() if df.shape[1] else ""
    text = head + " " + left

    # 1) Financial statements
    if any(k in text for k in [
        "consolidated statements of operations",
        "cost of sales", "gross margin", "gross profit"
    ]):
        if "net sales" in text or "revenue" in text:
            return "income_statement"
        return "income_statement"

    if any(k in text for k in [
        "consolidated balance sheets", "total assets", "total liabilities",
        "stockholders' equity", "shareholders' equity"
    ]):
        return "balance_sheet"

    if any(k in text for k in [
        "consolidated statements of cash flows", "cash flows from",
        "net cash provided", "cash and cash equivalents at end"
    ]):
        return "cash_flow"

    # 2) Segment operating income (catch before other 'segment' cases)
    if "operating income" in text and "segment" in text:
        return "segment_operating_income"

    # 3) Geographic revenue
    if any(k in text for k in [
        "americas", "europe", "greater china", "japan",
        "rest of asia pacific", "emea", "apac", "geographic"
    ]) and ("net sales" in text or "revenue" in text or "sales" in text):
        return "geographic_revenue"

    # 4) Product/category revenue (iPhone, Services, etc.)
    product_terms = ["iphone","ipad","mac","services","wearables","home","accessories","watch","music","tv+"]
    if any(p in text for p in product_terms) and any(k in text for k in ["net sales","revenue","sales"]):
        return "product_revenue"  # <- new, more precise than "segment_revenue"

    # 5) Generic segment revenue (fallback if it really says segment + revenue)
    if "segment" in text and any(k in text for k in ["net sales","revenue","sales"]):
        return "segment_revenue"

    return "other"

def _drop_boilerplate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are unit descriptors or boilerplate (e.g., '(in millions)')."""
    if df.empty:
        return df
    lbl = df.columns[0]
    bad = df[lbl].astype(str).str.lower().str.contains(
        r"\b(in thousands|in millions|in billions|unaudited|see accompanying notes)\b"
    )
    return df.loc[~bad].reset_index(drop=True)

def _normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common headers to consistent names."""
    ren = {}
    for c in list(df.columns):
        cl = c.lower()
        if re.search(r"\biphone\b", cl):        ren[c] = "iPhone"
        elif re.search(r"\bservices?\b", cl):   ren[c] = "Services"
        elif re.search(r"\bmac\b", cl):         ren[c] = "Mac"
        elif re.search(r"\bipad\b", cl):        ren[c] = "iPad"
        elif re.search(r"^period|quarter|three months ended|fiscal quarter", cl):
            ren[c] = "Period"
        elif re.search(r"\bnet sales\b|\brevenue(s)?\b", cl) and c not in {"iPhone","Services","Mac","iPad"}:
            # generic revenue column when not a segment column
            ren[c] = "Revenue"
    if ren:
        df = df.rename(columns=ren)
    return df

def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Make a table plot-ready: repair headers, parse numbers, drop empty cols/rows."""
    if df is None or df.empty:
        return pd.DataFrame()

    # 1) Flatten multiindex → strings
    df = df.copy()
    df.columns = _flatten_columns(df)

    # 2) If mostly unnamed, promote the best row to header
    if _looks_unnamed(df.columns):
        df = _promote_best_header(df)

    # 3) Strip whitespace in object cols
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: _norm(x) if isinstance(x, str) else x)

    # 4) Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # 5) Convert numerics (except the presumed label column)
    if df.shape[1] < 2:
        return pd.DataFrame()
    label_col = df.columns[0]
    for c in df.columns[1:]:
        ser = df[c]
        df[c] = ser.map(_to_number)

    # 6) Drop all-NaN numeric columns (keep label col)
    keep = [label_col] + [c for c in df.columns[1:] if not df[c].isna().all()]
    df = df[keep]

    # 7) Drop boilerplate rows like "(in millions)"
    df = _drop_boilerplate_rows(df)

    # 8) If label column is empty-ish, fill it
    if df[label_col].isna().all():
        df[label_col] = [f"Row {i+1}" for i in range(len(df))]

    # 9) Normalize common names
    df = _normalize_names(df)

    # 10) Validate minimally plot-ready
    numeric_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 1 or df.shape[0] < 2:
        return pd.DataFrame()

    return df

def _load_meta(path: str) -> Dict:
    meta_path = pathlib.Path(path).with_suffix(".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _detect_ticker(meta: Dict) -> str:
    cik = meta.get("cik")
    if cik is None:
        return ""
    try:
        cik_int = int(cik)
    except Exception:
        return ""
    return CIK_TO_TICKER.get(cik_int, "")

def _quarter_from_date(date_str: str) -> Optional[str]:
    # Expect YYYY-MM-DD
    if not date_str or len(date_str) < 7:
        return None
    try:
        y, m = int(date_str[:4]), int(date_str[5:7])
        q = (m - 1) // 3 + 1
        return f"{y}_Q{q}"
    except Exception:
        return None

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s or "").strip("_")

# --------------- Main ---------------

def extract_all(raw_dir: str = RAW_DIR_DEFAULT,
                out_dir: str = OUT_DIR_DEFAULT,
                catalog_path: str = CATALOG_PATH_DEFAULT) -> Tuple[int, int]:
    """
    Parse saved EDGAR HTML filings into plot-ready CSVs.
    - Fixes headers (promotes best header row), flattens multiindex
    - Cleans text; parses numbers (commas, $, (), %, dashes)
    - Drops empty cols/rows and boilerplate rows
    - Classifies tables
    - Normalizes common column names (Period/iPhone/Services/etc.)
    - Writes CSVs: data/csv_clean/{TICKER_or_CIK}/{YYYY_Qn}_{FORM}_{type}_{i}.csv
    - Builds catalog.json with schema (path, ticker, cik, form, date, table_type, columns)

    Returns: (html_files_processed, csv_files_written)
    """
    raw = pathlib.Path(raw_dir)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    html_files = [p for p in raw.glob("*.html") if p.with_suffix(".meta.json").exists()]
    catalog: List[Dict] = []
    csv_count = 0

    for html_path in html_files:
        meta = _load_meta(str(html_path))
        form = (meta.get("form") or "").upper()  # "10-Q" / "10-K"
        date = meta.get("date") or ""            # "YYYY-MM-DD"
        cik = str(meta.get("cik") or "")
        ticker = _detect_ticker(meta)

        company_key = ticker if ticker else (f"CIK{cik}" if cik else "UNKNOWN")
        company_dir = out / _safe_name(company_key)
        company_dir.mkdir(parents=True, exist_ok=True)

        raw_tables = _read_html_tables(str(html_path))
        if not raw_tables:
            continue

        idx = 0
        for raw_df in raw_tables:
            if raw_df is None or raw_df.empty:
                continue

            df = _clean_table(raw_df)
            if df.empty:
                continue

            idx += 1
            table_type = _classify_table(df)

            period = _quarter_from_date(date) or (date[:10] if date else "unknown")
            fname = f"{_safe_name(period)}_{_safe_name(form)}_{table_type}_{idx}.csv"
            fpath = company_dir / fname

            df.to_csv(fpath, index=False)
            csv_count += 1

            label_col = "Period" if "Period" in df.columns else df.columns[0]
            catalog.append({
                "path": str(fpath),
                "ticker": ticker or None,
                "cik": cik or None,
                "form": form,
                "date": date,
                "period": period,
                "table_type": table_type,
                "label_col": label_col,
                "numeric_cols": [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])],
                "columns": list(df.columns),
                "source_html": html_path.name,
            })

    pathlib.Path(catalog_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(catalog_path).write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    return (len(html_files), csv_count)

if __name__ == "__main__":
    files, csvs = extract_all()
    print(f"Processed HTML filings: {files}")
    print(f"Wrote plot-ready CSVs: {csvs}")