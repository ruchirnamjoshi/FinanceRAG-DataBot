# -*- coding: utf-8 -*-
import os, io, json, pathlib, re
from typing import List, Dict, Tuple, Optional
import requests
import pandas as pd
import numpy as np

OUT_DIR_DEFAULT = "data/csv_clean"
CATALOG_PATH_DEFAULT = "data/csv_clean/catalog.json"

# ============= 1) Add your sources here (multi-company) =================
# Replace `url` placeholders with raw CSV links.
# table_type: "product_revenue", "geographic_revenue", "segment_revenue", etc.
# period: "YYYY_Qn" (or "YYYY" for annual) — consistent naming helps selection.
SOURCES: List[Dict] = [
    # --- Apple ---
    {
        "ticker": "AAPL",
        "form": "10-K",
        "table_type": "product_revenue",   # this set is unit sales, but useful for testing
        "period": "2011-2018",
        "url": "https://raw.githubusercontent.com/kjhealy/apple/master/data/apple-all-products-quarterly-sales.csv",
        "preferred_columns": ["Period", "iPhone", "Mac", "iPad", "Wearables", "Services"]
    },
    {
        "ticker": "AAPL",
        "form": "10-K",
        "table_type": "geographic_revenue",
        "period": "2024_Q4",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/aapl_2024q4_geo_revenue.csv",
        "preferred_columns": ["Period","Americas","Europe","Greater China","Japan","Rest of Asia Pacific"],
    },

    # --- Microsoft ---
    {
        "ticker": "MSFT",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q3",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/msft_2025q3_segment_revenue.csv",
        "preferred_columns": ["Period","Productivity and Business Processes","Intelligent Cloud","More Personal Computing"],
    },

    # --- Amazon ---
    {
        "ticker": "AMZN",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/amzn_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","North America","International","AWS"],
    },

    # --- Alphabet ---
    {
        "ticker": "GOOGL",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/googl_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","Google Services","Google Cloud","Other Bets"],
    },

    # --- Meta ---
    {
        "ticker": "META",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/meta_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","Family of Apps","Reality Labs"],
    },

    # --- Nvidia ---
    {
        "ticker": "NVDA",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/nvda_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","Data Center","Gaming","Professional Visualization","Automotive","OEM & Other"],
    },

    # --- Tesla ---
    {
        "ticker": "TSLA",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/tsla_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","Automotive","Energy generation and storage","Services and other"],
    },

    # --- Netflix ---
    {
        "ticker": "NFLX",
        "form": "10-Q",
        "table_type": "geographic_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/nflx_2025q2_geo_revenue.csv",
        "preferred_columns": ["Period","UCAN","EMEA","LATAM","APAC"],
    },

    # --- Salesforce ---
    {
        "ticker": "CRM",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/crm_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","Subscription and support","Professional services and other"],
    },

    # --- Adobe ---
    {
        "ticker": "ADBE",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/adbe_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","Digital Media","Digital Experience","Publishing and Advertising"],
    },

    # --- Intel ---
    {
        "ticker": "INTC",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/intc_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","Client Computing","Data Center and AI","Network and Edge","Foundry Services","Mobileye","Others"],
    },

    # --- AMD ---
    {
        "ticker": "AMD",
        "form": "10-Q",
        "table_type": "segment_revenue",
        "period": "2025_Q2",
        "url": "https://raw.githubusercontent.com/<org>/<repo>/main/amd_2025q2_segment_revenue.csv",
        "preferred_columns": ["Period","Data Center","Client","Gaming","Embedded"],
    },
]

# ============= 2) Helpers =============

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "")).strip("_")

def _norm(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _to_number(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int,float,np.number)): return float(val)
    x = str(val).strip()
    if x in {"—","–","-","--",""}: return np.nan
    x = x.replace("$","").replace(",","")
    if x.endswith("%"):
        try: return float(x[:-1])/100.0
        except: return np.nan
    m = re.match(r"^\(([-+]?\d*\.?\d+)\)$", x)
    if m:
        try: return -float(m.group(1))
        except: return np.nan
    try: return float(x)
    except: return np.nan

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common field names across companies for easier plotting."""
    ren = {}
    for c in df.columns:
        cl = _norm(c).lower()
        if cl in {"quarter","period","fiscal quarter","fiscal_quarter"}:
            ren[c] = "Period"
        elif "iphone" in cl:      ren[c] = "iPhone"
        elif "services" in cl:    ren[c] = "Services"
        elif re.fullmatch(r"macs?|mac revenue", cl): ren[c] = "Mac"
        elif "ipad" in cl:        ren[c] = "iPad"
        elif "wear" in cl or "accessor" in cl: ren[c] = "Wearables"
        elif "americas" in cl:    ren[c] = "Americas"
        elif "europe" in cl:      ren[c] = "Europe"
        elif "greater china" in cl or "china" in cl: ren[c] = "Greater China"
        elif "japan" in cl:       ren[c] = "Japan"
        elif "asia pacific" in cl or "apac" in cl or "rest of asia pacific" in cl:
            ren[c] = "Rest of Asia Pacific"
        elif cl in {"revenue","net sales","net_sales"}:
            ren[c] = "Revenue"
    if ren:
        df = df.rename(columns=ren)
    return df

def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            try_vals = out[c].dropna().astype(str).head(25)
            if len(try_vals) and (try_vals.str.contains(r"[\d$,%\-\(\)]", regex=True)).mean() > 0.5:
                out[c] = out[c].map(_to_number)
    return out

def _label_and_numeric(df: pd.DataFrame) -> Tuple[str, List[str]]:
    label = "Period" if "Period" in df.columns else df.columns[0]
    nums = [c for c in df.columns if c != label and pd.api.types.is_numeric_dtype(df[c])]
    return label, nums

def _download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.content
    # Try reading as UTF-8 CSV first; if fails, let pandas guess via BytesIO
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return pd.read_csv(io.BytesIO(data), encoding_errors="ignore")

# ============= 3) Main fetch/validate/install =============

def fetch_and_install(out_dir: str = OUT_DIR_DEFAULT,
                      catalog_path: str = CATALOG_PATH_DEFAULT) -> Tuple[int,int]:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog: List[Dict] = []
    written = 0

    for i, src in enumerate(SOURCES, 1):
        ticker = src["ticker"].upper().strip()
        form   = (src.get("form") or "10-Q").upper()
        ttype  = src.get("table_type") or "other"
        period = src.get("period") or "unknown"
        url    = src["url"]

        print(f"[{i}/{len(SOURCES)}] {ticker} {period} {ttype} → {url}")

        # 1) Download & parse
        try:
            df = _download_csv(url)
        except Exception as e:
            print("  skip (download/read failed):", e)
            continue

        if df is None or df.empty:
            print("  skip (empty csv)")
            continue

        # 2) Normalize
        df.columns = [_norm(c) for c in df.columns]
        df = _normalize_columns(df)
        df = _clean_numeric(df)

        # 3) Validate schema
        label_col, numeric_cols = _label_and_numeric(df)
        if len(df.columns) < 2 or len(numeric_cols) < 1 or df.shape[0] < 2:
            print("  skip (not plot-ready: need label + ≥1 numeric + ≥2 rows)")
            continue

        # Optional: if preferred columns specified, verify presence
        want = set(src.get("preferred_columns") or [])
        if want and not want.issubset(set(df.columns)):
            print(f"  warn: preferred columns {sorted(want)} not all found; actual columns = {list(df.columns)}")

        # 4) Write file in standard structure
        comp_dir = out / _safe_name(ticker)
        comp_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{_safe_name(period)}_{_safe_name(form)}_{_safe_name(ttype)}_1.csv"
        fpath = comp_dir / fname
        df.to_csv(fpath, index=False)
        written += 1

        # 5) Add to catalog
        catalog.append({
            "path": str(fpath),
            "ticker": ticker,
            "cik": None,
            "form": form,
            "date": None,
            "period": period,
            "table_type": ttype,
            "label_col": label_col,
            "numeric_cols": numeric_cols,
            "columns": list(df.columns),
            "source_html": None,
            "source_url": url,
        })

        print(f"  saved: {fpath.name} | columns={list(df.columns)}")

    # 6) Save/merge catalog
    out_catalog = pathlib.Path(catalog_path)
    out_catalog.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing if present
    if out_catalog.exists():
        try:
            existing = json.loads(out_catalog.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    else:
        existing = []

    # De-dup on (path)
    existing_paths = {it.get("path") for it in existing}
    merged = existing + [it for it in catalog if it["path"] not in existing_paths]

    out_catalog.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nCatalog updated: {out_catalog} (entries={len(merged)})")
    return (len(SOURCES), written)

if __name__ == "__main__":
    sources, ok = fetch_and_install()
    print(f"\nListed sources: {sources}")
    print(f"CSV files written: {ok}")