
from __future__ import annotations
from typing import Optional
import io, json, base64, contextlib

import numpy as np
import os, io, uuid, contextlib, sys
from langchain_core.tools import tool
from .session_state import STATE
from .csv_catalog import CATALOG  # your existing catalog helper
import re, json, os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, List,Any
from langchain_core.tools import tool

from .session_state import STATE
from .csv_catalog import CATALOG

from langchain.tools import tool
import base64, io





@tool("ingest_csv", return_direct=True)
def ingest_csv_tool(file_path: str) -> str:
    """Load a CSV file; register it as current. Returns summary (rows, columns)."""
    df = pd.read_csv(file_path)
    key = file_path  # could hash for shorter key
    STATE.add(key, df)
    CATALOG.add(key, file_path, df)
    return json.dumps({"key": key, "rows": df.shape[0], "columns": list(df.columns)})

@tool("summarize_csv", return_direct=True)
def summarize_csv_tool(target: Optional[str] = None) -> str:
    """Return a markdown summary of the selected CSV. If target is None, use current."""
    df = STATE.get(target)
    md = df.describe(include="all").round(3).to_markdown()
    return md


def _schema_from_df(df: pd.DataFrame):
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    label_col = df.columns[0] if len(df.columns) else None
    return {
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "label_col": label_col,
        "numeric_cols": numeric_cols,
    }

@tool("select_csv")
def select_csv_tool(
    query: Optional[str] = None,   # kept for signature compatibility; not used
    ticker: Optional[str] = None,
    form: Optional[str] = None,
    table_type: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """
    Select a single CSV from the catalog, set it as current (STATE), and return schema:
      {"selected": path, "schema": {"columns": [...], "dtypes": {...}, "label_col": str, "numeric_cols": [...]}}
    Selection is deterministic:
      1) Filter by ticker/form/table_type/year if provided.
      2) If multiple remain, pick the most recent by period (YYYY_Qn) or date.
    """


    CATALOG.load()
    items = CATALOG.items or []
    if not items:
        return json.dumps({"error": "catalog is empty; run the extractor/ingestor"})

    # --- helpers ---
    def _match(ci_val, user_val):
        if user_val is None: 
            return True
        return str(ci_val or "").strip().lower() == str(user_val).strip().lower()

    def _year_of(it):
        # prefer explicit date "YYYY-MM-DD"
        d = (it.get("date") or "")
        if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
            return int(d[:4])
        # else try period "YYYY_Qn"
        p = (it.get("period") or "")
        m = re.match(r"^(20\d{2})_?Q([1-4])$", str(p), re.I)
        if m: 
            return int(m.group(1))
        # else None
        return None

    def _period_key(it):
        # Higher sorts first (newest first)
        # Parse "YYYY_Qn" → (YYYY, Q) else fallback to date[:10]
        p = str(it.get("period") or "")
        m = re.match(r"^(20\d{2})_?Q([1-4])$", p, re.I)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        # Try date "YYYY-MM-DD"
        d = str(it.get("date") or "")
        if re.match(r"^20\d{2}-\d{2}-\d{2}$", d):
            return (int(d[:4]), int(d[5:7]) // 3 + 1)
        # Unknown → push to bottom
        return (0, 0)

    # --- filter by user constraints ---
    pool = []
    for it in items:
        if not _match(it.get("ticker"), ticker): 
            continue
        if not _match(it.get("form"), form): 
            continue
        if not _match(it.get("table_type"), table_type): 
            continue
        if year is not None:
            y = _year_of(it)
            if y is None or y != int(year):
                continue
        pool.append(it)

    if not pool:
        # If no matches, help caller by returning available combos
        seen = {
            "tickers": sorted({str(it.get("ticker") or "") for it in items}),
            "forms": sorted({str(it.get("form") or "") for it in items}),
            "table_types": sorted({str(it.get("table_type") or "") for it in items}),
            "years": sorted({str(_year_of(it)) for it in items if _year_of(it) is not None})
        }
        return json.dumps({"error": "no matching CSV", "available": seen})

    # --- pick most recent ---
    pool.sort(key=_period_key, reverse=True)
    chosen = pool[0]

    # --- load and set state ---
    path = chosen["path"]
    try:
        # tolerate odd encodings
        df = pd.read_csv(path, encoding_errors="ignore")
    except Exception as e:
        return json.dumps({"error": f"failed to read CSV: {e}", "path": path})

    # Store DF for subsequent plotting/code tools
    STATE.add(path, df)

    # --- derive schema for LLM ---
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    label_col = chosen.get("label_col") or ("Period" if "Period" in df.columns else df.columns[0])
    # numeric columns (exclude label)
    numeric_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]

    schema = {
        "columns": list(df.columns),
        "dtypes": dtypes,
        "label_col": label_col,
        "numeric_cols": numeric_cols,
    }
    return json.dumps({"selected": path, "schema": schema})

@tool("get_current_df_schema")
def get_current_df_schema() -> str:
    """
    Return the columns, dtypes, label_col, numeric_cols for the current DataFrame.
    """
    df = STATE.get()
    if df is None:
        return json.dumps({"error":"no current DataFrame; call select_csv first"})
    return json.dumps(_schema_from_df(df))





def _coerce_code(code: Any) -> str:
    if isinstance(code, str):
        return code.strip()
    try:
        return str(code).strip()
    except Exception:
        return ""

@tool("run_python_on_current_df", return_direct=True)
def run_python_on_current_df(code: str, save_path: Optional[str] = None) -> dict:
    """
    Execute python that operates on `df` and produces a matplotlib figure.

    Returns a plain dict (NOT JSON string):
      {"stdout": str, "image_path": str|None, "saved": str|None}
      or {"error": str, "columns": [...]}

    Rules:
      - Available names: df, pd, np, plt
      - No file I/O / dangerous builtins (blocked: open(, exec(, eval(, __, os., sys.)
      - Do not re-load CSV files; `df` is selected by select_csv()
      - NO AUTO-FALLBACK: if code draws nothing, return a clear error.
    """
    df = STATE.get()
    if df is None:
        return {"error": "No DataFrame selected. Call select_csv first."}

    code_str = _coerce_code(code)
    lowered = code_str.lower()
    banned = ["open(", "exec(", "eval(", "__", "os.", "sys."]
    if any(tok in lowered for tok in banned):
        return {
            "error": "Disallowed code. Use only df, pd, np, plt. Do not import or read files.",
            "columns": list(df.columns),
        }

    # Ensure output folder
    os.makedirs("data/plots", exist_ok=True)
    out_path = save_path or os.path.join("data/plots", f"plot_{uuid.uuid4().hex[:10]}.png")


    # Namespace shared with exec()
    ns = {"df": df, "pd": pd, "np": np, "plt": plt}

    
    stdout = io.StringIO()
    fig = None
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code_str, ns)

        # Collect all figs that actually have content
        fig = plt.gcf()
        img_bytes = None
        if fig:
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(out_path, format="png")
            plt.close(fig)
            buf.seek(0)
            img_bytes = buf.getvalue()
        return {"stdout": stdout.getvalue(), "image_path": out_path, "saved": out_path}
        
            

    except Exception as e:
        plt.close("all")
        return {"error": f"Plotting failed: {e}", "columns": list(df.columns)}
