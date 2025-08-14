# src/databot/session_state.py
from __future__ import annotations
from typing import Optional, Dict
import pandas as pd

class _State:
    def __init__(self) -> None:
        self._dfs: Dict[str, pd.DataFrame] = {}
        self.current_key: Optional[str] = None

    def add(self, key: str, df: pd.DataFrame):
        self._dfs[key] = df
        self.current_key = key

    def get(self, key: Optional[str] = None) -> Optional[pd.DataFrame]:
        if key is None:
            key = self.current_key
        if key is None:
            return None
        return self._dfs.get(key)

    def set_current(self, key: str):
        if key in self._dfs:
            self.current_key = key

STATE = _State()
