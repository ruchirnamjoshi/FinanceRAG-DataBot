# src/ingest/tickers.py
TICKER_TO_CIK = {
    "AAPL": 320193, "MSFT": 789019, "AMZN": 1018724, "GOOGL": 1652044, "GOOG": 1652044,
    "META": 1318605, "NVDA": 1045810, "TSLA": 1090872, "NFLX": 1326801, "CRM": 1329099,
    "ADBE": 796343, "INTC": 50863, "AMD": 2488,
    # add more as needed (we can maintain a JSON file later)
}
NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "meta": "META",
    "nvidia": "NVDA", "tesla": "TSLA", "netflix": "NFLX",
    "salesforce": "CRM", "adobe": "ADBE", "intel": "INTC", "advanced micro devices": "AMD",
}

def ticker_to_cik(s: str | None):
    if not s: return None
    return TICKER_TO_CIK.get(s.upper())

def name_to_cik(name: str | None):
    if not name: return None
    t = NAME_TO_TICKER.get(name.lower())
    return TICKER_TO_CIK.get(t) if t else None
