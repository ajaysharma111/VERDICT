"""Data freshness. Cached or old candles must never be labelled live (STALE blocks new automated entries)."""
from datetime import datetime

import pandas as pd

from ..timeutil import IST, as_utc

MAX_AGE_DAYS = 4  # covers a weekend plus one exchange holiday


def assess(df: pd.DataFrame, now: datetime) -> tuple[str, bool]:
    """Returns (status, fresh). DEMO data is labelled DEMO and allowed only because demo mode was chosen explicitly."""
    if df.attrs.get("source") == "DEMO":
        return "DEMO", True
    if df.empty:
        return "DISCONNECTED", False
    last = pd.Timestamp(df.index[-1])
    last = last.tz_localize(IST) if last.tzinfo is None else last
    age = (as_utc(now) - last.to_pydatetime()).days
    return ("LIVE", True) if age <= MAX_AGE_DAYS else ("STALE", False)
