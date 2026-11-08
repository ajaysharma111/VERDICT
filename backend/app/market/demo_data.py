"""Synthetic DEMO candles. Every frame carries attrs['source']='DEMO'. Never used unless explicitly selected."""
import numpy as np
import pandas as pd

WATCHLIST = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "SBIN", "LT", "WIPRO", "TATAMOTORS"]


def demo_candles(symbol: str, days: int = 750, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else sum(map(ord, symbol)) * 7919)
    drift = rng.normal(0.0004, 0.0002)
    # slow-moving regime so trend strategies have something to find (and to fail on)
    regime = np.sign(np.sin(np.linspace(0, rng.uniform(4, 9), days) + rng.uniform(0, 6)))
    rets = drift + regime * 0.0006 + rng.normal(0, 0.014, days)
    close = rng.uniform(300, 3000) * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, days))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, days)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, days)))
    vol = rng.integers(1_000_000, 8_000_000, days) * (1 + 0.5 * np.abs(rets) / 0.014)
    idx = pd.bdate_range(end=pd.Timestamp("2026-11-06"), periods=days, tz="Asia/Kolkata")
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx).round(2)
    df.attrs["source"] = "DEMO"
    df.attrs["symbol"] = symbol
    return df
