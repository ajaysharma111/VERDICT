"""Synthetic DEMO candles anchored to approximate real NSE prices.
Every frame carries attrs['source']='DEMO'. Never used unless explicitly selected.

Prices are seeded to match approximate Nov-2026 NSE levels so the demo feels
realistic (₹2600 for RELIANCE, ₹3800 for TCS, etc.) while still being fully
synthetic — no live data is fetched.
"""
import numpy as np
import pandas as pd

WATCHLIST = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "SBIN", "LT", "WIPRO", "TATAMOTORS"]

# Approximate real NSE closing prices (Nov 2026 reference levels).
# These are realistic anchors — the random walk starts from here.
_BASE_PRICES: dict[str, float] = {
    "RELIANCE":   2680.0,
    "TCS":        3820.0,
    "INFY":       1875.0,
    "HDFCBANK":   1740.0,
    "ICICIBANK":   985.0,
    "ITC":         470.0,
    "SBIN":        780.0,
    "LT":         3550.0,
    "WIPRO":       555.0,
    "TATAMOTORS":  970.0,
    # generic fallback for unlisted symbols: mid-cap range
}
_DEFAULT_BASE = 1200.0


# Per-symbol realistic volatility (annualised σ approximation divided to daily).
# Large-caps: ~1.2-1.5% daily σ; mid-caps: ~1.5-2%.
_DAILY_VOL: dict[str, float] = {
    "RELIANCE":   0.012,
    "TCS":        0.013,
    "INFY":       0.014,
    "HDFCBANK":   0.013,
    "ICICIBANK":  0.014,
    "ITC":        0.011,
    "SBIN":       0.016,
    "LT":         0.013,
    "WIPRO":      0.015,
    "TATAMOTORS": 0.018,
}
_DEFAULT_VOL = 0.015


def demo_candles(symbol: str, days: int = 750, seed: int | None = None) -> pd.DataFrame:
    sym = symbol.upper().strip()
    rng = np.random.default_rng(seed if seed is not None else sum(map(ord, sym)) * 7919)

    # Volatility and base price anchored to known NSE levels
    daily_vol = _DAILY_VOL.get(sym, _DEFAULT_VOL)
    base_price = _BASE_PRICES.get(sym, _DEFAULT_BASE)

    # Long-term drift: ~8-12% annual return for large-caps (realistic equity premium)
    annual_drift = rng.uniform(0.08, 0.12)
    daily_drift = annual_drift / 252

    # Regime: bull/bear cycles with realistic 3-18 month duration
    regime_period = rng.uniform(3, 8)   # number of sine cycles across the window
    regime_phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, regime_period * np.pi, days)
    regime_signal = np.sin(t + regime_phase)

    # Fat-tailed daily returns (Student-t with df=5 approximates equity tails)
    df_t = 5
    u = rng.standard_t(df_t, size=days)
    u = u / u.std()   # rescale so std ≈ 1
    rets = daily_drift + regime_signal * daily_vol * 0.4 + u * daily_vol

    # Build OHLCV starting from the anchored base price
    # The geometric random walk ends at a price ≈ base_price * e^(drift*days)
    # We reverse-anchor so the LAST close ≈ base_price (feels current)
    raw_close = np.exp(np.cumsum(rets))
    # Scale so last value == 1, then multiply by base_price
    close = base_price * (raw_close / raw_close[-1])

    # Realistic intraday range: open gaps ±0.3%, hi/lo wicks ±0.5-0.8%
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, days))
    open_ = np.clip(open_, close * 0.95, close * 1.05)   # cap overnight gaps at 5%

    # High/low with realistic intraday range proportional to daily vol
    intraday_range = np.abs(rng.normal(0, daily_vol * 0.6, days))
    hi = np.maximum(open_, close) * (1 + intraday_range)
    lo = np.minimum(open_, close) * (1 - intraday_range)

    # Volume: realistic NSE large-cap volumes with event spikes
    base_vol = {
        "RELIANCE": 6_000_000, "TCS": 1_500_000, "INFY": 4_000_000,
        "HDFCBANK": 8_000_000, "ICICIBANK": 10_000_000, "ITC": 15_000_000,
        "SBIN": 20_000_000, "LT": 2_000_000, "WIPRO": 5_000_000,
        "TATAMOTORS": 12_000_000,
    }.get(sym, 3_000_000)

    vol_multiplier = 1 + 2.0 * np.abs(rets) / (daily_vol * 2)   # higher vol on big moves
    volume = rng.integers(
        int(base_vol * 0.4), int(base_vol * 1.6), days
    ).astype(float) * vol_multiplier

    idx = pd.bdate_range(end=pd.Timestamp("2026-11-06"), periods=days, tz="Asia/Kolkata")
    df = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume.astype(int)},
        index=idx,
    ).round(2)
    df.attrs["source"] = "DEMO"
    df.attrs["symbol"] = sym
    return df
