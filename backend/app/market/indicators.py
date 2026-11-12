import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(dn.notna())


def macd(s: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(s: pd.Series, n=20, k=2.0) -> pd.DataFrame:
    m, sd = s.rolling(n).mean(), s.rolling(n).std()
    return pd.DataFrame({"mid": m, "upper": m + k * sd, "lower": m - k * sd})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def volatility(s: pd.Series, n: int = 20) -> pd.Series:
    """Annualised stdev of daily log returns."""
    return np.log(s).diff().rolling(n).std() * np.sqrt(252)


def indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = df.copy()
    out["sma20"], out["sma50"], out["ema20"] = sma(c, 20), sma(c, 50), ema(c, 20)
    out["ema9"] = ema(c, 9)
    out["sma200"] = sma(c, 200).bfill().fillna(c)
    out["rsi14"] = rsi(c)
    out = out.join(macd(c))
    out["atr14"] = atr(df)
    out["vol20"] = volatility(c)
    out["momentum20"] = c.pct_change(20)
    out["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    return out
