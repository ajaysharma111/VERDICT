"""Long-only daily backtester. Signals use data up to the close of bar t and trade at the OPEN of bar t+1,
so there is no look-ahead. Costs: SLIPPAGE_BPS + CHARGE_BPS per side (documented approximation).
Historical backtests are research results; they are not paper-portfolio performance."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd

from ..market.indicators import ema, rsi

SLIPPAGE_BPS = 5.0
CHARGE_BPS = 13.0  # STT 0.1% + brokerage/exchange/GST/stamp, delivery equity, per side (approx.)
COST_PER_SIDE = (SLIPPAGE_BPS + CHARGE_BPS) / 10_000


# ---- strategies: return 0/1 desired position known at each bar's close ----
def momentum_signal(df: pd.DataFrame, lookback: int = 20, rsi_cap: int = 80) -> pd.Series:
    mom = df["close"].pct_change(lookback)
    return ((mom > 0) & (rsi(df["close"]) < rsi_cap)).astype(int)


def ma_cross_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    return (ema(df["close"], fast) > ema(df["close"], slow)).astype(int)


def buy_hold_signal(df: pd.DataFrame, **_) -> pd.Series:
    return pd.Series(1, index=df.index)


STRATEGIES = {
    "momentum": (momentum_signal, {"lookback": [10, 20, 40], "rsi_cap": [70, 80]}),
    "ma_crossover": (ma_cross_signal, {"fast": [10, 20], "slow": [50, 100]}),
    "buy_and_hold": (buy_hold_signal, {}),
}


def strategy_returns(df: pd.DataFrame, signal: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Per-bar returns and per-bar number of sides traded (for cost accounting)."""
    pos = signal.shift(1).fillna(0)          # decided at close t-1, active during bar t
    prev = pos.shift(1).fillna(0)
    o, c, pc = df["open"], df["close"], df["close"].shift(1)
    r = pd.Series(0.0, index=df.index)
    hold = (pos == 1) & (prev == 1)
    enter = (pos == 1) & (prev == 0)
    exit_ = (pos == 0) & (prev == 1)
    r[hold] = (c / pc - 1)[hold]
    r[enter] = (c / o - 1)[enter]
    r[exit_] = (o / pc - 1)[exit_]
    sides = (pos != prev).astype(int)
    sides.iloc[0] = int(pos.iloc[0] == 1)    # the first buy-and-hold purchase also pays costs
    return r - sides * COST_PER_SIDE, sides


@dataclass
class Metrics:
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    trades: int
    avg_trade_pnl: float
    cost_paid: float


def compute_metrics(r: pd.Series, sides: pd.Series) -> Metrics:
    r = r.fillna(0)
    eq = (1 + r).cumprod()
    n = max(len(r), 1)
    total = float(eq.iloc[-1] - 1) if n else 0.0
    cagr = float(eq.iloc[-1] ** (252 / n) - 1) if n and eq.iloc[-1] > 0 else -1.0
    dd = float((eq / eq.cummax() - 1).min())
    sd = r.std()
    sharpe = float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0
    # trade = contiguous holding stretch
    held = (r != 0).astype(int)
    grp = (held.diff().fillna(held.iloc[0]) != 0).cumsum()
    stretches = [(1 + r[g.index]).prod() - 1 for _, g in held.groupby(grp) if g.iloc[0] == 1]
    wins = sum(1 for x in stretches if x > 0)
    return Metrics(total, cagr, dd, sharpe, wins / len(stretches) if stretches else 0.0, int(sides.sum() // 2 or len(stretches)),
                   float(np.mean(stretches)) if stretches else 0.0, float(sides.sum() * COST_PER_SIDE))


def _best_params(df: pd.DataFrame, fn, grid: dict) -> dict:
    if not grid:
        return {}
    best, best_s = {}, -1e9
    keys = list(grid)
    for combo in product(*grid.values()):
        p = dict(zip(keys, combo))
        if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
            continue
        r, sd = strategy_returns(df, fn(df, **p))
        s = compute_metrics(r, sd).sharpe
        if s > best_s:
            best, best_s = p, s
    return best


def walk_forward(df: pd.DataFrame, name: str, train: int = 252, test: int = 63) -> dict:
    """Rolling: pick params on `train` bars, trade the next `test` bars unseen; stitch the out-of-sample folds."""
    fn, grid = STRATEGIES[name]
    oos_r, oos_s, folds = [], [], []
    start = 0
    while start + train + test <= len(df):
        tr, te = df.iloc[start:start + train], df.iloc[start:start + train + test]
        p = _best_params(tr, fn, grid)
        sig = fn(te, **p)                       # warm-up uses the train window; signals still causal
        r, sd = strategy_returns(te, sig)
        oos_r.append(r.iloc[train:]); oos_s.append(sd.iloc[train:])
        folds.append({"train_end": str(tr.index[-1].date()), "test_end": str(te.index[-1].date()), "params": p})
        start += test
    if not oos_r:
        raise ValueError("Not enough history for walk-forward testing")
    return {"returns": pd.concat(oos_r), "sides": pd.concat(oos_s), "folds": folds}


def run_backtest(df: pd.DataFrame, source: str = "UNKNOWN") -> dict:
    """Out-of-sample comparison of each strategy against buy-and-hold over the same bars."""
    results, wf = {}, {}
    for name in STRATEGIES:
        w = walk_forward(df, name)
        wf[name] = w
        results[name] = asdict(compute_metrics(w["returns"], w["sides"]))
    bh = results["buy_and_hold"]
    gates = {n: bool(m["sharpe"] > bh["sharpe"] and n != "buy_and_hold") for n, m in results.items()}
    eq = {n: ((1 + wf[n]["returns"].fillna(0)).cumprod() * 100000).round(2).tolist() for n in results}
    return {"data_source": source, "kind": "HISTORICAL_BACKTEST", "out_of_sample": True,
            "metrics": results, "passes_gate": gates, "equity_curves": eq,
            "dates": [str(d.date()) for d in wf["buy_and_hold"]["returns"].index],
            "folds": {n: wf[n]["folds"] for n in wf},
            "assumptions": {"cost_per_side_bps": SLIPPAGE_BPS + CHARGE_BPS, "execution": "next-bar open"}}
