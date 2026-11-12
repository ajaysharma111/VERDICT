import ast
import pathlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pandas as pd
import pytest

from app.backtesting.engine import run_backtest, strategy_returns, ma_cross_signal
from app.config import UnsafeConfigError, assert_paper_only
from app.market.demo_data import demo_candles
from app.market.provider import AngelOneProvider, DemoProvider, MarketDataProvider
from app.risk.engine import AccountState, RiskEngine, RiskLimits, TradeProposal, IST
from app.risk.position_sizing import position_size
from app.trading.paper_broker import OrderError, PaperBroker

APP = pathlib.Path(__file__).parent.parent / "app"
OPEN_TIME = datetime(2026, 11, 9, 5, 0, tzinfo=timezone.utc)  # Mon 10:30 IST
NIGHT = datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)


def state(**kw):
    base = dict(equity=D(100000), cash=D(100000), start_of_day_equity=D(100000), last_checkin=OPEN_TIME)
    base.update(kw)
    return AccountState(**base)


def prop(**kw):
    base = dict(symbol="TCS", entry=D(100), stop=D(98), target=D(106))
    base.update(kw)
    return TradeProposal(**base)


R = RiskEngine()


# ---------- position sizing / risk ----------
def test_position_sizing_and_judge_multiplier():
    assert position_size(D(100000), D("0.01"), D(100), D(98)) == 500
    assert position_size(D(100000), D("0.01"), D(100), D(98), D("0.5")) == 250
    assert position_size(D(100000), D("0.01"), D(100), D(98), D("5")) == 500  # judge cannot enlarge
    assert position_size(D(100000), D("0.01"), D(100), D(100)) == 0


def test_exposure_clamp_to_40_percent():
    d = R.check_entry(prop(), state(), OPEN_TIME)
    assert d.approved and d.quantity == 400  # 40% of 1,00,000 / 100 = 400 (clamped from 500)


@pytest.mark.parametrize("kw,text", [
    (dict(stop=None), "Missing stop-loss"),
    (dict(stop=D(100)), "Stop-loss must"),
    (dict(stop=D(105)), "Stop-loss must"),
    (dict(entry=D(0)), "Entry price"),
    (dict(target=D(99)), "Target"),
])
def test_invalid_proposals_rejected(kw, text):
    d = R.check_entry(prop(**kw), state(), OPEN_TIME)
    assert not d.approved and any(text in r for r in d.reasons)


def test_daily_loss_limit_locks_entries():
    d = R.check_entry(prop(), state(equity=D(96900)), OPEN_TIME)
    assert not d.approved and any("Daily loss" in r for r in d.reasons)


def test_max_positions_and_insufficient_cash():
    full = state(positions={f"S{i}": D(1000) for i in range(10)})
    assert not R.check_entry(prop(), full, OPEN_TIME).approved
    assert not R.check_entry(prop(), state(cash=D(50)), OPEN_TIME).approved


def test_kill_switch_deadman_hours_stale_data():
    assert not R.check_entry(prop(), state(halted=True, halt_reason="x"), OPEN_TIME).approved
    R_with_checkin = RiskEngine(RiskLimits(checkin_timeout=timedelta(hours=24)))
    assert not R_with_checkin.check_entry(prop(), state(last_checkin=OPEN_TIME - timedelta(hours=25)), OPEN_TIME).approved
    assert not R_with_checkin.check_entry(prop(), state(last_checkin=None), OPEN_TIME).approved
    # By default, dead-man checkin is disabled
    assert R.check_entry(prop(), state(last_checkin=None), OPEN_TIME).approved
    assert not R.check_entry(prop(), state(), NIGHT).approved
    assert R.check_entry(prop(), state(), NIGHT, simulation_mode=True).approved
    assert not R.check_entry(prop(), state(data_fresh=False), OPEN_TIME).approved


# ---------- paper broker ----------
def broker():
    return PaperBroker()


def test_buy_sell_portfolio_and_charges():
    b = broker()
    o = b.submit("TCS", "BUY", 10, D(100), OPEN_TIME, stop=D(95))
    assert o.status == "FILLED" and o.execution_mode == "PAPER" and o.fill_price > D(100) and o.charges > 0
    assert b.account().cash < D(100000)
    s = b.submit("TCS", "SELL", 10, D(110), OPEN_TIME)
    assert s.status == "FILLED" and b.positions() == [] and b.account().realized_pnl > 0


def test_rejections_invalid_qty_cash_short_duplicate():
    b = broker()
    assert b.submit("TCS", "BUY", 0, D(100), OPEN_TIME, stop=D(95)).status == "REJECTED"
    assert b.submit("TCS", "BUY", 5000, D(100), OPEN_TIME, stop=D(95)).reason == "Insufficient virtual cash"
    assert b.submit("TCS", "SELL", 1, D(100), OPEN_TIME).status == "REJECTED"
    assert b.submit("TCS", "BUY", 1, D(100), OPEN_TIME).reason == "BUY requires a stop-loss"
    b.submit("TCS", "BUY", 1, D(100), OPEN_TIME, stop=D(95), client_order_id="a")
    with pytest.raises(OrderError):
        b.submit("TCS", "BUY", 1, D(100), OPEN_TIME, stop=D(95), client_order_id="a")


def test_limit_order_waits_then_fills():
    b = broker()
    o = b.submit("TCS", "BUY", 5, D(100), OPEN_TIME, order_type="LIMIT", limit_price=D(98), stop=D(95))
    assert o.status == "PENDING"
    assert b.on_prices({"TCS": D(99)}, OPEN_TIME) == []
    filled = b.on_prices({"TCS": D(97)}, OPEN_TIME)
    assert filled and filled[0].fill_price <= D(98)


def test_stop_and_target_trigger():
    b = broker()
    b.submit("TCS", "BUY", 10, D(100), OPEN_TIME, stop=D(95), target=D(110))
    out = b.on_prices({"TCS": D(94)}, OPEN_TIME)
    assert out[0].reason == "STOP triggered" and b.positions() == [] and b.account().realized_pnl < 0
    b.submit("INFY", "BUY", 10, D(100), OPEN_TIME, stop=D(95), target=D(110))
    assert b.on_prices({"INFY": D(111)}, OPEN_TIME)[0].reason == "TARGET triggered"


def test_kill_switch_persists_and_flattens(tmp_path):
    url = f"sqlite:///{tmp_path/'v.db'}"
    b = PaperBroker(url)
    b.submit("TCS", "BUY", 10, D(100), OPEN_TIME, stop=D(95))
    b.submit("INFY", "BUY", 5, D(100), OPEN_TIME, order_type="LIMIT", limit_price=D(90), stop=D(85))
    assert b.stop_trading({"TCS": D(101)}, OPEN_TIME) is True
    assert b.positions() == [] and all(o.status != "PENDING" for o in b.orders())
    assert PaperBroker(url).account().halted is True  # survives restart


def test_kill_switch_stays_halted_if_flatten_fails():
    b = broker()
    b.submit("TCS", "BUY", 10, D(100), OPEN_TIME, stop=D(95))
    assert b.stop_trading({}, OPEN_TIME) is False
    assert b.account().halted and len(b.positions()) == 1


# ---------- safety: nothing can reach a real broker ----------
def test_provider_interface_has_no_order_methods():
    names = {n for n in dir(MarketDataProvider) if not n.startswith("_")}
    assert names == {"connect", "historical", "quote", "status"}
    for cls in (DemoProvider, AngelOneProvider):
        assert not [n for n in dir(cls) if any(w in n.lower() for w in ("order", "place", "modify", "cancel"))]


def test_paper_modules_never_import_market_clients_or_order_apis():
    banned_imports = {"SmartApi", "requests", "httpx", "websocket"}
    for path in [APP / "trading" / "paper_broker.py", APP / "risk" / "engine.py", APP / "risk" / "position_sizing.py"]:
        tree = ast.parse(path.read_text())
        mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
               {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not mods & banned_imports, path
        assert "market" not in " ".join(m for m in {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)})


def test_no_order_placement_calls_anywhere_in_source():
    forbidden = ("placeOrder", "modifyOrder", "cancelOrder", "place_order", "gttCreateRule")
    for p in APP.rglob("*.py"):
        assert not any(f in p.read_text() for f in forbidden), p


def test_no_live_trading_switch_and_env_guard(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(UnsafeConfigError):
        assert_paper_only()


def test_all_orders_marked_paper():
    b = broker()
    b.submit("TCS", "BUY", 1, D(100), OPEN_TIME, stop=D(95))
    b.submit("TCS", "BUY", 0, D(100), OPEN_TIME, stop=D(95))
    assert {o.execution_mode for o in b.orders()} == {"PAPER"}


# ---------- backtest ----------
def test_no_lookahead_signal_is_lagged():
    df = demo_candles("TCS", 300)
    sig = ma_cross_signal(df)
    r, _ = strategy_returns(df, sig)
    # perturbing the LAST bar's close must not change any earlier strategy return
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] *= 1.5
    r2, _ = strategy_returns(df2, ma_cross_signal(df2))
    pd.testing.assert_series_equal(r.iloc[:-1], r2.iloc[:-1])


def test_backtest_reports_oos_vs_buy_and_hold():
    res = run_backtest(demo_candles("RELIANCE"), "DEMO")
    assert res["out_of_sample"] and res["kind"] == "HISTORICAL_BACKTEST"
    assert set(res["metrics"]) == {"momentum", "ma_crossover", "buy_and_hold"}
    for m in res["metrics"].values():
        assert {"total_return", "cagr", "max_drawdown", "sharpe", "win_rate", "trades", "cost_paid"} <= set(m)
    assert res["passes_gate"]["buy_and_hold"] is False


def test_pending_limit_bracket_survives_broker_restart(tmp_path):
    url = f"sqlite:///{tmp_path/'bracket.db'}"
    b = PaperBroker(url)
    pending = b.submit("TCS", "BUY", 5, D(100), OPEN_TIME, order_type="LIMIT",
                       limit_price=D(98), stop=D(92), target=D(112), invalidation="thesis broken")
    assert pending.status == "PENDING"
    restarted = PaperBroker(url)
    filled = restarted.on_prices({"TCS": D(97)}, OPEN_TIME)
    assert len(filled) == 1 and filled[0].status == "FILLED"
    pos = restarted.positions()[0]
    assert pos.stop == D(92) and pos.target == D(112) and pos.thesis_invalidation == "thesis broken"


def test_rejected_pending_limit_is_not_reported_as_filled():
    b = PaperBroker(initial_capital=D(100))
    pending = b.submit("TCS", "BUY", 100, D(100), OPEN_TIME, order_type="LIMIT",
                       limit_price=D(90), stop=D(80))
    assert pending.status == "PENDING"
    assert b.on_prices({"TCS": D(89)}, OPEN_TIME) == []
    assert b.orders()[-1].status == "REJECTED"
