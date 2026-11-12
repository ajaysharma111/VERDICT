"""Risk Engine: ordinary Python, no LLM. Every entry must pass here before the paper broker sees it."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Optional

from ..timeutil import IST, as_utc
from .position_sizing import position_size
OPEN, CLOSE = time(9, 15), time(15, 30)


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: Decimal = Decimal("0.01")
    daily_loss_limit: Decimal = Decimal("0.03")
    max_open_positions: int = 10
    max_stock_exposure: Decimal = Decimal("0.15")
    max_total_exposure: Decimal = Decimal("1.00")
    checkin_timeout: Optional[timedelta] = None


@dataclass
class AccountState:
    """Snapshot the engine judges against. Built by the broker, never by the LLM."""
    equity: Decimal
    cash: Decimal
    start_of_day_equity: Decimal
    positions: dict[str, Decimal] = field(default_factory=dict)  # symbol -> market value
    halted: bool = False
    halt_reason: str = ""
    last_checkin: Optional[datetime] = None
    data_fresh: bool = True


@dataclass
class TradeProposal:
    symbol: str
    entry: Decimal
    stop: Optional[Decimal]
    target: Optional[Decimal]
    multiplier: Decimal = Decimal("1")  # Judge: 1 = APPROVE, <1 = REDUCE
    side: str = "BUY"


@dataclass
class RiskDecision:
    approved: bool
    quantity: int = 0
    reasons: list[str] = field(default_factory=list)


def is_market_open(now: datetime) -> bool:
    now = as_utc(now).astimezone(IST)
    return now.weekday() < 5 and OPEN <= now.time() <= CLOSE  # exchange holidays: not modelled


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def daily_loss_hit(self, s: AccountState) -> bool:
        if s.start_of_day_equity <= 0:
            return True
        loss = (s.start_of_day_equity - s.equity) / s.start_of_day_equity
        return loss >= self.limits.daily_loss_limit

    def check_entry(self, p: TradeProposal, s: AccountState, now: datetime,
                    simulation_mode: bool = False) -> RiskDecision:
        r: list[str] = []
        L = self.limits
        if s.halted:
            r.append(f"HALTED: {s.halt_reason or 'kill switch active'}")
        if not s.data_fresh:
            r.append("Market data is not fresh; automated entries disabled")
        if L.checkin_timeout is not None:
            if s.last_checkin is None or as_utc(now) - as_utc(s.last_checkin) > L.checkin_timeout:
                r.append("Dead-man switch: no human check-in within 24h")
        if not simulation_mode and not is_market_open(now):
            r.append("Outside market hours (09:15-15:30 IST, Mon-Fri)")
        if self.daily_loss_hit(s):
            r.append("Daily loss limit reached; entries locked")
        if p.side != "BUY":
            r.append("Only long entries are supported")
        if p.entry <= 0:
            r.append("Entry price must be positive")
        if p.stop is None:
            r.append("Missing stop-loss")
        elif p.stop <= 0 or p.stop >= p.entry:
            r.append("Stop-loss must be positive and below entry for a long")
        if p.target is not None and p.target <= p.entry:
            r.append("Target must be above entry")
        if p.symbol not in s.positions and len(s.positions) >= L.max_open_positions:
            r.append(f"Max open positions ({L.max_open_positions}) reached")
        if r:
            return RiskDecision(False, 0, r)

        qty = position_size(s.equity, L.risk_per_trade, p.entry, p.stop, p.multiplier)
        if qty < 1:
            return RiskDecision(False, 0, ["Computed quantity is below one share"])
        # Clamp to cash, per-stock exposure and total exposure instead of rejecting outright.
        held = s.positions.get(p.symbol, Decimal(0))
        room_stock = (s.equity * L.max_stock_exposure - held) / p.entry
        room_total = (s.equity * L.max_total_exposure - sum(s.positions.values(), Decimal(0))) / p.entry
        cash_room = s.cash / p.entry
        capped = min(Decimal(qty), room_stock, room_total, cash_room)
        capped_qty = int(capped) if capped > 0 else 0
        if capped_qty < 1:
            if p.symbol in s.positions and held >= s.equity * L.max_stock_exposure:
                why = f"Max exposure for {p.symbol} reached ({int(L.max_stock_exposure * 100)}% of portfolio); close existing position to trade again"
            elif cash_room < 1:
                why = "Insufficient cash in paper wallet"
            elif len(s.positions) >= L.max_open_positions:
                why = f"Max open positions ({L.max_open_positions}) reached"
            else:
                why = "Exposure limit leaves no room"
            return RiskDecision(False, 0, [why])
        notes = []
        if capped_qty < qty:
            notes.append(f"Quantity reduced {qty} -> {capped_qty} by cash/exposure limits")
        return RiskDecision(True, capped_qty, notes)
