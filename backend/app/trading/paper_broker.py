"""Internal simulated broker. It never imports or calls any market-data or broker client.

Fill model (documented, simulated): market orders fill at reference price moved against the trader by
`slippage_bps`; limit BUYs fill at the limit price once the reference price is <= limit. Stops and targets
are checked on each price update (stop fills at the stop price less slippage; gap-through fills at the
reference price, whichever is worse). Charges approximate Indian equity delivery costs. These are
simulated fills, not exchange executions.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import delete, select

from ..database.models import Account, JournalEntry, Order, Position, RiskLog, make_session_factory
from ..timeutil import ist_date

Q = Decimal("0.01")


def q(x: Decimal) -> Decimal:
    return x.quantize(Q, rounding=ROUND_HALF_UP)


def charges(value: Decimal, side: str) -> Decimal:
    """Brokerage 0.03% (max 20), STT 0.1%, exchange 0.00297%, GST 18% on brokerage+exchange, stamp 0.015% on buys."""
    brokerage = min(Decimal("20"), value * Decimal("0.0003"))
    exch = value * Decimal("0.0000297")
    gst = (brokerage + exch) * Decimal("0.18")
    stt = value * Decimal("0.001")
    stamp = value * Decimal("0.00015") if side == "BUY" else Decimal(0)
    return q(brokerage + exch + gst + stt + stamp)


class OrderError(Exception):
    pass


class PaperBroker:
    def __init__(self, db_url: str = "sqlite:///:memory:", initial_capital: Decimal = Decimal("100000"),
                 slippage_bps: Decimal = Decimal("5")):
        self.Session = make_session_factory(db_url)
        self.initial_capital = Decimal(initial_capital)
        self.slip = slippage_bps / Decimal(10000)
        with self.Session.begin() as s:
            if s.get(Account, 1) is None:
                s.add(Account(id=1, cash=initial_capital, initial_capital=initial_capital,
                              start_of_day_equity=initial_capital))


    def reset_wallet(self) -> Decimal:
        """Reset this paper wallet and its history without affecting other broker databases."""
        with self.Session.begin() as s:
            s.execute(delete(Position))
            s.execute(delete(Order))
            s.execute(delete(RiskLog))
            s.execute(delete(JournalEntry))

            account = s.get(Account, 1)
            account.cash = self.initial_capital
            account.initial_capital = self.initial_capital
            account.start_of_day_equity = self.initial_capital
            account.realized_pnl = Decimal(0)
            account.halted = False
            account.halt_reason = ""
            account.last_checkin = None
            account.sod_date = ""

        return self.initial_capital

    # ---------- reads ----------
    def account(self) -> Account:
        with self.Session() as s:
            return s.get(Account, 1)

    def positions(self) -> list[Position]:
        with self.Session() as s:
            return list(s.scalars(select(Position)))

    def orders(self) -> list[Order]:
        with self.Session() as s:
            return list(s.scalars(select(Order).order_by(Order.id)))

    def equity(self, prices: dict[str, Decimal]) -> Decimal:
        acct = self.account()
        return acct.cash + sum((p.avg_price if p.symbol not in prices else prices[p.symbol]) * p.quantity
                               for p in self.positions())

    def unrealized(self, prices: dict[str, Decimal]) -> Decimal:
        return sum(((prices.get(p.symbol, p.avg_price) - p.avg_price) * p.quantity for p in self.positions()),
                   Decimal(0))

    def position_values(self, prices: dict[str, Decimal]) -> dict[str, Decimal]:
        return {p.symbol: prices.get(p.symbol, p.avg_price) * p.quantity for p in self.positions()}

    # ---------- audit ----------
    def log(self, event: str, symbol: str = "", detail: str = "", now: datetime | None = None, s=None):
        entry = RiskLog(at=now or datetime.utcnow(), event=event, symbol=symbol, detail=detail)
        if s is not None:
            s.add(entry)
        else:
            with self.Session.begin() as s2:
                s2.add(entry)

    # ---------- orders ----------
    def submit(self, symbol: str, side: str, quantity: int, reference_price: Decimal, now: datetime,
               order_type: str = "MARKET", limit_price: Decimal | None = None, stop: Decimal | None = None,
               target: Decimal | None = None, invalidation: str = "", client_order_id: str | None = None) -> Order:
        side, order_type = side.upper(), order_type.upper()
        cid = client_order_id or uuid.uuid4().hex
        with self.Session.begin() as s:
            if s.scalar(select(Order).where(Order.client_order_id == cid)):
                raise OrderError("Duplicate order id")
            base = dict(client_order_id=cid, execution_mode="PAPER", symbol=symbol, side=side,
                        order_type=order_type, quantity=quantity, limit_price=limit_price,
                        reference_price=reference_price, created_at=now,
                        bracket_stop=stop, bracket_target=target, thesis_invalidation=invalidation)
            problem = None
            if side not in ("BUY", "SELL") or order_type not in ("MARKET", "LIMIT"):
                problem = "Invalid side or order type"
            elif not isinstance(quantity, int) or quantity < 1:
                problem = "Invalid quantity"
            elif reference_price is None or reference_price <= 0:
                problem = "Invalid reference price"
            elif order_type == "LIMIT" and (limit_price is None or limit_price <= 0):
                problem = "Limit order needs a positive limit price"
            elif side == "BUY" and stop is None:
                problem = "BUY requires a stop-loss"
            elif side == "BUY" and s.get(Account, 1).halted:
                problem = "Trading halted: new buys are blocked"
            if problem:
                o = Order(**base, status="REJECTED", reason=problem)
                s.add(o)
                return o
            o = Order(**base, status="PENDING")
            s.add(o)
            s.flush()
            if side == "BUY":
                if order_type == "MARKET" or reference_price <= limit_price:
                    self._fill_buy(s, o, reference_price, now, stop, target, invalidation)
                else:
                    o.reason = "Limit not reached"
            else:
                self._fill_sell(s, o, reference_price, now)
            return o

    def _fill_buy(self, s, o: Order, ref: Decimal, now, stop, target, invalidation):
        acct = s.get(Account, 1)
        px = q(min(o.limit_price, ref * (1 + self.slip)) if o.order_type == "LIMIT" else ref * (1 + self.slip))
        value = px * o.quantity
        fee = charges(value, "BUY")
        if value + fee > acct.cash:
            o.status, o.reason = "REJECTED", "Insufficient virtual cash"
            return
        acct.cash -= value + fee
        pos = s.scalar(select(Position).where(Position.symbol == o.symbol))
        if pos:
            total = pos.quantity + o.quantity
            pos.avg_price = q((pos.avg_price * pos.quantity + px * o.quantity) / total)
            pos.quantity = total
            pos.stop, pos.target = stop or pos.stop, target or pos.target
        else:
            s.add(Position(symbol=o.symbol, quantity=o.quantity, avg_price=px, stop=stop, target=target,
                           thesis_invalidation=invalidation))
        o.fill_price, o.slippage, o.charges = px, q((px - ref) * o.quantity), fee
        o.status, o.executed_at, o.reason = "FILLED", now, "Simulated fill"

    def _fill_sell(self, s, o: Order, ref: Decimal, now, force_price: Decimal | None = None):
        pos = s.scalar(select(Position).where(Position.symbol == o.symbol))
        if not pos or pos.quantity < o.quantity:
            o.status, o.reason = "REJECTED", "No long position to sell (shorting not supported)"
            return
        px = force_price if force_price is not None else q(ref * (1 - self.slip))
        value = px * o.quantity
        fee = charges(value, "SELL")
        acct = s.get(Account, 1)
        acct.cash += value - fee
        acct.realized_pnl += (px - pos.avg_price) * o.quantity - fee
        pos.quantity -= o.quantity
        if pos.quantity == 0:
            s.delete(pos)
        o.fill_price, o.slippage, o.charges = px, q((ref - px) * o.quantity), fee
        o.status, o.executed_at, o.reason = "FILLED", now, o.reason or "Simulated fill"

    def cancel(self, order_id: int) -> bool:
        with self.Session.begin() as s:
            o = s.get(Order, order_id)
            if o and o.status == "PENDING":
                o.status, o.reason = "CANCELLED", "Cancelled by user"
                return True
            return False

    def cancel_all_pending(self, reason: str, s=None) -> int:
        def work(sess):
            n = 0
            for o in sess.scalars(select(Order).where(Order.status == "PENDING")):
                o.status, o.reason = "CANCELLED", reason
                n += 1
            return n
        if s is not None:
            return work(s)
        with self.Session.begin() as s2:
            return work(s2)

    # ---------- price updates ----------
    def on_prices(self, prices: dict[str, Decimal], now: datetime) -> list[Order]:
        """Fill reachable limit buys, then trigger stops/targets. Returns orders that filled."""
        filled: list[Order] = []
        with self.Session.begin() as s:
            for o in list(s.scalars(select(Order).where(Order.status == "PENDING", Order.side == "BUY"))):
                px = prices.get(o.symbol)
                if px is not None and px <= o.limit_price and not s.get(Account, 1).halted:
                    o.reference_price = px
                    self._fill_buy(s, o, px, now, o.bracket_stop, o.bracket_target, o.thesis_invalidation)
                    if o.status == "FILLED":
                        filled.append(o)
            for pos in list(s.scalars(select(Position))):
                px = prices.get(pos.symbol)
                if px is None:
                    continue
                hit = None
                if pos.stop is not None and px <= pos.stop:
                    hit, fill = "STOP", min(pos.stop, px)
                elif pos.target is not None and px >= pos.target:
                    hit, fill = "TARGET", max(pos.target, Decimal(0))
                if hit:
                    o = Order(client_order_id=uuid.uuid4().hex, execution_mode="PAPER", symbol=pos.symbol,
                              side="SELL", order_type="MARKET", quantity=pos.quantity, reference_price=px,
                              status="PENDING", reason=f"{hit} triggered", created_at=now)
                    s.add(o)
                    s.flush()
                    self._fill_sell(s, o, px, now, force_price=q(fill * (1 - self.slip)))
                    filled.append(o)
                    self.log(f"{hit}_HIT", pos.symbol, f"ref {px}", now, s)
        return filled

    # ---------- safety controls ----------
    def roll_day(self, prices: dict[str, Decimal], now: datetime) -> bool:
        """At the first call on a new IST day: reset the daily-loss baseline and release an automatic daily lock
        (a human kill switch is never released automatically)."""
        day = ist_date(now).isoformat()
        if self.account().sod_date == day:
            return False
        eq = self.equity(prices)
        with self.Session.begin() as s:
            a = s.get(Account, 1)
            a.sod_date, a.start_of_day_equity = day, eq
            if a.halted and a.halt_reason.startswith("Daily loss limit"):
                a.halted, a.halt_reason = False, ""
                self.log("DAILY_LOCK_RELEASED", detail="new trading day", now=now, s=s)
        return True

    def lock_daily_loss(self, now: datetime) -> None:
        with self.Session.begin() as s:
            a = s.get(Account, 1)
            a.halted, a.halt_reason = True, "Daily loss limit reached"
            n = self.cancel_all_pending("Daily loss lock", s)
            self.log("DAILY_LOSS_LOCK", detail=f"entries locked; cancelled {n} pending", now=now, s=s)

    def checkin(self, now: datetime) -> None:
        with self.Session.begin() as s:
            s.get(Account, 1).last_checkin = now
            self.log("CHECKIN", detail="human check-in", now=now, s=s)

    def close_position(self, symbol: str, price: Decimal, now: datetime, reason: str = "Manual close") -> Order | None:
        with self.Session.begin() as s:
            pos = s.scalar(select(Position).where(Position.symbol == symbol))
            if not pos or pos.quantity <= 0:
                return None
            o = Order(client_order_id=uuid.uuid4().hex, execution_mode="PAPER", symbol=pos.symbol,
                      side="SELL", order_type="MARKET", quantity=pos.quantity, reference_price=price,
                      status="PENDING", reason=reason, created_at=now)
            s.add(o)
            s.flush()
            self._fill_sell(s, o, price, now, force_price=q(price * (1 - self.slip)))
            self.log("POSITION_CLOSED", pos.symbol, f"Manual square off at {price}", now, s)
            return o

    def new_day(self, prices: dict[str, Decimal]) -> None:
        eq = self.equity(prices)
        with self.Session.begin() as s:
            s.get(Account, 1).start_of_day_equity = eq

    def stop_trading(self, prices: dict[str, Decimal], now: datetime, reason: str = "Human kill switch") -> bool:
        """Persist halt FIRST, cancel pending, then try to flatten. Stays halted regardless of flatten outcome."""
        with self.Session.begin() as s:
            a = s.get(Account, 1)
            a.halted, a.halt_reason = True, reason
            n = self.cancel_all_pending("Kill switch", s)
            self.log("KILL_SWITCH", detail=f"{reason}; cancelled {n} pending", now=now, s=s)
        ok = True
        for pos in self.positions():
            px = prices.get(pos.symbol)
            if px is None:
                ok = False
                self.log("FLATTEN_FAILED", pos.symbol, "no price available; staying halted", now)
                continue
            with self.Session.begin() as s:
                o = Order(client_order_id=uuid.uuid4().hex, execution_mode="PAPER", symbol=pos.symbol,
                          side="SELL", order_type="MARKET", quantity=pos.quantity, reference_price=px,
                          status="PENDING", reason="Kill switch flatten", created_at=now)
                s.add(o)
                s.flush()
                self._fill_sell(s, o, px, now)
                if o.status != "FILLED":
                    ok = False
        return ok

    def resume(self, now: datetime) -> None:
        with self.Session.begin() as s:
            a = s.get(Account, 1)
            a.halted, a.halt_reason = False, ""
            self.log("RESUME", detail="human resumed", now=now, s=s)

    def replenish(self, amount: Decimal = Decimal("100000"), now: datetime | None = None) -> Decimal:
        """Add virtual cash to the paper account and adjust sod baseline so daily loss lock is released if halted."""
        with self.Session.begin() as s:
            a = s.get(Account, 1)
            a.cash += amount
            a.start_of_day_equity += amount
            if a.halted and a.halt_reason.startswith("Daily loss limit"):
                a.halted, a.halt_reason = False, ""
                self.log("DAILY_LOCK_RELEASED", detail="replenished capital", now=now, s=s)
            self.log("REPLENISH", detail=f"added {amount} virtual cash; cash is now {a.cash}", now=now or datetime.utcnow(), s=s)
            return a.cash
