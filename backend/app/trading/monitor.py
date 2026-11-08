"""Periodic safety monitor: roll the trading day, trigger stops/targets, enforce the daily loss lock."""
from datetime import datetime
from decimal import Decimal

from ..risk.engine import AccountState, RiskEngine
from .paper_broker import PaperBroker


def monitor_tick(broker: PaperBroker, risk: RiskEngine, prices: dict[str, Decimal], now: datetime):
    broker.roll_day(prices, now)
    filled = broker.on_prices(prices, now)
    a = broker.account()
    if not a.halted:
        st = AccountState(equity=broker.equity(prices), cash=a.cash, start_of_day_equity=a.start_of_day_equity)
        if risk.daily_loss_hit(st):
            broker.lock_daily_loss(now)
    return filled
