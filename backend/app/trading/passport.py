"""Hashable audit snapshots for filled PAPER trades.

The digest is an integrity check for the stored snapshot, not an external
signature: an actor able to rewrite both data and digest can recompute it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_passport(order, trial: dict) -> dict:
    """Create a deterministic, self-contained audit passport for one simulated fill."""
    snapshot = {
        "schema": "verdict.trade-passport.v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "execution_mode": "PAPER",
        "trial": _json_value(trial),
        "order": _json_value({
            "id": order.id,
            "client_order_id": order.client_order_id,
            "execution_mode": order.execution_mode,
            "symbol": order.symbol,
            "side": order.side,
            "order_type": order.order_type,
            "quantity": order.quantity,
            "reference_price": order.reference_price,
            "fill_price": order.fill_price,
            "slippage": order.slippage,
            "charges": order.charges,
            "status": order.status,
            "created_at": order.created_at,
            "executed_at": order.executed_at,
            "bracket_stop": order.bracket_stop,
            "bracket_target": order.bracket_target,
            "thesis_invalidation": order.thesis_invalidation,
        }),
    }
    snapshot["sha256"] = _digest(snapshot)
    return snapshot


def verify_passport(passport: dict) -> bool:
    if not isinstance(passport, dict):
        return False
    supplied = passport.get("sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        return False
    payload = {key: value for key, value in passport.items() if key != "sha256"}
    try:
        return hashlib.compare_digest(_digest(payload), supplied)
    except (TypeError, ValueError):
        return False
