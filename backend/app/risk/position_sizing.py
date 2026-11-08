from decimal import Decimal, ROUND_DOWN


def position_size(equity: Decimal, risk_pct: Decimal, entry: Decimal, stop: Decimal,
                  multiplier: Decimal = Decimal("1")) -> int:
    """Whole-share quantity: floor(equity * risk% / (entry - stop) * judge multiplier). 0 if invalid."""
    if entry <= 0 or stop <= 0 or stop >= entry or equity <= 0:
        return 0
    multiplier = min(max(multiplier, Decimal("0")), Decimal("1"))  # a Judge can shrink, never enlarge
    qty = (equity * risk_pct / (entry - stop)) * multiplier
    return int(qty.to_integral_value(rounding=ROUND_DOWN))
