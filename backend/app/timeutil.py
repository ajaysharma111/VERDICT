from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def as_utc(dt: datetime) -> datetime:
    """Naive datetimes in this app are UTC (that is how they are stored); never let Python guess local time."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def ist_date(dt: datetime) -> date:
    return as_utc(dt).astimezone(IST).date()
