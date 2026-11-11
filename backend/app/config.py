"""VERDICT configuration. Paper trading only: there is deliberately NO setting that enables live trading."""
import os
from dataclasses import dataclass
from decimal import Decimal

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv optional
    pass

EXECUTION_MODE = "PAPER"  # constant, not configurable


class UnsafeConfigError(RuntimeError):
    pass


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def assert_paper_only() -> None:
    """Refuse to start if the environment tries to enable real orders."""
    for name in ("LIVE_TRADING_ENABLED", "ALLOW_REAL_ORDERS"):
        if _flag(name):
            raise UnsafeConfigError(f"{name} is set, but VERDICT is paper-trading only and refuses to start.")


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    database_url: str = "sqlite:///./verdict.db"
    initial_capital: Decimal = Decimal("100000")
    risk_per_trade: Decimal = Decimal("0.01")
    daily_loss_limit: Decimal = Decimal("0.03")
    max_open_positions: int = 5
    max_stock_exposure: Decimal = Decimal("0.20")
    slippage_bps: Decimal = Decimal("5")
    market_data_mode: str = "demo"  # demo | live  (never switches silently)
    timezone: str = "Asia/Kolkata"
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")


def load_settings() -> Settings:
    assert_paper_only()
    d = lambda k, v: Decimal(os.getenv(k, v))
    return Settings(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./verdict.db"),
        initial_capital=d("INITIAL_CAPITAL", "100000"),
        risk_per_trade=d("RISK_PER_TRADE", "0.01"),
        daily_loss_limit=d("DAILY_LOSS_LIMIT", "0.03"),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "5")),
        max_stock_exposure=d("MAX_STOCK_EXPOSURE", "0.20"),
        slippage_bps=d("SLIPPAGE_BPS", "5"),
        market_data_mode=os.getenv("MARKET_DATA_MODE", "demo").lower(),
        allowed_hosts=tuple(dict.fromkeys(
            ["127.0.0.1", "localhost", "::1"]
            + [
                host.strip().lower().rstrip(".")
                for host in os.getenv("ALLOWED_HOSTS", "").split(",")
                if host.strip()
            ]
        )),
    )
