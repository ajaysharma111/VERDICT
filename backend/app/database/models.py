from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (Boolean, DateTime, Integer, String, Text, TypeDecorator, create_engine,
                        ForeignKey, Index, inspect, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Money(TypeDecorator):
    """Decimal stored as text so SQLite never rounds money through floats."""
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else str(Decimal(value))

    def process_result_value(self, value, dialect):
        return None if value is None else Decimal(value)


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "portfolio"
    id: Mapped[int] = mapped_column(primary_key=True)
    cash: Mapped[Decimal] = mapped_column(Money)
    initial_capital: Mapped[Decimal] = mapped_column(Money)
    start_of_day_equity: Mapped[Decimal] = mapped_column(Money)
    realized_pnl: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))
    halted: Mapped[bool] = mapped_column(Boolean, default=False)       # persisted kill switch
    halt_reason: Mapped[str] = mapped_column(String(200), default="")
    last_checkin: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sod_date: Mapped[str] = mapped_column(String(10), default="")  # IST trading day start_of_day_equity belongs to


class Position(Base):
    __tablename__ = "positions"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True)
    quantity: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[Decimal] = mapped_column(Money)
    stop: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    target: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    thesis_invalidation: Mapped[str] = mapped_column(Text, default="")  # "I am wrong if..."


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    execution_mode: Mapped[str] = mapped_column(String(8), default="PAPER")
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(4))
    order_type: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer)
    limit_price: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    reference_price: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    fill_price: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    slippage: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    charges: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    status: Mapped[str] = mapped_column(String(12), index=True)  # PENDING FILLED CANCELLED REJECTED
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    bracket_stop: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    bracket_target: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    thesis_invalidation: Mapped[str] = mapped_column(Text, default="")


class RiskLog(Base):
    __tablename__ = "risk_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    event: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32), default="")
    detail: Mapped[str] = mapped_column(Text, default="")


class JournalEntry(Base):
    __tablename__ = "trading_journal"
    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    outcome: Mapped[str] = mapped_column(String(16))  # EXECUTED / REJECTED_BY_JUDGE / BLOCKED_BY_RISK / NO_TRADE
    summary: Mapped[str] = mapped_column(Text)        # plain-language journal text
    transcript_json: Mapped[str] = mapped_column(Text)  # evidence, 4 agent outputs, risk result
    process_score: Mapped[int] = mapped_column(Integer, default=0)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    salt: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class UserSession(Base):
    __tablename__ = "user_sessions"
    token: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class UserSelectedStock(Base):
    __tablename__ = "user_selected_stocks"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    selected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def make_session_factory(url: str):
    engine = create_engine(url, future=True,
                           connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    Base.metadata.create_all(engine)
    # create_all does not add columns to an existing SQLite database. Apply additive,
    # backwards-compatible columns so pending bracket metadata survives restarts.
    if url.startswith("sqlite"):
        columns = {c["name"] for c in inspect(engine).get_columns("orders")}
        with engine.begin() as conn:
            if "bracket_stop" not in columns:
                conn.execute(text("ALTER TABLE orders ADD COLUMN bracket_stop VARCHAR(64)"))
            if "bracket_target" not in columns:
                conn.execute(text("ALTER TABLE orders ADD COLUMN bracket_target VARCHAR(64)"))
            if "thesis_invalidation" not in columns:
                conn.execute(text("ALTER TABLE orders ADD COLUMN thesis_invalidation TEXT NOT NULL DEFAULT ''"))
    return sessionmaker(engine, expire_on_commit=False)
