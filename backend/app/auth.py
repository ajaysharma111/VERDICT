from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from .database.models import User, UserSession, UserSelectedStock


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Secure password hashing using PBKDF2-HMAC-SHA256 with salt and 100,000 iterations."""
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        100_000,
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Constant-time password verification."""
    pwd_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(pwd_hash, expected_hash)


def register_user(session: Session, username: str, email: str, password: str) -> User:
    username = username.strip().lower()
    email = email.strip().lower()

    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters long")
    if not username.replace("_", "").isalnum():
        raise ValueError("Username must contain only letters, numbers, and underscores")
    if "@" not in email or "." not in email:
        raise ValueError("Invalid email address")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters long")

    # Check if username or email already taken
    existing_user = session.scalar(select(User).where(User.username == username))
    if existing_user:
        raise ValueError("Username is already taken")

    existing_email = session.scalar(select(User).where(User.email == email))
    if existing_email:
        raise ValueError("Email is already registered")

    pwd_hash, salt = hash_password(password)
    user = User(
        username=username,
        email=email,
        password_hash=pwd_hash,
        salt=salt,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        is_active=True,
    )
    session.add(user)
    session.flush()
    return user


def authenticate_user(session: Session, username_or_email: str, password: str) -> User | None:
    ident = username_or_email.strip().lower()
    user = session.scalar(
        select(User).where(
            (User.username == ident) | (User.email == ident)
        )
    )
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.salt, user.password_hash):
        return None
    return user


def create_session(session: Session, user_id: int, duration_days: int = 7) -> str:
    token = secrets.token_hex(32)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    sess = UserSession(
        token=token,
        user_id=user_id,
        created_at=now,
        expires_at=now + timedelta(days=duration_days),
    )
    session.add(sess)
    session.flush()
    return token


def get_user_from_token(session: Session, token: str) -> User | None:
    if not token:
        return None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    sess = session.scalar(
        select(UserSession).where(
            (UserSession.token == token) & (UserSession.expires_at > now)
        )
    )
    if not sess:
        return None
    user = session.get(User, sess.user_id)
    if not user or not user.is_active:
        return None
    return user


def invalidate_session(session: Session, token: str) -> bool:
    if not token:
        return False
    sess = session.scalar(select(UserSession).where(UserSession.token == token))
    if sess:
        session.delete(sess)
        session.flush()
        return True
    return False


def get_selected_stocks(session: Session, user_id: int) -> list[str]:
    """Return all stock symbols selected for trading by the user."""
    rows = session.scalars(
        select(UserSelectedStock.symbol)
        .where(UserSelectedStock.user_id == user_id)
        .order_by(UserSelectedStock.symbol)
    ).all()
    return list(rows)


def toggle_selected_stock(session: Session, user_id: int, symbol: str) -> tuple[bool, list[str]]:
    """Toggle stock selection for trading. Returns (is_now_selected, all_selected_stocks)."""
    sym = symbol.strip().upper()
    existing = session.scalar(
        select(UserSelectedStock).where(
            (UserSelectedStock.user_id == user_id) & (UserSelectedStock.symbol == sym)
        )
    )
    if existing:
        session.delete(existing)
        session.flush()
        is_selected = False
    else:
        new_item = UserSelectedStock(
            user_id=user_id,
            symbol=sym,
            selected_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(new_item)
        session.flush()
        is_selected = True

    all_selected = get_selected_stocks(session, user_id)
    return is_selected, all_selected


def set_selected_stocks(session: Session, user_id: int, symbols: list[str]) -> list[str]:
    """Replace all selected stocks for trading for the user."""
    session.execute(
        delete(UserSelectedStock).where(UserSelectedStock.user_id == user_id)
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    clean_syms = sorted(list(set(s.strip().upper() for s in symbols if s.strip())))
    for sym in clean_syms:
        session.add(UserSelectedStock(user_id=user_id, symbol=sym, selected_at=now))
    session.flush()
    return clean_syms
