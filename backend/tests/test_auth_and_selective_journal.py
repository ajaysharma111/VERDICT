from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth import (
    authenticate_user,
    create_session,
    get_selected_stocks,
    get_user_from_token,
    hash_password,
    invalidate_session,
    register_user,
    set_selected_stocks,
    toggle_selected_stock,
    verify_password,
)
from app.database.models import make_session_factory
from app.main import app


@pytest.fixture
def test_db():
    factory = make_session_factory("sqlite:///:memory:")
    return factory


def test_password_hashing():
    pwd = "SecretPassword123"
    h, salt = hash_password(pwd)
    assert verify_password(pwd, salt, h) is True
    assert verify_password("WrongPassword", salt, h) is False


def test_user_registration_and_auth(test_db):
    with test_db() as s:
        user = register_user(s, "alice", "alice@example.com", "mypassword")
        s.commit()
        assert user.id is not None
        assert user.username == "alice"
        assert user.email == "alice@example.com"

        # Duplicate username
        with pytest.raises(ValueError, match="already taken"):
            register_user(s, "alice", "different@example.com", "mypassword")

        # Duplicate email
        with pytest.raises(ValueError, match="already registered"):
            register_user(s, "bob", "alice@example.com", "mypassword")

        # Short password
        with pytest.raises(ValueError, match="Password must be at least"):
            register_user(s, "charlie", "charlie@example.com", "123")

        # Authenticate with username or email
        u1 = authenticate_user(s, "alice", "mypassword")
        assert u1 is not None and u1.id == user.id

        u2 = authenticate_user(s, "alice@example.com", "mypassword")
        assert u2 is not None and u2.id == user.id

        # Bad password
        assert authenticate_user(s, "alice", "wrongpwd") is None


def test_session_lifecycle(test_db):
    with test_db() as s:
        user = register_user(s, "session_user", "sess@example.com", "pwd12345")
        token = create_session(s, user.id)
        s.commit()

        u = get_user_from_token(s, token)
        assert u is not None and u.id == user.id

        # Invalid token
        assert get_user_from_token(s, "badtoken") is None

        # Logout / Invalidate
        assert invalidate_session(s, token) is True
        s.commit()
        assert get_user_from_token(s, token) is None


def test_selected_stocks_toggle(test_db):
    with test_db() as s:
        user = register_user(s, "trader_dan", "dan@example.com", "pwd12345")
        s.commit()

        # Initial
        assert get_selected_stocks(s, user.id) == []

        # Toggle on RELIANCE
        sel, all_s = toggle_selected_stock(s, user.id, "RELIANCE")
        assert sel is True
        assert all_s == ["RELIANCE"]

        # Toggle on TCS
        sel, all_s = toggle_selected_stock(s, user.id, "TCS")
        assert sel is True
        assert set(all_s) == {"RELIANCE", "TCS"}

        # Toggle off RELIANCE
        sel, all_s = toggle_selected_stock(s, user.id, "RELIANCE")
        assert sel is False
        assert all_s == ["TCS"]

        # Bulk set
        new_list = set_selected_stocks(s, user.id, ["INFY", "WIPRO", "HDFCBANK"])
        s.commit()
        assert set(new_list) == {"INFY", "WIPRO", "HDFCBANK"}
        assert set(get_selected_stocks(s, user.id)) == {"INFY", "WIPRO", "HDFCBANK"}


def test_auth_api_flow():
    client = TestClient(app)
    headers = {"Host": "localhost", "X-Verdict-Client": "1"}

    import uuid
    uid = uuid.uuid4().hex[:8]
    uname = f"trader_{uid}"
    uemail = f"trader_{uid}@test.com"

    # 1. Register a new user
    reg_resp = client.post(
        "/api/auth/register",
        json={"username": uname, "email": uemail, "password": "password123"},
        headers=headers,
    )
    assert reg_resp.status_code == 200, reg_resp.text
    data = reg_resp.json()
    assert data["ok"] is True
    token = data["token"]
    assert token

    auth_headers = {**headers, "Authorization": f"Bearer {token}"}

    # 2. Check /api/auth/me
    me_resp = client.get("/api/auth/me", headers=auth_headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["authenticated"] is True
    assert me_resp.json()["user"]["username"] == uname

    # 3. Check selected stocks
    stocks_resp = client.get("/api/user/selected-stocks", headers=auth_headers)
    assert stocks_resp.status_code == 200
    selected = stocks_resp.json()["selected_stocks"]
    assert isinstance(selected, list)

    # 4. Set specific selected stocks: only INFY is selected
    set_resp = client.post(
        "/api/user/selected-stocks/set",
        json={"symbols": ["INFY"]},
        headers=auth_headers,
    )
    assert set_resp.status_code == 200
    assert set_resp.json()["selected_stocks"] == ["INFY"]

    # 5. Run trial on selected stock (INFY) -> Journal entry IS created
    trial_infy = client.post(
        "/api/trial/start",
        json={"symbol": "INFY", "simulation_mode": True, "trading_type": "INTRADAY"},
        headers=auth_headers,
    )
    assert trial_infy.status_code == 200
    infy_res = trial_infy.json()
    assert infy_res["journal_recorded"] is True
    assert infy_res["trial_id"] is not None

    # 6. Run trial on UNSELECTED stock (TCS) -> Journal entry is NOT created
    trial_tcs = client.post(
        "/api/trial/start",
        json={"symbol": "TCS", "simulation_mode": True, "trading_type": "INTRADAY"},
        headers=auth_headers,
    )
    assert trial_tcs.status_code == 200
    tcs_res = trial_tcs.json()
    assert tcs_res["journal_recorded"] is False
    assert tcs_res["trial_id"] is None
    assert tcs_res["transcript"]["journal"]["recorded"] is False

    # 7. Delete journal entry
    entry_id = infy_res["trial_id"]
    del_resp = client.delete(f"/api/journal/{entry_id}", headers=auth_headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 8. Check that journal entry is deleted
    get_del = client.get(f"/api/journal/{entry_id}", headers=auth_headers)
    assert get_del.status_code == 404

    # 9. Logout
    logout_resp = client.post("/api/auth/logout", headers=auth_headers)
    assert logout_resp.status_code == 200
    assert logout_resp.json()["ok"] is True

    # 10. Check me is unauthenticated
    me_after = client.get("/api/auth/me", headers=auth_headers)
    assert me_after.status_code == 200
    assert me_after.json()["authenticated"] is False


def test_user_account_isolation():
    client = TestClient(app)
    headers = {"Host": "localhost", "X-Verdict-Client": "1"}
    import uuid

    # User 1
    u1_name = f"user1_{uuid.uuid4().hex[:6]}"
    r1 = client.post("/api/auth/register", json={"username": u1_name, "email": f"{u1_name}@t.com", "password": "pass123"}, headers=headers)
    assert r1.status_code == 200
    t1 = r1.json()["token"]
    h1 = {**headers, "Authorization": f"Bearer {t1}"}

    # User 2
    u2_name = f"user2_{uuid.uuid4().hex[:6]}"
    r2 = client.post("/api/auth/register", json={"username": u2_name, "email": f"{u2_name}@t.com", "password": "pass123"}, headers=headers)
    assert r2.status_code == 200
    t2 = r2.json()["token"]
    h2 = {**headers, "Authorization": f"Bearer {t2}"}

    # Set User 1 selected stocks to ["RELIANCE"]
    client.post("/api/user/selected-stocks/set", json={"symbols": ["RELIANCE"]}, headers=h1)
    # Set User 2 selected stocks to ["TCS"]
    client.post("/api/user/selected-stocks/set", json={"symbols": ["TCS"]}, headers=h2)

    # Check that selected stocks are separate
    s1 = client.get("/api/user/selected-stocks", headers=h1).json()["selected_stocks"]
    s2 = client.get("/api/user/selected-stocks", headers=h2).json()["selected_stocks"]
    assert s1 == ["RELIANCE"]
    assert s2 == ["TCS"]

    # User 1 runs trial on RELIANCE -> recorded in User 1's journal
    tr1 = client.post("/api/trial/start", json={"symbol": "RELIANCE", "simulation_mode": True}, headers=h1)
    assert tr1.status_code == 200
    assert tr1.json()["journal_recorded"] is True

    # User 2's journal should NOT have User 1's entry
    j2 = client.get("/api/journal", headers=h2).json()
    assert all(e["symbol"] == "TCS" for e in j2)

