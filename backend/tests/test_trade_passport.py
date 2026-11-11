from copy import deepcopy

from app.trading.passport import create_passport, verify_passport


class FakeOrder:
    id = 17
    client_order_id = "paper-17"
    execution_mode = "PAPER"
    symbol = "TCS"
    side = "BUY"
    order_type = "MARKET"
    quantity = 3
    reference_price = "3500.00"
    fill_price = "3501.25"
    slippage = "1.25"
    charges = "2.10"
    status = "FILLED"
    created_at = None
    executed_at = None
    bracket_stop = "3400"
    bracket_target = "3700"
    thesis_invalidation = "Close below support"


def test_trade_passport_hash_verifies_and_detects_mutation():
    passport = create_passport(
        FakeOrder(),
        {
            "symbol": "TCS",
            "source": "DEMO",
            "evidence": [{"id": "E1", "finding": "trend"}],
            "strategist": {"action": "BUY"},
            "prosecutor": {"severity": 0.2},
            "defender": {"severity": 0.1},
            "judge": {"decision": "APPROVE"},
            "risk": {"approved": True, "quantity": 3},
        },
    )

    assert passport["execution_mode"] == "PAPER"
    assert passport["order"]["status"] == "FILLED"
    assert verify_passport(passport)

    altered = deepcopy(passport)
    altered["order"]["fill_price"] = "9999"
    assert not verify_passport(altered)


def test_trade_passport_rejects_missing_or_invalid_digest():
    assert not verify_passport({})
    assert not verify_passport({"sha256": "not-a-hash"})
