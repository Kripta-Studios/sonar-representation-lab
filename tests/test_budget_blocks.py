import hashlib
import json

import pytest

from budget import allocation_limits


def test_historical_cap_and_reserved_followup(tmp_path):
    old = (json.dumps({"event": "end", "charged_gpu_hours": 11.5}) + "\n").encode()
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(old)
    assert allocation_limits(ledger)["charge_ceiling_hours"] == 24
    block = tmp_path / "block.json"
    block.write_text(
        json.dumps(
            {
                "id": "spatial-motion-20261003",
                "baseline_gpu_hours": 11.5,
                "budget_gpu_hours": 24,
                "final_reserved_gpu_hours": 6,
                "historical_ledger_bytes": len(old),
                "historical_ledger_sha256": hashlib.sha256(old).hexdigest(),
            }
        )
    )
    limits = allocation_limits(ledger, block)
    assert limits["charge_ceiling_hours"] == 29.5
    assert limits["block_ceiling_hours"] == 35.5
    assert limits["remaining_development_hours"] == 18
    assert allocation_limits(ledger, block, original_operation=True)["charge_ceiling_hours"] == 24
    with ledger.open("a") as stream:
        stream.write(json.dumps({"event": "end", "charged_gpu_hours": 2, "status": "failed"}) + "\n")
    assert allocation_limits(ledger, block)["remaining_development_hours"] == 16
    assert allocation_limits(ledger)["charge_ceiling_hours"] == 24
    frozen = tmp_path / "freeze.json"
    frozen.write_text(
        json.dumps(
            {
                "block_id": "spatial-motion-20261003",
                "status": "frozen",
                "models": [{"checkpoint_sha256": "a" * 64}],
            }
        )
    )
    assert allocation_limits(ledger, block, "final", frozen)["charge_ceiling_hours"] == 35.5
    with pytest.raises(ValueError, match="freeze"):
        allocation_limits(ledger, block, "final")
    ledger.write_bytes(old.replace(b"11.5", b"10.5"))
    with pytest.raises(ValueError, match="Historical"):
        allocation_limits(ledger, block)


def test_reject_overall_reset_or_invalid_reserve(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("")
    block = tmp_path / "block.json"
    config = {
        "id": "x",
        "baseline_gpu_hours": 0,
        "budget_gpu_hours": 24,
        "final_reserved_gpu_hours": 5,
        "historical_ledger_bytes": 0,
        "historical_ledger_sha256": hashlib.sha256(b"").hexdigest(),
    }
    block.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="reserve"):
        allocation_limits(ledger, block)
