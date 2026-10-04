"""Explicit research-block accounting over one append-only historical ledger."""

import hashlib
import json
from pathlib import Path


def allocation_limits(ledger, allocation=None, phase="development", freeze=None, original_operation=False):
    raw = Path(ledger).read_bytes() if Path(ledger).exists() else b""
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    total = sum(row.get("charged_gpu_hours", 0) for row in rows)
    if allocation is None:
        return {
            "block_id": "original",
            "baseline_gpu_hours": 0,
            "block_ceiling_hours": 24,
            "charge_ceiling_hours": min(24, 100),
            "remaining_development_hours": 24 - total,
            "original_operation": True,
            "phase": "original",
        }
    record = json.loads(Path(allocation).read_text())
    prefix = raw[: record["historical_ledger_bytes"]]
    if (
        len(prefix) != record["historical_ledger_bytes"]
        or hashlib.sha256(prefix).hexdigest() != record["historical_ledger_sha256"]
    ):
        raise ValueError("Historical ledger prefix changed")
    baseline = sum(
        json.loads(line).get("charged_gpu_hours", 0) for line in prefix.splitlines() if line.strip()
    )
    if abs(baseline - record["baseline_gpu_hours"]) > 1e-10:
        raise ValueError("Historical balance mismatch")
    budget, reserve = record["budget_gpu_hours"], record["final_reserved_gpu_hours"]
    if not (6 <= reserve < budget <= 24):
        raise ValueError("Invalid follow-up budget or final reserve")
    ceiling = min(100, baseline + budget)
    development_ceiling = ceiling - reserve
    if phase == "final":
        if freeze is None or not Path(freeze).exists():
            raise ValueError("Final evaluation requires the fixed roster freeze")
        frozen = json.loads(Path(freeze).read_text())
        if (
            frozen.get("block_id") != record["id"]
            or frozen.get("status") != "frozen"
            or not 1 <= len(frozen.get("models", [])) <= 12
        ):
            raise ValueError("Invalid final roster freeze")
        current_ceiling = ceiling
    elif phase == "development":
        current_ceiling = development_ceiling
    else:
        raise ValueError("Unknown research phase")
    if original_operation:
        later = [json.loads(line) for line in raw[len(prefix) :].splitlines() if line.strip()]
        original_used = baseline + sum(
            r.get("charged_gpu_hours", 0)
            for r in later
            if r.get("original_operation", r.get("block_id", "original") == "original")
        )
        current_ceiling = min(current_ceiling, total + 24 - original_used)
    return {
        "block_id": record["id"],
        "baseline_gpu_hours": baseline,
        "block_ceiling_hours": ceiling,
        "charge_ceiling_hours": current_ceiling,
        "remaining_development_hours": development_ceiling - total,
        "original_operation": original_operation,
        "phase": phase,
        "allocation_sha256": hashlib.sha256(Path(allocation).read_bytes()).hexdigest(),
    }
