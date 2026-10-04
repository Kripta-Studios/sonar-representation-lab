"""Preserve the completed experiment and declare the owner-authorized follow-up."""

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ART = PROJECT / "artifacts"
sys.path.insert(0, str(PROJECT / "src"))
from acquire import save_json  # noqa: E402
from runtime import code_identity  # noqa: E402


def main():
    target = ART / "localization-block"
    if target.exists():
        raise ValueError("Follow-up already declared; recover rather than reset")
    if (ART / "gpu_process.lock").exists():
        raise ValueError("An existing GPU owner must finish first")
    raw = (ART / "resource_ledger.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    starts = {row["pid"] for row in rows if row["event"] == "start"}
    ends = {row["pid"] for row in rows if row["event"] == "end"}
    if starts - ends:
        raise ValueError("Unclosed historical intervals require recovery")
    baseline = sum(row.get("charged_gpu_hours", 0) for row in rows)
    frozen = json.loads((ART / "freeze.json").read_text())
    current = code_identity()
    assert current == frozen["code_identity"]
    assert hashlib.sha256((PROJECT / "PROTOCOL.md").read_bytes()).hexdigest() == frozen["protocol_sha256"]
    target.mkdir()
    files = [*sorted((PROJECT / "src").glob("*.py")), *sorted((PROJECT / "tests").glob("*.py"))]
    files += [PROJECT / name for name in ("README.md", "RESULTS.md", "MODEL_CARD.md", "PROTOCOL.md")]
    files += [
        ART / name for name in ("freeze.json", "final_roster.json", "final_execution_verification.json")
    ]
    with zipfile.ZipFile(target / "historical_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(PROJECT).as_posix())
    save_json(
        target / "historical_identity.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": current,
            "protocol_sha256": frozen["protocol_sha256"],
            "archive_sha256": hashlib.sha256((target / "historical_source.zip").read_bytes()).hexdigest(),
            "historical_ledger_sha256": hashlib.sha256(raw).hexdigest(),
            "channel_status": "completed and exposed; no further Channel inference in this block",
        },
    )
    save_json(
        target / "allocation.json",
        {
            "id": "localization-20261003",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "baseline_gpu_hours": baseline,
            "budget_gpu_hours": 24,
            "final_reserved_gpu_hours": 6,
            "historical_ledger_bytes": len(raw),
            "historical_ledger_sha256": hashlib.sha256(raw).hexdigest(),
            "authorization": "Owner: go ahead and implement and do all of that; adopts proposed bounded 24h block",
            "overall_initial_balance_hours": 100,
            "overall_remaining_at_start_hours": 100 - baseline,
        },
    )
    print(json.dumps({"baseline_hours": baseline, "allocation": str(target / "allocation.json")}))


if __name__ == "__main__":
    main()
