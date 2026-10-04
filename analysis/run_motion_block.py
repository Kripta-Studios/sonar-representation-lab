"""Execute the single declared paired continuation, only after complete admission."""

import json
import os
import shutil
import sys
from pathlib import Path

from run_spatial_block import ALLOCATION, ART, PROJECT, call, read, remaining


def charged(relative):
    return read(ART / relative / "resources.json")["charged_gpu_hours"]


def main():
    sys.stdout.reconfigure(encoding="utf8")
    os.chdir(PROJECT)
    os.environ["SONAR_RESEARCH_ALLOCATION"] = str(ALLOCATION)
    os.environ["SONAR_RESEARCH_PHASE"] = "development"
    os.environ["SONAR_ORIGINAL_BLOCK_OPERATION"] = "0"
    screen = read(ART / "spatial_B_final_screen.json")
    if screen["admissible_B_replication"]:
        raise ValueError("An admissible main replication takes priority over the optional pilot")
    if "Paired continuation recipe, before calibration" not in Path("PROTOCOL.md").read_text(encoding="utf8"):
        raise ValueError("Declare the complete paired recipe before calibration")
    if (ART / "motion_block_forecast.json").exists():
        raise ValueError("Existing attempt: recover its state; do not duplicate this script")
    parent = "artifacts/ssl-B-s7/checkpoint.pt"
    calibration = "artifacts/motion-calibration/calibration.json"
    frozen = max(charged(name) for name in ("det-published-r672-f010-s7", "det-b-r672-f010-s7")) * 1.25
    validation = (
        max(
            charged(name + "/val")
            for name in ("det-published-r672-f010-s7", "det-b-r672-f010-s7", "det-finetune-r672-f010-s7")
        )
        * 1.25
    )
    prior_ssl = charged("ssl-B-s7") / 2 * 1.25 + 0.05
    preliminary = 2 * (prior_ssl + frozen + validation) + 0.20
    space = shutil.disk_usage(ART).free
    initial = {
        "prior_ssl_per_arm_hours": prior_ssl,
        "frozen_train_per_arm_hours": frozen,
        "validation_per_arm_hours": validation,
        "preliminary_complete_pair_hours": preliminary,
        "remaining_development_hours": remaining(),
        "final_reserve_hours": 6,
        "free_artifact_bytes": space,
        "required_additional_artifact_bytes": 6 * 1024**3,
        "status": "preflight only; no motion research fitting yet",
    }
    if preliminary > remaining() or space < 6 * 1024**3:
        initial["status"] = "NOT_ADMITTED_COMPLETE_PAIR_BUDGET_OR_SPACE"
        (ART / "motion_block_forecast.json").write_text(json.dumps(initial, indent=2), encoding="utf8")
        print(json.dumps(initial), flush=True)
        return
    (ART / "motion_preflight_forecast.json").write_text(json.dumps(initial, indent=2), encoding="utf8")
    call(
        ["src/calibrate_motion.py", "--parent", parent, "--out", "artifacts/motion-calibration"],
        "artifacts/motion-calibration",
        0.05,
    )
    ssl_forecasts = {}
    for recipe, extra in (("control", []), ("motion", ["--dynamic"])):
        for ending, resume in (
            ("full", []),
            ("repeat", []),
            ("resume", ["--stop-after", "2"]),
            ("resume", ["--resume", f"artifacts/ssl-{recipe}-profile-resume/checkpoint.pt"]),
        ):
            out = f"artifacts/ssl-{recipe}-profile-{ending}"
            call(
                [
                    "src/adapt_motion.py",
                    "--parent",
                    parent,
                    "--calibration",
                    calibration,
                    "--seed",
                    "7",
                    "--profile",
                    "--out",
                    out,
                    *extra,
                    *resume,
                ],
                out,
                0.03,
            )
        call(
            [
                "analysis/check_ssl_resume.py",
                "--motion",
                "--full",
                f"artifacts/ssl-{recipe}-profile-full/checkpoint.pt",
                "--reference-repeat",
                f"artifacts/ssl-{recipe}-profile-repeat/checkpoint.pt",
                "--resumed",
                f"artifacts/ssl-{recipe}-profile-resume/checkpoint.pt",
                "--out",
                f"artifacts/ssl-{recipe}-resume-check.json",
                *(
                    ["--paired-control", "artifacts/ssl-control-profile-full/checkpoint.pt"]
                    if recipe == "motion"
                    else []
                ),
            ],
            f"artifacts/ssl-{recipe}-verification",
        )
        rows = [
            json.loads(line)
            for ending in ("full", "repeat")
            for line in (ART / f"ssl-{recipe}-profile-{ending}/curve.jsonl").read_text().splitlines()
        ]
        ssl_forecasts[recipe] = max(
            prior_ssl, max(row["step_seconds"] for row in rows) * 1000 / 3600 * 1.25 + 0.05
        )
    total = sum(ssl_forecasts.values()) + 2 * (frozen + validation)
    forecast = {
        **initial,
        "profiled_ssl_hours": ssl_forecasts,
        "complete_pair_hours": total,
        "remaining_development_hours": remaining(),
        "status": "ADMITTED" if total <= remaining() else "NOT_ADMITTED_PROFILED_COST",
    }
    (ART / "motion_block_forecast.json").write_text(json.dumps(forecast, indent=2), encoding="utf8")
    if total > remaining():
        print(json.dumps(forecast), flush=True)
        return
    for recipe, extra in (("control", []), ("motion", ["--dynamic"])):
        ssl = f"artifacts/ssl-{recipe}-s7"
        detector = f"artifacts/det-{recipe}-r672-f010-s7"
        still_required = ["control", "motion"] if recipe == "control" else ["motion"]
        remainder_forecast = sum(ssl_forecasts[name] + frozen + validation for name in still_required)
        if remainder_forecast > remaining():
            raise RuntimeError(
                "Remaining complete paired work no longer fits; preserve completed/partial arms"
            )
        call(
            [
                "src/adapt_motion.py",
                "--parent",
                parent,
                "--calibration",
                calibration,
                "--seed",
                "7",
                "--out",
                ssl,
                *extra,
            ],
            ssl,
            ssl_forecasts[recipe],
        )
        call(["analysis/snapshot_source.py", ssl], ssl)
        paired_check = (
            ["--paired-other", "artifacts/ssl-control-s7/checkpoint.pt"] if recipe == "motion" else []
        )
        call(["analysis/check_ssl_artifact.py", "--run", ssl, *paired_check], ssl)
        call(
            [
                "src/train_detector.py",
                "train",
                "--kind",
                "adapted",
                "--seed",
                "7",
                "--fraction",
                "10",
                "--size",
                "672",
                "--steps",
                "2000",
                "--batch",
                "8",
                "--accumulation",
                "1",
                "--adapted",
                ssl + "/encoder.pt",
                "--out",
                detector,
            ],
            detector,
            frozen,
        )
        call(["analysis/snapshot_source.py", detector], detector)
        call(
            [
                "src/train_detector.py",
                "predict",
                "--checkpoint",
                detector + "/checkpoint.pt",
                "--out",
                detector + "/val",
                "--batch",
                "8",
            ],
            detector + "/val",
            validation,
        )
    print("PAIRED_MOTION_BLOCK_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
