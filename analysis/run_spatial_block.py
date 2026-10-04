"""Serial execution of the declared B seed-7 block; no model/parameter search."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ART = PROJECT / "artifacts"
ALLOCATION = ART / "spatial_motion_allocation.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def remaining():
    allocation = read(ALLOCATION)
    used = sum(
        json.loads(line).get("charged_gpu_hours", 0)
        for line in (ART / "resource_ledger.jsonl").read_text().splitlines()
    )
    return allocation["baseline_gpu_hours"] + 24 - allocation["final_reserved_gpu_hours"] - used


def call(arguments, out, forecast=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if forecast is not None:
        if (ART / "gpu_process.lock").exists() or remaining() < forecast:
            raise RuntimeError("Active GPU owner or insufficient development allowance")
        admission = out / ("admission-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".json")
        admission.write_text(
            json.dumps(
                {
                    "arguments": arguments,
                    "forecast_hours": forecast,
                    "remaining_development_hours": remaining(),
                    "final_reserve_protected": True,
                },
                indent=2,
            ),
            encoding="utf8",
        )
    command = [sys.executable, "-u", *map(str, arguments)]
    print("EXECUTE", subprocess.list2cmdline(command), flush=True)
    with (out / "console.log").open("a", encoding="utf8") as log:
        result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Execution failed ({result.returncode}); preserve {out}")


def main():
    sys.stdout.reconfigure(encoding="utf8")
    os.chdir(PROJECT)
    os.environ["SONAR_RESEARCH_ALLOCATION"] = str(ALLOCATION)
    os.environ["SONAR_RESEARCH_PHASE"] = "development"
    os.environ["SONAR_ORIGINAL_BLOCK_OPERATION"] = "0"
    decision = read(ART / "resolution_decision.json")
    size = decision["selected_detector_size"]
    if size not in (448, 672) or remaining() < 8:
        raise RuntimeError("Spatial block admission requires the completed decision and eight-hour allowance")
    verification = ART / "ssl-B-resume-check-v2.json"
    if verification.exists():
        import hashlib

        config = read(ART / "ssl-B-profile-v2-full/config.json")
        for source in ("src/adapt.py", "src/spatial.py", "src/models.py", "src/runtime.py"):
            assert (
                hashlib.sha256((PROJECT / source).read_bytes()).hexdigest() == config["code_identity"][source]
            )
        assert read(verification)["checkpoint_state_restoration_and_numeric_continuation_pass"]
    else:
        for name, extra in [
            ("ssl-B-profile-v2-full", []),
            ("ssl-B-profile-v2-repeat", []),
            ("ssl-B-profile-v2-resume", ["--stop-after", "2"]),
            ("ssl-B-profile-v2-resume", ["--resume", "artifacts/ssl-B-profile-v2-resume/checkpoint.pt"]),
        ]:
            out = Path("artifacts") / name
            call(
                [
                    "src/adapt.py",
                    "--configuration",
                    "B",
                    "--seed",
                    "7",
                    "--batch",
                    "16",
                    "--steps",
                    "2000",
                    "--profile",
                    "--out",
                    str(out),
                    *extra,
                ],
                out,
                0.03,
            )
        call(
            [
                "analysis/check_ssl_resume.py",
                "--reference-repeat",
                "artifacts/ssl-B-profile-v2-repeat/checkpoint.pt",
                "--full",
                "artifacts/ssl-B-profile-v2-full/checkpoint.pt",
                "--resumed",
                "artifacts/ssl-B-profile-v2-resume/checkpoint.pt",
                "--out",
                "artifacts/ssl-B-resume-check-v2.json",
            ],
            "artifacts/ssl-B-resume-verification-v2",
        )
        # Checkpoint restoration must pass before an actual B research fit.
        assert read(ART / "ssl-B-resume-check-v2.json")[
            "checkpoint_state_restoration_and_numeric_continuation_pass"
        ]
    suffix = "" if size == 448 else "-r672"
    profile_path = ART / f"profile-b8-finetune{suffix}-v3/profile.json"
    if not profile_path.exists():
        call(
            [
                "src/profile_batch.py",
                "--out",
                f"artifacts/profile-b8-finetune{suffix}-v3",
                "--finetune",
                "--batch",
                "8",
                "--size",
                str(size),
                "--allocation",
                str(ALLOCATION),
            ],
            f"artifacts/profile-b8-finetune{suffix}-v3",
            0.03,
        )
    profile = read(profile_path)
    assert profile["checkpoint_reload_identical"]
    # Include a conservative cold-start batch in the training runtime estimator.
    fine_forecast = max(profile["microbatch_seconds"]) * 2000 / 3600 * 1.25 + 0.05
    ssl_resources = read(ART / "ssl-B-profile-v2-full/resources.json")
    ssl_forecast = ssl_resources["charged_gpu_hours"] / 4 * 2000 * 1.25 + 0.05
    reference = ART / ("det-published-f010-s7" if size == 448 else "det-published-r672-f010-s7")
    frozen_forecast = read(reference / "resources.json")["charged_gpu_hours"] * 1.25
    val_forecast = read(reference / "val/resources.json")["charged_gpu_hours"] * 1.25
    new_fine_reference = size != 448
    whole = (
        ssl_forecast
        + frozen_forecast
        + fine_forecast * (1 + new_fine_reference)
        + val_forecast * (2 + new_fine_reference)
    )
    forecast_record = ART / "spatial_block_forecast.json"
    if forecast_record.exists():
        raise ValueError("Preserve existing block forecast; inspect partial execution before any resume")
    forecast_record.write_text(
        json.dumps(
            {
                "size": size,
                "ssl_hours": ssl_forecast,
                "frozen_hours": frozen_forecast,
                "fine_hours": fine_forecast,
                "validation_hours": val_forecast,
                "whole_mandatory_block_hours": whole,
                "remaining_development_hours": remaining(),
                "reuse_published448_finetune": not new_fine_reference,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    if remaining() < whole:
        raise RuntimeError("Measured mandatory block forecast does not fit; retain probes and reserve")
    ssl_out = Path("artifacts/ssl-B-s7")
    call(
        [
            "src/adapt.py",
            "--configuration",
            "B",
            "--seed",
            "7",
            "--batch",
            "16",
            "--steps",
            "2000",
            "--out",
            str(ssl_out),
        ],
        ssl_out,
        ssl_forecast,
    )
    call(["analysis/snapshot_source.py", str(ssl_out)], ssl_out)
    call(
        [
            "analysis/check_ssl_artifact.py",
            "--run",
            str(ssl_out),
            "--compare-A",
            "artifacts/ssl-A-s7/checkpoint.pt",
        ],
        ssl_out,
    )
    jobs = [
        (f"det-b-r{size}-f010-s7", "adapted", True, frozen_forecast),
        (f"det-b-finetune-r{size}-f010-s7", "finetune", True, fine_forecast),
    ]
    if new_fine_reference:
        jobs.append(("det-finetune-r672-f010-s7", "finetune", False, fine_forecast))
    for name, kind, adapted, forecast in jobs:
        out = Path("artifacts") / name
        extra = ["--adapted", str(ssl_out / "encoder.pt")] if adapted else []
        call(
            [
                "src/train_detector.py",
                "train",
                "--kind",
                kind,
                "--seed",
                "7",
                "--fraction",
                "10",
                "--size",
                str(size),
                "--steps",
                "2000",
                "--batch",
                "8",
                "--accumulation",
                "1",
                "--out",
                str(out),
                *extra,
            ],
            out,
            forecast,
        )
        call(["analysis/snapshot_source.py", str(out)], out)
        call(
            [
                "src/train_detector.py",
                "predict",
                "--checkpoint",
                str(out / "checkpoint.pt"),
                "--out",
                str(out / "val"),
                "--batch",
                "8",
            ],
            out / "val",
            val_forecast,
        )
    print("SPATIAL_BLOCK_COMPLETED: assess the declared replication and optional motion rules", flush=True)


if __name__ == "__main__":
    main()
