"""Finite serial detector experiment; conditional cells follow the fixed screen."""

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
from acquire import save_json  # noqa: E402
from budget import allocation_limits  # noqa: E402
from runtime import LEDGER, code_identity, file_sha  # noqa: E402

ART = PROJECT / "artifacts"
BLOCK = ART / "localization-block"
ALLOCATION = BLOCK / "allocation.json"
DIAGNOSTICS = BLOCK / "diagnostics"
MICROBATCH = 8
ACCUMULATION = 1


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def remaining():
    return allocation_limits(LEDGER, ALLOCATION)["remaining_development_hours"]


def execute(arguments, directory, forecast):
    if (ART / "gpu_process.lock").exists() or remaining() < forecast:
        raise RuntimeError(
            "GPU owner exists or complete operation cannot fit protected development allowance"
        )
    directory.mkdir(parents=True, exist_ok=True)
    admission = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arguments": arguments,
        "forecast_hours": forecast,
        "remaining_development_hours": remaining(),
        "allocation_sha256": file_sha(ALLOCATION),
        "code_identity": code_identity(),
    }
    save_json(directory / f"admission-{time.time_ns()}.json", admission)
    command = [sys.executable, "-u", *map(str, arguments)]
    print("EXECUTE", subprocess.list2cmdline(command), flush=True)
    with (directory / "console.log").open("a", encoding="utf8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=PROJECT)
    if result.returncode:
        raise RuntimeError(f"Job failed ({result.returncode}); preserve {directory} and recover actual state")


def diagnose(directory):
    target = DIAGNOSTICS / f"{directory.name}.json"
    if not target.exists():
        command = [
            sys.executable,
            "analysis/diagnostics.py",
            "--runs",
            str(directory),
            "--out",
            str(DIAGNOSTICS),
        ]
        with (BLOCK / "diagnostics_console.log").open("a", encoding="utf8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return read(target)


def subset_recall(diagnostics):
    point = next(row for row in diagnostics["operating_points"] if row["score_threshold"] == 0.5)
    bins = [row for name, row in point["size_bins"].items() if name != ">=16px"]
    return sum(row["matched"] for row in bins) / sum(row["objects"] for row in bins)


def verify_training(directory, reference=None):
    import torch

    state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["step"] == state["config"]["steps"] == 2000
    assert state["config"]["batch"] * state["config"]["accumulation"] == 8
    assert sum(state["exposures"].values()) == 16000
    assert all(torch.isfinite(value).all() for value in state["model"].values())
    if reference:
        old = torch.load(reference / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert state["exposures"] == old["exposures"]
        assert state["config"]["manifest_sha256"] == old["config"]["manifest_sha256"]
        del old
    save_json(
        directory / "artifact_verification.json",
        {
            "successful_updates": state["step"],
            "presentations": sum(state["exposures"].values()),
            "unique_images": len(state["exposures"]),
            "finite": True,
            "exact_reference_exposures": str(reference) if reference else None,
            "checkpoint_sha256": file_sha(directory / "checkpoint.pt"),
        },
    )
    del state
    with zipfile.ZipFile(directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((PROJECT / "src").glob("*.py")):
            archive.write(path, path.relative_to(PROJECT).as_posix())
        archive.write(PROJECT / "PROTOCOL.md", "PROTOCOL.md")


def fit_and_score(name, neck, seed, fraction, train_forecast, val_forecast, adapted=None, reference=None):
    directory = ART / name
    if not (directory / "artifact_verification.json").exists():
        if remaining() < train_forecast + val_forecast:
            raise RuntimeError(
                "Full fit/evaluation forecast cannot fit; do not start partial scientific cell"
            )
        if (directory / "checkpoint.pt").exists():
            import torch

            state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
            step = state["step"]
            del state
        else:
            step = 0
        if step < 2000:
            args = [
                "src/train_detector.py",
                "train",
                "--kind",
                "finetune",
                "--size",
                "672",
                "--neck",
                neck,
                "--fraction",
                str(fraction),
                "--seed",
                str(seed),
                "--steps",
                "2000",
                "--batch",
                str(MICROBATCH),
                "--accumulation",
                str(ACCUMULATION),
                "--out",
                str(directory),
                "--allocation",
                str(ALLOCATION),
            ]
            if adapted:
                args += ["--adapted", str(adapted)]
            if step:
                args += ["--resume", str(directory / "checkpoint.pt")]
            if not step and neck == "none" and seed == 7 and fraction == 10 and MICROBATCH != 8:
                execute([*args, "--stop-after", "2"], directory, 0.05)
                execute([*args, "--resume", str(directory / "checkpoint.pt")], directory, train_forecast)
            else:
                execute(args, directory, train_forecast)
        verify_training(directory, reference)
    if not (directory / "val/metrics.json").exists():
        execute(
            [
                "src/train_detector.py",
                "predict",
                "--checkpoint",
                str(directory / "checkpoint.pt"),
                "--out",
                str(directory / "val"),
                "--batch",
                str(MICROBATCH),
                "--allocation",
                str(ALLOCATION),
            ],
            directory / "val",
            val_forecast,
        )
    diagnostics = diagnose(directory)
    return directory, read(directory / "val/metrics.json"), diagnostics


def main():
    global MICROBATCH, ACCUMULATION
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch", type=int, choices=(1, 4, 8), default=4)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    MICROBATCH, ACCUMULATION = args.microbatch, 8 // args.microbatch
    suffix = f"-mb{MICROBATCH}" if MICROBATCH != 8 else ""
    sys.stdout.reconfigure(encoding="utf8")
    os.environ["SONAR_RESEARCH_ALLOCATION"] = str(ALLOCATION)
    os.environ["SONAR_ORIGINAL_BLOCK_OPERATION"] = "0"
    os.environ["SONAR_RESEARCH_PHASE"] = "development"
    if (BLOCK / "development_closed.json").exists():
        raise ValueError("Development already closed; do not restart")
    profile = read(args.profile or BLOCK / f"profile-image{suffix}/profile.json")
    assert (
        profile["checkpoint_reload_identical"]
        and profile["real_detector_optimizer_resume_matches_next_update"]
    )
    # Cold startup happens once; warm maximum plus25% and0.10h covers full update/decode work.
    training = max(profile["microbatch_seconds"][1:]) * 2000 / 3600 * 1.25 + 0.10
    validation = (
        max(
            read(ART / run / "val/resources.json")["charged_gpu_hours"]
            for run in (
                "det-published-r672-f010-s7",
                "det-finetune-r672-f010-s7",
                "det-b-finetune-r672-f010-s7",
            )
        )
        * 1.25
        + 0.05
    )
    save_json(
        BLOCK / f"forecast{suffix}.json",
        {
            "training_hours": training,
            "validation_hours": validation,
            "cold_start_seconds": profile["microbatch_seconds"][0],
            "remaining_development_hours": remaining(),
            "final_reserve_hours": 6,
            "microbatch": MICROBATCH,
            "accumulation": ACCUMULATION,
        },
    )
    reference = ART / f"det-finetune-r672-f010-s7{suffix}"
    if MICROBATCH != 8:
        if not (reference / "val/metrics.json").exists() and remaining() < 2 * (training + validation):
            raise RuntimeError("Complete primary matched reference/candidate pair cannot fit")
        reference, original, original_diag = fit_and_score(
            reference.name,
            "none",
            7,
            10,
            training,
            validation,
            reference=ART / "det-finetune-r672-f010-s7",
        )
    else:
        original = read(reference / "val/metrics.json")
        original_diag = read(ART / "kenai_diagnostics/det-finetune-r672-f010-s7.json")
    image, metrics, diagnostic = fit_and_score(
        f"det-image-finetune-r672-f010-s7{suffix}", "image", 7, 10, training, validation, reference=reference
    )
    screen = {
        "AP50_gain_pp": 100 * (metrics["AP50"] - original["AP50"]),
        "AP95_gain_pp": 100 * (metrics["AP50:95"] - original["AP50:95"]),
        "small_recall": subset_recall(diagnostic),
        "reference_small_recall": subset_recall(original_diag),
        "rule": "AP50>=+1pp, AP95>=reference, fixed below16 recall>reference; resources fit",
        "metrics_sha256": file_sha(image / "val/metrics.json"),
    }
    screen["passes"] = (
        screen["AP50_gain_pp"] >= 1
        and screen["AP95_gain_pp"] >= 0
        and screen["small_recall"] > screen["reference_small_recall"]
    )
    save_json(BLOCK / "image_screen.json", screen)
    print("IMAGE_SCREEN", json.dumps(screen), flush=True)
    # Forecast later conditional bundles from completed, matched measured jobs.
    charged_intervals = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    training = (
        max(
            sum(
                row.get("charged_gpu_hours", 0)
                for row in charged_intervals
                if row.get("name") == directory.name
            )
            for directory in (reference, image)
        )
        * 1.25
        + 0.05
    )
    validation = (
        max(read(directory / "val/resources.json")["charged_gpu_hours"] for directory in (reference, image))
        * 1.25
        + 0.05
    )
    save_json(
        BLOCK / "conditional_forecast.json",
        {
            "training_hours": training,
            "validation_hours": validation,
            "basis": "slowest completed matched primary fit/evaluation +25% and0.05h; all fit intervals including retries/resume counted",
            "remaining_development_hours": remaining(),
        },
    )
    roster = [reference, image]
    omissions = {}
    if screen["passes"]:
        if remaining() >= training + validation:
            capacity_profile = BLOCK / f"profile-capacity{suffix}"
            if not (capacity_profile / "profile.json").exists():
                execute(
                    [
                        "src/profile_batch.py",
                        "--finetune",
                        "--neck",
                        "capacity",
                        "--batch",
                        str(MICROBATCH),
                        "--accumulation",
                        str(ACCUMULATION),
                        "--size",
                        "672",
                        "--allocation",
                        str(ALLOCATION),
                        "--out",
                        str(capacity_profile),
                    ],
                    capacity_profile,
                    0.05,
                )
            assert read(capacity_profile / "profile.json")[
                "real_detector_optimizer_resume_matches_next_update"
            ]
            control, _, _ = fit_and_score(
                f"det-capacity-finetune-r672-f010-s7{suffix}",
                "capacity",
                7,
                10,
                training,
                validation,
                reference=reference,
            )
            roster.append(control)
        else:
            omissions["capacity"] = "NOT_RUN_BUDGET; no fine-scale mechanism attribution"
        paired = 2 * (training + validation)
        # Admit both additional paired seeds together, never choose the better completed seed.
        if remaining() >= 2 * paired:
            for seed in (13, 23):
                published, _, _ = fit_and_score(
                    f"det-finetune-r672-f010-s{seed}{suffix}", "none", seed, 10, training, validation
                )
                adapted, _, _ = fit_and_score(
                    f"det-image-finetune-r672-f010-s{seed}{suffix}",
                    "image",
                    seed,
                    10,
                    training,
                    validation,
                    reference=published,
                )
                roster.extend((published, adapted))
        else:
            omissions["replication"] = "NOT_RUN_BUDGET; strongest contrast remains seed7 pilot"
    else:
        omissions["capacity"] = "NOT_QUALIFIED_PREDECLARED_SCREEN"
        omissions["replication"] = "NOT_QUALIFIED_PREDECLARED_SCREEN"
    # A B10 microbatch1 bridge is necessary for a controlled1%/10% recipe curve.
    label_cells = 3 if MICROBATCH != 8 else 2
    pair_forecast = label_cells * (training + validation)
    if remaining() >= pair_forecast:
        if MICROBATCH != 8:
            b_ten, _, _ = fit_and_score(
                f"det-b-finetune-r672-f010-s7{suffix}",
                "none",
                7,
                10,
                training,
                validation,
                adapted=ART / "ssl-B-s7/encoder.pt",
                reference=reference,
            )
            roster.append(b_ten)
        published, _, _ = fit_and_score(
            f"det-finetune-r672-f001-s7{suffix}", "none", 7, 1, training, validation
        )
        adapted, _, _ = fit_and_score(
            f"det-b-finetune-r672-f001-s7{suffix}",
            "none",
            7,
            1,
            training,
            validation,
            adapted=ART / "ssl-B-s7/encoder.pt",
            reference=published,
        )
        roster.extend((published, adapted))
    else:
        omissions["one_percent"] = "NOT_RUN_BUDGET; complete matched label curve including numerical bridge"
    save_json(
        BLOCK / "development_closed.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "development closed",
            "roster": [str(path.relative_to(PROJECT)) for path in roster],
            "screen": screen,
            "omissions": omissions,
            "original_100_percent": "DEFERRED_BY_OWNER_PRIORITY_AMENDMENT",
            "remaining_development_hours": remaining(),
            "no_new_channel_inference": True,
        },
    )
    print("LOCALIZATION_DEVELOPMENT_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
