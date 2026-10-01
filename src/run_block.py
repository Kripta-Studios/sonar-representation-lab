"""Plain serial execution of the declared experiment table, with logs and budget checks."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import platform_compat  # noqa: F401
from acquire import ROOT, save_json


PROJECT = Path(__file__).resolve().parents[1]


def hours():
    path = PROJECT / "artifacts/resource_ledger.jsonl"
    if not path.exists():
        return 0.0
    return sum(json.loads(s).get("charged_gpu_hours", 0) for s in path.read_text().splitlines())


def extension_fits():
    """Admission based only on measured runtime and public frame counts, never AP."""
    directories = list((PROJECT / "artifacts").glob("det-*"))
    train_hours = [
        json.loads((d / "resources.json").read_text())["charged_gpu_hours"]
        for d in directories
        if (d / "resources.json").exists() and ("published" in d.name or "adapted" in d.name)
    ]
    val_hours = [
        json.loads((d / "val/resources.json").read_text())["charged_gpu_hours"]
        for d in directories
        if (d / "val/resources.json").exists()
    ]
    if not train_hours or not val_hours:
        return False, {"reason": "missing measured runtime"}
    channel_frames = sum(
        bool(s.strip())
        for s in (ROOT / "metadata/file_lists_v1.1/kenai-channel.txt").read_text().splitlines()
    )
    val_frames = len(json.loads((ROOT / "manifests/val.json").read_text())["images"])
    heldout = (len(directories) + 2) * channel_frames / val_frames * max(val_hours) * 1.5 + 0.1
    pair = 2 * (max(train_hours) + max(val_hours)) * 1.25
    forecast = hours() + pair + heldout
    return forecast <= 23.8 and hours() < 20, {
        "charged_hours": hours(),
        "pair_hours_upper_estimate": pair,
        "heldout_hours_reserved": heldout,
        "projected_block_hours": forecast,
        "median_validation_hours": statistics.median(val_hours),
        "channel_frames": channel_frames,
    }


def call(script, arguments, log):
    command = [sys.executable, "-u", str(PROJECT / "src" / script), *map(str, arguments)]
    print("EXECUTE", subprocess.list2cmdline(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf8") as stream:
        result = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def ensure_dataset():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    if "kenai.tar" not in provenance:
        raise RuntimeError("Kenai archive integrity verification must finish before research training")
    full = json.loads((ROOT / "manifests/train-100.json").read_text())
    val = json.loads((ROOT / "manifests/val.json").read_text())
    missing = [
        i["file_name"]
        for i in full["images"] + val["images"]
        if not (ROOT / "images/kenai" / i["file_name"]).exists()
    ]
    if missing:
        raise RuntimeError(f"Kenai extraction incomplete: {len(missing)} missing frames")


def ssl(seed):
    out = PROJECT / f"artifacts/ssl-A-s{seed}"
    if not (out / "encoder.pt").exists():
        arguments = ["--seed", seed, "--out", out]
        if (out / "checkpoint.pt").exists():
            arguments += ["--resume", out / "checkpoint.pt"]
        call("adapt.py", arguments, out / "console.log")
    return out / "encoder.pt"


def detector_run(kind, seed, fraction=10, adapted=None):
    out = PROJECT / f"artifacts/det-{kind}-f{fraction:03d}-s{seed}"
    if (out / "val/metrics.json").exists():
        return
    arguments = [
        "train",
        "--kind",
        kind,
        "--seed",
        seed,
        "--fraction",
        fraction,
        "--batch",
        8,
        "--accumulation",
        1,
        "--steps",
        2000,
        "--out",
        out,
    ]
    if adapted:
        arguments += ["--adapted", adapted]
    if (out / "checkpoint.pt").exists():
        arguments += ["--resume", out / "checkpoint.pt"]
    call("train_detector.py", arguments, out / "console.log")
    call(
        "train_detector.py",
        ["predict", "--checkpoint", out / "checkpoint.pt", "--batch", 8, "--out", out / "val"],
        out / "val/console.log",
    )
    result = json.loads((out / "val/metrics.json").read_text())
    print(
        "RESULT",
        out.name,
        json.dumps({k: result[k] for k in ["AP50", "AP50:95", "precision", "recall"]}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["milestone", "replicate", "fractions"], required=True)
    args = parser.parse_args()
    ensure_dataset()
    save_json(
        PROJECT / "artifacts/execution_plan.json",
        {
            "protocol": "PROTOCOL.md",
            "seeds": [7, 13, 23],
            "ssl_configurations": ["A"],
            "detector_steps": 2000,
            "detector_batch": 8,
            "detector_accumulation": 1,
            "selection": "final step, no best seed",
            "initial_gpu_hour_cap": 24,
            "overall_initial_remaining_hours": 100,
        },
    )
    if args.stage == "milestone":
        detector_run("published", 7)
        adapted = ssl(7)
        detector_run("adapted", 7, adapted=adapted)
    elif args.stage == "replicate":
        for seed in [7, 13, 23]:
            if hours() > 20:
                raise RuntimeError("Reserve remaining hours for fixed cross-location evaluation")
            detector_run("published", seed)
            adapted = ssl(seed)
            detector_run("adapted", seed, adapted=adapted)
        # Prioritize the three-seed primary contrast; diagnostics use a fixed seed, never the best seed.
        detector_run("finetune", 7)
        detector_run("random", 7)
    else:
        for seed in [7, 13, 23]:
            for fraction in [1, 100]:
                # Extensions are conditional on measured budget; preserve remaining evaluation time.
                fits, forecast = extension_fits()
                if not fits:
                    save_json(
                        PROJECT / "artifacts/fraction_budget_stop.json",
                        {
                            "next_fraction": fraction,
                            "next_seed": seed,
                            "reason": "preserve initial allocation for fixed held-out evaluation",
                            "runtime_forecast": forecast,
                        },
                    )
                    print(
                        "FRACTION_EXTENSION_OMITTED_BUDGET", fraction, seed, json.dumps(forecast), flush=True
                    )
                    return
                detector_run("published", seed, fraction)
                detector_run("adapted", seed, fraction, ssl(seed))


if __name__ == "__main__":
    main()
