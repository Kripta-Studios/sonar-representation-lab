"""Freeze an explicit bounded roster after the finite Kenai development block."""

import argparse
import json
import re
import time
from pathlib import Path

import platform_compat  # noqa: F401
import torch

from acquire import ROOT, save_json
from budget import allocation_limits
from runtime import LEDGER, LOCK, code_identity, file_sha, used_hours


def validate_spec(spec):
    runs = spec["runs"]
    if not 1 <= len(runs) <= 12 or len(set(runs)) != len(runs):
        raise ValueError("Freeze requires 1-12 distinct detector checkpoints")
    if any(not re.fullmatch(r"det-[a-z0-9-]+", name) for name in runs):
        raise ValueError("Roster must name local detector artifact directories")
    if spec["selected_resolution"] not in (448, 672):
        raise ValueError("Unsupported resolution")
    if not spec.get("development_closed") or "omitted_cells" not in spec:
        raise ValueError("Close finite development and record omitted cells first")
    if not 0 < spec["forecast_final_gpu_hours"] <= spec["reserved_gpu_hours"]:
        raise ValueError("Final forecast exceeds its reservation")
    if spec["reserved_gpu_hours"] < 6:
        raise ValueError("At least six hours must remain reserved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, default=Path("artifacts/spatial_motion_allocation.json"))
    args = parser.parse_args()
    target = Path("artifacts/freeze.json")
    if target.exists() or LOCK.exists():
        raise RuntimeError("Preserve an existing freeze; close active GPU work before finalization")
    spec = json.loads(args.roster.read_text(encoding="utf8"))
    validate_spec(spec)
    allocation = json.loads(args.allocation.read_text())
    limits = allocation_limits(LEDGER, args.allocation)
    remaining = limits["block_ceiling_hours"] - used_hours()
    if (
        allocation["final_reserved_gpu_hours"] < spec["reserved_gpu_hours"]
        or remaining < spec["reserved_gpu_hours"]
    ):
        raise RuntimeError("The required final reserve is unavailable; do not expose Channel")
    models = []
    for run in spec["runs"]:
        directory = Path("artifacts") / run
        if not directory.resolve().is_relative_to(Path("artifacts").resolve()):
            raise ValueError("Checkpoint resolved outside this project's artifacts")
        checkpoint = directory / "checkpoint.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = json.loads((directory / "config.json").read_text())
        if state["config"] != config or state["step"] != config["steps"] or state["step"] != 2000:
            raise ValueError(f"Incomplete or inconsistent final checkpoint: {run}")
        if not all(torch.isfinite(value).all() for value in state["model"].values()):
            raise ValueError(f"Nonfinite checkpoint: {run}")
        metrics = json.loads((directory / "val/metrics.json").read_text())
        if metrics["images_evaluated"] != 30454:
            raise ValueError(f"Full Kenai validation is missing: {run}")
        checkpoint_sha = file_sha(checkpoint)
        prediction_identity = json.loads((directory / "val/prediction_identity.json").read_text())
        label_manifest = ROOT / "manifests" / f"train-{config['fraction']:03d}.json"
        if file_sha(label_manifest) != config["manifest_sha256"]:
            raise ValueError(f"Label manifest changed since training: {run}")
        if (
            prediction_identity["checkpoint_sha256"] != checkpoint_sha
            or prediction_identity["annotation_sha256"] != file_sha(ROOT / "manifests/val.json")
            or prediction_identity.get("detector_size", 448) != config["detector_size"]
        ):
            raise ValueError(f"Validation does not identify this checkpoint/input: {run}")
        models.append(
            {
                "run": run,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "config": config,
                "validation_metrics_sha256": file_sha(directory / "val/metrics.json"),
                "validation_prediction_identity_sha256": file_sha(directory / "val/prediction_identity.json"),
                "label_manifest": str(label_manifest),
                "ancestry_encoder_sha256": None
                if config["kind"] == "random"
                else config.get("adapted_sha256") or config["published_sha256"],
                "initialization": "same-architecture random"
                if config["kind"] == "random"
                else "adapted"
                if config.get("adapted")
                else "published",
            }
        )
        del state
    configs = {model["run"]: model["config"] for model in models}
    for comparison in spec.get("comparisons", []):
        candidate = [configs[name] for name in comparison["candidate"]]
        reference = [configs[name] for name in comparison["reference"]]
        required_seeds = {7, 13, 23} if comparison.get("replicated_claim") else {7}
        if {c["seed"] for c in candidate} != required_seeds or {
            r["seed"] for r in reference
        } != required_seeds:
            raise ValueError("All declared comparison seeds must be included, without best-seed substitution")
        for seed in required_seeds:
            c = next(row for row in candidate if row["seed"] == seed)
            r = next(row for row in reference if row["seed"] == seed)
            for key in [
                "fraction",
                "manifest_sha256",
                "steps",
                "batch",
                "accumulation",
                "detector_size",
                "frozen",
            ]:
                if c[key] != r[key]:
                    raise ValueError(f"Unmatched final comparison: {key}")
    kinds = [row["config"] for row in models]
    if not any(c["kind"] == "published" for c in kinds) or not any(
        c["kind"] == "finetune" and not c.get("adapted") for c in kinds
    ):
        raise ValueError("Published frozen and supervised references are required")
    if not any("ssl-A-" in (c.get("adapted") or "") for c in kinds):
        raise ValueError("An original A encoder reference is required")
    prior = "Initial metadata listing displayed several Channel annotation/clip headers; no author-executed Channel imagery or score before this freeze. Annotation bytes are hashed here without parsing entries."
    if Path("artifacts/channel_exposure.json").exists():
        prior = {
            "already_exposed": True,
            "existing_journal": json.loads(Path("artifacts/channel_exposure.json").read_text()),
        }
    save_json(
        target,
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "frozen",
            "block_id": allocation["id"],
            "models": models,
            "roster_spec": spec,
            "roster_spec_sha256": file_sha(args.roster),
            "allocation_sha256": file_sha(args.allocation),
            "code_identity": code_identity(),
            "protocol_sha256": file_sha("PROTOCOL.md"),
            "label_policy": "fixed nested complete Kenai TRAIN clips; no target-site adaptation",
            "preprocessing": "grayscale repeated into 3 channels; aspect-preserving top-left letterbox at each checkpoint's fixed size; rounded inverse axes; ImageNet normalization",
            "cache_policy": "existing validated 448 transform; no full 672 cache",
            "score_threshold_saved": 0.001,
            "operating_score": 0.5,
            "nms": 0.5,
            "max_detections": 100,
            "channel_annotations_sha256": file_sha(
                ROOT / "metadata/coco_annotations_v1.1/kenai-channel.json"
            ),
            "gpu_hours_charged_before_freeze": used_hours(),
            "prior_exposure": prior,
        },
    )
    print(
        json.dumps(
            {"models_frozen": len(models), "file": str(target), "reserved_hours": spec["reserved_gpu_hours"]}
        )
    )


if __name__ == "__main__":
    main()
