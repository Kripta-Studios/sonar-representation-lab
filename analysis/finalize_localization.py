"""Freeze this finite Kenai block and verify saved outputs without model selection."""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "analysis"))
from acquire import ROOT, save_json  # noqa: E402
from budget import allocation_limits  # noqa: E402
from runtime import LEDGER, code_identity, file_sha  # noqa: E402

BLOCK = PROJECT / "artifacts/localization-block"
FREEZE = BLOCK / "final_freeze.json"
ALLOCATION = BLOCK / "allocation.json"
FILES = (
    "checkpoint.pt",
    "config.json",
    "artifact_verification.json",
    "finetuning_verification.json",
    "source_snapshot.zip",
    "val/predictions.json",
    "val/metrics.json",
    "val/prediction_identity.json",
    "val/coco_output.txt",
    "val/coco_arrays.npz",
    "val/coco_eval_images.json.gz",
    "val/evaluator_params.json",
    "val/evaluator_identity.json",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def freeze():
    if FREEZE.exists():
        raise ValueError("Preserve existing freeze")
    if (PROJECT / "artifacts/gpu_process.lock").exists():
        raise RuntimeError("Recover live GPU owner before final freeze")
    closed = read(BLOCK / "development_closed.json")
    assert closed["status"] == "development closed" and closed["no_new_channel_inference"]
    models = []
    for name in closed["roster"]:
        directory = PROJECT / name
        config = read(directory / "config.json")
        verification = read(directory / "artifact_verification.json")
        assert read(directory / "finetuning_verification.json")["changed_encoder_tensor_count"] > 0
        metrics = read(directory / "val/metrics.json")
        assert read(directory / "resources.json")["status"] == "completed"
        assert read(directory / "val/resources.json")["status"] == "completed"
        assert verification["successful_updates"] == 2000 and verification["presentations"] == 16000
        assert config["batch"] == 4 and config["accumulation"] == 2 and config["detector_size"] == 672
        assert metrics["evaluator_version"] == "cfc-coco-positive-annotation-ids-v2"
        assert metrics["images_evaluated"] == 30454
        files = {name: file_sha(directory / name) for name in FILES}
        if (directory / "training_source_snapshot.zip").exists():
            files["training_source_snapshot.zip"] = file_sha(directory / "training_source_snapshot.zip")
        models.append(
            {
                "run": directory.name,
                "path": name,
                "files": files,
                "seed": config["seed"],
                "fraction": config["fraction"],
                "neck": config["neck"],
                "adapted_sha256": config["adapted_sha256"],
                "manifest_sha256": config["manifest_sha256"],
                "training_code_identity": config["code_identity"],
            }
        )
    assert 1 <= len(models) <= 12
    save_json(
        FREEZE,
        {
            "status": "frozen",
            "block_id": "localization-20261003",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": models,
            "code_identity": code_identity(),
            "analysis_identity": {p.name: file_sha(p) for p in sorted((PROJECT / "analysis").glob("*.py"))},
            "protocol_sha256": file_sha(PROJECT / "PROTOCOL.md"),
            "allocation_sha256": file_sha(ALLOCATION),
            "architecture_costs_sha256": file_sha(BLOCK / "architecture_costs.json"),
            "ledger_sha256_at_freeze": file_sha(LEDGER),
            "closure_sha256": file_sha(BLOCK / "development_closed.json"),
            "validation_manifest_sha256": file_sha(ROOT / "manifests/val.json"),
            "source_only": True,
            "no_new_channel_inference": True,
            "channel_status": "Previously exposed; historical results/evaluator limitations retained",
            "budget": allocation_limits(LEDGER, ALLOCATION),
            "replay_roster_rule": "Full CPU scorer replay of primary published reference; all roster files verified",
            "latency_roster_rule": "Primary published/image pair; fixed eight TRAIN frames, batch1/4",
            "proposal_roster_rule": "Primary published/image pair, same pre-recorded512 Kenai frames; descriptive diagnosis only",
            "replay_is_not_model_selection": True,
        },
    )
    print("FROZEN", file_sha(FREEZE), flush=True)


def verify():
    if (BLOCK / "final_verification.json").exists():
        raise ValueError("Preserve completed verification")
    frozen = read(FREEZE)
    assert frozen["code_identity"] == code_identity()
    assert frozen["analysis_identity"] == {
        p.name: file_sha(p) for p in sorted((PROJECT / "analysis").glob("*.py"))
    }
    assert frozen["protocol_sha256"] == file_sha(PROJECT / "PROTOCOL.md")
    assert frozen["allocation_sha256"] == file_sha(ALLOCATION)
    assert frozen["architecture_costs_sha256"] == file_sha(BLOCK / "architecture_costs.json")
    assert frozen["closure_sha256"] == file_sha(BLOCK / "development_closed.json")
    assert frozen["validation_manifest_sha256"] == file_sha(ROOT / "manifests/val.json")
    import torch
    import numpy as np
    import psutil
    from clip_bootstrap import compact_rows, pooled_ap, stream_rows
    from evaluate import score_predictions

    started = time.monotonic()
    gt = read(ROOT / "manifests/val.json")
    clips = sorted({Path(row["file_name"]).stem.rsplit("_", 1)[0] for row in gt["images"]})
    exposures, results = {}, []
    for item in frozen["models"]:
        directory = PROJECT / item["path"]
        for name, expected in item["files"].items():
            assert file_sha(directory / name) == expected, (directory, name)
        state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert state["step"] == 2000 and sum(state["exposures"].values()) == 16000
        assert all(torch.isfinite(value).all() for value in state["model"].values())
        assert all(int(item["step"]) == 2000 for item in state["optimizer"]["state"].values())
        training_source = directory / "training_source_snapshot.zip"
        if not training_source.exists():
            training_source = directory / "source_snapshot.zip"
        with zipfile.ZipFile(training_source) as archive:
            for source, expected in state["config"]["code_identity"].items():
                assert hashlib.sha256(archive.read(source)).hexdigest() == expected, (directory, source)
            assert (
                hashlib.sha256(archive.read("PROTOCOL.md")).hexdigest() == state["config"]["protocol_sha256"]
            )
        key = (item["seed"], item["fraction"])
        if key in exposures:
            assert state["exposures"] == exposures[key], directory
        else:
            exposures[key] = state["exposures"]
        del state
        compact = compact_rows(stream_rows(directory / "val/coco_eval_images.json.gz"), gt["images"])
        ap = pooled_ap(compact, clips, original_order=True)
        metric = read(directory / "val/metrics.json")
        assert abs(ap[0] - metric["AP50"]) < 1e-12
        assert abs(ap.mean() - metric["AP50:95"]) < 1e-12
        assert abs(ap[5] - metric["AP75"]) < 1e-12
        results.append(
            {
                "run": item["run"],
                "all_hashes_match": True,
                "checkpoint_finite": True,
                "presentations": 16000,
                "cached_full_AP_reproduced": True,
            }
        )
        del compact
        print("VERIFIED", item["run"], flush=True)
    # One complete canonical replay, not a second model inference or a score search.
    directory = PROJECT / frozen["models"][0]["path"]
    target = BLOCK / "final_replay"
    if (target / "metrics.json").exists():
        raise ValueError("Preserve existing replay and recover its actual state")
    predictions = read(directory / "val/predictions.json")
    replay = score_predictions(gt, predictions, target)
    original = read(directory / "val/metrics.json")
    assert replay == {key: original[key] for key in replay}
    del predictions
    with np.load(directory / "val/coco_arrays.npz") as first, np.load(target / "coco_arrays.npz") as second:
        assert first.files == second.files
        assert all(np.array_equal(first[key], second[key]) for key in first.files)
    assert file_sha(target / "evaluator_params.json") == file_sha(directory / "val/evaluator_params.json")
    assert file_sha(target / "evaluator_identity.json") == file_sha(directory / "val/evaluator_identity.json")
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss))
    assert peak < 22 * 1024**3, "Owned CPU evaluation RAM cap exceeded"
    save_json(
        BLOCK / "final_verification.json",
        {
            "status": "author-executed verification; not independent review",
            "freeze_sha256": file_sha(FREEZE),
            "models": results,
            "exact_exposures_within_seed_fraction": True,
            "canonical_full_replay": directory.name,
            "all_metric_fields_match": True,
            "precision_recall_scores_bitwise_equal": True,
            "evaluator_parameters_and_mapping_identical": True,
            "gpu_used": False,
            "owned_native_peak_bytes": peak,
            "cpu_wall_seconds": time.monotonic() - started,
            "channel_accessed": False,
            "budget": allocation_limits(LEDGER, ALLOCATION, "final", FREEZE),
        },
    )
    print("FINAL_VERIFICATION_COMPLETED", flush=True)


if __name__ == "__main__":
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "verify"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else verify()
