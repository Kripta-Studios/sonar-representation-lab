"""Preserve historical metrics and quantify positive-ID corrections from saved matching."""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analysis"))
sys.path.insert(0, str(PROJECT / "src"))
from acquire import ROOT, save_json  # noqa: E402
from clip_bootstrap import compact_rows, file_sha, pooled_ap, stream_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/localization-block/evaluator_correction_audit.json")
    )
    parser.add_argument("--runs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve the completed correction audit")
    started = time.monotonic()
    truth = json.loads((ROOT / "manifests/val.json").read_text())
    clips = sorted({Path(row["file_name"]).stem.rsplit("_", 1)[0] for row in truth["images"]})
    result = []
    for directory in sorted(args.runs or (PROJECT / "artifacts").glob("det-*")):
        metrics_path = directory / "val/metrics.json"
        cached = directory / "val/coco_eval_images.json.gz"
        if not cached.exists():
            # The earliest run predates gzip serialization. Its saved CPU replay
            # was verified bitwise identical to the original full evaluator.
            replay = directory / "val/saved_prediction_rescore_verification.json"
            if replay.exists():
                proof = json.loads(replay.read_text())
                assert proof["all_metric_fields_match"]
                assert proof["all_precision_recall_score_arrays_bitwise_equal"]
                cached = directory / "val/rescore_verification/coco_eval_images.json.gz"
        if not metrics_path.exists() or not cached.exists():
            continue
        old = json.loads(metrics_path.read_text())
        if old.get("evaluator_version") == "cfc-coco-positive-annotation-ids-v2":
            continue
        compact = compact_rows(stream_rows(cached), truth["images"])
        legacy = pooled_ap({**compact, "matched": compact["legacy_matched"]}, clips, original_order=True)
        corrected = pooled_ap(compact, clips, original_order=True)
        assert abs(legacy[0] - old["AP50"]) < 1e-12
        assert abs(legacy.mean() - old["AP50:95"]) < 1e-12
        row = {
            "run": directory.name,
            "source_metrics_sha256": file_sha(metrics_path),
            "source_per_image_coco_sha256": file_sha(cached),
            "legacy_AP50": old["AP50"],
            "legacy_AP95": old["AP50:95"],
            "positive_id_AP50": float(corrected[0]),
            "positive_id_AP95": float(corrected.mean()),
            "positive_id_AP75": float(corrected[5]),
            "delta_AP50_pp": float(100 * (corrected[0] - legacy[0])),
            "delta_AP95_pp": float(100 * (corrected.mean() - legacy.mean())),
            "zero_annotation_image_ids": compact["legacy_zero_annotation_image_ids"],
        }
        result.append(row)
        print(json.dumps(row), flush=True)
        del compact
    save_json(
        args.out,
        {
            "evaluator": "cfc-coco-positive-annotation-ids-v2",
            "derived_kenai_metrics": result,
            "historical_artifacts_overwritten": False,
            "channel_accessed": False,
            "gpu_used": False,
            "cpu_wall_seconds": time.monotonic() - started,
            "method": "Recovered matched ID0 through gtMatches/dtIds; known-answer exact agreement with positive-ID full scorer, original image-ID tie ordering retained",
        },
    )


if __name__ == "__main__":
    main()
