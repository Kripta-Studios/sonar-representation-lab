"""Kenai-only operating-point and error analysis from saved detections; no GPU."""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from acquire import ROOT, save_json  # noqa: E402
from evaluate import iou_xywh  # noqa: E402

THRESHOLDS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
BINS = ("<4px", "4-8px", "8-16px", ">=16px")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def size_bin(annotation, image):
    side = min(annotation["bbox"][2:]) * 448 / max(image["width"], image["height"])
    return BINS[0] if side < 4 else BINS[1] if side < 8 else BINS[2] if side < 16 else BINS[3]


def match(annotations, predictions, threshold=0.5, iou=0.5):
    """Use the same greedy score order/category/IoU rule as the main operating metric."""
    found, decisions = set(), []
    for prediction in sorted(predictions, key=lambda p: -p["score"]):
        if prediction["score"] < threshold:
            continue
        overlaps = [
            (iou_xywh(prediction["bbox"], a["bbox"]), index)
            for index, a in enumerate(annotations)
            if prediction["category_id"] == a["category_id"]
        ]
        overlap, index = max((pair for pair in overlaps if pair[1] not in found), default=(0, -1))
        if overlap >= iou:
            found.add(index)
            decisions.append("true_positive")
        else:
            best = max((pair[0] for pair in overlaps), default=0)
            decisions.append("duplicate" if best >= iou else "localization" if best >= 0.1 else "background")
    return found, decisions


def analyze(gt, predictions, thresholds=THRESHOLDS):
    images = {image["id"]: image for image in gt["images"]}
    cats = {c["id"] for c in gt["categories"]}
    annotations, by_image = defaultdict(list), defaultdict(list)
    for annotation in gt["annotations"]:
        if annotation.get("iscrowd", 0):
            raise ValueError("These diagnostics require the verified non-crowd CFC annotations")
        annotations[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        if prediction["image_id"] not in images or prediction["category_id"] not in cats:
            raise ValueError("Unknown image/category identity")
        if prediction["score"] >= min(thresholds):
            by_image[prediction["image_id"]].append(prediction)
    results, examples = [], {}
    for threshold in thresholds:
        buckets = {name: {"objects": 0, "matched": 0} for name in BINS}
        tp = fp = negative_fp = negative_hit = negative = 0
        fp_types = dict.fromkeys(("duplicate", "localization", "background"), 0)
        clips = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "frames": 0})
        for image_id in sorted(images):
            image, anns = images[image_id], annotations[image_id]
            found, decisions = match(anns, by_image[image_id], threshold)
            image_fp = len(decisions) - len(found)
            tp += len(found)
            fp += image_fp
            if not anns:
                negative += 1
                negative_fp += image_fp
                negative_hit += image_fp > 0
            clip = Path(image["file_name"]).stem.rsplit("_", 1)[0]
            for key, count in [
                ("tp", len(found)),
                ("fp", image_fp),
                ("fn", len(anns) - len(found)),
                ("frames", 1),
            ]:
                clips[clip][key] += count
            for decision in decisions:
                if decision in fp_types:
                    fp_types[decision] += 1
            for index, annotation in enumerate(anns):
                name = size_bin(annotation, image)
                buckets[name]["objects"] += 1
                buckets[name]["matched"] += index in found
                if threshold == 0.5 and index not in found:
                    overlap = max(
                        (
                            iou_xywh(p["bbox"], annotation["bbox"])
                            for p in by_image[image_id]
                            if p["score"] >= threshold
                        ),
                        default=0,
                    )
                    kind = "poor_localization" if 0.1 <= overlap < 0.5 else "missed"
                    key = name + ":" + kind
                    # Select the first image/annotation ID, never the most dramatic visual.
                    candidate = {
                        "image_id": image_id,
                        "annotation_id": annotation["id"],
                        "file_name": image["file_name"],
                        "bbox": annotation["bbox"],
                        "best_iou_at_operating_score": overlap,
                        "bin_448": name,
                        "kind": kind,
                    }
                    if key not in examples or (image_id, annotation["id"]) < (
                        examples[key]["image_id"],
                        examples[key]["annotation_id"],
                    ):
                        examples[key] = candidate
        total = len(gt["annotations"])
        for bucket in buckets.values():
            bucket["recall"] = bucket["matched"] / max(bucket["objects"], 1)
            bucket["missed"] = bucket["objects"] - bucket["matched"]
        results.append(
            {
                "score_threshold": threshold,
                "iou_threshold": 0.5,
                "tp": tp,
                "fp": fp,
                "fn": total - tp,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(total, 1),
                "negative_frames": negative,
                "negative_false_positives": negative_fp,
                "false_positives_per_negative_frame": negative_fp / max(negative, 1),
                "fraction_negative_frames_with_fp": negative_hit / max(negative, 1),
                "false_positive_types": fp_types,
                "size_bins": buckets,
                "clips": dict(clips),
            }
        )
    return {"operating_points": results, "error_examples": list(examples.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=Path("artifacts/kenai_diagnostics"))
    args = parser.parse_args()
    manifest = args.root / "manifests/val.json"
    gt = json.loads(manifest.read_text())
    summary_path = args.out / "summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else []
    for run in args.runs:
        if (args.out / f"{run.name}.json").exists():
            raise ValueError(f"Diagnostic output already exists: {run.name}; preserve it")
        saved = run / "val/metrics.json"
        raw = run / "val/predictions.json"
        if not saved.exists():
            raise ValueError(f"Incomplete Kenai evaluation: {run}")
        started = time.time()
        with raw.open(encoding="utf8") as stream:
            predictions = json.load(stream)
        result = analyze(gt, predictions)
        del predictions
        original = json.loads(saved.read_text())
        primary = next(r for r in result["operating_points"] if r["score_threshold"] == 0.5)
        for key in ["tp", "fp", "fn", "negative_false_positives", "negative_frames"]:
            if primary[key] != original[key]:
                raise ValueError(f"Operating-point reproduction failed: {run} {key}")
        result.update(
            {
                "run": run.name,
                "split": "Kenai validation only",
                "cpu_only": True,
                "size_definition": "fixed pre-rounding 448 scale at all detector resolutions",
                "thresholds_predeclared": list(THRESHOLDS),
                "threshold_selected": None,
                "selection": "first image ID then annotation ID per size/error bin",
                "author_status": "author evaluated; no independent review",
                "predictions_sha256": sha(raw),
                "annotations_sha256": sha(manifest),
                "analysis_sha256": sha(__file__),
                "cpu_wall_seconds": time.time() - started,
            }
        )
        save_json(args.out / (run.name + ".json"), result)
        summaries.append(
            {
                "run": run.name,
                "cpu_wall_seconds": result["cpu_wall_seconds"],
                "AP50": original["AP50"],
                "recall": primary["recall"],
                "false_positive_types": primary["false_positive_types"],
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
    save_json(args.out / "summary.json", summaries)


if __name__ == "__main__":
    main()
