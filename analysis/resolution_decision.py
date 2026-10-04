"""Apply the single predeclared Kenai resolution decision; never inspect Channel."""

import hashlib
import json
from pathlib import Path


def criteria(reference, candidate, reference_small, candidate_small, resources_fit):
    return {
        "published_AP50_gain_at_least_one_percentage_point": candidate["AP50"] - reference["AP50"]
        >= 0.01 - 1e-12,
        "published_AP50_95_not_lower": candidate["AP50:95"] >= reference["AP50:95"] - 1e-12,
        "fixed_below_16px_operating_recall_improves": candidate_small > reference_small,
        "runtime_and_memory_fit": resources_fit,
    }


def small_recall(path):
    result = json.loads(path.read_text(encoding="utf8"))
    row = next(item for item in result["operating_points"] if item["score_threshold"] == 0.5)
    bins = row["size_bins"]
    names = ["<4px", "4-8px", "8-16px"]
    total = sum(bins[name]["objects"] for name in names)
    assert total == 16613
    return sum(bins[name]["matched"] for name in names) / total


def main():
    output = Path("artifacts/resolution_decision.json")
    if output.exists():
        raise ValueError("Resolution decision is already recorded; do not overwrite")
    paths = {
        "published448": Path("artifacts/det-published-f010-s7"),
        "A448": Path("artifacts/det-adapted-f010-s7"),
        "published672": Path("artifacts/det-published-r672-f010-s7"),
        "A672": Path("artifacts/det-adapted-r672-f010-s7"),
    }
    scores = {key: json.loads((path / "val/metrics.json").read_text()) for key, path in paths.items()}
    assert all(value["images_evaluated"] == 30454 for value in scores.values())
    recall = {
        key: small_recall(Path("artifacts/kenai_diagnostics") / (path.name + ".json"))
        for key, path in paths.items()
    }
    resources = [
        json.loads((paths[key] / suffix).read_text())
        for key in ["published672", "A672"]
        for suffix in ["resources.json", "val/resources.json"]
    ]
    allocation = json.loads(Path("artifacts/spatial_motion_allocation.json").read_text())
    used = sum(
        json.loads(line).get("charged_gpu_hours", 0)
        for line in Path("artifacts/resource_ledger.jsonl").read_text().splitlines()
    )
    remaining = (
        allocation["baseline_gpu_hours"]
        + allocation["budget_gpu_hours"]
        - allocation["final_reserved_gpu_hours"]
        - used
    )
    # The owner's initial spatial-B/comparator allowance is the conservative admission floor.
    fits = remaining >= 8 and all(
        r["status"] == "completed"
        and r["physical_device_used_peak_bytes"] < 10 * 1024**3
        and r["owned_rss_peak_bytes"] < 22 * 1024**3
        for r in resources
    )
    tests = criteria(
        scores["published448"], scores["published672"], recall["published448"], recall["published672"], fits
    )
    record = {
        "selected_detector_size": 672 if all(tests.values()) else 448,
        "criteria": tests,
        "scores": {
            key: {
                name: value[name]
                for name in ["AP50", "AP50:95", "AP75", "precision", "recall", "negative_false_positives"]
            }
            for key, value in scores.items()
        },
        "below_16px_recall_fixed_448_bins": recall,
        "remaining_development_hours": remaining,
        "mandatory_spatial_allowance_hours": 8,
        "resolution_pair_gpu_process_hours": sum(r["charged_gpu_hours"] for r in resources),
        "final_reserved_hours": allocation["final_reserved_gpu_hours"],
        "metric_sha256": {
            key: hashlib.sha256((path / "val/metrics.json").read_bytes()).hexdigest()
            for key, path in paths.items()
        },
        "status": "post-hoc Kenai-informed development choice under prospectively declared rule; no Channel scores used",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
