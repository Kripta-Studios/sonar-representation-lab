"""Summarize only executed saved detection results and draw learning curves."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from acquire import ROOT, save_json
from evaluate import iou_xywh


def size_errors(directory):
    gt = json.loads((ROOT / "manifests/val.json").read_text())
    predictions = json.loads((directory / "val/predictions.json").read_text())
    by_image = defaultdict(list)
    for p in predictions:
        if p["score"] >= 0.5:
            by_image[p["image_id"]].append(p)
    annotations = defaultdict(list)
    for a in gt["annotations"]:
        annotations[a["image_id"]].append(a)
    buckets = {
        name: {"objects": 0, "matched": 0, "missed": 0} for name in ["<4px", "4-8px", "8-16px", ">=16px"]
    }
    for image in gt["images"]:
        anns = annotations[image["id"]]
        found = set()
        for p in sorted(by_image[image["id"]], key=lambda p: -p["score"]):
            pairs = [(iou_xywh(p["bbox"], a["bbox"]), j) for j, a in enumerate(anns) if j not in found]
            overlap, index = max(pairs, default=(0, -1))
            if overlap >= 0.5:
                found.add(index)
        scale = 448 / max(image["width"], image["height"])
        for j, a in enumerate(anns):
            side = min(a["bbox"][2:]) * scale
            name = "<4px" if side < 4 else "4-8px" if side < 8 else "8-16px" if side < 16 else ">=16px"
            buckets[name]["objects"] += 1
            buckets[name]["matched" if j in found else "missed"] += 1
    for value in buckets.values():
        value["recall"] = value["matched"] / max(1, value["objects"])
    save_json(
        directory / "val/errors_by_resized_short_side.json",
        {"score_threshold": 0.5, "iou": 0.5, "bins_use_pre_rounding_resize_scale": True, "buckets": buckets},
    )


def curves(directory):
    path = directory / "curve.jsonl"
    if not path.exists() or (directory / "learning_curves.png").exists():
        return
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    segments = [[]]
    for row in rows:
        if segments[-1] and row["step"] <= segments[-1][-1]["step"]:
            segments.append([])
        segments[-1].append(row)
    # Preserve every logged attempt; never connect backwards across checkpoint replay.
    for index, segment in enumerate(segments):
        axes[0].plot(
            [r["step"] for r in segment],
            [r.get("total_loss", r.get("loss", sum(r.get("losses", {}).values()))) for r in segment],
            label=f"Logged segment {index + 1}",
        )
    if len(segments) > 1:
        axes[0].legend(fontsize=7)
    axes[0].set(xlabel="Successful optimizer updates", ylabel="Training loss", title=directory.name)
    if "teacher_entropy" in rows[0]:
        for index, segment in enumerate(segments):
            x = [r["step"] for r in segment]
            axes[1].plot(x, [r["teacher_entropy"] for r in segment], label=f"Teacher / segment {index + 1}")
            axes[1].plot(
                x,
                [r["batch_marginal_entropy"] for r in segment],
                label=f"Marginal / segment {index + 1}",
            )
        axes[1].legend()
        axes[1].set(xlabel="Successful optimizer updates", ylabel="Entropy (nats)")
    else:
        for index, segment in enumerate(segments):
            x = [r["step"] for r in segment]
            for key in rows[0]["losses"]:
                axes[1].plot(x, [r["losses"][key] for r in segment], label=f"{key} / {index + 1}")
        axes[1].legend(fontsize=7)
        axes[1].set(xlabel="Successful optimizer updates", ylabel="Loss component")
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)


def recipe(config):
    if not config.get("adapted"):
        return "random" if config["kind"] == "random" else "published"
    parent = Path(config["adapted"]).parent / "config.json"
    metadata = json.loads(parent.read_text())
    return metadata.get("adaptation_configuration", "A")


def group_key(row):
    return (
        row["recipe"],
        row["kind"],
        row["detector_size"],
        row["fraction"],
        row["split"],
        row.get("neck", "none"),
        row.get("microbatch", 8),
        row.get("accumulation", 1),
        row.get("evaluator_version", "legacy-coco-v1"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("artifacts/continuation-summary"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for directory in sorted(Path("artifacts").glob("det-*")):
        config = json.loads((directory / "config.json").read_text())
        curves(directory)
        for split in ["val", "channel"]:
            path = directory / split / "metrics.json"
            if path.exists():
                metric = json.loads(path.read_text())
                results.append(
                    {
                        "run": directory.name,
                        "kind": config["kind"],
                        "recipe": recipe(config),
                        "detector_size": config.get("detector_size", 448),
                        "seed": config["seed"],
                        "fraction": config["fraction"],
                        "neck": config.get("neck", "none"),
                        "microbatch": config["batch"],
                        "accumulation": config["accumulation"],
                        "split": split,
                        "evaluator_version": metric.get("evaluator_version", "legacy-coco-v1"),
                        **{
                            k: metric[k]
                            for k in [
                                "AP50",
                                "AP50:95",
                                "precision",
                                "recall",
                                "false_positives_per_negative_frame",
                                "fraction_negative_frames_with_fp",
                            ]
                        },
                    }
                )
        if (
            args.errors
            and config["seed"] == 7
            and config["fraction"] == 10
            and (directory / "val/predictions.json").exists()
        ):
            size_errors(directory)
    for directory in Path("artifacts").glob("ssl-*"):
        curves(directory)
    if not results:
        raise RuntimeError("No executed saved detector evaluation exists")
    with (args.out / "results.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    groups = defaultdict(list)
    for row in results:
        groups[group_key(row)].append(row)
    summaries = []
    for (
        training_recipe,
        kind,
        size,
        fraction,
        split,
        neck,
        microbatch,
        accumulation,
        evaluator_version,
    ), rows in sorted(groups.items()):
        if len({row["seed"] for row in rows}) != len(rows):
            raise ValueError("Duplicate seed in a summary group; preserve separate experiments")
        summaries.append(
            {
                "kind": kind,
                "recipe": training_recipe,
                "detector_size": size,
                "fraction": fraction,
                "split": split,
                "neck": neck,
                "microbatch": microbatch,
                "accumulation": accumulation,
                "evaluator_version": evaluator_version,
                "seeds": [r["seed"] for r in rows],
                "n": len(rows),
                "AP50_mean": statistics.mean(r["AP50"] for r in rows),
                "AP50_sample_std": statistics.stdev(r["AP50"] for r in rows) if len(rows) > 1 else None,
                "AP50:95_mean": statistics.mean(r["AP50:95"] for r in rows),
                "AP50:95_sample_std": statistics.stdev(r["AP50:95"] for r in rows) if len(rows) > 1 else None,
                "precision_mean": statistics.mean(r["precision"] for r in rows),
                "recall_mean": statistics.mean(r["recall"] for r in rows),
                "false_positives_per_negative_frame_mean": statistics.mean(
                    r["false_positives_per_negative_frame"] for r in rows
                ),
            }
        )
    save_json(args.out / "results_summary.json", summaries)
    print(json.dumps({"per_seed": results, "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
