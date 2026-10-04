"""Plot the fixed roster's completed source/held-out AP, without model selection."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/final_comparison"))
    args = parser.parse_args()
    if args.out.with_suffix(".json").exists():
        raise ValueError("Preserve an existing completed figure")
    freeze_path = Path("artifacts/freeze.json")
    freeze = json.loads(freeze_path.read_text(encoding="utf8"))
    journal_path = Path("artifacts/channel_exposure.json")
    journal = json.loads(journal_path.read_text(encoding="utf8"))
    assert journal["status"] == "completed fixed held-out block"
    assert journal["freeze_sha256"] == sha(freeze_path)
    rows = []
    for model in freeze["models"]:
        run, config = model["run"], model["config"]
        assert config["fraction"] == 10, "This figure caption describes the fixed 10% roster"
        completed = [
            item for item in journal["attempts"] if item["run"] == run and item["status"] == "completed"
        ]
        assert len(completed) == 1
        paths = {
            "Kenai validation": Path("artifacts") / run / "val/metrics.json",
            "Channel": Path(completed[0]["out"]) / "metrics.json",
        }
        assert sha(paths["Channel"]) == completed[0]["metrics_sha256"]
        assert sha(paths["Kenai validation"]) == model["validation_metrics_sha256"]
        if config.get("adapted"):
            parent = json.loads((Path(config["adapted"]).parent / "config.json").read_text())
            recipe = parent.get("adaptation_configuration", "A")
        else:
            recipe = "Random" if config["kind"] == "random" else "Published"
        mode = "frozen" if config["frozen"] else "full fine-tune"
        rows.append(
            {
                "run": run,
                "label": f"{recipe} / {mode} / {config['detector_size']} / seed {config['seed']}",
                "scores": {split: json.loads(path.read_text()) for split, path in paths.items()},
                "metric_hashes": {split: sha(path) for split, path in paths.items()},
            }
        )
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4.5, len(rows) * 0.55 + 1.5)), sharey=True)
    for axis, metric in zip(axes, ("AP50", "AP50:95"), strict=True):
        for offset, split, color in ((-0.18, "Kenai validation", "#2166ac"), (0.18, "Channel", "#d97706")):
            values = [100 * row["scores"][split][metric] for row in rows]
            bars = axis.barh(y + offset, values, height=0.34, color=color, label=split)
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        axis.set(xlim=(0, 100), xlabel=f"{metric} (%)")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, [row["label"] for row in rows], fontsize=9)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("Fixed detector roster: Kenai validation and one-location Channel transfer", fontsize=13)
    fig.text(
        0.5,
        0.012,
        "Author-evaluated; fixed 10% Kenai clips; no target-site training. Bars are individual checkpoints, not confidence intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    for suffix in (".png", ".svg"):
        fig.savefig(args.out.with_suffix(suffix), dpi=180)
    plt.close(fig)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "freeze_sha256": sha(freeze_path),
                "exposure_journal_sha256": sha(journal_path),
                "script_sha256": sha(__file__),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    print(json.dumps({"models": len(rows), "figure": str(args.out.with_suffix(".png"))}))


if __name__ == "__main__":
    main()
