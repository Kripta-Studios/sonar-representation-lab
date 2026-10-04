"""Plot the completed seed-7 resolution contrast from its recorded metrics."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    source = Path("artifacts/resolution_decision.json")
    record = json.loads(source.read_text(encoding="utf8"))
    target = Path("artifacts/resolution_comparison")
    if any(target.with_suffix(suffix).exists() for suffix in (".png", ".svg", ".json")):
        raise ValueError("Preserve the existing figure")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    positions = np.arange(2)
    panels = [
        ("AP50", "AP50 (%)"),
        ("AP50:95", "AP50:95 (%)"),
        ("small", "Recall of fixed <16px group (%)"),
    ]
    for axis, (metric, title) in zip(axes, panels, strict=True):
        for offset, (recipe, label, color) in enumerate(
            [("published", "Published frozen", "#276291"), ("A", "A frozen", "#bd641e")]
        ):
            values = [
                100
                * (
                    record["below_16px_recall_fixed_448_bins"][f"{recipe}{size}"]
                    if metric == "small"
                    else record["scores"][f"{recipe}{size}"][metric]
                )
                for size in (448, 672)
            ]
            bars = axis.bar(positions + (offset - 0.5) * 0.36, values, 0.36, label=label, color=color)
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.set_xticks(positions, ["448", "672"])
        axis.set(xlabel="Detector input side (pixels)", ylabel=title)
        axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Kenai validation · seed 7 · 10% complete TRAIN clips · 2,000 detector updates")
    fig.text(
        0.5,
        0.02,
        "One seed, no uncertainty estimate. Recall: score 0.5 / IoU 0.5; size bins fixed at 448. Author-evaluated.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for suffix in (".png", ".svg"):
        fig.savefig(target.with_suffix(suffix), dpi=180)
    plt.close(fig)
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "gpu_used": False,
                "metrics_unchanged": True,
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
