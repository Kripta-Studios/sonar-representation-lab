"""Plot only completed, frozen Kenai measurements from this block."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from acquire import save_json  # noqa: E402
from runtime import file_sha  # noqa: E402


def main():
    block = PROJECT / "artifacts/localization-block"
    if (block / "comparison.png").exists():
        raise ValueError("Preserve the completed figure")
    frozen = json.loads((block / "final_freeze.json").read_text())
    rows = []
    for item in frozen["models"]:
        directory = PROJECT / item["path"]
        metric = json.loads((directory / "val/metrics.json").read_text())
        assert file_sha(directory / "val/metrics.json") == item["files"]["val/metrics.json"]
        name = "B" if item["adapted_sha256"] else "Published"
        if item["neck"] != "none":
            name = item["neck"].capitalize()
        rows.append(
            {
                "run": item["run"],
                "label": f"{name}\ns{item['seed']} / {item['fraction']}%",
                "AP50": 100 * metric["AP50"],
                "AP95": 100 * metric["AP50:95"],
                "neck": item["neck"],
                "metrics_sha256": item["files"]["val/metrics.json"],
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(rows) * 1.35), 4.5))
    colors = [{"none": "#465b72", "image": "#167d8d", "capacity": "#bc7735"}[r["neck"]] for r in rows]
    for axis, metric, title in zip(axes, ("AP50", "AP95"), ("AP50", "AP50:95"), strict=True):
        bars = axis.bar(range(len(rows)), [r[metric] for r in rows], color=colors)
        axis.set_xticks(range(len(rows)), [r["label"] for r in rows], fontsize=8)
        axis.set_ylim(0, max(r[metric] for r in rows) * 1.18)
        axis.set_ylabel("Detection AP (%)")
        axis.set_title(title)
        axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Kenai validation · matched microbatch4 / effective8 · 2,000 updates", fontsize=12)
    fig.text(
        0.5,
        0.015,
        "Each bar is one checkpoint; no confidence interval or new Channel score is shown.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    for suffix in ("png", "svg"):
        fig.savefig(block / f"comparison.{suffix}", dpi=180)
    plt.close(fig)
    save_json(
        block / "comparison.json",
        {
            "rows": rows,
            "freeze_sha256": file_sha(block / "final_freeze.json"),
            "script_sha256": file_sha(__file__),
            "author_evaluated": True,
        },
    )


if __name__ == "__main__":
    main()
