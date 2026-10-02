"""Record fixed model/preprocessing identities before held-out location exposure."""

import json
import time
from pathlib import Path

import platform_compat  # noqa: F401
from acquire import ROOT, save_json
from runtime import code_identity, file_sha, used_hours


def main():
    target = Path("artifacts/freeze.json")
    if target.exists():
        raise RuntimeError("Freeze already exists; do not silently replace held-out policy")
    models = []
    for directory in sorted(Path("artifacts").glob("det-*")):
        if not (directory / "val/metrics.json").exists():
            raise RuntimeError(f"Incomplete attempted detector: {directory}")
        config = json.loads((directory / "config.json").read_text())
        models.append(
            {
                "run": directory.name,
                "checkpoint": str(directory / "checkpoint.pt"),
                "checkpoint_sha256": file_sha(directory / "checkpoint.pt"),
                "config": config,
                "validation_metrics_sha256": file_sha(directory / "val/metrics.json"),
            }
        )
    if not models:
        raise RuntimeError("No completed trained detector to freeze")
    required = {(kind, seed) for kind in ["published", "adapted"] for seed in [7, 13, 23]}
    required.update({("finetune", 7), ("random", 7)})
    completed = {(m["config"]["kind"], m["config"]["seed"]) for m in models if m["config"]["fraction"] == 10}
    if not required.issubset(completed):
        raise RuntimeError(f"Main comparison incomplete: {sorted(required - completed)}")
    save_json(
        target,
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "models and policy fixed before Channel imagery evaluation",
            "models": models,
            "code_identity": code_identity(),
            "protocol_sha256": file_sha("PROTOCOL.md"),
            "label_policy": "fixed nested complete train clips; no target-site adaptation",
            "preprocessing": "448 grayscale top-left aspect-preserving letterbox; rounded axis scales; ImageNet normalization",
            "cache_policy": "Optional byte-identical grayscale 448 canvases; Channel cache preparation starts only inside the recorded held-out exposure block; no fitted target statistics",
            "score_threshold_saved": 0.001,
            "operating_score": 0.5,
            "nms": 0.5,
            "max_detections": 100,
            "channel_annotations_sha256": file_sha(
                ROOT / "metadata/coco_annotations_v1.1/kenai-channel.json"
            ),
            "gpu_hours_charged_before_freeze": used_hours(),
            "prior_exposure": "initial metadata listing displayed several Channel annotation/clip headers; no Channel image or model score",
        },
    )
    print(json.dumps({"models_frozen": len(models), "file": str(target)}))


if __name__ == "__main__":
    main()
