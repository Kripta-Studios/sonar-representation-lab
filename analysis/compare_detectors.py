"""Author checks on completed detector schedules, exposures and frozen weights."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import platform_compat  # noqa: E402,F401
import torch  # noqa: E402
from acquire import save_json  # noqa: E402
from models import PRETRAINED  # noqa: E402
from runtime import file_sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, previous_exposures, previous_settings = [], None, None
    keys = ["seed", "fraction", "steps", "batch", "accumulation", "manifest_sha256", "head_lr", "backbone_lr"]
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=True)
        config = state["config"]
        assert state["step"] == config["steps"] == 2000
        assert config["batch"] * config["accumulation"] == 8
        assert sum(state["exposures"].values()) == 16000
        settings = {key: config[key] for key in keys}
        if previous_settings is not None:
            assert settings == previous_settings, "Scientific training settings differ"
            assert state["exposures"] == previous_exposures, "Image exposure counters differ"
        previous_settings, previous_exposures = settings, state["exposures"]
        identity = None
        if config["frozen"] and config["kind"] != "random":
            initial_path = Path(config["adapted"]) if config.get("adapted") else PRETRAINED
            initial = torch.load(initial_path, map_location="cpu", weights_only=True)
            initial = initial.get("encoder", initial)
            assert all(
                torch.equal(value, state["model"]["backbone.encoder." + key])
                for key, value in initial.items()
            )
            identity = True
            del initial
        rows.append(
            {
                "checkpoint": str(path),
                "sha256": file_sha(path),
                "size": config["detector_size"],
                "kind": config["kind"],
                "frozen_encoder_identical_to_initialization": identity,
                "updates": state["step"],
                "presentations": sum(state["exposures"].values()),
                "unique_frames": len(state["exposures"]),
                "settings": settings,
            }
        )
        del state
    result = {
        "runs": rows,
        "matched_full_image_exposure_counters": True,
        "status": "author artifact verification, not independent review",
    }
    save_json(args.out, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
