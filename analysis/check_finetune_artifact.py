"""Verify completed supervised encoder updates against their recorded initialization."""

import argparse
import json
import math
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
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    target = args.run / "finetuning_verification.json"
    if target.exists():
        raise ValueError("Preserve existing artifact verification")
    torch.set_num_threads(4)
    path = args.run / "checkpoint.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state["config"]
    assert config["kind"] == "finetune" and not config["frozen"]
    assert state["step"] == config["steps"] == 2000
    parent = Path(config["adapted"]) if config.get("adapted") else PRETRAINED
    initial = torch.load(parent, map_location="cpu", weights_only=True)
    initial = initial.get("encoder", initial)
    changed, squared, norm = 0, 0.0, 0.0
    for key, value in initial.items():
        trained = state["model"]["backbone.encoder." + key]
        changed += not torch.equal(trained, value)
        squared += float((trained.double() - value.double()).square().sum())
        norm += float(value.double().square().sum())
    assert changed > 0 and all(torch.isfinite(v).all() for v in state["model"].values())
    optimizer = state["optimizer"]
    assert all(float(row["step"]) == 2000 for row in optimizer["state"].values())
    counts = [
        sum(index in optimizer["state"] for index in group["params"]) for group in optimizer["param_groups"]
    ]
    assert all(counts)
    result = {
        "checkpoint_sha256": file_sha(path),
        "parent_encoder_sha256": file_sha(parent),
        "updates": state["step"],
        "changed_encoder_tensor_count": changed,
        "relative_encoder_l2_change_from_initialization": math.sqrt(squared / norm),
        "optimizer_parameters_with_state_per_head_encoder_group": counts,
        "all_active_optimizer_counters_2000": True,
        "all_model_tensors_finite": True,
        "status": "author artifact verification, not independent review",
    }
    save_json(target, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
