"""Verify a real completed spatial encoder and its full training ancestry on CPU."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import platform_compat  # noqa: E402,F401
import torch  # noqa: E402
from acquire import ROOT, save_json  # noqa: E402
from models import PRETRAINED  # noqa: E402
from runtime import assert_state_equal, file_sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--compare-A", type=Path)
    parser.add_argument("--paired-other", type=Path)
    args = parser.parse_args()
    out = args.run / "artifact_verification.json"
    if out.exists():
        raise ValueError("Preserve the recorded artifact check")
    torch.set_num_threads(4)
    full = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=True)
    exported = torch.load(args.run / "encoder.pt", map_location="cpu", weights_only=True)
    recipe = full["config"]["adaptation_configuration"]
    assert recipe in ("B", "B-CONTROL", "C-MOTION")
    updates = 2000 if recipe == "B" else 1000
    assert full["step"] == exported["successful_updates"] == full["config"]["steps"] == updates
    assert full["config"] == exported["config"]
    assert not full["config"]["resource_profile_only"]
    assert exported["encoder"].keys() == full["teacher"].keys()
    assert all(torch.equal(value, full["teacher"][key]) for key, value in exported["encoder"].items())
    for key in [
        "student",
        "teacher",
        "student_head",
        "teacher_head",
        "student_patch_head",
        "teacher_patch_head",
        "objective",
        "patch_objective",
    ]:
        assert all(torch.isfinite(value).all() for value in full[key].values()), key
    assert not torch.equal(full["objective"]["center"], full["patch_objective"]["center"])
    names = set(json.loads((ROOT / "manifests/ssl-train.json").read_text())["filenames"])
    assert set(full["exposures"]) <= names
    assert sum(full["exposures"].values()) == updates * 16
    original = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
    assert original.keys() == exported["encoder"].keys()
    changed = [key for key, value in original.items() if not torch.equal(value, exported["encoder"][key])]
    assert changed and "mask_token" in changed
    record = {
        "checkpoint_sha256": file_sha(args.run / "checkpoint.pt"),
        "encoder_sha256": file_sha(args.run / "encoder.pt"),
        "updates": full["step"],
        "frame_presentations": updates * 16,
        "unique_frames": len(full["exposures"]),
        "teacher_export_exact": True,
        "all_training_modules_finite": True,
        "global_patch_centers_distinct": True,
        "changed_encoder_tensor_keys": changed,
        "status": "author-executed artifact checks, not independent review",
    }
    if recipe != "B":
        parent_path = Path(full["config"]["parent_checkpoint"])
        assert file_sha(parent_path) == full["config"]["parent_checkpoint_sha256"]
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        assert parent["step"] == 2000 and parent["config"]["adaptation_configuration"] == "B"
        incremental = [
            key for key, value in parent["teacher"].items() if not torch.equal(value, full["teacher"][key])
        ]
        assert incremental
        assert all(torch.isfinite(value).all() for value in full["dynamic_auxiliary"].values())
        for index, initial_state in parent["optimizer"]["state"].items():
            assert float(full["optimizer"]["state"][index]["step"]) == float(initial_state["step"]) + updates
        record.update(
            {
                "parent_checkpoint_sha256": file_sha(parent_path),
                "parent_updates": 2000,
                "incremental_teacher_changed_tensors": incremental,
                "parent_optimizer_counters_advanced_by_declared_updates": True,
            }
        )
        del parent
    if args.paired_other:
        assert recipe in ("B-CONTROL", "C-MOTION")
        other = torch.load(args.paired_other, map_location="cpu", weights_only=True)
        assert {recipe, other["config"]["adaptation_configuration"]} == {"B-CONTROL", "C-MOTION"}
        for key in ("step", "exposures", "rng", "mask_rng"):
            assert_state_equal(full[key], other[key], key)
        for key in (
            "parent_checkpoint_sha256",
            "calibration_sha256",
            "seed",
            "manifest_sha256",
            "batch",
            "steps",
            "code_identity",
            "protocol_sha256",
        ):
            assert_state_equal(full["config"][key], other["config"][key], key)
        record.update(
            {
                "paired_checkpoint_sha256": file_sha(args.paired_other),
                "paired_exposures_all_rng_masks_parent_and_settings_exact": True,
            }
        )
        del other
    if args.compare_A:
        assert recipe == "B"
        other = torch.load(args.compare_A, map_location="cpu", weights_only=True)
        assert other["step"] == 2000 and other["config"]["seed"] == full["config"]["seed"]
        assert full["exposures"] == other["exposures"]
        assert torch.equal(full["rng"]["generator"], other["rng"]["generator"])
        record.update(
            {
                "A_checkpoint_sha256": file_sha(args.compare_A),
                "image_exposure_counters_match_A": True,
                "image_sampler_state_matches_A": True,
                "final_CPU_augmentation_rng_matches_A": torch.equal(
                    full["rng"]["torch"], other["rng"]["torch"]
                ),
            }
        )
    save_json(out, record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
