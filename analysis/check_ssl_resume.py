"""Exact restored state plus bounded BF16 continuation versus fresh replay."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import platform_compat  # noqa: E402,F401
import torch  # noqa: E402
from acquire import save_json  # noqa: E402
from runtime import assert_state_equal, file_sha  # noqa: E402


def numerical_difference(reference, other):
    for key in ("config", "step", "exposures", "rng", "mask_rng"):
        assert_state_equal(reference[key], other[key], key)
    assert_state_equal(reference["optimizer"]["param_groups"], other["optimizer"]["param_groups"])
    groups = {}
    for name in reference:
        if name in ("config", "step", "exposures", "rng", "mask_rng", "optimizer"):
            continue
        assert reference[name].keys() == other[name].keys()
        groups[name] = [(value, other[name][key]) for key, value in reference[name].items()]
    assert reference["optimizer"]["state"].keys() == other["optimizer"]["state"].keys()
    for index, values in reference["optimizer"]["state"].items():
        actual = other["optimizer"]["state"][index]
        assert values.keys() == actual.keys()
        for key, value in values.items():
            if key == "step":
                assert_state_equal(value, actual[key])
            else:
                groups.setdefault("optimizer." + key, []).append((value, actual[key]))
    result = {}
    for name, pairs in groups.items():
        squared, reference_squared, maximum = 0.0, 0.0, 0.0
        for left, right in pairs:
            assert left.shape == right.shape and left.dtype == right.dtype
            assert torch.isfinite(left).all() and torch.isfinite(right).all()
            delta = left.double() - right.double()
            squared += float(delta.square().sum())
            reference_squared += float(left.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        result[name] = {"relative_l2": math.sqrt(squared / max(reference_squared, 1e-30)), "max_abs": maximum}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--reference-repeat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion", action="store_true", help="Prospective paired-continuation replay bounds")
    parser.add_argument("--paired-control", type=Path, help="Compare C profile's paired stream with control")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve prior resume-verification results")
    torch.set_num_threads(4)
    states = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in (args.full, args.resumed, args.reference_repeat)
    ]
    assert all(state["config"]["resource_profile_only"] and state["step"] == 4 for state in states)
    if args.motion:
        assert states[0]["config"]["adaptation_configuration"] in ("B-CONTROL", "C-MOTION")
    reload_record = json.loads((args.resumed.parent / "exact_reload_step2.json").read_text())
    assert reload_record["all_loaded_states_exact"]
    resumed = numerical_difference(states[0], states[1])
    repeated = numerical_difference(states[0], states[2])
    if args.paired_control:
        assert args.motion and states[0]["config"]["adaptation_configuration"] == "C-MOTION"
        control = torch.load(args.paired_control, map_location="cpu", weights_only=True)
        assert control["config"]["adaptation_configuration"] == "B-CONTROL"
        for key in ("step", "exposures", "rng", "mask_rng"):
            assert_state_equal(states[0][key], control[key], key)
        assert (
            states[0]["config"]["parent_checkpoint_sha256"] == control["config"]["parent_checkpoint_sha256"]
        )
        del control
    for name, values in resumed.items():
        ceiling = (
            0.05
            if name.startswith("optimizer.")
            else 0.005
            if "objective" in name
            else 1e-4
            if args.motion and name == "dynamic_auxiliary"
            else 1e-6
        )
        for row in (values, repeated[name]):
            assert row["relative_l2"] <= ceiling, (name, row, ceiling)
            if not name.startswith("optimizer.") and "objective" not in name:
                maximum = 1e-3 if args.motion and name == "dynamic_auxiliary" else 5e-6
                assert row["max_abs"] <= maximum, (name, row)
        assert values["relative_l2"] <= 3 * repeated[name]["relative_l2"] + 1e-8, (
            name,
            values,
            repeated[name],
        )
    record = {
        "uninterrupted_sha256": file_sha(args.full),
        "resumed_sha256": file_sha(args.resumed),
        "fresh_repeat_sha256": file_sha(args.reference_repeat),
        "successful_updates": 4,
        "exact_loaded_model_optimizer_centers_rng_verified": True,
        "rng_exposures_step_config_exact": True,
        "checkpoint_state_restoration_and_numeric_continuation_pass": True,
        "bitwise_training_trajectory_claimed": False,
        "resumed_differences": resumed,
        "fresh_replay_differences": repeated,
        "bounds": "module relative L2 <=1e-6/max abs <=5e-6; centers <=0.005; optimizer moments <=0.05; resumed relative error <=3*fresh replay error+1e-8",
        "dynamic_auxiliary_bounds": "relative L2 <=1e-4/max abs <=1e-3" if args.motion else None,
        "paired_control_stream_checked": bool(args.paired_control),
        "status": "author real TRAIN engineering test; preserved earlier failed bitwise check; not a detection result",
    }
    save_json(args.out, record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
