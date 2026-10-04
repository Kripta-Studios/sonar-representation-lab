import summarize as module


def test_summary_never_pools_recipes_resolutions_or_training_regimes():
    row = {"recipe": "A", "kind": "adapted", "detector_size": 448, "fraction": 10, "split": "val"}
    reference = module.group_key(row)
    for change in [
        {"recipe": "B"},
        {"detector_size": 672},
        {"kind": "finetune"},
        {"fraction": 1},
        {"split": "channel"},
        {"neck": "image"},
        {"microbatch": 1, "accumulation": 8},
        {"evaluator_version": "cfc-coco-positive-annotation-ids-v2"},
    ]:
        assert module.group_key({**row, **change}) != reference
    assert module.group_key({**row, "seed": 13}) == module.group_key({**row, "seed": 23})
