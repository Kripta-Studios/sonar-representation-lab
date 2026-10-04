import pytest

import freeze as module


def valid():
    return {
        "runs": ["det-published-f010-s7"],
        "selected_resolution": 448,
        "development_closed": True,
        "omitted_cells": [],
        "forecast_final_gpu_hours": 5,
        "reserved_gpu_hours": 6,
    }


def test_final_roster_spec_has_a_finite_size_and_protected_reserve():
    module.validate_spec(valid())
    for patch in [
        {"runs": []},
        {"runs": [f"det-run-{index}" for index in range(13)]},
        {"runs": ["det-a", "det-a"]},
        {"runs": ["../another-project/checkpoint"]},
        {"development_closed": False},
        {"reserved_gpu_hours": 5},
        {"forecast_final_gpu_hours": 7},
        {"selected_resolution": 896},
    ]:
        with pytest.raises(ValueError):
            module.validate_spec({**valid(), **patch})
