from analysis.resolution_decision import criteria


def test_all_resolution_criteria_are_required_and_ap_units_are_fractions():
    old = {"AP50": 0.15, "AP50:95": 0.04}
    new = {"AP50": 0.16, "AP50:95": 0.04}
    assert all(criteria(old, new, 0.05, 0.06, True).values())
    assert not all(criteria(old, {**new, "AP50": 0.159}, 0.05, 0.06, True).values())
    assert not all(criteria(old, {**new, "AP50:95": 0.039}, 0.05, 0.06, True).values())
    assert not all(criteria(old, new, 0.05, 0.05, True).values())
    assert not all(criteria(old, new, 0.05, 0.06, False).values())
