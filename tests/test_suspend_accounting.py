from runtime import suspended_overlap


def test_verified_suspend_intersections_and_no_double_charge():
    events = [
        {"sleep_time": "2026-10-01T23:00:00Z", "wake_time": "2026-10-02T07:00:00Z"},
        {"sleep_time": "2026-10-02T06:00:00Z", "wake_time": "2026-10-02T08:00:00Z"},
    ]
    from runtime import epoch

    start, end = epoch("2026-10-02T05:00:00Z"), epoch("2026-10-02T09:00:00Z")
    assert suspended_overlap(start, end, events) == 3 * 3600
    assert suspended_overlap(epoch("2026-10-02T08:00:00Z"), end, events) == 0
    assert suspended_overlap(start, end, []) == 0
