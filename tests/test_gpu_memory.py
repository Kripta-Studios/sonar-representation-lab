"""Memory-query failure must not turn into a permission bypass or zero use."""

import pytest

from gpu_memory import check_status


def test_read_only_permission_denial_and_driver_errors_are_not_ignored():
    check_status(0)
    with pytest.raises(PermissionError, match="denies"):
        check_status(4)
    with pytest.raises(RuntimeError, match="status15"):
        check_status(15)
