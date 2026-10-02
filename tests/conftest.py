import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import platform_compat  # noqa: E402,F401


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Avoid shared pytest-temp retention/cleanup affecting unrelated projects.
    if config.option.basetemp is None:
        root = Path(__file__).resolve().parents[1] / "artifacts" / "pytest-temp"
        root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(root / str(time.time_ns()))
