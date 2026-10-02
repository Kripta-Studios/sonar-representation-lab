from types import SimpleNamespace

import runtime
from runtime import Resources


def test_native_peak_survives_between_sampled_checks(tmp_path, monkeypatch):
    memory = SimpleNamespace(rss=200, peak_wset=900)
    child = SimpleNamespace(is_running=lambda: True, memory_info=lambda: SimpleNamespace(rss=50))
    process = SimpleNamespace(memory_info=lambda: memory, children=lambda recursive: [child])
    monkeypatch.setattr(runtime.psutil, "Process", lambda: process)
    resources = Resources("ram-known-answer", tmp_path)
    assert resources.observe_owned_ram() == 250
    assert resources.peak_rss == 950
    memory.rss = 100
    memory.peak_wset = 300
    assert resources.observe_owned_ram() == 150
    assert resources.peak_rss == 950


def test_peak_fallback_for_memory_info_without_windows_field(tmp_path, monkeypatch):
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=200), children=lambda recursive: [])
    monkeypatch.setattr(runtime.psutil, "Process", lambda: process)
    resources = Resources("ram-portable-known-answer", tmp_path)
    assert resources.observe_owned_ram() == 200
    assert resources.peak_rss == 200
