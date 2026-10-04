"""Single-process resource accounting, atomic resume and reproducible RNG."""

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import platform_compat  # noqa: F401
import psutil
import torch

from acquire import save_json
from budget import allocation_limits
from gpu_memory import physical_used_bytes

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
LEDGER = ARTIFACTS / "resource_ledger.jsonl"
LOCK = ARTIFACTS / "gpu_process.lock"
POWER_EVENTS = ARTIFACTS / "verified_suspend_intervals.json"


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def suspended_overlap(start, end, events):
    """Union of OS-confirmed sleep/wake intervals overlapping this process."""
    spans = []
    for event in events:
        lo, hi = epoch(event["sleep_time"]), epoch(event["wake_time"])
        if hi <= lo:
            raise ValueError("Invalid OS sleep/wake interval")
        lo, hi = max(start, lo), min(end, hi)
        if hi > lo:
            spans.append((lo, hi))
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return sum(hi - lo for lo, hi in merged)


def capture_suspend_events():
    """Read OS wake events; failure conservatively retains wall-time charging."""
    previous = json.loads(POWER_EVENTS.read_text()) if POWER_EVENTS.exists() else {"events": []}
    query = """
$events = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Power-Troubleshooter'; Id=1; StartTime=(Get-Date).AddDays(-7)} -ErrorAction SilentlyContinue)
@($events | ForEach-Object {
    $xml = [xml]$_.ToXml()
    $fields = @{}
    foreach ($item in $xml.Event.EventData.Data) { $fields[$item.Name] = $item.'#text' }
    [pscustomobject]@{record_id=$_.RecordId; sleep_time=$fields['SleepTime']; wake_time=$fields['WakeTime']}
}) | ConvertTo-Json -Compress
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        items = json.loads(result.stdout) if result.stdout.strip() else []
        if isinstance(items, dict):
            items = [items]
        events = {int(e["record_id"]): e for e in previous["events"]}
        for event in items:
            # Reject malformed records instead of inventing inactive time.
            if epoch(event["wake_time"]) <= epoch(event["sleep_time"]):
                raise ValueError("Invalid power event")
            events[int(event["record_id"])] = event
        previous = {
            "source": "Windows System / Microsoft-Windows-Power-Troubleshooter / EventID 1",
            "events": list(events.values()),
            "last_query_error": None,
        }
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        previous["last_query_error"] = str(error)
    previous["last_query_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(POWER_EVENTS, previous)
    return previous["events"]


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def rng_state(generator):
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "generator": generator.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    generator.set_state(state["generator"].cpu())
    random.setstate(state["python"])


def checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def assert_state_equal(left, right, path="state"):
    """Exact loaded-state identity, independent of CPU/CUDA placement."""
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu()), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_state_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_state_equal(a, b, f"{path}.{index}")
    else:
        assert left == right, path


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def code_identity():
    root = Path(__file__).resolve().parents[1]
    return {"src/" + p.name: file_sha(p) for p in sorted((root / "src").glob("*.py"))}


def used_hours():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line).get("charged_gpu_hours", 0) for line in LEDGER.read_text().splitlines())


class Resources:
    def __init__(self, name, directory, allocation=None):
        self.name = name
        self.out = Path(directory)
        self.out.mkdir(parents=True, exist_ok=True)
        self.start = None
        self.base_hours = used_hours()
        allocation = allocation or os.environ.get("SONAR_RESEARCH_ALLOCATION")
        self.budget = allocation_limits(
            LEDGER,
            allocation,
            os.environ.get("SONAR_RESEARCH_PHASE", "development"),
            os.environ.get("SONAR_FINAL_FREEZE"),
            os.environ.get("SONAR_ORIGINAL_BLOCK_OPERATION") == "1",
        )
        self.peak_rss = 0
        self.peak_device_used = 0

    def __enter__(self):
        if self.base_hours >= self.budget["charge_ceiling_hours"] - 0.01:
            raise RuntimeError("Research block limit or protected final reserve reached")
        ARTIFACTS.mkdir(exist_ok=True)
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({"pid": os.getpid(), "name": self.name}).encode())
        os.close(fd)
        self.start = time.monotonic()
        self.start_epoch = time.time()
        self.suspend_events = capture_suspend_events()
        self.last_power_refresh = time.monotonic()
        # Leave room for observed desktop GPU allocations; never kill other processes.
        torch.cuda.set_per_process_memory_fraction(0.50)
        torch.cuda.reset_peak_memory_stats()
        self.event(
            {
                "event": "start",
                "name": self.name,
                "pid": os.getpid(),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "started_epoch": self.start_epoch,
                **self.budget,
            }
        )
        return self

    def event(self, value):
        with LEDGER.open("a", encoding="utf8") as f:
            f.write(json.dumps(value) + "\n")

    def timing(self, refresh=False):
        if refresh or time.monotonic() - self.last_power_refresh >= 300:
            self.suspend_events = capture_suspend_events()
            self.last_power_refresh = time.monotonic()
        elapsed = time.monotonic() - self.start
        suspended = min(elapsed, suspended_overlap(self.start_epoch, time.time(), self.suspend_events))
        return elapsed, suspended, max(0, elapsed - suspended)

    def observe_owned_ram(self):
        process = psutil.Process()
        memory = process.memory_info()
        child_rss = sum(
            child.memory_info().rss for child in process.children(recursive=True) if child.is_running()
        )
        rss = memory.rss + child_rss
        # Windows peak_wset retains CPU evaluator peaks between sampled checks.
        native_peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss)) + child_rss
        self.peak_rss = max(self.peak_rss, native_peak)
        return rss

    def check(self):
        elapsed, suspended, active = self.timing()
        rss = self.observe_owned_ram()
        free, total = torch.cuda.mem_get_info()
        # WDDM's CUDA memory view excludes some desktop allocations. Query the physical total too.
        physical, physical_source = physical_used_bytes()
        self.peak_device_used = max(self.peak_device_used, physical)
        values = {
            "elapsed_seconds": elapsed,
            "verified_suspend_seconds": suspended,
            "active_gpu_process_seconds": active,
            "owned_rss_bytes": rss,
            "owned_rss_peak_bytes": self.peak_rss,
            "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
            "device_used_bytes": total - free,
            "physical_device_used_bytes": physical,
            "physical_memory_source": physical_source,
        }
        if self.peak_rss >= 22 * 1024**3 or max(total - free, physical) >= 10 * 1024**3:
            raise RuntimeError(f"Resource ceiling reached: {values}")
        if self.base_hours + active / 3600 >= self.budget["charge_ceiling_hours"] - 0.01:
            raise RuntimeError("Research allocation/reserve limit reached; preserve checkpoint")
        return values

    def __exit__(self, typ, exc, tb):
        self.observe_owned_ram()
        seconds, suspended, active = self.timing(refresh=True)
        result = {
            "event": "end",
            "name": self.name,
            "pid": os.getpid(),
            "status": "completed" if typ is None else "failed_or_interrupted",
            "error": str(exc) if exc else None,
            "elapsed_seconds": seconds,
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verified_suspend_seconds": suspended,
            "active_gpu_process_seconds": active,
            "charged_gpu_hours": active / 3600,
            "owned_rss_peak_bytes": self.peak_rss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "physical_device_used_peak_bytes": self.peak_device_used,
            "overall_remaining_hours": 100 - self.base_hours - active / 3600,
            "block_remaining_hours": self.budget["block_ceiling_hours"] - self.base_hours - active / 3600,
            **self.budget,
        }
        result["remaining_development_hours"] = self.budget["remaining_development_hours"] - active / 3600
        self.event(result)
        save_json(self.out / "resources.json", result)
        LOCK.unlink(missing_ok=True)


def append_curve(path, row):
    with Path(path).open("a", encoding="utf8") as f:
        f.write(json.dumps(row) + "\n")
