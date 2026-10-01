"""Single-process resource accounting, atomic resume and reproducible RNG."""

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import platform_compat  # noqa: F401
import psutil
import torch

from acquire import save_json

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
LEDGER = ARTIFACTS / "resource_ledger.jsonl"
LOCK = ARTIFACTS / "gpu_process.lock"


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
    def __init__(self, name, directory):
        self.name = name
        self.out = Path(directory)
        self.out.mkdir(parents=True, exist_ok=True)
        self.start = None
        self.base_hours = used_hours()
        self.peak_rss = 0
        self.peak_device_used = 0

    def __enter__(self):
        if self.base_hours >= 24:
            raise RuntimeError("Initial 24 GPU-hour allocation exhausted")
        ARTIFACTS.mkdir(exist_ok=True)
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({"pid": os.getpid(), "name": self.name}).encode())
        os.close(fd)
        self.start = time.monotonic()
        # Leave room for observed desktop GPU allocations; never kill other processes.
        torch.cuda.set_per_process_memory_fraction(0.50)
        torch.cuda.reset_peak_memory_stats()
        self.event(
            {
                "event": "start",
                "name": self.name,
                "pid": os.getpid(),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        return self

    def event(self, value):
        with LEDGER.open("a", encoding="utf8") as f:
            f.write(json.dumps(value) + "\n")

    def check(self):
        p = psutil.Process()
        rss = p.memory_info().rss + sum(
            c.memory_info().rss for c in p.children(recursive=True) if c.is_running()
        )
        self.peak_rss = max(self.peak_rss, rss)
        free, total = torch.cuda.mem_get_info()
        # WDDM's CUDA memory view excludes some desktop allocations. Query the physical total too.
        physical = (
            int(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
                ).splitlines()[0]
            )
            * 1024**2
        )
        self.peak_device_used = max(self.peak_device_used, physical)
        values = {
            "elapsed_seconds": time.monotonic() - self.start,
            "owned_rss_bytes": rss,
            "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
            "device_used_bytes": total - free,
            "physical_device_used_bytes": physical,
        }
        if rss >= 22 * 1024**3 or max(total - free, physical) >= 10 * 1024**3:
            raise RuntimeError(f"Resource ceiling reached: {values}")
        if self.base_hours + values["elapsed_seconds"] / 3600 >= 23.99:
            raise RuntimeError("Initial allocation limit reached; preserve checkpoint")
        return values

    def __exit__(self, typ, exc, tb):
        seconds = time.monotonic() - self.start
        result = {
            "event": "end",
            "name": self.name,
            "pid": os.getpid(),
            "status": "completed" if typ is None else "failed_or_interrupted",
            "error": str(exc) if exc else None,
            "elapsed_seconds": seconds,
            "charged_gpu_hours": seconds / 3600,
            "owned_rss_peak_bytes": self.peak_rss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "physical_device_used_peak_bytes": self.peak_device_used,
            "overall_remaining_hours": 100 - self.base_hours - seconds / 3600,
            "block_remaining_hours": 24 - self.base_hours - seconds / 3600,
        }
        self.event(result)
        save_json(self.out / "resources.json", result)
        LOCK.unlink(missing_ok=True)


def append_curve(path, row):
    with Path(path).open("a", encoding="utf8") as f:
        f.write(json.dumps(row) + "\n")
