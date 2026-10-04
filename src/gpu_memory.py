"""Read-only physical GPU memory queries; no driver or power-policy changes."""

import ctypes
import os
import subprocess


class MemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


_reader = None
_unsupported = False


def check_status(status):
    if status == 4:
        raise PermissionError("NVML denies this read-only query; do not bypass")
    if status != 0:
        raise RuntimeError(f"NVML query error status{status}")


def physical_used_bytes():
    global _reader, _unsupported
    if os.name == "nt" and not _unsupported:
        if _reader is None:
            try:
                library = ctypes.WinDLL("nvml.dll")
                check_status(library.nvmlInit_v2())
                handle = ctypes.c_void_p()
                library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                    ctypes.c_uint,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                check_status(library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)))
                query = library.nvmlDeviceGetMemoryInfo_v2
                query.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryV2)]
                _reader = library, handle, query
            except (OSError, AttributeError):
                _unsupported = True
        if _reader is not None:
            _, handle, query = _reader
            memory = MemoryV2()
            memory.version = ctypes.sizeof(MemoryV2) | (2 << 24)
            check_status(query(handle, ctypes.byref(memory)))
            if not 0 <= memory.used <= memory.total or memory.total == 0:
                raise RuntimeError("Invalid physical GPU memory query")
            # Match the conservative whole-MiB granularity of nvidia-smi output.
            rounded = ((int(memory.used) + 1024**2 - 1) // 1024**2) * 1024**2
            return rounded, "NVML nvmlDeviceGetMemoryInfo_v2 used, rounded upward to MiB"
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).splitlines()[0]
    return int(value) * 1024**2, "nvidia-smi memory.used fallback"
