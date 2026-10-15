"""Read-only host measurements. No torch import or GPU allocation required."""
from __future__ import annotations

import csv
import ctypes
import io
import platform
import subprocess
import sys
from datetime import datetime, timezone


def gpu_budget_mib(free_mib: int, reserve_mib: int = 1024, cap_mib: int = 4096) -> int:
    if min(free_mib, reserve_mib, cap_mib) < 0:
        raise ValueError("memory values cannot be negative")
    return max(0, min(cap_mib, free_mib - reserve_mib))


def hardware_report() -> dict:
    report = {"measured_at": datetime.now(timezone.utc).isoformat(),
              "os": platform.platform(), "python": sys.version, "gpus": []}
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, check=True, timeout=10)
        for row in csv.reader(io.StringIO(result.stdout)):
            name, total, free, used, driver = [item.strip() for item in row]
            report["gpus"].append({"name": name, "total_mib": int(total), "free_mib": int(free),
                                   "used_mib": int(used), "driver": driver,
                                   "provisional_budget_mib": gpu_budget_mib(int(free))})
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        report["gpu_error"] = str(exc)
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            report["ram"] = {"total_mib": status.total // 2**20,
                             "available_mib": status.available // 2**20, "load_percent": status.load}
    return report
