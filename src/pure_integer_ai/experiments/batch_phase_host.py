"""Windows endpoint counters for this process only, outside graph semantics.

No other PID is inspected. A cumulative process peak is never a phase peak.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time

from .batch_phase_evidence import BatchPhaseEvidenceError


@dataclass(frozen=True, slots=True)
class ProcessResourceEndpoint:
    pid: int
    creation_epoch_ns: int
    sampled_started_ns: int
    sampled_ended_ns: int
    sampled_epoch_ns: int
    cpu_ns: int
    working_set_bytes: int | None
    process_peak_working_set_bytes: int | None
    memory_error: str | None = None

    def __post_init__(self) -> None:
        for field in ("pid", "creation_epoch_ns", "sampled_epoch_ns"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise BatchPhaseEvidenceError(f"{field} must be a positive integer")
        if self.pid > 0xFFFFFFFF:
            raise BatchPhaseEvidenceError("PID exceeds DWORD")
        for field in ("sampled_started_ns", "sampled_ended_ns", "cpu_ns"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise BatchPhaseEvidenceError(f"{field} must be a nonnegative integer")
        if self.sampled_ended_ns < self.sampled_started_ns:
            raise BatchPhaseEvidenceError("endpoint sampling clock reversed")
        for field in ("working_set_bytes", "process_peak_working_set_bytes"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise BatchPhaseEvidenceError(f"{field} must be a nonnegative integer or None")
        current, peak = self.working_set_bytes, self.process_peak_working_set_bytes
        if current is not None and peak is not None and peak < current:
            raise BatchPhaseEvidenceError("process peak is below current working set")
        if self.memory_error is not None:
            if type(self.memory_error) is not str or not self.memory_error:
                raise BatchPhaseEvidenceError("memory error must be a nonempty string")
            if current is not None or peak is not None:
                raise BatchPhaseEvidenceError("failed memory query cannot supply counters")

    def to_value(self) -> dict[str, object]:
        return asdict(self)


def current_process_resource_endpoint() -> ProcessResourceEndpoint:
    """Read identity/CPU/memory from the same current-process pseudo handle.

    Sampling has a recorded interval; these are sequential API observations,
    not an atomic OS snapshot. Unsupported platforms do not supply fake zeros.
    """
    if os.name != "nt":
        raise NotImplementedError("Windows current-process counters are required")
    import ctypes
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (
        ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    started = time.perf_counter_ns()
    epoch = time.time_ns()
    handle = kernel.GetCurrentProcess()  # Pseudo handle: do not CloseHandle.
    creation, exited, kernel_cpu, user_cpu = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exited),
                                  ctypes.byref(kernel_cpu), ctypes.byref(user_cpu)):
        raise ctypes.WinError(ctypes.get_last_error())

    def ticks(value: wintypes.FILETIME) -> int:
        return (value.dwHighDateTime << 32) | value.dwLowDateTime

    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    current = int(counters.WorkingSetSize) if ok else None
    peak = int(counters.PeakWorkingSetSize) if ok else None
    memory_error = None if ok else f"GetProcessMemoryInfo_error_{ctypes.get_last_error()}"
    return ProcessResourceEndpoint(
        os.getpid(), (ticks(creation) - 116444736000000000) * 100,
        started, time.perf_counter_ns(), epoch,
        (ticks(kernel_cpu) + ticks(user_cpu)) * 100, current, peak, memory_error)
