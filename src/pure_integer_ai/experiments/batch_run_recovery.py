"""Read-only process probes for resuming a durable batch run.

On Windows, ``os.kill(pid, 0)`` is not a safe existence probe. Query a process
handle instead. A denied or incomplete probe must prevent a duplicate resume.
Neither this module nor its callers terminate or signal a Windows process.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Literal, Mapping


@dataclass(frozen=True, slots=True)
class ProcessProbe:
    pid: int
    status: Literal["LIVE", "EXITED", "UNKNOWN"]
    creation_epoch_ns: int | None = None
    reason: str = ""


def _windows_process_probe(pid: int) -> ProcessProbe:
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL

    # PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, with no mutation rights.
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists.
            return ProcessProbe(pid, "EXITED", reason="process_not_found")
        return ProcessProbe(pid, "UNKNOWN", reason=f"OpenProcess_error_{error}")
    try:
        wait_status = kernel.WaitForSingleObject(handle, 0)
        if wait_status == 0:  # WAIT_OBJECT_0: process has exited.
            return ProcessProbe(pid, "EXITED", reason="process_handle_signaled")
        if wait_status != 258:  # WAIT_TIMEOUT: a live process.
            return ProcessProbe(pid, "UNKNOWN", reason=f"wait_status_{wait_status}")
        creation, exit_time, kernel_time, user_time = (
            wintypes.FILETIME() for _ in range(4))
        created_ns = None
        if kernel.GetProcessTimes(
                handle, ctypes.byref(creation), ctypes.byref(exit_time),
                ctypes.byref(kernel_time), ctypes.byref(user_time)):
            ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            created_ns = (ticks - 116444736000000000) * 100
        return ProcessProbe(pid, "LIVE", creation_epoch_ns=created_ns)
    finally:
        kernel.CloseHandle(handle)


def probe_process(pid: int) -> ProcessProbe:
    """Return conservative liveness evidence, without a Windows signal call."""
    if type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
        raise ValueError("process PID must be a positive DWORD integer")
    if os.name == "nt":
        return _windows_process_probe(pid)
    try:
        os.kill(pid, 0)  # POSIX existence probe only; unreachable on Windows.
    except ProcessLookupError:
        return ProcessProbe(pid, "EXITED", reason="process_not_found")
    except OSError as error:
        return ProcessProbe(pid, "UNKNOWN", reason=type(error).__name__)
    return ProcessProbe(pid, "LIVE")


def require_inactive_run_owners(
        prior: Mapping[str, object],
        *, probe: Callable[[int], ProcessProbe] = probe_process,
        ) -> tuple[ProcessProbe, ...]:
    """Refuse resume while either owner may still write the run's artifacts.

    Legacy states lack process creation identity. Treat any live PID as a live
    owner, even if the PID might have been reused; that uncertainty cannot
    authorize another writer. Both the parent and worker must have exited.
    """
    status = prior.get("status")
    if status not in {"CASE_RUNNING", "CASE_WORKER_RUNNING"}:
        return ()
    fields = ("pid", "worker_pid") if status == "CASE_WORKER_RUNNING" else ("pid",)
    evidence = []
    for field in fields:
        pid = prior.get(field)
        if type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
            raise RuntimeError(f"prior {field} is unknown; refusing duplicate run")
        result = probe(pid)
        if result.pid != pid or result.status not in {"LIVE", "EXITED", "UNKNOWN"}:
            raise RuntimeError("process probe identity/status drift; refusing resume")
        if result.status == "LIVE":
            raise RuntimeError(f"prior {field} PID {pid} is still active; refusing duplicate run")
        if result.status == "UNKNOWN":
            raise RuntimeError(f"prior {field} PID {pid} liveness is unknown; refusing duplicate run")
        evidence.append(result)
    return tuple(evidence)
