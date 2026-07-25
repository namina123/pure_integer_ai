"""V-02 外层时钟、预注册清单和可中断恢复结果存储。"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def canonical_json_bytes(value: Any) -> bytes:
    """把无浮点诊断对象编码为排序键、紧凑分隔符的 UTF-8 JSON。"""
    _assert_no_float(value, where="canonical_json_bytes")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (payload + "\n").encode("utf-8")


def _assert_no_float(value: Any, *, where: str) -> None:
    """递归拒绝浮点，避免外层报告绕过纯整数度量约束。"""
    if isinstance(value, float):
        raise TypeError(f"{where} 含浮点")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_float(key, where=where)
            _assert_no_float(item, where=where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_float(item, where=where)


def sha256_path(path: str | Path) -> str:
    """流式计算文件 SHA-256，不把大课程产物整体读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """同目录写临时文件并原子替换，防中断留下半份结果。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}")
    payload = canonical_json_bytes(value)
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


class HostMonotonicClock:
    """在 Windows 实验边界读取 QueryPerformanceCounter 整数纳秒。"""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("当前宿主未提供 V-02 Windows 单调时钟适配")
        frequency = ctypes.c_longlong()
        if not ctypes.windll.kernel32.QueryPerformanceFrequency(
                ctypes.byref(frequency)):
            raise OSError("QueryPerformanceFrequency 失败")
        if frequency.value <= 0:
            raise OSError("QueryPerformanceFrequency 返回非法频率")
        self._frequency = frequency.value

    def __call__(self) -> int:
        """返回当前单调计数换算后的严格整数纳秒。"""
        counter = ctypes.c_longlong()
        if not ctypes.windll.kernel32.QueryPerformanceCounter(
                ctypes.byref(counter)):
            raise OSError("QueryPerformanceCounter 失败")
        return counter.value * 1_000_000_000 // self._frequency


class HostProcessMemory:
    """读取当前和进程级峰值工作集，供外层性能报告使用。"""

    class _Counters(ctypes.Structure):
        """Windows ``PROCESS_MEMORY_COUNTERS`` 的本地结构声明。"""

        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    def __init__(self) -> None:
        """绑定 64-bit 安全的 Windows 工作集查询签名。"""
        if os.name != "nt":
            self._get_process_memory_info = None
            self._get_current_process = None
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = kernel32.K32GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._Counters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        self._get_current_process = get_current_process
        self._get_process_memory_info = get_process_memory_info

    def __call__(self) -> dict[str, int]:
        """返回当前工作集和不可复位的进程峰值工作集。"""
        if os.name != "nt":
            return {
                "current_working_set_bytes": 0,
                "process_peak_working_set_bytes": 0,
            }
        if (self._get_process_memory_info is None
                or self._get_current_process is None):
            return {
                "current_working_set_bytes": 0,
                "process_peak_working_set_bytes": 0,
            }
        counters = self._Counters()
        counters.cb = ctypes.sizeof(counters)
        ok = self._get_process_memory_info(
            self._get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return {
                "current_working_set_bytes": 0,
                "process_peak_working_set_bytes": 0,
            }
        return {
            "current_working_set_bytes": int(counters.WorkingSetSize),
            "process_peak_working_set_bytes": int(
                counters.PeakWorkingSetSize),
        }


class V02RunStore:
    """管理一次 V-02 run 的预注册、逐点结果和最终摘要。"""

    def __init__(self, output_root: str | Path, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("V-02 run_id 不能为空")
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("V-02 run_id 必须是单层安全目录名")
        self.output_root = Path(output_root).resolve()
        self.run_id = run_id
        self.run_root = (self.output_root / run_id).resolve()
        if not self.run_root.is_relative_to(self.output_root):
            raise ValueError("V-02 run 目录越界")
        self.points_root = self.run_root / "points"
        self.artifacts_root = self.run_root / "artifacts"
        self.dumps_root = self.run_root / "dumps"

    @property
    def preregistration_path(self) -> Path:
        """返回跑前冻结清单路径。"""
        return self.run_root / "preregistered.json"

    @property
    def summary_path(self) -> Path:
        """返回最终或增量摘要路径。"""
        return self.run_root / "summary.json"

    def preregister(self, payload: dict[str, Any]) -> None:
        """首次运行先落预注册；恢复时要求内容逐字段完全一致。"""
        path = self.preregistration_path
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != payload:
                raise ValueError("既有 V-02 预注册与本次配置不一致，拒绝覆盖")
            return
        atomic_write_json(path, payload)

    def point_path(self, lane: str, n: int) -> Path:
        """返回 lane/规模唯一结果路径。"""
        if lane not in {"observe", "curriculum"}:
            raise ValueError("未知 V-02 lane")
        assert_int(n, _where="V02RunStore.point_path.n")
        if n <= 0:
            raise ValueError("V-02 规模必须为正")
        return self.points_root / f"{lane}-{n:06d}.json"

    def has_point(self, lane: str, n: int) -> bool:
        """判断一个规模点是否已有完整原子结果。"""
        return self.point_path(lane, n).is_file()

    def read_point(self, lane: str, n: int) -> dict[str, Any]:
        """读取一个已完成规模点。"""
        with self.point_path(lane, n).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("V-02 point 根必须是对象")
        return value

    def write_point(self, lane: str, n: int,
                    payload: dict[str, Any]) -> None:
        """原子写入一个完成点，已存在时要求 bit-identical。"""
        path = self.point_path(lane, n)
        if path.is_file():
            existing = self.read_point(lane, n)
            if existing != payload:
                raise ValueError("既有 V-02 point 与重算结果不一致")
            return
        atomic_write_json(path, payload)

    def write_named_result(self, name: str,
                           payload: dict[str, Any]) -> Path:
        """写 provider、遥测等非规模点结果，并拒绝路径穿越。"""
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("V-02 result 名必须是安全文件名")
        path = self.run_root / name
        atomic_write_json(path, payload)
        return path

    def write_summary(self, payload: dict[str, Any]) -> None:
        """每完成一个施工切片就刷新可恢复摘要。"""
        atomic_write_json(self.summary_path, payload)


__all__ = [
    "HostMonotonicClock",
    "HostProcessMemory",
    "V02RunStore",
    "atomic_write_json",
    "canonical_json_bytes",
    "sha256_path",
]
