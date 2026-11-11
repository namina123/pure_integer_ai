"""真实 case 进度点的资源 owner；不使用定时轮询或业务语义时钟。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from shutil import disk_usage
import stat
from typing import Iterator

from pure_integer_ai.experiments.batch_phase_host import current_process_resource_endpoint
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


class RelationResourceBudgetError(RuntimeError):
    """当前真实进度点超过冻结预算，必须退出该 case。"""


@dataclass(frozen=True, slots=True)
class RelationResourceLimits:
    """所有整数均为物理 bytes；进度检查不等于任意指令间硬限。"""

    max_artifact_bytes: int
    max_memory_database_bytes: int
    max_case_memory_growth_bytes: int
    max_working_set_bytes: int
    max_process_peak_working_set_bytes: int
    minimum_free_disk_bytes: int
    diagnostic_reserve_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        """拒绝 bool、缺失预算和不足以保留失败诊断的配置。"""
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("all resource limits must be positive strict integer bytes")
        if not 65536 <= self.diagnostic_reserve_bytes < self.max_artifact_bytes:
            raise ValueError("resource diagnostic reserve must fit inside the artifact budget")


_CURRENT: ContextVar[RelationCaseResources | None] = ContextVar("relation_case_resources", default=None)


def _tree_bytes(root: Path) -> int:
    """逐次完整枚举已解析 root；拒绝链接/reparse，不跟随目录跳转或缓存总量。"""
    total = 0
    pending = [root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                if (stat.S_ISLNK(metadata.st_mode)
                        or getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                    raise RelationResourceBudgetError(
                        "resource tree contains a symbolic link or reparse path: "
                        + Path(entry.path).relative_to(root).as_posix())
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(entry.path)
                elif stat.S_ISREG(metadata.st_mode):
                    total += metadata.st_size
    return total


def preflight_relation_resource_tree(root: Path, limits: RelationResourceLimits) -> dict[str, int]:
    """准备和训练前复用完整资源树规则；拒绝保留链接，不替代 case 进度门。"""
    root = root.resolve(strict=True)
    if root.drive.upper() != "K:" or root.parent == Path("K:/") or not root.is_dir():
        raise ValueError("resource preflight requires a named explicit K root")
    total = _tree_bytes(root)
    free = disk_usage(root).free
    if total > limits.max_artifact_bytes - limits.diagnostic_reserve_bytes:
        raise RelationResourceBudgetError("ARTIFACT_PREFLIGHT_LIMIT")
    if free < limits.minimum_free_disk_bytes:
        raise RelationResourceBudgetError("FREE_DISK_PREFLIGHT_LIMIT")
    return {"artifact_bytes": total, "free_disk_bytes": free}


# object-model: resource_owner; representation=host; interop=relation-case-resources-v1
class RelationCaseResources:
    """拥有一次 Memory 事务、资源样本和失败记录；成功必须显式提交。"""

    __slots__ = ("root", "directory", "memory", "limits", "transaction", "initial_memory",
                 "samples", "check_count", "failed", "committed", "sample_bytes")

    def __init__(self, root: Path, directory: Path, memory, limits: RelationResourceLimits) -> None:
        """要求独占新 K 诊断目录和同 root 内的 Memory；不创建训练身份。"""
        self.root, self.directory = root.resolve(strict=True), directory.resolve()
        if (self.root.drive.upper() != "K:" or self.root.parent == Path("K:/")
                or self.directory == self.root or not self.directory.is_relative_to(self.root)
                or not memory.path.resolve(strict=True).is_relative_to(self.root)):
            raise ValueError("case resource scope requires one explicit named K root")
        self.memory, self.limits = memory, limits
        self.initial_memory = memory.path.stat().st_size
        ceiling = min(limits.max_memory_database_bytes,
                      self.initial_memory + limits.max_case_memory_growth_bytes)
        self.transaction = memory.backend.bounded_case(max_database_bytes=ceiling)
        self.samples: list[dict[str, object]] = []
        self.check_count = self.sample_bytes = 0
        self.failed = self.committed = False
        self.directory.mkdir(exist_ok=False)

    def _save(self, name: str, value: object) -> None:
        """失败证据只消费预留诊断空间，独占保存且不递归触发预算。"""
        raw = canonical_json_bytes(value) + b"\n"
        if self.sample_bytes + len(raw) > self.limits.diagnostic_reserve_bytes:
            raise RelationResourceBudgetError("resource diagnostic reserve exhausted")
        with (self.directory / name).open("xb") as stream:
            stream.write(raw)
        self.sample_bytes += len(raw)

    def checkpoint(self, label: str, *, retain: bool = True) -> None:
        """在实际进度边界读取完整 root、Memory 页数及当前/累计进程峰值。"""
        if self.failed:
            return  # 异常展开只允许已有诊断代码保留失败边界，不再进入业务。
        self.check_count += 1
        endpoint = current_process_resource_endpoint().to_value()
        usage = self.transaction.usage()
        memory_files = {suffix or "main": (self.memory.path.parent / (self.memory.path.name + suffix)).stat().st_size
                        if (self.memory.path.parent / (self.memory.path.name + suffix)).exists() else 0
                        for suffix in ("", "-journal", "-wal", "-shm")}
        total = _tree_bytes(self.root)
        row = {"ordinal": self.check_count, "progress_point": label, "artifact_bytes": total,
               "memory_files": memory_files, "initial_memory_bytes": self.initial_memory,
               "memory_main_growth_bytes": memory_files["main"] - self.initial_memory,
               "memory_logical_growth_bytes": usage["logical_database_bytes"] - self.initial_memory,
               "sqlite": usage, "resource_endpoint": endpoint,
               "free_disk_bytes": disk_usage(self.root).free}
        reasons = []
        if total > self.limits.max_artifact_bytes - self.limits.diagnostic_reserve_bytes:
            reasons.append("ARTIFACT_PROGRESS_LIMIT")
        if usage["logical_database_bytes"] > usage["hard_main_database_bytes"]:
            reasons.append("MEMORY_DATABASE_LIMIT")
        current, peak = endpoint["working_set_bytes"], endpoint["process_peak_working_set_bytes"]
        if current is None or peak is None:
            reasons.append("MEMORY_COUNTER_UNAVAILABLE")
        else:
            if current > self.limits.max_working_set_bytes:
                reasons.append("WORKING_SET_PROGRESS_LIMIT")
            if peak > self.limits.max_process_peak_working_set_bytes:
                reasons.append("CUMULATIVE_PEAK_PROGRESS_LIMIT")
        if row["free_disk_bytes"] < self.limits.minimum_free_disk_bytes:
            reasons.append("FREE_DISK_PROGRESS_LIMIT")
        if retain or reasons:
            if len(self.samples) >= 512:
                reasons.append("RESOURCE_SAMPLE_BUDGET")
            else:
                self.samples.append(row)
        if reasons:
            self.failed = True
            self._save("resource-stop.json", {"reasons": reasons, "last_sample": row,
                "restart_authorized": 0, "free_dialogue_complete": 0})
            raise RelationResourceBudgetError(",".join(reasons))

    def before_write(self, path: Path, final_byte_count: int) -> None:
        """在真实文件写入前核验最终长度；不把 SQLite journal 当作已覆盖硬限。"""
        if self.failed:
            return
        target = path.resolve()
        if (not target.is_relative_to(self.root) or type(final_byte_count) is not int
                or final_byte_count < 0):
            raise RelationResourceBudgetError("managed write escaped its declared root or byte budget")
        previous = target.stat().st_size if target.is_file() else 0
        projected = _tree_bytes(self.root) - previous + final_byte_count
        if projected > self.limits.max_artifact_bytes - self.limits.diagnostic_reserve_bytes:
            self.failed = True
            self._save("resource-stop.json", {"reasons": ["MANAGED_WRITE_PROJECTED_LIMIT"],
                "target": target.relative_to(self.root).as_posix(), "projected_bytes": projected,
                "restart_authorized": 0, "free_dialogue_complete": 0})
            raise RelationResourceBudgetError("MANAGED_WRITE_PROJECTED_LIMIT")

    def commit(self) -> None:
        """完整查询、交付及保存成功后提交唯一事务，先过当前进度门。"""
        from pure_integer_ai.experiments.relation_runtime_phase import relation_phase
        if self.failed or self.committed:
            raise RelationResourceBudgetError("failed or committed resource scope cannot commit")
        self.checkpoint("case_before_commit")
        with relation_phase("memory_commit", "RelationCaseResources.commit"):
            self.transaction.commit()
            self.committed = True
        self.checkpoint("case_after_commit")

    def finish(self, error: BaseException | None) -> None:
        """保留成功/失败及硬限实际作用域，不把诊断检查称为瞬时硬上界。"""
        self._save("receipt.json", {"status": "PASS" if error is None and self.committed else "FAIL",
            "failure_type": None if error is None else type(error).__name__,
            "limits": asdict(self.limits), "initial_memory_bytes": self.initial_memory,
            "hard_main_database_bytes": self.transaction.actual_bytes,
            "sqlite_hard_limit_scope": "MAIN_DATABASE_ONLY_EXCLUDES_JOURNAL_AND_TEMP",
            "artifact_limit_scope": "MANAGED_PREWRITE_AND_PROGRESS_CHECKS",
            "working_set_limit_scope": "PROGRESS_CHECKS_ONLY_NO_OS_HARD_LIMIT",
            "peak_scope": "PROCESS_CUMULATIVE_NOT_PHASE_PEAK", "process_poll_count": 0,
            "progress_check_count": self.check_count, "memory_committed": int(self.transaction.committed),
            "samples": self.samples, "restart_authorized": 0, "free_dialogue_complete": 0})


@contextmanager
def bounded_relation_resources(root: Path, directory: Path, memory,
                               limits: RelationResourceLimits | None) -> Iterator[RelationCaseResources | None]:
    """显式选择有界事务；默认路径保持既有提交合同。"""
    if limits is None:
        yield None
        return
    if _CURRENT.get() is not None:
        raise RuntimeError("case resource scopes cannot nest")
    owner = RelationCaseResources(root, directory, memory, limits)
    original = None
    token = _CURRENT.set(owner)
    try:
        with owner.transaction:
            owner.checkpoint("case_begin")
            yield owner
            if not owner.committed:
                raise RelationResourceBudgetError("case ended without an explicit complete commit")
    except BaseException as error:
        original = error
        try:
            memory.abandon()
        except BaseException as cleanup_error:
            error.add_note("Memory abandon failed: " + type(cleanup_error).__name__)
        raise
    finally:
        _CURRENT.reset(token)
        try:
            owner.finish(original)
        except BaseException as collection_error:
            if original is None:
                raise
            original.add_note("Resource receipt failed: " + type(collection_error).__name__)


def resource_checkpoint(label: str, *, retain: bool = True) -> None:
    """从真实调用点触发当前 owner；无 owner 时不产生观测或副作用。"""
    owner = _CURRENT.get()
    if owner is not None:
        owner.checkpoint(label, retain=retain)


def before_resource_write(path: Path, final_byte_count: int) -> None:
    """共享文件写入门；与 phase 及 trace 保存使用同一 ContextVar owner。"""
    owner = _CURRENT.get()
    if owner is not None:
        owner.before_write(path, final_byte_count)
