"""课程阶段报告的 append-only 持久化和 header-last 可见性。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    StorageBackend,
    register_extension_table,
)


CURRICULUM_STAGE_REPORT_TABLE = "curriculum_stage_report"
CURRICULUM_STAGE_REPORT_PART_TABLE = "curriculum_stage_report_part"

FAULT_CURRICULUM_REPORT_AFTER_PARTS = 1
FAULT_CURRICULUM_REPORT_AFTER_HEADER = 2

_REPORT_COLUMNS = [
    ("report_hash", TYPE_INT),
    ("stage_hash", TYPE_INT),
    ("version_hash", TYPE_INT),
    ("evaluator_hash", TYPE_INT),
    ("report_seq", TYPE_INT),
    ("passed", TYPE_INT),
    ("payload_count", TYPE_INT),
]
_REPORT_INDEXES = [
    ("report_hash",),
    ("stage_hash", "version_hash"),
    ("stage_hash", "version_hash", "report_seq"),
]
_PART_COLUMNS = [
    ("report_hash", TYPE_INT),
    ("ordinal", TYPE_INT),
    ("value", TYPE_INT),
]
_PART_INDEXES = [
    ("report_hash",),
    ("report_hash", "ordinal"),
]


class CurriculumMasteryIntegrityError(RuntimeError):
    """阶段报告出现半写、重复序号、hash 碰撞或 payload 漂移。"""


@runtime_checkable
class CurriculumReportFaultInjector(Protocol):
    """在阶段报告可见性边界注入故障的最小协议。"""

    def hit(self, point: int, context: dict[str, int]) -> None:
        """观察边界；需要中断时由实现直接抛出异常。"""
        ...


def _hit(
        injector: CurriculumReportFaultInjector | None,
        point: int,
        context: dict[str, int],
        ) -> None:
    """调用可选故障注入器并隔离可变上下文。"""
    if injector is None:
        return
    if not isinstance(injector, CurriculumReportFaultInjector):
        raise TypeError("curriculum report fault injector 协议错误")
    injector.hit(point, dict(context))


@dataclass(frozen=True)
class CurriculumStageReportRecord:
    """一个已完整封存的阶段评测报告物理记录。"""

    report_hash: int
    stage_hash: int
    version_hash: int
    evaluator_hash: int
    report_seq: int
    passed: int
    payload: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验索引、逻辑序、通过位和非空严格整数 payload。"""
        values = (
            self.report_hash,
            self.stage_hash,
            self.version_hash,
            self.evaluator_hash,
            self.report_seq,
            self.passed,
        )
        assert_int(*values, _where="CurriculumStageReportRecord")
        if (any(type(value) is not int for value in values)
                or min(values[:5]) <= 0
                or self.passed not in {0, 1}):
            raise ValueError("curriculum stage report 索引或通过位非法")
        if (not isinstance(self.payload, tuple) or not self.payload
                or any(type(value) is not int for value in self.payload)):
            raise ValueError("curriculum stage report payload 必须是非空严格整数 tuple")
        assert_int(*self.payload, _where="CurriculumStageReportRecord.payload")


def register_curriculum_mastery_tables(backend: StorageBackend) -> None:
    """注册阶段报告 header/part 表，header 是唯一正式可见点。"""
    register_extension_table(
        backend,
        CURRICULUM_STAGE_REPORT_TABLE,
        _REPORT_COLUMNS,
        discipline=disc.DISC_APPEND_ONLY,
        indexes=_REPORT_INDEXES,
        recovery_key=("report_hash",),
    )
    register_extension_table(
        backend,
        CURRICULUM_STAGE_REPORT_PART_TABLE,
        _PART_COLUMNS,
        discipline=disc.DISC_APPEND_ONLY,
        indexes=_PART_INDEXES,
        recovery_key=("report_hash", "ordinal"),
    )


class CurriculumMasteryStore:
    """以 parts-first/header-last 协议持久化和恢复阶段报告。"""

    def __init__(self, backend: StorageBackend) -> None:
        """绑定已注册课程报告表的后端。"""
        self.backend = backend

    def next_sequence(self, stage_hash: int, version_hash: int) -> int:
        """返回指定阶段和版本的下一确定性报告逻辑序。"""
        self._require_hash(stage_hash, "stage_hash")
        self._require_hash(version_hash, "version_hash")
        rows = self.backend.select(
            CURRICULUM_STAGE_REPORT_TABLE,
            where={
                "stage_hash": stage_hash,
                "version_hash": version_hash,
            },
            order_by="report_seq",
            descending=True,
            limit=1,
        )
        return 1 if not rows else rows[0]["report_seq"] + 1

    def append(
            self,
            record: CurriculumStageReportRecord,
            *,
            fault_injector: CurriculumReportFaultInjector | None = None,
            ) -> CurriculumStageReportRecord:
        """幂等写入完整 payload parts，最后追加 header 发布正式报告。"""
        if not isinstance(record, CurriculumStageReportRecord):
            raise TypeError("record 必须是 CurriculumStageReportRecord")
        existing = self.optional(record.report_hash)
        if existing is not None:
            if existing != record:
                raise CurriculumMasteryIntegrityError(
                    "同一 report_hash 命中不同阶段报告")
            return existing
        sequence_rows = self.backend.select(
            CURRICULUM_STAGE_REPORT_TABLE,
            where={
                "stage_hash": record.stage_hash,
                "version_hash": record.version_hash,
                "report_seq": record.report_seq,
            },
        )
        if sequence_rows:
            raise CurriculumMasteryIntegrityError(
                "同一阶段版本的 report_seq 已被其他报告占用")
        expected_parts = tuple(
            {
                "report_hash": record.report_hash,
                "ordinal": ordinal,
                "value": value,
            }
            for ordinal, value in enumerate(record.payload)
        )
        persisted_parts = self.backend.select(
            CURRICULUM_STAGE_REPORT_PART_TABLE,
            where={"report_hash": record.report_hash},
            order_by="ordinal",
        )
        persisted_by_ordinal = {}
        for row in persisted_parts:
            ordinal = row["ordinal"]
            if ordinal in persisted_by_ordinal:
                raise CurriculumMasteryIntegrityError(
                    "阶段报告 part ordinal 重复")
            persisted_by_ordinal[ordinal] = row
        if set(persisted_by_ordinal) - set(range(len(expected_parts))):
            raise CurriculumMasteryIntegrityError(
                "阶段报告存在超出 payload 的孤立 part")
        for part in expected_parts:
            previous = persisted_by_ordinal.get(part["ordinal"])
            if previous is not None:
                if previous != part:
                    raise CurriculumMasteryIntegrityError(
                        "阶段报告孤立 part 与重试 payload 漂移")
                continue
            self.backend.insert(CURRICULUM_STAGE_REPORT_PART_TABLE, part)
        _hit(fault_injector, FAULT_CURRICULUM_REPORT_AFTER_PARTS, {
            "report_hash": record.report_hash,
            "report_seq": record.report_seq,
        })
        self.backend.insert(CURRICULUM_STAGE_REPORT_TABLE, {
            "report_hash": record.report_hash,
            "stage_hash": record.stage_hash,
            "version_hash": record.version_hash,
            "evaluator_hash": record.evaluator_hash,
            "report_seq": record.report_seq,
            "passed": record.passed,
            "payload_count": len(record.payload),
        })
        _hit(fault_injector, FAULT_CURRICULUM_REPORT_AFTER_HEADER, {
            "report_hash": record.report_hash,
            "report_seq": record.report_seq,
        })
        return self.read(record.report_hash)

    def optional(self, report_hash: int) -> CurriculumStageReportRecord | None:
        """按报告 hash 读取正式 header；孤立 parts 不构成可见报告。"""
        self._require_hash(report_hash, "report_hash")
        rows = self.backend.select(
            CURRICULUM_STAGE_REPORT_TABLE,
            where={"report_hash": report_hash},
        )
        if len(rows) > 1:
            raise CurriculumMasteryIntegrityError("阶段报告 header 重复")
        return None if not rows else self._restore(rows[0])

    def read(self, report_hash: int) -> CurriculumStageReportRecord:
        """按报告 hash 回读完整 header 和连续 payload parts。"""
        record = self.optional(report_hash)
        if record is None:
            raise CurriculumMasteryIntegrityError("阶段报告不存在或尚未发布 header")
        return record

    def latest(
            self,
            stage_hash: int,
            version_hash: int,
            ) -> CurriculumStageReportRecord | None:
        """返回指定阶段和版本的最新正式报告。"""
        self._require_hash(stage_hash, "stage_hash")
        self._require_hash(version_hash, "version_hash")
        rows = self.backend.select(
            CURRICULUM_STAGE_REPORT_TABLE,
            where={
                "stage_hash": stage_hash,
                "version_hash": version_hash,
            },
            order_by="report_seq",
            descending=True,
            limit=1,
        )
        return None if not rows else self._restore(rows[0])

    def all_reports(self) -> tuple[CurriculumStageReportRecord, ...]:
        """按阶段、版本和逻辑序返回全部正式报告。"""
        records = tuple(
            self._restore(row)
            for row in self.backend.select(
                CURRICULUM_STAGE_REPORT_TABLE,
                where=None,
            )
        )
        return tuple(sorted(records, key=lambda item: (
            item.stage_hash, item.version_hash, item.report_seq,
            item.report_hash,
        )))

    def _restore(self, row) -> CurriculumStageReportRecord:
        """从唯一 header 和连续 part 行恢复规范物理记录。"""
        required = {
            "report_hash", "stage_hash", "version_hash", "evaluator_hash",
            "report_seq", "passed", "payload_count",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise CurriculumMasteryIntegrityError("阶段报告 header 字段漂移")
        values = tuple(row[name] for name in required)
        if any(type(value) is not int for value in values):
            raise CurriculumMasteryIntegrityError("阶段报告 header 必须是严格整数")
        if row["payload_count"] <= 0:
            raise CurriculumMasteryIntegrityError("阶段报告 payload_count 必须为正")
        report_hash = row["report_hash"]
        parts = self.backend.select(
            CURRICULUM_STAGE_REPORT_PART_TABLE,
            where={"report_hash": report_hash},
            order_by="ordinal",
        )
        expected_ordinals = tuple(range(row["payload_count"]))
        actual_ordinals = tuple(part["ordinal"] for part in parts)
        if actual_ordinals != expected_ordinals:
            raise CurriculumMasteryIntegrityError(
                "阶段报告 payload parts 缺失、重复或不连续")
        return CurriculumStageReportRecord(
            report_hash,
            row["stage_hash"],
            row["version_hash"],
            row["evaluator_hash"],
            row["report_seq"],
            row["passed"],
            tuple(part["value"] for part in parts),
        )

    @staticmethod
    def _require_hash(value: int, label: str) -> None:
        """核验持久层索引为严格正整数。"""
        assert_int(value, _where=f"CurriculumMasteryStore.{label}")
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} 必须是正严格整数")


__all__ = [
    "CURRICULUM_STAGE_REPORT_PART_TABLE",
    "CURRICULUM_STAGE_REPORT_TABLE",
    "FAULT_CURRICULUM_REPORT_AFTER_HEADER",
    "FAULT_CURRICULUM_REPORT_AFTER_PARTS",
    "CurriculumMasteryIntegrityError",
    "CurriculumMasteryStore",
    "CurriculumReportFaultInjector",
    "CurriculumStageReportRecord",
    "register_curriculum_mastery_tables",
]
