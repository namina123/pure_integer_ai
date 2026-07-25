"""PW-00 dry-run 启动清单、入口请求和运行报告协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


POST_WEANING_OPERATION_COMMITTED = 1
POST_WEANING_OPERATION_FAILED = 2


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验稳定键为非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给嵌套稳定键增加长度边界。"""
    return len(value), *value


def _instruction(value: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """要求入口和设施维度均由一等最小指令标识。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return value


@dataclass(frozen=True)
class PostWeaningRouteProtocol:
    """把四类 runtime 入口绑定到调用方注入的一等指令。"""

    reading: ObjectIdentity
    interaction: ObjectIdentity
    external_define: ObjectIdentity
    question: ObjectIdentity

    def __post_init__(self) -> None:
        """核验四类入口是互异且完整的最小指令。"""
        values = tuple(
            _instruction(item, label="post-weaning route")
            for item in (
                self.reading,
                self.interaction,
                self.external_define,
                self.question,
            )
        )
        keys = tuple(item.stable_key() for item in values)
        if len(set(keys)) != len(keys):
            raise ValueError("post-weaning 四类入口指令必须互异")

    def stable_key(self) -> tuple[int, ...]:
        """返回四类入口的长度分帧稳定身份。"""
        result = [1]
        for item in (
                self.reading,
                self.interaction,
                self.external_define,
                self.question):
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True, order=True)
class PostWeaningFacilityCheck:
    """一个来源化设施维度的通过状态和实际状态摘要。"""

    requirement: ObjectIdentity
    passed: bool
    state_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝无身份、非严格布尔或空状态摘要。"""
        _instruction(self.requirement, label="facility requirement")
        if type(self.passed) is not bool:
            raise TypeError("facility passed 必须是严格 bool")
        _strict_key(self.state_key, label="facility state_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回设施维度、判定和状态摘要。"""
        return (
            *_packed(self.requirement.stable_key()),
            1 if self.passed else 0,
            *_packed(self.state_key),
        )


@dataclass(frozen=True)
class PostWeaningFacilityProbe:
    """启动前由独立设施探针提交的逐维结果，不合成 readiness。"""

    checks: tuple[PostWeaningFacilityCheck, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求探针非空、维度唯一且按完整身份稳定排序。"""
        if (not isinstance(self.checks, tuple) or not self.checks
                or any(not isinstance(item, PostWeaningFacilityCheck)
                       for item in self.checks)):
            raise TypeError("facility checks 必须是非空 typed tuple")
        keys = tuple(item.requirement.stable_key() for item in self.checks)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("facility checks 必须唯一且稳定有序")
        _strict_key(self.trace, label="facility probe trace")

    @property
    def complete(self) -> bool:
        """只有全部独立维度通过时才允许 PH1 dry-run 启动。"""
        return all(item.passed for item in self.checks)

    def stable_key(self) -> tuple[int, ...]:
        """返回全部逐维结果和探针 trace。"""
        result = [1, len(self.checks)]
        for item in self.checks:
            result.extend(_packed(item.stable_key()))
        result.extend(_packed(self.trace))
        return tuple(result)


@dataclass(frozen=True)
class PostWeaningResourceBudget:
    """单个 dry-run 实例允许的调用数和逐调用持久状态增长上限。"""

    operation_limit: int
    memory_event_growth: int
    source_record_growth: int
    companion_item_growth: int

    def __post_init__(self) -> None:
        """要求全部预算为正严格整数，避免零预算伪启动。"""
        values = (
            self.operation_limit,
            self.memory_event_growth,
            self.source_record_growth,
            self.companion_item_growth,
        )
        assert_int(*values, _where="post-weaning resource budget")
        if any(type(item) is not int or item <= 0 for item in values):
            raise ValueError("post-weaning 资源预算必须是正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回全部资源上限。"""
        return (
            self.operation_limit,
            self.memory_event_growth,
            self.source_record_growth,
            self.companion_item_growth,
        )


@dataclass(frozen=True)
class PostWeaningDryRunManifest:
    """绑定一个隔离 fixture 与当前全部承重 owner 的不可变启动清单。"""

    runtime_owner: ObjectIdentity
    fixture_artifact_key: tuple[int, ...]
    core_state_key: tuple[int, ...]
    schema_state_key: tuple[int, ...]
    backend_state_key: tuple[int, ...]
    component_state_key: tuple[int, ...]
    routes: PostWeaningRouteProtocol
    probe: PostWeaningFacilityProbe
    budget: PostWeaningResourceBudget
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 fixture、版本、owner、探针和预算均被实例清单完整绑定。"""
        _instruction(self.runtime_owner, label="post-weaning runtime owner")
        for label, value in (
                ("fixture artifact", self.fixture_artifact_key),
                ("core state", self.core_state_key),
                ("schema state", self.schema_state_key),
                ("backend state", self.backend_state_key),
                ("component state", self.component_state_key),
                ("manifest trace", self.trace)):
            _strict_key(value, label=label)
        if not isinstance(self.routes, PostWeaningRouteProtocol):
            raise TypeError("manifest routes 类型错误")
        if not isinstance(self.probe, PostWeaningFacilityProbe):
            raise TypeError("manifest probe 类型错误")
        if not isinstance(self.budget, PostWeaningResourceBudget):
            raise TypeError("manifest budget 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回隔离 artifact、承重设施与预算的完整启动身份。"""
        result = [1]
        for value in (
                self.runtime_owner.stable_key(),
                self.fixture_artifact_key,
                self.core_state_key,
                self.schema_state_key,
                self.backend_state_key,
                self.component_state_key,
                self.routes.stable_key(),
                self.probe.stable_key(),
                self.budget.stable_key(),
                self.trace):
            result.extend(_packed(value))
        return tuple(result)


@dataclass(frozen=True)
class PostWeaningIntakeRequest:
    """阅读、交互或 define 入口的一次来源化摄入请求。"""

    route_kind: ObjectIdentity
    source: SourceRef
    raw_text: str
    license_id: str
    batch_id: int
    parser: Any = None
    supersedes_source: SourceRef | None = None
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验来源、许可、批次、入口和调用 trace，不解释文本语义。"""
        _instruction(self.route_kind, label="post-weaning intake route")
        if not isinstance(self.source, SourceRef):
            raise TypeError("post-weaning intake source 类型错误")
        if not isinstance(self.raw_text, str):
            raise TypeError("post-weaning raw_text 必须是字符串")
        if not isinstance(self.license_id, str) or not self.license_id:
            raise ValueError("post-weaning intake 必须提供非空许可")
        assert_int(self.batch_id, _where="post-weaning intake batch")
        if type(self.batch_id) is not int or self.batch_id < 0:
            raise ValueError("post-weaning batch_id 必须是非负严格整数")
        if (self.supersedes_source is not None
                and not isinstance(self.supersedes_source, SourceRef)):
            raise TypeError("supersedes_source 类型错误")
        if not isinstance(self.trace, tuple):
            raise TypeError("post-weaning intake trace 必须是 tuple")
        if self.trace:
            _strict_key(self.trace, label="post-weaning intake trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回不包含原文和 parser 实例的来源化请求身份。"""
        supersedes = (() if self.supersedes_source is None
                      else self.supersedes_source.stable_key())
        return (
            *_packed(self.route_kind.stable_key()),
            *_packed(self.source.stable_key()),
            self.batch_id,
            *_packed(supersedes),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class PostWeaningStateSnapshot:
    """一次调用边界的 Core 摘要和 Memory/Companion 物理计数。"""

    core_state_key: tuple[int, ...]
    memory_event_count: int
    source_record_count: int
    companion_item_count: int

    def __post_init__(self) -> None:
        """要求状态摘要非空且三个物理计数为非负严格整数。"""
        _strict_key(self.core_state_key, label="post-weaning core snapshot")
        values = (
            self.memory_event_count,
            self.source_record_count,
            self.companion_item_count,
        )
        assert_int(*values, _where="post-weaning state counters")
        if any(type(item) is not int or item < 0 for item in values):
            raise ValueError("post-weaning 状态计数必须是非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回 Core 摘要和全部物理计数。"""
        return (
            *_packed(self.core_state_key),
            self.memory_event_count,
            self.source_record_count,
            self.companion_item_count,
        )


@dataclass(frozen=True)
class PostWeaningOperationReport:
    """一次已提交或已失败 dry-run 调用的边界证据。"""

    ordinal: int
    request_key: tuple[int, ...]
    route_kind: ObjectIdentity
    status: int
    result_key: tuple[int, ...]
    before: PostWeaningStateSnapshot
    after: PostWeaningStateSnapshot
    query_closed: bool
    physical_metrics_key: tuple[int, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验报告状态、序号、快照和 query 关闭证明。"""
        assert_int(self.ordinal, self.status, _where="post-weaning report")
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ValueError("post-weaning report ordinal 必须为正")
        if self.status not in {
                POST_WEANING_OPERATION_COMMITTED,
                POST_WEANING_OPERATION_FAILED}:
            raise ValueError("post-weaning report status 未注册")
        _strict_key(self.request_key, label="post-weaning report request")
        _instruction(self.route_kind, label="post-weaning report route")
        if not isinstance(self.result_key, tuple):
            raise TypeError("post-weaning result_key 必须是 tuple")
        if self.result_key:
            _strict_key(self.result_key, label="post-weaning result_key")
        if (not isinstance(self.before, PostWeaningStateSnapshot)
                or not isinstance(self.after, PostWeaningStateSnapshot)):
            raise TypeError("post-weaning report snapshot 类型错误")
        if type(self.query_closed) is not bool:
            raise TypeError("post-weaning query_closed 必须是严格 bool")
        if not isinstance(self.physical_metrics_key, tuple):
            raise TypeError("physical_metrics_key 必须是 tuple")
        if self.physical_metrics_key:
            _strict_key(
                self.physical_metrics_key,
                label="post-weaning physical metrics",
            )
        _strict_key(self.trace, label="post-weaning report trace")

    @property
    def core_unchanged(self) -> bool:
        """判断本次调用前后 Core canonical 摘要是否完全一致。"""
        return self.before.core_state_key == self.after.core_state_key

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、结果、状态增量和关闭证明的完整报告身份。"""
        result = [
            self.ordinal,
            *_packed(self.request_key),
            *_packed(self.route_kind.stable_key()),
            self.status,
            *_packed(self.result_key),
            *_packed(self.before.stable_key()),
            *_packed(self.after.stable_key()),
            1 if self.query_closed else 0,
            *_packed(self.physical_metrics_key),
            *_packed(self.trace),
        ]
        return tuple(result)


@dataclass(frozen=True)
class PostWeaningOperationRun:
    """把领域结果与同次 PW-00 操作报告成对返回。"""

    result: Any
    report: PostWeaningOperationReport

    def __post_init__(self) -> None:
        """要求成功调用必须携带已提交报告。"""
        if not isinstance(self.report, PostWeaningOperationReport):
            raise TypeError("post-weaning operation report 类型错误")
        if self.report.status != POST_WEANING_OPERATION_COMMITTED:
            raise ValueError("PostWeaningOperationRun 只能携带已提交报告")


__all__ = [
    "POST_WEANING_OPERATION_COMMITTED",
    "POST_WEANING_OPERATION_FAILED",
    "PostWeaningDryRunManifest",
    "PostWeaningFacilityCheck",
    "PostWeaningFacilityProbe",
    "PostWeaningIntakeRequest",
    "PostWeaningOperationReport",
    "PostWeaningOperationRun",
    "PostWeaningResourceBudget",
    "PostWeaningRouteProtocol",
    "PostWeaningStateSnapshot",
]
