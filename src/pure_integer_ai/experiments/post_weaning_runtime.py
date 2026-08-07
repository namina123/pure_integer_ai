"""PW-00 独立 dry-run runtime、启动闸和来源摄入入口。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass

from pure_integer_ai.cognition.shared.memory_batch import MemoryBatchRuntime
from pure_integer_ai.cognition.shared.post_weaning import (
    POST_WEANING_OPERATION_COMMITTED,
    POST_WEANING_OPERATION_FAILED,
    PostWeaningDryRunManifest,
    PostWeaningFacilityProbe,
    PostWeaningIntakeRequest,
    PostWeaningOperationReport,
    PostWeaningOperationRun,
    PostWeaningResourceBudget,
    PostWeaningRouteProtocol,
    PostWeaningStateSnapshot,
)
from pure_integer_ai.cognition.shared.formal_post_weaning import (
    FormalPostWeaningManifest,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.types import WEANING_PRE
from pure_integer_ai.cognition.understanding.memory_intake import (
    MemoryIntakeResult,
    MemorySourceIntake,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.attractor_runtime import AttractorRuntime
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    MemoryHotSetRuntime,
)
from pure_integer_ai.experiments.memory_generation_runtime import (
    MemoryAwareQuestionDialogueRuntime,
    MemoryQuestionDialogueRun,
)
from pure_integer_ai.experiments.memory_query_runtime import MemoryQueryRuntime
from pure_integer_ai.experiments.memory_resolver_runtime import (
    MemoryResolverRuntime,
)
from pure_integer_ai.experiments.memory_use_runtime import MemoryUseRuntime
from pure_integer_ai.experiments.source_trust_runtime import (
    SourceAdmissionRuntime,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.backend_capability import capability_profile
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE
from pure_integer_ai.storage.source_record import (
    SOURCE_RECORD_TABLE,
    SourceRecordStorage,
)
from pure_integer_ai.storage.spaces.companion import TEXT_ASSOC_TABLE
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.spaces.registry import SpaceRegistry


class PostWeaningStartupError(RuntimeError):
    """PW-00 启动清单、设施 owner 或正式状态闸不满足。"""


class PostWeaningRuntimeError(RuntimeError):
    """PW-00 调用污染 Core、击穿预算或泄漏 query 资源。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给稳定键增加长度边界。"""
    return len(value), *value


def _canonical_key(value: object) -> tuple[int, ...]:
    """把只含稳定结构的数据规范编码为 SHA-256 整数摘要。"""
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return tuple(hashlib.sha256(payload).digest())


def _dataclass_key(value: object) -> tuple[int, ...]:
    """递归提取配置 dataclass 的稳定键，不接受自由对象 repr。"""
    stable_key = getattr(value, "stable_key", None)
    if callable(stable_key):
        key = stable_key()
        if (not isinstance(key, tuple) or not key
                or any(type(item) is not int for item in key)):
            raise PostWeaningStartupError("组件 stable_key 非法")
        return key
    if is_dataclass(value):
        result = [1, len(fields(value))]
        for item in fields(value):
            current = getattr(value, item.name)
            if type(current) is int:
                result.extend((1, current))
            elif isinstance(current, tuple):
                if any(type(part) is not int for part in current):
                    raise PostWeaningStartupError("配置 tuple 含非整数")
                result.extend((2, len(current), *current))
            else:
                key = _dataclass_key(current)
                result.extend((3, *_packed(key)))
        return tuple(result)
    raise PostWeaningStartupError("组件缺少稳定状态协议")


class CoreCanonicalStateReader:
    """读取所有显式指向 Core space 的持久行并形成 canonical 摘要。"""

    def __init__(self, ctx: TrainContext) -> None:
        """绑定一个上下文的 Core 稳定空间身份。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("Core state reader ctx 类型错误")
        self._backend = ctx.backend
        self._core_space_id = ctx.core_space.space_id
        self._core_identity = SpaceRegistry(ctx.backend).identity(
            self._core_space_id)

    def state_key(self) -> tuple[int, ...]:
        """返回 reader 版本和 Core 稳定空间身份。"""
        return 1, *self._core_identity.stable_key()

    def read(self) -> tuple[int, ...]:
        """摘要显式指向 Core owner 的行。"""
        schema = self._backend.schema_snapshot()
        selected: dict[str, list[dict[str, object]]] = {}
        for table in sorted(schema):
            columns = tuple(schema[table]["columns"])
            owner_columns = tuple(
                column for column in columns
                if (column == "space_id"
                    or column.startswith("space_id_")
                    or column.endswith("_space_id"))
            )
            if not owner_columns:
                continue
            rows = self._backend.select(table, where=None)
            owned = [
                row for row in rows
                if any(row.get(column) == self._core_space_id
                       for column in owner_columns)
            ]
            if owned:
                selected[table] = sorted(
                    owned,
                    key=lambda row: json.dumps(
                        row,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
        return _canonical_key(selected)


def _schema_state_key(ctx: TrainContext) -> tuple[int, ...]:
    """返回当前 backend 完整注册 schema 的规范摘要。"""
    return _canonical_key(ctx.backend.schema_snapshot())


def _batch_config_key(ctx: TrainContext) -> tuple[int, ...]:
    """返回 M-10/K-02 批次配置的稳定状态。"""
    config = ctx.memory_batch_config
    if config is None:
        raise PostWeaningStartupError("PW-00 缺少 M-10 batch config")
    return _dataclass_key(config)


def post_weaning_component_state_key(ctx: TrainContext) -> tuple[int, ...]:
    """交叉核验全部承重 runtime 的真实类型、同 owner 绑定和稳定状态。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("post-weaning component ctx 类型错误")
    read_intake = ctx.memory_read_intake
    interact_intake = ctx.memory_interact_intake
    if (not isinstance(read_intake, MemorySourceIntake)
            or not isinstance(interact_intake, MemorySourceIntake)):
        raise PostWeaningStartupError("PW-00 缺少成对 M-05 intake")
    if read_intake.source_intake is not interact_intake.source_intake:
        raise PostWeaningStartupError("双层 M-05 未共享同一 Companion 来源 owner")
    if (not isinstance(ctx.memory_read_batch_runtime, MemoryBatchRuntime)
            or not isinstance(
                ctx.memory_interact_batch_runtime, MemoryBatchRuntime)):
        raise PostWeaningStartupError("PW-00 缺少成对 M-10 runtime")
    if (read_intake.batch_runtime is not ctx.memory_read_batch_runtime
            or interact_intake.batch_runtime
            is not ctx.memory_interact_batch_runtime):
        raise PostWeaningStartupError("M-05 未绑定当前 M-10 runtime")
    source_trust = ctx.source_trust_runtime
    if (not isinstance(source_trust, SourceAdmissionRuntime)
            or source_trust.ctx is not ctx):
        raise PostWeaningStartupError("PW-00 缺少当前上下文的 A-05 来源准入")
    components = (
        (1, ctx.memory_query_runtime, MemoryQueryRuntime),
        (2, ctx.memory_resolver_runtime, MemoryResolverRuntime),
        (3, ctx.attractor_runtime, AttractorRuntime),
        (4, ctx.memory_use_runtime, MemoryUseRuntime),
        (5, ctx.memory_hot_set_runtime, MemoryHotSetRuntime),
        (6, source_trust, SourceAdmissionRuntime),
    )
    result = [2]
    for tag, component, expected in components:
        if not isinstance(component, expected):
            raise PostWeaningStartupError(
                f"PW-00 缺少承重 runtime {expected.__name__}")
        result.extend((tag, *_packed(component.state_key())))
    companion = read_intake.source_intake.companion
    if companion is not interact_intake.source_intake.companion:
        raise PostWeaningStartupError("双层 M-05 Companion 实例漂移")
    registry = SpaceRegistry(ctx.backend)
    for tag, space_id in (
            (7, ctx.core_space.space_id),
            (8, ctx.memory_read.space_id),
            (9, ctx.memory_interact.space_id),
            (10, companion.space_id)):
        result.extend((tag, *_packed(registry.identity(space_id).stable_key())))
    result.extend((11, *_packed(_batch_config_key(ctx))))
    return tuple(result)


def _validate_resource_budget(
        ctx: TrainContext,
        budget: PostWeaningResourceBudget,
        ) -> None:
    """要求 PW-00 预算覆盖已绑定 M-10 单元和单个来源留档的最大增长。"""
    runtimes = (
        ctx.memory_read_batch_runtime,
        ctx.memory_interact_batch_runtime,
    )
    if any(not isinstance(item, MemoryBatchRuntime) for item in runtimes):
        raise PostWeaningStartupError("PW-00 预算校验缺少成对 M-10 runtime")
    event_limit = max(item.write_budget.object_limit for item in runtimes)
    if budget.memory_event_growth < event_limit:
        raise PostWeaningStartupError(
            "PW-00 Memory 增长预算小于 M-10 单元对象上限")
    if budget.source_record_growth < 1 or budget.companion_item_growth < 1:
        raise PostWeaningStartupError(
            "PW-00 来源和 Companion 增长预算必须覆盖单次留档")


def build_post_weaning_dry_run_manifest(
        ctx: TrainContext,
        *,
        runtime_owner,
        fixture_artifact_key: tuple[int, ...],
        routes: PostWeaningRouteProtocol,
        probe: PostWeaningFacilityProbe,
        budget: PostWeaningResourceBudget,
        trace: tuple[int, ...],
        core_reader: CoreCanonicalStateReader | None = None,
        ) -> PostWeaningDryRunManifest:
    """从当前实际 owner 捕获一个不可变 PH1 fixture 启动清单。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("build dry-run manifest ctx 类型错误")
    reader = core_reader or CoreCanonicalStateReader(ctx)
    if not isinstance(reader, CoreCanonicalStateReader):
        raise TypeError("core_reader 类型错误")
    return PostWeaningDryRunManifest(
        runtime_owner,
        fixture_artifact_key,
        reader.read(),
        _schema_state_key(ctx),
        capability_profile(ctx.backend).stable_key(),
        post_weaning_component_state_key(ctx),
        routes,
        probe,
        budget,
        trace,
    )


# object-model: lifecycle; owner=post-weaning-context; cleanup=backend-close
class PostWeaningOperationRuntime:
    """dry-run 与正式状态共用的入口、回滚、预算和 Core 边界。"""

    def __init__(
            self,
            ctx: TrainContext,
            manifest: PostWeaningDryRunManifest | FormalPostWeaningManifest,
            *,
            core_reader: CoreCanonicalStateReader | None = None,
            required_phase: int = WEANING_PRE,
            ) -> None:
        """核验状态、实际设施 owner、artifact 清单和全部探针后绑定。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("post-weaning runtime ctx 类型错误")
        if not isinstance(
                manifest,
                (PostWeaningDryRunManifest, FormalPostWeaningManifest)):
            raise TypeError("post-weaning manifest 类型错误")
        if ctx.teacher is not None:
            raise PostWeaningStartupError("post-weaning runtime 禁止安装教师")
        if ctx.weaning_phase != required_phase:
            raise PostWeaningStartupError("post-weaning runtime 阶段不匹配")
        if (ctx.work_memory.active_query_scope is not None
                or ctx.work_memory.active_generation_scope is not None
                or ctx.work_memory.attractor_state is not None):
            raise PostWeaningStartupError("PW-00 启动前存在未关闭 query 资源")
        reader = core_reader or CoreCanonicalStateReader(ctx)
        if not isinstance(reader, CoreCanonicalStateReader):
            raise TypeError("core_reader 类型错误")
        actual = (
            reader.read(),
            _schema_state_key(ctx),
            capability_profile(ctx.backend).stable_key(),
            post_weaning_component_state_key(ctx),
        )
        expected = (
            manifest.core_state_key,
            manifest.schema_state_key,
            manifest.backend_state_key,
            manifest.component_state_key,
        )
        if actual != expected:
            raise PostWeaningStartupError("PW-00 manifest 与当前设施状态漂移")
        if not manifest.probe.complete:
            raise PostWeaningStartupError("PW-00 设施探针存在未通过维度")
        _validate_resource_budget(ctx, manifest.budget)
        self.ctx = ctx
        self.manifest = manifest
        self.core_reader = reader
        self._reports: list[PostWeaningOperationReport] = []

    def reports(self) -> tuple[PostWeaningOperationReport, ...]:
        """按调用顺序返回不可变操作报告。"""
        return tuple(self._reports)

    def run_intake(
            self,
            request: PostWeaningIntakeRequest,
            ) -> PostWeaningOperationRun:
        """分派阅读、交互或 define，并核验 Core、预算和失败清理边界。"""
        if not isinstance(request, PostWeaningIntakeRequest):
            raise TypeError("post-weaning intake request 类型错误")
        self._require_operation_budget()
        before = self._snapshot()
        route = request.route_kind
        recovery_state = self.ctx.backend.recovery_state_snapshot()
        try:
            if route == self.manifest.routes.reading:
                result = self.ctx.source_trust_runtime.ingest_batch(
                    (request,), cursor_commit=lambda: None).results[0]
            elif route == self.manifest.routes.interaction:
                result = self.ctx.source_trust_runtime.ingest_batch(
                    (request,), cursor_commit=lambda: None).results[0]
            elif route == self.manifest.routes.external_define:
                result = self.ctx.source_trust_runtime.admit_record_only(request)
                self.ctx.backend.commit()
            else:
                raise ValueError("intake route 不属于阅读、交互或 external define")
            after = self._snapshot()
            report = self._report(
                request.stable_key(),
                request.route_kind,
                POST_WEANING_OPERATION_COMMITTED,
                self._result_key(result),
                before,
                after,
            )
            self._validate_committed(report)
            self._reports.append(report)
            return PostWeaningOperationRun(result, report)
        except BaseException as exc:
            self._close_query_resources()
            self.ctx.backend.restore_recovery_state(recovery_state)
            self.ctx.backend.commit()
            self._restore_runtime_state()
            after = self._snapshot()
            report = self._report(
                request.stable_key(),
                request.route_kind,
                POST_WEANING_OPERATION_FAILED,
                (),
                before,
                after,
                failure=exc,
            )
            self._reports.append(report)
            if not report.core_unchanged:
                raise PostWeaningRuntimeError(
                    "失败入口污染 Core 且无法通过") from exc
            raise

    def run_question(
            self,
            dialogue: MemoryAwareQuestionDialogueRuntime,
            request: QuestionRequest,
            ) -> PostWeaningOperationRun:
        """调用同上下文 J-G dialogue，并记录 Use、冷页指标和完整资源关闭状态。"""
        if not isinstance(dialogue, MemoryAwareQuestionDialogueRuntime):
            raise TypeError("PW-00 question dialogue 类型错误")
        if dialogue.context is not self.ctx:
            raise ValueError("PW-00 question dialogue 属于其他上下文")
        if not isinstance(request, QuestionRequest):
            raise TypeError("PW-00 question request 类型错误")
        self._require_operation_budget()
        before = self._snapshot()
        recovery_state = self.ctx.backend.recovery_state_snapshot()
        try:
            result = dialogue.run(request)
            after = self._snapshot()
            report = self._report(
                request.stable_key(),
                self.manifest.routes.question,
                POST_WEANING_OPERATION_COMMITTED,
                result.stable_key(),
                before,
                after,
            )
            self._validate_committed(report)
            self._reports.append(report)
            return PostWeaningOperationRun(result, report)
        except BaseException as exc:
            self._close_query_resources()
            self.ctx.backend.restore_recovery_state(recovery_state)
            self.ctx.backend.commit()
            self._restore_runtime_state()
            after = self._snapshot()
            report = self._report(
                request.stable_key(),
                self.manifest.routes.question,
                POST_WEANING_OPERATION_FAILED,
                (),
                before,
                after,
                failure=exc,
            )
            self._reports.append(report)
            if not report.core_unchanged:
                raise PostWeaningRuntimeError(
                    "失败 question 污染 Core 且无法通过") from exc
            raise

    def _snapshot(self) -> PostWeaningStateSnapshot:
        """读取 Core 摘要、双层 Memory event、SourceRecord 和 Companion 计数。"""
        memory_count = sum(
            self.ctx.backend.count(
                MEMORY_EVENT_TABLE,
                where={"space_id": space.space_id},
            )
            for space in (self.ctx.memory_read, self.ctx.memory_interact)
        )
        companion = self.ctx.memory_read_intake.source_intake.companion
        return PostWeaningStateSnapshot(
            self.core_reader.read(),
            memory_count,
            self.ctx.backend.count(SOURCE_RECORD_TABLE, where=None),
            self.ctx.backend.count(
                TEXT_ASSOC_TABLE,
                where={"space_id": companion.space_id},
            ),
        )

    def _report(
            self,
            request_key: tuple[int, ...],
            route_kind: ObjectIdentity,
            status: int,
            result_key: tuple[int, ...],
            before: PostWeaningStateSnapshot,
            after: PostWeaningStateSnapshot,
            *,
            failure: BaseException | None = None,
            ) -> PostWeaningOperationReport:
        """形成不含异常文本的稳定操作报告，并证明 query 资源已关闭。"""
        ordinal = len(self._reports) + 1
        failure_key = ()
        if failure is not None:
            failure_key = (
                Hasher("post_weaning.failure.v1").h63(
                    failure.__class__.__module__),
                Hasher("post_weaning.failure.v1").h63(
                    failure.__class__.__qualname__),
            )
        trace = (
            *self.manifest.trace,
            ordinal,
            status,
            *_packed(failure_key),
        )
        return PostWeaningOperationReport(
            ordinal,
            request_key,
            route_kind,
            status,
            result_key,
            before,
            after,
            self._query_closed(),
            self._physical_metrics_key(),
            trace,
        )

    def _validate_committed(self, report: PostWeaningOperationReport) -> None:
        """拒绝 Core 漂移、资源泄漏和任一持久增长预算击穿。"""
        if not report.core_unchanged:
            raise PostWeaningRuntimeError("post-weaning 调用改变了 Core")
        if not report.query_closed:
            raise PostWeaningRuntimeError("post-weaning 调用泄漏 query 资源")
        budget = self.manifest.budget
        growth = (
            report.after.memory_event_count
            - report.before.memory_event_count,
            report.after.source_record_count
            - report.before.source_record_count,
            report.after.companion_item_count
            - report.before.companion_item_count,
        )
        limits = (
            budget.memory_event_growth,
            budget.source_record_growth,
            budget.companion_item_growth,
        )
        if any(value < 0 or value > limit
               for value, limit in zip(growth, limits)):
            raise PostWeaningRuntimeError("post-weaning 调用击穿持久增长预算")

    def _require_operation_budget(self) -> None:
        """在执行前拒绝超过实例级调用总数预算。"""
        if len(self._reports) >= self.manifest.budget.operation_limit:
            raise PostWeaningRuntimeError("post-weaning 实例调用预算已耗尽")

    def _query_closed(self) -> bool:
        """核验 query、generation、AttractorState 和 K-04 reader 均已关闭。"""
        work = self.ctx.work_memory
        hot = self.ctx.memory_hot_set_runtime
        return (
            work.active_query_scope is None
            and work.active_generation_scope is None
            and work.attractor_state is None
            and hot.query_resources_closed()
        )

    def _close_query_resources(self) -> None:
        """异常兜底关闭仍活动的 generation/query，由 WorkMemory 释放注册资源。"""
        work = self.ctx.work_memory
        if work.active_generation_scope is not None:
            work.end_generation()
        if work.active_query_scope is not None:
            work.end_query()

    def _physical_metrics_key(self) -> tuple[int, ...]:
        """返回最近 K-04 query 的页、对象、字节和 fault 指标；尚未查询则为空。"""
        metrics = self.ctx.memory_hot_set_runtime.metrics()
        if metrics is None:
            return ()
        observations = metrics.observations()
        result = [1, len(observations)]
        for item in observations:
            key = item.stable_key()
            result.extend(_packed(key))
        return tuple(result)

    def _restore_runtime_state(self) -> None:
        """后端回滚后重挂 Companion 水位并清空来源和 scope 的运行期缓存。"""
        source_intake = self.ctx.memory_read_intake.source_intake
        restored = CompanionSpace(
            SpaceRegistry(self.ctx.backend),
            self.ctx.backend,
            source_intake.companion.space_id,
        )
        source_intake.companion = restored
        self.ctx.memory_interact_intake.source_intake.companion = restored
        source_intake.repository.clear_runtime_caches()
        self.ctx.source_trust_records.clear_runtime_caches()
        self.ctx.scoped_identity_store.clear_runtime_caches()

    @staticmethod
    def _result_key(result: object) -> tuple[int, ...]:
        """从 M-05 或 SourceRecord 结果提取可恢复身份，不使用原文。"""
        if isinstance(result, MemoryIntakeResult):
            refs = (
                result.manifest_ref,
                *((result.observation_ref,)
                  if result.observation_ref is not None else ()),
                *result.hypothesis_refs,
                *result.evidence_refs,
                *((result.failure_ref,)
                  if result.failure_ref is not None else ()),
                *result.superseded_refs,
            )
            payload = [
                1,
                *_packed(result.source_record.source_key),
                result.outcome_kind,
                len(refs),
            ]
            for ref in refs:
                payload.extend(_packed(ref.stable_key()))
            return tuple(payload)
        if isinstance(result, SourceRecordStorage):
            return 2, *_packed(result.source_key), result.source_hash
        if isinstance(result, MemoryQuestionDialogueRun):
            return 3, *_packed(result.stable_key())
        raise TypeError("PW-00 不支持该入口结果类型")


class PostWeaningDryRunRuntime(PostWeaningOperationRuntime):
    """只允许 WEANING_PRE 隔离 fixture 的 PW-00 dry-run 薄门面。"""

    def __init__(
            self,
            ctx: TrainContext,
            manifest: PostWeaningDryRunManifest,
            *,
            core_reader: CoreCanonicalStateReader | None = None,
            ) -> None:
        """拒绝正式清单，并委托共用操作核心核验 dry-run。"""
        if not isinstance(manifest, PostWeaningDryRunManifest):
            raise TypeError("PW-00 dry-run manifest 类型错误")
        super().__init__(
            ctx,
            manifest,
            core_reader=core_reader,
            required_phase=WEANING_PRE,
        )

    @classmethod
    def start_formal(cls, *args, **kwargs):
        """薄转发到 PW-00A verifier，错误 dry-run 请求仍会 fail closed。"""
        from pure_integer_ai.cognition.shared.formal_post_weaning import (
            FormalPostWeaningLoadRequest,
        )
        if (len(args) < 2
                or not isinstance(args[1], FormalPostWeaningLoadRequest)):
            raise PostWeaningStartupError(
                "PW-00A formal start 必须使用正式装载请求")
        from pure_integer_ai.experiments.pw00a_formal_runtime import (
            PW00AFormalRuntime,
        )
        return PW00AFormalRuntime.start(*args, **kwargs)


__all__ = [
    "CoreCanonicalStateReader",
    "PostWeaningDryRunRuntime",
    "PostWeaningOperationRuntime",
    "PostWeaningRuntimeError",
    "PostWeaningStartupError",
    "build_post_weaning_dry_run_manifest",
    "post_weaning_component_state_key",
]
