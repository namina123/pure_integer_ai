"""A-05 来源准入、M-10 组摄入和 V-06 克隆运行时。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.cognition.shared.memory_batch import (
    MemoryBatchCoordinator,
    MemoryBatchFaultInjector,
)
from pure_integer_ai.cognition.shared.post_weaning import PostWeaningIntakeRequest
from pure_integer_ai.cognition.shared.source_trust import (
    SourceTrustAssessment,
    SourceTrustPolicy,
    SourceTrustRequest,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    MemoryIntakeResult,
    MemorySourceIntake,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.source_record import SourceRecordStorage
from pure_integer_ai.storage.source_trust import SourceTrustStorageRepository


SOURCE_ADMISSION_RUNTIME_VERSION = 1


class SourceAdmissionError(RuntimeError):
    """来源准入 policy、图身份或批次协议不一致。"""


class SourceAdmissionRejected(SourceAdmissionError):
    """来源在任何 Companion、SourceRecord 或 Memory 写入前被拒绝。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(value), *value


@dataclass(frozen=True)
class SourceAdmissionBatchRun:
    """一次完整 M-10 组摄入的准入裁决和逐来源结果。"""

    assessments: tuple[SourceTrustAssessment, ...]
    results: tuple[MemoryIntakeResult, ...]

    def __post_init__(self) -> None:
        """要求整批裁决与结果一一对应且全部已经接受。"""
        if not self.assessments or len(self.assessments) != len(self.results):
            raise ValueError("来源组摄入裁决与结果数量不一致")
        if any(not item.accepted for item in self.assessments):
            raise ValueError("来源组摄入结果包含未接受裁决")
        if any(not isinstance(item, MemoryIntakeResult) for item in self.results):
            raise TypeError("来源组摄入包含错误结果类型")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含原文和 parser 实例的完整组运行键。"""
        result = [SOURCE_ADMISSION_RUNTIME_VERSION, len(self.assessments)]
        for assessment, intake in zip(self.assessments, self.results):
            result.extend(_packed(assessment.stable_key()))
            result.extend(_packed((
                *intake.source_record.source_key,
                intake.source_record.source_hash,
                *intake.manifest_ref.stable_key(),
            )))
        return tuple(result)


class SourceAdmissionRuntime:
    """在 M-01/M-05 写入前执行来源 policy，并编排 M-10 组事务。"""

    def __init__(
            self,
            ctx: TrainContext,
            policy: SourceTrustPolicy,
            *,
            reading_route,
            interaction_route,
            record_only_routes: tuple[object, ...] = (),
            ) -> None:
        """绑定上下文、版本化 policy 和调用方注入的入口路由。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("来源准入 ctx 必须是 TrainContext")
        for name in ("assess", "state_key", "clone_for_context"):
            if not callable(getattr(policy, name, None)):
                raise TypeError(f"来源准入 policy 缺少 {name}")
        policy_key = policy.state_key()
        if (not isinstance(policy_key, tuple) or not policy_key
                or any(type(item) is not int for item in policy_key)):
            raise TypeError("来源准入 policy state_key 必须是非空严格整数元组")
        if (not isinstance(ctx.memory_read_intake, MemorySourceIntake)
                or not isinstance(ctx.memory_interact_intake, MemorySourceIntake)):
            raise SourceAdmissionError("来源准入缺少成对 M-05 intake")
        if not isinstance(ctx.memory_batch_coordinator, MemoryBatchCoordinator):
            raise SourceAdmissionError("来源准入缺少 M-10 batch coordinator")
        if not isinstance(
                ctx.source_trust_records, SourceTrustStorageRepository):
            raise SourceAdmissionError("来源准入缺少持久化 assessment 仓库")
        route_values = (reading_route, interaction_route, *record_only_routes)
        if any(not hasattr(item, "stable_key") for item in route_values):
            raise TypeError("来源准入 route 缺少稳定身份")
        route_keys = tuple(item.stable_key() for item in route_values)
        if len(set(route_keys)) != len(route_keys):
            raise ValueError("来源准入 route 不得重复")
        if not isinstance(record_only_routes, tuple):
            raise TypeError("record_only_routes 必须是 tuple")
        self.ctx = ctx
        self.policy = policy
        self.records = ctx.source_trust_records
        self.reading_route = reading_route
        self.interaction_route = interaction_route
        self.record_only_routes = record_only_routes
        self._memory_routes = {
            reading_route.stable_key(): ctx.memory_read_intake,
            interaction_route.stable_key(): ctx.memory_interact_intake,
        }
        self._record_only_route_keys = frozenset(
            item.stable_key() for item in record_only_routes)
        self._assessments: dict[
            tuple[tuple[int, ...], tuple[int, ...], str, str, int,
                  tuple[int, ...]],
            SourceTrustAssessment,
        ] = {}

    def state_key(self) -> tuple[int, ...]:
        """返回 policy、双 Memory 路由和只留档路由的稳定装配键。"""
        policy_key = self.policy.state_key()
        result = [
            SOURCE_ADMISSION_RUNTIME_VERSION,
            *_packed(policy_key),
            *_packed(self.reading_route.stable_key()),
            *_packed(self.interaction_route.stable_key()),
            len(self.record_only_routes),
        ]
        for route in sorted(
                self.record_only_routes, key=lambda item: item.stable_key()):
            result.extend(_packed(route.stable_key()))
        return tuple(result)

    def preflight(
            self,
            request: PostWeaningIntakeRequest,
            ) -> SourceTrustAssessment:
        """在任何来源写入前执行 policy、图身份和裁决稳定性核验。"""
        if not isinstance(request, PostWeaningIntakeRequest):
            raise TypeError("来源准入请求必须是 PostWeaningIntakeRequest")
        route_key = request.route_kind.stable_key()
        if (route_key not in self._memory_routes
                and route_key not in self._record_only_route_keys):
            raise SourceAdmissionError("来源准入请求使用了未注册 route")
        trust_request = SourceTrustRequest(
            request.route_kind,
            request.source,
            request.raw_text,
            request.license_id,
            request.batch_id,
            request.trace,
        )
        assessment = self.policy.assess(trust_request)
        if not isinstance(assessment, SourceTrustAssessment):
            raise TypeError("来源准入 policy 返回类型错误")
        if assessment.request_key != trust_request.stable_key():
            raise SourceAdmissionError("来源准入 policy 替换或截断了请求身份")
        if assessment.policy_state_key != self.policy.state_key():
            raise SourceAdmissionError("来源准入 assessment 与 policy 状态漂移")
        self._validate_graph_refs(assessment)
        cache_key = (
            route_key,
            request.source.stable_key(),
            request.raw_text,
            request.license_id,
            request.batch_id,
            request.trace,
        )
        previous = self._assessments.get(cache_key)
        if previous is not None and previous != assessment:
            raise SourceAdmissionError("同一完整来源请求的准入裁决发生漂移")
        self._assessments[cache_key] = assessment
        stored = self.records.find(request.source.stable_key())
        if stored is not None and stored.assessment_key != assessment.stable_key():
            raise SourceAdmissionError(
                "既有 SourceRef 的持久化准入 assessment 发生漂移")
        if not assessment.accepted:
            raise SourceAdmissionRejected("来源未通过许可、信任或异常准入")
        return assessment

    def ingest_batch(
            self,
            requests: tuple[PostWeaningIntakeRequest, ...],
            *,
            cursor_commit: Callable[[], None],
            fault_injector: MemoryBatchFaultInjector | None = None,
            ) -> SourceAdmissionBatchRun:
        """先零写预检整批，再经 M-10 group commit 原子摄入全部来源。"""
        if not isinstance(requests, tuple) or not requests:
            raise TypeError("来源组摄入 requests 必须是非空 tuple")
        if any(not isinstance(item, PostWeaningIntakeRequest)
               for item in requests):
            raise TypeError("来源组摄入包含错误请求类型")
        batch_ids = {item.batch_id for item in requests}
        if len(batch_ids) != 1:
            raise ValueError("来源组摄入必须共享同一 batch_id")
        source_keys = tuple(item.source.stable_key() for item in requests)
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("来源组摄入不得重复完整 SourceRef")
        if not callable(cursor_commit):
            raise TypeError("来源组摄入 cursor_commit 必须可调用")
        if any(item.route_kind.stable_key() not in self._memory_routes
               for item in requests):
            raise ValueError("M-10 来源组只接受 reading/interaction Memory route")

        assessments = tuple(self.preflight(item) for item in requests)
        actions = tuple(
            self._unit_action(
                item,
                assessment,
                fault_injector=fault_injector,
            )
            for item, assessment in zip(requests, assessments)
        )
        results = self.ctx.memory_batch_coordinator.execute(
            next(iter(batch_ids)),
            actions,
            cursor_commit=cursor_commit,
            fault_injector=fault_injector,
        )
        return SourceAdmissionBatchRun(assessments, results)

    def admit_record_only(
            self,
            request: PostWeaningIntakeRequest,
            ) -> SourceRecordStorage:
        """准入只留档来源，并在 SourceRecord 后保存 assessment/cluster。"""
        if not isinstance(request, PostWeaningIntakeRequest):
            raise TypeError("只留档来源请求必须是 PostWeaningIntakeRequest")
        if request.route_kind.stable_key() not in self._record_only_route_keys:
            raise ValueError("只留档来源使用了非 record-only route")
        if request.parser is not None or request.supersedes_source is not None:
            raise ValueError("只留档来源不接受 parser 或 supersede")
        assessment = self.preflight(request)
        intake = self.ctx.memory_read_intake.source_intake
        record = intake.ensure(
            request.source,
            request.raw_text,
            license_id=request.license_id,
            batch_id=request.batch_id,
        )
        self._persist_assessment(request, assessment)
        return record

    def clone_for_context(self, ctx: TrainContext) -> "SourceAdmissionRuntime":
        """为 V-06 克隆 policy 和路由，并重绑独立 M-05/M-10 owner。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("来源准入 clone ctx 必须是 TrainContext")
        cloned_policy = self.policy.clone_for_context(ctx)
        runtime = SourceAdmissionRuntime(
            ctx,
            cloned_policy,
            reading_route=self.reading_route,
            interaction_route=self.interaction_route,
            record_only_routes=self.record_only_routes,
        )
        if runtime.state_key() != self.state_key():
            raise SourceAdmissionError("来源准入 V-06 clone 改变了 policy 状态")
        return runtime

    def _unit_action(
            self,
            request: PostWeaningIntakeRequest,
            assessment: SourceTrustAssessment,
            *,
            fault_injector: MemoryBatchFaultInjector | None,
            ) -> Callable[[], MemoryIntakeResult]:
        """把一个已预检请求封装为无隐藏状态的 M-05 单元动作。"""
        intake = self._memory_routes[request.route_kind.stable_key()]

        def action() -> MemoryIntakeResult:
            """执行当前来源的 M-05/M-10 单元摄入。"""
            if request.parser is None or not callable(
                    getattr(request.parser, "parse", None)):
                raise TypeError("来源组摄入必须提供 MemoryIntakeParser")
            intake.source_intake.ensure(
                request.source,
                request.raw_text,
                license_id=request.license_id,
                batch_id=request.batch_id,
            )
            self._persist_assessment(request, assessment)
            return intake.ingest(
                request.source,
                request.raw_text,
                license_id=request.license_id,
                batch_id=request.batch_id,
                parser=request.parser,
                supersedes_source=request.supersedes_source,
                materialize=None,
                batch_fault_injector=fault_injector,
            )

        return action

    def _persist_assessment(
            self,
            request: PostWeaningIntakeRequest,
            assessment: SourceTrustAssessment,
            ) -> None:
        """在 SourceRecord 已存在后保存不可变 assessment 与来源簇身份。"""
        if assessment.request_key != SourceTrustRequest(
                request.route_kind,
                request.source,
                request.raw_text,
                request.license_id,
                request.batch_id,
                request.trace,
                ).stable_key():
            raise SourceAdmissionError("持久化前来源准入请求身份漂移")
        self.records.put(
            request.source.stable_key(),
            assessment.stable_key(),
            assessment.source_cluster_key,
        )

    def _validate_graph_refs(self, assessment: SourceTrustAssessment) -> None:
        """回读 assessment 的全部图引用，拒绝伪造或跨图编址。"""
        refs = (
            assessment.source_kind_ref,
            assessment.license_ref,
            assessment.trust_ref,
            *assessment.reason_refs,
            *assessment.blocking_anomaly_refs,
        )
        for ref in refs:
            self.ctx.graph_ontology.identity_of(ref)


def install_source_admission_runtime(
        ctx: TrainContext,
        policy: SourceTrustPolicy,
        *,
        reading_route,
        interaction_route,
        record_only_routes: tuple[object, ...] = (),
        ) -> SourceAdmissionRuntime:
    """在已安装 M-05/M-10 的上下文上安装唯一 A-05 准入 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("安装来源准入的 ctx 必须是 TrainContext")
    if ctx.source_trust_runtime is not None:
        raise ValueError("TrainContext 已安装来源准入 runtime")
    runtime = SourceAdmissionRuntime(
        ctx,
        policy,
        reading_route=reading_route,
        interaction_route=interaction_route,
        record_only_routes=record_only_routes,
    )
    ctx.source_trust_runtime = runtime
    return runtime


__all__ = [
    "SOURCE_ADMISSION_RUNTIME_VERSION",
    "SourceAdmissionBatchRun",
    "SourceAdmissionError",
    "SourceAdmissionRejected",
    "SourceAdmissionRuntime",
    "install_source_admission_runtime",
]
