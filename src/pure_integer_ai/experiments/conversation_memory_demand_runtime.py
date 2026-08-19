"""把一次会话的显式 typed read 接到有界 Memory query consumer。

本模块只执行当前 query 的 M-06 编译、M-07 只读解析和可选 K-04 热集读取，
不写 Core、Memory、Use 或长期会话。没有精确查询索引时，receipt 明确把
``zero_unrelated_scan_proven`` 置为 0，不能把普通过滤路径宣传为零无关扫描。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_overlay import (
    MemoryAccessContext,
)
from pure_integer_ai.cognition.shared.memory_query import (
    FederatedMemoryQueryCompilation,
    MemoryQueryCompilation,
    MemoryCurrentQuery,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    FederatedMemoryResolution,
    MemoryResolution,
    RESOLUTION_ORIGIN_MEMORY,
    ResolvedCandidateSet,
)
from pure_integer_ai.cognition.shared.memory_generation import (
    MemoryGenerationEvidence,
    MemoryGenerationSource,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
)
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    memory_hot_set_runtime_for,
)
from pure_integer_ai.experiments.memory_query_runtime import MemoryQueryRuntime
from pure_integer_ai.experiments.memory_resolver_runtime import (
    MemoryResolverRuntime,
)
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DirectionalMemoryCenter,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    MemoryExpansionChannelBudget,
    MemoryExpansionProfile,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.source_record import SourceRecordRepository


_DIGEST_SIZE = 32
_STATUS = ("HIT", "UNKNOWN")
_CHANNEL_OVERLAY = "L3_MEMORY_OVERLAY"
_CHANNEL_PAGE = "L4_SEALED_PAGE"


class ConversationMemoryDemandError(RuntimeError):
    """会话 Memory read 的 center、预算、scope 或回滚证据不闭合。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ConversationMemoryDemandError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ConversationMemoryDemandError(f"{label} 必须使用严格整数")
    return value


def _digest_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验固定摘要键。"""
    if (not isinstance(value, tuple) or len(value) != _DIGEST_SIZE
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise ConversationMemoryDemandError(f"{label} 非法")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长整数键加边界。"""
    return len(value), *value


def _requests(
        compilation: MemoryQueryCompilation | FederatedMemoryQueryCompilation,
        ) -> tuple:
    """按空间顺序展平 query compilation 的 typed activation requests。"""
    if isinstance(compilation, MemoryQueryCompilation):
        return compilation.requests
    if isinstance(compilation, FederatedMemoryQueryCompilation):
        return tuple(
            request
            for child in compilation.compilations
            for request in child.requests
        )
    raise TypeError("conversation Memory compilation 类型错误")


def _sets(
        resolution: MemoryResolution | FederatedMemoryResolution,
        ) -> tuple[ResolvedCandidateSet, ...]:
    """按同一 compilation 顺序展平解析结果。"""
    if isinstance(resolution, (MemoryResolution, FederatedMemoryResolution)):
        return resolution.sets if isinstance(
            resolution, FederatedMemoryResolution) else resolution.sets
    raise TypeError("conversation Memory resolution 类型错误")


@dataclass(frozen=True, slots=True)
class MemoryDemandRequestReceipt:
    """一个 activation request 的预算、考虑数和选中候选键。"""

    query_kind_key: tuple[int, ...]
    hypothesis_kind: tuple[int, ...]
    budget: int
    considered_count: int
    selected_candidate_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _strict_key(self.query_kind_key, label="demand request query kind")
        _strict_key(self.hypothesis_kind, label="demand request hypothesis kind")
        for label, value in (
                ("demand request budget", self.budget),
                ("demand request considered count", self.considered_count)):
            if type(value) is not int or value < 0:
                raise ConversationMemoryDemandError(f"{label} 非法")
        if self.budget <= 0:
            raise ConversationMemoryDemandError("demand request budget 必须为正")
        if (not isinstance(self.selected_candidate_keys, tuple)
                or any(not isinstance(item, tuple) or not item
                       for item in self.selected_candidate_keys)):
            raise TypeError("demand request candidate keys 类型错误")
        for item in self.selected_candidate_keys:
            _strict_key(item, label="demand request candidate key")
        if self.considered_count < len(self.selected_candidate_keys):
            raise ConversationMemoryDemandError(
                "demand request considered count 小于 selected count")
        if len(self.selected_candidate_keys) > self.budget:
            raise ConversationMemoryDemandError(
                "demand request selected count 超出 request budget")

    def stable_key(self) -> tuple[int, ...]:
        """返回 request receipt 的纯整数稳定键。"""
        result = [*_packed(self.query_kind_key), *_packed(self.hypothesis_kind),
                  self.budget, self.considered_count,
                  len(self.selected_candidate_keys)]
        for key in self.selected_candidate_keys:
            result.extend(_packed(key))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class MemoryDemandReceipt:
    """一次中心单 ring Memory read 的来源、预算、物理计数和回滚证明。"""

    center_key: tuple[int, ...]
    profile_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    channel_key: str
    context_revision: int
    context_read_digest: tuple[int, ...]
    compilation_key: tuple[int, ...]
    requests: tuple[MemoryDemandRequestReceipt, ...]
    considered_count: int
    selected_candidate_keys: tuple[tuple[int, ...], ...]
    page_read_count: int
    cold_read_bytes: int
    zero_unrelated_scan_proven: int
    rollback_before_key: tuple[int, ...]
    rollback_after_key: tuple[int, ...]
    status: str

    def __post_init__(self) -> None:
        _strict_key(self.center_key, label="demand receipt center key")
        _strict_key(self.profile_key, label="demand receipt profile key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("demand receipt source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("demand receipt scope 类型错误")
        if self.scope.source != self.source:
            raise ConversationMemoryDemandError(
                "demand receipt scope/source 不一致")
        if self.channel_key not in {_CHANNEL_OVERLAY, _CHANNEL_PAGE}:
            raise ConversationMemoryDemandError("demand receipt channel 未注册")
        if type(self.context_revision) is not int or self.context_revision < 0:
            raise ConversationMemoryDemandError("demand receipt context revision 非法")
        _digest_key(self.context_read_digest,
                    label="demand receipt context read digest")
        _strict_key(self.compilation_key, label="demand receipt compilation key")
        if (not isinstance(self.requests, tuple)
                or any(not isinstance(item, MemoryDemandRequestReceipt)
                       for item in self.requests)):
            raise TypeError("demand receipt requests 类型错误")
        for label, value in (
                ("demand receipt considered count", self.considered_count),
                ("demand receipt page reads", self.page_read_count),
                ("demand receipt cold bytes", self.cold_read_bytes)):
            if type(value) is not int or value < 0:
                raise ConversationMemoryDemandError(f"{label} 非法")
        if (not isinstance(self.selected_candidate_keys, tuple)
                or any(not isinstance(item, tuple) or not item
                       for item in self.selected_candidate_keys)):
            raise TypeError("demand receipt selected keys 类型错误")
        for key in self.selected_candidate_keys:
            _strict_key(key, label="demand receipt selected key")
        if self.considered_count != sum(
                item.considered_count for item in self.requests):
            raise ConversationMemoryDemandError(
                "demand receipt considered count 不闭合")
        if len(self.selected_candidate_keys) != sum(
                len(item.selected_candidate_keys) for item in self.requests):
            raise ConversationMemoryDemandError(
                "demand receipt selected count 不闭合")
        if type(self.zero_unrelated_scan_proven) is not int or (
                self.zero_unrelated_scan_proven not in (0, 1)):
            raise ConversationMemoryDemandError(
                "demand receipt zero scan proof 必须为 0/1")
        _strict_key(self.rollback_before_key,
                    label="demand receipt rollback before")
        _strict_key(self.rollback_after_key,
                    label="demand receipt rollback after")
        if self.rollback_before_key != self.rollback_after_key:
            raise ConversationMemoryDemandError(
                "demand receipt rollback 前后状态漂移")
        if self.status not in _STATUS:
            raise ConversationMemoryDemandError("demand receipt status 未注册")
        if self.status == "HIT" and not self.selected_candidate_keys:
            raise ConversationMemoryDemandError("HIT 必须有 selected candidate")
        if self.status == "UNKNOWN" and self.selected_candidate_keys:
            raise ConversationMemoryDemandError("UNKNOWN 不得携带 selected candidate")

    def stable_key(self) -> tuple[int, ...]:
        """返回 receipt 的完整纯整数审计键。"""
        result = [1, *_packed(self.center_key), *_packed(self.profile_key),
                  *_packed(self.source.stable_key()),
                  *_packed(self.scope.stable_key()),
                  1 if self.channel_key == _CHANNEL_PAGE else 2,
                  self.context_revision, *_packed(self.context_read_digest),
                  *_packed(self.compilation_key), len(self.requests)]
        for request in self.requests:
            result.extend(_packed(request.stable_key()))
        result.extend((self.considered_count, len(self.selected_candidate_keys)))
        for key in self.selected_candidate_keys:
            result.extend(_packed(key))
        result.extend((self.page_read_count, self.cold_read_bytes,
                       self.zero_unrelated_scan_proven,
                       *_packed(self.rollback_before_key),
                       *_packed(self.rollback_after_key),
                       1 if self.status == "HIT" else 2))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class MemoryDemandRead:
    """保留同次 resolver 输出及其 receipt，供后续理解/生成层显式消费。"""

    resolution: MemoryResolution | FederatedMemoryResolution
    receipt: MemoryDemandReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.resolution,
                          (MemoryResolution, FederatedMemoryResolution)):
            raise TypeError("demand read resolution 类型错误")
        if not isinstance(self.receipt, MemoryDemandReceipt):
            raise TypeError("demand read receipt 类型错误")
        if self.resolution.compilation.current.source != self.receipt.source:
            raise ConversationMemoryDemandError("demand read source 漂移")
        if self.resolution.compilation.current.scope != self.receipt.scope:
            raise ConversationMemoryDemandError("demand read scope 漂移")


class ConversationMemoryQuestionExecutor:
    """把同次 DLG-04 read 的 Memory 候选投影为 G-00 generation candidates。

    该 executor 不重新编译或重跑 M-06/M-07/A-10；候选只能来自调用方显式
    授权的本次 ``MemoryDemandRead``，来源缺少完整 SourceRecord 时直接失败。
    """

    def __init__(
            self,
            read: MemoryDemandRead,
            target: BoundProposition,
            *,
            authorized_candidate_keys: tuple[tuple[int, ...], ...],
            executed_reason: ObjectIdentity,
            binding_reason: ObjectIdentity,
            trace_prefix: tuple[int, ...],
            source_records: SourceRecordRepository,
            ) -> None:
        if not isinstance(read, MemoryDemandRead):
            raise TypeError("conversation Memory executor read 类型错误")
        if not isinstance(target, BoundProposition):
            raise TypeError("conversation Memory executor target 类型错误")
        if (not isinstance(authorized_candidate_keys, tuple)
                or any(not isinstance(item, tuple) or not item
                       for item in authorized_candidate_keys)):
            raise TypeError("conversation Memory executor authorized keys 类型错误")
        if len(set(authorized_candidate_keys)) != len(authorized_candidate_keys):
            raise ConversationMemoryDemandError(
                "conversation Memory executor authorized keys 不得重复")
        for key in authorized_candidate_keys:
            _strict_key(key, label="conversation Memory executor authorized key")
        for identity, label in (
                (executed_reason, "executed reason"),
                (binding_reason, "binding reason")):
            if (not isinstance(identity, ObjectIdentity)
                    or identity.object_kind != OBJECT_MINIMAL_INSTRUCTION):
                raise ValueError(
                    f"conversation Memory executor {label} 必须是 MinimalInstruction")
        if (not isinstance(trace_prefix, tuple) or not trace_prefix
                or any(type(item) is not int for item in trace_prefix)):
            raise ValueError("conversation Memory executor trace prefix 非法")
        if not isinstance(source_records, SourceRecordRepository):
            raise TypeError("conversation Memory executor SourceRecord 类型错误")
        self.read = read
        self.target = target
        self.authorized_candidate_keys = authorized_candidate_keys
        self.executed_reason = executed_reason
        self.binding_reason = binding_reason
        self.trace_prefix = trace_prefix
        self.source_records = source_records

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """只投影同次 read 的授权 Memory candidates，不读取 labels 或文本。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("conversation Memory executor query 类型错误")
        request = query.request
        if request.target != self.target:
            raise ConversationMemoryDemandError(
                "conversation Memory executor target 漂移")
        if (request.source != self.read.receipt.source
                or request.response_scope != self.read.receipt.scope):
            raise ConversationMemoryDemandError(
                "conversation Memory executor source/scope 漂移")
        selected_by_key = {
            candidate.stable_key(): candidate
            for candidate_set in _sets(self.read.resolution)
            for candidate in candidate_set.candidates
            if candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
        }
        selected_receipt_keys = set(
            self.read.receipt.selected_candidate_keys)
        if not set(selected_by_key).issubset(selected_receipt_keys):
            raise ConversationMemoryDemandError(
                "conversation Memory executor candidate 不在本次 receipt")
        candidates = []
        evidence_items = []
        for candidate_key in self.authorized_candidate_keys:
            candidate = selected_by_key.get(candidate_key)
            if candidate is None:
                raise ConversationMemoryDemandError(
                    "conversation Memory executor authorized candidate 未命中")
            sources = []
            for trace in candidate.memory_source_traces:
                record = self.source_records.find(trace.source.stable_key())
                if record is None:
                    raise ConversationMemoryDemandError(
                        "conversation Memory executor 缺少 SourceRecord")
                sources.append(MemoryGenerationSource.from_record(trace, record))
            if not sources:
                raise ConversationMemoryDemandError(
                    "conversation Memory executor candidate 缺少来源分账")
            evidence_items.append(MemoryGenerationEvidence(
                candidate,
                self.target,
                self.binding_reason,
                (*self.trace_prefix, *_packed(candidate_key)),
                tuple(sorted(sources, key=lambda item: item.stable_key())),
            ))
        if evidence_items:
            state = LogicEvidenceState(
                any(item.state.support for item in evidence_items),
                any(item.state.refute for item in evidence_items),
            )
            candidates.append(GenerationCandidate(
                self.target,
                state,
                request.source,
                request.response_scope,
                (),
                memory_evidence=tuple(evidence_items),
            ))
        trace = (
            *self.trace_prefix,
            *_packed(self.read.receipt.stable_key()),
            len(candidates),
            *(_packed(candidates[0].stable_key()) if candidates else ()),
        )
        return QuestionExecutionResult(
            query,
            self.executed_reason,
            tuple(candidates),
            trace,
        )


class ConversationMemoryDemandConsumer:
    """执行单中心、单 ring 的有界 Memory demand read。"""

    def __init__(
            self,
            ctx: TrainContext,
            query_runtime: MemoryQueryRuntime,
            resolver_runtime: MemoryResolverRuntime,
            ) -> None:
        """绑定同一 TrainContext 的 M-06/M-07，拒绝跨上下文 facade。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("demand consumer ctx 类型错误")
        if not isinstance(query_runtime, MemoryQueryRuntime):
            raise TypeError("demand consumer query runtime 类型错误")
        if not isinstance(resolver_runtime, MemoryResolverRuntime):
            raise TypeError("demand consumer resolver runtime 类型错误")
        if ctx.memory_query_runtime is not query_runtime:
            raise ConversationMemoryDemandError(
                "demand consumer query runtime 不属于当前上下文")
        if ctx.memory_resolver_runtime is not resolver_runtime:
            raise ConversationMemoryDemandError(
                "demand consumer resolver runtime 不属于当前上下文")
        self.ctx = ctx
        self.query_runtime = query_runtime
        self.resolver_runtime = resolver_runtime

    def read(
            self,
            current: MemoryCurrentQuery,
            center: DirectionalMemoryCenter,
            profile: MemoryExpansionProfile,
            *,
            access: MemoryAccessContext,
            context_read: ConversationContextRead,
            ) -> MemoryDemandRead:
        """消费一个显式 center，并返回真实 query/resolution 与只读 receipt。"""
        if not isinstance(current, MemoryCurrentQuery):
            raise TypeError("demand current 类型错误")
        if not isinstance(center, DirectionalMemoryCenter):
            raise TypeError("demand center 类型错误")
        if not isinstance(profile, MemoryExpansionProfile):
            raise TypeError("demand profile 类型错误")
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("demand access 类型错误")
        if not isinstance(context_read, ConversationContextRead):
            raise TypeError("demand context read 类型错误")
        if center.center.activation_only != 1:
            raise ConversationMemoryDemandError("demand center 必须是 activation-only")
        if center.write_boundary.host_learning_write_count != 0:
            raise ConversationMemoryDemandError("demand center 已有 host learning write")
        if center.center.expansion_profile_key != profile.profile_key:
            raise ConversationMemoryDemandError(
                "demand center/profile expansion key 漂移")
        if current.scope != self.ctx.work_memory.active_query_scope:
            raise ConversationMemoryDemandError(
                "demand current 不属于活动 query scope")
        if not access.can_read(current.scope.owner):
            raise PermissionError("demand access 不可读取当前 scope")
        hot_set = memory_hot_set_runtime_for(
            self.ctx, self.resolver_runtime.resolver)
        channel = _CHANNEL_PAGE if hot_set is not None else _CHANNEL_OVERLAY
        channel_budget = self._channel_budget(profile, channel)
        before = self._rollback_key(hot_set)
        compilation = self.query_runtime.compile(current, access=access)
        resolution = self.resolver_runtime.resolve(compilation)
        if resolution.compilation.current != current:
            raise ConversationMemoryDemandError("demand resolution 替换 current query")
        requests = _requests(compilation)
        sets = _sets(resolution)
        if len(requests) != len(sets):
            raise ConversationMemoryDemandError(
                "demand resolution 未逐 request 覆盖")
        request_receipts = tuple(
            MemoryDemandRequestReceipt(
                request.query_kind.stable_key(),
                request.hypothesis_kind,
                request.budget,
                candidate_set.considered_count,
                tuple(candidate.stable_key()
                      for candidate in candidate_set.candidates),
            )
            for request, candidate_set in zip(requests, sets)
        )
        considered = sum(item.considered_count for item in sets)
        selected = tuple(
            candidate.stable_key()
            for item in sets
            for candidate in item.candidates
        )
        metrics = None if hot_set is None else hot_set.metrics()
        page_reads = 0 if metrics is None else metrics.page_faults
        cold_bytes = 0 if metrics is None else metrics.cold_read_bytes
        self._enforce_budget(
            channel_budget,
            considered=considered,
            selected=len(selected),
            page_reads=page_reads,
            cold_bytes=cold_bytes,
        )
        after = self._rollback_key(hot_set)
        if before != after:
            raise ConversationMemoryDemandError(
                "demand read 改变了 Memory semantic state")
        exact_index = bool(
            hot_set is not None
            and isinstance(compilation, MemoryQueryCompilation)
            and hot_set.exact_index_available_for(requests)
        )
        receipt = MemoryDemandReceipt(
            center.center.center_key.stable_key(),
            profile.profile_key.stable_key(),
            current.source,
            current.scope,
            channel,
            context_read.revision,
            context_read.digest,
            compilation.stable_key(),
            request_receipts,
            considered,
            selected,
            page_reads,
            cold_bytes,
            int(exact_index),
            before,
            after,
            "HIT" if selected else "UNKNOWN",
        )
        return MemoryDemandRead(resolution, receipt)

    @staticmethod
    def _channel_budget(
            profile: MemoryExpansionProfile,
            channel: str,
            ) -> MemoryExpansionChannelBudget:
        """按 profile 精确找当前物理通道预算。"""
        matches = tuple(item for item in profile.channel_budgets
                        if item.channel_key == channel)
        if len(matches) != 1:
            raise ConversationMemoryDemandError(
                "demand profile 缺少唯一 Memory channel budget")
        budget = matches[0]
        if budget.admission_enabled != 1:
            raise ConversationMemoryDemandError("当前 Memory channel 未准入")
        return budget

    @staticmethod
    def _enforce_budget(
            budget: MemoryExpansionChannelBudget,
            *,
            considered: int,
            selected: int,
            page_reads: int,
            cold_bytes: int,
            ) -> None:
        """在返回 read 前执行 channel 的扫描、候选、分页和冷字节硬预算。"""
        checks = (
            (considered, budget.max_scanned_objects, "scanned objects"),
            (selected, budget.max_candidates, "candidates"),
            (page_reads, budget.max_page_reads, "page reads"),
            (cold_bytes, budget.max_cold_bytes, "cold bytes"),
        )
        for actual, limit, label in checks:
            if actual > limit:
                raise ConversationMemoryDemandError(
                    f"demand {label} 超出 channel budget")

    def _rollback_key(self, hot_set) -> tuple[int, ...]:
        """只取 query 读取前后可重放的整数状态，不把物理缓存计数写入语义。"""
        values = [
            *self.query_runtime.state_key(),
            *self.resolver_runtime.state_key(),
        ]
        for resolver in self.resolver_runtime.resolvers:
            values.extend(resolver.aggregates.event_log.projection_state_key())
        if hot_set is not None:
            values.extend(hot_set.state_key())
        return integer_tuple_fingerprint(
            tuple(values), domain="conversation.memory.demand.rollback.v1")


__all__ = [
    "ConversationMemoryDemandConsumer",
    "ConversationMemoryDemandError",
    "ConversationMemoryQuestionExecutor",
    "MemoryDemandRead",
    "MemoryDemandReceipt",
    "MemoryDemandRequestReceipt",
]
