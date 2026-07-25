"""把同次 G-04 分维结果精确追加到 G-05 已采用的 M-08 Use。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME,
    MemoryLinkedRef,
    MemoryObjectRef,
    UseOutcomePayload,
    UsePayload,
)
from pure_integer_ai.cognition.shared.memory_generation import (
    MemoryGenerationCommitReport,
    MemoryGenerationOutcomeCommit,
    MemoryGenerationOutcomeReport,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.scope_identity import LogicalTimestamp
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRun,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageRewardSignal,
)
from pure_integer_ai.experiments.memory_use_runtime import MemoryUseRuntime
from pure_integer_ai.experiments.verification_orchestration import (
    VerificationResult,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为开放整数键增加长度边界。"""
    return len(key), *key


def _strict_int(value: int, *, label: str) -> int:
    """核验开放协议值使用严格整数，不在桥内解释其语义。"""
    assert_int(value, _where=label)
    if type(value) is not int:
        raise ValueError(f"{label} 必须是严格整数")
    return value


def _access_for(ref: MemoryObjectRef) -> MemoryAccessContext:
    """按目标 Use owner 构造不提升可见性的 Memory 读取上下文。"""
    return MemoryAccessContext(
        ref.owner.tenant_id,
        ref.owner.user_id,
        ref.owner.session_id,
    )


@dataclass(frozen=True)
class MemoryGenerationOutcomeValueRoute:
    """把一个注入的 applicability/verdict 对映射到一等结果引用。"""

    applicability: int
    verdict: int
    outcome_ref: MemoryLinkedRef

    def __post_init__(self) -> None:
        """核验开放结果对和一等引用，不写死正负或 unknown 含义。"""
        _strict_int(self.applicability, label="Memory outcome applicability")
        _strict_int(self.verdict, label="Memory outcome verdict")
        if not isinstance(self.outcome_ref, MemoryLinkedRef):
            raise TypeError("Memory outcome value 必须是一等引用")

    def stable_key(self) -> tuple[int, ...]:
        """返回结果对及其图外一等引用的稳定配置键。"""
        return (
            self.applicability,
            self.verdict,
            *_packed(self.outcome_ref.stable_key()),
        )


@dataclass(frozen=True)
class MemoryGenerationOutcomeRoute:
    """为一个 exact dimension/verifier 注入 outcome kind 和结果映射。"""

    dimension: ProtocolKey
    verifier: ProtocolKey
    outcome_kind: MemoryLinkedRef
    values: tuple[MemoryGenerationOutcomeValueRoute, ...]

    def __post_init__(self) -> None:
        """核验维度、verifier、kind 和结果映射均唯一且可回查。"""
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("Memory outcome dimension 必须是 ProtocolKey")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("Memory outcome verifier 必须是 ProtocolKey")
        if not isinstance(self.outcome_kind, MemoryLinkedRef):
            raise TypeError("Memory outcome kind 必须是一等引用")
        if (not isinstance(self.values, tuple) or not self.values
                or any(not isinstance(item, MemoryGenerationOutcomeValueRoute)
                       for item in self.values)):
            raise TypeError("Memory outcome values 必须是非空分型映射")
        pairs = tuple((item.applicability, item.verdict) for item in self.values)
        refs = tuple(item.outcome_ref.stable_key() for item in self.values)
        if len(set(pairs)) != len(pairs):
            raise ValueError("Memory outcome 结果对不得重复")
        if len(set(refs)) != len(refs):
            raise ValueError("Memory outcome 结果引用不得复用不同结果对")

    def value_for(
            self,
            applicability: int,
            verdict: int,
            ) -> MemoryGenerationOutcomeValueRoute:
        """返回 exact 结果对的注入引用；缺映射时 fail closed。"""
        matches = tuple(
            item for item in self.values
            if (item.applicability, item.verdict) == (applicability, verdict)
        )
        if len(matches) != 1:
            raise ValueError("G-04 结果缺少唯一 Memory outcome 映射")
        return matches[0]

    def stable_key(self) -> tuple[int, ...]:
        """返回 exact verifier route 及全部结果映射的配置键。"""
        result = [
            *_packed(self.dimension.stable_key()),
            *_packed(self.verifier.stable_key()),
            *_packed(self.outcome_kind.stable_key()),
            len(self.values),
        ]
        for value in self.values:
            result.extend(_packed(value.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryGenerationOutcomeProtocol:
    """保存 G-04 分维信号到 M-08 outcome 的完整注入式路由。"""

    routes: tuple[MemoryGenerationOutcomeRoute, ...]

    def __post_init__(self) -> None:
        """核验 verifier route 和 outcome kind 均全局唯一。"""
        if (not isinstance(self.routes, tuple) or not self.routes
                or any(not isinstance(item, MemoryGenerationOutcomeRoute)
                       for item in self.routes)):
            raise TypeError("Memory outcome protocol routes 必须非空")
        pairs = tuple(
            (item.dimension.stable_key(), item.verifier.stable_key())
            for item in self.routes
        )
        kinds = tuple(item.outcome_kind.stable_key() for item in self.routes)
        if len(set(pairs)) != len(pairs):
            raise ValueError("Memory outcome verifier route 不得重复")
        if len(set(kinds)) != len(kinds):
            raise ValueError("不同 verifier 不得复用同一 outcome kind")

    def route_for(
            self,
            result: VerificationResult,
            ) -> MemoryGenerationOutcomeRoute:
        """按 exact dimension/verifier 返回唯一路由，拒绝忽略新维度。"""
        if not isinstance(result, VerificationResult):
            raise TypeError("Memory outcome 只能映射 VerificationResult")
        matches = tuple(
            item for item in self.routes
            if (item.dimension == result.dimension
                and item.verifier == result.verifier)
        )
        if len(matches) != 1:
            raise ValueError("G-04 dimension/verifier 缺少唯一 Memory outcome route")
        return matches[0]

    def stable_key(self) -> tuple[int, ...]:
        """返回稳定有序的全部注入式结果路由。"""
        ordered = tuple(sorted(
            self.routes,
            key=lambda item: (
                item.dimension.stable_key(),
                item.verifier.stable_key(),
            ),
        ))
        result = [len(ordered)]
        for route in ordered:
            result.extend(_packed(route.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class _PreparedOutcome:
    """首写前完成核验的一条 exact Use 分维 outcome。"""

    candidate_key: tuple[int, ...]
    use_ref: MemoryObjectRef
    payload: UseOutcomePayload
    signal_key: tuple[int, ...]


class MemoryQuestionOutcomeCommitter:
    """把同次 G-04 claim 逐维归因到已实际采用的 exact Memory Use。"""

    def __init__(
            self,
            memory_use: MemoryUseRuntime,
            protocol: MemoryGenerationOutcomeProtocol,
            *,
            trace_prefix: tuple[int, ...],
            ) -> None:
        """绑定 M-08 event log、注入式结果路由和调用方 trace。"""
        if not isinstance(memory_use, MemoryUseRuntime):
            raise TypeError("Memory outcome committer 需要 M-08 runtime")
        if not isinstance(protocol, MemoryGenerationOutcomeProtocol):
            raise TypeError("Memory outcome committer protocol 类型错误")
        if (not isinstance(trace_prefix, tuple) or not trace_prefix
                or any(type(item) is not int for item in trace_prefix)):
            raise ValueError("Memory outcome trace prefix 必须是非空严格整数 tuple")
        assert_int(*trace_prefix, _where="Memory outcome trace prefix")
        self.memory_use = memory_use
        self.protocol = protocol
        self.trace_prefix = trace_prefix

    def commit(
            self,
            request: QuestionRequest,
            query: QuestionQuery,
            result: QuestionExecutionResult,
            selection: AnswerContentSelection,
            generation: TypedGenerationExecution,
            selection_commit: MemoryGenerationCommitReport,
            postcheck: GenerationPostcheckRun,
            ) -> MemoryGenerationOutcomeReport:
        """先完整预检同次性与逐 claim 归属，再幂等追加分维 outcome。"""
        self._validate_upstream(
            request,
            query,
            result,
            selection,
            generation,
            selection_commit,
            postcheck,
        )
        prepared = self._prepare(
            request,
            selection,
            generation,
            selection_commit,
            postcheck,
        )
        existing = tuple(self._existing(item) for item in prepared)
        materialized = []
        for item, replay in zip(prepared, existing):
            event = replay
            if event is None:
                event = self.memory_use.record_outcome(
                    item.use_ref,
                    scope=request.response_scope,
                    outcome_kind=item.payload.outcome_kind,
                    outcome_ref=item.payload.outcome_ref,
                    observed_at=item.payload.observed_at,
                    outcome_trace_key=item.payload.outcome_trace_key,
                )
            if event.event.payload != item.payload:
                raise ValueError("Memory outcome 持久化结果与预检 payload 漂移")
            materialized.append(MemoryGenerationOutcomeCommit(
                item.candidate_key,
                item.use_ref,
                item.payload.outcome_kind,
                item.payload.outcome_ref,
                item.signal_key,
                integer_tuple_fingerprint(
                    event.event.stable_key(),
                    domain="memory.generation.outcome.event.v1",
                ),
            ))
        selection_key = integer_tuple_fingerprint(
            selection.stable_key(),
            domain="question.commit.selection.v1",
        )
        generation_key = integer_tuple_fingerprint(
            generation.stable_key(),
            domain="question.commit.generation.v1",
        )
        postcheck_key = integer_tuple_fingerprint(
            postcheck.stable_key(),
            domain="memory.generation.outcome.postcheck.v1",
        )
        trace = (
            *self.trace_prefix,
            *_packed(integer_tuple_fingerprint(
                self.protocol.stable_key(),
                domain="memory.generation.outcome.protocol.v1",
            )),
            len(materialized),
        )
        return MemoryGenerationOutcomeReport(
            selection_key,
            generation_key,
            postcheck_key,
            tuple(materialized),
            trace,
        )

    def _validate_upstream(
            self,
            request: QuestionRequest,
            query: QuestionQuery,
            result: QuestionExecutionResult,
            selection: AnswerContentSelection,
            generation: TypedGenerationExecution,
            selection_commit: MemoryGenerationCommitReport,
            postcheck: GenerationPostcheckRun,
            ) -> None:
        """核验问答、selection、generation、Use 报告和 G-04 属于同一次 query。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("Memory outcome request 类型错误")
        if not isinstance(query, QuestionQuery) or query.request != request:
            raise ValueError("Memory outcome query 替换了原请求")
        if (not isinstance(result, QuestionExecutionResult)
                or result.query != query):
            raise ValueError("Memory outcome result 替换了同次 query")
        planning = result.planning_request()
        if (not isinstance(selection, AnswerContentSelection)
                or selection.request != planning):
            raise ValueError("Memory outcome selection 替换了查询结果")
        if (not isinstance(generation, TypedGenerationExecution)
                or generation.plan.request != planning
                or not generation.complete):
            raise ValueError("Memory outcome 只接受同次完整 generation")
        if not isinstance(selection_commit, MemoryGenerationCommitReport):
            raise TypeError("Memory outcome selection commit 类型错误")
        if selection_commit.selection_key != integer_tuple_fingerprint(
                selection.stable_key(),
                domain="question.commit.selection.v1"):
            raise ValueError("Memory outcome selection 内容引用漂移")
        if selection_commit.generation_key != integer_tuple_fingerprint(
                generation.stable_key(),
                domain="question.commit.generation.v1"):
            raise ValueError("Memory outcome generation 内容引用漂移")
        if (not isinstance(postcheck, GenerationPostcheckRun)
                or postcheck.request.execution != generation):
            raise ValueError("Memory outcome postcheck 替换了同次 generation")
        if not postcheck.report.read_only:
            raise ValueError("Memory outcome 只能消费只读 G-04 report")

    def _prepare(
            self,
            request: QuestionRequest,
            selection: AnswerContentSelection,
            generation: TypedGenerationExecution,
            selection_commit: MemoryGenerationCommitReport,
            postcheck: GenerationPostcheckRun,
            ) -> tuple[_PreparedOutcome, ...]:
        """双向核验 candidate/evidence/processing/Use/claim 并构造零写计划。"""
        candidates = {
            item.stable_key(): item for item in selection.request.candidates
        }
        selected_keys = set(selection.selected_candidate_keys)
        planned_keys = set(candidates)
        if not selected_keys.issubset(planned_keys):
            raise ValueError("Memory outcome selection 含请求外候选")
        routes = {}
        value_routes = {}
        for verification in postcheck.report.results:
            route = self.protocol.route_for(verification)
            value = route.value_for(
                verification.applicability,
                verification.verdict,
            )
            routes[(verification.dimension, verification.verifier)] = route
            value_routes[(verification.dimension, verification.verifier)] = value
            goal = generation.plan.request.goal
            if verification.claim_keys and (
                    verification.source != goal.source
                    or verification.scope != goal.scope):
                raise ValueError("G-04 claimed 结果的 source/scope 与当前 query 漂移")
            if (verification.source is not None
                    and verification.source != goal.source):
                raise ValueError("G-04 结果 source 与当前 generation 漂移")
            if (verification.scope is not None
                    and verification.scope != goal.scope):
                raise ValueError("G-04 结果 scope 与当前 generation 漂移")

        use_rows = []
        seen_candidates = set()
        seen_uses = set()
        for commit in selection_commit.commits:
            if commit.candidate_key not in selected_keys:
                raise ValueError("Memory outcome commit 指向未选候选")
            if commit.candidate_key in seen_candidates:
                raise ValueError("Memory outcome 同一 selected candidate 重复提交")
            if commit.use_ref.stable_key() in seen_uses:
                raise ValueError("Memory outcome 多个提交复用了同一 Use")
            candidate = candidates[commit.candidate_key]
            evidence = tuple(
                item for item in candidate.memory_evidence
                if item.stable_key() == commit.evidence_key
            )
            if len(evidence) != 1:
                raise ValueError("Memory outcome 无法唯一恢复 selected Memory evidence")
            use_event = self._use_event(commit.use_ref)
            use = use_event.event.payload
            if (use_event.event.scope != request.response_scope
                    or use.used_at.clock.scope != request.response_scope):
                raise ValueError("Memory outcome Use 不属于当前 query scope")
            if (use.memory_ref != evidence[0].candidate.memory_ref
                    or use.decision_trace_key != commit.processing_key):
                raise ValueError("Memory outcome Use 与 evidence/processing 漂移")
            seen_candidates.add(commit.candidate_key)
            seen_uses.add(commit.use_ref.stable_key())
            use_rows.append((commit, use))

        max_use_seq = max(
            (use.used_at.seq for _, use in use_rows),
            default=0,
        )
        prepared = []
        for commit, use in use_rows:
            claimed = tuple(
                verification
                for verification in postcheck.report.results
                if commit.candidate_key in verification.claim_keys
            )
            if not claimed:
                raise ValueError("已采用 Memory candidate 没有 G-04 claimed 结果")
            for verification in claimed:
                key = (verification.dimension, verification.verifier)
                route = routes[key]
                value = value_routes[key]
                signal_key = integer_tuple_fingerprint(
                    TypedLanguageRewardSignal.from_verification(
                        verification).stable_key(),
                    domain="memory.generation.outcome.signal.v1",
                )
                observed_at = LogicalTimestamp(
                    use.used_at.clock,
                    max_use_seq + len(prepared) + 1,
                )
                payload = UseOutcomePayload(
                    commit.use_ref,
                    use.decision_trace_key,
                    use.query_kind,
                    use.context_key,
                    route.outcome_kind,
                    value.outcome_ref,
                    observed_at,
                    signal_key,
                )
                prepared.append(_PreparedOutcome(
                    commit.candidate_key,
                    commit.use_ref,
                    payload,
                    signal_key,
                ))
        return tuple(prepared)

    def _use_event(self, use_ref: MemoryObjectRef):
        """读取 exact Use 唯一声明，拒绝兼容或伪造引用。"""
        events = self.memory_use.event_log.query(
            access=_access_for(use_ref),
            event_kind=MEMORY_EVENT_USE,
            object_ref=use_ref,
        )
        if (len(events) != 1
                or not isinstance(events[0].event.payload, UsePayload)):
            raise ValueError("Memory outcome use_ref 没有唯一 M-08 Use 声明")
        return events[0]

    def _existing(self, prepared: _PreparedOutcome):
        """在首写前识别 exact replay，并拒绝同 kind 的竞争延迟结果。"""
        events = self.memory_use.event_log.query(
            access=_access_for(prepared.use_ref),
            event_kind=MEMORY_EVENT_USE_OUTCOME,
            object_ref=prepared.use_ref,
        )
        matches = tuple(
            item for item in events
            if (isinstance(item.event.payload, UseOutcomePayload)
                and item.event.payload.outcome_kind
                == prepared.payload.outcome_kind)
        )
        if not matches:
            return None
        if len(matches) != 1 or matches[0].event.payload != prepared.payload:
            raise ValueError("同一 Use/outcome kind 已存在竞争结果")
        return matches[0]


__all__ = [
    "MemoryGenerationOutcomeProtocol",
    "MemoryGenerationOutcomeRoute",
    "MemoryGenerationOutcomeValueRoute",
    "MemoryQuestionOutcomeCommitter",
]
