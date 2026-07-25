"""G-05 将成功生成实际采用的 Memory 候选提交到 A-10 与 M-08。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorConsumptionDecision,
)
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_event import (
    MemoryLinkedRef,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_generation import (
    MemoryGenerationCommitReport,
    MemoryGenerationEvidence,
    MemoryGenerationSource,
    MemoryGenerationUseCommit,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import MemoryCurrentQuery
from pure_integer_ai.cognition.shared.memory_resolver import (
    RESOLUTION_ORIGIN_MEMORY,
)
from pure_integer_ai.cognition.shared.reasoning_planner import ReasoningObligation
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    SCOPE_QUERY,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.memory_use_runtime import MemoryUseRuntime
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerRun,
    QuestionAnswerRuntime,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.source_record import SourceRecordRepository


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为开放稳定键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 route trace 使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class MemoryQuestionDialogueRun:
    """一次已关闭 query 生命周期的问答报告及实际采用 SourceRecord 引用。"""

    question: QuestionAnswerRun
    sources: tuple[MemoryGenerationSource, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验问答报告、来源集合和 caller trace 都可稳定回放。"""
        if not isinstance(self.question, QuestionAnswerRun):
            raise TypeError("Memory dialogue question report 类型错误")
        if (not isinstance(self.sources, tuple)
                or any(not isinstance(item, MemoryGenerationSource)
                       for item in self.sources)):
            raise TypeError("Memory dialogue sources 类型错误")
        keys = tuple(item.stable_key() for item in self.sources)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("Memory dialogue sources 必须唯一且稳定有序")
        _strict_key(self.trace, label="Memory dialogue trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回问答执行、实际来源引用和 caller 生命周期 trace。"""
        result = [*_packed(self.question.stable_key()), len(self.sources)]
        for source in self.sources:
            result.extend(_packed(source.stable_key()))
        result.extend(_packed(self.trace))
        return tuple(result)


class ResolvedMemoryQuestionExecutor:
    """把当前 M-06/M-07/A-10 结果投影为来源完整的 G-00 Memory 候选。"""

    def __init__(
            self,
            ctx: TrainContext,
            current: MemoryCurrentQuery,
            access: MemoryAccessContext,
            obligations: tuple[ReasoningObligation, ...],
            *,
            executed_reason: ObjectIdentity,
            binding_reason: ObjectIdentity,
            trace_prefix: tuple[int, ...],
            source_records: SourceRecordRepository | None = None,
            ) -> None:
        """绑定 typed 当前输入、目标义务、来源仓库和两类注入式原因。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("Resolved Memory question ctx 类型错误")
        if not isinstance(current, MemoryCurrentQuery):
            raise TypeError("Resolved Memory question current 类型错误")
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("Resolved Memory question access 类型错误")
        if (not isinstance(obligations, tuple) or not obligations
                or any(not isinstance(item, ReasoningObligation)
                       for item in obligations)):
            raise TypeError("Resolved Memory question obligations 类型错误")
        for identity, label in (
                (executed_reason, "executed reason"),
                (binding_reason, "binding reason")):
            if (not isinstance(identity, ObjectIdentity)
                    or identity.object_kind != OBJECT_MINIMAL_INSTRUCTION):
                raise ValueError(f"Resolved Memory question {label} 必须是 MinimalInstruction")
        _strict_key(trace_prefix, label="Resolved Memory question trace prefix")
        repository = source_records or SourceRecordRepository(ctx.backend)
        if not isinstance(repository, SourceRecordRepository):
            raise TypeError("Resolved Memory question SourceRecord repository 类型错误")
        if repository.backend is not ctx.backend:
            raise ValueError("Resolved Memory question SourceRecord 必须属于同一 backend")
        if (ctx.memory_query_runtime is None
                or ctx.memory_resolver_runtime is None
                or ctx.attractor_runtime is None):
            raise ValueError("Resolved Memory question 缺少 M-06/M-07/A-10 runtime")
        self._ctx = ctx
        self.current = current
        self.access = access
        self.obligations = obligations
        self.executed_reason = executed_reason
        self.binding_reason = binding_reason
        self.trace_prefix = trace_prefix
        self.source_records = repository

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """执行 M-06/M-07/A-10，并只投影当前目标的真实 Memory activation。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("Resolved Memory question executor 需要 QuestionQuery")
        request = query.request
        if (request.response_scope != self.current.scope
                or request.source != self.current.source):
            raise ValueError("Resolved Memory question 请求与当前 typed 输入漂移")
        if all(item.proposition != request.target for item in self.obligations):
            raise ValueError("Resolved Memory question 缺少当前 target obligation")
        compilation = self._ctx.memory_query_runtime.compile(
            self.current,
            access=self.access,
        )
        state = self._ctx.attractor_runtime.resolve_and_activate(
            compilation,
            self.obligations,
        )
        evidence_items = []
        for activation in state.activations():
            if (activation.candidate.origin_kind != RESOLUTION_ORIGIN_MEMORY
                    or activation.obligation.proposition != request.target):
                continue
            sources = []
            for trace in activation.candidate.memory_source_traces:
                record = self.source_records.find(trace.source.stable_key())
                if record is None:
                    raise ValueError("Resolved Memory question 缺少 SourceRecord")
                sources.append(MemoryGenerationSource.from_record(trace, record))
            evidence = MemoryGenerationEvidence(
                activation.candidate,
                request.target,
                self.binding_reason,
                (
                    *self.trace_prefix,
                    *_packed(query.stable_key()),
                    *_packed(activation.identity_key()),
                ),
                tuple(sources),
            )
            evidence_items.append(evidence)
        candidates = ()
        if evidence_items:
            derived = LogicEvidenceState(
                any(item.state.support for item in evidence_items),
                any(item.state.refute for item in evidence_items),
            )
            candidates = (GenerationCandidate(
                request.target,
                derived,
                request.source,
                request.response_scope,
                (),
                memory_evidence=tuple(evidence_items),
            ),)
        candidate_keys = tuple(item.stable_key() for item in candidates)
        if len(set(candidate_keys)) != len(candidate_keys):
            raise ValueError("Resolved Memory question 形成重复 G-00 候选")
        trace = [
            *self.trace_prefix,
            *_packed(compilation.stable_key()),
            *_packed(state.stable_key()),
            len(candidate_keys),
        ]
        for key in sorted(candidate_keys):
            trace.extend(_packed(key))
        return QuestionExecutionResult(
            query,
            self.executed_reason,
            tuple(candidates),
            tuple(trace),
        )


class MemoryAwareQuestionDialogueRuntime:
    """管理完整 query 生命周期并调用统一问答链，支持 Memory OFF/ON 装配。"""

    def __init__(
            self,
            ctx: TrainContext,
            runtime: QuestionAnswerRuntime,
            *,
            trace_prefix: tuple[int, ...],
            source_records: SourceRecordRepository | None = None,
            ) -> None:
        """绑定同一上下文的问答 runtime 与只读 SourceRecord 回查入口。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("Memory dialogue ctx 类型错误")
        if not isinstance(runtime, QuestionAnswerRuntime):
            raise TypeError("Memory dialogue question runtime 类型错误")
        _strict_key(trace_prefix, label="Memory dialogue trace prefix")
        repository = source_records or SourceRecordRepository(ctx.backend)
        if not isinstance(repository, SourceRecordRepository):
            raise TypeError("Memory dialogue SourceRecord repository 类型错误")
        if repository.backend is not ctx.backend:
            raise ValueError("Memory dialogue SourceRecord 必须属于同一 backend")
        self._ctx = ctx
        self.runtime = runtime
        self.trace_prefix = trace_prefix
        self.source_records = repository

    @property
    def context(self) -> TrainContext:
        """返回该 dialogue caller 唯一绑定的运行上下文。"""
        return self._ctx

    def _selected_sources(
            self,
            run: QuestionAnswerRun,
            ) -> tuple[MemoryGenerationSource, ...]:
        """回查 G-01 实际采用项的全部 SourceRecord，不读取未选候选。"""
        if run.selection is None or run.planning_request is None:
            return ()
        selected_keys = set(run.selection.selected_candidate_keys)
        sources = {
            source
            for candidate in run.planning_request.candidates
            if candidate.stable_key() in selected_keys
            for evidence in candidate.memory_evidence
            for source in evidence.sources
        }
        ordered = tuple(sorted(sources, key=lambda item: item.stable_key()))
        for source in ordered:
            record = self.source_records.read(source.source_record_hash)
            if record.source_key != source.trace.source.stable_key():
                raise ValueError("Memory dialogue SourceRecord 回查来源漂移")
        return ordered

    def run(self, request: QuestionRequest) -> MemoryQuestionDialogueRun:
        """打开并最终关闭 query；成功、unsupported、postcheck 失败和异常均不泄漏状态。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("Memory dialogue 需要 QuestionRequest")
        scope = request.response_scope
        if scope.scope_kind != SCOPE_QUERY:
            raise ValueError("Memory dialogue response scope 必须是 query")
        if self._ctx.work_memory.active_query_scope is not None:
            raise RuntimeError("Memory dialogue 不得嵌套活动 query")
        self._ctx.scoped_identity_store.register_scope(scope)
        self._ctx.work_memory.begin_query(scope)
        try:
            question = self.runtime.run(request)
            sources = self._selected_sources(question)
            trace = (
                *self.trace_prefix,
                *_packed(scope.stable_key()),
                len(sources),
            )
            return MemoryQuestionDialogueRun(question, sources, trace)
        finally:
            if self._ctx.work_memory.active_generation_scope is not None:
                self._ctx.work_memory.end_generation()
            self._ctx.work_memory.end_query()


class MemoryQuestionSelectionCommitter:
    """只把 G-01 已选且完成 surface 的当前 frontier Memory 项写成 Use。"""

    def __init__(
            self,
            ctx: TrainContext,
            *,
            consumer: ObjectIdentity,
            input_observation_ref: MemoryObjectRef,
            influence_kind: MemoryLinkedRef,
            trace_prefix: tuple[int, ...],
            ) -> None:
        """绑定当前 query 的 A-10/M-08 owner、输入 Observation 和采用指令。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("Memory question committer ctx 类型错误")
        if (not isinstance(consumer, ObjectIdentity)
                or consumer.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise ValueError("Memory question committer consumer 必须是 MinimalInstruction")
        if not isinstance(input_observation_ref, MemoryObjectRef):
            raise TypeError("Memory question input Observation 引用类型错误")
        if not isinstance(influence_kind, MemoryLinkedRef):
            raise TypeError("Memory question influence_kind 类型错误")
        _strict_key(trace_prefix, label="Memory question committer trace prefix")
        if not isinstance(ctx.memory_use_runtime, MemoryUseRuntime):
            raise ValueError("Memory question committer 需要已安装 M-08 runtime")
        if ctx.attractor_runtime is None:
            raise ValueError("Memory question committer 需要已安装 A-10 runtime")
        self._ctx = ctx
        self.consumer = consumer
        self.input_observation_ref = input_observation_ref
        self.influence_kind = influence_kind
        self.trace_prefix = trace_prefix

    @staticmethod
    def _selected_memory_evidence(
            selection: AnswerContentSelection,
            ) -> tuple[tuple[tuple[int, ...], MemoryGenerationEvidence], ...]:
        """返回 G-01 已选候选携带的 Memory evidence，不读取未选候选。"""
        selected_keys = set(selection.selected_candidate_keys)
        selected = tuple(
            candidate
            for candidate in selection.request.candidates
            if candidate.stable_key() in selected_keys
        )
        if {item.stable_key() for item in selected} != selected_keys:
            raise ValueError("Memory question selection 含请求外候选")
        result = tuple(
            (candidate.stable_key(), evidence)
            for candidate in selected
            for evidence in candidate.memory_evidence
        )
        evidence_keys = tuple(item.stable_key() for _, item in result)
        if len(set(evidence_keys)) != len(evidence_keys):
            raise ValueError("同次选择不得重复采用同一 Memory evidence")
        return result

    def commit(
            self,
            request: QuestionRequest,
            query: QuestionQuery,
            result: QuestionExecutionResult,
            selection: AnswerContentSelection,
            generation: TypedGenerationExecution,
            ) -> MemoryGenerationCommitReport:
        """预检 frontier 后提交 consumed 与唯一 Use；无 Memory 采用时返回零提交。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("Memory question request 类型错误")
        if not isinstance(query, QuestionQuery) or query.request != request:
            raise ValueError("Memory question query 替换了原请求")
        if (not isinstance(result, QuestionExecutionResult)
                or result.query != query):
            raise ValueError("Memory question result 替换了同次 query")
        planning = result.planning_request()
        if (not isinstance(selection, AnswerContentSelection)
                or selection.request != planning):
            raise ValueError("Memory question selection 替换了查询结果")
        if (not isinstance(generation, TypedGenerationExecution)
                or generation.plan.request != planning
                or not generation.complete):
            raise ValueError("Memory question 只提交同次成功 generation")

        selected = self._selected_memory_evidence(selection)
        report_trace = (
            *self.trace_prefix,
            len(selected),
        )
        if not selected:
            return MemoryGenerationCommitReport(
                integer_tuple_fingerprint(
                    selection.stable_key(),
                    domain="question.commit.selection.v1"),
                integer_tuple_fingerprint(
                    generation.stable_key(),
                    domain="question.commit.generation.v1"),
                (),
                report_trace,
            )

        state = self._ctx.work_memory.require_attractor_state()
        if state.scope != request.response_scope:
            raise ValueError("Memory question A-10 state 不属于当前 response scope")
        activations = state.activations()
        by_activation: dict[tuple[int, ...], tuple[tuple[int, ...], MemoryGenerationEvidence]] = {}
        for candidate_key, evidence in selected:
            matches = tuple(
                activation
                for activation in activations
                if (activation.candidate == evidence.candidate
                    and activation.obligation.proposition == evidence.target)
            )
            if len(matches) != 1:
                raise ValueError("Memory evidence 未唯一绑定当前 A-10 activation")
            activation_key = matches[0].identity_key()
            if activation_key in by_activation:
                raise ValueError("同一 A-10 activation 不得形成多个 Memory Use")
            by_activation[activation_key] = (candidate_key, evidence)

        frontier = state.frontier()
        frontier_prefix = tuple(
            item.identity_key() for item in frontier[:len(by_activation)])
        if set(frontier_prefix) != set(by_activation):
            raise ValueError("已选 Memory evidence 不是当前 A-10 frontier 前缀")

        commits = []
        for activation_key in frontier_prefix:
            candidate_key, evidence = by_activation[activation_key]
            decision_trace = (
                *self.trace_prefix,
                len(commits) + 1,
            )
            processing = state.commit_consumption(AttractorConsumptionDecision(
                activation_key,
                self.consumer,
                state.protocol.consumed,
                decision_trace,
            ))
            used_at = LogicalTimestamp(
                state.current_timestamp.clock,
                state.current_timestamp.seq + processing.ordinal,
            )
            attribution = self._ctx.memory_use_runtime.record_selection_use(
                processing,
                input_observation_ref=self.input_observation_ref,
                influence_kind=self.influence_kind,
                used_at=used_at,
            )
            commits.append(MemoryGenerationUseCommit(
                candidate_key,
                evidence.stable_key(),
                processing.stable_key(),
                attribution.use.event.object_ref,
            ))
        return MemoryGenerationCommitReport(
            integer_tuple_fingerprint(
                selection.stable_key(),
                domain="question.commit.selection.v1"),
            integer_tuple_fingerprint(
                generation.stable_key(),
                domain="question.commit.generation.v1"),
            tuple(commits),
            report_trace,
        )


__all__ = [
    "MemoryAwareQuestionDialogueRuntime",
    "MemoryQuestionDialogueRun",
    "MemoryQuestionSelectionCommitter",
    "ResolvedMemoryQuestionExecutor",
]
