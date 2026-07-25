"""正式语言课程的 S-02 图物化、Evidence、S-03 绑定与 G-00 请求 owner。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisLedger,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import query_scope
from pure_integer_ai.cognition.shared.semantic_graph import SemanticGraph
from pure_integer_ai.cognition.shared.semantic_template_scope import (
    SemanticTemplateScopeDefinition,
    SemanticTemplateScopeGraph,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
)
from pure_integer_ai.cognition.shared.types import (
    MODALITY_LANGUAGE,
    InputPayload,
    ObserveResult,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    LocalSemanticRef,
    SemanticBuildResult,
    SemanticCandidateBuilder,
)
from pure_integer_ai.cognition.understanding.semantic_builder_graph import (
    MaterializedSemanticBuild,
    SemanticCandidateGraphAdapter,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationRequestDecision,
)
from pure_integer_ai.experiments.language_semantic_course import (
    ActiveSenseCourseView,
    LanguageSemanticCourseDecision,
    LanguageSemanticCourseInput,
    LanguageSemanticCourseProtocol,
    LanguageSemanticLesson,
)
from pure_integer_ai.experiments.language_semantic_query import (
    LanguageSemanticQueryProtocol,
    LanguageSemanticQueryRun,
    LanguageSemanticQueryRuntime,
)
from pure_integer_ai.experiments.language_sense_candidate_runtime import (
    SenseCandidateRecognitionTrace,
)
from pure_integer_ai.experiments.train_context import TrainContext


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为 mapper trace 的可变长片段增加长度边界。"""
    return len(key), *key


@dataclass(frozen=True)
class LanguageSemanticCourseRun:
    """一次正式课程映射的编译、图、Evidence、绑定和请求完整产物。"""

    input_value: LanguageSemanticCourseInput
    decision: LanguageSemanticCourseDecision
    build: SemanticBuildResult | None = None
    materialized: MaterializedSemanticBuild | None = None
    evidence: tuple[EvidenceRecord, ...] = ()
    bound_propositions: tuple[
        tuple[LocalSemanticRef, BoundProposition], ...] = ()
    request: GenerationPlanningRequest | None = None
    recovery: LanguageSemanticQueryRun | None = None

    def __post_init__(self) -> None:
        has_lesson = self.decision.lesson is not None
        training_products = (
            self.build is not None,
            self.materialized is not None,
            bool(self.evidence),
            bool(self.bound_propositions),
        )
        if has_lesson != all(training_products):
            raise ValueError("semantic course lesson 与完整运行产物状态不一致")
        if has_lesson and self.request is None:
            raise ValueError("semantic course lesson 缺 generation request")
        if not has_lesson and any(training_products):
            raise ValueError("无 semantic lesson 时不得携带训练产物")
        if has_lesson:
            if self.recovery is not None:
                raise ValueError("训练 lesson 不得同时携带只读 recovery")
            if not isinstance(self.build, SemanticBuildResult):
                raise TypeError("semantic course build 类型错误")
            if not isinstance(self.materialized, MaterializedSemanticBuild):
                raise TypeError("semantic course materialized 类型错误")
            if any(not isinstance(item, EvidenceRecord) for item in self.evidence):
                raise TypeError("semantic course evidence 产物类型错误")
            if any(not isinstance(local, LocalSemanticRef)
                   or not isinstance(bound, BoundProposition)
                   for local, bound in self.bound_propositions):
                raise TypeError("semantic course bound proposition 类型错误")
        elif self.recovery is None:
            if self.request is not None:
                raise ValueError("无 lesson/recovery 时不得携带 request")
        else:
            if not isinstance(self.recovery, LanguageSemanticQueryRun):
                raise TypeError("semantic course recovery 类型错误")
            if self.request != self.recovery.request:
                raise ValueError("semantic course recovery 替换了只读 request")


class LanguageSemanticCourseRuntime:
    """以课程显式声明驱动 S-02/S-03，并保留独立 H-00 Evidence 账。"""

    def __init__(
            self,
            ctx: TrainContext,
            protocol: LanguageSemanticCourseProtocol,
            *,
            ledger: HypothesisLedger | None = None,
            query_protocol: LanguageSemanticQueryProtocol | None = None,
            ) -> None:
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        if not isinstance(protocol, LanguageSemanticCourseProtocol):
            raise TypeError("semantic course protocol 类型错误")
        if ctx.span_index is None or ctx.occurrence_index is None:
            raise ValueError("semantic course runtime 需要 L-03 occurrence 和 L-04 Span")
        atomic, trace, template_scope = protocol.graph_protocols(
            ctx.graph_ontology)
        semantic_graph = SemanticGraph(ctx.graph_ontology, atomic)
        self.protocol = protocol
        self.builder = SemanticCandidateBuilder(
            ctx.span_index,
            protocol.builder,
            ctx.occurrence_index,
        )
        self.graph = SemanticCandidateGraphAdapter(semantic_graph, trace)
        self.template_scopes = SemanticTemplateScopeGraph(
            ctx.graph_ontology, template_scope)
        self.ledger = ledger or HypothesisLedger()
        self.substituter = PropositionSubstituter(protocol.substitution)
        self.query_protocol = query_protocol
        self.query_runtime = (
            None
            if query_protocol is None
            else LanguageSemanticQueryRuntime(
                self.graph,
                self.ledger,
                self.substituter,
                query_protocol,
                template_scopes=self.template_scopes,
                hypothesis_kind=protocol.builder.semantic_hypothesis_kind,
                builder=protocol.builder.builder,
            )
        )

    def process(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            input_payload: InputPayload,
            observation: ObserveResult,
            ) -> LanguageSemanticCourseRun:
        """整批预检课程声明后写 S-02 图和 Evidence，并形成同源 G-00 请求。"""
        input_value = self._input(
            ctx, item, input_payload, observation)
        decision = self.protocol.mapper.map(input_value)
        if not isinstance(decision, LanguageSemanticCourseDecision):
            raise TypeError("semantic course mapper 返回类型错误")
        lesson = decision.lesson
        if lesson is None:
            if input_value.read_only and self.query_runtime is not None:
                recovery = self.query_runtime.process(input_value)
                return LanguageSemanticCourseRun(
                    input_value,
                    decision,
                    request=recovery.request,
                    recovery=recovery,
                )
            return LanguageSemanticCourseRun(input_value, decision)
        if input_value.read_only:
            raise ValueError("评测只读调用不得重新注入 semantic lesson Evidence")
        self._validate_lesson_input(lesson, input_value)
        build = self.builder.compile(lesson.root_anchor, lesson.plan)
        candidates = self._candidate_by_local(build)
        template_scopes = self._template_scope_definitions(
            lesson, candidates)
        evidence = self._evidence(lesson, candidates)
        self._preflight_ledger(build, evidence)
        self.template_scopes.preflight_many(
            template_scopes,
            scope=build.scope,
            provenance_kind=self.protocol.provenance_kind,
            epistemic_origin=self.protocol.epistemic_origin,
            content_version=self.protocol.content_version,
            qualifiers=self.protocol.qualifiers,
        )
        bound = self._bound(lesson, build, candidates)
        request = self._request(
            lesson,
            input_value,
            build,
            candidates,
            evidence,
            bound,
        )
        materialized = self.graph.materialize(
            build,
            provenance_kind=self.protocol.provenance_kind,
            epistemic_origin=self.protocol.epistemic_origin,
            content_version=self.protocol.content_version,
            qualifiers=self.protocol.qualifiers,
        )
        self.template_scopes.materialize_many(
            template_scopes,
            scope=build.scope,
            provenance_kind=self.protocol.provenance_kind,
            epistemic_origin=self.protocol.epistemic_origin,
            content_version=self.protocol.content_version,
            qualifiers=self.protocol.qualifiers,
        )
        self._apply_ledger(self.ledger, build, evidence)
        return LanguageSemanticCourseRun(
            input_value,
            decision,
            build,
            materialized,
            evidence,
            tuple(sorted(bound.items(), key=lambda item: item[0])),
            request,
        )

    def clone_for_context(
            self, ctx: TrainContext,
            ) -> "LanguageSemanticCourseRuntime":
        """把协议、课程状态和 H-00 历史复制到独立评测上下文。"""
        mapper = self.protocol.mapper.clone_for_evaluation()
        cloned_protocol = LanguageSemanticCourseProtocol(
            self.protocol.builder,
            self.protocol.atomic_predicates,
            self.protocol.trace_predicates,
            self.protocol.scope_predicates,
            self.protocol.substitution,
            mapper,
            self.protocol.provenance_kind,
            self.protocol.epistemic_origin,
            self.protocol.content_version,
            self.protocol.qualifiers,
        )
        cloned_query = (
            None
            if self.query_protocol is None
            else self.query_protocol.clone_for_evaluation()
        )
        return LanguageSemanticCourseRuntime(
            ctx,
            cloned_protocol,
            ledger=self.ledger.clone(),
            query_protocol=cloned_query,
        )

    def state_key(self) -> tuple:
        """返回 mapper 与 H-00 账本状态，供 V-06 证明宿主零污染。"""
        query_state = (
            ()
            if self.query_protocol is None
            else self.query_protocol.state_key()
        )
        return (
            self.protocol.mapper.state_key(),
            query_state,
            self.ledger.state_key(),
        )

    @staticmethod
    def _input(
            ctx: TrainContext,
            item: CollectedItem,
            input_payload: InputPayload,
            observation: ObserveResult,
            ) -> LanguageSemanticCourseInput:
        """从正式 observation 投影无词面的 Span/Occurrence/Sense 输入。"""
        if item.modality != MODALITY_LANGUAGE:
            raise ValueError("semantic course runtime 只接受语言模态")
        if input_payload.source_ref is None:
            raise ValueError("semantic course runtime 缺少 SourceRef")
        if input_payload.occurrence_scope_identity is None:
            raise ValueError("semantic course runtime 缺少 occurrence scope")
        if input_payload.scope_identity is None:
            raise ValueError("semantic course runtime 缺少 runtime scope")
        active_senses = []
        for trace in observation.sense_candidate_traces:
            if not isinstance(trace, SenseCandidateRecognitionTrace):
                raise TypeError("ObserveResult 混入非法 Sense trace")
            if not trace.adopted:
                continue
            active_senses.append(ActiveSenseCourseView(
                trace.input_value.occurrence,
                trace.mapped.candidate,
                trace.mapped.predicted,
                trace.projection.candidate.hypothesis,
                trace.projection,
            ))
        return LanguageSemanticCourseInput(
            input_payload.source_ref,
            input_payload.occurrence_scope_identity,
            input_payload.scope_identity,
            tuple(observation.occurrence_refs),
            tuple(observation.span_refs),
            tuple(active_senses),
            ctx.scope_owner is not None,
        )

    def _validate_lesson_input(
            self,
            lesson: LanguageSemanticLesson,
            input_value: LanguageSemanticCourseInput,
            ) -> None:
        """要求 mapper 精确引用本次 typed anchor，不允许按排序私选外部对象。"""
        anchors = set(input_value.occurrences + input_value.spans)
        if lesson.root_anchor not in anchors:
            raise ValueError("semantic lesson root anchor 不属于当前 typed 输入")
        if (lesson.plan.upstream_hypothesis.observation != input_value.source
                or lesson.plan.upstream_hypothesis.scope
                != input_value.occurrence_scope):
            raise ValueError("semantic lesson upstream 与当前来源或 scope 不一致")

    @staticmethod
    def _candidate_by_local(build: SemanticBuildResult) -> dict:
        """按课程局部 Proposition ref 建唯一候选索引。"""
        result = {
            candidate.spec.local_ref: candidate
            for candidate in build.propositions
        }
        if len(result) != len(build.propositions):
            raise RuntimeError("S-02 build 产生重复局部 Proposition")
        return result

    @staticmethod
    def _template_scope_definitions(
            lesson: LanguageSemanticLesson,
            candidates: dict,
            ) -> tuple[SemanticTemplateScopeDefinition, ...]:
        """要求课程逐 Proposition 显式声明空或非空词法 scope。"""
        declarations = {}
        for item in lesson.template_scopes:
            if item.proposition in declarations:
                raise ValueError("semantic lesson 重复声明 template scope")
            declarations[item.proposition] = item
        if set(declarations) != set(candidates):
            raise ValueError("semantic lesson 必须完整声明每个 Proposition scope")
        return tuple(
            SemanticTemplateScopeDefinition(
                candidates[local].definition.proposition,
                declarations[local].scope,
                declarations[local].introduced_binders,
            )
            for local in sorted(candidates)
        )

    @staticmethod
    def _evidence(lesson, candidates) -> tuple[EvidenceRecord, ...]:
        """把局部 Evidence 规格绑定到 S-02 实际 Hypothesis。"""
        result = []
        for spec in lesson.evidence:
            candidate = candidates.get(spec.proposition)
            if candidate is None:
                raise ValueError("课程 Evidence 指向 plan 外的 Proposition")
            result.append(EvidenceRecord(
                spec.evidence_id,
                candidate.hypothesis,
                spec.stance,
                spec.reason_key,
                spec.source,
                spec.timestamp_seq,
                spec.payload,
                spec.supersedes_evidence_id,
            ))
        if len({item.evidence_id for item in result}) != len(result):
            raise ValueError("同一 semantic lesson 不得重复 evidence_id")
        return tuple(sorted(result, key=lambda item: item.evidence_id))

    def _preflight_ledger(
            self,
            build: SemanticBuildResult,
            evidence: tuple[EvidenceRecord, ...],
            ) -> None:
        """在任何语义图写入前用克隆账本验证全部 Evidence。"""
        probe = self.ledger.clone()
        self._apply_ledger(probe, build, evidence)

    @staticmethod
    def _apply_ledger(
            ledger: HypothesisLedger,
            build: SemanticBuildResult,
            evidence: tuple[EvidenceRecord, ...],
            ) -> None:
        """按完整稳定键登记 S-02 候选并追加课程 Evidence。"""
        for candidate in sorted(
                build.propositions,
                key=lambda item: item.hypothesis.stable_key()):
            ledger.register(candidate.hypothesis)
        for item in sorted(evidence, key=lambda value: value.stable_key()):
            ledger.append_evidence(item)

    def _bound(self, lesson, build, candidates) -> dict:
        """按显式 Binder scope 形成所有 generation candidate 的运行期 bound view。"""
        scope_by_local = {}
        for item in lesson.template_scopes:
            if item.proposition in scope_by_local:
                raise ValueError("semantic lesson 重复声明 template scope")
            scope_by_local[item.proposition] = item.introduced_binders
        unknown_scopes = set(scope_by_local).difference(candidates)
        if unknown_scopes:
            raise ValueError("template scope 指向 plan 外的 Proposition")
        templates = PropositionTemplateGraph(tuple(
            ScopedPropositionTemplate(
                candidate.definition,
                candidate.spec.structure,
                scope_by_local.get(local, ()),
            )
            for local, candidate in candidates.items()
        ))
        bound = {}
        for local in lesson.generation_candidates:
            candidate = candidates.get(local)
            if candidate is None:
                raise ValueError("generation candidate 指向 plan 外的 Proposition")
            bound[local] = self.substituter.substitute(
                candidate.definition.proposition,
                templates,
                lesson.binding_environment,
            )
        return bound

    def _request(
            self, lesson, input_value, build, candidates, evidence, bound):
        """只从 active H-00 快照与未被替代 Evidence 构造 G-00 请求。"""
        request_scope = query_scope(1, parent=input_value.runtime_scope)
        probe = self.ledger.clone()
        self._apply_ledger(probe, build, evidence)
        generation_candidates = []
        for local in lesson.generation_candidates:
            candidate = candidates[local]
            snapshot = probe.snapshot(candidate.hypothesis)
            if snapshot.lifecycle != LIFECYCLE_ACTIVE:
                raise ValueError("非 active semantic candidate 不得进入 generation request")
            active_ids = set(
                snapshot.support_evidence_ids
                + snapshot.refute_evidence_ids
                + snapshot.unknown_evidence_ids
            )
            active_evidence = tuple(
                item for item in probe.evidence_history(candidate.hypothesis)
                if item.evidence_id in active_ids
            )
            if not active_evidence:
                raise ValueError("generation candidate 缺少课程 active Evidence")
            generation_candidates.append(GenerationCandidate(
                bound[local],
                LogicEvidenceState.from_status(snapshot.epistemic_status),
                candidate.definition.source,
                request_scope,
                active_evidence,
            ))
        goal_candidate = candidates.get(lesson.goal_proposition)
        if goal_candidate is None:
            raise ValueError("semantic lesson goal 指向 plan 外的 Proposition")
        goal = AnswerGenerationGoal(
            lesson.goal_kind,
            bound[lesson.goal_proposition],
            lesson.required,
            goal_candidate.definition.source,
            request_scope,
            lesson.target_branch,
        )
        return GenerationPlanningRequest(goal, tuple(generation_candidates))


class SemanticCourseGenerationRequestMapper:
    """把同次 ObserveResult 的 semantic course run 交给 production owner。"""

    def build(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            input_payload: InputPayload,
            observation: ObserveResult,
            ) -> ProductionGenerationRequestDecision:
        """核验来源和 runtime scope 后转交请求，不重建或替换语义候选。"""
        run = observation.semantic_course_run
        if not isinstance(run, LanguageSemanticCourseRun):
            raise RuntimeError("production semantic mapper 缺少同次课程运行结果")
        if item.source_ref != run.input_value.source:
            raise ValueError("production semantic mapper 的 item 来源发生替换")
        if input_payload.scope_identity != run.input_value.runtime_scope:
            raise ValueError("production semantic mapper 的 runtime scope 发生替换")
        active_query = ctx.work_memory.active_query_scope
        if (active_query is not None and run.request is not None
                and run.request.goal.scope != active_query):
            raise ValueError("production semantic mapper 的请求未绑定当前 query scope")
        trace = (
            1,
            *_packed(run.decision.trace),
            0 if run.request is None else 1,
        )
        return ProductionGenerationRequestDecision(
            run.decision.reason,
            trace,
            run.request,
        )


def install_language_semantic_course_runtime(
        ctx: TrainContext,
        protocol: LanguageSemanticCourseProtocol,
        query_protocol: LanguageSemanticQueryProtocol | None = None,
        ) -> LanguageSemanticCourseRuntime:
    """在正式上下文安装 run-local 语义课程 owner。"""
    if ctx.language_semantic_course_runtime is not None:
        raise ValueError("TrainContext 已安装 language semantic course runtime")
    runtime = LanguageSemanticCourseRuntime(
        ctx,
        protocol,
        query_protocol=query_protocol,
    )
    ctx.language_semantic_course_runtime = runtime
    return runtime


__all__ = [
    "LanguageSemanticCourseRun",
    "LanguageSemanticCourseRuntime",
    "SemanticCourseGenerationRequestMapper",
    "install_language_semantic_course_runtime",
]
