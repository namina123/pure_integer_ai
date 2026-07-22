"""旧结构发现输出到 H-05 typed 候选全链的显式课程适配边界。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveredOperator,
    Recognition,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateHistoryUnavailableError,
    CandidateLearningOutcome,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_OCCURRENCE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.understanding.language_candidate import (
    CueStructureCandidateSpec,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.train_context import TrainContext


@dataclass(frozen=True)
class StructureFormationInput:
    """交给课程 mapper 的 legacy operator 与独立 forming 来源全集。"""

    operator_name: str
    skeleton_ref: ConceptRef
    arity: int
    sample_count: int
    forming_roots: tuple[ConceptRef, ...]
    forming_sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operator_name, str) or not self.operator_name:
            raise ValueError("operator_name 必须是非空字符串")
        assert_int(
            *self.skeleton_ref,
            self.arity,
            self.sample_count,
            _where="StructureFormationInput",
        )
        if self.arity < 0 or self.sample_count <= 0:
            raise ValueError("structure arity/sample_count 非法")
        if not isinstance(self.forming_roots, tuple):
            raise TypeError("forming_roots 必须是 ConceptRef tuple")
        for root in self.forming_roots:
            if not isinstance(root, tuple) or len(root) != 2:
                raise ValueError("forming root 必须是二元 ConceptRef")
            assert_int(*root, _where="StructureFormationInput.forming_root")
        if not isinstance(self.forming_sources, tuple):
            raise TypeError("forming_sources 必须是 SourceRef tuple")
        if any(not isinstance(item, SourceRef)
               for item in self.forming_sources):
            raise TypeError("forming_sources 只能包含 SourceRef")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("forming_sources 必须按真实 SourceRef 去重")


@dataclass(frozen=True)
class StructureRecognitionInput:
    """不含图或候选状态的 held-out 结构识别输入。"""

    operator_name: str
    input_root: ConceptRef
    concept_binding: tuple
    observation: SourceRef
    scope: ScopeIdentity
    occurrences: tuple[ObjectIdentity, ...]
    token_spans: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operator_name, str) or not self.operator_name:
            raise ValueError("operator_name 必须是非空字符串")
        if not isinstance(self.input_root, tuple) or len(self.input_root) != 2:
            raise ValueError("input_root 必须是二元 ConceptRef")
        assert_int(*self.input_root, _where="StructureRecognitionInput.input_root")
        if not isinstance(self.concept_binding, tuple):
            raise TypeError("concept_binding 必须是 tuple")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("recognition observation 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("recognition scope 必须是 ScopeIdentity")
        if self.scope.source != self.observation:
            raise ValueError("recognition scope 必须指向同一 observation")
        if (not isinstance(self.occurrences, tuple)
                or not self.occurrences
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_OCCURRENCE
                       for item in self.occurrences)):
            raise ValueError("structure recognition 必须保存真实 Occurrence 全集")
        if (not isinstance(self.token_spans, tuple)
                or len(self.token_spans) != len(self.occurrences)):
            raise ValueError("structure token span 必须与 Occurrence 一一对应")
        previous_end = -1
        for index, span in enumerate(self.token_spans):
            if not isinstance(span, tuple) or len(span) != 3:
                raise ValueError("structure token span 必须是三元组")
            start, end, ordinal = span
            assert_int(
                start, end, ordinal,
                _where=f"StructureRecognitionInput.token_spans[{index}]",
            )
            if (type(start) is not int or type(end) is not int
                    or type(ordinal) is not int or start < 0
                    or end <= start or ordinal < 0 or start < previous_end):
                raise ValueError("structure token span 范围或顺序非法")
            previous_end = end


@dataclass(frozen=True)
class MappedStructureRecognition:
    """mapper 产出的候选定位、冻结输入、目标和独立 reveal。"""

    candidate: ObjectIdentity
    visible_inputs: tuple[ObjectIdentity, ...]
    predicted: ObjectIdentity
    revealed: RevealedObjectObservation
    archive_refuted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ObjectIdentity):
            raise TypeError("mapped candidate 必须是 ObjectIdentity")
        if not isinstance(self.visible_inputs, tuple) or not self.visible_inputs:
            raise ValueError("mapped visible_inputs 必须是非空对象 tuple")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.visible_inputs):
            raise TypeError("mapped visible_inputs 只能包含 ObjectIdentity")
        if not isinstance(self.predicted, ObjectIdentity):
            raise TypeError("mapped predicted 必须是 ObjectIdentity")
        if not isinstance(self.revealed, RevealedObjectObservation):
            raise TypeError("mapped revealed 类型非法")
        if type(self.archive_refuted) is not bool:
            raise TypeError("archive_refuted 必须是 bool")


@runtime_checkable
class StructureCandidateCourseMapper(Protocol):
    """课程实现的 legacy 全集迁移和独立 recognition 映射协议。"""

    def form(
            self,
            input_value: StructureFormationInput,
            ) -> tuple[CueStructureCandidateSpec, ...]:
        """把一个旧 operator 全量映射为零个或多个完整 typed 候选。"""
        ...

    def recognize(
            self,
            input_value: StructureRecognitionInput,
            ) -> tuple[MappedStructureRecognition, ...]:
        """只依据来源输入构造 prediction/reveal，不读取候选图或旧 tally。"""
        ...

    def clone_for_evaluation(self) -> "StructureCandidateCourseMapper":
        """返回不共享可变课程状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple:
        """返回 mapper 的可比较课程状态供 V-06 隔离核验。"""
        ...


@dataclass(frozen=True)
class StructureCandidateIntegrationReport:
    """生产结构候选形成与 recognition 三态计数。"""

    formation_inputs: int = 0
    candidates_registered: int = 0
    recognition_inputs: int = 0
    prediction_count: int = 0
    support_count: int = 0
    refute_count: int = 0
    unknown_count: int = 0
    recovered_read_count: int = 0
    adopted_count: int = 0


@dataclass(frozen=True)
class StructureCandidateRecognitionTrace:
    """一条 legacy 对齐输入经 typed mapper 后的写链或只读恢复结果。"""

    recognition: Recognition
    input_value: StructureRecognitionInput
    mapped: MappedStructureRecognition
    outcome: CandidateLearningOutcome | None
    projection: CandidateGraphProjection | None
    read_only: bool
    adopted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.recognition, Recognition):
            raise TypeError("trace recognition 类型非法")
        if not isinstance(self.input_value, StructureRecognitionInput):
            raise TypeError("trace input_value 类型非法")
        if not isinstance(self.mapped, MappedStructureRecognition):
            raise TypeError("trace mapped 类型非法")
        if self.outcome is not None and not isinstance(
                self.outcome, CandidateLearningOutcome):
            raise TypeError("trace outcome 类型非法")
        if (self.projection is not None
                and not isinstance(self.projection, CandidateGraphProjection)):
            raise TypeError("trace projection 类型非法")
        if type(self.read_only) is not bool or type(self.adopted) is not bool:
            raise TypeError("trace read_only/adopted 必须是 bool")
        if self.read_only and self.projection is None:
            raise ValueError("只读恢复 trace 必须携带图内 lifecycle 投影")
        if self.adopted and self.projection is None:
            raise ValueError("adopted trace 必须携带 active 图投影")


def _formation_input(
        operator: DiscoveredOperator,
        root_sources: dict[ConceptRef, SourceRef],
        ) -> StructureFormationInput | None:
    """恢复 forming root 的真实 SourceRef；任一缺失时拒绝伪造 aggregate 来源。"""
    if not operator.forming_roots:
        return None
    sources: dict[SourceRef, None] = {}
    for root in operator.forming_roots:
        source = root_sources.get(root)
        if source is None:
            return None
        sources[source] = None
    return StructureFormationInput(
        operator.name,
        operator.skeleton_ref,
        operator.arity,
        operator.sample_count,
        operator.forming_roots,
        tuple(sorted(sources, key=SourceRef.stable_key)),
    )


def register_structure_candidates(
        ctx: TrainContext,
        discovered: tuple[DiscoveredOperator, ...], *,
        root_sources: dict[ConceptRef, SourceRef],
        mapper: StructureCandidateCourseMapper,
        ) -> tuple[int, int, int]:
    """登记 forming unknown；遇已恢复图时只核验定义并阻断伪续写。"""
    if ctx.structure_candidate_runtime is None:
        raise ValueError("structure candidate runtime 尚未安装")
    if not isinstance(mapper, StructureCandidateCourseMapper):
        raise TypeError("mapper 必须实现 StructureCandidateCourseMapper")
    formation_count = 0
    registered_count = 0
    recovered_count = 0
    for operator in discovered:
        formation = _formation_input(operator, root_sources)
        if formation is None:
            continue
        formation_count += 1
        specs = mapper.form(formation)
        if not isinstance(specs, tuple):
            raise TypeError("structure mapper.form 必须返回 tuple")
        for spec in specs:
            if not isinstance(spec, CueStructureCandidateSpec):
                raise TypeError("structure mapper.form 返回了非法候选规格")
            if spec.forming_sources != formation.forming_sources:
                raise ValueError("typed 候选必须保留 forming SourceRef 全集")
            definition = spec.definition(
                ctx.structure_candidate_consumer.protocol)
            timestamp_base = ctx.structure_candidate_runtime.next_timestamps(1)[0]
            try:
                ctx.structure_candidate_runtime.register(
                    definition,
                    timestamp_base=timestamp_base,
                )
            except CandidateHistoryUnavailableError:
                ctx.structure_candidate_runtime.read_only_definition(
                    definition)
                recovered_count += 1
            else:
                registered_count += 1
    return formation_count, registered_count, recovered_count


def recognize_structure_candidates(
        ctx: TrainContext,
        recognitions: tuple[Recognition, ...], *,
        origin_sources: dict[ConceptRef, SourceRef],
        origin_of: dict[ConceptRef, ConceptRef],
        origin_occurrences: dict[ConceptRef, tuple[ObjectIdentity, ...]],
        origin_token_spans: dict[
            ConceptRef, tuple[tuple[int, int, int], ...]],
        mapper: StructureCandidateCourseMapper,
        ) -> tuple[StructureCandidateRecognitionTrace, ...]:
    """执行 typed 全链；图恢复但历史缺失时只读投影且不追加 Evidence。"""
    if ctx.structure_candidate_runtime is None:
        raise ValueError("structure candidate runtime 尚未安装")
    if not isinstance(mapper, StructureCandidateCourseMapper):
        raise TypeError("mapper 必须实现 StructureCandidateCourseMapper")
    traces: list[StructureCandidateRecognitionTrace] = []
    for recognition in recognitions:
        origin = origin_of.get(recognition.input_root, recognition.input_root)
        observation = origin_sources.get(origin)
        if observation is None:
            continue
        occurrences = origin_occurrences.get(origin, ())
        token_spans = origin_token_spans.get(origin, ())
        if not occurrences or len(occurrences) != len(token_spans):
            continue
        scope = document_scope(observation)
        ctx.scoped_identity_store.register_scope(scope)
        input_value = StructureRecognitionInput(
            recognition.operator_name,
            origin,
            recognition.concept_binding,
            observation,
            scope,
            occurrences,
            token_spans,
        )
        mapped_values = mapper.recognize(input_value)
        if not isinstance(mapped_values, tuple):
            raise TypeError("structure mapper.recognize 必须返回 tuple")
        for mapped in mapped_values:
            if not isinstance(mapped, MappedStructureRecognition):
                raise TypeError("structure mapper.recognize 返回了非法结果")
            visible_occurrences = tuple(
                item for item in mapped.visible_inputs
                if item.object_kind == OBJECT_OCCURRENCE)
            if (not visible_occurrences
                    or any(item not in input_value.occurrences
                           for item in visible_occurrences)):
                raise ValueError("structure prediction 必须保存当前真实 Occurrence")
            try:
                hypothesis = (
                    ctx.structure_candidate_runtime.hypothesis_for_candidate(
                        mapped.candidate))
            except CandidateHistoryUnavailableError:
                projection = (
                    ctx.structure_candidate_runtime
                    .lifecycle_projection_if_available(mapped.candidate))
                if projection is None:
                    continue
                definition = projection.candidate.definition
                outcome = None
                read_only = True
            else:
                definition = ctx.structure_candidate_runtime.engine.definition(
                    hypothesis)
                outcome = None
                projection = None
                read_only = False
            target_binding = (
                ctx.structure_candidate_consumer.protocol.target_relation.binding(
                    mapped.predicted))
            if target_binding not in definition.bindings:
                raise ValueError("prediction target 不属于候选的 typed relation 字段")
            if not read_only:
                evidence_seq, decision_seq, projection_seq = (
                    ctx.structure_candidate_runtime.next_timestamps(3))
                outcome = ctx.structure_candidate_runtime.recognize(
                    hypothesis,
                    observation=input_value.observation,
                    scope=input_value.scope,
                    event_key=mapped.revealed.event_key,
                    visible_inputs=mapped.visible_inputs,
                    predicted=mapped.predicted,
                    revealed=mapped.revealed,
                    timestamp_seq=evidence_seq,
                    resolve_timestamp_seq=decision_seq,
                    projection_timestamp_seq=projection_seq,
                    archive_refuted=mapped.archive_refuted,
                )
                projection = outcome.projection
                if projection is None:
                    projection = (
                        ctx.structure_candidate_runtime
                        .lifecycle_projection_if_available(mapped.candidate))
            adopted = (
                projection is not None
                and projection.state
                == ctx.candidate_projection_graph.protocol.active_state)
            traces.append(StructureCandidateRecognitionTrace(
                recognition,
                input_value,
                mapped,
                outcome,
                projection,
                read_only,
                adopted,
            ))
    return tuple(traces)


def integration_report(
        *, formation_inputs: int, candidates_registered: int,
        candidates_recovered: int,
        recognition_inputs: int,
        traces: tuple[StructureCandidateRecognitionTrace, ...],
        ) -> StructureCandidateIntegrationReport:
    """从全链 outcome 派生三态报告，不以旧 tally 或边 tier 计成功。"""
    outcomes = tuple(
        item.outcome for item in traces if item.outcome is not None)
    stances = tuple(item.verification.stance for item in outcomes)
    return StructureCandidateIntegrationReport(
        formation_inputs,
        candidates_registered,
        recognition_inputs,
        len(outcomes),
        stances.count(EVIDENCE_SUPPORT),
        stances.count(EVIDENCE_REFUTE),
        stances.count(EVIDENCE_UNKNOWN),
        candidates_recovered + sum(item.read_only for item in traces),
        sum(item.adopted for item in traces),
    )


__all__ = [
    "MappedStructureRecognition",
    "StructureCandidateCourseMapper",
    "StructureCandidateIntegrationReport",
    "StructureCandidateRecognitionTrace",
    "StructureFormationInput",
    "StructureRecognitionInput",
    "integration_report",
    "recognize_structure_candidates",
    "register_structure_candidates",
]
