"""旧 sense 目录到 H-05 typed 候选及 occurrence Evidence 的课程边界。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateHistoryUnavailableError,
    CandidateLearningOutcome,
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
    OBJECT_LANGUAGE_ATOM,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_OCCURRENCE,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.understanding.language_candidate import (
    SenseCandidateSpec,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.sense_candidates import (
    read_sense_candidates,
    record_legacy_sense_bridge,
    sense_surface_hash,
)


def _legacy_ref(value: ConceptRef, *, where: str) -> ConceptRef:
    """校验旧目录引用只作完整迁移键，不把它冒充 Sense 或 Concept 身份。"""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{where} 必须是二元 ConceptRef")
    assert_int(*value, _where=where)
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{where} 必须使用严格正整数")
    return value


def _atom_belongs_to_branch(
        atom: ObjectIdentity, branch: ObjectIdentity) -> bool:
    """按公开身份编码核验 LanguageAtom 确实属于 mapper 声明的分支。"""
    if (atom.object_kind != OBJECT_LANGUAGE_ATOM
            or branch.object_kind != OBJECT_LANGUAGE_BRANCH
            or atom.owner != branch.owner
            or atom.versions != branch.versions):
        return False
    prefix = (len(branch.components), *branch.components)
    return atom.components[:len(prefix)] == prefix


@dataclass(frozen=True)
class LegacySenseCandidateInput:
    """旧目录中的一个候选及其只供审计的统计列。"""

    legacy_ref: ConceptRef
    base_count: int
    success_count: int
    observation_count: int

    def __post_init__(self) -> None:
        _legacy_ref(self.legacy_ref, where="LegacySenseCandidateInput.legacy_ref")
        assert_int(
            self.base_count,
            self.success_count,
            self.observation_count,
            _where="LegacySenseCandidateInput",
        )
        if any(type(item) is not int or item < 0 for item in (
                self.base_count,
                self.success_count,
                self.observation_count)):
            raise ValueError("旧 sense 统计必须为非负严格整数")


@dataclass(frozen=True)
class SenseFormationInput:
    """交给课程 mapper 的 surface 提示、分支提示和旧候选全集。"""

    runtime_language: int
    surface: str
    branch_hint: ObjectIdentity | None
    representation_hint: ObjectIdentity | None
    legacy_candidates: tuple[LegacySenseCandidateInput, ...]

    def __post_init__(self) -> None:
        assert_int(self.runtime_language, _where="SenseFormationInput.language")
        if type(self.runtime_language) is not int or self.runtime_language <= 0:
            raise ValueError("runtime_language 必须为严格正整数")
        if not isinstance(self.surface, str) or not self.surface:
            raise ValueError("sense formation surface 必须是非空字符串")
        if (self.branch_hint is not None
                and self.branch_hint.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ValueError("branch_hint 必须是 LanguageBranch 或 None")
        if (self.representation_hint is not None
                and self.representation_hint.object_kind != OBJECT_REPRESENTATION):
            raise ValueError("representation_hint 必须是 Representation 或 None")
        if (not isinstance(self.legacy_candidates, tuple)
                or not self.legacy_candidates):
            raise ValueError("sense formation 必须携带非空旧候选全集")
        if any(not isinstance(item, LegacySenseCandidateInput)
               for item in self.legacy_candidates):
            raise TypeError("legacy_candidates 类型非法")
        refs = tuple(item.legacy_ref for item in self.legacy_candidates)
        if len(set(refs)) != len(refs):
            raise ValueError("旧 sense 候选全集不得重复 ref")


@dataclass(frozen=True)
class MappedSenseFormation:
    """一个旧候选到 branch/LanguageAtom/Sense/Concept 的完整映射。"""

    legacy_ref: ConceptRef
    branch: ObjectIdentity
    spec: SenseCandidateSpec

    def __post_init__(self) -> None:
        _legacy_ref(self.legacy_ref, where="MappedSenseFormation.legacy_ref")
        if not isinstance(self.branch, ObjectIdentity):
            raise TypeError("mapped sense branch 必须是 ObjectIdentity")
        if self.branch.object_kind != OBJECT_LANGUAGE_BRANCH:
            raise ValueError("mapped sense branch 必须是 LanguageBranch")
        if not isinstance(self.spec, SenseCandidateSpec):
            raise TypeError("mapped sense spec 类型非法")
        if not _atom_belongs_to_branch(self.spec.atom, self.branch):
            raise ValueError("mapped LanguageAtom 不属于声明的语言分支")


@dataclass(frozen=True)
class SenseRecognitionInput:
    """不含候选状态的真实 occurrence、表示提示和可用候选全集。"""

    runtime_language: int
    surface: str
    observation: SourceRef
    scope: ScopeIdentity
    occurrence: ObjectIdentity
    representation: ObjectIdentity | None
    span_inputs: tuple[ObjectIdentity, ...]
    available_candidates: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        assert_int(self.runtime_language, _where="SenseRecognitionInput.language")
        if type(self.runtime_language) is not int or self.runtime_language <= 0:
            raise ValueError("runtime_language 必须为严格正整数")
        if not isinstance(self.surface, str) or not self.surface:
            raise ValueError("sense recognition surface 必须是非空字符串")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("sense recognition observation 类型非法")
        if (not isinstance(self.scope, ScopeIdentity)
                or self.scope.source != self.observation):
            raise ValueError("sense recognition scope 必须指向 observation")
        if (not isinstance(self.occurrence, ObjectIdentity)
                or self.occurrence.object_kind != OBJECT_OCCURRENCE):
            raise ValueError("sense recognition 必须绑定真实 Occurrence")
        if (self.representation is not None
                and self.representation.object_kind != OBJECT_REPRESENTATION):
            raise ValueError("sense recognition representation 类型非法")
        for name, values in (
                ("span_inputs", self.span_inputs),
                ("available_candidates", self.available_candidates)):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} 必须是 ObjectIdentity tuple")
            if any(not isinstance(item, ObjectIdentity) for item in values):
                raise TypeError(f"{name} 含非法对象")
        if not self.available_candidates:
            raise ValueError("sense recognition 必须携带已形成候选")
        if len(set(self.available_candidates)) != len(self.available_candidates):
            raise ValueError("available_candidates 不得重复")


@dataclass(frozen=True)
class MappedSenseRecognition:
    """mapper 产出的候选定位、冻结输入、Concept 预测和独立揭示。"""

    candidate: ObjectIdentity
    visible_inputs: tuple[ObjectIdentity, ...]
    predicted: ObjectIdentity
    revealed: RevealedObjectObservation
    archive_refuted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ObjectIdentity):
            raise TypeError("mapped Sense candidate 类型非法")
        if (not isinstance(self.visible_inputs, tuple)
                or not self.visible_inputs
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.visible_inputs)):
            raise ValueError("mapped Sense visible_inputs 必须是非空对象 tuple")
        if not isinstance(self.predicted, ObjectIdentity):
            raise TypeError("mapped Sense predicted 类型非法")
        if not isinstance(self.revealed, RevealedObjectObservation):
            raise TypeError("mapped Sense revealed 类型非法")
        if type(self.archive_refuted) is not bool:
            raise TypeError("archive_refuted 必须是 bool")


@runtime_checkable
class SenseCandidateCourseMapper(Protocol):
    """课程实现的旧目录全集迁移和独立 occurrence 核验协议。"""

    def form(
            self, input_value: SenseFormationInput,
            ) -> tuple[MappedSenseFormation, ...]:
        """把一个 surface 的旧候选全集映射为完整 typed 候选。"""
        ...

    def recognize(
            self, input_value: SenseRecognitionInput,
            ) -> tuple[MappedSenseRecognition, ...]:
        """只依据来源输入给出 prediction/reveal，不读取 active 投影。"""
        ...

    def clone_for_evaluation(self) -> "SenseCandidateCourseMapper":
        """返回不共享可变课程状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple:
        """返回 mapper 的完整可比较课程状态。"""
        ...


@dataclass(frozen=True)
class SenseCandidateRecognitionTrace:
    """一次 Sense recognition 的完整写链或只读恢复投影。"""

    input_value: SenseRecognitionInput
    mapped: MappedSenseRecognition
    outcome: CandidateLearningOutcome | None
    projection: CandidateGraphProjection
    read_only: bool
    adopted: bool


@dataclass(frozen=True)
class SenseCandidateIntegrationReport:
    """typed Sense 形成、预测、Evidence、恢复读取和采用计数。"""

    formation_inputs: int = 0
    candidates_registered: int = 0
    candidates_recovered: int = 0
    recognition_inputs: int = 0
    prediction_count: int = 0
    support_count: int = 0
    refute_count: int = 0
    unknown_count: int = 0
    recovered_read_count: int = 0
    adopted_count: int = 0


def _mapped_key(mapped: MappedSenseRecognition) -> tuple:
    """构造重复 round 精确核验所需的完整 mapper 输出键。"""
    return (
        mapped.candidate.stable_key(),
        tuple(item.stable_key() for item in mapped.visible_inputs),
        mapped.predicted.stable_key(),
        mapped.revealed,
        mapped.archive_refuted,
    )


class SenseCandidateCourseRuntime:
    """持有课程 mapper、surface 路由和每个来源只计一次的 typed Sense 编排。"""

    def __init__(self, mapper: SenseCandidateCourseMapper) -> None:
        if not isinstance(mapper, SenseCandidateCourseMapper):
            raise TypeError("mapper 必须实现 SenseCandidateCourseMapper")
        self.mapper = mapper
        self._formations: dict[
            tuple[int, str], tuple[MappedSenseFormation, ...]] = {}
        self._processed: dict[
            tuple[ObjectIdentity, SourceRef, tuple[int, ...]],
            tuple[tuple, SenseCandidateRecognitionTrace],
        ] = {}
        self._formation_inputs = 0
        self._registered = 0
        self._recovered = 0

    def form_legacy(
            self, ctx, *, runtime_language: int, surface: str,
            ) -> tuple[MappedSenseFormation, ...]:
        """读取一个旧候选全集，经 mapper 全量预检后登记 forming 或恢复只读定义。"""
        rows = read_sense_candidates(
            ctx.backend, ctx.space_id, sense_surface_hash(surface))
        if not rows:
            return ()
        provider = (
            None if ctx.word_form_providers is None
            else ctx.word_form_providers.provider(runtime_language))
        branch_hint = (
            None if provider is None
            else ctx.graph_ontology.identity_of(provider.branch))
        representation_hint = None
        if provider is not None:
            representation_ref = provider.index.lookup(
                surface, branch=provider.branch)
            if representation_ref is not None:
                representation_hint = ctx.graph_ontology.identity_of(
                    representation_ref)
        formation_input = SenseFormationInput(
            runtime_language,
            surface,
            branch_hint,
            representation_hint,
            tuple(LegacySenseCandidateInput(
                legacy_ref, base, success, observed)
                for legacy_ref, base, success, observed in rows),
        )
        mapped_values = self.mapper.form(formation_input)
        if not isinstance(mapped_values, tuple):
            raise TypeError("sense mapper.form 必须返回 tuple")
        if any(not isinstance(item, MappedSenseFormation)
               for item in mapped_values):
            raise TypeError("sense mapper.form 返回了非法映射")
        legacy_refs = {item.legacy_ref for item in formation_input.legacy_candidates}
        mapped_refs = tuple(item.legacy_ref for item in mapped_values)
        if len(set(mapped_refs)) != len(mapped_refs) or set(mapped_refs) != legacy_refs:
            raise ValueError("sense mapper 必须无重复地精确覆盖旧候选全集")
        key = (runtime_language, surface)
        existing = self._formations.get(key)
        ordered = tuple(sorted(
            mapped_values, key=lambda item: item.legacy_ref))
        if existing is not None:
            if existing != ordered:
                raise ValueError("同一语言和 surface 的 typed Sense 映射发生漂移")
            return existing

        definitions = tuple(
            item.spec.definition(ctx.sense_candidate_consumer.protocol)
            for item in ordered)
        probe = ctx.sense_candidate_runtime.engine.clone()
        timestamp_base = ctx.sense_candidate_runtime.next_timestamps(1)[0]
        for definition in definitions:
            probe.register(definition, timestamp_base=timestamp_base)
            timestamp_base += max(len(definition.forming_sources), 1)

        registered = 0
        recovered = 0
        for mapped, definition in zip(ordered, definitions, strict=True):
            timestamp = ctx.sense_candidate_runtime.next_timestamps(1)[0]
            try:
                ctx.sense_candidate_runtime.register(
                    definition, timestamp_base=timestamp)
            except CandidateHistoryUnavailableError:
                ctx.sense_candidate_runtime.read_only_definition(definition)
                recovered += 1
            else:
                registered += 1
            sense_ref = ctx.graph_ontology.resolve(mapped.spec.sense)
            concept_ref = ctx.graph_ontology.resolve(mapped.spec.concept)
            if sense_ref is None or concept_ref is None:
                raise RuntimeError("typed Sense 定义写入后缺少 Sense/Concept 对象")
            for object_ref in (sense_ref, concept_ref):
                record_legacy_sense_bridge(
                    ctx.backend,
                    legacy_ref=mapped.legacy_ref,
                    object_ref=(
                        object_ref.object_kind,
                        object_ref.space_id,
                        object_ref.local_id,
                    ),
                )
        self._formations[key] = ordered
        self._formation_inputs += 1
        self._registered += registered
        self._recovered += recovered
        return ordered

    def observe_item(self, ctx, item, observed) -> tuple[
            SenseCandidateRecognitionTrace, ...]:
        """把 item 的真实 occurrence 逐个送入 mapper 和 H-05 全链，重复 round 精确幂等。"""
        if item.source_ref is None or not observed.occurrence_refs:
            return ()
        if len(item.tokens) != len(observed.occurrence_refs):
            raise ValueError("Sense recognition 的 token 与 occurrence 数量不一致")
        provider = (
            None if ctx.word_form_providers is None
            else ctx.word_form_providers.provider(item.lang))
        span_inputs = tuple(
            ctx.graph_ontology.identity_of(ref) for ref in observed.span_refs)
        traces: list[SenseCandidateRecognitionTrace] = []
        for surface, occurrence_ref in zip(
                item.tokens, observed.occurrence_refs, strict=True):
            formations = self._formations.get((item.lang, surface), ())
            if not formations:
                continue
            occurrence_record = ctx.occurrence_index.read(occurrence_ref)
            occurrence = ctx.graph_ontology.identity_of(occurrence_ref)
            representation = None
            if provider is not None:
                representation_ref = provider.index.lookup(
                    surface, branch=provider.branch)
                if representation_ref is not None:
                    representation = ctx.graph_ontology.identity_of(
                        representation_ref)
            input_value = SenseRecognitionInput(
                item.lang,
                surface,
                item.source_ref,
                occurrence_record.scope,
                occurrence,
                representation,
                span_inputs,
                tuple(mapped.spec.sense for mapped in formations),
            )
            mapped_values = self.mapper.recognize(input_value)
            if not isinstance(mapped_values, tuple):
                raise TypeError("sense mapper.recognize 必须返回 tuple")
            for mapped in mapped_values:
                if not isinstance(mapped, MappedSenseRecognition):
                    raise TypeError("sense mapper.recognize 返回了非法结果")
                if mapped.candidate not in input_value.available_candidates:
                    raise ValueError("sense recognition 引用了未形成候选")
                if occurrence not in mapped.visible_inputs:
                    raise ValueError("Sense prediction 必须保存当前 Occurrence")
                route = (
                    mapped.candidate,
                    input_value.observation,
                    mapped.revealed.event_key,
                )
                prior = self._processed.get(route)
                mapped_key = _mapped_key(mapped)
                if prior is not None:
                    if prior[0] != mapped_key:
                        raise ValueError("同一 Sense recognition route 的 mapper 输出发生漂移")
                    traces.append(prior[1])
                    continue
                try:
                    hypothesis = (
                        ctx.sense_candidate_runtime.hypothesis_for_candidate(
                            mapped.candidate))
                except CandidateHistoryUnavailableError:
                    projection = (
                        ctx.sense_candidate_runtime
                        .lifecycle_projection_if_available(mapped.candidate))
                    if projection is None:
                        continue
                    definition = projection.candidate.definition
                    outcome = None
                    read_only = True
                else:
                    definition = ctx.sense_candidate_runtime.engine.definition(
                        hypothesis)
                    target_binding = (
                        ctx.sense_candidate_consumer.protocol.concept.binding(
                            mapped.predicted))
                    if target_binding not in definition.bindings:
                        raise ValueError("Sense prediction target 不属于候选 Concept 字段")
                    evidence_seq, decision_seq, projection_seq = (
                        ctx.sense_candidate_runtime.next_timestamps(3))
                    outcome = ctx.sense_candidate_runtime.recognize(
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
                            ctx.sense_candidate_runtime.projection_for_candidate(
                                mapped.candidate))
                    read_only = False
                target_binding = (
                    ctx.sense_candidate_consumer.protocol.concept.binding(
                        mapped.predicted))
                if target_binding not in definition.bindings:
                    raise ValueError("Sense prediction target 不属于候选 Concept 字段")
                adopted = (
                    projection.state
                    == ctx.candidate_projection_graph.protocol.active_state)
                trace = SenseCandidateRecognitionTrace(
                    input_value,
                    mapped,
                    outcome,
                    projection,
                    read_only,
                    adopted,
                )
                self._processed[route] = (mapped_key, trace)
                traces.append(trace)
        return tuple(traces)

    def active_concept_refs(
            self, ctx, *, runtime_language: int, surface: str,
            ) -> tuple[ConceptRef, ...]:
        """严格消费 active typed Sense；多 Sense 或无 Sense 时不向旧单值 caller 私选。"""
        formations = self._formations.get((runtime_language, surface), ())
        candidates = {}
        for formation in formations:
            for candidate in ctx.sense_candidate_consumer.lookup(
                    formation.spec.atom):
                candidates[candidate.sense] = candidate
        if len(candidates) != 1:
            return ()
        selected = next(iter(candidates.values()))
        concept_ref = ctx.graph_ontology.resolve(selected.concept)
        if concept_ref is None:
            raise RuntimeError("active Sense 的 Concept 无法从图恢复")
        return (concept_ref.node_ref(),)

    def report(self) -> SenseCandidateIntegrationReport:
        """从已处理 route 派生全链计数，不把旧频次或 base_count 算作支持。"""
        traces = tuple(item[1] for item in self._processed.values())
        outcomes = tuple(
            trace.outcome for trace in traces if trace.outcome is not None)
        stances = tuple(item.verification.stance for item in outcomes)
        return SenseCandidateIntegrationReport(
            self._formation_inputs,
            self._registered,
            self._recovered,
            len(traces),
            len(outcomes),
            stances.count(EVIDENCE_SUPPORT),
            stances.count(EVIDENCE_REFUTE),
            stances.count(EVIDENCE_UNKNOWN),
            sum(trace.read_only for trace in traces),
            sum(trace.adopted for trace in traces),
        )

    def clone_for_evaluation(self) -> "SenseCandidateCourseRuntime":
        """复制 mapper、形成路由和幂等 route，避免评测共享可变课程状态。"""
        cloned = SenseCandidateCourseRuntime(
            self.mapper.clone_for_evaluation())
        cloned._formations = dict(self._formations)
        cloned._processed = dict(self._processed)
        cloned._formation_inputs = self._formation_inputs
        cloned._registered = self._registered
        cloned._recovered = self._recovered
        return cloned

    def state_key(self) -> tuple:
        """返回课程 mapper、形成映射和处理 route 的完整隔离状态。"""
        formations = tuple(sorted(
            (
                runtime_language,
                surface,
                tuple((
                    item.legacy_ref,
                    item.branch.stable_key(),
                    item.spec.sense.stable_key(),
                    item.spec.atom.stable_key(),
                    item.spec.concept.stable_key(),
                    item.spec.context.stable_key(),
                    item.spec.competition_key,
                    tuple(source.stable_key()
                          for source in item.spec.forming_sources),
                ) for item in values),
            )
            for (runtime_language, surface), values in self._formations.items()
        ))
        routes = tuple(sorted(
            (
                candidate.stable_key(),
                source.stable_key(),
                event_key,
                mapped_key,
                trace.read_only,
                trace.adopted,
            )
            for (candidate, source, event_key), (mapped_key, trace)
            in self._processed.items()
        ))
        return (
            self.mapper.state_key(),
            formations,
            routes,
            self._formation_inputs,
            self._registered,
            self._recovered,
        )


def observe_sense_lookup(ctx, *, runtime_language: int):
    """选择 typed active Sense reader；未启 H-05 时才退回旧目录兼容 hook。"""
    runtime = ctx.sense_candidate_course_runtime
    if runtime is None:
        from pure_integer_ai.cognition.understanding.sense_lookup_hook import (
            make_sense_lookup,
        )
        return make_sense_lookup(ctx.backend, ctx.space_id), True

    def _typed_lookup(surface: str):
        """只返回唯一 active typed Sense 的 ConceptRef，多解保持无选择。"""
        return list(runtime.active_concept_refs(
            ctx,
            runtime_language=runtime_language,
            surface=surface,
        ))

    return _typed_lookup, False


__all__ = [
    "LegacySenseCandidateInput",
    "MappedSenseFormation",
    "MappedSenseRecognition",
    "SenseCandidateCourseMapper",
    "SenseCandidateCourseRuntime",
    "SenseCandidateIntegrationReport",
    "SenseCandidateRecognitionTrace",
    "SenseFormationInput",
    "SenseRecognitionInput",
    "observe_sense_lookup",
]
