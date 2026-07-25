"""R-08 逻辑作用候选、执行采用和生成归因的纯领域对象。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding,
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvaluation,
    LogicOperatorDefinition,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """校验开放协议键和 trace 使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度前缀。"""
    return len(value), *value


@dataclass(frozen=True)
class LogicOperatorCandidateProtocol:
    """声明候选 Proposition 到结构、指令和有序槽的图 predicate。"""

    structure_predicate: ObjectIdentity
    instruction_predicate: ObjectIdentity
    slot_predicate: ObjectIdentity

    def __post_init__(self) -> None:
        """核验三类 predicate 都是一等、互异 Concept。"""
        predicates = (
            self.structure_predicate,
            self.instruction_predicate,
            self.slot_predicate,
        )
        if any(not isinstance(item, ObjectIdentity) for item in predicates):
            raise TypeError("logic candidate predicate 必须是 ObjectIdentity")
        if any(item.object_kind != OBJECT_CONCEPT for item in predicates):
            raise ValueError("logic candidate predicate 必须是一等 Concept")
        if len(set(predicates)) != len(predicates):
            raise ValueError("logic candidate predicate 必须互不相同")

    def stable_key(self) -> tuple[int, ...]:
        """返回三类 predicate 的完整协议键。"""
        result: list[int] = []
        for item in (
                self.structure_predicate,
                self.instruction_predicate,
                self.slot_predicate):
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class LogicOperatorCandidateSpec:
    """一个来源化逻辑作用候选及其 S-04 执行 adapter。"""

    candidate: ObjectIdentity
    definition: LogicOperatorDefinition
    competition_key: tuple[int, ...]
    forming_sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        """拒绝把 StructureConcept 本体直接冒充可竞争候选 Proposition。"""
        if not isinstance(self.candidate, ObjectIdentity):
            raise TypeError("logic candidate 必须是 ObjectIdentity")
        if self.candidate.object_kind != OBJECT_PROPOSITION:
            raise ValueError("logic candidate 必须是一等 Proposition")
        if not isinstance(self.definition, LogicOperatorDefinition):
            raise TypeError("logic candidate definition 类型错误")
        _strict_key(self.competition_key, label="logic competition_key")
        if not isinstance(self.forming_sources, tuple):
            raise TypeError("logic forming_sources 必须是 tuple")
        if any(not isinstance(item, SourceRef)
               for item in self.forming_sources):
            raise TypeError("logic forming_sources 只能包含 SourceRef")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("logic forming_sources 不得重复")

    def candidate_definition(
            self, protocol: LogicOperatorCandidateProtocol,
            ) -> EvidenceCandidateDefinition:
        """把候选本体映射为 H-05 可持久化的结构、指令和槽位图绑定。"""
        if not isinstance(protocol, LogicOperatorCandidateProtocol):
            raise TypeError("logic candidate protocol 类型错误")
        bindings = [
            CandidateBinding(
                protocol.structure_predicate,
                self.definition.structure,
            ),
            CandidateBinding(
                protocol.instruction_predicate,
                self.definition.instruction,
            ),
        ]
        for index, slot in enumerate(self.definition.slots):
            bindings.append(CandidateBinding(
                protocol.slot_predicate,
                slot.role,
                ordinal=index,
            ))
        return EvidenceCandidateDefinition(
            self.candidate,
            self.competition_key,
            tuple(bindings),
            self.forming_sources,
        )

    def stable_key(
            self, protocol: LogicOperatorCandidateProtocol,
            ) -> tuple[int, ...]:
        """返回不包含 Python handler 对象身份的完整来源化候选键。"""
        return self.candidate_definition(protocol).stable_key()


@dataclass(frozen=True)
class LogicOperatorAdoption:
    """一个已通过 H-04/H-05 且可进入 S-04 registry 的候选快照。"""

    spec: LogicOperatorCandidateSpec
    hypothesis: HypothesisKey
    evidence: tuple[EvidenceRecord, ...]
    decision: ResolverDecision
    projection: CandidateGraphProjection

    def __post_init__(self) -> None:
        """交叉核验 active 快照的 Hypothesis、Evidence 和图投影。"""
        if not isinstance(self.spec, LogicOperatorCandidateSpec):
            raise TypeError("logic adoption spec 类型错误")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("logic adoption hypothesis 类型错误")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("logic adoption 必须携带非空 Evidence")
        if any(not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise TypeError("logic adoption evidence 元素类型错误")
        if any(item.hypothesis != self.hypothesis for item in self.evidence):
            raise ValueError("logic adoption Evidence 混入其他 Hypothesis")
        if not isinstance(self.decision, ResolverDecision):
            raise TypeError("logic adoption decision 类型错误")
        if not isinstance(self.projection, CandidateGraphProjection):
            raise TypeError("logic adoption projection 类型错误")
        materialized = self.projection.candidate
        if materialized.hypothesis != self.hypothesis:
            raise ValueError("logic adoption 图投影属于其他 Hypothesis")
        if materialized.definition.candidate != self.spec.candidate:
            raise ValueError("logic adoption 图投影属于其他候选")
        object.__setattr__(self, "evidence", tuple(sorted(
            self.evidence, key=lambda item: item.evidence_id)))

    def stable_key(self) -> tuple[int, ...]:
        """返回采用候选、Evidence、decision 和当前图状态完整键。"""
        result = [
            *_packed(self.hypothesis.stable_key()),
            len(self.evidence),
        ]
        for item in self.evidence:
            result.extend(_packed(item.stable_key()))
        result.extend(_packed(self.decision.stable_key()))
        result.extend(_packed(self.projection.state.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class LogicOperatorExecutionUse:
    """一次逻辑执行对实际采用 operator 候选的完整归因。"""

    use_key: tuple[int, ...]
    evaluation: LogicEvaluation
    adoptions: tuple[LogicOperatorAdoption, ...]
    conflicted_structures: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        """核验 derivation 中每个 operator 都有且只有一个采用候选。"""
        _strict_key(self.use_key, label="logic execution use_key")
        if not isinstance(self.evaluation, LogicEvaluation):
            raise TypeError("logic execution evaluation 类型错误")
        if any(not isinstance(item, LogicOperatorAdoption)
               for item in self.adoptions):
            raise TypeError("logic execution adoptions 类型错误")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.conflicted_structures):
            raise TypeError("conflicted_structures 元素类型错误")
        by_structure = {
            item.spec.definition.structure: item for item in self.adoptions
        }
        if len(by_structure) != len(self.adoptions):
            raise ValueError("同一结构不得记录多个已采用 operator")
        used = {item.operator for item in self.evaluation.derivation}
        if not used.issubset(by_structure):
            raise ValueError("logic derivation 含未归因的 operator")
        if used.intersection(self.conflicted_structures):
            raise ValueError("冲突结构不得同时进入 logic derivation")
        object.__setattr__(self, "adoptions", tuple(sorted(
            self.adoptions, key=lambda item: item.stable_key())))
        object.__setattr__(self, "conflicted_structures", tuple(sorted(
            set(self.conflicted_structures),
            key=ObjectIdentity.stable_key,
        )))

    def stable_key(self) -> tuple[int, ...]:
        """返回执行结果、实际采用候选和冲突结构的完整归因键。"""
        result = [
            *_packed(self.use_key),
            *_packed(self.evaluation.stable_key()),
            len(self.adoptions),
        ]
        for item in self.adoptions:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.conflicted_structures))
        for item in self.conflicted_structures:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class LogicDerivedEvidenceSeed:
    """调用方为一次 provisional 派生 Evidence 注入的事件身份和理由。"""

    stance: int
    evidence_id: int
    reason_key: tuple[int, ...]
    timestamp_seq: int
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验派生 Evidence 事件不使用隐式 id、理由或墙钟。"""
        assert_int(
            self.stance,
            self.evidence_id,
            self.timestamp_seq,
            _where="LogicDerivedEvidenceSeed",
        )
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("派生 Evidence stance 未注册")
        if type(self.evidence_id) is not int or self.evidence_id <= 0:
            raise ValueError("派生 Evidence id 必须为严格正整数")
        if type(self.timestamp_seq) is not int or self.timestamp_seq < 0:
            raise ValueError("派生 Evidence timestamp 必须为非负严格整数")
        _strict_key(self.reason_key, label="logic derived reason_key")
        _strict_key(self.trace, label="logic derived trace")


@dataclass(frozen=True)
class LogicDerivedEvidenceBundle:
    """当前根命题的执行结果、派生 Evidence 和 G-00 typed 候选。"""

    execution: LogicOperatorExecutionUse
    hypothesis: HypothesisKey
    evidence: tuple[EvidenceRecord, ...]
    candidate: GenerationCandidate

    def __post_init__(self) -> None:
        """核验派生 Evidence、执行四态和 GenerationCandidate 三者一致。"""
        if not isinstance(self.execution, LogicOperatorExecutionUse):
            raise TypeError("derived bundle execution 类型错误")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("derived bundle hypothesis 类型错误")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("derived bundle evidence 不能为空")
        if any(not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise TypeError("derived bundle evidence 元素类型错误")
        if any(item.hypothesis != self.hypothesis for item in self.evidence):
            raise ValueError("derived bundle Evidence 混入其他 Hypothesis")
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("derived bundle candidate 类型错误")
        evaluation = self.execution.evaluation
        if (self.candidate.proposition != evaluation.proposition
                or self.candidate.state != evaluation.state
                or self.candidate.source != evaluation.source
                or self.candidate.scope != evaluation.scope
                or self.candidate.evidence != self.evidence):
            raise ValueError("GenerationCandidate 与逻辑执行派生 Evidence 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回执行、根 Hypothesis、派生 Evidence 和生成候选完整键。"""
        result = [
            *_packed(self.execution.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            len(self.evidence),
        ]
        for item in self.evidence:
            result.extend(_packed(item.stable_key()))
        result.extend(_packed(self.candidate.stable_key()))
        return tuple(result)


def build_logic_derived_evidence(
        execution: LogicOperatorExecutionUse,
        hypothesis: HypothesisKey,
        seeds: tuple[LogicDerivedEvidenceSeed, ...],
        ) -> LogicDerivedEvidenceBundle:
    """按执行四态生成当前来源 Evidence，并构造 G-00 可消费候选。"""
    if not isinstance(execution, LogicOperatorExecutionUse):
        raise TypeError("execution 必须是 LogicOperatorExecutionUse")
    if not isinstance(hypothesis, HypothesisKey):
        raise TypeError("hypothesis 必须是 HypothesisKey")
    evaluation = execution.evaluation
    if hypothesis.observation != evaluation.source:
        raise ValueError("派生 Hypothesis observation 与执行来源不一致")
    if hypothesis.scope != evaluation.scope:
        raise ValueError("派生 Hypothesis scope 与执行 scope 不一致")
    if hypothesis.candidate_key != evaluation.proposition.stable_key():
        raise ValueError("派生 Hypothesis candidate_key 必须绑定根 BoundProposition")
    if not isinstance(seeds, tuple) or not seeds:
        raise ValueError("派生 Evidence seeds 不能为空")
    if any(not isinstance(item, LogicDerivedEvidenceSeed) for item in seeds):
        raise TypeError("派生 Evidence seeds 元素类型错误")
    required = set()
    if evaluation.state.support:
        required.add(EVIDENCE_SUPPORT)
    if evaluation.state.refute:
        required.add(EVIDENCE_REFUTE)
    if not required:
        required.add(EVIDENCE_UNKNOWN)
    stances = tuple(item.stance for item in seeds)
    if set(stances) != required or len(stances) != len(required):
        raise ValueError("派生 Evidence stance 必须与执行四态一一对应")
    ids = tuple(item.evidence_id for item in seeds)
    if len(set(ids)) != len(ids):
        raise ValueError("派生 Evidence id 不得重复")
    evidence = tuple(sorted((
        EvidenceRecord(
            item.evidence_id,
            hypothesis,
            item.stance,
            item.reason_key,
            evaluation.source,
            item.timestamp_seq,
            item.trace,
        )
        for item in seeds
    ), key=lambda item: item.evidence_id))
    candidate = GenerationCandidate(
        evaluation.proposition,
        evaluation.state,
        evaluation.source,
        evaluation.scope,
        evidence,
    )
    return LogicDerivedEvidenceBundle(
        execution,
        hypothesis,
        evidence,
        candidate,
    )


__all__ = [
    "LogicDerivedEvidenceBundle",
    "LogicDerivedEvidenceSeed",
    "LogicOperatorAdoption",
    "LogicOperatorCandidateProtocol",
    "LogicOperatorCandidateSpec",
    "LogicOperatorExecutionUse",
    "build_logic_derived_evidence",
]
