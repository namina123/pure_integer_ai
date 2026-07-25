"""H-05 cue 结构和 `LanguageAtom -> Sense -> Concept` typed 候选适配器。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CANDIDATE_AS_OBJECT,
    CANDIDATE_AS_SUBJECT,
    CandidateBinding,
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class LanguageCandidateError(RuntimeError):
    """语言候选字段、图方向或 active 消费投影不完整。"""


def _strict_key(value, *, where: str) -> tuple[int, ...]:
    """校验领域候选竞争键使用严格开放整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _strict_kinds(value, *, where: str) -> tuple[int, ...]:
    """校验由图协议注入的开放 object kind 集合。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空 object kind tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{where} 必须使用严格正整数")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} 不得重复 object kind")
    return tuple(sorted(value))


@dataclass(frozen=True)
class CandidateFieldProtocol:
    """一个领域字段的一等 predicate、候选端点、ordinal 和值类型契约。"""

    predicate: ObjectIdentity
    candidate_endpoint: int
    ordinal: int
    value_kinds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, ObjectIdentity):
            raise TypeError("field predicate 必须是 ObjectIdentity")
        if self.predicate.object_kind != OBJECT_CONCEPT:
            raise ValueError("field predicate 必须是一等 Concept")
        assert_int(
            self.candidate_endpoint,
            self.ordinal,
            _where="CandidateFieldProtocol",
        )
        if self.candidate_endpoint not in {
                CANDIDATE_AS_SUBJECT, CANDIDATE_AS_OBJECT}:
            raise ValueError("field candidate_endpoint 非法")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("field ordinal 必须为非负严格整数")
        object.__setattr__(
            self,
            "value_kinds",
            _strict_kinds(
                self.value_kinds,
                where="CandidateFieldProtocol.value_kinds",
            ),
        )

    def binding(self, value: ObjectIdentity) -> CandidateBinding:
        """校验字段值类型并构造保留边方向的 CandidateBinding。"""
        if not isinstance(value, ObjectIdentity):
            raise TypeError("candidate field value 必须是 ObjectIdentity")
        if value.object_kind not in self.value_kinds:
            raise ValueError("candidate field value object kind 不符合注入协议")
        return CandidateBinding(
            self.predicate,
            value,
            self.ordinal,
            self.candidate_endpoint,
        )

    def slot_key(self) -> tuple:
        """返回字段在候选定义中的 predicate/endpoint/ordinal 唯一槽。"""
        return self.predicate, self.candidate_endpoint, self.ordinal


def _validate_fields(fields: tuple[CandidateFieldProtocol, ...]) -> None:
    """拒绝领域必需字段复用同一图槽。"""
    if any(not isinstance(item, CandidateFieldProtocol) for item in fields):
        raise TypeError("领域字段必须是 CandidateFieldProtocol")
    slots = tuple(item.slot_key() for item in fields)
    if len(set(slots)) != len(slots):
        raise ValueError("领域必需字段不得复用同一 predicate/endpoint/ordinal")


@dataclass(frozen=True)
class CueStructureCandidateProtocol:
    """cue、目标关系、结构族和 context 的开放 typed 字段协议。"""

    candidate_kinds: tuple[int, ...]
    cue: CandidateFieldProtocol
    target_relation: CandidateFieldProtocol
    structure_family: CandidateFieldProtocol
    context: CandidateFieldProtocol

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_kinds",
            _strict_kinds(
                self.candidate_kinds,
                where="CueStructureCandidateProtocol.candidate_kinds",
            ),
        )
        _validate_fields((
            self.cue,
            self.target_relation,
            self.structure_family,
            self.context,
        ))


@dataclass(frozen=True)
class SenseCandidateProtocol:
    """显式 `atom -> Sense -> Concept` 和 context 的开放 typed 字段协议。"""

    candidate_kinds: tuple[int, ...]
    atom: CandidateFieldProtocol
    concept: CandidateFieldProtocol
    context: CandidateFieldProtocol

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_kinds",
            _strict_kinds(
                self.candidate_kinds,
                where="SenseCandidateProtocol.candidate_kinds",
            ),
        )
        _validate_fields((self.atom, self.concept, self.context))
        if self.atom.candidate_endpoint != CANDIDATE_AS_OBJECT:
            raise ValueError("atom 字段必须把 Sense 放在 object 端")
        if self.concept.candidate_endpoint != CANDIDATE_AS_SUBJECT:
            raise ValueError("concept 字段必须把 Sense 放在 subject 端")


@dataclass(frozen=True)
class CueStructureCandidateSpec:
    """一个 cue-bearing structure 的完整 typed 候选输入。"""

    candidate: ObjectIdentity
    competition_key: tuple[int, ...]
    cue: ObjectIdentity
    target_relation: ObjectIdentity
    structure_family: ObjectIdentity
    context: ObjectIdentity
    slot_bindings: tuple[CandidateBinding, ...]
    forming_sources: tuple[SourceRef, ...]

    def definition(
            self,
            protocol: CueStructureCandidateProtocol,
            ) -> EvidenceCandidateDefinition:
        """按注入字段构造通用候选定义，不解释 cue 或 Role 的具体语义。"""
        if not isinstance(protocol, CueStructureCandidateProtocol):
            raise TypeError("protocol 必须是 CueStructureCandidateProtocol")
        if not isinstance(self.candidate, ObjectIdentity):
            raise TypeError("structure candidate 必须是 ObjectIdentity")
        if self.candidate.object_kind not in protocol.candidate_kinds:
            raise ValueError("structure candidate object kind 不符合注入协议")
        _strict_key(self.competition_key, where="structure competition_key")
        if not isinstance(self.slot_bindings, tuple):
            raise TypeError("slot_bindings 必须是 CandidateBinding tuple")
        if any(not isinstance(item, CandidateBinding)
               for item in self.slot_bindings):
            raise TypeError("slot_bindings 只能包含 CandidateBinding")
        return EvidenceCandidateDefinition(
            self.candidate,
            self.competition_key,
            (
                protocol.cue.binding(self.cue),
                protocol.target_relation.binding(self.target_relation),
                protocol.structure_family.binding(self.structure_family),
                protocol.context.binding(self.context),
                *self.slot_bindings,
            ),
            self.forming_sources,
        )


@dataclass(frozen=True)
class SenseCandidateSpec:
    """一个来源化 Sense、LanguageAtom、Concept 和 context 候选输入。"""

    sense: ObjectIdentity
    competition_key: tuple[int, ...]
    atom: ObjectIdentity
    concept: ObjectIdentity
    context: ObjectIdentity
    forming_sources: tuple[SourceRef, ...]

    def definition(
            self,
            protocol: SenseCandidateProtocol,
            ) -> EvidenceCandidateDefinition:
        """构造方向严格为 atom->Sense->Concept 的通用候选图定义。"""
        if not isinstance(protocol, SenseCandidateProtocol):
            raise TypeError("protocol 必须是 SenseCandidateProtocol")
        if not isinstance(self.sense, ObjectIdentity):
            raise TypeError("sense candidate 必须是 ObjectIdentity")
        if self.sense.object_kind not in protocol.candidate_kinds:
            raise ValueError("sense candidate object kind 不符合注入协议")
        _strict_key(self.competition_key, where="sense competition_key")
        return EvidenceCandidateDefinition(
            self.sense,
            self.competition_key,
            (
                protocol.atom.binding(self.atom),
                protocol.concept.binding(self.concept),
                protocol.context.binding(self.context),
            ),
            self.forming_sources,
        )


def _field_value(
        projection: CandidateGraphProjection,
        field: CandidateFieldProtocol) -> ObjectIdentity:
    """从已恢复候选定义读取唯一领域字段值，缺失或重复均 fail closed。"""
    matches = tuple(
        binding.value
        for binding in projection.candidate.definition.bindings
        if (binding.predicate == field.predicate
            and binding.candidate_endpoint == field.candidate_endpoint
            and binding.ordinal == field.ordinal)
    )
    if len(matches) != 1 or matches[0].object_kind not in field.value_kinds:
        raise LanguageCandidateError("active 候选缺少唯一合法领域字段")
    return matches[0]


@dataclass(frozen=True)
class ActiveCueStructureCandidate:
    """可回溯到 active 图事件的 cue-bearing structure 消费对象。"""

    structure: ObjectIdentity
    cue: ObjectIdentity
    target_relation: ObjectIdentity
    structure_family: ObjectIdentity
    context: ObjectIdentity
    projection: CandidateGraphProjection


class ActiveCueStructureConsumer:
    """只读 typed active 投影，不读取 REALIZES、PRIMARY、频次或 specificity。"""

    def __init__(
            self, graph: CandidateProjectionGraph,
            protocol: CueStructureCandidateProtocol) -> None:
        if not isinstance(graph, CandidateProjectionGraph):
            raise TypeError("graph 必须是 CandidateProjectionGraph")
        if not isinstance(protocol, CueStructureCandidateProtocol):
            raise TypeError("protocol 必须是 CueStructureCandidateProtocol")
        self.graph = graph
        self.protocol = protocol

    def lookup(
            self, cue: ObjectIdentity, *,
            target_relation: ObjectIdentity | None = None,
            context: ObjectIdentity | None = None,
            ) -> tuple[ActiveCueStructureCandidate, ...]:
        """按 cue 查询全部 active 结构，并用完整 typed 字段作可选过滤。"""
        projections = self.graph.active_for_binding(
            self.protocol.cue.binding(cue))
        results: list[ActiveCueStructureCandidate] = []
        for projection in projections:
            definition = projection.candidate.definition
            if definition.candidate.object_kind not in self.protocol.candidate_kinds:
                raise LanguageCandidateError("active structure candidate kind 非法")
            relation = _field_value(
                projection, self.protocol.target_relation)
            candidate_context = _field_value(
                projection, self.protocol.context)
            if target_relation is not None and relation != target_relation:
                continue
            if context is not None and candidate_context != context:
                continue
            results.append(ActiveCueStructureCandidate(
                definition.candidate,
                _field_value(projection, self.protocol.cue),
                relation,
                _field_value(projection, self.protocol.structure_family),
                candidate_context,
                projection,
            ))
        return tuple(sorted(
            results,
            key=lambda item: item.projection.candidate.hypothesis.stable_key(),
        ))

    def require_unique(
            self, cue: ObjectIdentity, *,
            target_relation: ObjectIdentity | None = None,
            context: ObjectIdentity | None = None,
            ) -> ActiveCueStructureCandidate:
        """为要求唯一结构的消费者返回一项，多解或无解均 fail closed。"""
        candidates = self.lookup(
            cue,
            target_relation=target_relation,
            context=context,
        )
        if len(candidates) != 1:
            raise LookupError("当前 cue/context 没有唯一 active typed 结构")
        return candidates[0]


@dataclass(frozen=True)
class ActiveSenseCandidate:
    """可回溯到 active 图事件的 atom、Sense、Concept 和 context 绑定。"""

    atom: ObjectIdentity
    sense: ObjectIdentity
    concept: ObjectIdentity
    context: ObjectIdentity
    projection: CandidateGraphProjection


class ActiveSenseConsumer:
    """只从 active typed 图恢复多 Sense 集合，严格入口不私选首项。"""

    def __init__(
            self, graph: CandidateProjectionGraph,
            protocol: SenseCandidateProtocol) -> None:
        if not isinstance(graph, CandidateProjectionGraph):
            raise TypeError("graph 必须是 CandidateProjectionGraph")
        if not isinstance(protocol, SenseCandidateProtocol):
            raise TypeError("protocol 必须是 SenseCandidateProtocol")
        self.graph = graph
        self.protocol = protocol

    def lookup(
            self, atom: ObjectIdentity, *,
            context: ObjectIdentity | None = None,
            ) -> tuple[ActiveSenseCandidate, ...]:
        """沿 atom->Sense 反查全部 active 候选并恢复 Sense->Concept。"""
        projections = self.graph.active_for_binding(
            self.protocol.atom.binding(atom))
        results: list[ActiveSenseCandidate] = []
        for projection in projections:
            definition = projection.candidate.definition
            if definition.candidate.object_kind not in self.protocol.candidate_kinds:
                raise LanguageCandidateError("active Sense candidate kind 非法")
            candidate_context = _field_value(
                projection, self.protocol.context)
            if context is not None and candidate_context != context:
                continue
            results.append(ActiveSenseCandidate(
                _field_value(projection, self.protocol.atom),
                definition.candidate,
                _field_value(projection, self.protocol.concept),
                candidate_context,
                projection,
            ))
        return tuple(sorted(
            results,
            key=lambda item: item.projection.candidate.hypothesis.stable_key(),
        ))

    def require_unique(
            self, atom: ObjectIdentity, *,
            context: ObjectIdentity | None = None,
            ) -> ActiveSenseCandidate:
        """要求唯一 Sense 时严格返回一项，多解或无解均抛错。"""
        candidates = self.lookup(atom, context=context)
        if len(candidates) != 1:
            raise LookupError("当前 LanguageAtom/context 没有唯一 active Sense")
        return candidates[0]


__all__ = [
    "ActiveCueStructureCandidate",
    "ActiveCueStructureConsumer",
    "ActiveSenseCandidate",
    "ActiveSenseConsumer",
    "CandidateFieldProtocol",
    "CueStructureCandidateProtocol",
    "CueStructureCandidateSpec",
    "LanguageCandidateError",
    "SenseCandidateProtocol",
    "SenseCandidateSpec",
]
