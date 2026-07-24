"""R-03 PROPERTY 的六维 typed claim、四态聚合和有界选择。

本模块不读取自然语言词面、旧 PROPERTY 边或宿主极性/模态枚举。调用方必须
注入 relation、六个 Role、完整 schema 和 intensity resolver；运行期只聚合 R-00
提供的当前 Evidence，任何采用都由上层 runtime 经正式 relation Use 完成。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSchemaError,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.valtypes import Rational


class PropertyRelationError(RuntimeError):
    """PROPERTY 协议、强度解释或查询结果不完整。"""


class PropertyRelationBudgetExceeded(PropertyRelationError):
    """PROPERTY 查询超过调用方预算，拒绝返回部分候选。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长完整键增加长度边界。"""
    return len(value), *value


def _strict_key(
        value: tuple[int, ...], *, label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验开放协议键只包含严格整数。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _authoritative(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """要求对象具有权威 identity contract，可承担 PROPERTY filler。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    contract = object_contracts_by_kind().get(identity.object_kind)
    if contract is None or not contract.authoritative_identity:
        raise ValueError(f"{label} 缺少权威 object contract")
    return identity


@dataclass(frozen=True)
class PropertyRelationProtocol:
    """一个 PROPERTY relation 的 schema 和六个有名 Role。"""

    schema: RelationSchema
    subject_role: ObjectIdentity
    attribute_role: ObjectIdentity
    value_role: ObjectIdentity
    polarity_role: ObjectIdentity
    modality_role: ObjectIdentity
    intensity_role: ObjectIdentity

    def __post_init__(self) -> None:
        """要求 schema 恰好声明六个互异且单值的注入 Role。"""
        if not isinstance(self.schema, RelationSchema):
            raise TypeError("property relation schema 类型错误")
        roles = self.roles()
        if len(set(roles)) != len(roles):
            raise ValueError("property relation Role 不得重复")
        if len(self.schema.slots) != len(roles):
            raise RelationSchemaError("property schema 必须恰好声明六个 Role")
        if {slot.role for slot in self.schema.slots} != set(roles):
            raise RelationSchemaError("property schema Role 与协议不一致")
        for role in roles:
            if not isinstance(role, ObjectIdentity):
                raise TypeError("property relation Role 类型错误")
            if role.object_kind != OBJECT_ROLE:
                raise RelationSchemaError("property relation Role 必须是一等 Role")
            slot = self.schema.slot(role)
            if slot is None or (slot.min_count, slot.max_count) != (1, 1):
                raise RelationSchemaError("property Role 必须恰好出现一次")

    @property
    def relation(self) -> ObjectIdentity:
        """返回 schema 绑定的一等 PROPERTY relation。"""
        return self.schema.relation

    def roles(self) -> tuple[ObjectIdentity, ...]:
        """按 claim 字段顺序返回六个完整 Role。"""
        return (
            self.subject_role,
            self.attribute_role,
            self.value_role,
            self.polarity_role,
            self.modality_role,
            self.intensity_role,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation、schema 和六个 Role 的完整状态键。"""
        values = [
            *_packed(self.schema.schema.stable_key()),
            *_packed(self.relation.stable_key()),
            len(self.roles()),
        ]
        for role in self.roles():
            values.extend(_packed(role.stable_key()))
        return tuple(values)


@runtime_checkable
class PropertyIntensityResolver(Protocol):
    """把一等 intensity filler 解释为纯整数 Rational。"""

    def resolve(self, intensity: ObjectIdentity) -> Rational | None:
        """返回 intensity 的当前 Rational 解释，未知时返回 None。"""
        ...

    def clone_for_evaluation(self) -> "PropertyIntensityResolver":
        """返回不共享可变宿主状态的评测副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 resolver 协议和全部映射的完整整数键。"""
        ...


@dataclass(frozen=True)
class MappingPropertyIntensityResolver:
    """最小 fixture 使用的一等 intensity 到 Rational 显式映射。"""

    values: tuple[tuple[ObjectIdentity, Rational], ...]

    def __post_init__(self) -> None:
        """核验 intensity 身份、Rational 类型和映射唯一性。"""
        if not isinstance(self.values, tuple):
            raise TypeError("property intensity values 必须是 tuple")
        normalized = []
        for item in self.values:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("property intensity mapping 必须是二元组")
            identity, value = item
            _authoritative(identity, label="property intensity")
            if not isinstance(value, Rational):
                raise TypeError("property intensity value 必须是 Rational")
            normalized.append((identity, value))
        if len({item[0] for item in normalized}) != len(normalized):
            raise ValueError("property intensity mapping 不得重复身份")
        object.__setattr__(self, "values", tuple(sorted(
            normalized,
            key=lambda item: item[0].stable_key(),
        )))

    def resolve(self, intensity: ObjectIdentity) -> Rational | None:
        """按完整 intensity 身份读取 Rational，不按数值或名称降级。"""
        _authoritative(intensity, label="property intensity")
        return next((
            value for identity, value in self.values
            if identity == intensity
        ), None)

    def clone_for_evaluation(self) -> "MappingPropertyIntensityResolver":
        """复制不可变映射供 held-out runtime 使用。"""
        return MappingPropertyIntensityResolver(self.values)

    def state_key(self) -> tuple[int, ...]:
        """返回全部 intensity 身份和 Rational 数值的稳定键。"""
        result = [1, len(self.values)]
        for identity, value in self.values:
            result.extend(_packed(identity.stable_key()))
            result.extend((value.num, value.den))
        return tuple(result)


@dataclass(frozen=True)
class PropertyQueryBudget:
    """限制单次 PROPERTY owner 快照和候选数量。"""

    max_direct_facts: int
    max_options: int

    def __post_init__(self) -> None:
        """要求两个预算均为严格正整数。"""
        assert_int(
            self.max_direct_facts,
            self.max_options,
            _where="PropertyQueryBudget",
        )
        if (type(self.max_direct_facts) is not int
                or self.max_direct_facts <= 0
                or type(self.max_options) is not int
                or self.max_options <= 0):
            raise ValueError("property query budget 必须为严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回直接事实与候选预算。"""
        return self.max_direct_facts, self.max_options


@dataclass(frozen=True)
class PropertyClaim:
    """一个完整六维 PROPERTY 内容，不以 Proposition 身份替代语义字段。"""

    subject: ObjectIdentity
    attribute: ObjectIdentity
    value: ObjectIdentity
    polarity: ObjectIdentity
    modality: ObjectIdentity
    intensity: ObjectIdentity

    def __post_init__(self) -> None:
        """要求六个字段均是一等权威对象。"""
        for label, identity in zip(
                ("subject", "attribute", "value", "polarity", "modality", "intensity"),
                self.values(),
                strict=True):
            _authoritative(identity, label=f"property {label}")

    def values(self) -> tuple[ObjectIdentity, ...]:
        """按协议字段顺序返回完整 filler。"""
        return (
            self.subject,
            self.attribute,
            self.value,
            self.polarity,
            self.modality,
            self.intensity,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回六个一等对象的无歧义完整键。"""
        result = [len(self.values())]
        for identity in self.values():
            result.extend(_packed(identity.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class PropertyPattern:
    """以 subject/attribute 锚定并可精确过滤其余字段的查询模式。"""

    subject: ObjectIdentity
    attribute: ObjectIdentity
    value: ObjectIdentity | None = None
    polarity: ObjectIdentity | None = None
    modality: ObjectIdentity | None = None
    intensity: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验必需锚和全部可选过滤器均为权威对象。"""
        _authoritative(self.subject, label="property pattern subject")
        _authoritative(self.attribute, label="property pattern attribute")
        for label, identity in (
                ("value", self.value),
                ("polarity", self.polarity),
                ("modality", self.modality),
                ("intensity", self.intensity)):
            if identity is not None:
                _authoritative(identity, label=f"property pattern {label}")

    def matches(self, claim: PropertyClaim) -> bool:
        """按完整对象身份匹配，不按词面、局部编号或 Rational 值合并。"""
        if not isinstance(claim, PropertyClaim):
            raise TypeError("property pattern claim 类型错误")
        return (
            claim.subject == self.subject
            and claim.attribute == self.attribute
            and (self.value is None or claim.value == self.value)
            and (self.polarity is None or claim.polarity == self.polarity)
            and (self.modality is None or claim.modality == self.modality)
            and (self.intensity is None or claim.intensity == self.intensity)
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回必需锚与可选过滤器的完整模式键。"""
        result = []
        for identity in (
                self.subject,
                self.attribute,
                self.value,
                self.polarity,
                self.modality,
                self.intensity):
            key = () if identity is None else identity.stable_key()
            result.extend(_packed(key))
        return tuple(result)


@dataclass(frozen=True)
class PropertyEvidence:
    """从一个 R-00 forming 候选恢复的当前 PROPERTY 直接证据。"""

    claim: PropertyClaim
    proposition: ObjectIdentity
    semantic_context: ObjectIdentity
    scope: ScopeIdentity
    hypothesis: HypothesisKey
    state: LogicEvidenceState
    evidence: tuple[EvidenceRecord, ...]
    forming_sources: tuple[SourceRef, ...]
    active_supported: bool

    def __post_init__(self) -> None:
        """核验 Evidence、Hypothesis、来源和 active 投影一致。"""
        if not isinstance(self.claim, PropertyClaim):
            raise TypeError("property evidence claim 类型错误")
        if not isinstance(self.proposition, ObjectIdentity):
            raise TypeError("property evidence proposition 类型错误")
        if not isinstance(self.semantic_context, ObjectIdentity):
            raise TypeError("property evidence semantic_context 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("property evidence scope 类型错误")
        proposition_source = semantic_source(self.proposition)
        if (semantic_source(self.semantic_context) != proposition_source
                or self.scope.source != proposition_source):
            raise ValueError("property Evidence context/scope 必须绑定 Proposition 来源")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("property evidence hypothesis 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("property evidence state 类型错误")
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise TypeError("property evidence records 类型错误")
        if any(item.hypothesis != self.hypothesis for item in self.evidence):
            raise ValueError("property Evidence 属于其他 Hypothesis")
        if not isinstance(self.forming_sources, tuple) or any(
                not isinstance(item, SourceRef) for item in self.forming_sources):
            raise TypeError("property forming_sources 类型错误")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("property forming_sources 不得重复")
        if type(self.active_supported) is not bool:
            raise TypeError("property active_supported 必须是严格 bool")
        support = any(item.stance == EVIDENCE_SUPPORT for item in self.evidence)
        refute = any(item.stance == EVIDENCE_REFUTE for item in self.evidence)
        if (support, refute) != (self.state.support, self.state.refute):
            raise ValueError("property Evidence 与四态不一致")
        if self.active_supported and self.state.status != EPISTEMIC_SUPPORTED:
            raise ValueError("property active_supported 只能标记纯支持候选")

    def stable_key(self) -> tuple[int, ...]:
        """返回 claim、Proposition、Hypothesis、Evidence 和来源完整键。"""
        result = [
            *_packed(self.claim.stable_key()),
            *_packed(self.proposition.stable_key()),
            *_packed(self.semantic_context.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            *self.state.stable_key(),
            int(self.active_supported),
            len(self.evidence),
        ]
        for record in self.evidence:
            result.extend(_packed(record.stable_key()))
        result.append(len(self.forming_sources))
        for source in self.forming_sources:
            result.extend(_packed(source.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class PropertyKnowledge:
    """一次查询从 R-00 读取的不可变 PROPERTY Evidence 集。"""

    evidence: tuple[PropertyEvidence, ...]

    def __post_init__(self) -> None:
        """按完整 Evidence 键拒绝重复并稳定排序。"""
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, PropertyEvidence) for item in self.evidence):
            raise TypeError("property knowledge 类型错误")
        keyed = {item.stable_key(): item for item in self.evidence}
        if len(keyed) != len(self.evidence):
            raise ValueError("property knowledge 不得重复 Evidence")
        object.__setattr__(self, "evidence", tuple(
            keyed[key] for key in sorted(keyed)
        ))


@dataclass(frozen=True)
class PropertyClaimEvaluation:
    """一个完整 PROPERTY claim 的四态聚合结果。"""

    claim: PropertyClaim
    state: LogicEvidenceState
    evidence: tuple[PropertyEvidence, ...]

    def __post_init__(self) -> None:
        """要求 Evidence 均属于当前 claim 且证据位与状态一致。"""
        if not isinstance(self.claim, PropertyClaim):
            raise TypeError("property evaluation claim 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("property evaluation state 类型错误")
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, PropertyEvidence) for item in self.evidence):
            raise TypeError("property evaluation evidence 类型错误")
        if any(item.claim != self.claim for item in self.evidence):
            raise ValueError("property evaluation 混入其他 claim")
        support = any(item.state.support for item in self.evidence)
        refute = any(item.state.refute for item in self.evidence)
        if (support, refute) != (self.state.support, self.state.refute):
            raise ValueError("property evaluation 与 Evidence 四态不一致")

    def active_premises(self) -> tuple[PropertyEvidence, ...]:
        """返回当前 claim 中可由 R-00 consumer 正式采用的前提。"""
        return tuple(
            item for item in self.evidence if item.active_supported
        )


@dataclass(frozen=True)
class PropertySelectionOption:
    """一个完整 claim、Rational 强度和其四态证据。"""

    evaluation: PropertyClaimEvaluation
    intensity_value: Rational

    def __post_init__(self) -> None:
        """核验 evaluation 与纯整数 Rational 强度类型。"""
        if not isinstance(self.evaluation, PropertyClaimEvaluation):
            raise TypeError("property option evaluation 类型错误")
        if not isinstance(self.intensity_value, Rational):
            raise TypeError("property option intensity 必须是 Rational")

    @property
    def claim(self) -> PropertyClaim:
        """返回选项对应的完整 claim。"""
        return self.evaluation.claim


@dataclass(frozen=True)
class PropertySelection:
    """一个查询模式下全部审计选项和保守唯一采用结果。"""

    pattern: PropertyPattern
    options: tuple[PropertySelectionOption, ...]

    def __post_init__(self) -> None:
        """要求选项稳定有序、互不重复且全部匹配查询模式。"""
        if not isinstance(self.pattern, PropertyPattern):
            raise TypeError("property selection pattern 类型错误")
        if not isinstance(self.options, tuple) or any(
                not isinstance(item, PropertySelectionOption)
                for item in self.options):
            raise TypeError("property selection options 类型错误")
        claims = tuple(item.claim for item in self.options)
        if any(not self.pattern.matches(claim) for claim in claims):
            raise ValueError("property selection 包含模式外 claim")
        if len(set(claims)) != len(claims):
            raise ValueError("property selection 不得重复 claim")
        if claims != tuple(sorted(claims, key=PropertyClaim.stable_key)):
            raise ValueError("property selection options 顺序不稳定")

    def support_bearing(self) -> tuple[PropertySelectionOption, ...]:
        """返回具有 support 证据位的全部选项，不掩盖 conflicted 候选。"""
        return tuple(
            item for item in self.options
            if item.evaluation.state.support
        )

    def selected(self) -> PropertySelectionOption | None:
        """仅当唯一 support-bearing 选项为纯 supported 时返回它。"""
        supported = self.support_bearing()
        if len(supported) != 1 or supported[0].evaluation.state.refute:
            return None
        return supported[0]

    def ambiguous(self) -> bool:
        """报告多个 support-bearing 完整 claim，禁止调用方私选。"""
        return len(self.support_bearing()) > 1

    def conflicted(self) -> bool:
        """报告任一匹配 claim 同时含 support/refute。"""
        return any(
            item.evaluation.state.support
            and item.evaluation.state.refute
            for item in self.options
        )


class PropertyRelationEngine:
    """在不可变 R-00 快照上执行有界 PROPERTY 查询和保守选择。"""

    def __init__(
            self,
            protocol: PropertyRelationProtocol,
            budget: PropertyQueryBudget,
            knowledge: PropertyKnowledge,
            intensity_resolver: PropertyIntensityResolver,
            ) -> None:
        """绑定注入协议、预算、当前知识和 intensity 解释 owner。"""
        if not isinstance(protocol, PropertyRelationProtocol):
            raise TypeError("property protocol 类型错误")
        if not isinstance(budget, PropertyQueryBudget):
            raise TypeError("property budget 类型错误")
        if not isinstance(knowledge, PropertyKnowledge):
            raise TypeError("property knowledge 类型错误")
        if not isinstance(intensity_resolver, PropertyIntensityResolver):
            raise TypeError("property intensity resolver 协议不完整")
        resolver_key = intensity_resolver.state_key()
        _strict_key(
            resolver_key,
            label="PropertyIntensityResolver.state_key",
            allow_empty=True,
        )
        if len(knowledge.evidence) > budget.max_direct_facts:
            raise PropertyRelationBudgetExceeded(
                "property 直接事实预算耗尽")
        self.protocol = protocol
        self.budget = budget
        self.knowledge = knowledge
        self.intensity_resolver = intensity_resolver
        grouped: dict[PropertyClaim, list[PropertyEvidence]] = {}
        for item in knowledge.evidence:
            grouped.setdefault(item.claim, []).append(item)
        self._by_claim = {
            claim: tuple(sorted(values, key=PropertyEvidence.stable_key))
            for claim, values in grouped.items()
        }

    def evaluate(self, claim: PropertyClaim) -> PropertyClaimEvaluation:
        """聚合一个完整六维 claim 的多来源四态 Evidence。"""
        if not isinstance(claim, PropertyClaim):
            raise TypeError("property evaluate claim 类型错误")
        evidence = self._by_claim.get(claim, ())
        return PropertyClaimEvaluation(
            claim,
            LogicEvidenceState(
                any(item.state.support for item in evidence),
                any(item.state.refute for item in evidence),
            ),
            evidence,
        )

    def select(self, pattern: PropertyPattern) -> PropertySelection:
        """完整发现匹配 claim，解释强度并保守确定唯一可采用选项。"""
        if not isinstance(pattern, PropertyPattern):
            raise TypeError("property select pattern 类型错误")
        claims = tuple(sorted(
            (claim for claim in self._by_claim if pattern.matches(claim)),
            key=PropertyClaim.stable_key,
        ))
        if len(claims) > self.budget.max_options:
            raise PropertyRelationBudgetExceeded("property option 预算耗尽")
        options = []
        for claim in claims:
            intensity = self.intensity_resolver.resolve(claim.intensity)
            if intensity is None:
                raise PropertyRelationError(
                    "property intensity 缺少 Rational 解释")
            options.append(PropertySelectionOption(
                self.evaluate(claim),
                intensity,
            ))
        return PropertySelection(pattern, tuple(options))


__all__ = [
    "MappingPropertyIntensityResolver",
    "PropertyClaim",
    "PropertyClaimEvaluation",
    "PropertyEvidence",
    "PropertyIntensityResolver",
    "PropertyKnowledge",
    "PropertyPattern",
    "PropertyQueryBudget",
    "PropertyRelationBudgetExceeded",
    "PropertyRelationEngine",
    "PropertyRelationError",
    "PropertyRelationProtocol",
    "PropertySelection",
    "PropertySelectionOption",
]
