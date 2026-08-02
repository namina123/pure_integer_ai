"""W06-R03 adapter slice、PROPERTY protocol 与共享 active view。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ROLE,
    ObjectIdentity,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.property_relation import (
    MappingPropertyIntensityResolver,
    PropertyClaim,
    PropertyIntensityResolver,
    PropertyPattern,
    PropertyQueryBudget,
    PropertyRelationProtocol,
)
from pure_integer_ai.crosscut.integer.valtypes import Rational
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    ROLE_PROPERTY_ATTRIBUTE,
    ROLE_PROPERTY_INTENSITY,
    ROLE_PROPERTY_MODALITY,
    ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_SUBJECT,
    ROLE_PROPERTY_VALUE,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06RelationCandidate,
    W06TypedAdapterOutput,
    W06_IDENTITY_VERSIONS,
    W06_NAMESPACE,
)
from pure_integer_ai.experiments.ph2_w06_learning import (
    W06RelationLearningRuntime,
)
from pure_integer_ai.experiments.ph2_w06_r03_contract import (
    W06R03ConsumerProtocol,
    W06R03ContractError,
    W06_R03_SUBSTAGE,
    pack_key,
)
from pure_integer_ai.experiments.property_relation_runtime import (
    PropertyQuery,
    PropertyRelationRuntime,
)


W06_R03_RUNTIME_NAMESPACE = 50603
W06_R03_RELATION = "PROPERTY"
W06_R03_ROLE_KINDS = (
    ROLE_PROPERTY_SUBJECT,
    ROLE_PROPERTY_ATTRIBUTE,
    ROLE_PROPERTY_VALUE,
    ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_MODALITY,
    ROLE_PROPERTY_INTENSITY,
)


def _binding(
        candidate: W06RelationCandidate,
        role: ObjectIdentity,
        ) -> ObjectIdentity:
    """从 canonical proposition bindings 读取指定 Role 的唯一 filler。"""
    values = tuple(
        item.filler for item in candidate.proposition.canonical_bindings()
        if item.role == role
    )
    if len(values) != 1:
        raise W06R03ContractError("R03 property Role 未恰好绑定一次")
    return values[0]


def candidate_role_fillers(
        candidate: W06RelationCandidate,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    """按六个 typed Role 返回规范排序的 filler 对。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R03_SUBSTAGE):
        raise W06R03ContractError("R03 candidate 不属于 PROPERTY")
    return tuple(sorted(
        ((item.role, item.filler)
         for item in candidate.proposition.canonical_bindings()),
        key=lambda item: item[0].stable_key(),
    ))


def candidate_claim(
        candidate: W06RelationCandidate,
        protocol: PropertyRelationProtocol,
        ) -> PropertyClaim:
    """按注入协议从 candidate Proposition 恢复完整六维 claim。"""
    if not isinstance(protocol, PropertyRelationProtocol):
        raise TypeError("R03 property protocol 类型非法")
    return PropertyClaim(*tuple(
        _binding(candidate, role)
        for role in protocol.roles()
    ))


def w06_r03_language_branch(candidate: W06RelationCandidate) -> ObjectIdentity:
    """从来源 Observation 的语言字段构造一等语言分支。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R03_SUBSTAGE):
        raise W06R03ContractError("R03 language branch candidate 非法")
    value = candidate.observation.language
    return language_branch_identity(
        (W06_NAMESPACE, 923, len(value), *(ord(item) for item in value)),
        versions=W06_IDENTITY_VERSIONS,
    )


def candidate_construction(candidate: W06RelationCandidate) -> ObjectIdentity:
    """把来源 template family 提升为可审计的结构身份。"""
    if candidate.substage_key != W06_R03_SUBSTAGE:
        raise W06R03ContractError("R03 construction candidate 非法")
    key = candidate.observation.template_group_key.components
    return structure_concept_identity(
        (W06_NAMESPACE, 924, len(key), *key),
        versions=W06_IDENTITY_VERSIONS,
    )


def _property_roles() -> tuple[ObjectIdentity, ...]:
    """返回 authored PROPERTY 六个一等 Role。"""
    return tuple(
        authored_relation_role_identity(kind)
        for kind in W06_R03_ROLE_KINDS
    )


def w06_r03_property_protocol(
        adapter: W06TypedAdapterOutput,
        ) -> PropertyRelationProtocol:
    """从 R03 唯一公开 schema 注入现役 PropertyRelationProtocol。"""
    candidates = adapter.candidates_for_substage(W06_R03_SUBSTAGE)
    schemas = {item.schema.schema: item.schema for item in candidates}
    if len(schemas) != 1:
        raise W06R03ContractError("R03 PROPERTY schema 必须唯一")
    roles = _property_roles()
    return PropertyRelationProtocol(next(iter(schemas.values())), *roles)


def w06_r03_intensity_resolver(
        adapter: W06TypedAdapterOutput,
        protocol: PropertyRelationProtocol,
        ) -> MappingPropertyIntensityResolver:
    """从每条 train candidate 的 rational role value 建立显式强度映射。"""
    values: dict[ObjectIdentity, Rational] = {}
    for candidate in adapter.candidates_for_substage(W06_R03_SUBSTAGE):
        raw = candidate.rational_role_values
        if raw is None:
            raise W06R03ContractError("R03 candidate 缺少 intensity Rational")
        payload = raw.to_value()
        rows = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or len(rows) != 1:
            raise W06R03ContractError("R03 intensity Rational 数量漂移")
        row = rows[0]
        if not isinstance(row, dict):
            raise W06R03ContractError("R03 intensity Rational row 非 object")
        role = ObjectIdentity.from_stable_key(tuple(row["role_key"]))
        filler = ObjectIdentity.from_stable_key(tuple(row["filler_key"]))
        if role != protocol.intensity_role:
            raise W06R03ContractError("R03 intensity Role 漂移")
        if filler != candidate_claim(candidate, protocol).intensity:
            raise W06R03ContractError("R03 intensity filler 与 claim 漂移")
        rational = Rational(row["num"], row["den"])
        prior = values.get(filler)
        if prior is not None and prior != rational:
            raise W06R03ContractError("R03 同一 intensity 映射冲突")
        values[filler] = rational
    return MappingPropertyIntensityResolver(tuple(values.items()))


def slice_w06_r03_adapter(
        adapter: W06TypedAdapterOutput,
        ) -> W06TypedAdapterOutput:
    """只保留 7 条 PROPERTY train candidate 并隔离其他关系。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("R03 slice 需要 W06TypedAdapterOutput")
    candidates = adapter.candidates_for_substage(W06_R03_SUBSTAGE)
    if len(candidates) != 7 or any(
            item.relation_family != W06_R03_RELATION for item in candidates):
        raise W06R03ContractError("R03 PROPERTY train inventory 漂移")
    if {item.observation.split for item in candidates} != {"train"}:
        raise W06R03ContractError("R03 只允许 train candidate")
    candidate_ids = {item.proposition.proposition for item in candidates}
    sources = {item.source_record.stable_key for item in candidates}
    real_schemas = {item.schema.schema: item.schema for item in candidates}
    if len(real_schemas) != 1:
        raise W06R03ContractError("R03 PROPERTY schema identity 不唯一")
    rejections = tuple(
        item for item in adapter.rejections
        if item.substage_key == W06_R03_SUBSTAGE)
    if rejections:
        raise W06R03ContractError("R03 PROPERTY 不应有 schema rejection")
    return W06TypedAdapterOutput(
        tuple(item for item in adapter.source_bindings
              if item.record.stable_key in sources),
        tuple(item for item in adapter.observations
              if item.candidate.proposition.proposition in candidate_ids),
        candidates,
        tuple(real_schemas.values()),
        tuple(item for item in adapter.evidence
              if item.candidate in candidate_ids),
        (),
        (),
        adapter.execution_state,
    )


class W06R03View:
    """共享同一 W06 learning owner 的 direct PROPERTY 只读视图。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            protocol: W06R03ConsumerProtocol,
            ) -> None:
        """绑定隔离 candidate 集、PROPERTY protocol 与强度 resolver。"""
        if not isinstance(learning, W06RelationLearningRuntime):
            raise TypeError("R03 learning 类型非法")
        if not isinstance(adapter, W06TypedAdapterOutput):
            raise TypeError("R03 adapter 类型非法")
        if not isinstance(protocol, W06R03ConsumerProtocol):
            raise TypeError("R03 consumer protocol 类型非法")
        candidates = adapter.candidates_for_substage(W06_R03_SUBSTAGE)
        registered = learning.registered_candidates()
        if {item.proposition.proposition for item in registered} != {
                item.proposition.proposition for item in candidates}:
            raise W06R03ContractError(
                "R03 runtime 必须绑定隔离的 PROPERTY learning owner")
        if learning.closure is None:
            raise W06R03ContractError("R03 learning 缺少 R-00 closure")
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.candidates = candidates
        self.candidate_by_id = {
            item.proposition.proposition: item for item in candidates
        }
        self.property_protocol = w06_r03_property_protocol(adapter)
        self.intensity_resolver = w06_r03_intensity_resolver(
            adapter, self.property_protocol)

    def runtime(self, budget: PropertyQueryBudget) -> PropertyRelationRuntime:
        """按调用方预算建立共享 R-00 owner 的 PROPERTY facade。"""
        assert self.learning.closure is not None
        return PropertyRelationRuntime(
            self.learning.closure,
            self.property_protocol,
            budget,
            self.intensity_resolver,
        )

    def select(
            self,
            pattern: PropertyPattern,
            budget: PropertyQueryBudget,
            use_key: tuple[int, ...] | None = None,
            ):
        """读取或采用 direct PROPERTY selection，不执行属性路径闭包。"""
        return self.runtime(budget).select(PropertyQuery(pattern, use_key))

    def claim_for(self, candidate: W06RelationCandidate) -> PropertyClaim:
        """返回 candidate 的完整六维 typed claim。"""
        return candidate_claim(candidate, self.property_protocol)

    def pattern_for(self, candidate: W06RelationCandidate) -> PropertyPattern:
        """以 candidate subject/attribute 构造 direct Understanding pattern。"""
        claim = self.claim_for(candidate)
        return PropertyPattern(claim.subject, claim.attribute)

    def exact_pattern(self, claim: PropertyClaim) -> PropertyPattern:
        """以完整六维 claim 构造 direct Reasoning/Generation pattern。"""
        return PropertyPattern(
            claim.subject,
            claim.attribute,
            claim.value,
            claim.polarity,
            claim.modality,
            claim.intensity,
        )

    def authorization_key(
            self, candidate: W06RelationCandidate,
            ) -> LosslessIntegerKey | None:
        """从 current active fact 构造 source-grounded generation 授权键。"""
        snapshot = self.learning.snapshot_for(
            candidate.proposition.proposition)
        fact = snapshot.active_fact
        if fact is None:
            return None
        values = [
            W06_R03_RUNTIME_NAMESPACE,
            700,
            *pack_key(fact.proposition.proposition.stable_key()),
            *pack_key(fact.hypothesis.stable_key()),
            len(fact.evidence_keys),
        ]
        for item in fact.evidence_keys:
            values.extend(pack_key(item))
        values.extend(pack_key(fact.decision_key))
        return LosslessIntegerKey(tuple(values))


__all__ = [
    "W06R03View",
    "W06_R03_RELATION",
    "W06_R03_ROLE_KINDS",
    "W06_R03_RUNTIME_NAMESPACE",
    "candidate_claim",
    "candidate_construction",
    "candidate_role_fillers",
    "slice_w06_r03_adapter",
    "w06_r03_intensity_resolver",
    "w06_r03_language_branch",
    "w06_r03_property_protocol",
]
