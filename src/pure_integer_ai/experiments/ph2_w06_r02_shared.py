"""W06-R02 adapter slice、R-02 protocol 与共享 active view。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ROLE,
    OBJECT_SET_EXPR,
    ObjectIdentity,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.set_relation import (
    MappingMemberTypeResolver,
    SetBinaryRelationProtocol,
    SetRelationProtocol,
    SetRelationRules,
    SetRelationStatement,
    SetUnaryRelationProtocol,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    ROLE_MEMBER_ELEMENT,
    ROLE_MEMBER_SET,
    ROLE_SUBSET_CHILD,
    ROLE_SUBSET_PARENT,
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
from pure_integer_ai.experiments.ph2_w06_r02_contract import (
    W06R02ConsumerProtocol,
    W06R02ContractError,
    W06R02SetQuery,
    W06_R02_RELATION_FAMILIES,
    W06_R02_SUBSTAGE,
    pack_key,
)
from pure_integer_ai.experiments.set_relation_runtime import (
    SetRelationEndpointResolver,
    SetRelationQuery,
    SetRelationRuntime,
)


W06_R02_RUNTIME_NAMESPACE = 50602


def candidate_role_fillers(
        candidate: W06RelationCandidate,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    """按规范 Role 排序恢复 relation 结构，不读取 surface cue。"""
    return tuple(sorted(
        ((item.role, item.filler)
         for item in candidate.proposition.canonical_bindings()),
        key=lambda item: item[0].stable_key(),
    ))


def _binding(
        candidate: W06RelationCandidate,
        role: ObjectIdentity,
        ) -> ObjectIdentity:
    values = tuple(
        item.filler for item in candidate.proposition.canonical_bindings()
        if item.role == role
    )
    if len(values) != 1:
        raise W06R02ContractError("W06-R02 relation Role 未恰好绑定一次")
    return values[0]


def candidate_endpoints(
        candidate: W06RelationCandidate,
        ) -> tuple[ObjectIdentity, ObjectIdentity]:
    """按 SUBSET/MEMBER 有向 Role 返回左、右端点。"""
    roles = {
        "SUBSET": (ROLE_SUBSET_CHILD, ROLE_SUBSET_PARENT),
        "MEMBER": (ROLE_MEMBER_ELEMENT, ROLE_MEMBER_SET),
    }.get(candidate.relation_family)
    if roles is None:
        raise W06R02ContractError("candidate 不属于 W06-R02")
    return tuple(
        _binding(candidate, authored_relation_role_identity(role))
        for role in roles
    )


def w06_r02_language_branch(candidate: W06RelationCandidate) -> ObjectIdentity:
    """从来源记录的语言标识构造一等 branch。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R02_SUBSTAGE):
        raise W06R02ContractError("language branch candidate 不属于 R02")
    value = candidate.observation.language
    return language_branch_identity(
        (W06_NAMESPACE, 921, len(value), *(ord(item) for item in value)),
        versions=W06_IDENTITY_VERSIONS,
    )


def candidate_construction(candidate: W06RelationCandidate) -> ObjectIdentity:
    """把来源化 template family 提升为可复用结构身份。"""
    key = candidate.observation.template_group_key.components
    return structure_concept_identity(
        (W06_NAMESPACE, 922, len(key), *key),
        versions=W06_IDENTITY_VERSIONS,
    )


def _placeholder_role(ordinal: int) -> ObjectIdentity:
    return ObjectIdentity(
        OBJECT_ROLE,
        (W06_R02_RUNTIME_NAMESPACE, 10, ordinal),
        versions=W06_IDENTITY_VERSIONS,
    )


def _placeholder_binary(
        ordinal: int,
        left_role: ObjectIdentity,
        right_role: ObjectIdentity,
        ) -> RelationSchema:
    """建立 R-02 完整协议所需但本片不形成事实的 typed schema。"""
    return RelationSchema(
        structure_concept_identity(
            (W06_R02_RUNTIME_NAMESPACE, 20, ordinal),
            versions=W06_IDENTITY_VERSIONS,
        ),
        concept_identity(
            (W06_R02_RUNTIME_NAMESPACE, 21, ordinal),
            versions=W06_IDENTITY_VERSIONS,
        ),
        (
            RelationSlotSchema(
                left_role, frozenset({OBJECT_SET_EXPR}), 1, 1),
            RelationSlotSchema(
                right_role, frozenset({OBJECT_SET_EXPR}), 1, 1),
        ),
    )


def w06_r02_placeholder_schemas() -> tuple[RelationSchema, ...]:
    """返回 PROPER/EQUAL/CLOSED 空 schema，不赋予它们训练真值。"""
    roles = tuple(_placeholder_role(index) for index in range(1, 6))
    proper = _placeholder_binary(1, roles[0], roles[1])
    equal = _placeholder_binary(2, roles[2], roles[3])
    closed = RelationSchema(
        structure_concept_identity(
            (W06_R02_RUNTIME_NAMESPACE, 20, 3),
            versions=W06_IDENTITY_VERSIONS,
        ),
        concept_identity(
            (W06_R02_RUNTIME_NAMESPACE, 21, 3),
            versions=W06_IDENTITY_VERSIONS,
        ),
        (RelationSlotSchema(
            roles[4], frozenset({OBJECT_SET_EXPR}), 1, 1),),
    )
    return proper, equal, closed


def slice_w06_r02_adapter(
        adapter: W06TypedAdapterOutput,
        ) -> W06TypedAdapterOutput:
    """只保留 R02 train 候选和类型拒绝，并注册 R-02 空协议 schema。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("R02 slice 需要 W06TypedAdapterOutput")
    candidates = adapter.candidates_for_substage(W06_R02_SUBSTAGE)
    if ({item.relation_family for item in candidates}
            != set(W06_R02_RELATION_FAMILIES)):
        raise W06R02ContractError("R02 train 未同时覆盖 SUBSET/MEMBER")
    candidate_ids = {item.proposition.proposition for item in candidates}
    sources = {item.source_record.stable_key for item in candidates}
    rejections = tuple(
        item for item in adapter.rejections
        if item.substage_key == W06_R02_SUBSTAGE)
    rejection_observations = {item.observation for item in rejections}
    real_schemas = {item.schema.schema: item.schema for item in candidates}
    placeholders = w06_r02_placeholder_schemas()
    schemas = tuple(sorted(
        (*real_schemas.values(), *placeholders),
        key=lambda item: item.schema.stable_key(),
    ))
    if len(schemas) != 5:
        raise W06R02ContractError("R02 protocol schema 必须恰有五类")
    return W06TypedAdapterOutput(
        tuple(item for item in adapter.source_bindings
              if item.record.stable_key in sources),
        tuple(item for item in adapter.observations
              if item.candidate.proposition.proposition in candidate_ids),
        candidates,
        schemas,
        tuple(item for item in adapter.evidence
              if item.candidate in candidate_ids),
        rejections,
        tuple(item for item in adapter.rejection_evidence
              if item.observation in rejection_observations),
        adapter.execution_state,
    )


def w06_r02_set_protocol(
        adapter: W06TypedAdapterOutput,
        ) -> SetRelationProtocol:
    """把 W-06 SUBSET/MEMBER schema 注入现役完整 R-02 协议。"""
    candidates = adapter.candidates_for_substage(W06_R02_SUBSTAGE)
    by_family: dict[str, RelationSchema] = {}
    for family in W06_R02_RELATION_FAMILIES:
        schemas = {
            item.schema.schema: item.schema
            for item in candidates if item.relation_family == family
        }
        if len(schemas) != 1:
            raise W06R02ContractError("R02 relation family schema 不唯一")
        by_family[family] = next(iter(schemas.values()))
    proper_schema, equal_schema, closed_schema = w06_r02_placeholder_schemas()
    instructions = tuple(
        minimal_instruction_identity(
            (W06_R02_RUNTIME_NAMESPACE, 30, ordinal),
            versions=W06_IDENTITY_VERSIONS,
        )
        for ordinal in range(1, 15)
    )
    proper_roles = tuple(item.role for item in proper_schema.slots)
    equal_roles = tuple(item.role for item in equal_schema.slots)
    return SetRelationProtocol(
        SetBinaryRelationProtocol(
            by_family["SUBSET"],
            authored_relation_role_identity(ROLE_SUBSET_CHILD),
            authored_relation_role_identity(ROLE_SUBSET_PARENT),
        ),
        SetBinaryRelationProtocol(
            proper_schema, proper_roles[0], proper_roles[1]),
        SetBinaryRelationProtocol(
            by_family["MEMBER"],
            authored_relation_role_identity(ROLE_MEMBER_ELEMENT),
            authored_relation_role_identity(ROLE_MEMBER_SET),
        ),
        SetBinaryRelationProtocol(
            equal_schema, equal_roles[0], equal_roles[1]),
        SetUnaryRelationProtocol(closed_schema, closed_schema.slots[0].role),
        SetRelationRules(*instructions),
    )


class W06R02View:
    """共享同一 learning owner 与现役 SetRelationRuntime 的只读视图。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            protocol: W06R02ConsumerProtocol,
            endpoint_resolver: SetRelationEndpointResolver,
            ) -> None:
        if not isinstance(learning, W06RelationLearningRuntime):
            raise TypeError("R02 learning 类型非法")
        if not isinstance(adapter, W06TypedAdapterOutput):
            raise TypeError("R02 adapter 类型非法")
        if not isinstance(protocol, W06R02ConsumerProtocol):
            raise TypeError("R02 consumer protocol 类型非法")
        if not isinstance(endpoint_resolver, SetRelationEndpointResolver):
            raise TypeError("R02 endpoint resolver 类型非法")
        candidates = adapter.candidates_for_substage(W06_R02_SUBSTAGE)
        registered = learning.registered_candidates()
        if ({item.proposition.proposition for item in registered}
                != {item.proposition.proposition for item in candidates}):
            raise W06R02ContractError(
                "R02 runtime 必须绑定隔离的 SUBSET_MEMBER learning owner")
        if learning.closure is None:
            raise W06R02ContractError("R02 learning 缺少 R-00 closure")
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.endpoint_resolver = endpoint_resolver
        self.candidates = candidates
        self.candidate_by_id = {
            item.proposition.proposition: item for item in candidates
        }
        self.set_protocol = w06_r02_set_protocol(adapter)
        self.member_type_resolver = MappingMemberTypeResolver(())

    def statement_for(self, query: W06R02SetQuery) -> SetRelationStatement:
        """把公开 family 映射为 R-02 注入 relation identity。"""
        relation = (
            self.set_protocol.subset_eq.relation
            if query.relation_family == "SUBSET"
            else self.set_protocol.member.relation
        )
        return SetRelationStatement(relation, query.left, query.right)

    def set_runtime(self, query: W06R02SetQuery) -> SetRelationRuntime:
        """按调用方显式预算创建共享 owner 的 R-02 facade。"""
        assert self.learning.closure is not None
        return SetRelationRuntime(
            self.learning.closure,
            self.set_protocol,
            query.budget,
            self.member_type_resolver,
            self.endpoint_resolver,
        )

    def endpoints_for(
            self, candidate: W06RelationCandidate,
            ) -> tuple[ObjectIdentity, ObjectIdentity]:
        """经显式 projection 返回可跨命题比较的 canonical 端点。"""
        return tuple(
            self.endpoint_resolver.resolve(item)
            for item in candidate_endpoints(candidate)
        )

    def evaluate(self, query: W06R02SetQuery):
        """无写入地执行一次完整集合闭包查询。"""
        return self.set_runtime(query).evaluate(
            SetRelationQuery(self.statement_for(query)))

    def commit(self, query: W06R02SetQuery, use_key: LosslessIntegerKey):
        """重算查询并原子提交全部 active proof 前提 Use。"""
        return self.set_runtime(query).evaluate(SetRelationQuery(
            self.statement_for(query), use_key.components))

    def current_conflicts(
            self, query: W06R02SetQuery,
            ) -> tuple[W06RelationCandidate, ...]:
        """返回与查询内容精确匹配的 current ACTIVE+CONFLICTED 候选。"""
        result = []
        for candidate in self.candidates:
            if candidate.relation_family != query.relation_family:
                continue
            if self.endpoints_for(candidate) != (query.left, query.right):
                continue
            snapshot = self.learning.snapshot_for(
                candidate.proposition.proposition)
            if (snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
                    and snapshot.snapshot.epistemic_status
                    == EPISTEMIC_CONFLICTED):
                result.append(candidate)
        return tuple(sorted(
            result, key=lambda item: item.proposition.proposition.stable_key()))

    def authorization_key(
            self, candidate: W06RelationCandidate,
            ) -> LosslessIntegerKey | None:
        """从 current active fact 的 Hypothesis/Evidence/H-04 构造授权键。"""
        snapshot = self.learning.snapshot_for(
            candidate.proposition.proposition)
        fact = snapshot.active_fact
        if fact is None:
            return None
        values = [
            W06_R02_RUNTIME_NAMESPACE,
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
    "W06R02View",
    "W06_R02_RUNTIME_NAMESPACE",
    "candidate_construction",
    "candidate_endpoints",
    "candidate_role_fillers",
    "slice_w06_r02_adapter",
    "w06_r02_language_branch",
    "w06_r02_placeholder_schemas",
    "w06_r02_set_protocol",
]
