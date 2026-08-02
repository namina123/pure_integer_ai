"""W06-R01 adapter slice、typed endpoint 与共享 active view。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProtocol,
    AliasResolutionSelector,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ROLE,
    ObjectIdentity,
    language_branch_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.experiments.alias_relation_runtime import AliasRelationRuntime
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_PURE_ALIAS,
    RELATION_REFERS,
    ROLE_ALIAS_LEFT,
    ROLE_ALIAS_RIGHT,
    ROLE_REFERS_FROM,
    ROLE_REFERS_TO,
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
from pure_integer_ai.experiments.ph2_w06_r01_contract import (
    W06R01ConsumerProtocol,
    W06R01ContractError,
    W06_R01_RELATION_FAMILIES,
    W06_R01_SUBSTAGE,
)


W06_R01_RUNTIME_NAMESPACE = 50601


def pack_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def candidate_role_fillers(
        candidate: W06RelationCandidate,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    """从规范 RoleBinding 恢复 relation 结构，不读取 surface cue。"""
    return tuple(sorted(
        ((item.role, item.filler)
         for item in candidate.proposition.canonical_bindings()),
        key=lambda item: item[0].stable_key(),
    ))


def _binding(
        candidate: W06RelationCandidate,
        role: ObjectIdentity,
        ) -> ObjectIdentity:
    """读取一个一对一 relation Role filler，缺失或重复即失败。"""
    values = tuple(
        item.filler for item in candidate.proposition.canonical_bindings()
        if item.role == role
    )
    if len(values) != 1:
        raise W06R01ContractError("W06-R01 relation Role 未恰好绑定一次")
    return values[0]


def candidate_endpoints(
        candidate: W06RelationCandidate,
        ) -> tuple[ObjectIdentity, ObjectIdentity]:
    """按 PURE_ALIAS 对称或 REFERS 有向 Role 返回查询端点。"""
    if candidate.relation_family == "PURE_ALIAS":
        return (
            _binding(candidate, authored_relation_role_identity(ROLE_ALIAS_LEFT)),
            _binding(candidate, authored_relation_role_identity(ROLE_ALIAS_RIGHT)),
        )
    if candidate.relation_family == "REFERS":
        return (
            _binding(candidate, authored_relation_role_identity(ROLE_REFERS_FROM)),
            _binding(candidate, authored_relation_role_identity(ROLE_REFERS_TO)),
        )
    raise W06R01ContractError("candidate 不属于 W06-R01")


def w06_r01_language_branch(candidate: W06RelationCandidate) -> ObjectIdentity:
    """把来源记录的语言标识映射为一等 branch，不从 surface 猜语言。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R01_SUBSTAGE):
        raise W06R01ContractError("language branch candidate 不属于 R01")
    value = candidate.observation.language
    return language_branch_identity(
        (W06_NAMESPACE, 901, len(value), *(ord(item) for item in value)),
        versions=W06_IDENTITY_VERSIONS,
    )


def candidate_construction(candidate: W06RelationCandidate) -> ObjectIdentity:
    """把来源化 template family 提升为可复用结构身份。"""
    key = candidate.observation.template_group_key.components
    return structure_concept_identity(
        (W06_NAMESPACE, 902, len(key), *key),
        versions=W06_IDENTITY_VERSIONS,
    )


def slice_w06_r01_adapter(
        adapter: W06TypedAdapterOutput,
        ) -> W06TypedAdapterOutput:
    """只保留 R01 train 记录，禁止把后续关系作为本阶段已学事实。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("R01 slice 需要 W06TypedAdapterOutput")
    candidates = adapter.candidates_for_substage(W06_R01_SUBSTAGE)
    if ({item.relation_family for item in candidates}
            != set(W06_R01_RELATION_FAMILIES)):
        raise W06R01ContractError("R01 train 未同时覆盖 PURE_ALIAS/REFERS")
    candidate_ids = {item.proposition.proposition for item in candidates}
    sources = {item.source_record.stable_key for item in candidates}
    schemas = tuple(sorted(
        {item.schema for item in candidates},
        key=lambda item: item.schema.stable_key(),
    ))
    return W06TypedAdapterOutput(
        tuple(item for item in adapter.source_bindings
              if item.record.stable_key in sources),
        tuple(item for item in adapter.observations
              if item.candidate.proposition.proposition in candidate_ids),
        candidates,
        schemas,
        tuple(item for item in adapter.evidence
              if item.candidate in candidate_ids),
        tuple(item for item in adapter.rejections
              if item.substage_key == W06_R01_SUBSTAGE),
        (),
        adapter.execution_state,
    )


def w06_r01_alias_protocol(
        adapter: W06TypedAdapterOutput,
        ) -> AliasResolutionProtocol:
    """从 R01 schema 注入现役 alias selector，不注册篇章 reference。"""
    candidates = adapter.candidates_for_substage(W06_R01_SUBSTAGE)
    alias_schemas = tuple(sorted(
        {item.schema.schema for item in candidates
         if item.relation_family == "PURE_ALIAS"},
        key=ObjectIdentity.stable_key,
    ))
    refers_schemas = tuple(sorted(
        {item.schema.schema for item in candidates
         if item.relation_family == "REFERS"},
        key=ObjectIdentity.stable_key,
    ))
    if not alias_schemas or not refers_schemas:
        raise W06R01ContractError("R01 alias/refers schema 不完整")
    placeholder_relation = ObjectIdentity(
        authored_relation_identity(RELATION_REFERS).object_kind,
        (W06_NAMESPACE, 910, 1),
        versions=W06_IDENTITY_VERSIONS,
    )
    placeholder_schema = structure_concept_identity(
        (W06_NAMESPACE, 910, 2), versions=W06_IDENTITY_VERSIONS)
    placeholder_roles = tuple(
        ObjectIdentity(
            OBJECT_ROLE,
            (W06_NAMESPACE, 910, ordinal),
            versions=W06_IDENTITY_VERSIONS,
        )
        for ordinal in range(3, 6)
    )
    instructions = tuple(
        minimal_instruction_identity(
            (W06_NAMESPACE, 911, ordinal),
            versions=W06_IDENTITY_VERSIONS,
        )
        for ordinal in range(1, 7)
    )
    return AliasResolutionProtocol(
        authored_relation_identity(RELATION_PURE_ALIAS),
        alias_schemas,
        authored_relation_role_identity(ROLE_ALIAS_LEFT),
        authored_relation_role_identity(ROLE_ALIAS_RIGHT),
        instructions[0],
        authored_relation_identity(RELATION_REFERS),
        refers_schemas,
        authored_relation_role_identity(ROLE_REFERS_FROM),
        authored_relation_role_identity(ROLE_REFERS_TO),
        instructions[1],
        placeholder_relation,
        (placeholder_schema,),
        placeholder_roles[0],
        placeholder_roles[1],
        placeholder_roles[2],
        instructions[2],
        instructions[3],
        instructions[4],
        instructions[5],
    )


class W06R01View:
    """共享同一 learning owner、candidate 集和 alias route selector 的只读视图。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            protocol: W06R01ConsumerProtocol,
            ) -> None:
        if not isinstance(learning, W06RelationLearningRuntime):
            raise TypeError("R01 learning 类型非法")
        if not isinstance(adapter, W06TypedAdapterOutput):
            raise TypeError("R01 adapter 类型非法")
        if not isinstance(protocol, W06R01ConsumerProtocol):
            raise TypeError("R01 consumer protocol 类型非法")
        candidates = adapter.candidates_for_substage(W06_R01_SUBSTAGE)
        registered = learning.registered_candidates()
        if ({item.proposition.proposition for item in registered}
                != {item.proposition.proposition for item in candidates}):
            raise W06R01ContractError(
                "R01 runtime 必须绑定隔离的 PURE_ALIAS_REFERS learning owner")
        if learning.closure is None:
            raise W06R01ContractError("R01 learning 缺少 R-00 closure")
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.candidates = candidates
        self.candidate_by_id = {
            item.proposition.proposition: item for item in candidates
        }
        self.alias = AliasRelationRuntime(
            learning.closure,
            AliasResolutionSelector(w06_r01_alias_protocol(adapter)),
        )


__all__ = [
    "W06R01View",
    "W06_R01_RUNTIME_NAMESPACE",
    "candidate_construction",
    "candidate_endpoints",
    "candidate_role_fillers",
    "pack_key",
    "slice_w06_r01_adapter",
    "w06_r01_alias_protocol",
    "w06_r01_language_branch",
]
