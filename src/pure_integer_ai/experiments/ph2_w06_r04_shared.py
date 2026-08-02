"""W06-R04 adapter slice、R-04 protocol 与共享 active view。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ROLE,
    ObjectIdentity,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.mereology_relation import (
    MereologyBudget,
    MereologyPattern,
    MereologyProtocol,
    MereologyRelationProtocol,
    MereologyStatement,
)
from pure_integer_ai.cognition.shared.typed_relation import InverseRule
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
    authored_relation_rule_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_HAS_PART,
    RELATION_PART_OF,
    ROLE_HAS_PART_PART,
    ROLE_HAS_PART_WHOLE,
    ROLE_PART_OF_PART,
    ROLE_PART_OF_WHOLE,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.mereology_relation_runtime import (
    MereologyEndpointResolver,
    MereologyQuery,
    MereologyRelationRuntime,
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
from pure_integer_ai.experiments.ph2_w06_r04_contract import (
    W06R04ConsumerProtocol,
    W06R04ContractError,
    W06R04MereologyQuery,
    W06_R04_RELATION_FAMILIES,
    W06_R04_SUBSTAGE,
    pack_key,
)


W06_R04_RUNTIME_NAMESPACE = 50604


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
    """读取某个有名 Role 的唯一 filler。"""
    values = tuple(
        item.filler for item in candidate.proposition.canonical_bindings()
        if item.role == role
    )
    if len(values) != 1:
        raise W06R04ContractError("W06-R04 relation Role 未恰好绑定一次")
    return values[0]


def candidate_endpoints(
        candidate: W06RelationCandidate,
        ) -> tuple[ObjectIdentity, ObjectIdentity]:
    """按 PART_OF/HAS_PART 的 canonical part、whole 返回端点。"""
    roles = {
        "PART_OF": (ROLE_PART_OF_PART, ROLE_PART_OF_WHOLE),
        "HAS_PART": (ROLE_HAS_PART_PART, ROLE_HAS_PART_WHOLE),
    }.get(candidate.relation_family)
    if roles is None:
        raise W06R04ContractError("candidate 不属于 W06-R04")
    return tuple(
        _binding(candidate, authored_relation_role_identity(role))
        for role in roles
    )


def w06_r04_language_branch(candidate: W06RelationCandidate) -> ObjectIdentity:
    """从来源记录的语言标识构造一等 branch。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R04_SUBSTAGE):
        raise W06R04ContractError("language branch candidate 不属于 R04")
    value = candidate.observation.language
    return language_branch_identity(
        (W06_NAMESPACE, 941, len(value), *(ord(item) for item in value)),
        versions=W06_IDENTITY_VERSIONS,
    )


def candidate_construction(candidate: W06RelationCandidate) -> ObjectIdentity:
    """把来源化 template family 提升为可复用结构身份。"""
    key = candidate.observation.template_group_key.components
    return structure_concept_identity(
        (W06_NAMESPACE, 942, len(key), *key),
        versions=W06_IDENTITY_VERSIONS,
    )


def _placeholder_role(ordinal: int) -> ObjectIdentity:
    """建立本片不用但协议 state 需要的占位 Role。"""
    return ObjectIdentity(
        OBJECT_ROLE,
        (W06_R04_RUNTIME_NAMESPACE, 10, ordinal),
        versions=W06_IDENTITY_VERSIONS,
    )


def slice_w06_r04_adapter(
        adapter: W06TypedAdapterOutput,
        ) -> W06TypedAdapterOutput:
    """只保留 R04 train 候选；本片没有 schema rejection。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("R04 slice 需要 W06TypedAdapterOutput")
    candidates = adapter.candidates_for_substage(W06_R04_SUBSTAGE)
    if len(candidates) != 7:
        raise W06R04ContractError("R04 train candidate inventory 漂移")
    if ({item.relation_family for item in candidates}
            != set(W06_R04_RELATION_FAMILIES)):
        raise W06R04ContractError("R04 train 未同时覆盖 PART_OF/HAS_PART")
    candidate_ids = {item.proposition.proposition for item in candidates}
    sources = {item.source_record.stable_key for item in candidates}
    rejections = tuple(
        item for item in adapter.rejections
        if item.substage_key == W06_R04_SUBSTAGE)
    if rejections:
        raise W06R04ContractError("R04 不应含 schema rejection")
    real_schemas = {item.schema.schema: item.schema for item in candidates}
    if len(real_schemas) != 2:
        raise W06R04ContractError("R04 protocol schema 必须恰有两类")
    return W06TypedAdapterOutput(
        tuple(item for item in adapter.source_bindings
              if item.record.stable_key in sources),
        tuple(item for item in adapter.observations
              if item.candidate.proposition.proposition in candidate_ids),
        candidates,
        tuple(sorted(real_schemas.values(),
                     key=lambda item: item.schema.stable_key())),
        tuple(item for item in adapter.evidence
              if item.candidate in candidate_ids),
        (),
        (),
        adapter.execution_state,
    )


def w06_r04_mereology_protocol(
        adapter: W06TypedAdapterOutput,
        ) -> MereologyProtocol:
    """把 W-06 MEREOLOGY schema 注入现役 R-04 协议。"""
    candidates = adapter.candidates_for_substage(W06_R04_SUBSTAGE)
    by_family = {}
    for family in W06_R04_RELATION_FAMILIES:
        schemas = {
            item.schema.schema: item.schema
            for item in candidates if item.relation_family == family
        }
        if len(schemas) != 1:
            raise W06R04ContractError("R04 relation family schema 不唯一")
        by_family[family] = next(iter(schemas.values()))
    part_of = MereologyRelationProtocol(
        by_family["PART_OF"],
        authored_relation_role_identity(ROLE_PART_OF_PART),
        authored_relation_role_identity(ROLE_PART_OF_WHOLE),
    )
    has_part = MereologyRelationProtocol(
        by_family["HAS_PART"],
        authored_relation_role_identity(ROLE_HAS_PART_PART),
        authored_relation_role_identity(ROLE_HAS_PART_WHOLE),
    )
    inverse_rule = InverseRule(
        authored_relation_rule_identity(1),
        authored_relation_identity(RELATION_PART_OF),
        authored_relation_role_identity(ROLE_PART_OF_PART),
        authored_relation_role_identity(ROLE_PART_OF_WHOLE),
        authored_relation_identity(RELATION_HAS_PART),
        authored_relation_role_identity(ROLE_HAS_PART_WHOLE),
        authored_relation_role_identity(ROLE_HAS_PART_PART),
    )
    return MereologyProtocol(
        (part_of, has_part),
        inverse_rules=(inverse_rule,),
    )


class W06R04View:
    """共享同一 learning owner 与现役 MereologyRelationRuntime 的只读视图。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            protocol: W06R04ConsumerProtocol,
            endpoint_resolver: MereologyEndpointResolver,
            ) -> None:
        """绑定 R04 隔离 learning owner、consumer 协议和 endpoint resolver。"""
        if not isinstance(learning, W06RelationLearningRuntime):
            raise TypeError("R04 learning 类型非法")
        if not isinstance(adapter, W06TypedAdapterOutput):
            raise TypeError("R04 adapter 类型非法")
        if not isinstance(protocol, W06R04ConsumerProtocol):
            raise TypeError("R04 consumer protocol 类型非法")
        if not isinstance(endpoint_resolver, MereologyEndpointResolver):
            raise TypeError("R04 endpoint resolver 类型非法")
        candidates = adapter.candidates_for_substage(W06_R04_SUBSTAGE)
        registered = learning.registered_candidates()
        if ({item.proposition.proposition for item in registered}
                != {item.proposition.proposition for item in candidates}):
            raise W06R04ContractError(
                "R04 runtime 必须绑定隔离的 MEREOLOGY learning owner")
        if learning.closure is None:
            raise W06R04ContractError("R04 learning 缺少 R-00 closure")
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.endpoint_resolver = endpoint_resolver
        self.candidates = candidates
        self.candidate_by_id = {
            item.proposition.proposition: item for item in candidates
        }
        self.mereology_protocol = w06_r04_mereology_protocol(adapter)

    def relation_for(self, relation_family: str) -> ObjectIdentity:
        """把公开 family 映射为 R-04 注入 relation identity。"""
        if relation_family == "PART_OF":
            return authored_relation_identity(RELATION_PART_OF)
        if relation_family == "HAS_PART":
            return authored_relation_identity(RELATION_HAS_PART)
        raise W06R04ContractError("R04 relation family 未注册")

    def statement_for(self, query: W06R04MereologyQuery) -> MereologyStatement:
        """把 public query 转成 canonical MereologyStatement。"""
        return MereologyStatement(
            self.relation_for(query.relation_family),
            query.part,
            query.whole,
        )

    def mereology_runtime(
            self, budget: MereologyBudget,
            ) -> MereologyRelationRuntime:
        """按调用方显式预算创建共享 owner 的 R-04 facade。"""
        assert self.learning.closure is not None
        return MereologyRelationRuntime(
            self.learning.closure,
            self.mereology_protocol,
            budget,
            self.endpoint_resolver,
            False,
        )

    def endpoints_for(
            self, candidate: W06RelationCandidate,
            ) -> tuple[ObjectIdentity, ObjectIdentity]:
        """经显式 projection 返回可跨命题比较的 canonical 端点。"""
        return tuple(
            self.endpoint_resolver.resolve(item)
            for item in candidate_endpoints(candidate)
        )

    def evaluate(self, query: W06R04MereologyQuery):
        """无写入地执行一次完整部分整体闭包查询。"""
        runtime = self.mereology_runtime(query.budget)
        statement = self.statement_for(query)
        selection = runtime.engine().select(MereologyPattern(
            statement.relation,
            statement.part,
            statement.whole,
        ))
        if len(selection.evaluations) != 1:
            raise W06R04ContractError("R04 exact query 未返回唯一 evaluation")
        return selection.evaluations[0]

    def commit(
            self, query: W06R04MereologyQuery,
            use_key: LosslessIntegerKey,
            ):
        """重算查询并原子提交全部 active proof 前提 Use。"""
        runtime = self.mereology_runtime(query.budget)
        statement = self.statement_for(query)
        return runtime.query(MereologyQuery(MereologyPattern(
            statement.relation,
            statement.part,
            statement.whole,
        ), use_key.components))

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
            W06_R04_RUNTIME_NAMESPACE,
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
    "W06R04View",
    "W06_R04_RUNTIME_NAMESPACE",
    "candidate_construction",
    "candidate_endpoints",
    "candidate_role_fillers",
    "slice_w06_r04_adapter",
    "w06_r04_language_branch",
    "w06_r04_mereology_protocol",
]
