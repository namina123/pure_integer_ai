"""W06-R05 adapter slice、双 channel protocol 与共享 active view。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    concept_identity,
    language_branch_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricPairPattern,
    SymmetricRelationBudget,
    SymmetricRelationProtocol,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    IrreflexiveRule,
    SymmetricRule,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
    authored_relation_rule_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_ANTONYM,
    RELATION_SIMILAR,
    ROLE_ANTONYM_LEFT,
    ROLE_ANTONYM_RIGHT,
    ROLE_SIMILAR_LEFT,
    ROLE_SIMILAR_RIGHT,
)
from pure_integer_ai.experiments.ph2_authored_semantic_pair_course import (
    ANTONYM_IRREFLEXIVE_RULE_KIND,
    ANTONYM_SYMMETRIC_RULE_KIND,
    SIMILAR_SYMMETRIC_RULE_KIND,
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
from pure_integer_ai.experiments.ph2_w06_r05_contract import (
    W06R05ConsumerProtocol,
    W06R05ContractError,
    W06R05PairQuery,
    W06_R05_CONSUMERS,
    W06_R05_RELATION_FAMILIES,
    W06_R05_SUBSTAGE,
    pack_key,
)
from pure_integer_ai.experiments.symmetric_relation_runtime import (
    SymmetricPairQuery,
    SymmetricRelationChannelRuntime,
)


W06_R05_RUNTIME_NAMESPACE = 50605


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
        raise W06R05ContractError("W06-R05 relation Role 未恰好绑定一次")
    return values[0]


def candidate_endpoints(
        candidate: W06RelationCandidate,
        ) -> tuple[ObjectIdentity, ObjectIdentity]:
    """按 SIMILAR/ANTONYM 的 left、right Role 返回原始 typed 端点。"""
    roles = {
        "SIMILAR": (ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT),
        "ANTONYM": (ROLE_ANTONYM_LEFT, ROLE_ANTONYM_RIGHT),
    }.get(candidate.relation_family)
    if roles is None:
        raise W06R05ContractError("candidate 不属于 W06-R05")
    return tuple(
        _binding(candidate, authored_relation_role_identity(role))
        for role in roles
    )


def w06_r05_language_branch(candidate: W06RelationCandidate) -> ObjectIdentity:
    """从来源记录的语言标识构造一等 branch。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R05_SUBSTAGE):
        raise W06R05ContractError("language branch candidate 不属于 R05")
    value = candidate.observation.language
    return language_branch_identity(
        (W06_NAMESPACE, 951, len(value), *(ord(item) for item in value)),
        versions=W06_IDENTITY_VERSIONS,
    )


def candidate_construction(candidate: W06RelationCandidate) -> ObjectIdentity:
    """把来源化 template family 提升为可复用结构身份。"""
    key = candidate.observation.template_group_key.components
    return structure_concept_identity(
        (W06_NAMESPACE, 952, len(key), *key),
        versions=W06_IDENTITY_VERSIONS,
    )


def slice_w06_r05_adapter(
        adapter: W06TypedAdapterOutput,
        ) -> W06TypedAdapterOutput:
    """只保留 R05 train 候选；本片没有 schema rejection。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("R05 slice 需要 W06TypedAdapterOutput")
    candidates = adapter.candidates_for_substage(W06_R05_SUBSTAGE)
    if len(candidates) != 7:
        raise W06R05ContractError("R05 train candidate inventory 漂移")
    if ({item.relation_family for item in candidates}
            != set(W06_R05_RELATION_FAMILIES)):
        raise W06R05ContractError("R05 train 未同时覆盖 SIMILAR/ANTONYM")
    candidate_ids = {item.proposition.proposition for item in candidates}
    sources = {item.source_record.stable_key for item in candidates}
    rejections = tuple(
        item for item in adapter.rejections
        if item.substage_key == W06_R05_SUBSTAGE)
    if rejections:
        raise W06R05ContractError("R05 不应含 schema rejection")
    real_schemas = {item.schema.schema: item.schema for item in candidates}
    if len(real_schemas) != 2:
        raise W06R05ContractError("R05 protocol schema 必须恰有两类")
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


def w06_r05_symmetric_protocols(
        adapter: W06TypedAdapterOutput,
        ) -> tuple[SymmetricRelationProtocol, SymmetricRelationProtocol]:
    """把 W-06 SIMILAR/ANTONYM schema 注入现役 R-05 协议。"""
    candidates = adapter.candidates_for_substage(W06_R05_SUBSTAGE)
    by_family = {}
    for family in W06_R05_RELATION_FAMILIES:
        schemas = {
            item.schema.schema: item.schema
            for item in candidates if item.relation_family == family
        }
        if len(schemas) != 1:
            raise W06R05ContractError("R05 relation family schema 不唯一")
        by_family[family] = next(iter(schemas.values()))

    similar_left = authored_relation_role_identity(ROLE_SIMILAR_LEFT)
    similar_right = authored_relation_role_identity(ROLE_SIMILAR_RIGHT)
    similar_relation = authored_relation_identity(RELATION_SIMILAR)
    similar = SymmetricRelationProtocol(
        by_family["SIMILAR"],
        similar_left,
        similar_right,
        SymmetricRule(
            authored_relation_rule_identity(SIMILAR_SYMMETRIC_RULE_KIND),
            similar_relation,
            similar_left,
            similar_right,
        ),
    )

    antonym_left = authored_relation_role_identity(ROLE_ANTONYM_LEFT)
    antonym_right = authored_relation_role_identity(ROLE_ANTONYM_RIGHT)
    antonym_relation = authored_relation_identity(RELATION_ANTONYM)
    antonym = SymmetricRelationProtocol(
        by_family["ANTONYM"],
        antonym_left,
        antonym_right,
        SymmetricRule(
            authored_relation_rule_identity(ANTONYM_SYMMETRIC_RULE_KIND),
            antonym_relation,
            antonym_left,
            antonym_right,
        ),
        IrreflexiveRule(
            authored_relation_rule_identity(ANTONYM_IRREFLEXIVE_RULE_KIND),
            antonym_relation,
            antonym_left,
            antonym_right,
        ),
    )
    return similar, antonym


class W06R05View:
    """共享同一 W06 learning owner 与现役 symmetric channel facade。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            protocol: W06R05ConsumerProtocol,
            ) -> None:
        """绑定隔离的 R05 learning owner、双 channel 协议和 consumer 门。"""
        if not isinstance(learning, W06RelationLearningRuntime):
            raise TypeError("R05 learning 类型非法")
        if not isinstance(adapter, W06TypedAdapterOutput):
            raise TypeError("R05 adapter 类型非法")
        if not isinstance(protocol, W06R05ConsumerProtocol):
            raise TypeError("R05 consumer protocol 类型非法")
        candidates = adapter.candidates_for_substage(W06_R05_SUBSTAGE)
        registered = learning.registered_candidates()
        if ({item.proposition.proposition for item in registered}
                != {item.proposition.proposition for item in candidates}):
            raise W06R05ContractError(
                "R05 runtime 必须绑定隔离的 SIMILAR_ANTONYM learning owner")
        if learning.closure is None:
            raise W06R05ContractError("R05 learning 缺少 R-00 closure")
        similar, antonym = w06_r05_symmetric_protocols(adapter)
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.candidates = candidates
        self.candidate_by_id = {
            item.proposition.proposition: item for item in candidates
        }
        self.protocols = {"SIMILAR": similar, "ANTONYM": antonym}

    def symmetric_protocol(self, channel: str) -> SymmetricRelationProtocol:
        """返回指定 channel 的 relation/schema/rule 协议。"""
        result = self.protocols.get(channel)
        if result is None:
            raise W06R05ContractError("R05 channel 未注册")
        return result

    def channel_runtime(
            self,
            channel: str,
            budget: SymmetricRelationBudget,
            ) -> SymmetricRelationChannelRuntime:
        """在共享 R-00 owner 上创建一个只读 symmetric channel facade。"""
        assert self.learning.closure is not None
        return SymmetricRelationChannelRuntime(
            self.learning.closure,
            self.symmetric_protocol(channel),
            budget,
        )

    def evaluate(self, query: W06R05PairQuery):
        """无写入执行一次 exact pair+channel 四态查询。"""
        runtime = self.channel_runtime(query.channel, query.budget)
        selection = runtime.select_many((SymmetricPairQuery(
            SymmetricPairPattern(query.left, query.right)),))[0]
        if len(selection.evaluations) != 1:
            raise W06R05ContractError("R05 exact query 未返回唯一 evaluation")
        return selection.evaluations[0]

    @staticmethod
    def _use_context(
            query: W06R05PairQuery,
            consumer: str,
            ) -> RelationUseContext:
        """从来源化 query 构造 consumer/purpose 均显式的 Use context。"""
        if consumer not in W06_R05_CONSUMERS:
            raise W06R05ContractError("R05 Use consumer 未注册")
        ordinal = W06_R05_CONSUMERS.index(consumer) + 1
        return RelationUseContext(
            query.source,
            document_scope(query.source),
            concept_identity((
                W06_R05_RUNTIME_NAMESPACE, 80, ordinal,
                *pack_key(query.request_key.components),
            )),
            concept_identity((
                W06_R05_RUNTIME_NAMESPACE, 81, ordinal,
                *pack_key(query.request_key.components),
            )),
        )

    def commit(
            self,
            query: W06R05PairQuery,
            use_key: LosslessIntegerKey,
            consumer: str,
            ):
        """重算 exact pair 并原子提交全部 current active premise Use。"""
        runtime = self.channel_runtime(query.channel, query.budget)
        return runtime.query(SymmetricPairQuery(
            SymmetricPairPattern(query.left, query.right),
            use_key.components,
            self._use_context(query, consumer),
        ))

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
            W06_R05_RUNTIME_NAMESPACE,
            700,
            *pack_key(fact.proposition.proposition.stable_key()),
            *pack_key(fact.hypothesis.stable_key()),
            len(fact.evidence_keys),
        ]
        for item in fact.evidence_keys:
            values.extend(pack_key(item))
        values.extend(pack_key(fact.decision_key))
        return LosslessIntegerKey(tuple(values))

    def consume_candidate(
            self,
            candidate: W06RelationCandidate,
            use_key: LosslessIntegerKey,
            ):
        """为 generation 精确提交一个 direct active target Proposition。"""
        if candidate not in self.candidates:
            raise W06R05ContractError("generation target 不属于 R05 view")
        if self.authorization_key(candidate) is None:
            raise W06R05ContractError("generation target 已失去 current authorization")
        assert self.learning.closure is not None
        query = W06R05PairQuery(
            LosslessIntegerKey((
                W06_R05_RUNTIME_NAMESPACE, 790, *use_key.components)),
            candidate.relation_family,
            *candidate_endpoints(candidate),
            SymmetricRelationBudget(32, 32),
            candidate.source_ref,
        )
        context = RelationUseContext(
            query.source,
            document_scope(query.source),
            concept_identity((W06_R05_RUNTIME_NAMESPACE, 82, 3)),
            concept_identity((
                W06_R05_RUNTIME_NAMESPACE, 83, *pack_key(use_key.components))),
        )
        return self.learning.closure.consume_many(((
            candidate.proposition.proposition,
            use_key.components,
            context,
        ),))


__all__ = [
    "W06R05View",
    "W06_R05_RUNTIME_NAMESPACE",
    "candidate_construction",
    "candidate_endpoints",
    "candidate_role_fillers",
    "slice_w06_r05_adapter",
    "w06_r05_language_branch",
    "w06_r05_symmetric_protocols",
]
