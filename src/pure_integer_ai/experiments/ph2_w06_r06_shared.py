"""W06-R06 adapter slice、event-time qualifier facade 与 active view。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_DIRECTION_UNKNOWN,
    EVENT_TIME_SAME,
    ResolvedEventTimeRelation,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    concept_identity,
    language_branch_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    ROLE_EVENT_AFTER_OBJECT,
    ROLE_EVENT_AFTER_SUBJECT,
    ROLE_EVENT_BEFORE_OBJECT,
    ROLE_EVENT_BEFORE_SUBJECT,
    ROLE_EVENT_SAME_OBJECT,
    ROLE_EVENT_SAME_SUBJECT,
    ROLE_EVENT_UNKNOWN_OBJECT,
    ROLE_EVENT_UNKNOWN_SUBJECT,
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
from pure_integer_ai.experiments.ph2_w06_r06_contract import (
    W06R06BudgetExceeded,
    W06R06ConsumerProtocol,
    W06R06ContractError,
    W06R06EventTimeEvaluation,
    W06R06EventTimeQuery,
    W06_R06_CONSUMERS,
    W06_R06_RELATION_FAMILIES,
    W06_R06_RUNTIME_NAMESPACE,
    W06_R06_SUBSTAGE,
    pack_key,
)


_ROLES = {
    "EVENT_BEFORE": (ROLE_EVENT_BEFORE_SUBJECT, ROLE_EVENT_BEFORE_OBJECT),
    "EVENT_AFTER": (ROLE_EVENT_AFTER_SUBJECT, ROLE_EVENT_AFTER_OBJECT),
    "EVENT_SAME": (ROLE_EVENT_SAME_SUBJECT, ROLE_EVENT_SAME_OBJECT),
    "EVENT_UNKNOWN": (ROLE_EVENT_UNKNOWN_SUBJECT, ROLE_EVENT_UNKNOWN_OBJECT),
}
_DIRECTIONS = {
    "EVENT_BEFORE": EVENT_TIME_BEFORE,
    "EVENT_AFTER": EVENT_TIME_AFTER,
    "EVENT_SAME": EVENT_TIME_SAME,
    "EVENT_UNKNOWN": EVENT_TIME_DIRECTION_UNKNOWN,
}
_PROTOCOL_FIELDS = {
    "causes_effect",
    "detail_key",
    "dimension_key",
    "direction",
    "hypothesis_kind_key",
    "object_role_key",
    "occurrence_order_consumed",
    "relation_key",
    "scope_key",
    "structure_order_consumed",
    "subject_role_key",
    "verifier_key",
}


def _strict_int_key(value, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W06R06ContractError(f"{where} 必须是非空严格整数列表")
    return tuple(value)


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
        raise W06R06ContractError("R06 event-time Role 未恰好绑定一次")
    return values[0]


def candidate_endpoints(
        candidate: W06RelationCandidate,
        ) -> tuple[ObjectIdentity, ObjectIdentity]:
    """按 family-specific subject/object Role 返回 typed 端点。"""
    roles = _ROLES.get(candidate.relation_family)
    if roles is None:
        raise W06R06ContractError("candidate 不属于 W06-R06")
    return tuple(
        _binding(candidate, authored_relation_role_identity(role))
        for role in roles
    )


def candidate_event_time_qualifier(
        candidate: W06RelationCandidate,
        ) -> ResolvedEventTimeRelation:
    """严格恢复 authored protocol 的方向和 qualifier detail。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R06_SUBSTAGE):
        raise W06R06ContractError("event-time qualifier candidate 不属于 R06")
    value = candidate.domain_protocols.to_value()
    if not isinstance(value, dict) or set(value) != {"event_time_protocol"}:
        raise W06R06ContractError("R06 domain protocol 信封漂移")
    protocol = value["event_time_protocol"]
    if not isinstance(protocol, dict) or set(protocol) != _PROTOCOL_FIELDS:
        raise W06R06ContractError("R06 event-time protocol 字段漂移")
    relation = ObjectIdentity.from_stable_key(_strict_int_key(
        protocol["relation_key"], where="event-time relation_key"))
    subject_role, object_role = tuple(
        authored_relation_role_identity(item)
        for item in _ROLES[candidate.relation_family]
    )
    if (relation != candidate.proposition.predicate
            or ObjectIdentity.from_stable_key(_strict_int_key(
                protocol["subject_role_key"], where="subject_role_key"))
            != subject_role
            or ObjectIdentity.from_stable_key(_strict_int_key(
                protocol["object_role_key"], where="object_role_key"))
            != object_role):
        raise W06R06ContractError("R06 relation/Role qualifier 漂移")
    if tuple(_strict_int_key(
            protocol["scope_key"], where="event-time scope_key")) != (
            document_scope(candidate.source_ref).stable_key()):
        raise W06R06ContractError("R06 qualifier scope 漂移")
    direction = protocol["direction"]
    if (type(direction) is not int
            or direction != _DIRECTIONS[candidate.relation_family]):
        raise W06R06ContractError("R06 event-time direction 漂移")
    for name in (
            "causes_effect", "occurrence_order_consumed",
            "structure_order_consumed"):
        if protocol[name] != 0 or type(protocol[name]) is not int:
            raise W06R06ContractError(
                "R06 event-time 不得消费 causal/occurrence/structure order")
    for name in ("dimension_key", "hypothesis_kind_key", "verifier_key"):
        _strict_int_key(protocol[name], where=name)
    return ResolvedEventTimeRelation(
        relation,
        direction,
        _strict_int_key(protocol["detail_key"], where="detail_key"),
    )


def w06_r06_language_branch(candidate: W06RelationCandidate) -> ObjectIdentity:
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R06_SUBSTAGE):
        raise W06R06ContractError("language branch candidate 不属于 R06")
    value = candidate.observation.language
    return language_branch_identity(
        (W06_NAMESPACE, 961, len(value), *(ord(item) for item in value)),
        versions=W06_IDENTITY_VERSIONS,
    )


def candidate_construction(candidate: W06RelationCandidate) -> ObjectIdentity:
    key = candidate.observation.template_group_key.components
    return structure_concept_identity(
        (W06_NAMESPACE, 962, len(key), *key),
        versions=W06_IDENTITY_VERSIONS,
    )


def slice_w06_r06_adapter(
        adapter: W06TypedAdapterOutput,
        ) -> W06TypedAdapterOutput:
    """只保留 PRECEDES/train 的九个合法候选。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("R06 slice 需要 W06TypedAdapterOutput")
    candidates = adapter.candidates_for_substage(W06_R06_SUBSTAGE)
    if len(candidates) != 9:
        raise W06R06ContractError("R06 train candidate inventory 漂移")
    if ({item.relation_family for item in candidates}
            != set(W06_R06_RELATION_FAMILIES)):
        raise W06R06ContractError("R06 train 未覆盖四种 event-time state")
    candidate_ids = {item.proposition.proposition for item in candidates}
    sources = {item.source_record.stable_key for item in candidates}
    rejections = tuple(
        item for item in adapter.rejections
        if item.substage_key == W06_R06_SUBSTAGE)
    if rejections:
        raise W06R06ContractError("R06 不应含 schema rejection")
    real_schemas = {item.schema.schema: item.schema for item in candidates}
    if len(real_schemas) != 4:
        raise W06R06ContractError("R06 protocol schema 必须恰有四类")
    sliced = W06TypedAdapterOutput(
        tuple(item for item in adapter.source_bindings
              if item.record.stable_key in sources),
        tuple(item for item in adapter.observations
              if item.candidate.proposition.proposition in candidate_ids),
        candidates,
        tuple(sorted(real_schemas.values(), key=lambda item: item.schema.stable_key())),
        tuple(item for item in adapter.evidence
              if item.candidate in candidate_ids),
        (),
        (),
        adapter.execution_state,
    )
    for candidate in sliced.candidates:
        candidate_event_time_qualifier(candidate)
    return sliced


class W06R06View:
    """共享唯一 W06 truth owner 的只读 event-time direction facade。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            protocol: W06R06ConsumerProtocol,
            endpoint_resolver,
            ) -> None:
        if not isinstance(learning, W06RelationLearningRuntime):
            raise TypeError("R06 learning 类型非法")
        if not isinstance(adapter, W06TypedAdapterOutput):
            raise TypeError("R06 adapter 类型非法")
        if not isinstance(protocol, W06R06ConsumerProtocol):
            raise TypeError("R06 consumer protocol 类型非法")
        if (not callable(getattr(endpoint_resolver, "resolve", None))
                or not callable(getattr(endpoint_resolver, "state_key", None))):
            raise TypeError("R06 endpoint resolver 协议非法")
        candidates = adapter.candidates_for_substage(W06_R06_SUBSTAGE)
        registered = learning.registered_candidates()
        if ({item.proposition.proposition for item in registered}
                != {item.proposition.proposition for item in candidates}):
            raise W06R06ContractError(
                "R06 runtime 必须绑定隔离的 PRECEDES learning owner")
        if learning.closure is None:
            raise W06R06ContractError("R06 learning 缺少 R-00 closure")
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.endpoint_resolver = endpoint_resolver
        self.candidates = candidates
        self.candidate_by_id = {
            item.proposition.proposition: item for item in candidates
        }
        self.qualifiers = {
            item.proposition.proposition: candidate_event_time_qualifier(item)
            for item in candidates
        }

    def endpoints_for(
            self, candidate: W06RelationCandidate,
            ) -> tuple[ObjectIdentity, ObjectIdentity]:
        return tuple(
            self.endpoint_resolver.resolve(item)
            for item in candidate_endpoints(candidate)
        )

    @staticmethod
    def _normalization(
            query: W06R06EventTimeQuery,
            ) -> tuple[tuple[ObjectIdentity, ...], tuple[ObjectIdentity, ...]]:
        if query.qualifier.direction == EVENT_TIME_BEFORE:
            return (query.subject, query.object_identity), ()
        if query.qualifier.direction == EVENT_TIME_AFTER:
            return (query.object_identity, query.subject), ()
        if query.qualifier.direction == EVENT_TIME_SAME:
            return (), tuple(sorted(
                (query.subject, query.object_identity),
                key=ObjectIdentity.stable_key,
            ))
        return (), ()

    def evaluate(
            self, query: W06R06EventTimeQuery,
            ) -> W06R06EventTimeEvaluation:
        """按完整 raw family/endpoints/qualifier 执行四态 event-time 查询。"""
        if query.budget.max_candidates < len(self.candidates):
            raise W06R06BudgetExceeded("R06 candidate scan budget 耗尽")
        matched = []
        snapshots = []
        for candidate in self.candidates:
            if (candidate.relation_family == query.relation_family
                    and self.endpoints_for(candidate) == (
                        query.subject, query.object_identity)
                    and self.qualifiers[candidate.proposition.proposition]
                    == query.qualifier):
                snapshot = self.learning.snapshot_for(
                    candidate.proposition.proposition)
                if snapshot.snapshot.lifecycle != LIFECYCLE_SUPERSEDED:
                    matched.append(candidate)
                    snapshots.append(snapshot)
        evidence_count = sum(len(item.evidence) for item in snapshots)
        if evidence_count > query.budget.max_evidence:
            raise W06R06BudgetExceeded("R06 Evidence scan budget 耗尽")

        support = any(
            item.active_fact is not None
            or item.snapshot.epistemic_status == EPISTEMIC_CONFLICTED
            for item in snapshots
        )
        refute = any(
            item.snapshot.epistemic_status in {
                EPISTEMIC_REFUTED, EPISTEMIC_CONFLICTED}
            for item in snapshots
        )
        explicit_unknown = any(
            item.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
            for item in snapshots
        )
        active = tuple(sorted(
            (
                candidate.proposition.proposition
                for candidate, snapshot in zip(matched, snapshots, strict=True)
                if snapshot.active_fact is not None
            ),
            key=ObjectIdentity.stable_key,
        ))
        propositions = tuple(sorted(
            (item.proposition.proposition for item in matched),
            key=ObjectIdentity.stable_key,
        ))
        evidence = tuple(sorted({
            record.stable_key()
            for snapshot in snapshots
            for record in snapshot.evidence
        }))
        before_edge, same_group = self._normalization(query)
        return W06R06EventTimeEvaluation(
            query,
            LogicEvidenceState(support, refute),
            explicit_unknown,
            before_edge,
            same_group,
            active,
            propositions,
            evidence,
        )

    @staticmethod
    def _use_context(
            query: W06R06EventTimeQuery,
            consumer: str,
            ) -> RelationUseContext:
        if consumer not in W06_R06_CONSUMERS:
            raise W06R06ContractError("R06 Use consumer 未注册")
        ordinal = W06_R06_CONSUMERS.index(consumer) + 1
        return RelationUseContext(
            query.source,
            document_scope(query.source),
            concept_identity((
                W06_R06_RUNTIME_NAMESPACE, 80, ordinal,
                *pack_key(query.request_key.components),
            )),
            concept_identity((
                W06_R06_RUNTIME_NAMESPACE, 81, ordinal,
                *pack_key(query.request_key.components),
            )),
        )

    def commit(
            self,
            query: W06R06EventTimeQuery,
            use_key: LosslessIntegerKey,
            consumer: str,
            ):
        evaluation = self.evaluate(query)
        if not evaluation.active_propositions:
            raise W06R06ContractError("R06 query 没有 current active premise")
        context = self._use_context(query, consumer)
        assert self.learning.closure is not None
        return self.learning.closure.consume_many(tuple(
            (
                proposition,
                (*use_key.components, ordinal),
                context,
            )
            for ordinal, proposition in enumerate(
                evaluation.active_propositions, start=1)
        ))

    def authorization_key(
            self, candidate: W06RelationCandidate,
            ) -> LosslessIntegerKey | None:
        snapshot = self.learning.snapshot_for(
            candidate.proposition.proposition)
        fact = snapshot.active_fact
        if fact is None:
            return None
        values = [
            W06_R06_RUNTIME_NAMESPACE,
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
        if candidate not in self.candidates:
            raise W06R06ContractError("generation target 不属于 R06 view")
        if self.authorization_key(candidate) is None:
            raise W06R06ContractError("generation target 已失去 current authorization")
        assert self.learning.closure is not None
        context = RelationUseContext(
            candidate.source_ref,
            document_scope(candidate.source_ref),
            concept_identity((W06_R06_RUNTIME_NAMESPACE, 82, 3)),
            concept_identity((
                W06_R06_RUNTIME_NAMESPACE, 83,
                *pack_key(use_key.components),
            )),
        )
        return self.learning.closure.consume_many(((
            candidate.proposition.proposition,
            use_key.components,
            context,
        ),))

    def state_key(self) -> tuple:
        assert self.learning.closure is not None
        return (
            self.learning.closure.state_key(),
            self.protocol.stable_key(),
            self.endpoint_resolver.state_key(),
            tuple(
                (item.proposition.proposition.stable_key(),
                 self.qualifiers[item.proposition.proposition].direction,
                 self.qualifiers[item.proposition.proposition].detail_key)
                for item in self.candidates
            ),
        )


__all__ = [
    "W06R06View",
    "candidate_construction",
    "candidate_endpoints",
    "candidate_event_time_qualifier",
    "candidate_role_fillers",
    "slice_w06_r06_adapter",
    "w06_r06_language_branch",
]
