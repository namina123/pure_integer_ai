"""W06-R07 adapter slice、causal protocol facade 与 active view。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.causal_execution import (
    CausalEndpointProtocol,
    causal_endpoints,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
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
from pure_integer_ai.experiments.causal_relation_runtime import (
    CausalIndependentWitness,
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
from pure_integer_ai.experiments.ph2_w06_r07_contract import (
    W06R07BudgetExceeded,
    W06R07CausalEvaluation,
    W06R07CausalQuery,
    W06R07ConsumerProtocol,
    W06R07ContractError,
    W06R07WitnessAccount,
    W06_R07_CONSUMERS,
    W06_R07_RUNTIME_NAMESPACE,
    W06_R07_SUBSTAGE,
    pack_key,
)


_PROTOCOL_FIELDS = {
    "causal_implies_event_time_fact",
    "cause_role_key",
    "counterfactual_verdict_claimed",
    "dimension_key",
    "effect_role_key",
    "evidence_target_kind_key",
    "execution_instruction_key",
    "forming_source_reusable_as_witness",
    "independent_witness_required",
    "occurrence_order_consumed",
    "precedence_implies_causation",
    "relation_key",
    "scope_key",
    "structure_order_consumed",
    "temporal_support_sufficient",
    "verifier_key",
}


def _strict_int_key(value, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W06R07ContractError(f"{where} 必须是非空严格整数列表")
    return tuple(value)


def candidate_role_fillers(
        candidate: W06RelationCandidate,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    return tuple(sorted(
        ((item.role, item.filler)
         for item in candidate.proposition.canonical_bindings()),
        key=lambda item: item[0].stable_key(),
    ))


def candidate_causal_protocol(
        candidate: W06RelationCandidate,
        ) -> CausalEndpointProtocol:
    """从 authored causal protocol 恢复不可替换的 relation/Role/指令。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R07_SUBSTAGE
            or candidate.relation_family != W06_R07_SUBSTAGE):
        raise W06R07ContractError("causal protocol candidate 不属于 R07")
    envelope = candidate.domain_protocols.to_value()
    if not isinstance(envelope, dict) or set(envelope) != {"causal_protocol"}:
        raise W06R07ContractError("R07 domain protocol 信封漂移")
    value = envelope["causal_protocol"]
    if not isinstance(value, dict) or set(value) != _PROTOCOL_FIELDS:
        raise W06R07ContractError("R07 causal protocol 字段漂移")
    protocol = CausalEndpointProtocol(
        ObjectIdentity.from_stable_key(_strict_int_key(
            value["relation_key"], where="causal relation_key")),
        ObjectIdentity.from_stable_key(_strict_int_key(
            value["cause_role_key"], where="cause_role_key")),
        ObjectIdentity.from_stable_key(_strict_int_key(
            value["effect_role_key"], where="effect_role_key")),
        ObjectIdentity.from_stable_key(_strict_int_key(
            value["execution_instruction_key"],
            where="execution_instruction_key")),
    )
    if protocol.relation != candidate.proposition.predicate:
        raise W06R07ContractError("R07 causal relation 漂移")
    if tuple(_strict_int_key(value["scope_key"], where="causal scope_key")) != (
            document_scope(candidate.source_ref).stable_key()):
        raise W06R07ContractError("R07 causal scope 漂移")
    required = {
        "independent_witness_required": 1,
        "causal_implies_event_time_fact": 0,
        "counterfactual_verdict_claimed": 0,
        "forming_source_reusable_as_witness": 0,
        "occurrence_order_consumed": 0,
        "precedence_implies_causation": 0,
        "structure_order_consumed": 0,
        "temporal_support_sufficient": 0,
    }
    if any(type(value[name]) is not int or value[name] != expected
           for name, expected in required.items()):
        raise W06R07ContractError("R07 causal 隔离边界漂移")
    for name in ("dimension_key", "evidence_target_kind_key", "verifier_key"):
        _strict_int_key(value[name], where=name)
    causal_endpoints(candidate.proposition, protocol)
    return protocol


def candidate_endpoints(
        candidate: W06RelationCandidate,
        ) -> tuple[ObjectIdentity, ObjectIdentity]:
    return causal_endpoints(candidate.proposition, candidate_causal_protocol(candidate))


def w06_r07_language_branch(candidate: W06RelationCandidate) -> ObjectIdentity:
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R07_SUBSTAGE):
        raise W06R07ContractError("language branch candidate 不属于 R07")
    value = candidate.observation.language
    return language_branch_identity(
        (W06_NAMESPACE, 971, len(value), *(ord(item) for item in value)),
        versions=W06_IDENTITY_VERSIONS,
    )


def candidate_construction(candidate: W06RelationCandidate) -> ObjectIdentity:
    key = candidate.observation.template_group_key.components
    return structure_concept_identity(
        (W06_NAMESPACE, 972, len(key), *key),
        versions=W06_IDENTITY_VERSIONS,
    )


def slice_w06_r07_adapter(
        adapter: W06TypedAdapterOutput,
        ) -> W06TypedAdapterOutput:
    """只保留 CAUSES/train 的十个合法候选。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("R07 slice 需要 W06TypedAdapterOutput")
    candidates = adapter.candidates_for_substage(W06_R07_SUBSTAGE)
    if len(candidates) != 10:
        raise W06R07ContractError("R07 train candidate inventory 漂移")
    if {item.relation_family for item in candidates} != {W06_R07_SUBSTAGE}:
        raise W06R07ContractError("R07 train relation family 漂移")
    candidate_ids = {item.proposition.proposition for item in candidates}
    sources = {item.source_record.stable_key for item in candidates}
    rejections = tuple(
        item for item in adapter.rejections
        if item.substage_key == W06_R07_SUBSTAGE)
    if rejections:
        raise W06R07ContractError("R07 不应含 schema rejection")
    schemas = {item.schema.schema: item.schema for item in candidates}
    if len(schemas) != 1:
        raise W06R07ContractError("R07 protocol schema 必须恰有一类")
    sliced = W06TypedAdapterOutput(
        tuple(item for item in adapter.source_bindings
              if item.record.stable_key in sources),
        tuple(item for item in adapter.observations
              if item.candidate.proposition.proposition in candidate_ids),
        candidates,
        tuple(schemas.values()),
        tuple(item for item in adapter.evidence
              if item.candidate in candidate_ids),
        (),
        (),
        adapter.execution_state,
    )
    for candidate in sliced.candidates:
        candidate_causal_protocol(candidate)
    return sliced


class W06R07View:
    """共享唯一 W06 truth owner 的只读 direct CAUSES facade。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            protocol: W06R07ConsumerProtocol,
            endpoint_resolver,
            ) -> None:
        if not isinstance(learning, W06RelationLearningRuntime):
            raise TypeError("R07 learning 类型非法")
        if not isinstance(adapter, W06TypedAdapterOutput):
            raise TypeError("R07 adapter 类型非法")
        if not isinstance(protocol, W06R07ConsumerProtocol):
            raise TypeError("R07 consumer protocol 类型非法")
        if (not callable(getattr(endpoint_resolver, "resolve", None))
                or not callable(getattr(endpoint_resolver, "state_key", None))):
            raise TypeError("R07 endpoint resolver 协议非法")
        candidates = adapter.candidates_for_substage(W06_R07_SUBSTAGE)
        registered = learning.registered_candidates()
        if ({item.proposition.proposition for item in registered}
                != {item.proposition.proposition for item in candidates}):
            raise W06R07ContractError(
                "R07 runtime 必须绑定隔离的 CAUSES learning owner")
        if learning.closure is None:
            raise W06R07ContractError("R07 learning 缺少 R-00 closure")
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.endpoint_resolver = endpoint_resolver
        self.candidates = candidates
        self.candidate_by_id = {
            item.proposition.proposition: item for item in candidates
        }
        self.endpoint_protocols = {
            item.proposition.proposition: candidate_causal_protocol(item)
            for item in candidates
        }

    def endpoints_for(
            self, candidate: W06RelationCandidate,
            ) -> tuple[ObjectIdentity, ObjectIdentity]:
        return tuple(
            self.endpoint_resolver.resolve(item)
            for item in candidate_endpoints(candidate)
        )

    def witness_accounts(
            self, candidate: W06RelationCandidate,
            ) -> tuple[W06R07WitnessAccount, ...]:
        """只投影仍存在于 current snapshot 的 teacher Evidence。"""
        snapshot = self.learning.snapshot_for(candidate.proposition.proposition)
        current = set(snapshot.evidence)
        result = []
        for application in self.learning.applications():
            for account in application.accounts:
                if (account.candidate != candidate.proposition.proposition
                        or account.derived_supersede
                        or account.trace.outcome.evidence not in current):
                    continue
                recognition = account.trace.input
                witness = CausalIndependentWitness(
                    account.stance,
                    recognition.revealed.verifier_source,
                    recognition.visible_inputs,
                    account.event_key,
                )
                result.append(W06R07WitnessAccount(
                    candidate.proposition.proposition,
                    witness,
                    candidate.spec.forming_sources,
                    candidate.source_ref,
                ))
        return tuple(sorted(result, key=W06R07WitnessAccount.stable_key))

    def evaluate(
            self, query: W06R07CausalQuery,
            ) -> W06R07CausalEvaluation:
        if query.budget.max_candidates < len(self.candidates):
            raise W06R07BudgetExceeded("R07 candidate scan budget 耗尽")
        matched = []
        snapshots = []
        witnesses = []
        for candidate in self.candidates:
            if self.endpoints_for(candidate) != (query.cause, query.effect):
                continue
            snapshot = self.learning.snapshot_for(
                candidate.proposition.proposition)
            if snapshot.snapshot.lifecycle == LIFECYCLE_SUPERSEDED:
                continue
            matched.append(candidate)
            snapshots.append(snapshot)
            witnesses.extend(self.witness_accounts(candidate))
        evidence_count = sum(len(item.evidence) for item in snapshots)
        if evidence_count > query.budget.max_evidence:
            raise W06R07BudgetExceeded("R07 Evidence scan budget 耗尽")
        witness_input_count = sum(
            len(item.witness.input_objects) for item in witnesses)
        if witness_input_count > query.budget.max_witness_inputs:
            raise W06R07BudgetExceeded("R07 witness input scan budget 耗尽")
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
        return W06R07CausalEvaluation(
            query,
            LogicEvidenceState(support, refute),
            active,
            propositions,
            evidence,
            tuple(sorted(witnesses, key=W06R07WitnessAccount.stable_key)),
            True,
            False,
        )

    @staticmethod
    def _use_context(
            query: W06R07CausalQuery,
            consumer: str,
            ) -> RelationUseContext:
        if consumer not in W06_R07_CONSUMERS:
            raise W06R07ContractError("R07 Use consumer 未注册")
        ordinal = W06_R07_CONSUMERS.index(consumer) + 1
        return RelationUseContext(
            query.source,
            document_scope(query.source),
            concept_identity((
                W06_R07_RUNTIME_NAMESPACE, 80, ordinal,
                *pack_key(query.request_key.components),
            )),
            concept_identity((
                W06_R07_RUNTIME_NAMESPACE, 81, ordinal,
                *pack_key(query.request_key.components),
            )),
        )

    def commit(
            self,
            query: W06R07CausalQuery,
            use_key: LosslessIntegerKey,
            consumer: str,
            ):
        evaluation = self.evaluate(query)
        if not evaluation.active_propositions:
            raise W06R07ContractError("R07 query 没有 current active premise")
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

    def witness_keys(
            self, candidate: W06RelationCandidate,
            ) -> tuple[tuple[int, ...], ...]:
        return tuple(
            item.stable_key() for item in self.witness_accounts(candidate))

    def authorization_key(
            self, candidate: W06RelationCandidate,
            ) -> LosslessIntegerKey | None:
        snapshot = self.learning.snapshot_for(candidate.proposition.proposition)
        fact = snapshot.active_fact
        witnesses = self.witness_keys(candidate)
        if fact is None or not witnesses:
            return None
        values = [
            W06_R07_RUNTIME_NAMESPACE,
            700,
            *pack_key(fact.proposition.proposition.stable_key()),
            *pack_key(fact.hypothesis.stable_key()),
            len(fact.evidence_keys),
        ]
        for item in fact.evidence_keys:
            values.extend(pack_key(item))
        values.extend(pack_key(fact.decision_key))
        values.append(len(witnesses))
        for item in witnesses:
            values.extend(pack_key(item))
        return LosslessIntegerKey(tuple(values))

    def consume_candidate(
            self,
            candidate: W06RelationCandidate,
            use_key: LosslessIntegerKey,
            ):
        if candidate not in self.candidates:
            raise W06R07ContractError("generation target 不属于 R07 view")
        if self.authorization_key(candidate) is None:
            raise W06R07ContractError("generation target 已失去 current authorization")
        assert self.learning.closure is not None
        context = RelationUseContext(
            candidate.source_ref,
            document_scope(candidate.source_ref),
            concept_identity((W06_R07_RUNTIME_NAMESPACE, 82, 3)),
            concept_identity((
                W06_R07_RUNTIME_NAMESPACE, 83,
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
                 self.endpoint_protocols[
                     item.proposition.proposition].execution_instruction.stable_key())
                for item in self.candidates
            ),
        )


__all__ = [
    "W06R07View",
    "candidate_causal_protocol",
    "candidate_construction",
    "candidate_endpoints",
    "candidate_role_fillers",
    "slice_w06_r07_adapter",
    "w06_r07_language_branch",
]
