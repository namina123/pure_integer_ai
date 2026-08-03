"""W07-L03 CONDITION 的薄 facade、P2-G certificate 与 bounded report。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.logic_executor import (
    STATE_PROVISIONAL,
    LogicEvaluation,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w07_adapter import (
    W07LogicProposal,
    W07TypedAdapterOutput,
)
from pure_integer_ai.experiments.ph2_w07_learning import (
    W07LogicLearningRuntime,
    build_w07_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w07_logic_consumer import (
    W07LogicReasoningRuntime,
    W07LogicUnderstandingRuntime,
)
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicConsumerProtocol,
    W07LogicContractError,
    W07LogicExecution,
    W07LogicRequest,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w07_logic_generation import (
    W07LogicGenerationRuntime,
    generation_request_for_proposal,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    W07LogicView,
    role_tree_key,
    slice_w07_adapter,
    structure_tree_key,
    w07_logic_language_branch,
)
from pure_integer_ai.experiments.typed_proof_family_contracts import (
    CONDITION_ASSERTION,
    CONDITION_MATERIAL,
    CONDITION_NECESSARY,
    CONDITION_SUFFICIENT,
    PROOF_FAMILY_CONDITION,
    ConditionProofCertificate,
    ProofWorkBudget,
)
from pure_integer_ai.experiments.typed_proof_family_runtime import (
    ProofDispatchReceipt,
    TypedProofFamilyDispatcher,
)


W07_L03_PREFIX = ("NOT", "AND_OR", "CONDITION")
W07_L03_SUBSTAGE = "CONDITION"
W07_L03_PROOF_FAMILY = PROOF_FAMILY_CONDITION
W07_L03_PROOF_BUDGET = 128


@dataclass(frozen=True)
class W07ConditionProof:
    """一个有序 CONDITION certificate、S-04 execution 和 checker receipt。"""

    kind: int
    execution: W07LogicExecution
    certificate: ConditionProofCertificate
    receipt: ProofDispatchReceipt
    operator_evidence_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in {
                CONDITION_MATERIAL, CONDITION_SUFFICIENT,
                CONDITION_NECESSARY}:
            raise W07LogicContractError("CONDITION proof kind 未注册")
        if not isinstance(self.execution, W07LogicExecution):
            raise TypeError("CONDITION proof execution 类型非法")
        if not isinstance(self.certificate, ConditionProofCertificate):
            raise TypeError("CONDITION certificate 类型非法")
        if not isinstance(self.receipt, ProofDispatchReceipt):
            raise TypeError("CONDITION proof receipt 类型非法")
        if (self.certificate.declared_family != W07_L03_PROOF_FAMILY
                or self.certificate.condition_kind != self.kind
                or self.certificate.inference_kind != CONDITION_ASSERTION):
            raise W07LogicContractError("CONDITION certificate family/kind 漂移")
        if (not isinstance(self.operator_evidence_ids, tuple)
                or not self.operator_evidence_ids
                or any(type(item) is not int or item <= 0
                       for item in self.operator_evidence_ids)
                or tuple(sorted(set(self.operator_evidence_ids)))
                != self.operator_evidence_ids):
            raise W07LogicContractError("CONDITION operator Evidence id 非法")
        if self.operator_evidence_ids != _operator_evidence_ids(self.execution):
            raise W07LogicContractError(
                "CONDITION proof 未绑定 execution 的 active Evidence")
        expected = (
            () if self.kind == CONDITION_MATERIAL
            else self.operator_evidence_ids)
        if self.certificate.kind_evidence_ids != expected:
            raise W07LogicContractError(
                "CONDITION certificate kind Evidence 未绑定当前 adoption")
        if (self.receipt.result.certificate_key
                != self.certificate.stable_key()):
            raise W07LogicContractError("CONDITION receipt/certificate 漂移")
        if (self.certificate.evaluation != self.execution.evaluation
                or self.certificate.structure
                not in self.execution.executed_structures
                or self.receipt.result.family != W07_L03_PROOF_FAMILY
                or self.receipt.result.work_units > self.receipt.budget.used):
            raise W07LogicContractError(
                "CONDITION proof execution/family/budget 归因漂移")

    @property
    def provisional_content_evidence(self) -> bool:
        """S-04 求值是 provisional 时才可作为当前 scope 内容结果。"""
        return self.execution.evaluation.status == STATE_PROVISIONAL

    def stable_key(self) -> tuple[int, ...]:
        return (
            self.kind,
            *pack_key(self.execution.stable_key()),
            *pack_key(self.certificate.stable_key()),
            *pack_key(self.receipt.result.stable_key()),
            self.receipt.budget.limit,
            self.receipt.budget.used,
        )


@dataclass(frozen=True)
class W07L03RuntimeReport:
    operator_digest: tuple[int, ...]
    source_evidence_digest: tuple[int, ...]
    execution_digest: tuple[int, ...]
    certificate_digest: tuple[int, ...]
    candidate_count: int
    active_candidate_count: int
    operator_profile_count: int
    executable_proposal_count: int
    supported_count: int
    refuted_count: int
    unknown_count: int
    conflict_count: int
    material_certificate_count: int
    sufficient_certificate_count: int
    necessary_certificate_count: int
    provisional_content_evidence_count: int
    understanding_use_count: int
    reasoning_use_count: int
    generation_use_count: int
    generation_outcome_count: int
    modus_ponens_count: int = 0
    global_truth_count: int = 0
    causal_fact_count: int = 0
    temporal_fact_count: int = 0
    action_fact_count: int = 0
    private_read_count: int = 0
    formal_guard_read_count: int = 0
    future_substage_claim_count: int = 0
    w07_started: int = 0


def logic_request_for_proposal(
        proposal: W07LogicProposal,
        *,
        request_key: LosslessIntegerKey,
        ) -> W07LogicRequest:
    if (not isinstance(proposal, W07LogicProposal)
            or proposal.observation.substage != W07_L03_SUBSTAGE):
        raise TypeError("W07-L03 request 只接受 CONDITION proposal")
    return W07LogicRequest(
        request_key,
        W07_L03_SUBSTAGE,
        proposal.bound_root.template,
        proposal.source_binding.source_ref,
        proposal.request_scope,
    )


def _operator_evidence_ids(
        execution: W07LogicExecution,
        ) -> tuple[int, ...]:
    values = tuple(sorted({
        evidence.evidence_id
        for adoption in execution.operator_adoptions
        for evidence in adoption.evidence
    }))
    if not values:
        raise W07LogicContractError("CONDITION 缺 operator adoption Evidence")
    return values


class W07L03Runtime:
    """共享 prefix owner，显式暴露 CONDITION certificate/U/R/G。"""

    def __init__(
            self,
            learning: W07LogicLearningRuntime,
            adapter: W07TypedAdapterOutput,
            *,
            protocol: W07LogicConsumerProtocol = (
                W07LogicConsumerProtocol(W07_L03_PREFIX)),
            ) -> None:
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.view = W07LogicView(learning, adapter, protocol)
        self.understanding = W07LogicUnderstandingRuntime(self.view)
        self.reasoning = W07LogicReasoningRuntime(self.view)
        self.generation = W07LogicGenerationRuntime(self.view)
        self._proofs: list[W07ConditionProof] = []
        self._proof_keys: set[tuple[int, ...]] = set()

    @property
    def proposals(self) -> tuple[W07LogicProposal, ...]:
        return tuple(
            item for item in self.adapter.proposals
            if item.observation.substage == W07_L03_SUBSTAGE)

    def resolve_understanding(self, request):
        return self.understanding.resolve(request)

    def adopt_understanding(self, resolution):
        return self.understanding.adopt(resolution)

    def verify_understanding(self, use):
        return self.understanding.verify(use)

    def resolve_reasoning(self, request):
        return self.reasoning.resolve(request)

    def adopt_reasoning(self, resolution):
        return self.reasoning.adopt(resolution)

    def verify_reasoning(self, use):
        return self.reasoning.verify(use)

    def choose_generation(self, request):
        return self.generation.choose(request)

    def adopt_generation(self, choice, option_key):
        return self.generation.adopt(choice, option_key)

    def verify_generation(self, use):
        return self.generation.verify(use)

    def _parts(
            self,
            request: W07LogicRequest,
            ) -> tuple[W07LogicExecution, LogicEvaluation, LogicEvaluation,
                       tuple[int, ...]] | None:
        proposal = self.view.proposal_for(request)
        execution = self.view.execute(request)
        if proposal is None or execution is None:
            return None
        adoptions = execution.operator_adoptions
        definitions = {item.spec.definition.stable_key()
                       for item in adoptions}
        if len(definitions) != 1 or len(adoptions[0].spec.definition.slots) != 2:
            raise W07LogicContractError(
                "CONDITION active definition 必须是唯一二元 profile")
        definition = adoptions[0].spec.definition
        by_slot = {
            (item.role, item.ordinal): item.filler
            for item in proposal.bound_root.bindings
        }
        try:
            first = by_slot[(definition.slots[0].role,
                             definition.slots[0].ordinal)]
            second = by_slot[(definition.slots[1].role,
                              definition.slots[1].ordinal)]
        except KeyError as error:
            raise W07LogicContractError(
                "CONDITION antecedent/consequent Role slot 缺失") from error
        if (not isinstance(first, BoundProposition)
                or not isinstance(second, BoundProposition)):
            raise W07LogicContractError(
                "CONDITION antecedent/consequent 必须是 bound proposition")
        condition = self.view.evaluate_bound(request, first)
        conditioned = self.view.evaluate_bound(request, second)
        if condition is None or conditioned is None:
            return None
        return (
            execution,
            condition,
            conditioned,
            _operator_evidence_ids(execution),
        )

    def certificate_for(
            self,
            request: W07LogicRequest,
            condition_kind: int,
            *,
            kind_evidence_ids: tuple[int, ...] | None = None,
            ) -> ConditionProofCertificate | None:
        """从真实根/child execution 形成有序 P2-G certificate。"""
        if request.substage != W07_L03_SUBSTAGE:
            raise TypeError("W07-L03 certificate 只接受 CONDITION request")
        parts = self._parts(request)
        if parts is None:
            return None
        return self._certificate_from_parts(
            parts,
            condition_kind,
            kind_evidence_ids=kind_evidence_ids,
        )

    @staticmethod
    def _certificate_from_parts(
            parts: tuple[
                W07LogicExecution, LogicEvaluation, LogicEvaluation,
                tuple[int, ...]],
            condition_kind: int,
            *,
            kind_evidence_ids: tuple[int, ...] | None = None,
            ) -> ConditionProofCertificate:
        execution, condition, conditioned, evidence_ids = parts
        if condition_kind == CONDITION_NECESSARY:
            cert_condition, cert_conditioned = conditioned, condition
        elif condition_kind in {CONDITION_MATERIAL, CONDITION_SUFFICIENT}:
            cert_condition, cert_conditioned = condition, conditioned
        else:
            raise W07LogicContractError("CONDITION kind 未注册")
        if kind_evidence_ids is None:
            kind_evidence_ids = (
                () if condition_kind == CONDITION_MATERIAL else evidence_ids)
        return ConditionProofCertificate(
            W07_L03_PROOF_FAMILY,
            condition_kind,
            execution.operator_adoptions[0].spec.definition.structure,
            cert_condition,
            cert_conditioned,
            execution.evaluation,
            CONDITION_ASSERTION,
            kind_evidence_ids,
        )

    def prove(
            self,
            request: W07LogicRequest,
            condition_kind: int,
            *,
            budget: ProofWorkBudget = ProofWorkBudget(W07_L03_PROOF_BUDGET),
            ) -> W07ConditionProof | None:
        if request.substage != W07_L03_SUBSTAGE:
            raise TypeError("W07-L03 proof 只接受 CONDITION request")
        parts = self._parts(request)
        if parts is None:
            return None
        execution = parts[0]
        cert = self._certificate_from_parts(parts, condition_kind)
        key = (
            *request.stable_key(), condition_kind, CONDITION_ASSERTION)
        if key in self._proof_keys:
            raise W07LogicContractError("CONDITION proof request 重放")
        receipt = TypedProofFamilyDispatcher().check(cert, budget)
        proof = W07ConditionProof(
            condition_kind,
            execution,
            cert,
            receipt,
            _operator_evidence_ids(execution),
        )
        self._proof_keys.add(key)
        self._proofs.append(proof)
        return proof

    def state_key(self) -> tuple:
        return (
            self.learning.logic.state_key(),
            self.protocol.stable_key(),
            self.understanding.state_key(),
            self.reasoning.state_key(),
            self.generation.state_key(),
            tuple(item.stable_key() for item in self._proofs),
        )

    def report(self) -> W07L03RuntimeReport:
        specs = tuple(
            spec for proposal in self.proposals for spec in proposal.specs)
        active = tuple(
            spec for spec in specs
            if self.learning.logic.adoption(spec) is not None)
        executable = self.view.executable_proposals(W07_L03_SUBSTAGE)
        executions = tuple(
            self.view.execute(logic_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((70703, index)),
            ))
            for index, proposal in enumerate(executable, start=1)
        )
        executions = tuple(item for item in executions if item is not None)
        states = tuple(item.evaluation.state.stable_key() for item in executions)
        cert_payload = []
        for index, proposal in enumerate(executable, start=1):
            request = logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((70703, 100 + index)))
            parts = self._parts(request)
            assert parts is not None
            for kind in (CONDITION_MATERIAL, CONDITION_SUFFICIENT,
                         CONDITION_NECESSARY):
                cert = self._certificate_from_parts(parts, kind)
                receipt = TypedProofFamilyDispatcher().check(
                    cert, ProofWorkBudget(W07_L03_PROOF_BUDGET))
                cert_payload.append({
                    "kind": kind,
                    "certificate": list(cert.stable_key()),
                    "result": list(receipt.result.stable_key()),
                })
        return W07L03RuntimeReport(
            digest_value([{
                "definition": list(item.definition.stable_key()),
                "family": proposal.operator_families[0],
                "structure_tree": list(structure_tree_key(
                    proposal.bound_root)),
                "role_tree": list(role_tree_key(
                    proposal.bound_root, include_bound_provenance=True)),
            } for proposal in self.proposals for item in proposal.specs]),
            digest_value([{
                "candidate": list(item.spec.candidate.stable_key()),
                "evidence": [list(value.stable_key())
                             for value in item.evidence],
            } for item in (
                self.learning.logic.adoption(spec) for spec in specs)
                if item is not None]),
            digest_value([list(item.stable_key()) for item in executions]),
            digest_value(cert_payload),
            len(specs),
            len(active),
            len({item.definition.stable_key() for item in active}),
            len(executable),
            states.count((1, 0)),
            states.count((0, 1)),
            states.count((0, 0)),
            states.count((1, 1)),
            len(executable),
            len(executable),
            len(executable),
            states.count((1, 0)),
            len(self.understanding.uses),
            len(self.reasoning.uses),
            len(self.generation.uses),
            len(self.generation.outcomes),
        )


def build_w07_l03_runtime(
        backend,
        adapter: W07TypedAdapterOutput,
        *,
        protocol: W07LogicConsumerProtocol = (
            W07LogicConsumerProtocol(W07_L03_PREFIX)),
        ) -> W07L03Runtime:
    sliced = slice_w07_adapter(adapter, W07_L03_PREFIX)
    learning = build_w07_learning_runtime(backend, sliced)
    return W07L03Runtime(learning, sliced, protocol=protocol)


__all__ = [
    "W07ConditionProof",
    "W07L03Runtime",
    "W07L03RuntimeReport",
    "W07_L03_PREFIX",
    "W07_L03_PROOF_BUDGET",
    "W07_L03_SUBSTAGE",
    "build_w07_l03_runtime",
    "generation_request_for_proposal",
    "logic_request_for_proposal",
]
