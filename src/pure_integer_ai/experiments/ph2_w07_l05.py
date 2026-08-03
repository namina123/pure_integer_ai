"""W07-L05 FORALL 的 typed facade 与 bounded public report。"""
from __future__ import annotations

from dataclasses import dataclass

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
)


W07_L05_PREFIX = ("NOT", "AND_OR", "CONDITION", "EXISTS", "FORALL")
W07_L05_SUBSTAGE = "FORALL"


@dataclass(frozen=True)
class W07L05RuntimeReport:
    """FORALL 的 operator/content 分账、分支和 consumer 摘要。"""

    operator_digest: tuple[int, ...]
    source_evidence_digest: tuple[int, ...]
    execution_digest: tuple[int, ...]
    candidate_count: int
    active_candidate_count: int
    operator_profile_count: int
    executable_proposal_count: int
    branch_count: int
    counterexample_count: int
    closed_domain_count: int
    open_domain_count: int
    empty_domain_count: int
    supported_count: int
    refuted_count: int
    unknown_count: int
    conflict_count: int
    understanding_use_count: int
    reasoning_use_count: int
    generation_use_count: int
    generation_outcome_count: int
    operator_evidence_count: int = 0
    content_evidence_count: int = 0
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
    """从来源化 FORALL proposal 构造不携 expected state 的执行请求。"""
    if (not isinstance(proposal, W07LogicProposal)
            or proposal.observation.substage != W07_L05_SUBSTAGE):
        raise TypeError("W07-L05 request 只接受 FORALL proposal")
    return W07LogicRequest(
        request_key,
        W07_L05_SUBSTAGE,
        proposal.bound_root.template,
        proposal.source_binding.source_ref,
        proposal.request_scope,
    )


def _quantifier(proposal: W07LogicProposal):
    if len(proposal.quantifiers) != 1:
        raise W07LogicContractError(
            "FORALL proposal 必须包含唯一 Binder/Variable/domain")
    quantifier = proposal.quantifiers[0]
    if quantifier.definition.body_slot != proposal.specs[0].definition.slots[0]:
        raise W07LogicContractError("FORALL body Role 与 operator slot 漂移")
    return quantifier


class W07L05Runtime:
    """共享 prefix learning owner，显式暴露 FORALL 分支和 U/R/G。"""

    def __init__(
            self,
            learning: W07LogicLearningRuntime,
            adapter: W07TypedAdapterOutput,
            *,
            protocol: W07LogicConsumerProtocol = (
                W07LogicConsumerProtocol(W07_L05_PREFIX)),
            ) -> None:
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.view = W07LogicView(learning, adapter, protocol)
        self.understanding = W07LogicUnderstandingRuntime(self.view)
        self.reasoning = W07LogicReasoningRuntime(self.view)
        self.generation = W07LogicGenerationRuntime(self.view)

    @property
    def proposals(self) -> tuple[W07LogicProposal, ...]:
        return tuple(
            item for item in self.adapter.proposals
            if item.observation.substage == W07_L05_SUBSTAGE)

    def quantifier_for(self, proposal: W07LogicProposal):
        """返回 proposal 的完整量词定义，拒绝模糊或错向绑定。"""
        if proposal not in self.proposals:
            raise W07LogicContractError("FORALL quantifier proposal 不属于当前 runtime")
        return _quantifier(proposal)

    def execute(self, request: W07LogicRequest) -> W07LogicExecution | None:
        """执行一次真实 S-04 FORALL，并保留完整 branch trace。"""
        if request.substage != W07_L05_SUBSTAGE:
            raise TypeError("W07-L05 execute 只接受 FORALL request")
        return self.view.execute(request)

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

    def state_key(self) -> tuple:
        return (
            self.learning.logic.state_key(),
            self.protocol.stable_key(),
            self.understanding.state_key(),
            self.reasoning.state_key(),
            self.generation.state_key(),
        )

    def report(self) -> W07L05RuntimeReport:
        specs = tuple(
            spec for proposal in self.proposals for spec in proposal.specs)
        active = tuple(
            spec for spec in specs
            if self.learning.logic.adoption(spec) is not None)
        executable = self.view.executable_proposals(W07_L05_SUBSTAGE)
        executions = tuple(
            self.execute(logic_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((70705, index)),
            ))
            for index, proposal in enumerate(executable, start=1)
        )
        executions = tuple(item for item in executions if item is not None)
        states = tuple(item.evaluation.state.stable_key() for item in executions)
        quantifiers = tuple(_quantifier(item) for item in self.proposals)
        branch_count = sum(len(item.evaluation.branches) for item in executions)
        counterexample_count = sum(
            any(branch.state.refute for branch in item.evaluation.branches)
            for item in executions)
        operator_payload = [{
            "definition": list(item.definition.stable_key()),
            "family": proposal.operator_families[0],
            "structure_tree": list(structure_tree_key(proposal.bound_root)),
            "role_tree": list(role_tree_key(
                proposal.bound_root, include_bound_provenance=True)),
            "binder": list(_quantifier(proposal).definition.binder.stable_key()),
            "variable": list(_quantifier(proposal).definition.variable.stable_key()),
            "domain": list(_quantifier(proposal).definition.domain.stable_key()),
            "body_role": list(_quantifier(proposal).value_role.stable_key()),
        } for proposal in self.proposals for item in proposal.specs]
        source_payload = [{
            "candidate": list(spec.candidate.stable_key()),
            "operator_evidence": [list(value.stable_key())
                                  for value in adoption.evidence],
            "domain_values": [list(value.value.stable_key())
                              for value in _quantifier(binding.proposal).value_evidence],
            "domain_states": [value.state.stable_key()
                              for value in _quantifier(binding.proposal).value_evidence],
        } for binding in self.adapter.evidence
            if binding.proposal.observation.substage == W07_L05_SUBSTAGE
            for spec in binding.proposal.specs
            if (adoption := self.learning.logic.adoption(spec)) is not None]
        return W07L05RuntimeReport(
            digest_value(operator_payload),
            digest_value(source_payload),
            digest_value([list(item.stable_key()) for item in executions]),
            len(specs),
            len(active),
            len({item.definition.stable_key() for item in active}),
            len(executable),
            branch_count,
            counterexample_count,
            sum(item.definition.domain.closed for item in quantifiers),
            sum(not item.definition.domain.closed for item in quantifiers),
            sum(not item.definition.domain.values for item in quantifiers),
            states.count((1, 0)),
            states.count((0, 1)),
            states.count((0, 0)),
            states.count((1, 1)),
            len(self.understanding.uses),
            len(self.reasoning.uses),
            len(self.generation.uses),
            len(self.generation.outcomes),
            sum(len(item.operator_premise_keys) for item in executions),
            sum(len(item.content_premise_keys) for item in executions),
        )


def build_w07_l05_runtime(
        backend,
        adapter: W07TypedAdapterOutput,
        *,
        protocol: W07LogicConsumerProtocol = (
            W07LogicConsumerProtocol(W07_L05_PREFIX)),
        ) -> W07L05Runtime:
    """构建严格 L01..L05 prefix 的共享学习/执行 owner。"""
    sliced = slice_w07_adapter(adapter, W07_L05_PREFIX)
    learning = build_w07_learning_runtime(backend, sliced)
    return W07L05Runtime(learning, sliced, protocol=protocol)


__all__ = [
    "W07L05Runtime",
    "W07L05RuntimeReport",
    "W07_L05_PREFIX",
    "W07_L05_SUBSTAGE",
    "build_w07_l05_runtime",
    "generation_request_for_proposal",
    "logic_request_for_proposal",
]
