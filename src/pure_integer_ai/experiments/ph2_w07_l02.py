"""W07-L02 AND_OR 的薄 facade 与 public bounded report。"""
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


W07_L02_PREFIX = ("NOT", "AND_OR")
W07_L02_SUBSTAGE = "AND_OR"


@dataclass(frozen=True)
class W07L02RuntimeReport:
    operator_digest: tuple[int, ...]
    source_evidence_digest: tuple[int, ...]
    execution_digest: tuple[int, ...]
    candidate_count: int
    active_candidate_count: int
    operator_profile_count: int
    executable_proposal_count: int
    and_execution_count: int
    or_execution_count: int
    supported_count: int
    refuted_count: int
    unknown_count: int
    conflict_count: int
    understanding_use_count: int
    reasoning_use_count: int
    generation_use_count: int
    generation_outcome_count: int
    algebraic_rewrite_count: int = 0
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
            or proposal.observation.substage != W07_L02_SUBSTAGE):
        raise TypeError("W07-L02 request 只接受 AND_OR proposal")
    return W07LogicRequest(
        request_key,
        W07_L02_SUBSTAGE,
        proposal.bound_root.template,
        proposal.source_binding.source_ref,
        proposal.request_scope,
    )


class W07L02Runtime:
    """共享 prefix learning owner，分别暴露 U/R/G consumer。"""

    def __init__(
            self,
            learning: W07LogicLearningRuntime,
            adapter: W07TypedAdapterOutput,
            *,
            protocol: W07LogicConsumerProtocol = (
                W07LogicConsumerProtocol(W07_L02_PREFIX)),
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
            if item.observation.substage == W07_L02_SUBSTAGE)

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

    def report(self) -> W07L02RuntimeReport:
        specs = tuple(
            spec for proposal in self.proposals for spec in proposal.specs)
        active = tuple(
            spec for spec in specs
            if self.learning.logic.adoption(spec) is not None)
        executable = self.view.executable_proposals(W07_L02_SUBSTAGE)
        executions = tuple(
            self.view.execute(logic_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((70702, index)),
            ))
            for index, proposal in enumerate(executable, start=1)
        )
        executions = tuple(item for item in executions if item is not None)
        states = tuple(item.evaluation.state.stable_key() for item in executions)
        families = tuple(
            proposal.operator_families[0] for proposal in executable)
        return W07L02RuntimeReport(
            digest_value([{
                "definition": list(item.definition.stable_key()),
                "family": proposal.operator_families[0],
                "structure_tree": list(structure_tree_key(proposal.bound_root)),
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
            len(specs),
            len(active),
            len({item.definition.stable_key() for item in active}),
            len(executable),
            families.count("AND"),
            families.count("OR"),
            states.count((1, 0)),
            states.count((0, 1)),
            states.count((0, 0)),
            states.count((1, 1)),
            len(self.understanding.uses),
            len(self.reasoning.uses),
            len(self.generation.uses),
            len(self.generation.outcomes),
        )


def build_w07_l02_runtime(
        backend,
        adapter: W07TypedAdapterOutput,
        *,
        protocol: W07LogicConsumerProtocol = (
            W07LogicConsumerProtocol(W07_L02_PREFIX)),
        ) -> W07L02Runtime:
    sliced = slice_w07_adapter(adapter, W07_L02_PREFIX)
    learning = build_w07_learning_runtime(backend, sliced)
    return W07L02Runtime(learning, sliced, protocol=protocol)


__all__ = [
    "W07L02Runtime",
    "W07L02RuntimeReport",
    "W07_L02_PREFIX",
    "W07_L02_SUBSTAGE",
    "build_w07_l02_runtime",
    "generation_request_for_proposal",
    "logic_request_for_proposal",
]
