"""W07-L06 MODAL 的 typed facade 与 bounded public report。"""
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


W07_L06_PREFIX = ("NOT", "AND_OR", "CONDITION", "EXISTS", "FORALL", "MODAL")
W07_L06_SUBSTAGE = "MODAL"


@dataclass(frozen=True)
class W07L06RuntimeReport:
    """MODAL 的 resolver/content/operator 分账和 scope 摘要。"""

    operator_digest: tuple[int, ...]
    source_evidence_digest: tuple[int, ...]
    execution_digest: tuple[int, ...]
    candidate_count: int
    active_candidate_count: int
    operator_profile_count: int
    executable_proposal_count: int
    resolved_count: int
    unresolved_count: int
    output_scope_count: int
    scope_shift_count: int
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
    reality_fact_count: int = 0
    private_read_count: int = 0
    formal_guard_read_count: int = 0
    future_substage_claim_count: int = 0
    w07_started: int = 0


def logic_request_for_proposal(
        proposal: W07LogicProposal,
        *,
        request_key: LosslessIntegerKey,
        ) -> W07LogicRequest:
    """从来源化 MODAL proposal 构造不携 expected state 的执行请求。"""
    if (not isinstance(proposal, W07LogicProposal)
            or proposal.observation.substage != W07_L06_SUBSTAGE):
        raise TypeError("W07-L06 request 只接受 MODAL proposal")
    return W07LogicRequest(
        request_key,
        W07_L06_SUBSTAGE,
        proposal.bound_root.template,
        proposal.source_binding.source_ref,
        proposal.request_scope,
    )


def _plan(proposal):
    if len(proposal.modal_plans) != 1:
        raise W07LogicContractError(
            "MODAL proposal 必须包含唯一 typed ModalResolver plan")
    plan = proposal.modal_plans[0]
    if plan.source != proposal.source_binding.source_ref:
        raise W07LogicContractError("MODAL plan/source 漂移")
    if plan.input_scope != proposal.request_scope:
        raise W07LogicContractError("MODAL plan/input scope 漂移")
    return plan


class W07L06Runtime:
    """共享 prefix learning owner，显式暴露 MODAL resolver 和 U/R/G。"""

    def __init__(
            self,
            learning: W07LogicLearningRuntime,
            adapter: W07TypedAdapterOutput,
            *,
            protocol: W07LogicConsumerProtocol = (
                W07LogicConsumerProtocol(W07_L06_PREFIX)),
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
            if item.observation.substage == W07_L06_SUBSTAGE)

    def modal_plan_for(self, proposal: W07LogicProposal):
        """返回 proposal 的 typed resolver plan，不能由 surface/cue 推断。"""
        if proposal not in self.proposals:
            raise W07LogicContractError("MODAL plan proposal 不属于当前 runtime")
        return _plan(proposal)

    def execute(self, request: W07LogicRequest) -> W07LogicExecution | None:
        """执行一次真实 S-04 MODAL，保留 resolver output scope。"""
        if request.substage != W07_L06_SUBSTAGE:
            raise TypeError("W07-L06 execute 只接受 MODAL request")
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

    def report(self) -> W07L06RuntimeReport:
        specs = tuple(
            spec for proposal in self.proposals for spec in proposal.specs)
        active = tuple(
            spec for spec in specs
            if self.learning.logic.adoption(spec) is not None)
        executable = self.view.executable_proposals(W07_L06_SUBSTAGE)
        executions = tuple(
            self.execute(logic_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((70706, index)),
            ))
            for index, proposal in enumerate(executable, start=1)
        )
        executions = tuple(item for item in executions if item is not None)
        states = tuple(item.evaluation.state.stable_key() for item in executions)
        plans = tuple(_plan(item) for item in self.proposals)
        resolved = sum(item.status == "RESOLVED" for item in plans)
        output_scopes = tuple(
            item.output_scope for item in plans if item.output_scope is not None)
        operator_payload = [{
            "definition": list(item.definition.stable_key()),
            "family": proposal.operator_families[0],
            "structure_tree": list(structure_tree_key(proposal.bound_root)),
            "role_tree": list(role_tree_key(
                proposal.bound_root, include_bound_provenance=True)),
            "modal_kind": proposal.observation.typed_payload.to_value().get(
                "operator_kind"),
            "plan": {
                "status": _plan(proposal).status,
                "input_scope": list(_plan(proposal).input_scope.stable_key()),
                "output_scope": (None if _plan(proposal).output_scope is None
                                 else list(_plan(proposal).output_scope.stable_key())),
            },
        } for proposal in self.proposals for item in proposal.specs]
        source_payload = [{
            "candidate": list(spec.candidate.stable_key()),
            "operator_evidence": [list(value.stable_key())
                                  for value in adoption.evidence],
            "modal_evidence": list(_plan(binding.proposal).evidence_ids),
        } for binding in self.adapter.evidence
            if binding.proposal.observation.substage == W07_L06_SUBSTAGE
            for spec in binding.proposal.specs
            if (adoption := self.learning.logic.adoption(spec)) is not None]
        return W07L06RuntimeReport(
            digest_value(operator_payload),
            digest_value(source_payload),
            digest_value([list(item.stable_key()) for item in executions]),
            len(specs),
            len(active),
            len({item.definition.stable_key() for item in active}),
            len(executable),
            resolved,
            len(plans) - resolved,
            len(output_scopes),
            sum(item.output_scope != item.input_scope
                for item in plans if item.output_scope is not None),
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
            0,
        )


def build_w07_l06_runtime(
        backend,
        adapter: W07TypedAdapterOutput,
        *,
        protocol: W07LogicConsumerProtocol = (
            W07LogicConsumerProtocol(W07_L06_PREFIX)),
        ) -> W07L06Runtime:
    """构建严格 L01..L06 prefix 的共享学习/执行 owner。"""
    sliced = slice_w07_adapter(adapter, W07_L06_PREFIX)
    learning = build_w07_learning_runtime(backend, sliced)
    return W07L06Runtime(learning, sliced, protocol=protocol)


__all__ = [
    "W07L06Runtime",
    "W07L06RuntimeReport",
    "W07_L06_PREFIX",
    "W07_L06_SUBSTAGE",
    "build_w07_l06_runtime",
    "generation_request_for_proposal",
    "logic_request_for_proposal",
]
