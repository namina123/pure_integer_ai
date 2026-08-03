"""W07-L07 NESTED_SCOPE 的 typed facade、逐层 Use 与 bounded report。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
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
    W07_LOGIC_CONSUMERS,
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
    W07_LOGIC_RUNTIME_NAMESPACE,
    role_tree_key,
    slice_w07_adapter,
    structure_tree_key,
)


W07_L07_PREFIX = (
    "NOT", "AND_OR", "CONDITION", "EXISTS", "FORALL", "MODAL",
    "NESTED_SCOPE",
)
W07_L07_SUBSTAGE = "NESTED_SCOPE"


@dataclass(frozen=True)
class W07NestedLayerUse:
    """一个 consumer 对 nested derivation 中一个实际 operator 层的 exact Use。"""

    consumer: str
    ordinal: int
    family: str
    structure: ObjectIdentity
    proposition: ObjectIdentity
    operator_premise_keys: tuple[tuple[int, ...], ...]
    source: SourceRef
    scope: ScopeIdentity
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if self.consumer not in W07_LOGIC_CONSUMERS:
            raise W07LogicContractError("nested layer consumer 未注册")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise W07LogicContractError("nested layer ordinal 非法")
        if not isinstance(self.family, str) or not self.family:
            raise W07LogicContractError("nested layer family 非法")
        if not isinstance(self.structure, ObjectIdentity):
            raise W07LogicContractError("nested layer structure 非法")
        if not isinstance(self.proposition, ObjectIdentity):
            raise W07LogicContractError("nested layer proposition 非法")
        if (not self.operator_premise_keys
                or self.operator_premise_keys
                != tuple(sorted(set(self.operator_premise_keys)))):
            raise W07LogicContractError("nested layer operator Evidence 未规范化")
        if not isinstance(self.source, SourceRef):
            raise W07LogicContractError("nested layer source 非法")
        if not isinstance(self.scope, ScopeIdentity):
            raise W07LogicContractError("nested layer scope 非法")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W07LogicContractError("nested layer use key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            W07_LOGIC_CONSUMERS.index(self.consumer) + 1,
            self.ordinal,
            len(self.family),
            *(ord(item) for item in self.family),
            *pack_key(self.structure.stable_key()),
            *pack_key(self.proposition.stable_key()),
            len(self.operator_premise_keys),
        ]
        for item in self.operator_premise_keys:
            values.extend(pack_key(item))
        values.extend((
            *pack_key(self.source.stable_key()),
            *pack_key(self.scope.stable_key()),
            *pack_key(self.use_key.components),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W07L07RuntimeReport:
    operator_digest: tuple[int, ...]
    source_evidence_digest: tuple[int, ...]
    execution_digest: tuple[int, ...]
    candidate_count: int
    active_candidate_count: int
    operator_profile_count: int
    executable_proposal_count: int
    executed_layer_count: int
    branch_count: int
    binder_assignment_count: int
    supported_count: int
    refuted_count: int
    unknown_count: int
    conflict_count: int
    schema_rejection_count: int
    conflict_candidate_count: int
    superseded_candidate_count: int
    understanding_use_count: int
    reasoning_use_count: int
    generation_use_count: int
    generation_outcome_count: int
    understanding_layer_use_count: int
    reasoning_layer_use_count: int
    generation_layer_use_count: int
    cycle_claim_count: int = 0
    carrier_projection_count: int = 0
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
    if (not isinstance(proposal, W07LogicProposal)
            or proposal.observation.substage != W07_L07_SUBSTAGE):
        raise TypeError("W07-L07 request 只接受 NESTED_SCOPE proposal")
    return W07LogicRequest(
        request_key,
        W07_L07_SUBSTAGE,
        proposal.bound_root.template,
        proposal.source_binding.source_ref,
        proposal.request_scope,
    )


class W07L07Runtime:
    """共享完整 prefix owner，并为每个实际 nested layer 形成 exact Use。"""

    def __init__(
            self,
            learning: W07LogicLearningRuntime,
            adapter: W07TypedAdapterOutput,
            *,
            protocol: W07LogicConsumerProtocol = (
                W07LogicConsumerProtocol(W07_L07_PREFIX)),
            ) -> None:
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.view = W07LogicView(learning, adapter, protocol)
        self.understanding = W07LogicUnderstandingRuntime(self.view)
        self.reasoning = W07LogicReasoningRuntime(self.view)
        self.generation = W07LogicGenerationRuntime(self.view)
        self._layer_uses: list[W07NestedLayerUse] = []

    @property
    def proposals(self) -> tuple[W07LogicProposal, ...]:
        return tuple(
            item for item in self.adapter.proposals
            if item.observation.substage == W07_L07_SUBSTAGE)

    @property
    def layer_uses(self) -> tuple[W07NestedLayerUse, ...]:
        return tuple(self._layer_uses)

    def execute(self, request: W07LogicRequest) -> W07LogicExecution | None:
        if request.substage != W07_L07_SUBSTAGE:
            raise TypeError("W07-L07 execute 只接受 NESTED_SCOPE request")
        return self.view.execute(request)

    def _record_layer_uses(
            self,
            consumer: str,
            execution: W07LogicExecution,
            ) -> tuple[W07NestedLayerUse, ...]:
        proposal = self.view.proposal_for(execution.request)
        if proposal is None:
            raise W07LogicContractError("nested execution 缺 proposal")
        by_structure = {
            spec.definition.structure: family
            for family, spec in zip(
                proposal.operator_families, proposal.specs, strict=True)
        }
        uses = []
        for ordinal, step in enumerate(execution.evaluation.derivation):
            family = by_structure.get(step.operator)
            if family is None:
                raise W07LogicContractError("nested derivation 命中未声明 operator")
            keys = tuple(sorted({
                (1, *evidence.stable_key())
                for adoption in execution.operator_adoptions
                if adoption.spec.definition.structure == step.operator
                for evidence in adoption.evidence
            }))
            if not keys:
                raise W07LogicContractError("nested layer 缺 current adoption Evidence")
            use = W07NestedLayerUse(
                consumer,
                ordinal,
                family,
                step.operator,
                step.proposition,
                keys,
                step.source,
                step.scope,
                LosslessIntegerKey((
                    W07_LOGIC_RUNTIME_NAMESPACE,
                    910 + W07_LOGIC_CONSUMERS.index(consumer),
                    len(self._layer_uses) + len(uses) + 1,
                    *pack_key(execution.request.request_key.components),
                    ordinal,
                )),
            )
            uses.append(use)
        self._layer_uses.extend(uses)
        return tuple(uses)

    def resolve_understanding(self, request):
        return self.understanding.resolve(request)

    def adopt_understanding(self, resolution):
        use = self.understanding.adopt(resolution)
        assert resolution.execution is not None
        self._record_layer_uses("UNDERSTANDING", resolution.execution)
        return use

    def verify_understanding(self, use):
        return self.understanding.verify(use)

    def resolve_reasoning(self, request):
        return self.reasoning.resolve(request)

    def adopt_reasoning(self, resolution):
        use = self.reasoning.adopt(resolution)
        assert resolution.execution is not None
        self._record_layer_uses("REASONING", resolution.execution)
        return use

    def verify_reasoning(self, use):
        return self.reasoning.verify(use)

    def choose_generation(self, request):
        return self.generation.choose(request)

    def adopt_generation(self, choice, option_key):
        use = self.generation.adopt(choice, option_key)
        self._record_layer_uses("GENERATION", use.execution)
        return use

    def verify_generation(self, use):
        return self.generation.verify(use)

    def state_key(self) -> tuple:
        return (
            self.learning.logic.state_key(),
            self.protocol.stable_key(),
            self.understanding.state_key(),
            self.reasoning.state_key(),
            self.generation.state_key(),
            tuple(item.stable_key() for item in self._layer_uses),
        )

    def report(self) -> W07L07RuntimeReport:
        learning_report = self.learning.report()
        specs = tuple(
            spec for proposal in self.proposals for spec in proposal.specs)
        active = tuple(
            spec for spec in specs
            if self.learning.logic.adoption(spec) is not None)
        executable = self.view.executable_proposals(W07_L07_SUBSTAGE)
        executions = tuple(
            self.execute(logic_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((70707, index)),
            ))
            for index, proposal in enumerate(executable, start=1)
        )
        executions = tuple(item for item in executions if item is not None)
        states = tuple(item.evaluation.state.stable_key() for item in executions)
        operator_payload = [{
            "families": list(proposal.operator_families),
            "definitions": [list(item.definition.stable_key())
                            for item in proposal.specs],
            "structure_tree": list(structure_tree_key(proposal.bound_root)),
            "role_tree": list(role_tree_key(
                proposal.bound_root, include_bound_provenance=True)),
        } for proposal in self.proposals]
        source_payload = [{
            "candidate": list(spec.candidate.stable_key()),
            "family": family,
            "operator_evidence": [list(value.stable_key())
                                  for value in adoption.evidence],
        } for proposal in self.proposals
            for family, spec in zip(
                proposal.operator_families, proposal.specs, strict=True)
            if (adoption := self.learning.logic.adoption(spec)) is not None]
        return W07L07RuntimeReport(
            digest_value(operator_payload),
            digest_value(source_payload),
            digest_value([list(item.stable_key()) for item in executions]),
            len(specs),
            len(active),
            len({item.definition.stable_key() for item in active}),
            len(executable),
            sum(len(item.evaluation.derivation) for item in executions),
            sum(len(item.evaluation.branches) for item in executions),
            sum(branch.assignment is not None
                for item in executions for branch in item.evaluation.branches),
            states.count((1, 0)),
            states.count((0, 1)),
            states.count((0, 0)),
            states.count((1, 1)),
            learning_report.schema_rejection_count,
            learning_report.conflict_candidate_count,
            learning_report.superseded_candidate_count,
            len(self.understanding.uses),
            len(self.reasoning.uses),
            len(self.generation.uses),
            len(self.generation.outcomes),
            sum(item.consumer == "UNDERSTANDING" for item in self._layer_uses),
            sum(item.consumer == "REASONING" for item in self._layer_uses),
            sum(item.consumer == "GENERATION" for item in self._layer_uses),
        )


def build_w07_l07_runtime(
        backend,
        adapter: W07TypedAdapterOutput,
        *,
        protocol: W07LogicConsumerProtocol = (
            W07LogicConsumerProtocol(W07_L07_PREFIX)),
        ) -> W07L07Runtime:
    """构建完整 L01..L07 prefix 的共享学习/执行 owner。"""
    sliced = slice_w07_adapter(adapter, W07_L07_PREFIX)
    learning = build_w07_learning_runtime(backend, sliced)
    return W07L07Runtime(learning, sliced, protocol=protocol)


__all__ = [
    "W07L07Runtime",
    "W07L07RuntimeReport",
    "W07NestedLayerUse",
    "W07_L07_PREFIX",
    "W07_L07_SUBSTAGE",
    "build_w07_l07_runtime",
    "generation_request_for_proposal",
    "logic_request_for_proposal",
]
