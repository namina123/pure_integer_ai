"""W-07 evaluator 对七个已发布 U/R/G facade 的隔离消费。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pure_integer_ai.cognition.shared.scope_identity import query_scope
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w07_adapter import W07TypedAdapterOutput
from pure_integer_ai.experiments.ph2_w07_contract import W07_SUBSTAGE_ORDER
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07PrivateEvaluationError,
)
from pure_integer_ai.experiments.ph2_w07_l01 import (
    W07L01Runtime,
    logic_request_for_proposal as l01_logic_request,
)
from pure_integer_ai.experiments.ph2_w07_l02 import (
    W07L02Runtime,
    W07_L02_PREFIX,
    logic_request_for_proposal as l02_logic_request,
)
from pure_integer_ai.experiments.ph2_w07_l03 import (
    W07L03Runtime,
    W07_L03_PREFIX,
    logic_request_for_proposal as l03_logic_request,
)
from pure_integer_ai.experiments.ph2_w07_l04 import (
    W07L04Runtime,
    W07_L04_PREFIX,
    logic_request_for_proposal as l04_logic_request,
)
from pure_integer_ai.experiments.ph2_w07_l05 import (
    W07L05Runtime,
    W07_L05_PREFIX,
    logic_request_for_proposal as l05_logic_request,
)
from pure_integer_ai.experiments.ph2_w07_l06 import (
    W07L06Runtime,
    W07_L06_PREFIX,
    logic_request_for_proposal as l06_logic_request,
)
from pure_integer_ai.experiments.ph2_w07_l07 import (
    W07L07Runtime,
    W07_L07_PREFIX,
    logic_request_for_proposal as l07_logic_request,
)
from pure_integer_ai.experiments.ph2_w07_learning import build_w07_learning_runtime
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w07_logic_generation import (
    generation_request_for_proposal,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    role_tree_key,
    slice_w07_adapter,
    structure_tree_key,
    w07_logic_language_branch,
)
from pure_integer_ai.experiments.typed_proof_family_contracts import (
    CONDITION_MATERIAL,
    CONDITION_NECESSARY,
    CONDITION_SUFFICIENT,
    PROOF_ACCEPTED,
)
from pure_integer_ai.storage.backend import StorageBackend


_EVALUATOR_NAMESPACE = 50761
_PREFIXES = {
    "NOT": ("NOT",),
    "AND_OR": W07_L02_PREFIX,
    "CONDITION": W07_L03_PREFIX,
    "EXISTS": W07_L04_PREFIX,
    "FORALL": W07_L05_PREFIX,
    "MODAL": W07_L06_PREFIX,
    "NESTED_SCOPE": W07_L07_PREFIX,
}
_RUNTIME_TYPES = {
    "NOT": W07L01Runtime,
    "AND_OR": W07L02Runtime,
    "CONDITION": W07L03Runtime,
    "EXISTS": W07L04Runtime,
    "FORALL": W07L05Runtime,
    "MODAL": W07L06Runtime,
    "NESTED_SCOPE": W07L07Runtime,
}
_LOGIC_REQUESTS = {
    "NOT": l01_logic_request,
    "AND_OR": l02_logic_request,
    "CONDITION": l03_logic_request,
    "EXISTS": l04_logic_request,
    "FORALL": l05_logic_request,
    "MODAL": l06_logic_request,
    "NESTED_SCOPE": l07_logic_request,
}


@dataclass(frozen=True)
class _LogicBundle:
    substage: str
    backend: StorageBackend
    adapter: W07TypedAdapterOutput
    learning: object


class W07EvaluatorConsumerSuite:
    """持有七个物理 learning ledger，并调用各自已发布 facade。"""

    def __init__(self, bundles: tuple[_LogicBundle, ...]) -> None:
        if (not isinstance(bundles, tuple)
                or tuple(item.substage for item in bundles)
                != W07_SUBSTAGE_ORDER):
            raise W07PrivateEvaluationError("W-07 evaluator suite order drift")
        self._bundles = {item.substage: item for item in bundles}
        self._audit = {
            "generation_choices": 0,
            "generation_outcomes": 0,
            "generation_uses": 0,
            "nested_generation_layer_uses": 0,
            "nested_reasoning_layer_uses": 0,
            "nested_understanding_layer_uses": 0,
            "reasoning_outcomes": 0,
            "reasoning_uses": 0,
            "understanding_outcomes": 0,
            "understanding_uses": 0,
        }

    def close(self) -> None:
        for bundle in self._bundles.values():
            bundle.backend.close()

    def audit(self) -> dict[str, int]:
        return dict(self._audit)

    def _bundle(self, substage: str) -> _LogicBundle:
        try:
            return self._bundles[substage]
        except KeyError as error:
            raise W07PrivateEvaluationError(
                "W-07 evaluator substage is not registered") from error

    def _runtime(
            self,
            substage: str,
            *,
            target_connected: bool,
            generation_connected: bool,
            ):
        bundle = self._bundle(substage)
        prefix = _PREFIXES[substage]
        protocol = W07LogicConsumerProtocol(
            prefix,
            () if target_connected else (substage,),
            True,
            True,
            generation_connected,
            True,
        )
        return _RUNTIME_TYPES[substage](
            bundle.learning, bundle.adapter, protocol=protocol)

    @staticmethod
    def _key(
            challenge: tuple[int, ...],
            evaluation_ordinal: int,
            consumer_ordinal: int,
            proposal_ordinal: int,
            extra: int = 0,
            ) -> LosslessIntegerKey:
        return LosslessIntegerKey((
            _EVALUATOR_NAMESPACE,
            evaluation_ordinal,
            consumer_ordinal,
            proposal_ordinal,
            extra,
            len(challenge),
            *challenge,
        ))

    @staticmethod
    def _constraints(proposal) -> GenerationExpressionConstraints:
        branch = w07_logic_language_branch(proposal)
        return GenerationExpressionConstraints(
            branch,
            tuple(item.definition.structure for item in proposal.specs),
            (branch,),
            0,
            0,
            0,
            256,
        )

    def _representatives(self, substage: str, runtime) -> tuple:
        executable = runtime.view.executable_proposals(substage)
        if substage == "AND_OR":
            result = []
            for family in ("AND", "OR"):
                result.append(next(
                    item for item in executable
                    if item.operator_families == (family,)
                    and item.observation.perturbation_kind == "NONE"))
            return tuple(result)
        normal = tuple(
            item for item in executable
            if item.observation.perturbation_kind == "NONE")
        if normal:
            return (normal[0],)
        if not executable:
            raise W07PrivateEvaluationError(
                f"{substage} has no executable proposal")
        return (executable[0],)

    def _logic_request(
            self,
            substage: str,
            proposal,
            challenge: tuple[int, ...],
            evaluation_ordinal: int,
            consumer_ordinal: int,
            proposal_ordinal: int,
            ):
        return _LOGIC_REQUESTS[substage](
            proposal,
            request_key=self._key(
                challenge, evaluation_ordinal, consumer_ordinal,
                proposal_ordinal),
        )

    def _consume_logic(self, runtime, request, *, consumer: str) -> bool:
        if consumer == "UNDERSTANDING":
            resolution = runtime.resolve_understanding(request)
            if resolution.status == "NO_ADOPTION":
                return False
            use = runtime.adopt_understanding(resolution)
            self._audit["understanding_uses"] += 1
            outcome = runtime.verify_understanding(use)
            self._audit["understanding_outcomes"] += 1
            return outcome.verdict == "SUPPORT"
        if consumer == "REASONING":
            resolution = runtime.resolve_reasoning(request)
            if resolution.status == "NO_ADOPTION":
                return False
            use = runtime.adopt_reasoning(resolution)
            self._audit["reasoning_uses"] += 1
            outcome = runtime.verify_reasoning(use)
            self._audit["reasoning_outcomes"] += 1
            return outcome.verdict == "SUPPORT"
        raise W07PrivateEvaluationError("W-07 evaluator consumer drift")

    def _consume_generation(self, runtime, request) -> tuple[bool, object | None]:
        choice = runtime.choose_generation(request)
        self._audit["generation_choices"] += 1
        if choice.status != "READY" or not choice.options:
            return False, None
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        self._audit["generation_uses"] += 1
        outcome = runtime.verify_generation(use)
        self._audit["generation_outcomes"] += 1
        return outcome.verdict == "SUPPORT", outcome

    def _consume_urg(
            self,
            substage: str,
            runtime,
            proposals: tuple,
            challenge: tuple[int, ...],
            evaluation_ordinal: int,
            ) -> tuple[list[bool], list[bool], list[bool], list[object]]:
        understanding = []
        reasoning = []
        generation = []
        outcomes = []
        before_layers = len(getattr(runtime, "layer_uses", ()))
        for ordinal, proposal in enumerate(proposals, start=1):
            understanding.append(self._consume_logic(
                runtime,
                self._logic_request(
                    substage, proposal, challenge, evaluation_ordinal,
                    1, ordinal),
                consumer="UNDERSTANDING",
            ))
            reasoning.append(self._consume_logic(
                runtime,
                self._logic_request(
                    substage, proposal, challenge, evaluation_ordinal,
                    2, ordinal),
                consumer="REASONING",
            ))
            passed, outcome = self._consume_generation(
                runtime,
                generation_request_for_proposal(
                    proposal,
                    request_key=self._key(
                        challenge, evaluation_ordinal, 3, ordinal),
                    logic_request_key=self._key(
                        challenge, evaluation_ordinal, 4, ordinal),
                    constraints=self._constraints(proposal),
                ),
            )
            generation.append(passed)
            if outcome is not None:
                outcomes.append(outcome)
        if substage == "NESTED_SCOPE":
            new_layers = runtime.layer_uses[before_layers:]
            self._audit["nested_understanding_layer_uses"] += sum(
                item.consumer == "UNDERSTANDING" for item in new_layers)
            self._audit["nested_reasoning_layer_uses"] += sum(
                item.consumer == "REASONING" for item in new_layers)
            self._audit["nested_generation_layer_uses"] += sum(
                item.consumer == "GENERATION" for item in new_layers)
        return understanding, reasoning, generation, outcomes

    def _execution(
            self,
            substage: str,
            runtime,
            proposal,
            challenge: tuple[int, ...],
            evaluation_ordinal: int,
            ordinal: int,
            ):
        request = _LOGIC_REQUESTS[substage](
            proposal,
            request_key=self._key(
                challenge, evaluation_ordinal, 8, ordinal),
        )
        return runtime.view.execute(request)

    def _hard_checks(
            self,
            substage: str,
            runtime,
            probe_runtime,
            challenge: tuple[int, ...],
            evaluation_ordinal: int,
            ) -> dict[str, int]:
        proposals = probe_runtime.view.executable_proposals(substage)
        executions = [
            self._execution(
                substage, runtime, proposal, challenge,
                evaluation_ordinal, ordinal)
            for ordinal, proposal in enumerate(proposals, start=1)
        ]
        valid = [item for item in executions if item is not None]
        states = {item.evaluation.state.stable_key() for item in valid}
        expected_states = (
            {(1, 0), (0, 1), (0, 0)}
            if substage == "NESTED_SCOPE"
            else {(1, 0), (0, 1), (0, 0), (1, 1)}
        )
        checks: dict[str, bool] = {
            "four_state_preserved": states == expected_states,
        }
        if substage == "NOT":
            proposal = self._representatives(substage, probe_runtime)[0]
            request = _LOGIC_REQUESTS[substage](
                proposal,
                request_key=self._key(
                    challenge, evaluation_ordinal, 9, 1),
            )
            flipped = type(request)(
                request.request_key,
                request.substage,
                request.target_proposition,
                request.source,
                query_scope(99007, parent=request.scope),
                request.budget,
            )
            checks["scope_flip_rejected"] = (
                runtime.understanding.preview(flipped).status == "NO_ADOPTION")
            checks["unknown_conflict_preserved"] = {
                (0, 0), (1, 1)}.issubset(states)
        elif substage == "AND_OR":
            by_family = {
                family: {
                    execution.evaluation.state.stable_key()
                    for proposal, execution in zip(
                        proposals, executions, strict=True)
                    if execution is not None
                    and proposal.operator_families == (family,)
                }
                for family in ("AND", "OR")
            }
            checks["and_or_distinct"] = set(by_family) == {"AND", "OR"}
            checks["and_or_four_state"] = all(
                values == {(1, 0), (0, 1), (0, 0), (1, 1)}
                for values in by_family.values())
        elif substage == "CONDITION":
            invalid = tuple(
                item for item in runtime.proposals
                if item.observation.perturbation_kind in {
                    "ANTECEDENT_CONSEQUENT_SWAP",
                    "CAUSAL_CONFUSION",
                    "TEMPORAL_CONFUSION",
                })
            checks["condition_causal_temporal_isolated"] = (
                len(invalid) == 3 and all(
                    runtime.understanding.preview(_LOGIC_REQUESTS[substage](
                        proposal,
                        request_key=self._key(
                            challenge, evaluation_ordinal, 10, ordinal),
                    )).status == "NO_ADOPTION"
                    for ordinal, proposal in enumerate(invalid, start=1)))
            support = next(
                (proposal for proposal, execution in zip(
                    proposals, executions, strict=True)
                 if execution is not None
                 and execution.evaluation.state.stable_key() == (1, 0)),
                None,
            )
            proofs = []
            if support is not None:
                for ordinal, kind in enumerate((
                        CONDITION_MATERIAL,
                        CONDITION_SUFFICIENT,
                        CONDITION_NECESSARY), start=1):
                    proofs.append(runtime.prove(
                        _LOGIC_REQUESTS[substage](
                            support,
                            request_key=self._key(
                                challenge, evaluation_ordinal, 11, ordinal),
                        ),
                        kind,
                    ))
            checks["condition_certificates"] = (
                len(proofs) == 3 and all(
                    item is not None
                    and item.receipt.result.status == PROOF_ACCEPTED
                    for item in proofs))
        elif substage == "EXISTS":
            open_proposal = next((
                item for item in runtime.proposals
                if item.observation.perturbation_kind
                == "DOMAIN_CLOSURE_CONFUSION"), None)
            open_execution = None if open_proposal is None else self._execution(
                substage, runtime, open_proposal, challenge,
                evaluation_ordinal, 101)
            checks["open_domain_unknown"] = bool(
                open_execution is not None
                and open_execution.evaluation.state.stable_key() == (0, 0)
                and open_execution.evaluation.failures)
            confused = next((
                item for item in runtime.proposals
                if item.observation.perturbation_kind == "QUANTIFIER_SWAP"), None)
            checks["quantifier_swap_rejected"] = bool(
                confused is not None
                and runtime.understanding.preview(_LOGIC_REQUESTS[substage](
                    confused,
                    request_key=self._key(
                        challenge, evaluation_ordinal, 12, 1),
                )).status == "NO_ADOPTION")
        elif substage == "FORALL":
            open_proposal = next((
                item for item in runtime.proposals
                if item.observation.perturbation_kind
                == "DOMAIN_CLOSURE_CONFUSION"), None)
            open_execution = None if open_proposal is None else self._execution(
                substage, runtime, open_proposal, challenge,
                evaluation_ordinal, 102)
            counterexample = next((
                execution for execution in valid
                if execution.evaluation.state.stable_key() == (0, 1)
                and any(branch.state.refute
                        for branch in execution.evaluation.branches)), None)
            checks["open_domain_unknown"] = bool(
                open_execution is not None
                and open_execution.evaluation.state.stable_key() == (0, 0))
            checks["counterexample_refutes"] = counterexample is not None
            checks["quantifier_exchange_distinct"] = (
                any(item.observation.perturbation_kind == "QUANTIFIER_SWAP"
                    for item in runtime.proposals))
        elif substage == "MODAL":
            unresolved = []
            for ordinal, kind in enumerate((
                    "RESOLVER_MISSING", "RESOLVER_DENIED",
                    "BUDGET_UNDECIDED"), start=110):
                proposal = next((
                    item for item in runtime.proposals
                    if item.observation.perturbation_kind == kind), None)
                execution = None if proposal is None else self._execution(
                    substage, runtime, proposal, challenge,
                    evaluation_ordinal, ordinal)
                unresolved.append(bool(
                    execution is not None
                    and execution.evaluation.state.stable_key() == (0, 0)
                    and execution.evaluation.derivation == ()
                    and execution.evaluation.failures))
            shifted = next((
                item for item in runtime.proposals
                if item.observation.perturbation_kind == "MODAL_SCOPE_SHIFT"),
                None,
            )
            shifted_execution = None if shifted is None else self._execution(
                substage, runtime, shifted, challenge,
                evaluation_ordinal, 120)
            checks["modal_certificate_required"] = all(unresolved)
            checks["modal_scope_shift_preserved"] = bool(
                shifted_execution is not None
                and shifted_execution.evaluation.scope != shifted.request_scope)
        elif substage == "NESTED_SCOPE":
            exchanged = tuple(
                item for item in runtime.proposals
                if item.observation.perturbation_kind == "QUANTIFIER_SWAP")
            shifted = next((
                item for item in runtime.proposals
                if item.observation.perturbation_kind == "MODAL_SCOPE_SHIFT"),
                None,
            )
            shifted_execution = None if shifted is None else self._execution(
                substage, runtime, shifted, challenge,
                evaluation_ordinal, 130)
            checks["quantifier_exchange_distinct"] = bool(
                len(exchanged) == 2
                and exchanged[0].operator_families
                != exchanged[1].operator_families
                and structure_tree_key(exchanged[0].bound_root)
                != structure_tree_key(exchanged[1].bound_root)
                and role_tree_key(
                    exchanged[0].bound_root,
                    include_bound_provenance=True,
                ) != role_tree_key(
                    exchanged[1].bound_root,
                    include_bound_provenance=True,
                ))
            checks["nested_scope_shift_preserved"] = bool(
                shifted_execution is not None
                and shifted_execution.evaluation.scope != shifted.request_scope)
            layer_uses = getattr(runtime, "layer_uses", ())
            checks["nested_layer_trace"] = bool(
                layer_uses
                and {item.consumer for item in layer_uses}
                == {"UNDERSTANDING", "REASONING", "GENERATION"}
                and all(item.operator_premise_keys for item in layer_uses))
        return {key: int(value) for key, value in sorted(checks.items())}

    def evaluate_logic_dimension(
            self,
            substage: str,
            challenge: tuple[int, ...],
            *,
            target_connected: bool,
            evaluation_ordinal: int,
            ) -> tuple[bool, dict[str, object]]:
        """执行一个已发布 facade 的 U/R/G 路径与承重不变量。"""
        probe = self._runtime(
            substage, target_connected=True, generation_connected=True)
        runtime = self._runtime(
            substage,
            target_connected=target_connected,
            generation_connected=True,
        )
        representatives = self._representatives(substage, probe)
        understanding, reasoning, generation, outcomes = self._consume_urg(
            substage,
            runtime,
            representatives,
            challenge,
            evaluation_ordinal,
        )
        hard_checks = self._hard_checks(
            substage, runtime, probe, challenge, evaluation_ordinal)
        postchecks = [
            all((
                item.adoption_current,
                item.structure_preserved,
                item.role_order_preserved,
                item.state_preserved,
                item.source_scope_preserved,
                item.surface_valid,
                item.recovered_target,
            ))
            for item in outcomes
        ]
        passed = all((
            bool(representatives),
            len(understanding) == len(representatives),
            all(understanding),
            len(reasoning) == len(representatives),
            all(reasoning),
            len(generation) == len(representatives),
            all(generation),
            len(postchecks) == len(representatives),
            all(postchecks),
            all(hard_checks.values()),
        ))
        return passed, {
            "generation_postcheck_support_count": sum(postchecks),
            "generation_use_count": len(generation),
            "hard_checks": hard_checks,
            "reasoning_support_count": sum(reasoning),
            "representative_count": len(representatives),
            "substage": substage,
            "target_connected": int(target_connected),
            "understanding_support_count": sum(understanding),
        }

    def evaluate_generation_hard_conjunct(
            self,
            challenge: tuple[int, ...],
            *,
            generation_connected: bool,
            evaluation_ordinal: int,
            ) -> tuple[bool, dict[str, object]]:
        """通过每个 L01-L07 facade 执行真实 choice/use/postcheck。"""
        support = []
        full_checks = []
        for ordinal, substage in enumerate(W07_SUBSTAGE_ORDER, start=1):
            probe = self._runtime(
                substage, target_connected=True, generation_connected=True)
            runtime = self._runtime(
                substage,
                target_connected=True,
                generation_connected=generation_connected,
            )
            proposal = self._representatives(substage, probe)[0]
            passed, outcome = self._consume_generation(
                runtime,
                generation_request_for_proposal(
                    proposal,
                    request_key=self._key(
                        challenge, evaluation_ordinal, 20, ordinal),
                    logic_request_key=self._key(
                        challenge, evaluation_ordinal, 21, ordinal),
                    constraints=self._constraints(proposal),
                ),
            )
            support.append(passed)
            full_checks.append(bool(
                outcome is not None and all((
                    outcome.adoption_current,
                    outcome.structure_preserved,
                    outcome.role_order_preserved,
                    outcome.state_preserved,
                    outcome.source_scope_preserved,
                    outcome.surface_valid,
                    outcome.recovered_target,
                ))))
        passed = (
            len(support) == len(W07_SUBSTAGE_ORDER)
            and all(support) and all(full_checks))
        return passed, {
            "choice_use_postcheck_count": sum(full_checks),
            "generation_connected": int(generation_connected),
            "postcheck_support_count": sum(support),
            "substage_count": len(support),
        }


def build_w07_evaluator_consumer_suite(
        repository_root: str | Path,
        adapter: W07TypedAdapterOutput,
        *,
        backend_factory: Callable[[str], StorageBackend],
        ) -> W07EvaluatorConsumerSuite:
    """从 public adapter 建立七个 owner 隔离的 prefix ledger。"""
    del repository_root
    if not isinstance(adapter, W07TypedAdapterOutput):
        raise TypeError("W-07 evaluator suite needs a complete adapter")
    if not callable(backend_factory):
        raise TypeError("W-07 evaluator backend factory is invalid")
    bundles = []
    try:
        for substage in W07_SUBSTAGE_ORDER:
            sliced = slice_w07_adapter(adapter, _PREFIXES[substage])
            backend = backend_factory(substage)
            bundles.append(_LogicBundle(
                substage,
                backend,
                sliced,
                build_w07_learning_runtime(backend, sliced),
            ))
    except Exception:
        for bundle in bundles:
            bundle.backend.close()
        raise
    return W07EvaluatorConsumerSuite(tuple(bundles))


__all__ = [
    "W07EvaluatorConsumerSuite",
    "build_w07_evaluator_consumer_suite",
]
