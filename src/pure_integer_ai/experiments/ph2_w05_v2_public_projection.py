"""Public W-05 occurrence, proposition, role, scope and generation probes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    language_branch_identity,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
)
from pure_integer_ai.experiments.ph2_evaluation_public_plugin import (
    EvaluationPublicCapabilityRun,
    EvaluationPublicProbe,
    build_evaluation_public_probe,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05_IDENTITY_VERSIONS,
    W05AtomicPropositionCandidate,
    W05TypedAdapterOutput,
    adapt_w05_training_payload,
)
from pure_integer_ai.experiments.ph2_w05_generation import (
    build_w05_generation_runtime,
    generation_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_generation_contract import (
    W05_GENERATION_ADOPTED,
    W05_GENERATION_HARD_CASES,
    W05_GENERATION_OUTCOME_SUPPORT,
    W05_GENERATION_READY,
    W05GenerationCaseResult,
    W05GenerationProtocol,
    run_w05_generation_hard_conjunct,
)
from pure_integer_ai.experiments.ph2_w05_learning import (
    W05AtomicPropositionLearningRuntime,
    W05LearningError,
    build_w05_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_reasoning import (
    W05_REASONING_AUTHORIZED,
    W05_REASONING_CONFLICT,
    W05_REASONING_OUTCOME_SUPPORT,
    W05_REASONING_REJECTED,
    W05ReasoningProtocol,
    build_w05_reasoning_runtime,
    reasoning_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_understanding import (
    W05_UNDERSTANDING_CONFLICT,
    W05_UNDERSTANDING_OUTCOME_SUPPORT,
    W05_UNDERSTANDING_UNIQUE,
    W05_UNDERSTANDING_UNKNOWN,
    W05UnderstandingProtocol,
    build_w05_understanding_runtime,
    understanding_request_for_candidate,
)
from pure_integer_ai.storage.backend import DictBackend


W05V2PublicCapabilityRun = EvaluationPublicCapabilityRun
W05V2PublicProbe = EvaluationPublicProbe
_PUBLIC_REQUEST_NAMESPACE = 50550


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-05"
    )


def _probe(
        key: str,
        *,
        evaluated: bool,
        passed: bool,
        evidence: dict[str, int],
        operations: int,
        ) -> W05V2PublicProbe:
    return build_evaluation_public_probe(
        key,
        evaluated=evaluated,
        passed=passed,
        evidence=evidence,
        operations=operations,
    )


def _request_key(ordinal: int) -> LosslessIntegerKey:
    return LosslessIntegerKey((_PUBLIC_REQUEST_NAMESPACE, ordinal))


def _candidate(
        runtime: W05AtomicPropositionLearningRuntime,
        perturbation_kind: str,
        ) -> W05AtomicPropositionCandidate | None:
    values = tuple(
        item for item in runtime.registered_candidates()
        if item.perturbation_kind == perturbation_kind
    )
    return values[0] if len(values) == 1 else None


def _state_signature(runtime: W05AtomicPropositionLearningRuntime) -> str:
    report = runtime.report()
    applications = runtime.applications()
    return _sha({
        "active": [
            list(item.candidate.stable_key())
            for item in runtime.active_candidates()
        ],
        "applications": [
            {
                "accounts": [
                    {
                        "candidate": list(account.candidate.stable_key()),
                        "derived_supersede": int(account.derived_supersede),
                        "stance": account.stance,
                    }
                    for account in application.accounts
                ],
                "teacher": list(
                    application.binding.teacher_record.stable_key.stable_key()),
            }
            for application in applications
        ],
        "registered": [
            {
                "candidate": list(item.candidate.stable_key()),
                "context": list(
                    item.proposition_definition.context.stable_key()),
                "occurrences": [
                    list(value.identity.stable_key())
                    for value in item.occurrences
                ],
                "perturbation": item.perturbation_kind,
                "roles": [
                    list(value.stable_key())
                    for value in item.role_binding_identities()
                ],
            }
            for item in runtime.registered_candidates()
        ],
        "report": {
            "account_count": report.account_count,
            "active_candidate_count": report.active_candidate_count,
            "candidate_count": report.candidate_count,
            "conflict_candidate_count": report.conflict_candidate_count,
            "evidence_application_count": report.evidence_application_count,
            "occurrence_count": report.occurrence_count,
            "role_binding_count": report.role_binding_count,
            "superseded_candidate_count": report.superseded_candidate_count,
            "unknown_candidate_count": report.unknown_candidate_count,
        },
        "superseded": [
            list(item.candidate.stable_key())
            for item in runtime.superseded_candidates()
        ],
    })


def _generation_evidence(
        runtime: W05AtomicPropositionLearningRuntime,
        supported: W05AtomicPropositionCandidate,
        *,
        protocol: W05GenerationProtocol,
        request_ordinal: int,
        ) -> tuple[bool, dict[str, int], int]:
    branch = language_branch_identity(
        (_PUBLIC_REQUEST_NAMESPACE, request_ordinal, 1),
        versions=W05_IDENTITY_VERSIONS,
    )
    uncertainty = concept_identity(
        (_PUBLIC_REQUEST_NAMESPACE, request_ordinal, 2),
        versions=W05_IDENTITY_VERSIONS,
    )
    constraints = GenerationExpressionConstraints(
        branch, (), (), 0, 0, 0, 128)
    generation = build_w05_generation_runtime(runtime, protocol=protocol)
    choice = generation.choose(generation_request_for_candidate(
        supported,
        request_key=_request_key(request_ordinal),
        uncertainty=uncertainty,
        constraints=constraints,
    ))
    outcomes = ()
    if choice.status == W05_GENERATION_READY and choice.options:
        uses = generation.adopt(
            choice, tuple(item.stable_key() for item in choice.options))
        adopted = tuple(
            item for item in uses
            if item.decision.action == W05_GENERATION_ADOPTED
        )
        independent = build_w05_understanding_runtime(runtime)
        outcomes = tuple(
            generation.verify_use(item, understanding=independent)
            for item in adopted
        )
    case_values = (
        choice.status == W05_GENERATION_READY and bool(choice.options),
        bool(outcomes) and all(item.occurrence_preserved for item in outcomes),
        bool(outcomes) and all(item.role_preserved for item in outcomes),
        bool(outcomes) and all(item.scope_preserved for item in outcomes),
        bool(outcomes) and all(
            item.understanding_status == W05_UNDERSTANDING_UNIQUE
            for item in outcomes),
        bool(outcomes) and all(
            item.verdict == W05_GENERATION_OUTCOME_SUPPORT
            for item in outcomes),
    )
    cases = tuple(
        W05GenerationCaseResult(
            name,
            passed,
            LosslessIntegerKey((
                _PUBLIC_REQUEST_NAMESPACE,
                request_ordinal,
                100 + ordinal,
            )),
        )
        for ordinal, (name, passed) in enumerate(
            zip(W05_GENERATION_HARD_CASES, case_values, strict=True),
            start=1,
        )
    )
    hard = run_w05_generation_hard_conjunct(cases, protocol=protocol)
    return hard.status == "PASS", {
        "adopted_use_count": len(outcomes),
        "case_count": len(hard.cases),
        "case_pass_count": sum(int(item.passed) for item in hard.cases),
        "choice_option_count": len(choice.options),
        "choice_ready": int(choice.status == W05_GENERATION_READY),
        "outcome_support_count": sum(
            int(item.verdict == W05_GENERATION_OUTCOME_SUPPORT)
            for item in outcomes),
    }, 3 + len(choice.options) + len(outcomes) + len(cases)


def _occurrence_probe(
        runtime: W05AtomicPropositionLearningRuntime,
        ) -> W05V2PublicProbe:
    supported = _candidate(runtime, "NONE")
    role_swap = _candidate(runtime, "ROLE_SWAP")
    omission = _candidate(runtime, "OCCURRENCE_OMISSION")
    restore = _candidate(runtime, "OCCURRENCE_RESTORE")
    evaluated = all(item is not None for item in (
        supported, role_swap, omission, restore))
    if not evaluated:
        return _probe(
            _policy().bearing_dimension_keys[0],
            evaluated=False,
            passed=False,
            evidence={
                "required_candidate_count": 4,
                "visible_required_candidate_count": sum(
                    int(item is not None) for item in (
                        supported, role_swap, omission, restore)),
            },
            operations=len(runtime.registered_candidates()),
        )
    assert supported is not None
    assert role_swap is not None
    assert omission is not None
    assert restore is not None
    report = runtime.report()
    supported_ids = {item.identity for item in supported.occurrences}
    swapped_ids = {item.identity for item in role_swap.occurrences}
    all_occurrences = tuple(
        value.identity
        for candidate in runtime.registered_candidates()
        for value in candidate.occurrences
    )
    active = {item.candidate for item in runtime.active_candidates()}
    superseded = {item.candidate for item in runtime.superseded_candidates()}
    same_surface_disjoint = (
        supported.surface == role_swap.surface
        and supported_ids.isdisjoint(swapped_ids)
    )
    restore_linked = (
        restore.supersedes_observation_key == omission.observation.stable_key)
    occurrence_inventory_exact = (
        report.occurrence_count == len(all_occurrences)
        and len(set(all_occurrences)) == len(all_occurrences)
    )
    passed = all((
        same_surface_disjoint,
        occurrence_inventory_exact,
        omission.candidate in superseded,
        restore.candidate in active,
        restore_linked,
    ))
    return _probe(
        _policy().bearing_dimension_keys[0],
        evaluated=True,
        passed=passed,
        evidence={
            "active_restore": int(restore.candidate in active),
            "occurrence_count": report.occurrence_count,
            "occurrence_inventory_exact": int(occurrence_inventory_exact),
            "omission_superseded": int(omission.candidate in superseded),
            "restore_linked": int(restore_linked),
            "same_surface_disjoint": int(same_surface_disjoint),
        },
        operations=len(runtime.registered_candidates()) + len(all_occurrences),
    )


def _proposition_probe(
        runtime: W05AtomicPropositionLearningRuntime,
        ) -> W05V2PublicProbe:
    supported = _candidate(runtime, "NONE")
    if supported is None:
        return _probe(
            _policy().bearing_dimension_keys[1],
            evaluated=False,
            passed=False,
            evidence={"supported_candidate_count": 0},
            operations=len(runtime.registered_candidates()),
        )
    understanding = build_w05_understanding_runtime(
        runtime,
        protocol=W05UnderstandingProtocol(
            proposition_consumer_connected=True),
    )
    resolution = understanding.resolve(understanding_request_for_candidate(
        supported, request_key=_request_key(20)))
    understanding_ok = False
    if resolution.status == W05_UNDERSTANDING_UNIQUE:
        use = understanding.adopt(resolution, supported)
        understanding_ok = (
            understanding.verify_use(use).verdict
            == W05_UNDERSTANDING_OUTCOME_SUPPORT
        )
    reasoning = build_w05_reasoning_runtime(
        runtime,
        protocol=W05ReasoningProtocol(
            proposition_consumer_connected=True),
    )
    reasoning_use = reasoning.authorize(reasoning_request_for_candidate(
        supported, request_key=_request_key(21)))
    reasoning_ok = (
        reasoning_use.status == W05_REASONING_AUTHORIZED
        and reasoning.verify_use(reasoning_use).verdict
        == W05_REASONING_OUTCOME_SUPPORT
    )
    generation_ok, generation, operations = _generation_evidence(
        runtime,
        supported,
        protocol=W05GenerationProtocol(
            proposition_consumer_connected=True),
        request_ordinal=22,
    )
    return _probe(
        _policy().bearing_dimension_keys[1],
        evaluated=True,
        passed=understanding_ok and reasoning_ok and generation_ok,
        evidence={
            "generation_case_count": generation["case_count"],
            "generation_case_pass_count": generation["case_pass_count"],
            "generation_passed": int(generation_ok),
            "reasoning_authorized": int(reasoning_ok),
            "understanding_unique": int(understanding_ok),
        },
        operations=operations + 4,
    )


def _role_probe(
        runtime: W05AtomicPropositionLearningRuntime,
        ) -> W05V2PublicProbe:
    supported = _candidate(runtime, "NONE")
    role_swap = _candidate(runtime, "ROLE_SWAP")
    evaluated = supported is not None and role_swap is not None
    if not evaluated:
        return _probe(
            _policy().bearing_dimension_keys[2],
            evaluated=False,
            passed=False,
            evidence={
                "required_candidate_count": 2,
                "visible_required_candidate_count": sum(
                    int(item is not None) for item in (supported, role_swap)),
            },
            operations=len(runtime.registered_candidates()),
        )
    assert supported is not None
    assert role_swap is not None
    understanding = build_w05_understanding_runtime(
        runtime, protocol=W05UnderstandingProtocol(role_bridge_connected=True))
    normal_resolution = understanding.resolve(
        understanding_request_for_candidate(
            supported, request_key=_request_key(30)))
    swapped_resolution = understanding.resolve(
        understanding_request_for_candidate(
            role_swap, request_key=_request_key(31)))
    reasoning = build_w05_reasoning_runtime(
        runtime, protocol=W05ReasoningProtocol(role_bridge_connected=True))
    normal_use = reasoning.authorize(reasoning_request_for_candidate(
        supported, request_key=_request_key(32)))
    swapped_use = reasoning.authorize(reasoning_request_for_candidate(
        role_swap, request_key=_request_key(33)))
    role_bindings_differ = (
        supported.role_binding_identities()
        != role_swap.role_binding_identities())
    passed = all((
        role_bindings_differ,
        normal_resolution.status == W05_UNDERSTANDING_UNIQUE,
        swapped_resolution.status == W05_UNDERSTANDING_UNKNOWN,
        normal_use.status == W05_REASONING_AUTHORIZED,
        swapped_use.status == W05_REASONING_REJECTED,
    ))
    return _probe(
        _policy().bearing_dimension_keys[2],
        evaluated=True,
        passed=passed,
        evidence={
            "normal_authorized": int(
                normal_use.status == W05_REASONING_AUTHORIZED),
            "normal_unique": int(
                normal_resolution.status == W05_UNDERSTANDING_UNIQUE),
            "role_bindings_differ": int(role_bindings_differ),
            "swap_rejected": int(
                swapped_use.status == W05_REASONING_REJECTED),
            "swap_unknown": int(
                swapped_resolution.status == W05_UNDERSTANDING_UNKNOWN),
        },
        operations=6,
    )


def _scope_probe(
        runtime: W05AtomicPropositionLearningRuntime,
        ) -> W05V2PublicProbe:
    supported = _candidate(runtime, "NONE")
    scope_shift = _candidate(runtime, "SCOPE_SHIFT")
    evaluated = supported is not None and scope_shift is not None
    if not evaluated:
        return _probe(
            _policy().bearing_dimension_keys[3],
            evaluated=False,
            passed=False,
            evidence={
                "required_candidate_count": 2,
                "visible_required_candidate_count": sum(
                    int(item is not None) for item in (
                        supported, scope_shift)),
            },
            operations=len(runtime.registered_candidates()),
        )
    assert supported is not None
    assert scope_shift is not None
    understanding = build_w05_understanding_runtime(
        runtime,
        protocol=W05UnderstandingProtocol(scope_projection_connected=True),
    )
    normal_resolution = understanding.resolve(
        understanding_request_for_candidate(
            supported, request_key=_request_key(40)))
    shifted_resolution = understanding.resolve(
        understanding_request_for_candidate(
            scope_shift, request_key=_request_key(41)))
    reasoning = build_w05_reasoning_runtime(
        runtime,
        protocol=W05ReasoningProtocol(scope_projection_connected=True),
    )
    normal_use = reasoning.authorize(reasoning_request_for_candidate(
        supported, request_key=_request_key(42)))
    shifted_use = reasoning.authorize(reasoning_request_for_candidate(
        scope_shift, request_key=_request_key(43)))
    contexts_differ = (
        supported.proposition_definition.context
        != scope_shift.proposition_definition.context)
    passed = all((
        contexts_differ,
        normal_resolution.status == W05_UNDERSTANDING_UNIQUE,
        shifted_resolution.status == W05_UNDERSTANDING_CONFLICT,
        normal_use.status == W05_REASONING_AUTHORIZED,
        shifted_use.status == W05_REASONING_CONFLICT,
    ))
    return _probe(
        _policy().bearing_dimension_keys[3],
        evaluated=True,
        passed=passed,
        evidence={
            "contexts_differ": int(contexts_differ),
            "normal_authorized": int(
                normal_use.status == W05_REASONING_AUTHORIZED),
            "normal_unique": int(
                normal_resolution.status == W05_UNDERSTANDING_UNIQUE),
            "shift_conflict_reasoning": int(
                shifted_use.status == W05_REASONING_CONFLICT),
            "shift_conflict_understanding": int(
                shifted_resolution.status == W05_UNDERSTANDING_CONFLICT),
        },
        operations=6,
    )


def _generation_probe(
        runtime: W05AtomicPropositionLearningRuntime,
        ) -> W05V2PublicProbe:
    supported = _candidate(runtime, "NONE")
    if supported is None:
        return _probe(
            _policy().generation_hard_conjunct_key,
            evaluated=False,
            passed=False,
            evidence={"supported_candidate_count": 0},
            operations=len(runtime.registered_candidates()),
        )
    passed, evidence, operations = _generation_evidence(
        runtime,
        supported,
        protocol=W05GenerationProtocol(),
        request_ordinal=70,
    )
    return _probe(
        _policy().generation_hard_conjunct_key,
        evaluated=True,
        passed=passed,
        evidence=evidence,
        operations=operations,
    )


def _rollback_probe(
        output: W05TypedAdapterOutput,
        runtime: W05AtomicPropositionLearningRuntime,
        ) -> W05V2PublicProbe:
    safe = tuple(
        item for item in output.evidence
        if item.supersedes_observation_key is None
    )
    before = _state_signature(runtime)
    rejected = False
    if safe and output.candidates:
        invalid = replace(
            safe[0],
            candidates=(
                output.candidates[0].proposition_definition.predicate,),
        )
        try:
            runtime.apply_evidence(invalid)
        except W05LearningError:
            rejected = True
    after = _state_signature(runtime)
    evaluated = bool(safe and output.candidates)
    return _probe(
        _policy().hard_conjunct_keys[-3],
        evaluated=evaluated,
        passed=evaluated and rejected and before == after,
        evidence={
            "invalid_evidence_rejected": int(rejected),
            "state_unchanged": int(before == after),
        },
        operations=len(output.candidates) + len(output.evidence),
    )


def run_w05_v2_public_capability(
        payload: W05TrainingPayload,
        ) -> W05V2PublicCapabilityRun:
    """Run current W-05 learning and all public V2 hard-conjunct probes."""
    output = adapt_w05_training_payload(payload)
    backend = DictBackend()
    try:
        runtime = build_w05_learning_runtime(backend, output)
        probes = (
            _occurrence_probe(runtime),
            _proposition_probe(runtime),
            _role_probe(runtime),
            _scope_probe(runtime),
            _generation_probe(runtime),
        )
        rollback = _rollback_probe(output, runtime)
        operations = sum(item.operations for item in probes) + rollback.operations
        return W05V2PublicCapabilityRun(
            probes, rollback, _state_signature(runtime), operations)
    finally:
        backend.close()


__all__ = [
    "W05V2PublicCapabilityRun",
    "W05V2PublicProbe",
    "run_w05_v2_public_capability",
]
