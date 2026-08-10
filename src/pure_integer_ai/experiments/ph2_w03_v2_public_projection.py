"""Pure capability projections for the public-only W-03 V2 consumer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w03_adapter import (
    W03SenseCandidateEnvelope,
    adapt_w03_training_payload,
)
from pure_integer_ai.experiments.ph2_w03_generation import (
    W03_GENERATION_CLARIFY,
    W03_GENERATION_HARD_CASES,
    W03_GENERATION_OUTCOME_REFUTE,
    W03_GENERATION_OUTCOME_SUPPORT,
    W03_GENERATION_READY,
    W03_GENERATION_UNKNOWN,
    W03ExpressionConstraints,
    W03GenerationCaseResult,
    W03GenerationRequest,
    build_w03_generation_runtime,
    run_w03_generation_hard_conjunct,
)
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.experiments.ph2_w03_understanding import (
    W03_UNDERSTANDING_AMBIGUOUS,
    W03_UNDERSTANDING_UNIQUE,
    W03_UNDERSTANDING_UNKNOWN,
    W03UnderstandingError,
    W03UnderstandingRuntime,
    build_w03_understanding_runtime,
)
from pure_integer_ai.storage.backend import DictBackend


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-03"
    )


def _groups(values: tuple[Any, ...], key_fn) -> tuple[tuple[Any, ...], ...]:
    grouped: dict[object, list[Any]] = {}
    for item in values:
        grouped.setdefault(key_fn(item), []).append(item)
    return tuple(
        tuple(sorted(group, key=lambda item: item.sense.stable_key()))
        for _, group in sorted(grouped.items(), key=lambda item: repr(item[0]))
    )


def _active(
        runtime: W03UnderstandingRuntime,
        candidate: W03SenseCandidateEnvelope,
        ) -> bool:
    return any(
        item.sense == candidate.sense
        for item in runtime.consumer.lookup(
            candidate.anchor.atom, context=candidate.context)
    )


def _request(
        candidate: W03SenseCandidateEnvelope,
        ordinal: int,
        *,
        context: object = ...,
        ) -> W03GenerationRequest:
    selected_context = candidate.context if context is ... else context
    return W03GenerationRequest(
        LosslessIntegerKey((303_200, ordinal)),
        candidate.sense,
        candidate.concept,
        selected_context,
        candidate.anchor.branch,
        W03ExpressionConstraints(True, True, 64),
        candidate.source_ref,
        document_scope(candidate.source_ref),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicProbe:
    key: str
    status: str
    evidence_sha256: str
    operations: int

    def __post_init__(self) -> None:
        if (not isinstance(self.key, str) or not self.key
                or self.status not in {"PASS", "FAIL", "NE", "BLOCKED"}
                or not isinstance(self.evidence_sha256, str)
                or len(self.evidence_sha256) != 64
                or type(self.operations) is not int
                or self.operations < 0):
            raise EvaluationKernelContractError("W-03 public probe drifted")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicCapabilityRun:
    probes: tuple[W03V2PublicProbe, ...]
    rollback_probe: W03V2PublicProbe
    state_signature: str
    operations: int


def _probe(
        key: str,
        *,
        evaluated: bool,
        passed: bool,
        evidence: dict[str, int],
        operations: int,
        ) -> W03V2PublicProbe:
    status = "PASS" if evaluated and passed else "FAIL" if evaluated else "NE"
    return W03V2PublicProbe(
        key,
        status,
        _sha({
            "evidence": evidence,
            "evaluated": int(evaluated),
            "passed": int(passed),
            "result_key": key,
            "status": status,
        }),
        operations,
    )


def _bearing_probes(runtime: W03UnderstandingRuntime) -> tuple[W03V2PublicProbe, ...]:
    policy = _policy()
    candidates = runtime.output.candidates
    contextual = _groups(
        candidates, lambda item: (item.anchor.atom, item.context))
    competition = tuple(group for group in contextual if len(group) > 1)
    separated = tuple(
        group for group in competition
        if (len({item.sense for item in group}) == len(group)
            and len({item.concept for item in group}) > 1)
    )
    concept_split = _probe(
        policy.bearing_dimension_keys[0],
        evaluated=bool(competition),
        passed=bool(separated),
        evidence={
            "candidate_count": len(candidates),
            "competition_group_count": len(competition),
            "separated_group_count": len(separated),
            "unique_sense_count": len({item.sense for item in candidates}),
        },
        operations=len(candidates) + sum(map(len, competition)),
    )

    unique_count = 0
    ambiguous_count = 0
    illegal_selection_count = 0
    for group in contextual:
        resolution = runtime.resolve(
            group[0].anchor.atom, context=group[0].context)
        if resolution.status == W03_UNDERSTANDING_UNIQUE:
            unique_count += 1
        if (len(group) > 1
                and resolution.status in {
                    W03_UNDERSTANDING_AMBIGUOUS,
                    W03_UNDERSTANDING_UNKNOWN,
                }):
            ambiguous_count += 1
            illegal_selection_count += int(resolution.selected is not None)
    polysemy = _probe(
        policy.bearing_dimension_keys[1],
        evaluated=bool(competition),
        passed=(unique_count >= 1 and ambiguous_count >= 1
                and illegal_selection_count == 0),
        evidence={
            "ambiguous_or_unknown_group_count": ambiguous_count,
            "competition_group_count": len(competition),
            "illegal_selection_count": illegal_selection_count,
            "unique_group_count": unique_count,
        },
        operations=sum(map(len, contextual)),
    )

    conflict_count = 0
    conflict_illegal_selection_count = 0
    for candidate in candidates:
        stances = {
            item.stance for item in runtime.evidence_accounts(candidate.sense)
        }
        if not {EVIDENCE_SUPPORT, EVIDENCE_REFUTE}.issubset(stances):
            continue
        conflict_count += 1
        resolution = runtime.resolve(
            candidate.anchor.atom, context=candidate.context)
        conflict_illegal_selection_count += int(resolution.selected is not None)
    source_conflict = _probe(
        policy.bearing_dimension_keys[2],
        evaluated=conflict_count > 0,
        passed=conflict_count > 0 and conflict_illegal_selection_count == 0,
        evidence={
            "conflict_candidate_count": conflict_count,
            "illegal_selection_count": conflict_illegal_selection_count,
        },
        operations=len(candidates),
    )

    pairs = []
    for new_candidate in candidates:
        old_key = runtime.supersedes_observation(new_candidate.sense)
        if old_key is None:
            continue
        pairs.extend(
            (old_candidate, new_candidate)
            for old_candidate in runtime.candidate_for_observation(old_key)
        )
    valid = 0
    for old_candidate, new_candidate in pairs:
        derived_refute = any(
            item.derived_supersede and item.stance == EVIDENCE_REFUTE
            for item in runtime.evidence_accounts(old_candidate.sense)
        )
        valid += int(
            old_candidate.anchor.atom == new_candidate.anchor.atom
            and old_candidate.sense != new_candidate.sense
            and not _active(runtime, old_candidate)
            and _active(runtime, new_candidate)
            and derived_refute
        )
    supersede = _probe(
        policy.bearing_dimension_keys[3],
        evaluated=bool(pairs),
        passed=valid > 0,
        evidence={
            "supersede_pair_count": len(pairs),
            "valid_same_atom_pair_count": valid,
        },
        operations=len(candidates) + len(pairs),
    )
    return concept_split, polysemy, source_conflict, supersede


def _withdrawal_probe(output) -> tuple[bool, bool, dict[str, int], int]:
    candidates = output.candidates
    same_atom_pairs = tuple(sorted(
        (
            (old_candidate, new_candidate)
            for new_candidate in candidates
            if new_candidate.supersedes_observation_key is not None
            for old_candidate in candidates
            if (old_candidate.observation.stable_key
                == new_candidate.supersedes_observation_key
                and old_candidate.anchor.atom == new_candidate.anchor.atom)
        ),
        key=lambda pair: (
            pair[0].sense.stable_key(), pair[1].sense.stable_key()),
    ))
    if not same_atom_pairs:
        return False, False, {"same_atom_pair_count": 0}, len(candidates)
    old_candidate, new_candidate = same_atom_pairs[0]
    evidence_by_observation = {
        item.observation.stable_key: item for item in output.evidence}
    old_evidence = evidence_by_observation.get(old_candidate.observation.stable_key)
    new_evidence = evidence_by_observation.get(new_candidate.observation.stable_key)
    if old_evidence is None or new_evidence is None:
        return False, False, {
            "same_atom_pair_count": len(same_atom_pairs),
        }, len(candidates)
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        runtime = build_w03_understanding_runtime(
            output, context.graph_ontology)
        runtime.apply_evidence(old_evidence)
        generation = build_w03_generation_runtime(runtime)
        before = generation.choose(_request(old_candidate, 501))
        if before.status != W03_GENERATION_READY or not before.options:
            return True, False, {
                "before_ready": 0,
                "same_atom_pair_count": len(same_atom_pairs),
            }, len(candidates) + 1
        old_option = next(
            (item for item in before.options
             if item.sense == old_candidate.sense),
            None,
        )
        if old_option is None:
            return True, False, {
                "before_ready": 1,
                "old_option_present": 0,
                "same_atom_pair_count": len(same_atom_pairs),
            }, len(candidates) + len(before.options)
        use = generation.adopt(before, (old_option.stable_key(),))[0]
        supported = generation.verify_use(use)
        runtime.apply_evidence(new_evidence)
        after_old = generation.choose(_request(old_candidate, 502))
        after_new = generation.choose(_request(new_candidate, 503))
        refuted = generation.verify_use(use)
        passed = (
            supported.verdict == W03_GENERATION_OUTCOME_SUPPORT
            and after_old.status == W03_GENERATION_UNKNOWN
            and after_old.options == ()
            and after_new.status == W03_GENERATION_READY
            and any(item.sense == new_candidate.sense
                    for item in after_new.options)
            and refuted.verdict == W03_GENERATION_OUTCOME_REFUTE
            and supported.use.ref.use_key == refuted.use.ref.use_key
        )
        return True, passed, {
            "after_new_ready": int(after_new.status == W03_GENERATION_READY),
            "after_old_unknown": int(after_old.status == W03_GENERATION_UNKNOWN),
            "same_atom_pair_count": len(same_atom_pairs),
            "support_then_refute": int(
                supported.verdict == W03_GENERATION_OUTCOME_SUPPORT
                and refuted.verdict == W03_GENERATION_OUTCOME_REFUTE),
        }, len(candidates) * 2 + len(before.options) + len(after_new.options)
    finally:
        backend.close()


def _generation_probe(
        runtime: W03UnderstandingRuntime,
        withdrawal: tuple[bool, bool, dict[str, int], int],
        ) -> W03V2PublicProbe:
    policy = _policy()
    candidates = runtime.output.candidates
    active = tuple(item for item in candidates if _active(runtime, item))
    generation = build_w03_generation_runtime(runtime)

    target_available = bool(active)
    target_ok = False
    if target_available:
        target = active[0]
        choice = generation.choose(_request(target, 1))
        target_ok = (
            choice.status == W03_GENERATION_READY
            and any(item.sense == target.sense for item in choice.options)
            and choice.selected is None
        )

    homograph_groups = tuple(
        group for group in _groups(candidates, lambda item: item.anchor.atom)
        if (len({item.sense for item in group}) > 1
            and len({item.concept for item in group}) > 1
            and any(_active(runtime, item) for item in group))
    )
    homograph_available = bool(homograph_groups)
    isolation_ok = ambiguity_ok = False
    if homograph_available:
        group = homograph_groups[0]
        target = next(item for item in group if _active(runtime, item))
        exact = generation.choose(_request(target, 2))
        missing = generation.choose(_request(target, 3, context=None))
        other_senses = {
            item.sense for item in group if item.sense != target.sense}
        isolation_ok = (
            exact.status == W03_GENERATION_READY
            and not other_senses.intersection(
                item.sense for item in exact.options)
        )
        ambiguity_ok = (
            missing.status == W03_GENERATION_CLARIFY
            and missing.options == () and missing.selected is None
        )

    active_groups = tuple(
        group for group in _groups(
            active,
            lambda item: (item.concept, item.context, item.anchor.branch),
        )
        if len({item.anchor.extracted.surface for item in group}) >= 2
    )
    multiple_available = bool(active_groups)
    multiple_ok = False
    if multiple_available:
        target = active_groups[0][0]
        choice = generation.choose(_request(target, 4))
        multiple_ok = (
            choice.status == W03_GENERATION_READY
            and len({item.surface for item in choice.options}) >= 2
            and len({item.sense for item in choice.options}) >= 2
            and choice.selected is None
        )

    withdrawal_available, withdrawal_ok, withdrawal_evidence, withdrawal_ops = (
        withdrawal)
    available = (
        target_available,
        homograph_available,
        homograph_available,
        multiple_available,
        withdrawal_available,
    )
    passed = (
        target_ok, isolation_ok, ambiguity_ok, multiple_ok, withdrawal_ok)
    cases = tuple(
        W03GenerationCaseResult(
            name,
            case_passed,
            LosslessIntegerKey((303_201, ordinal)),
        )
        for ordinal, (name, case_passed) in enumerate(
            zip(W03_GENERATION_HARD_CASES, passed, strict=True), start=1)
    )
    hard_report = run_w03_generation_hard_conjunct(
        cases,
        sense_consumer_connected=True,
        choice_bridge_connected=True,
    )
    evaluated_failure = any(
        case_available and not case_passed
        for case_available, case_passed in zip(available, passed, strict=True)
    )
    evaluated = all(available)
    final_passed = hard_report.status == "PASS"
    if evaluated_failure:
        evaluated = True
        final_passed = False
    evidence = {
        "active_candidate_count": len(active),
        "available_case_count": sum(available),
        "hard_case_pass_count": sum(passed),
        "homograph_group_count": len(homograph_groups),
        "multi_surface_group_count": len(active_groups),
        "withdrawal_pair_count": withdrawal_evidence.get(
            "same_atom_pair_count", 0),
    }
    return _probe(
        policy.generation_hard_conjunct_key,
        evaluated=evaluated,
        passed=final_passed,
        evidence=evidence,
        operations=len(candidates) * 3 + withdrawal_ops,
    )


def _state_signature(runtime: W03UnderstandingRuntime) -> str:
    report = runtime.report()
    active_senses = tuple(sorted(
        item.sense.stable_key()
        for item in runtime.output.candidates
        if _active(runtime, item)
    ))
    resolutions = []
    for group in _groups(
            runtime.output.candidates,
            lambda item: (item.anchor.atom, item.context)):
        result = runtime.resolve(
            group[0].anchor.atom, context=group[0].context)
        resolutions.append({
            "candidate_count": len(group),
            "selected": (
                [] if result.selected is None
                else list(result.selected.sense.stable_key())),
            "status": result.status,
        })
    return _sha({
        "active_senses": active_senses,
        "applied_evidence_count": report.applied_observation_evidence_count,
        "candidate_count": report.candidate_count,
        "conflict_count": report.source_conflict_candidate_count,
        "execution_state": report.execution_state,
        "resolutions": resolutions,
        "unbound_evidence_count": report.unbound_evidence_count,
    })


def run_w03_v2_public_capability(
        payload: W03TrainingPayload,
        ) -> W03V2PublicCapabilityRun:
    output = adapt_w03_training_payload(payload)
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        runtime = build_w03_understanding_runtime(
            output, context.graph_ontology)
        runtime.apply_all_evidence()
        bearings = _bearing_probes(runtime)
        withdrawal = _withdrawal_probe(output)
        generation = _generation_probe(runtime, withdrawal)
        before = runtime.candidate_runtime_state_key()
        duplicate_rejected = False
        if output.evidence:
            try:
                runtime.apply_evidence(output.evidence[0])
            except W03UnderstandingError:
                duplicate_rejected = True
        after = runtime.candidate_runtime_state_key()
        rollback = _probe(
            _policy().hard_conjunct_keys[-3],
            evaluated=bool(output.evidence),
            passed=duplicate_rejected and before == after,
            evidence={
                "duplicate_rejected": int(duplicate_rejected),
                "state_unchanged": int(before == after),
            },
            operations=len(output.evidence) + len(output.candidates),
        )
        probes = (*bearings, generation)
        operations = sum(item.operations for item in probes) + rollback.operations
        return W03V2PublicCapabilityRun(
            probes, rollback, _state_signature(runtime), operations)
    finally:
        backend.close()



__all__ = [
    "W03V2PublicCapabilityRun",
    "W03V2PublicProbe",
    "run_w03_v2_public_capability",
]
