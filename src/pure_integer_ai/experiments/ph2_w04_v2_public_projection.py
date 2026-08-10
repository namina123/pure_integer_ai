"""Pure capability projections for the public-only W-04 V2 consumer."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

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
from pure_integer_ai.experiments.ph2_w04_adapter import (
    W04TypedAdapterOutput,
    adapt_w04_training_payload,
)
from pure_integer_ai.experiments.ph2_w04_generation import (
    build_w04_generation_runtime,
)
from pure_integer_ai.experiments.ph2_w04_generation_contract import (
    W04_GENERATION_READY,
    W04GenerationRequest,
)
from pure_integer_ai.experiments.ph2_w04_learning import (
    W04LearningError,
    W04PrimitiveSurfaceLearningRuntime,
    build_w04_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_reasoning import (
    W04_REASONING_AUTHORIZED,
    build_w04_reasoning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_understanding import (
    W04_UNDERSTANDING_UNIQUE,
    build_w04_understanding_runtime,
)
from pure_integer_ai.storage.backend import DictBackend


W04V2PublicProbe = EvaluationPublicProbe
W04V2PublicCapabilityRun = EvaluationPublicCapabilityRun


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _policy():
    return next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == "W-04"
    )


def _probe(
        key: str,
        *,
        evaluated: bool,
        passed: bool,
        evidence: dict[str, int],
        operations: int,
        ) -> W04V2PublicProbe:
    return build_evaluation_public_probe(
        key,
        evaluated=evaluated,
        passed=passed,
        evidence=evidence,
        operations=operations,
    )


def _candidate_rows(candidates) -> tuple[dict[str, object], ...]:
    return tuple(sorted((
        {
            "candidate": item.candidate.stable_key(),
            "context": item.context_text,
            "coordinate": item.coordinate_key(),
            "observation": item.observation.stable_key.stable_key(),
            "surface": item.surface_form,
        }
        for item in candidates
    ), key=lambda item: tuple(item["candidate"])))


def _state_signature(runtime: W04PrimitiveSurfaceLearningRuntime) -> str:
    report = runtime.report()
    return _sha({
        "active": _candidate_rows(runtime.active_candidates()),
        "registered": _candidate_rows(runtime.registered_candidates()),
        "report": {
            "account_count": report.account_count,
            "active_candidate_count": report.active_candidate_count,
            "candidate_count": report.candidate_count,
            "conflict_candidate_count": report.conflict_candidate_count,
            "evidence_application_count": report.evidence_application_count,
            "superseded_candidate_count": report.superseded_candidate_count,
            "unknown_candidate_count": report.unknown_candidate_count,
        },
        "superseded": _candidate_rows(runtime.superseded_candidates()),
    })


def _content_replacement_probe(
        runtime: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04V2PublicProbe:
    superseded = runtime.superseded_candidates()
    active = runtime.active_candidates()
    replacements = tuple(
        (old, new)
        for old in superseded
        for new in active
        if new.coordinate_key() == old.coordinate_key()
        and new.supersedes_observation_key == old.observation.stable_key
        and (new.surface_form, new.context_text)
        != (old.surface_form, old.context_text)
    )
    return _probe(
        _policy().bearing_dimension_keys[0],
        evaluated=bool(superseded),
        passed=bool(superseded) and len(replacements) == len(superseded),
        evidence={
            "active_candidate_count": len(active),
            "content_replacement_count": len(replacements),
            "superseded_candidate_count": len(superseded),
        },
        operations=len(active) * max(1, len(superseded)),
    )


def _cue_replacement_probe(
        runtime: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04V2PublicProbe:
    candidates = runtime.registered_candidates()
    surface_groups: dict[str, set[tuple[str, int]]] = {}
    primitive_groups: dict[tuple[str, int], set[str]] = {}
    for item in candidates:
        surface_groups.setdefault(item.surface_form, set()).add(
            item.coordinate_key())
        primitive_groups.setdefault(item.coordinate_key(), set()).add(
            item.surface_form)
    competing_surfaces = sum(
        1 for values in surface_groups.values() if len(values) > 1)
    replacement_cues = sum(
        1 for values in primitive_groups.values() if len(values) > 1)
    active_count = len(runtime.active_candidates())
    evaluated = bool(competing_surfaces and replacement_cues)
    return _probe(
        _policy().bearing_dimension_keys[1],
        evaluated=evaluated,
        passed=evaluated and active_count == 1,
        evidence={
            "active_candidate_count": active_count,
            "candidate_count": len(candidates),
            "competing_surface_count": competing_surfaces,
            "replacement_cue_count": replacement_cues,
        },
        operations=len(candidates) * 2,
    )


def _evidence_ablation_probe(
        output: W04TypedAdapterOutput,
        runtime: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04V2PublicProbe:
    backend = DictBackend()
    try:
        ablated = W04PrimitiveSurfaceLearningRuntime(backend)
        ablated.register_adapter_output(output)
        without_evidence = ablated.report()
    finally:
        backend.close()
    full = runtime.report()
    evaluated = bool(output.evidence)
    passed = (
        evaluated
        and without_evidence.active_candidate_count == 0
        and full.active_candidate_count > 0
        and full.evidence_application_count == len(output.evidence)
        and full.account_count >= len(output.evidence)
    )
    return _probe(
        _policy().bearing_dimension_keys[2],
        evaluated=evaluated,
        passed=passed,
        evidence={
            "active_with_evidence": full.active_candidate_count,
            "active_without_evidence": without_evidence.active_candidate_count,
            "evidence_account_count": full.account_count,
            "evidence_application_count": full.evidence_application_count,
        },
        operations=len(output.candidates) + len(output.evidence),
    )


def _seed_ablation_probe(
        payload: W04TrainingPayload,
        runtime: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04V2PublicProbe:
    superseding = tuple(
        item for item in payload.observations
        if item.supersedes_key is not None
    )
    if not superseding:
        return _probe(
            _policy().bearing_dimension_keys[3],
            evaluated=False,
            passed=False,
            evidence={
                "candidate_count": runtime.report().candidate_count,
                "superseding_seed_count": 0,
            },
            operations=len(payload.observations),
        )
    removed = superseding[0]
    ablated_payload = W04TrainingPayload(
        payload.source_refs,
        tuple(item for item in payload.observations
              if item.stable_key != removed.stable_key),
        tuple(item for item in payload.teacher_evidence
              if item.observation_key != removed.stable_key),
    )
    backend = DictBackend()
    try:
        ablated_output = adapt_w04_training_payload(ablated_payload)
        ablated = build_w04_learning_runtime(backend, ablated_output)
        ablated_signature = _state_signature(ablated)
        ablated_report = ablated.report()
    finally:
        backend.close()
    full_report = runtime.report()
    full_signature = _state_signature(runtime)
    passed = (
        full_report.candidate_count == ablated_report.candidate_count + 1
        and full_report.superseded_candidate_count
        > ablated_report.superseded_candidate_count
        and full_signature != ablated_signature
    )
    return _probe(
        _policy().bearing_dimension_keys[3],
        evaluated=True,
        passed=passed,
        evidence={
            "ablated_candidate_count": ablated_report.candidate_count,
            "ablated_superseded_count": ablated_report.superseded_candidate_count,
            "full_candidate_count": full_report.candidate_count,
            "full_superseded_count": full_report.superseded_candidate_count,
            "state_changed": int(full_signature != ablated_signature),
        },
        operations=(len(payload.observations)
                    + ablated_report.candidate_count
                    + full_report.candidate_count),
    )


def _generation_probe(
        runtime: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04V2PublicProbe:
    active = runtime.active_candidates()
    understanding = build_w04_understanding_runtime(runtime)
    reasoning = build_w04_reasoning_runtime(runtime)
    generation = build_w04_generation_runtime(runtime)
    supported = 0
    authorized = 0
    unique = 0
    for candidate in active:
        resolution = understanding.resolve(
            candidate.surface_form, candidate.context_text)
        unique += int(
            resolution.status == W04_UNDERSTANDING_UNIQUE
            and resolution.selected == candidate)
        use = reasoning.authorize(
            candidate.primitive_registry, candidate.primitive_kind)
        authorized += int(use.status == W04_REASONING_AUTHORIZED)
        choice = generation.choose(W04GenerationRequest(
            candidate.primitive_registry,
            candidate.primitive_kind,
            candidate.context_text,
            True,
        ))
        selected = tuple(
            item for item in choice.options if item.candidate == candidate)
        if choice.status == W04_GENERATION_READY and selected:
            outcome = generation.verify_use(generation.adopt(choice, selected))
            supported += int(outcome.verdict == "SUPPORT")
    evaluated = bool(active)
    passed = (
        evaluated
        and unique == len(active)
        and authorized == len(active)
        and supported == len(active)
    )
    return _probe(
        _policy().generation_hard_conjunct_key,
        evaluated=evaluated,
        passed=passed,
        evidence={
            "active_candidate_count": len(active),
            "authorized_count": authorized,
            "supported_generation_count": supported,
            "unique_understanding_count": unique,
        },
        operations=len(active) * 4,
    )


def _rollback_probe(
        output: W04TypedAdapterOutput,
        runtime: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04V2PublicProbe:
    safe = tuple(
        item for item in output.evidence
        if item.supersedes_observation_key is None
    )
    before = _state_signature(runtime)
    rejected = False
    if safe and output.candidates:
        invalid = replace(
            safe[0],
            candidates=(output.candidates[0].primitive,),
        )
        try:
            runtime.apply_evidence(invalid)
        except W04LearningError:
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


def run_w04_v2_public_capability(
        payload: W04TrainingPayload,
        ) -> W04V2PublicCapabilityRun:
    """Run current W-04 learning and all public V2 hard-conjunct probes."""
    output = adapt_w04_training_payload(payload)
    backend = DictBackend()
    try:
        runtime = build_w04_learning_runtime(backend, output)
        probes = (
            _content_replacement_probe(runtime),
            _cue_replacement_probe(runtime),
            _evidence_ablation_probe(output, runtime),
            _seed_ablation_probe(payload, runtime),
            _generation_probe(runtime),
        )
        rollback = _rollback_probe(output, runtime)
        operations = sum(item.operations for item in probes) + rollback.operations
        return W04V2PublicCapabilityRun(
            probes, rollback, _state_signature(runtime), operations)
    finally:
        backend.close()


__all__ = [
    "W04V2PublicCapabilityRun",
    "W04V2PublicProbe",
    "run_w04_v2_public_capability",
]
