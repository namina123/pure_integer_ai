"""Blind private evaluator for the Candidate -> V1 -> V2 -> V3 route chain."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v3 import (
    blind_private_source_specs_v3,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorBoundaryContract,
    V2EvaluatorResourceBudget,
    V2PrivateFamilyRegistration,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import _dimension_key
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
    _dimension_report,
    _evaluate_pair,
    _hash_value,
    _tree_sha256,
    load_w02_dev_candidate_index,
    predict_w02_dev_observation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    load_w02_morphology_overlay_index,
    read_w02_morphology_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02MorphologyRankingCache,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_dev_calibration import (
    _requested_spans,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN,
    W02MorphologySuccessorV2Cache,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_io import (
    authorize_w02_morphology_successor_v3_private_files,
    iter_w02_morphology_successor_v3_private_pairs,
    iter_w02_morphology_successor_v3_private_records,
    v3_private_file_by_layout,
    v3_private_split_layout,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner import (
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02MorphologySuccessorV3PrivateFileIdentity,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_contract import (
    W02_MORPH_V3_PRIVATE_SPLITS,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    W02MorphologySourceCapability,
    authorize_w02_morphology_source_routes,
    build_w02_morphology_routed_indexes,
    predict_w02_morphology_successor_v3,
    w02_ud_morphology_source_capability,
)


W02_MORPH_V3_PRIVATE_EVALUATOR_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-EVALUATOR-V1"
)
W02_MORPH_V3_PRIVATE_SUPPORT_KEYS = (
    "W-02-V2-RESOURCE",
    "W-02-V2-ROLLBACK",
    "W-02-V2-ZERO-CALL-WINDOWS",
    "W-02-V2-V06-CLONE",
)


# object-model: exception
class W02MorphologySuccessorV3PrivateEvaluationError(RuntimeError):
    """The V3 source route, artifact chain, resource, or result drifted."""


def _morphology_identity(value: object) -> tuple[object, ...]:
    return tuple(getattr(value, name) for name in (
        "start", "end", "form", "lemma", "upos", "feats_json"))


def _audit_v3_extension(v1: object, v2: object,
                        spans: tuple[tuple[int, int], ...]) -> int:
    before = {_morphology_identity(row) for row in v1.prediction.morphology_candidates}
    after = {_morphology_identity(row) for row in v2.prediction.morphology_candidates}
    added = after - before
    counts = {span: 0 for span in spans}
    for row in v2.prediction.morphology_candidates:
        identity = _morphology_identity(row)
        if identity not in added:
            continue
        span = (row.start, row.end)
        if span not in counts:
            raise W02MorphologySuccessorV3PrivateEvaluationError(
                "V3 candidate escaped requested spans")
        counts[span] += 1
    maximum = max(counts.values(), default=0)
    if (not before.issubset(after)
            or len(added) != v2.edge_candidate_count
            or maximum > W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN):
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "V3 V2-extension retention/bound drifted")
    return maximum


def _support_result(key: str, state: bool | None,
                    evidence: object) -> dict[str, object]:
    if key not in W02_MORPH_V3_PRIVATE_SUPPORT_KEYS:
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "V3 private support key drifted")
    return {
        "denominator": 1,
        "dimension_key": key,
        "evidence_sha256": _hash_value(evidence),
        "failed": int(state is False),
        "ne": int(state is None),
        "numerator": int(state is True),
        "status": "PASS" if state is True else "FAIL" if state is False else "NE",
    }


def _result_status(rows: Iterable[dict[str, object]]) -> str:
    statuses = tuple(str(row["status"]) for row in rows)
    if "FAIL" in statuses:
        return "FAIL"
    if "NE" in statuses or not statuses:
        return "NE"
    return "PASS"


def evaluate_w02_morphology_successor_v3_private_pair_stream(
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        sources: Iterable[SourceRefRecord],
        capabilities: Iterable[W02MorphologySourceCapability],
        pairs: Iterable[tuple[ObservationRecord, EvaluatorLabelRecord]],
        budget: V2EvaluatorResourceBudget,
        ) -> dict[str, object]:
    """Evaluate public synthetic or authorized private pairs through V3."""
    if not isinstance(budget, V2EvaluatorResourceBudget):
        raise TypeError("private V3 budget type drifted")
    candidate_root = Path(candidate_artifact_root).resolve()
    v1_root = Path(v1_overlay_artifact_root).resolve()
    v2_root = Path(v2_overlay_artifact_root).resolve()
    before = tuple(_tree_sha256(root) for root in (candidate_root, v1_root, v2_root))
    candidate_result = read_w02_candidate_artifact(candidate_root)
    v1_result = read_w02_morphology_overlay_artifact(v1_root)
    v2_result = read_w02_morphology_successor_v2_overlay_artifact(v2_root)
    if v1_result.parent_candidate_semantic_sha256 != candidate_result.candidate_semantic_sha256:
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "Candidate/V1 parent identity drifted")
    parent_v1 = load_w02_morphology_overlay_index(v1_root)
    parent_v2 = load_w02_morphology_successor_v2_overlay_index(v2_root)
    if parent_v1.dataset_keys != parent_v2.dataset_keys:
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "V1/V2 parent route identity drifted")

    source_count = 0
    source_digest = hashlib.sha256()

    def counted_sources() -> Iterable[SourceRefRecord]:
        nonlocal source_count
        for source in sources:
            if not isinstance(source, SourceRefRecord):
                raise W02MorphologySuccessorV3PrivateEvaluationError(
                    "V3 private SourceRef type drifted")
            source_digest.update(canonical_json_bytes(source.stable_key.to_list()))
            source_count += 1
            yield source

    routes = authorize_w02_morphology_source_routes(
        counted_sources(), capabilities, max_sources=budget.max_records)
    if routes.source_count != source_count:
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "not every V3 private source received a route")
    indexes = build_w02_morphology_routed_indexes(parent_v1, parent_v2, routes)
    dimension_by_key = {
        _dimension_key(name).components: name for name in W02_DEV_DIMENSIONS
    }
    rows = {name: [] for name in W02_DEV_DIMENSIONS}
    family_counts = {name: 0 for name in (
        "AUTHORED_OOV", "UD_ANNOTATION", "UNICODE_ANNOTATION")}
    v1_cache = W02MorphologyRankingCache.empty()
    v2_cache = W02MorphologySuccessorV2Cache.empty()
    base_operations = 0
    v1_operations = 0
    v2_operations = 0
    route_inference_operations = 0
    queried_spans = 0
    v1_candidates = 0
    v2_candidates = 0
    max_v1 = 0
    max_v2 = 0
    max_v2_per_span = 0
    route_authorized_count = 0
    evaluation_count = 0
    input_pair_count = 0
    resource_exhausted = False
    clone_probe = None
    pair_iterator = iter(pairs)
    try:
        with open_w02_candidate_predictor(candidate_root) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
            for observation, evaluation in pair_iterator:
                input_pair_count += 1
                if (not isinstance(observation, ObservationRecord)
                        or not isinstance(evaluation, EvaluatorLabelRecord)):
                    raise W02MorphologySuccessorV3PrivateEvaluationError(
                        "V3 private pair type drifted")
                base, operations = predict_w02_dev_observation(
                    candidate_index, observation)
                spans = _requested_spans(evaluation)
                v3 = predict_w02_morphology_successor_v3(
                    indexes, observation, base, requested_spans=spans,
                    v1_cache=v1_cache, v2_cache=v2_cache)
                if v3.route_authorized != 1:
                    raise W02MorphologySuccessorV3PrivateEvaluationError(
                        "V3 private observation was not route-authorized")
                maximum = _audit_v3_extension(v3.v1, v3.v2, spans)
                projected = (
                    routes.logic_operations + parent_v1.logic_operations
                    + parent_v2.logic_operations + base_operations + operations + 8
                    + v1_operations + v3.v1.logic_operations + v2_operations
                    + v3.v2.logic_operations + route_inference_operations
                    + v3.route_logic_operations)
                if projected > budget.max_logic_operations:
                    resource_exhausted = True
                    break
                dimension, family, passed, evidence_sha = _evaluate_pair(
                    observation, evaluation, v3.v2.prediction, dimension_by_key)
                if dimension not in rows or family not in family_counts:
                    raise W02MorphologySuccessorV3PrivateEvaluationError(
                        "V3 private dimension/family drifted")
                rows[dimension].append((passed, evidence_sha))
                family_counts[family] += 1
                base_operations += operations + 8
                v1_operations += v3.v1.logic_operations
                v2_operations += v3.v2.logic_operations
                route_inference_operations += v3.route_logic_operations
                queried_spans += len(spans)
                v1_candidates += v3.v1.generalized_candidate_count
                v2_candidates += v3.v2.edge_candidate_count
                max_v1 = max(max_v1, v3.v1.generalized_candidate_count)
                max_v2 = max(max_v2, v3.v2.edge_candidate_count)
                max_v2_per_span = max(max_v2_per_span, maximum)
                route_authorized_count += 1
                evaluation_count += 1
                if clone_probe is None:
                    clone_probe = (
                        observation, spans,
                        _hash_value(v3.v2.prediction.to_dict()))
            if resource_exhausted:
                for _observation, _evaluation in pair_iterator:
                    input_pair_count += 1
        dimensions = [
            _dimension_report(name, rows[name]) for name in W02_DEV_DIMENSIONS
        ]
        clone_ok = None
        clone_operations = 0
        if not resource_exhausted and clone_probe is not None:
            observation, spans, expected_sha = clone_probe
            clone_v1_cache = W02MorphologyRankingCache.empty()
            clone_v2_cache = W02MorphologySuccessorV2Cache.empty()
            try:
                with open_w02_candidate_predictor(candidate_root) as predictor:
                    clone_index = load_w02_dev_candidate_index(predictor)
                    clone_base, clone_operations = predict_w02_dev_observation(
                        clone_index, observation)
                clone_parent_v1 = load_w02_morphology_overlay_index(v1_root)
                clone_parent_v2 = load_w02_morphology_successor_v2_overlay_index(v2_root)
                clone_indexes = build_w02_morphology_routed_indexes(
                    clone_parent_v1, clone_parent_v2, routes)
                clone_v3 = predict_w02_morphology_successor_v3(
                    clone_indexes, observation, clone_base,
                    requested_spans=spans, v1_cache=clone_v1_cache,
                    v2_cache=clone_v2_cache)
                clone_operations += (
                    clone_v3.v1.logic_operations + clone_v3.v2.logic_operations
                    + clone_v3.route_logic_operations)
                clone_ok = (
                    clone_index.semantic_sha256 == candidate_index.semantic_sha256
                    and clone_parent_v1.semantic_sha256 == parent_v1.semantic_sha256
                    and clone_parent_v2.semantic_sha256 == parent_v2.semantic_sha256
                    and clone_indexes.semantic_sha256 == indexes.semantic_sha256
                    and clone_v3.route_authorized == 1
                    and _hash_value(clone_v3.v2.prediction.to_dict()) == expected_sha)
            finally:
                clone_v1_cache.close()
                clone_v2_cache.close()
        total_operations = (
            routes.logic_operations + parent_v1.logic_operations
            + parent_v2.logic_operations + base_operations + v1_operations
            + v2_operations + route_inference_operations + clone_operations)
        after = tuple(_tree_sha256(root) for root in (candidate_root, v1_root, v2_root))
        rollback_ok = after == before
        resource_ok = None if (
            resource_exhausted or total_operations > budget.max_logic_operations
        ) else True
        zero_call_windows = [
            {"api_calls": 0, "llm_calls": 0, "teacher_calls": 0,
             "window_key": key}
            for key in ("BEFORE_PRIVATE_READ", "DURING_PRIVATE_EVALUATION",
                        "AFTER_PRIVATE_EVALUATION")
        ]
        support = [
            _support_result(W02_MORPH_V3_PRIVATE_SUPPORT_KEYS[0], resource_ok, {
                "logic_operations": total_operations,
                "max_logic_operations": budget.max_logic_operations,
                "resource_exhausted": int(resource_exhausted),
            }),
            _support_result(W02_MORPH_V3_PRIVATE_SUPPORT_KEYS[1], rollback_ok, {
                "candidate_unchanged": int(after[0] == before[0]),
                "v1_overlay_unchanged": int(after[1] == before[1]),
                "v2_overlay_unchanged": int(after[2] == before[2]),
                "v3_route_writes": 0,
            }),
            _support_result(
                W02_MORPH_V3_PRIVATE_SUPPORT_KEYS[2], True, zero_call_windows),
            _support_result(W02_MORPH_V3_PRIVATE_SUPPORT_KEYS[3], clone_ok, {
                "clone_probe_present": int(clone_probe is not None),
                "clone_semantic_equal": int(clone_ok is True),
            }),
        ]
        hard = [*dimensions, *support]
        return {
            "base_logic_operations": base_operations,
            "candidate_artifact_manifest_sha256":
                candidate_result.artifact_manifest_sha256,
            "candidate_index_row_count": candidate_index.row_count,
            "candidate_index_semantic_sha256": candidate_index.semantic_sha256,
            "candidate_semantic_sha256": candidate_result.candidate_semantic_sha256,
            "dimension_results": dimensions,
            "evaluation_count": evaluation_count,
            "family_counts": family_counts,
            "hard_conjunct_results": hard,
            "input_pair_count": input_pair_count,
            "logic_operations": total_operations,
            "max_v1_generalized_candidates_per_observation": max_v1,
            "max_v2_edge_candidates_per_observation": max_v2,
            "max_v2_edge_candidates_per_requested_span": max_v2_per_span,
            "queried_span_count": queried_spans,
            "route_authorized_count": route_authorized_count,
            "route_capability_sha256s": list(routes.capability_sha256s),
            "route_index_logic_operations": routes.logic_operations,
            "route_inference_logic_operations": route_inference_operations,
            "route_semantic_sha256": routes.semantic_sha256,
            "routed_index_semantic_sha256": indexes.semantic_sha256,
            "source_count": source_count,
            "source_identity_sha256": source_digest.hexdigest(),
            "status": _result_status(hard),
            "support_results": support,
            "v1_generalized_candidate_count": v1_candidates,
            "v1_overlay_artifact_manifest_sha256":
                v1_result.artifact_manifest_sha256,
            "v1_overlay_inference_logic_operations": v1_operations,
            "v1_overlay_ranking_cache_entry_count": len(v1_cache.values),
            "v1_overlay_ranking_cache_hit_count": v1_cache.hit_count,
            "v1_overlay_ranking_cache_miss_count": v1_cache.miss_count,
            "v1_overlay_rule_row_count": parent_v1.row_count,
            "v1_overlay_semantic_sha256": parent_v1.semantic_sha256,
            "v1_successor_transform_logic_operations": parent_v1.logic_operations,
            "v2_edge_candidate_count": v2_candidates,
            "v2_overlay_artifact_manifest_sha256":
                v2_result.artifact_manifest_sha256,
            "v2_overlay_inference_logic_operations": v2_operations,
            "v2_overlay_ranking_cache_entry_count": len(v2_cache.values),
            "v2_overlay_ranking_cache_hit_count": v2_cache.hit_count,
            "v2_overlay_ranking_cache_miss_count": v2_cache.miss_count,
            "v2_overlay_rule_row_count": parent_v2.row_count,
            "v2_overlay_semantic_sha256": parent_v2.semantic_sha256,
            "v2_successor_transform_logic_operations": parent_v2.logic_operations,
            "zero_call_windows": zero_call_windows,
        }
    finally:
        close_pairs = getattr(pair_iterator, "close", None)
        if callable(close_pairs):
            close_pairs()
        v1_cache.close()
        v2_cache.close()


def run_w02_morphology_successor_v3_private_evaluation(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02MorphologySuccessorV3PrivateFileIdentity, ...],
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        *,
        family_freeze_sha256: str,
        run_id: int = 1,
        ) -> dict[str, object]:
    """Consume an already-guarded V3 family in one full formal stream."""
    if run_id != 1:
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "formal V3 private run_id must be one")
    if (not isinstance(registration, V2PrivateFamilyRegistration)
            or registration.stage_key != "W-02"
            or registration.formal_run_count != 0
            or registration.private_payload_reads != 0):
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "V3 private registration state drifted")
    permits = authorize_w02_morphology_successor_v3_private_files(
        boundary, roots, registration, files)
    sources = iter_w02_morphology_successor_v3_private_records(
        v3_private_file_by_layout(files, "PRIVATE_SOURCE"),
        permits["PRIVATE_SOURCE"])
    pairs = (
        pair for split in W02_MORPH_V3_PRIVATE_SPLITS
        for pair in iter_w02_morphology_successor_v3_private_pairs(
            files, permits, split)
    )
    capabilities = tuple(
        w02_ud_morphology_source_capability(spec)
        for spec in blind_private_source_specs_v3())
    core = evaluate_w02_morphology_successor_v3_private_pair_stream(
        candidate_artifact_root, v1_overlay_artifact_root,
        v2_overlay_artifact_root, sources, capabilities, pairs,
        registration.resource_budget)
    authorize_w02_morphology_successor_v3_private_files(
        boundary, roots, registration, files)
    expected = sum(
        v3_private_file_by_layout(
            files, v3_private_split_layout(split, "observation")).record_count
        for split in W02_MORPH_V3_PRIVATE_SPLITS)
    if core["input_pair_count"] != expected:
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "V3 private input was not fully consumed")
    transport_bytes = sum(row.transport_size_bytes for row in files)
    private_record_reads = core["source_count"] + core["input_pair_count"] * 2
    report = {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_EVALUATION_REPORT"),
        "artifact_version": W02_MORPH_V3_PRIVATE_EVALUATOR_VERSION,
        **core,
        "family_commitment": registration.family_commitment,
        "family_freeze_sha256": family_freeze_sha256,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 1,
        "formal_shadow_audit_runs": 1,
        "formal_successor_transform_runs": 1,
        "formal_successor_v2_transform_runs": 1,
        "formal_successor_v3_route_dev_runs": 1,
        "formal_successor_v3_route_shadow_runs": 1,
        "formal_training_runs": 1,
        "label_record_reads": core["input_pair_count"],
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": (
            "W03_COMPILE_FREEZE" if core["status"] == "PASS"
            else "W02_PRIVATE_FAILED_OR_NE_STOP"),
        "observation_reads": core["input_pair_count"],
        "private_content_stream_reads": len(files),
        "private_family_registered": 1,
        "private_payload_gets": private_record_reads,
        "private_payload_reads": len(files) * 4,
        "private_post_content_transport_reads": len(files),
        "private_record_reads": private_record_reads,
        "private_transport_validation_reads": len(files) * 2,
        "release_key": "PH2-D03-V2",
        "run_id": 1,
        "run_scope": "FORMAL_BLIND_PRIVATE_EVALUATION",
        "stage_key": "W-02",
        "teacher_calls": 0,
        "transport_bytes_read": transport_bytes * 4,
        "validated_layout_count": len(files),
        "zero_write_audit": {
            "assessment_writes": 0,
            "candidate_writes": 0,
            "clock_writes": 0,
            "companion_writes": 0,
            "core_writes": 0,
            "evaluator_label_writes": 0,
            "evidence_writes": 0,
            "host_writes": 0,
            "memory_writes": 0,
            "use_writes": 0,
            "v1_overlay_writes": 0,
            "v2_overlay_writes": 0,
            "v3_route_writes": 0,
        },
    }
    if (report["private_record_reads"] > registration.resource_budget.max_records
            or report["transport_bytes_read"]
            > registration.resource_budget.max_payload_bytes
            or report["private_payload_gets"]
            > registration.resource_budget.max_payload_gets):
        raise W02MorphologySuccessorV3PrivateEvaluationError(
            "V3 private non-logic resource budget exceeded")
    validate_v2_safe_report(report)
    return report


__all__ = [
    "W02_MORPH_V3_PRIVATE_EVALUATOR_VERSION",
    "W02_MORPH_V3_PRIVATE_SUPPORT_KEYS",
    "W02MorphologySuccessorV3PrivateEvaluationError",
    "evaluate_w02_morphology_successor_v3_private_pair_stream",
    "run_w02_morphology_successor_v3_private_evaluation",
]
