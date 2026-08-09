"""Blind private evaluator for the Candidate -> V1 -> V2 morphology chain."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Iterable, Iterator

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension import (
    validate_blind_private_owner_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorBoundaryContract,
    V2EvaluatorResourceBudget,
    V2PrivateFamilyRegistration,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2AccessPermit,
    V2AccessRequest,
    V2PhysicalRoots,
    V2WriteAccount,
    authorize_v2_access,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import _dimension_key
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import W02FileFreeze
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
    predict_w02_morphology_successor,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_dev_calibration import (
    _requested_spans,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN,
    W02MorphologySuccessorV2Cache,
    predict_w02_morphology_successor_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_private_owner import (
    W02_MORPH_V2_PRIVATE_LAYOUTS,
    W02_MORPH_V2_PRIVATE_PATHS,
    W02_MORPH_V2_PRIVATE_SPLITS,
)


W02_MORPH_V2_PRIVATE_EVALUATOR_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-PRIVATE-EVALUATOR-V1"
)
W02_MORPH_V2_PRIVATE_SUPPORT_KEYS = (
    "W-02-V2-RESOURCE",
    "W-02-V2-ROLLBACK",
    "W-02-V2-ZERO-CALL-WINDOWS",
    "W-02-V2-V06-CLONE",
)


# object-model: exception
class W02MorphologySuccessorV2PrivateEvaluationError(RuntimeError):
    """Private input, artifact chain, resource, or result contract drifted."""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _file_by_layout(
        files: tuple[W02FileFreeze, ...], layout_key: str,
        ) -> W02FileFreeze:
    matches = tuple(row for row in files if row.layout_key == layout_key)
    if len(matches) != 1 or matches[0].root_key != "PRIVATE_EVALUATOR_ROOT":
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private owner layout is not unique")
    return matches[0]


def _split_layout(split: str, kind: str) -> str:
    if split not in W02_MORPH_V2_PRIVATE_SPLITS or kind not in {
            "observation", "label"}:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private split/kind is not registered")
    return f"PRIVATE_{split.upper()}_{'OBSERVATION' if kind == 'observation' else 'LABEL'}"


def authorize_w02_morphology_successor_v2_private_files(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02FileFreeze, ...],
        ) -> dict[str, V2AccessPermit]:
    """Authorize the seven owner files by public transport identity only."""
    if tuple(row.layout_key for row in files) != W02_MORPH_V2_PRIVATE_LAYOUTS:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private file inventory drifted")
    permits = {}
    for layout_key in W02_MORPH_V2_PRIVATE_LAYOUTS:
        identity = _file_by_layout(files, layout_key)
        split = identity.split or "held_out"
        request = V2AccessRequest(
            "W-02", "PH2_V2_PRIVATE_EVALUATOR", split,
            identity.record_kind, W02_MORPH_V2_PRIVATE_PATHS[layout_key],
            identity.transport_sha256, identity.transport_size_bytes,
            "PRIVATE_EVALUATION", registration.candidate_freeze_sha256,
            registration.code_freeze_sha256, V2WriteAccount())
        permits[layout_key] = authorize_v2_access(
            boundary, roots, request, registration=registration)
    return permits


def iter_w02_morphology_successor_v2_private_records(
        identity: W02FileFreeze,
        permit: V2AccessPermit,
        ) -> Iterator[object]:
    """Read one authorized gzip and close content plus transport identity."""
    if (not isinstance(identity, W02FileFreeze)
            or not isinstance(permit, V2AccessPermit)
            or permit.root_key != "PRIVATE_EVALUATOR_ROOT"
            or permit.record_kind != identity.record_kind
            or permit.content_sha256 != identity.transport_sha256
            or permit.content_size_bytes != identity.transport_size_bytes):
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private permit does not match frozen identity")
    content_digest = hashlib.sha256()
    content_size = 0
    count = 0
    first_key = None
    last_key = None
    previous_key = None
    try:
        with permit.target_path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise W02MorphologySuccessorV2PrivateEvaluationError(
                            f"private JSONL line {line_number} newline drifted")
                    content_digest.update(line)
                    content_size += len(line)
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = validate_blind_private_owner_record(value)
                    if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                        raise W02MorphologySuccessorV2PrivateEvaluationError(
                            "private record kind drifted")
                    if (identity.split
                            and getattr(record, "split", identity.split)
                            != identity.split):
                        raise W02MorphologySuccessorV2PrivateEvaluationError(
                            "private record split drifted")
                    key = record.stable_key.components
                    if previous_key is not None and key <= previous_key:
                        raise W02MorphologySuccessorV2PrivateEvaluationError(
                            "private stable keys are not strictly ordered")
                    previous_key = key
                    first_key = key if first_key is None else first_key
                    last_key = key
                    count += 1
                    yield record
    except W02MorphologySuccessorV2PrivateEvaluationError:
        raise
    except (OSError, EOFError, ValueError) as error:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private gzip/JSONL read failed") from error
    if (count != identity.record_count
            or content_size != identity.content_size_bytes
            or content_digest.hexdigest() != identity.content_sha256
            or first_key != identity.first_record_key
            or last_key != identity.last_record_key):
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private content identity drifted")
    size, digest = _sha256_file(permit.target_path)
    if size != identity.transport_size_bytes or digest != identity.transport_sha256:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private transport drifted during content read")


def iter_w02_morphology_successor_v2_private_pairs(
        files: tuple[W02FileFreeze, ...],
        permits: dict[str, V2AccessPermit],
        split: str,
        ) -> Iterator[tuple[ObservationRecord, EvaluatorLabelRecord]]:
    """Pair observation and label streams without buffering private content."""
    observation_key = _split_layout(split, "observation")
    label_key = _split_layout(split, "label")
    observations = iter_w02_morphology_successor_v2_private_records(
        _file_by_layout(files, observation_key), permits[observation_key])
    labels = iter_w02_morphology_successor_v2_private_records(
        _file_by_layout(files, label_key), permits[label_key])
    count = 0
    for observation, evaluation in zip(observations, labels, strict=True):
        if (not isinstance(observation, ObservationRecord)
                or not isinstance(evaluation, EvaluatorLabelRecord)
                or observation.split != split
                or evaluation.observation_key != observation.stable_key
                or evaluation.visible_stage != "W-02"
                or evaluation.owner_mode != "read_only"):
            raise W02MorphologySuccessorV2PrivateEvaluationError(
                "private pair owner/binding drifted")
        count += 1
        yield observation, evaluation
    if count != _file_by_layout(files, observation_key).record_count:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private pair count drifted")


def _morphology_identity(value: object) -> tuple[object, ...]:
    return tuple(getattr(value, name) for name in (
        "start", "end", "form", "lemma", "upos", "feats_json"))


def _audit_v2_extension(v1: object, v2: object, spans: tuple[tuple[int, int], ...]) -> int:
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
            raise W02MorphologySuccessorV2PrivateEvaluationError(
                "V2 candidate escaped requested spans")
        counts[span] += 1
    maximum = max(counts.values(), default=0)
    if (not before.issubset(after)
            or len(added) != v2.edge_candidate_count
            or maximum > W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN):
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "V2 extension retention/bound drifted")
    return maximum


def _support_result(key: str, state: bool | None, evidence: object) -> dict[str, object]:
    if key not in W02_MORPH_V2_PRIVATE_SUPPORT_KEYS:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private support key drifted")
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


def evaluate_w02_morphology_successor_v2_private_pair_stream(
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        pairs: Iterable[tuple[ObservationRecord, EvaluatorLabelRecord]],
        budget: V2EvaluatorResourceBudget,
        ) -> dict[str, object]:
    """Evaluate public synthetic or authorized private pairs through all layers."""
    if not isinstance(budget, V2EvaluatorResourceBudget):
        raise TypeError("private V2 budget type drifted")
    candidate_root = Path(candidate_artifact_root).resolve()
    v1_root = Path(v1_overlay_artifact_root).resolve()
    v2_root = Path(v2_overlay_artifact_root).resolve()
    before = tuple(_tree_sha256(root) for root in (candidate_root, v1_root, v2_root))
    candidate_result = read_w02_candidate_artifact(candidate_root)
    v1_result = read_w02_morphology_overlay_artifact(v1_root)
    v2_result = read_w02_morphology_successor_v2_overlay_artifact(v2_root)
    if v1_result.parent_candidate_semantic_sha256 != candidate_result.candidate_semantic_sha256:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "Candidate/V1 parent identity drifted")
    v1_index = load_w02_morphology_overlay_index(v1_root)
    v2_index = load_w02_morphology_successor_v2_overlay_index(v2_root)
    if v1_index.dataset_keys != v2_index.dataset_keys:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "V1/V2 route identity drifted")
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
    queried_spans = 0
    v1_candidates = 0
    v2_candidates = 0
    max_v1 = 0
    max_v2 = 0
    max_v2_per_span = 0
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
                    raise W02MorphologySuccessorV2PrivateEvaluationError(
                        "private pair type drifted")
                base, operations = predict_w02_dev_observation(
                    candidate_index, observation)
                spans = _requested_spans(evaluation)
                v1 = predict_w02_morphology_successor(
                    v1_index, observation, base, requested_spans=spans,
                    ranking_cache=v1_cache)
                v2 = predict_w02_morphology_successor_v2(
                    v2_index, observation, v1, requested_spans=spans,
                    cache=v2_cache)
                maximum = _audit_v2_extension(v1, v2, spans)
                projected = (
                    v1_index.logic_operations + v2_index.logic_operations
                    + base_operations + operations + 8 + v1_operations
                    + v1.logic_operations + v2_operations + v2.logic_operations)
                if projected > budget.max_logic_operations:
                    resource_exhausted = True
                    break
                dimension, family, passed, evidence_sha = _evaluate_pair(
                    observation, evaluation, v2.prediction, dimension_by_key)
                if dimension not in rows or family not in family_counts:
                    raise W02MorphologySuccessorV2PrivateEvaluationError(
                        "private dimension/family drifted")
                rows[dimension].append((passed, evidence_sha))
                family_counts[family] += 1
                base_operations += operations + 8
                v1_operations += v1.logic_operations
                v2_operations += v2.logic_operations
                queried_spans += len(spans)
                v1_candidates += v1.generalized_candidate_count
                v2_candidates += v2.edge_candidate_count
                max_v1 = max(max_v1, v1.generalized_candidate_count)
                max_v2 = max(max_v2, v2.edge_candidate_count)
                max_v2_per_span = max(max_v2_per_span, maximum)
                evaluation_count += 1
                if clone_probe is None:
                    clone_probe = (
                        observation, spans, _hash_value(v2.prediction.to_dict()))
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
                clone_v1_index = load_w02_morphology_overlay_index(v1_root)
                clone_v2_index = load_w02_morphology_successor_v2_overlay_index(v2_root)
                clone_v1 = predict_w02_morphology_successor(
                    clone_v1_index, observation, clone_base,
                    requested_spans=spans, ranking_cache=clone_v1_cache)
                clone_v2 = predict_w02_morphology_successor_v2(
                    clone_v2_index, observation, clone_v1,
                    requested_spans=spans, cache=clone_v2_cache)
                clone_operations += clone_v1.logic_operations + clone_v2.logic_operations
                clone_ok = (
                    clone_index.semantic_sha256 == candidate_index.semantic_sha256
                    and clone_v1_index.semantic_sha256 == v1_index.semantic_sha256
                    and clone_v2_index.semantic_sha256 == v2_index.semantic_sha256
                    and _hash_value(clone_v2.prediction.to_dict()) == expected_sha)
            finally:
                clone_v1_cache.close()
                clone_v2_cache.close()
        total_operations = (
            v1_index.logic_operations + v2_index.logic_operations
            + base_operations + v1_operations + v2_operations + clone_operations)
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
            _support_result(W02_MORPH_V2_PRIVATE_SUPPORT_KEYS[0], resource_ok, {
                "logic_operations": total_operations,
                "max_logic_operations": budget.max_logic_operations,
                "resource_exhausted": int(resource_exhausted),
            }),
            _support_result(W02_MORPH_V2_PRIVATE_SUPPORT_KEYS[1], rollback_ok, {
                "candidate_unchanged": int(after[0] == before[0]),
                "v1_overlay_unchanged": int(after[1] == before[1]),
                "v2_overlay_unchanged": int(after[2] == before[2]),
            }),
            _support_result(
                W02_MORPH_V2_PRIVATE_SUPPORT_KEYS[2], True, zero_call_windows),
            _support_result(W02_MORPH_V2_PRIVATE_SUPPORT_KEYS[3], clone_ok, {
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
            "status": _result_status(hard),
            "support_results": support,
            "v1_generalized_candidate_count": v1_candidates,
            "v1_overlay_artifact_manifest_sha256":
                v1_result.artifact_manifest_sha256,
            "v1_overlay_inference_logic_operations": v1_operations,
            "v1_overlay_ranking_cache_entry_count": len(v1_cache.values),
            "v1_overlay_ranking_cache_hit_count": v1_cache.hit_count,
            "v1_overlay_ranking_cache_miss_count": v1_cache.miss_count,
            "v1_overlay_rule_row_count": v1_index.row_count,
            "v1_overlay_semantic_sha256": v1_index.semantic_sha256,
            "v1_successor_transform_logic_operations": v1_index.logic_operations,
            "v2_edge_candidate_count": v2_candidates,
            "v2_overlay_artifact_manifest_sha256":
                v2_result.artifact_manifest_sha256,
            "v2_overlay_inference_logic_operations": v2_operations,
            "v2_overlay_ranking_cache_entry_count": len(v2_cache.values),
            "v2_overlay_ranking_cache_hit_count": v2_cache.hit_count,
            "v2_overlay_ranking_cache_miss_count": v2_cache.miss_count,
            "v2_overlay_rule_row_count": v2_index.row_count,
            "v2_overlay_semantic_sha256": v2_index.semantic_sha256,
            "v2_successor_transform_logic_operations": v2_index.logic_operations,
            "zero_call_windows": zero_call_windows,
        }
    finally:
        close_pairs = getattr(pair_iterator, "close", None)
        if callable(close_pairs):
            close_pairs()
        v1_cache.close()
        v2_cache.close()


def run_w02_morphology_successor_v2_private_evaluation(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02FileFreeze, ...],
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        *,
        family_freeze_sha256: str,
        run_id: int = 1,
        ) -> dict[str, object]:
    """Consume an already-guarded family in one full formal stream."""
    if run_id != 1:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "formal private run_id must be one")
    if (not isinstance(registration, V2PrivateFamilyRegistration)
            or registration.stage_key != "W-02"
            or registration.formal_run_count != 0
            or registration.private_payload_reads != 0):
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private registration state drifted")
    permits = authorize_w02_morphology_successor_v2_private_files(
        boundary, roots, registration, files)
    source_count = 0
    source_digest = hashlib.sha256()
    for record in iter_w02_morphology_successor_v2_private_records(
            _file_by_layout(files, "PRIVATE_SOURCE"), permits["PRIVATE_SOURCE"]):
        if not isinstance(record, SourceRefRecord):
            raise W02MorphologySuccessorV2PrivateEvaluationError(
                "private SourceRef type drifted")
        source_digest.update(canonical_json_bytes(record.stable_key.to_list()))
        source_count += 1
    pairs = (
        pair for split in W02_MORPH_V2_PRIVATE_SPLITS
        for pair in iter_w02_morphology_successor_v2_private_pairs(
            files, permits, split)
    )
    core = evaluate_w02_morphology_successor_v2_private_pair_stream(
        candidate_artifact_root, v1_overlay_artifact_root,
        v2_overlay_artifact_root, pairs, registration.resource_budget)
    authorize_w02_morphology_successor_v2_private_files(
        boundary, roots, registration, files)
    expected = sum(
        _file_by_layout(files, _split_layout(split, "observation")).record_count
        for split in W02_MORPH_V2_PRIVATE_SPLITS)
    if core["input_pair_count"] != expected:
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private input was not fully consumed")
    transport_bytes = sum(row.transport_size_bytes for row in files)
    private_record_reads = source_count + core["input_pair_count"] * 2
    report = {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_PRIVATE_EVALUATION_REPORT"),
        "artifact_version": W02_MORPH_V2_PRIVATE_EVALUATOR_VERSION,
        **core,
        "family_commitment": registration.family_commitment,
        "family_freeze_sha256": family_freeze_sha256,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 1,
        "formal_shadow_audit_attempts": 2,
        "formal_shadow_audit_passes": 1,
        "formal_shadow_audit_runs": 1,
        "formal_shadow_recovery_runs": 1,
        "formal_successor_transform_runs": 1,
        "formal_successor_v2_transform_runs": 1,
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
        "source_count": source_count,
        "source_identity_sha256": source_digest.hexdigest(),
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
        },
    }
    if (report["private_record_reads"] > registration.resource_budget.max_records
            or report["transport_bytes_read"]
            > registration.resource_budget.max_payload_bytes
            or report["private_payload_gets"]
            > registration.resource_budget.max_payload_gets):
        raise W02MorphologySuccessorV2PrivateEvaluationError(
            "private non-logic resource budget exceeded")
    validate_v2_safe_report(report)
    return report


__all__ = [
    "W02_MORPH_V2_PRIVATE_EVALUATOR_VERSION",
    "W02_MORPH_V2_PRIVATE_SUPPORT_KEYS",
    "W02MorphologySuccessorV2PrivateEvaluationError",
    "authorize_w02_morphology_successor_v2_private_files",
    "evaluate_w02_morphology_successor_v2_private_pair_stream",
    "iter_w02_morphology_successor_v2_private_pairs",
    "iter_w02_morphology_successor_v2_private_records",
    "run_w02_morphology_successor_v2_private_evaluation",
]
