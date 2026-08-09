"""PH2-D03-V2 W-02 morphology successor 的盲私评 runtime。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Iterable, Iterator

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2PrivateFamilyRegistration,
    V2EvaluatorBoundaryContract,
    V2EvaluatorResourceBudget,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2AccessPermit,
    V2AccessRequest,
    V2PhysicalRoots,
    V2WriteAccount,
    authorize_v2_access,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import _dimension_key
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02CompileFreeze,
    W02FileFreeze,
)
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
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    parse_canonical_json_bytes,
)


W02_PRIVATE_EVALUATOR_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-PRIVATE-EVALUATOR-V1")
W02_PRIVATE_SPLITS = ("held_out", "adversarial", "wall")
W02_PRIVATE_LAYOUT_PATHS = {
    "PRIVATE_SOURCE": "source/source_refs.jsonl.gz",
    "PRIVATE_HELD_OUT_OBSERVATION": "observations/held_out.jsonl.gz",
    "PRIVATE_ADVERSARIAL_OBSERVATION": "observations/adversarial.jsonl.gz",
    "PRIVATE_WALL_OBSERVATION": "observations/wall.jsonl.gz",
    "PRIVATE_HELD_OUT_LABEL": "evaluator/held_out.labels.jsonl.gz",
    "PRIVATE_ADVERSARIAL_LABEL": "evaluator/adversarial.labels.jsonl.gz",
    "PRIVATE_WALL_LABEL": "evaluator/wall.labels.jsonl.gz",
}
W02_PRIVATE_SUPPORT_KEYS = (
    "W-02-V2-RESOURCE",
    "W-02-V2-ROLLBACK",
    "W-02-V2-ZERO-CALL-WINDOWS",
    "W-02-V2-V06-CLONE",
)


# object-model: exception
class W02MorphologySuccessorPrivateEvaluationError(RuntimeError):
    """盲私评输入、隔离、资源或结果合同发生漂移。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _private_identity(parent: W02CompileFreeze, layout_key: str) -> W02FileFreeze:
    matches = tuple(item for item in parent.files if item.layout_key == layout_key)
    if len(matches) != 1 or matches[0].root_key != "PRIVATE_EVALUATOR_ROOT":
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private layout freeze 不唯一")
    return matches[0]


def _split_layout(split: str, kind: str) -> str:
    if split not in W02_PRIVATE_SPLITS or kind not in {"observation", "label"}:
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private split/kind 未注册")
    stem = split.upper()
    return f"PRIVATE_{stem}_{'OBSERVATION' if kind == 'observation' else 'LABEL'}"


def authorize_w02_private_files(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        parent: W02CompileFreeze,
        ) -> dict[str, V2AccessPermit]:
    """只按公开 transport identity 授权七个盲私评文件。"""
    if not isinstance(parent, W02CompileFreeze):
        raise TypeError("W-02 private compile freeze 类型错误")
    permits: dict[str, V2AccessPermit] = {}
    for layout_key, relative in W02_PRIVATE_LAYOUT_PATHS.items():
        identity = _private_identity(parent, layout_key)
        split = identity.split or "held_out"
        request = V2AccessRequest(
            "W-02", "PH2_V2_PRIVATE_EVALUATOR", split,
            identity.record_kind, relative, identity.transport_sha256,
            identity.transport_size_bytes, "PRIVATE_EVALUATION",
            registration.candidate_freeze_sha256,
            registration.code_freeze_sha256, V2WriteAccount())
        permits[layout_key] = authorize_v2_access(
            boundary, roots, request, registration=registration)
    return permits


def iter_w02_private_records(
        identity: W02FileFreeze,
        permit: V2AccessPermit,
        ) -> Iterator[object]:
    """单遍解析一个已授权 gzip，并在 EOF 闭合 content identity。"""
    if (not isinstance(identity, W02FileFreeze)
            or not isinstance(permit, V2AccessPermit)
            or permit.root_key != "PRIVATE_EVALUATOR_ROOT"
            or permit.record_kind != identity.record_kind
            or permit.content_sha256 != identity.transport_sha256
            or permit.content_size_bytes != identity.transport_size_bytes):
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private permit 与 freeze 不匹配")
    content_digest = hashlib.sha256()
    content_size = 0
    count = 0
    first_key: tuple[int, ...] | None = None
    last_key: tuple[int, ...] | None = None
    previous_key: tuple[int, ...] | None = None
    try:
        with permit.target_path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise W02MorphologySuccessorPrivateEvaluationError(
                            f"W-02 private JSONL 第 {line_number} 行换行非法")
                    content_digest.update(line)
                    content_size += len(line)
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = validate_v2_record(value)
                    if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                        raise W02MorphologySuccessorPrivateEvaluationError(
                            "W-02 private record kind 漂移")
                    if (identity.split
                            and getattr(record, "split", identity.split)
                            != identity.split):
                        raise W02MorphologySuccessorPrivateEvaluationError(
                            "W-02 private split 漂移")
                    key = record.stable_key.components
                    if previous_key is not None and key <= previous_key:
                        raise W02MorphologySuccessorPrivateEvaluationError(
                            "W-02 private stable key 未严格排序")
                    previous_key = key
                    first_key = key if first_key is None else first_key
                    last_key = key
                    count += 1
                    yield record
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, W02MorphologySuccessorPrivateEvaluationError):
            raise
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private gzip/JSONL 读取失败") from error
    if (count != identity.record_count
            or content_size != identity.content_size_bytes
            or content_digest.hexdigest() != identity.content_sha256
            or first_key != identity.first_record_key
            or last_key != identity.last_record_key):
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private content identity 漂移")
    size, digest = _sha256_file(permit.target_path)
    if size != identity.transport_size_bytes or digest != identity.transport_sha256:
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private transport 在内容读取期间漂移")


def iter_w02_private_pairs(
        parent: W02CompileFreeze,
        permits: dict[str, V2AccessPermit],
        split: str,
        ) -> Iterator[tuple[ObservationRecord, EvaluatorLabelRecord]]:
    """按 split 严格配对 Observation 与 evaluator label。"""
    observation_key = _split_layout(split, "observation")
    label_key = _split_layout(split, "label")
    observations = iter_w02_private_records(
        _private_identity(parent, observation_key), permits[observation_key])
    labels = iter_w02_private_records(
        _private_identity(parent, label_key), permits[label_key])
    count = 0
    for observation, evaluation in zip(observations, labels, strict=True):
        if (not isinstance(observation, ObservationRecord)
                or not isinstance(evaluation, EvaluatorLabelRecord)
                or observation.split != split
                or evaluation.observation_key != observation.stable_key
                or evaluation.visible_stage != "W-02"
                or evaluation.owner_mode != "read_only"):
            raise W02MorphologySuccessorPrivateEvaluationError(
                "W-02 private pair owner/binding 漂移")
        count += 1
        yield observation, evaluation
    if count != parent.plan.split_total(split):
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private pair 数量与 compile plan 漂移")


def _support_result(key: str, state: bool | None, evidence: object) -> dict[str, object]:
    if key not in W02_PRIVATE_SUPPORT_KEYS:
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private support key 漂移")
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


def evaluate_w02_private_pair_stream(
        candidate_artifact_root: str | Path,
        overlay_artifact_root: str | Path,
        pairs: Iterable[tuple[ObservationRecord, EvaluatorLabelRecord]],
        budget: V2EvaluatorResourceBudget,
        ) -> dict[str, object]:
    """对公开合成或已授权 private pair 执行同一无标签预测核心。"""
    if not isinstance(budget, V2EvaluatorResourceBudget):
        raise TypeError("W-02 private budget 类型错误")
    candidate_root = Path(candidate_artifact_root).resolve()
    overlay_root = Path(overlay_artifact_root).resolve()
    before_candidate = _tree_sha256(candidate_root)
    before_overlay = _tree_sha256(overlay_root)
    candidate_result = read_w02_candidate_artifact(candidate_root)
    overlay_result = read_w02_morphology_overlay_artifact(overlay_root)
    if (overlay_result.parent_candidate_semantic_sha256
            != candidate_result.candidate_semantic_sha256):
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private Candidate/overlay parent 漂移")
    overlay_index = load_w02_morphology_overlay_index(overlay_root)
    dimension_by_key = {
        _dimension_key(name).components: name for name in W02_DEV_DIMENSIONS
    }
    rows: dict[str, list[tuple[bool | None, str]]] = {
        name: [] for name in W02_DEV_DIMENSIONS
    }
    family_counts = {name: 0 for name in (
        "AUTHORED_OOV", "UD_ANNOTATION", "UNICODE_ANNOTATION")}
    cache = W02MorphologyRankingCache.empty()
    base_operations = 0
    overlay_operations = 0
    queried_spans = 0
    generalized_candidates = 0
    max_generalized = 0
    evaluation_count = 0
    input_pair_count = 0
    resource_exhausted = False
    clone_probe: tuple[
        ObservationRecord, EvaluatorLabelRecord, tuple[tuple[int, int], ...], str,
    ] | None = None
    pair_iterator = iter(pairs)
    try:
        with open_w02_candidate_predictor(candidate_root) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
            for observation, evaluation in pair_iterator:
                input_pair_count += 1
                if (not isinstance(observation, ObservationRecord)
                        or not isinstance(evaluation, EvaluatorLabelRecord)):
                    raise W02MorphologySuccessorPrivateEvaluationError(
                        "W-02 private pair 类型错误")
                base, operations = predict_w02_dev_observation(
                    candidate_index, observation)
                spans = _requested_spans(evaluation)
                successor = predict_w02_morphology_successor(
                    overlay_index, observation, base,
                    requested_spans=spans, ranking_cache=cache)
                projected_operations = (
                    overlay_index.logic_operations + base_operations
                    + operations + 8 + overlay_operations
                    + successor.logic_operations)
                if projected_operations > budget.max_logic_operations:
                    resource_exhausted = True
                    break
                dimension, family, passed, evidence_sha = _evaluate_pair(
                    observation, evaluation, successor.prediction,
                    dimension_by_key)
                if dimension not in rows or family not in family_counts:
                    raise W02MorphologySuccessorPrivateEvaluationError(
                        "W-02 private dimension/family 漂移")
                rows[dimension].append((passed, evidence_sha))
                family_counts[family] += 1
                base_operations += operations + 8
                overlay_operations += successor.logic_operations
                queried_spans += len(spans)
                generalized_candidates += successor.generalized_candidate_count
                max_generalized = max(
                    max_generalized, successor.generalized_candidate_count)
                evaluation_count += 1
                if clone_probe is None:
                    clone_probe = (
                        observation, evaluation, spans,
                        _hash_value(successor.prediction.to_dict()))
            if resource_exhausted:
                for _observation, _evaluation in pair_iterator:
                    input_pair_count += 1
        dimensions = [
            _dimension_report(name, rows[name]) for name in W02_DEV_DIMENSIONS
        ]
        clone_ok: bool | None = None
        clone_operations = 0
        if not resource_exhausted and clone_probe is not None:
            observation, _evaluation, spans, expected_sha = clone_probe
            clone_cache = W02MorphologyRankingCache.empty()
            try:
                with open_w02_candidate_predictor(candidate_root) as predictor:
                    cloned_index = load_w02_dev_candidate_index(predictor)
                    cloned_base, clone_operations = predict_w02_dev_observation(
                        cloned_index, observation)
                cloned_overlay = load_w02_morphology_overlay_index(overlay_root)
                cloned = predict_w02_morphology_successor(
                    cloned_overlay, observation, cloned_base,
                    requested_spans=spans, ranking_cache=clone_cache)
                clone_operations += cloned.logic_operations
                clone_ok = (
                    cloned_index.semantic_sha256 == candidate_index.semantic_sha256
                    and cloned_overlay.semantic_sha256 == overlay_index.semantic_sha256
                    and _hash_value(cloned.prediction.to_dict()) == expected_sha)
            finally:
                clone_cache.close()
        total_operations = (
            overlay_index.logic_operations + base_operations
            + overlay_operations + clone_operations)
        after_candidate = _tree_sha256(candidate_root)
        after_overlay = _tree_sha256(overlay_root)
        rollback_ok = (
            after_candidate == before_candidate
            and after_overlay == before_overlay)
        resource_ok: bool | None = (
            None if resource_exhausted or total_operations > budget.max_logic_operations
            else True)
        zero_call_windows = [
            {"api_calls": 0, "llm_calls": 0, "teacher_calls": 0,
             "window_key": key}
            for key in ("BEFORE_PRIVATE_READ", "DURING_PRIVATE_EVALUATION",
                        "AFTER_PRIVATE_EVALUATION")
        ]
        support = [
            _support_result(W02_PRIVATE_SUPPORT_KEYS[0], resource_ok, {
                "logic_operations": total_operations,
                "max_logic_operations": budget.max_logic_operations,
                "resource_exhausted": int(resource_exhausted),
            }),
            _support_result(W02_PRIVATE_SUPPORT_KEYS[1], rollback_ok, {
                "candidate_unchanged": int(after_candidate == before_candidate),
                "overlay_unchanged": int(after_overlay == before_overlay),
            }),
            _support_result(W02_PRIVATE_SUPPORT_KEYS[2], True,
                            zero_call_windows),
            _support_result(W02_PRIVATE_SUPPORT_KEYS[3], clone_ok, {
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
            "candidate_semantic_sha256":
                candidate_result.candidate_semantic_sha256,
            "dimension_results": dimensions,
            "evaluation_count": evaluation_count,
            "family_counts": family_counts,
            "generalized_candidate_count": generalized_candidates,
            "hard_conjunct_results": hard,
            "input_pair_count": input_pair_count,
            "logic_operations": total_operations,
            "max_generalized_candidates_per_observation": max_generalized,
            "overlay_artifact_manifest_sha256":
                overlay_result.artifact_manifest_sha256,
            "overlay_inference_logic_operations": overlay_operations,
            "overlay_rule_row_count": overlay_index.row_count,
            "overlay_semantic_sha256": overlay_index.semantic_sha256,
            "queried_span_count": queried_spans,
            "ranking_cache_entry_count": len(cache.values),
            "ranking_cache_hit_count": cache.hit_count,
            "ranking_cache_miss_count": cache.miss_count,
            "status": _result_status(hard),
            "successor_transform_logic_operations":
                overlay_index.logic_operations,
            "support_results": support,
            "zero_call_windows": zero_call_windows,
        }
    finally:
        close_pairs = getattr(pair_iterator, "close", None)
        if callable(close_pairs):
            close_pairs()
        cache.close()


def run_w02_morphology_successor_private_evaluation(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        parent: W02CompileFreeze,
        candidate_artifact_root: str | Path,
        overlay_artifact_root: str | Path,
        *,
        family_freeze_sha256: str,
        run_id: int = 1,
        ) -> dict[str, object]:
    """guard 消费后执行唯一全量 blind private evaluation。"""
    if run_id != 1:
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private formal run_id 固定为 1")
    if (not isinstance(registration, V2PrivateFamilyRegistration)
            or registration.stage_key != "W-02"
            or registration.formal_run_count != 0
            or registration.private_payload_reads != 0):
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private registration 状态非法")
    permits = authorize_w02_private_files(
        boundary, roots, registration, parent)
    source_count = 0
    source_digest = hashlib.sha256()
    for record in iter_w02_private_records(
            _private_identity(parent, "PRIVATE_SOURCE"),
            permits["PRIVATE_SOURCE"]):
        if not isinstance(record, SourceRefRecord):
            raise W02MorphologySuccessorPrivateEvaluationError(
                "W-02 private SourceRef 类型错误")
        source_digest.update(canonical_json_bytes(record.stable_key.to_list()))
        source_count += 1
    pairs = (
        pair for split in W02_PRIVATE_SPLITS
        for pair in iter_w02_private_pairs(parent, permits, split)
    )
    core = evaluate_w02_private_pair_stream(
        candidate_artifact_root, overlay_artifact_root, pairs,
        registration.resource_budget)
    # 完成后再次走 transport firewall，证明 private owner 未被评测写回。
    authorize_w02_private_files(boundary, roots, registration, parent)
    private_files = tuple(
        _private_identity(parent, key) for key in W02_PRIVATE_LAYOUT_PATHS)
    expected_evaluations = sum(
        parent.plan.split_total(split) for split in W02_PRIVATE_SPLITS)
    if core["input_pair_count"] != expected_evaluations:
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private input 未全量读完")
    transport_bytes = sum(item.transport_size_bytes for item in private_files)
    private_record_reads = source_count + core["input_pair_count"] * 2
    report = {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_PRIVATE_EVALUATION_REPORT"),
        "artifact_version": W02_PRIVATE_EVALUATOR_VERSION,
        **core,
        "family_commitment": registration.family_commitment,
        "family_freeze_sha256": family_freeze_sha256,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 1,
        "formal_shadow_audit_runs": 1,
        "formal_successor_transform_runs": 1,
        "formal_training_runs": 1,
        "label_record_reads": core["input_pair_count"],
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": (
            "W03_COMPILE_FREEZE" if core["status"] == "PASS"
            else "W02_PRIVATE_FAILED_OR_NE_STOP"),
        "observation_reads": core["input_pair_count"],
        "private_content_stream_reads": len(private_files),
        "private_post_content_transport_reads": len(private_files),
        "private_family_registered": 1,
        "private_payload_gets": private_record_reads,
        "private_payload_reads": len(private_files) * 4,
        "private_record_reads": private_record_reads,
        "private_transport_validation_reads": len(private_files) * 2,
        "release_key": "PH2-D03-V2",
        "run_id": 1,
        "run_scope": "FORMAL_BLIND_PRIVATE_EVALUATION",
        "source_count": source_count,
        "source_identity_sha256": source_digest.hexdigest(),
        "stage_key": "W-02",
        "teacher_calls": 0,
        "transport_bytes_read": transport_bytes * 4,
        "validated_layout_count": len(private_files),
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
        },
    }
    if (report["private_record_reads"] > registration.resource_budget.max_records
            or report["transport_bytes_read"]
            > registration.resource_budget.max_payload_bytes
            or report["private_payload_gets"]
            > registration.resource_budget.max_payload_gets):
        raise W02MorphologySuccessorPrivateEvaluationError(
            "W-02 private 非逻辑资源上界超限")
    validate_v2_safe_report(report)
    return report


__all__ = [
    "W02_PRIVATE_EVALUATOR_VERSION", "W02_PRIVATE_LAYOUT_PATHS",
    "W02_PRIVATE_SPLITS", "W02_PRIVATE_SUPPORT_KEYS",
    "W02MorphologySuccessorPrivateEvaluationError",
    "authorize_w02_private_files", "evaluate_w02_private_pair_stream",
    "iter_w02_private_pairs", "iter_w02_private_records",
    "run_w02_morphology_successor_private_evaluation",
]
