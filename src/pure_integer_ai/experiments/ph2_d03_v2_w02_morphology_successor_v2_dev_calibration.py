"""PH2-D03-V2 W-02 morphology successor V2 的只读 dev calibration。"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_publication import (
    W02_CANDIDATE_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import _dimension_key
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
    W02_DEV_LAYOUT_PATHS,
    W02DevInputRoot,
    _dev_freeze,
    _dimension_report,
    _evaluate_pair,
    _expected_family,
    _hash_value,
    _sha256_file,
    _tree_sha256,
    iter_w02_dev_pairs,
    load_w02_dev_candidate_index,
    predict_w02_dev_observation,
    scan_w02_dev_sources,
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
    W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH,
    W02_MORPH_SUCCESSOR_DEV_REPORT_PATH,
    read_w02_morphology_successor_dev_calibration_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_publication import (
    W02_MORPH_SUCCESSOR_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN,
    W02MorphologySuccessorV2Cache,
    predict_w02_morphology_successor_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_contract import (
    W02_MORPH_V2_FREEZE_PATH,
    read_w02_morphology_successor_v2_runtime_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_publication import (
    W02_MORPH_V2_RECEIPT_PATH,
)


W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-DEV-CALIBRATION-FREEZE-V1")
W02_MORPH_SUCCESSOR_V2_DEV_REPORT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-DEV-CALIBRATION-REPORT-V1")
W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_dev_calibration_freeze_v1.json")
W02_MORPH_SUCCESSOR_V2_DEV_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_dev_calibration_report_v1.json")
W02_MORPH_SUCCESSOR_V2_DEV_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v2_dev_calibration.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_v2_dev_calibration.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v2_dev_calibration.py",
)
# 只在完整开发预演闭合后填入并冻结。
W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_COUNTS = {
    "base_logic_operations": 1_108_763,
    "candidate_index_row_count": 19_545,
    "combined_added_candidate_count": 280_380,
    "evaluator_record_reads": 7_306,
    "logic_operations": 3_620_217,
    "max_combined_added_candidates_per_observation": 1_908,
    "max_v1_generalized_candidates_per_observation": 1_538,
    "max_v2_edge_candidates_per_observation": 370,
    "max_v2_edge_candidates_per_requested_span": 8,
    "observation_reads": 7_306,
    "queried_span_count": 12_665,
    "source_count": 6_234,
    "transport_bytes_read": 2_211_445,
    "v1_candidates_retained": 1,
    "v1_generalized_candidate_count": 232_229,
    "v1_overlay_inference_logic_operations": 334_171,
    "v1_overlay_ranking_cache_entry_count": 174,
    "v1_overlay_ranking_cache_hit_count": 12_491,
    "v1_overlay_ranking_cache_miss_count": 174,
    "v1_overlay_rule_row_count": 47_975,
    "v1_successor_transform_logic_operations": 1_574_251,
    "v2_edge_candidate_count": 48_151,
    "v2_overlay_inference_logic_operations": 598_421,
    "v2_overlay_ranking_cache_entry_count": 4_305,
    "v2_overlay_ranking_cache_hit_count": 8_360,
    "v2_overlay_ranking_cache_miss_count": 4_305,
    "v2_overlay_rule_row_count": 383,
    "v2_successor_transform_logic_operations": 4_611,
    "validated_layout_count": 3,
}
W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_FAMILY_COUNTS = {
    "AUTHORED_OOV": 1_206,
    "UD_ANNOTATION": 500,
    "UNICODE_ANNOTATION": 5_600,
}
W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_DIMENSIONS = (
    ("W-02-V2-BOUNDARY-WITHDRAWAL", 1_462),
    ("W-02-V2-MULTI-CANDIDATE", 1_461),
    ("W-02-V2-NEW-CONTENT-MORPHOLOGY", 1_461),
    ("W-02-V2-OOV", 1_461),
    ("W-02-V2-GENERATION-HARD-CONJUNCT", 1_461),
)


# object-model: exception
class W02MorphologySuccessorV2DevCalibrationError(RuntimeError):
    """V2 successor dev 输入、冻结或公开报告不满足严格合同。"""


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative
            or target.is_symlink() or not target.is_relative_to(repository)
            or not target.is_file()):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev repository file 非法")
    return target


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    for relative in W02_MORPH_SUCCESSOR_V2_DEV_CODE_PATHS:
        size, digest = _sha256_file(_repository_file(repository, relative))
        rows.append({
            "repository_file": relative,
            "sha256": digest,
            "size_bytes": size,
        })
    return rows, _hash_value(rows)


def _requested_spans(evaluation: object) -> tuple[tuple[int, int], ...]:
    """只取得冻结边界，不把 lemma、UPOS 或 feats 传给预测链。"""
    expected_payload = getattr(evaluation, "expected_payload", None)
    if expected_payload is None:
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev evaluator payload 缺失")
    expected = expected_payload.to_value()
    if _expected_family(expected) != "UD_ANNOTATION":
        return ()
    rows = expected.get("boundary_spans")
    if not isinstance(rows, list):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev boundary spans 非 list")
    spans: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise W02MorphologySuccessorV2DevCalibrationError(
                "successor V2 dev boundary span 非 object")
        start = row.get("start")
        end = row.get("end")
        if (type(start) is not int or type(end) is not int
                or start < 0 or end <= start):
            raise W02MorphologySuccessorV2DevCalibrationError(
                "successor V2 dev boundary span 非法")
        spans.add((start, end))
    if not spans:
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev UD span 为空")
    return tuple(sorted(spans))


def _dependency_state(repository: Path) -> dict[str, object]:
    parent = read_w02_compile_freeze(repository)
    v1_dev_freeze = read_w02_morphology_successor_dev_calibration_freeze(
        repository)
    v2_runtime = read_w02_morphology_successor_v2_runtime_freeze(repository)
    candidate_path = _repository_file(repository, W02_CANDIDATE_RECEIPT_PATH)
    v1_path = _repository_file(repository, W02_MORPH_SUCCESSOR_RECEIPT_PATH)
    v2_path = _repository_file(repository, W02_MORPH_V2_RECEIPT_PATH)
    v1_dev_freeze_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH)
    v1_dev_report_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_DEV_REPORT_PATH)
    v2_runtime_path = _repository_file(
        repository, W02_MORPH_V2_FREEZE_PATH)
    candidate = read_canonical_object(candidate_path)
    v1 = read_canonical_object(v1_path)
    v2 = read_canonical_object(v2_path)
    v1_dev_report = read_canonical_object(v1_dev_report_path)
    expected_projection = v1_dev_freeze["expected_dimensions"]
    actual_projection = [{
        "denominator": row.get("denominator"),
        "dimension_key": row.get("dimension_key"),
        "failed": row.get("failed"),
        "ne": row.get("ne"),
        "numerator": row.get("numerator"),
        "status": row.get("status"),
    } for row in v1_dev_report.get("dimension_results", ())
        if isinstance(row, dict)]
    zero_fields = (
        candidate.get("formal_private_evaluation_runs"),
        candidate.get("private_payload_reads"), candidate.get("teacher_calls"),
        v1.get("formal_private_evaluation_runs"),
        v1.get("private_payload_reads"), v1.get("teacher_calls"),
        v2.get("formal_private_evaluation_runs"),
        v2.get("private_payload_reads"), v2.get("teacher_calls"),
        v1_dev_report.get("formal_private_evaluation_runs"),
        v1_dev_report.get("private_payload_reads"),
        v1_dev_report.get("teacher_calls"),
    )
    if (candidate.get("status") != "W02_CANDIDATE_ARTIFACT_FROZEN"
            or candidate.get("formal_training_runs") != 1
            or v1.get("status") != "W02_MORPHOLOGY_SUCCESSOR_ARTIFACT_FROZEN"
            or v1.get("formal_successor_transform_runs") != 1
            or v1.get("parent_candidate_manifest_sha256")
            != candidate.get("candidate_artifact_manifest_sha256")
            or v1.get("parent_candidate_semantic_sha256")
            != candidate.get("candidate_semantic_sha256")
            or v2.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_V2_ARTIFACT_FROZEN"
            or v2.get("formal_successor_v2_transform_runs") != 1
            or v2.get("parent_candidate_manifest_sha256")
            != candidate.get("candidate_artifact_manifest_sha256")
            or v2.get("parent_candidate_semantic_sha256")
            != candidate.get("candidate_semantic_sha256")
            or v2.get("parent_v1_overlay_manifest_sha256")
            != v1.get("overlay_artifact_manifest_sha256")
            or v2.get("parent_v1_overlay_semantic_sha256")
            != v1.get("overlay_semantic_sha256")
            or v2.get("runtime_freeze_sha256") != v2_runtime.sha256()
            or v1_dev_report.get("status") != "PASS"
            or v1_dev_report.get("formal_dev_calibration_runs") != 1
            or actual_projection != expected_projection
            or v1_dev_freeze.get("dev_input_commitment")
            != v1_dev_report.get("dev_input_commitment")
            or any(value != 0 for value in zero_fields)):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev dependency state 漂移")
    return {
        "candidate": candidate,
        "candidate_receipt_file_sha256": _sha256_file(candidate_path)[1],
        "compile_freeze_sha256": parent.sha256(),
        "dev_input_commitment": v1_dev_freeze["dev_input_commitment"],
        "dev_input_files": v1_dev_freeze["dev_input_files"],
        "v1": v1,
        "v1_dev_freeze_file_sha256": _sha256_file(v1_dev_freeze_path)[1],
        "v1_dev_report_file_sha256": _sha256_file(v1_dev_report_path)[1],
        "v1_receipt_file_sha256": _sha256_file(v1_path)[1],
        "v2": v2,
        "v2_receipt_file_sha256": _sha256_file(v2_path)[1],
        "v2_runtime_freeze_file_sha256": _sha256_file(v2_runtime_path)[1],
    }


def build_w02_morphology_successor_v2_dev_calibration_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """冻结 V2 dev 的代码、三层 artifact、dev owner 与五维 1/1 门。"""
    repository = Path(repository_root).resolve()
    if not W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_COUNTS:
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev exact counts 尚未冻结")
    dependency = _dependency_state(repository)
    candidate = dependency["candidate"]
    v1 = dependency["v1"]
    v2 = dependency["v2"]
    code_rows, code_sha = _code_rows(repository)
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-02")
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_DEV_CALIBRATION_FREEZE"),
        "artifact_version": W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_VERSION,
        "candidate_artifact_manifest_sha256":
            candidate["candidate_artifact_manifest_sha256"],
        "candidate_receipt_file_sha256":
            dependency["candidate_receipt_file_sha256"],
        "candidate_semantic_sha256": candidate["candidate_semantic_sha256"],
        "code_files": code_rows,
        "code_freeze_sha256": code_sha,
        "compile_freeze_sha256": dependency["compile_freeze_sha256"],
        "dev_input_commitment": dependency["dev_input_commitment"],
        "dev_input_files": dependency["dev_input_files"],
        "evaluator_policy": policy.to_dict(),
        "expected_counts": dict(W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_COUNTS),
        "expected_family_counts": dict(
            W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_FAMILY_COUNTS),
        "expected_dimensions": [
            {"denominator": denominator, "dimension_key": name,
             "failed": 0, "ne": 0, "numerator": denominator,
             "status": "PASS"}
            for name, denominator in
            W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_DIMENSIONS
        ],
        "formal_dev_calibration_runs": 0,
        "formal_private_evaluation_runs": 0,
        "formal_successor_transform_runs": 1,
        "formal_successor_v2_transform_runs": 1,
        "formal_training_runs": 1,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": "W02_FORMAL_MORPHOLOGY_SUCCESSOR_V2_DEV_CALIBRATION",
        "parent_formal_successor_dev_calibration_runs": 1,
        "private_family_registered": 0,
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "resource_budget": {
            "max_logic_operations": 9_000_000,
            "max_payload_bytes": 536_870_912,
            "max_payload_gets": 300_000,
            "max_records": 100_000,
            "max_workers": 4,
        },
        "shadow_started": 0,
        "stage_key": "W-02",
        "status": (
            "W02_MORPHOLOGY_SUCCESSOR_V2_DEV_CALIBRATION_FREEZE_COMPLETE"),
        "teacher_calls": 0,
        "v1_dev_freeze_file_sha256":
            dependency["v1_dev_freeze_file_sha256"],
        "v1_dev_report_file_sha256":
            dependency["v1_dev_report_file_sha256"],
        "v1_overlay_artifact_manifest_sha256":
            v1["overlay_artifact_manifest_sha256"],
        "v1_overlay_receipt_file_sha256":
            dependency["v1_receipt_file_sha256"],
        "v1_overlay_semantic_sha256": v1["overlay_semantic_sha256"],
        "v2_overlay_artifact_manifest_sha256":
            v2["v2_overlay_artifact_manifest_sha256"],
        "v2_overlay_receipt_file_sha256":
            dependency["v2_receipt_file_sha256"],
        "v2_overlay_semantic_sha256": v2["semantic_sha256"],
        "v2_runtime_freeze_file_sha256":
            dependency["v2_runtime_freeze_file_sha256"],
    }


def publish_w02_morphology_successor_v2_dev_calibration_freeze(
        repository_root: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_v2_dev_calibration_freeze(repository)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    return target


def read_w02_morphology_successor_v2_dev_calibration_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    target = _repository_file(
        repository, W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_PATH)
    value = read_canonical_object(target)
    if value != build_w02_morphology_successor_v2_dev_calibration_freeze(
            repository):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev freeze 与 live identity 漂移")
    return value


def _candidate_identity(candidate: object) -> tuple[object, ...]:
    return tuple(getattr(candidate, name) for name in (
        "start", "end", "form", "lemma", "upos", "feats_json",
        "support_count"))


def _assert_v1_retained(
        v1: object,
        v2: object,
        requested_spans: tuple[tuple[int, int], ...],
        ) -> int:
    before = tuple(_candidate_identity(item)
                   for item in v1.prediction.morphology_candidates)
    after = tuple(_candidate_identity(item)
                  for item in v2.prediction.morphology_candidates)
    before_set = set(before)
    added = tuple(identity for identity in after if identity not in before_set)
    per_span = {span: 0 for span in requested_spans}
    for identity in added:
        span = (identity[0], identity[1])
        if span not in per_span:
            raise W02MorphologySuccessorV2DevCalibrationError(
                "successor V2 在非请求 span 追加候选")
        per_span[span] += 1
    maximum = max(per_span.values(), default=0)
    if (not set(before).issubset(after)
            or len(added) != v2.edge_candidate_count
            or maximum > W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 未保持 V1 候选或超出逐 span 上界")
    return maximum


def _expected_dimension_projection() -> list[dict[str, object]]:
    return [
        {"denominator": denominator, "dimension_key": name,
         "failed": 0, "ne": 0, "numerator": denominator, "status": "PASS"}
        for name, denominator in W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_DIMENSIONS
    ]


def _dimension_projection(report: dict[str, object]) -> list[dict[str, object]]:
    rows = report.get("dimension_results")
    if not isinstance(rows, list):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev dimension result 非 list")
    return [{
        "denominator": row.get("denominator"),
        "dimension_key": row.get("dimension_key"),
        "failed": row.get("failed"),
        "ne": row.get("ne"),
        "numerator": row.get("numerator"),
        "status": row.get("status"),
    } for row in rows if isinstance(row, dict)]


def _assert_preflight_gates(report: dict[str, object]) -> None:
    zero_write = report.get("zero_write_audit")
    if (_dimension_projection(report) != _expected_dimension_projection()
            or report.get("status") != "PASS"
            or report.get("v1_candidates_retained") != 1
            or report.get("max_v2_edge_candidates_per_requested_span")
            > W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN
            or report.get("logic_operations") > 9_000_000
            or not isinstance(zero_write, dict)
            or any(value != 0 for value in zero_write.values())
            or any(report.get(name) != 0 for name in (
                "formal_private_evaluation_runs", "private_payload_reads",
                "teacher_calls"))):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev preflight gate 未闭合")


def _assert_expected_report(
        report: dict[str, object], freeze: dict[str, object]) -> None:
    for key, expected in freeze["expected_counts"].items():
        if report.get(key) != expected:
            raise W02MorphologySuccessorV2DevCalibrationError(
                f"successor V2 dev expected count 漂移: {key}")
    if report.get("family_counts") != freeze["expected_family_counts"]:
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev family count 漂移")
    if _dimension_projection(report) != freeze["expected_dimensions"]:
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev 五维未达到冻结 1/1 门")


def _run_calibration(
        repository: Path,
        dev_root: str | Path,
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        *,
        formal: bool,
        ) -> dict[str, object]:
    freeze = (read_w02_morphology_successor_v2_dev_calibration_freeze(
        repository) if formal else None)
    dependency = _dependency_state(repository)
    parent = read_w02_compile_freeze(repository)
    dev = W02DevInputRoot(Path(dev_root))
    candidate_root = Path(candidate_artifact_root).resolve()
    v1_root = Path(v1_overlay_artifact_root).resolve()
    v2_root = Path(v2_overlay_artifact_root).resolve()
    before_dev = _tree_sha256(dev.root)
    before_candidate = _tree_sha256(candidate_root)
    before_v1 = _tree_sha256(v1_root)
    before_v2 = _tree_sha256(v2_root)
    candidate_result = read_w02_candidate_artifact(candidate_root)
    v1_result = read_w02_morphology_overlay_artifact(v1_root)
    v2_result = read_w02_morphology_successor_v2_overlay_artifact(v2_root)
    candidate = dependency["candidate"]
    v1_receipt = dependency["v1"]
    v2_receipt = dependency["v2"]
    if (candidate_result.artifact_manifest_sha256
            != candidate["candidate_artifact_manifest_sha256"]
            or candidate_result.candidate_semantic_sha256
            != candidate["candidate_semantic_sha256"]
            or v1_result.artifact_manifest_sha256
            != v1_receipt["overlay_artifact_manifest_sha256"]
            or v1_result.overlay_semantic_sha256
            != v1_receipt["overlay_semantic_sha256"]
            or v2_result.artifact_manifest_sha256
            != v2_receipt["v2_overlay_artifact_manifest_sha256"]
            or v2_result.semantic_sha256 != v2_receipt["semantic_sha256"]):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev artifact identity 漂移")
    v1_index = load_w02_morphology_overlay_index(v1_root)
    v2_index = load_w02_morphology_successor_v2_overlay_index(v2_root)
    source_count, source_sha = scan_w02_dev_sources(parent, dev)
    dimension_by_key = {
        _dimension_key(name).components: name for name in W02_DEV_DIMENSIONS
    }
    rows: dict[str, list[tuple[bool | None, str]]] = {
        name: [] for name in W02_DEV_DIMENSIONS
    }
    family_counts = {name: 0 for name in (
        "AUTHORED_OOV", "UD_ANNOTATION", "UNICODE_ANNOTATION")}
    base_operations = 0
    v1_inference_operations = 0
    v2_inference_operations = 0
    queried_spans = 0
    v1_candidates = 0
    v2_candidates = 0
    max_v1_candidates = 0
    max_v2_candidates = 0
    max_combined_candidates = 0
    max_v2_per_span = 0
    evaluation_count = 0
    v1_cache = W02MorphologyRankingCache.empty()
    v2_cache = W02MorphologySuccessorV2Cache.empty()
    try:
        with open_w02_candidate_predictor(candidate_root) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
            for observation, evaluation in iter_w02_dev_pairs(parent, dev):
                base, operations = predict_w02_dev_observation(
                    candidate_index, observation)
                spans = _requested_spans(evaluation)
                v1 = predict_w02_morphology_successor(
                    v1_index, observation, base, requested_spans=spans,
                    ranking_cache=v1_cache)
                v2 = predict_w02_morphology_successor_v2(
                    v2_index, observation, v1, requested_spans=spans,
                    cache=v2_cache)
                per_span_maximum = _assert_v1_retained(v1, v2, spans)
                dimension, family, passed, evidence_sha = _evaluate_pair(
                    observation, evaluation, v2.prediction, dimension_by_key)
                if dimension not in rows or family not in family_counts:
                    raise W02MorphologySuccessorV2DevCalibrationError(
                        "successor V2 dev dimension/family 漂移")
                rows[dimension].append((passed, evidence_sha))
                family_counts[family] += 1
                base_operations += operations + 8
                v1_inference_operations += v1.logic_operations
                v2_inference_operations += v2.logic_operations
                queried_spans += len(spans)
                v1_candidates += v1.generalized_candidate_count
                v2_candidates += v2.edge_candidate_count
                max_v1_candidates = max(
                    max_v1_candidates, v1.generalized_candidate_count)
                max_v2_candidates = max(
                    max_v2_candidates, v2.edge_candidate_count)
                combined = (v1.generalized_candidate_count
                            + v2.edge_candidate_count)
                max_combined_candidates = max(max_combined_candidates, combined)
                max_v2_per_span = max(max_v2_per_span, per_span_maximum)
                evaluation_count += 1
                total = (v1_index.logic_operations + v2_index.logic_operations
                         + base_operations + v1_inference_operations
                         + v2_inference_operations)
                if total > 9_000_000:
                    raise W02MorphologySuccessorV2DevCalibrationError(
                        "successor V2 dev logic resource stop")
        dimensions = [
            _dimension_report(name, rows[name]) for name in W02_DEV_DIMENSIONS
        ]
        failed = sum(row["status"] == "FAIL" for row in dimensions)
        ne = sum(row["status"] == "NE" for row in dimensions)
        status = "FAIL" if failed else "NE" if ne else "PASS"
        input_bytes = sum(
            _dev_freeze(parent, key).transport_size_bytes
            for key in W02_DEV_LAYOUT_PATHS)
        report = {
            "artifact_kind": (
                "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_DEV_CALIBRATION_REPORT"),
            "artifact_version": W02_MORPH_SUCCESSOR_V2_DEV_REPORT_VERSION,
            "base_logic_operations": base_operations,
            "candidate_artifact_manifest_sha256":
                candidate["candidate_artifact_manifest_sha256"],
            "candidate_index_row_count": candidate_index.row_count,
            "candidate_index_semantic_sha256": candidate_index.semantic_sha256,
            "candidate_semantic_sha256": candidate["candidate_semantic_sha256"],
            "combined_added_candidate_count": v1_candidates + v2_candidates,
            "compile_freeze_sha256": dependency["compile_freeze_sha256"],
            "dev_input_commitment": dependency["dev_input_commitment"],
            "dimension_results": dimensions,
            "evaluator_record_reads": evaluation_count,
            "family_counts": family_counts,
            "formal_dev_calibration_runs": 1 if formal else 0,
            "formal_private_evaluation_runs": 0,
            "formal_successor_transform_runs": 1,
            "formal_successor_v2_transform_runs": 1,
            "formal_training_runs": 1,
            "language_capability_mastered": 0,
            "language_readiness": 0,
            "logic_operations": (
                v1_index.logic_operations + v2_index.logic_operations
                + base_operations + v1_inference_operations
                + v2_inference_operations),
            "max_combined_added_candidates_per_observation":
                max_combined_candidates,
            "max_v1_generalized_candidates_per_observation": max_v1_candidates,
            "max_v2_edge_candidates_per_observation": max_v2_candidates,
            "max_v2_edge_candidates_per_requested_span": max_v2_per_span,
            "next_action": (
                "W02_SUCCESSOR_V2_SHADOW_AUDIT" if formal and status == "PASS"
                else "W02_SUCCESSOR_V2_DEV_FAILED_STOP" if formal
                else "W02_SUCCESSOR_V2_DEV_CALIBRATION_FREEZE"),
            "observation_reads": evaluation_count,
            "parent_formal_successor_dev_calibration_runs": 1,
            "private_family_registered": 0,
            "private_payload_reads": 0,
            "queried_span_count": queried_spans,
            "release_key": "PH2-D03-V2",
            "run_id": 1 if formal else 0,
            "run_scope": "FORMAL" if formal else "DEVELOPMENT_PREFLIGHT",
            "shadow_started": 0,
            "source_count": source_count,
            "source_identity_sha256": source_sha,
            "stage_key": "W-02",
            "status": status,
            "teacher_calls": 0,
            "transport_bytes_read": input_bytes,
            "v1_candidates_retained": 1,
            "v1_generalized_candidate_count": v1_candidates,
            "v1_overlay_artifact_manifest_sha256":
                v1_receipt["overlay_artifact_manifest_sha256"],
            "v1_overlay_inference_logic_operations": v1_inference_operations,
            "v1_overlay_ranking_cache_entry_count": len(v1_cache.values),
            "v1_overlay_ranking_cache_hit_count": v1_cache.hit_count,
            "v1_overlay_ranking_cache_miss_count": v1_cache.miss_count,
            "v1_overlay_rule_row_count": v1_index.row_count,
            "v1_overlay_semantic_sha256": v1_index.semantic_sha256,
            "v1_successor_transform_logic_operations": v1_index.logic_operations,
            "v2_edge_candidate_count": v2_candidates,
            "v2_overlay_artifact_manifest_sha256":
                v2_receipt["v2_overlay_artifact_manifest_sha256"],
            "v2_overlay_inference_logic_operations": v2_inference_operations,
            "v2_overlay_ranking_cache_entry_count": len(v2_cache.values),
            "v2_overlay_ranking_cache_hit_count": v2_cache.hit_count,
            "v2_overlay_ranking_cache_miss_count": v2_cache.miss_count,
            "v2_overlay_rule_row_count": v2_index.row_count,
            "v2_overlay_semantic_sha256": v2_index.semantic_sha256,
            "v2_successor_transform_logic_operations": v2_index.logic_operations,
            "validated_layout_count": len(W02_DEV_LAYOUT_PATHS),
            "zero_write_audit": {
                "candidate_writes": 0,
                "companion_writes": 0,
                "core_writes": 0,
                "dev_owner_writes": 0,
                "evidence_writes": 0,
                "memory_writes": 0,
                "use_writes": 0,
                "v1_overlay_writes": 0,
                "v2_overlay_writes": 0,
            },
        }
    finally:
        v1_cache.close()
        v2_cache.close()
    if (_tree_sha256(candidate_root) != before_candidate
            or _tree_sha256(v1_root) != before_v1
            or _tree_sha256(v2_root) != before_v2
            or _tree_sha256(dev.root) != before_dev):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev calibration 产生非授权写入")
    _assert_preflight_gates(report)
    if formal:
        assert freeze is not None
        report["code_freeze_sha256"] = freeze["code_freeze_sha256"]
        _assert_expected_report(report, freeze)
    validate_v2_safe_report(report)
    return report


def run_w02_morphology_successor_v2_dev_preflight(
        repository_root: str | Path,
        dev_root: str | Path,
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        ) -> dict[str, object]:
    return _run_calibration(
        Path(repository_root).resolve(), dev_root, candidate_artifact_root,
        v1_overlay_artifact_root, v2_overlay_artifact_root, formal=False)


def run_w02_morphology_successor_v2_dev_calibration(
        repository_root: str | Path,
        dev_root: str | Path,
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        *,
        run_id: int = 1,
        ) -> dict[str, object]:
    if run_id != 1:
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev formal run_id 固定为 1")
    return _run_calibration(
        Path(repository_root).resolve(), dev_root, candidate_artifact_root,
        v1_overlay_artifact_root, v2_overlay_artifact_root, formal=True)


def publish_w02_morphology_successor_v2_dev_calibration_report(
        repository_root: str | Path,
        external_report: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = read_canonical_object(external_report)
    validate_v2_safe_report(value)
    freeze_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_PATH)
    freeze = read_w02_morphology_successor_v2_dev_calibration_freeze(repository)
    freeze_size, freeze_sha = _sha256_file(freeze_path)
    if (value.get("artifact_version")
            != W02_MORPH_SUCCESSOR_V2_DEV_REPORT_VERSION
            or value.get("run_scope") != "FORMAL"
            or value.get("run_id") != 1
            or value.get("formal_dev_calibration_runs") != 1
            or value.get("formal_private_evaluation_runs") != 0
            or value.get("private_payload_reads") != 0
            or value.get("teacher_calls") != 0
            or value.get("code_freeze_sha256")
            != freeze["code_freeze_sha256"]):
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev 正式报告状态非法")
    _assert_preflight_gates(value)
    _assert_expected_report(value, freeze)
    public = dict(value)
    public["dev_freeze_file_sha256"] = freeze_sha
    public["dev_freeze_size_bytes"] = freeze_size
    validate_v2_safe_report(public)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_V2_DEV_REPORT_PATH).parts)
    write_immutable_json(public, target)
    if target.read_bytes() != canonical_json_bytes(public) + b"\n":
        raise W02MorphologySuccessorV2DevCalibrationError(
            "successor V2 dev report 发布字节漂移")
    return target


__all__ = [
    "W02_MORPH_SUCCESSOR_V2_DEV_CODE_PATHS",
    "W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_COUNTS",
    "W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_PATH",
    "W02_MORPH_SUCCESSOR_V2_DEV_REPORT_PATH",
    "W02MorphologySuccessorV2DevCalibrationError",
    "build_w02_morphology_successor_v2_dev_calibration_freeze",
    "publish_w02_morphology_successor_v2_dev_calibration_freeze",
    "publish_w02_morphology_successor_v2_dev_calibration_report",
    "read_w02_morphology_successor_v2_dev_calibration_freeze",
    "run_w02_morphology_successor_v2_dev_calibration",
    "run_w02_morphology_successor_v2_dev_preflight",
]
