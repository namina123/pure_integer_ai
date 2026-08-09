"""PH2-D03-V2 W-02 morphology successor 的只读 dev calibration。"""
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
    W02_DEV_FREEZE_PATH,
    W02_DEV_LAYOUT_PATHS,
    W02_DEV_REPORT_PATH,
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
    read_w02_dev_calibration_freeze,
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
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_contract import (
    W02_MORPH_SUCCESSOR_FREEZE_PATH,
    read_w02_morphology_successor_runtime_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_publication import (
    W02_MORPH_SUCCESSOR_RECEIPT_PATH,
)


W02_MORPH_SUCCESSOR_DEV_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-DEV-CALIBRATION-FREEZE-V1")
W02_MORPH_SUCCESSOR_DEV_REPORT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-DEV-CALIBRATION-REPORT-V1")
W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_dev_calibration_freeze_v1.json")
W02_MORPH_SUCCESSOR_DEV_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_dev_calibration_report_v1.json")
W02_MORPH_SUCCESSOR_DEV_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_dev_calibration.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_dev_calibration.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_dev_calibration.py",
)
W02_MORPH_SUCCESSOR_DEV_EXPECTED_COUNTS = {
    "base_logic_operations": 1_108_763,
    "candidate_index_row_count": 19_545,
    "evaluator_record_reads": 7_306,
    "generalized_candidate_count": 232_229,
    "logic_operations": 3_017_185,
    "max_generalized_candidates_per_observation": 1_538,
    "observation_reads": 7_306,
    "overlay_inference_logic_operations": 334_171,
    "overlay_rule_row_count": 47_975,
    "queried_span_count": 12_665,
    "ranking_cache_entry_count": 174,
    "ranking_cache_hit_count": 12_491,
    "ranking_cache_miss_count": 174,
    "source_count": 6_234,
    "successor_transform_logic_operations": 1_574_251,
    "transport_bytes_read": 2_211_445,
}
W02_MORPH_SUCCESSOR_DEV_EXPECTED_DIMENSIONS = (
    ("W-02-V2-BOUNDARY-WITHDRAWAL", 1_462),
    ("W-02-V2-MULTI-CANDIDATE", 1_461),
    ("W-02-V2-NEW-CONTENT-MORPHOLOGY", 1_461),
    ("W-02-V2-OOV", 1_461),
    ("W-02-V2-GENERATION-HARD-CONJUNCT", 1_461),
)


# object-model: exception
class W02MorphologySuccessorDevCalibrationError(RuntimeError):
    """Successor dev 输入、冻结或公开报告不满足严格合同。"""


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative
            or target.is_symlink() or not target.is_relative_to(repository)
            or not target.is_file()):
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev repository file 非法")
    return target


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    for relative in W02_MORPH_SUCCESSOR_DEV_CODE_PATHS:
        size, digest = _sha256_file(_repository_file(repository, relative))
        rows.append({
            "repository_file": relative,
            "sha256": digest,
            "size_bytes": size,
        })
    return rows, _hash_value(rows)


def _requested_spans(evaluation: object) -> tuple[tuple[int, int], ...]:
    """只从冻结边界标签取得 span，不向预测器传形态答案。"""
    expected_payload = getattr(evaluation, "expected_payload", None)
    if expected_payload is None:
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev evaluator payload 缺失")
    expected = expected_payload.to_value()
    if _expected_family(expected) != "UD_ANNOTATION":
        return ()
    rows = expected.get("boundary_spans")
    if not isinstance(rows, list):
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev boundary spans 非 list")
    spans: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise W02MorphologySuccessorDevCalibrationError(
                "successor dev boundary span 非 object")
        start = row.get("start")
        end = row.get("end")
        if (type(start) is not int or type(end) is not int
                or start < 0 or end <= start):
            raise W02MorphologySuccessorDevCalibrationError(
                "successor dev boundary span 非法")
        spans.add((start, end))
    if not spans:
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev UD span 为空")
    return tuple(sorted(spans))


def _dependency_state(repository: Path) -> dict[str, object]:
    parent = read_w02_compile_freeze(repository)
    old_dev = read_w02_dev_calibration_freeze(repository)
    successor_runtime = read_w02_morphology_successor_runtime_freeze(repository)
    candidate_receipt_path = _repository_file(
        repository, W02_CANDIDATE_RECEIPT_PATH)
    overlay_receipt_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_RECEIPT_PATH)
    old_dev_freeze_path = _repository_file(repository, W02_DEV_FREEZE_PATH)
    old_dev_report_path = _repository_file(repository, W02_DEV_REPORT_PATH)
    successor_runtime_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_FREEZE_PATH)
    candidate_receipt = read_canonical_object(candidate_receipt_path)
    overlay_receipt = read_canonical_object(overlay_receipt_path)
    old_report = read_canonical_object(old_dev_report_path)
    morphology_rows = tuple(
        row for row in old_report.get("dimension_results", ())
        if isinstance(row, dict)
        and row.get("dimension_key") == W02_DEV_DIMENSIONS[2])
    if (candidate_receipt.get("status") != "W02_CANDIDATE_ARTIFACT_FROZEN"
            or candidate_receipt.get("formal_training_runs") != 1
            or overlay_receipt.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_ARTIFACT_FROZEN"
            or overlay_receipt.get("formal_successor_transform_runs") != 1
            or overlay_receipt.get("formal_training_runs") != 0
            or overlay_receipt.get("parent_candidate_manifest_sha256")
            != candidate_receipt.get("candidate_artifact_manifest_sha256")
            or overlay_receipt.get("parent_candidate_semantic_sha256")
            != candidate_receipt.get("candidate_semantic_sha256")
            or overlay_receipt.get("runtime_freeze_sha256")
            != successor_runtime.sha256()
            or old_report.get("status") != "FAIL"
            or old_report.get("formal_dev_calibration_runs") != 1
            or len(morphology_rows) != 1
            or morphology_rows[0].get("failed") != 90
            or old_dev.get("dev_input_commitment")
            != old_report.get("dev_input_commitment")
            or any(value != 0 for value in (
                candidate_receipt.get("formal_private_evaluation_runs"),
                candidate_receipt.get("private_payload_reads"),
                candidate_receipt.get("teacher_calls"),
                overlay_receipt.get("formal_private_evaluation_runs"),
                overlay_receipt.get("private_payload_reads"),
                overlay_receipt.get("teacher_calls"),
            ))):
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev dependency state 漂移")
    return {
        "candidate_receipt": candidate_receipt,
        "candidate_receipt_file_sha256": _sha256_file(
            candidate_receipt_path)[1],
        "compile_freeze_sha256": parent.sha256(),
        "dev_input_commitment": old_dev["dev_input_commitment"],
        "dev_input_files": old_dev["dev_input_files"],
        "old_dev_failure_report_file_sha256": _sha256_file(
            old_dev_report_path)[1],
        "old_dev_freeze_file_sha256": _sha256_file(old_dev_freeze_path)[1],
        "overlay_receipt": overlay_receipt,
        "overlay_receipt_file_sha256": _sha256_file(overlay_receipt_path)[1],
        "successor_runtime_freeze_file_sha256": _sha256_file(
            successor_runtime_path)[1],
    }


def build_w02_morphology_successor_dev_calibration_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """冻结 successor dev 的代码、双 artifact、dev owner 和 1/1 门。"""
    repository = Path(repository_root).resolve()
    dependency = _dependency_state(repository)
    candidate = dependency["candidate_receipt"]
    overlay = dependency["overlay_receipt"]
    code_rows, code_sha = _code_rows(repository)
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-02")
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_DEV_CALIBRATION_FREEZE"),
        "artifact_version": W02_MORPH_SUCCESSOR_DEV_FREEZE_VERSION,
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
        "expected_counts": dict(W02_MORPH_SUCCESSOR_DEV_EXPECTED_COUNTS),
        "expected_dimensions": [
            {"denominator": denominator, "dimension_key": name,
             "failed": 0, "ne": 0, "numerator": denominator,
             "status": "PASS"}
            for name, denominator in W02_MORPH_SUCCESSOR_DEV_EXPECTED_DIMENSIONS
        ],
        "formal_dev_calibration_runs": 0,
        "formal_private_evaluation_runs": 0,
        "formal_successor_transform_runs": 1,
        "formal_training_runs": 1,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": "W02_FORMAL_MORPHOLOGY_SUCCESSOR_DEV_CALIBRATION",
        "old_dev_failure_report_file_sha256":
            dependency["old_dev_failure_report_file_sha256"],
        "old_dev_freeze_file_sha256":
            dependency["old_dev_freeze_file_sha256"],
        "overlay_artifact_manifest_sha256":
            overlay["overlay_artifact_manifest_sha256"],
        "overlay_receipt_file_sha256":
            dependency["overlay_receipt_file_sha256"],
        "overlay_semantic_sha256": overlay["overlay_semantic_sha256"],
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
            "W02_MORPHOLOGY_SUCCESSOR_DEV_CALIBRATION_FREEZE_COMPLETE"),
        "successor_runtime_freeze_file_sha256":
            dependency["successor_runtime_freeze_file_sha256"],
        "teacher_calls": 0,
    }


def publish_w02_morphology_successor_dev_calibration_freeze(
        repository_root: str | Path,
        ) -> Path:
    """不可覆盖发布 successor dev 正式运行前冻结。"""
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_dev_calibration_freeze(repository)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    return target


def read_w02_morphology_successor_dev_calibration_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """严格回读 successor dev freeze，并重算全部公开依赖。"""
    repository = Path(repository_root).resolve()
    target = _repository_file(repository, W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH)
    value = read_canonical_object(target)
    if value != build_w02_morphology_successor_dev_calibration_freeze(repository):
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev freeze 与 live identity 漂移")
    return value


def _assert_expected_report(
        report: dict[str, object], freeze: dict[str, object]) -> None:
    for key, expected in freeze["expected_counts"].items():
        if report.get(key) != expected:
            raise W02MorphologySuccessorDevCalibrationError(
                f"successor dev expected count 漂移: {key}")
    expected_dimensions = freeze["expected_dimensions"]
    actual_dimensions = report.get("dimension_results")
    if not isinstance(actual_dimensions, list):
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev dimension result 非 list")
    projection = [{
        "denominator": row.get("denominator"),
        "dimension_key": row.get("dimension_key"),
        "failed": row.get("failed"),
        "ne": row.get("ne"),
        "numerator": row.get("numerator"),
        "status": row.get("status"),
    } for row in actual_dimensions if isinstance(row, dict)]
    if projection != expected_dimensions:
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev 五维未达到冻结 1/1 门")


def _run_calibration(
        repository: Path,
        dev_root: str | Path,
        candidate_artifact_root: str | Path,
        overlay_artifact_root: str | Path,
        *,
        formal: bool,
        ) -> dict[str, object]:
    freeze = (
        read_w02_morphology_successor_dev_calibration_freeze(repository)
        if formal else
        build_w02_morphology_successor_dev_calibration_freeze(repository))
    parent = read_w02_compile_freeze(repository)
    dev = W02DevInputRoot(Path(dev_root))
    candidate_root = Path(candidate_artifact_root).resolve()
    overlay_root = Path(overlay_artifact_root).resolve()
    before_dev = _tree_sha256(dev.root)
    before_candidate = _tree_sha256(candidate_root)
    before_overlay = _tree_sha256(overlay_root)
    candidate_result = read_w02_candidate_artifact(candidate_root)
    overlay_result = read_w02_morphology_overlay_artifact(overlay_root)
    if (candidate_result.artifact_manifest_sha256
            != freeze["candidate_artifact_manifest_sha256"]
            or candidate_result.candidate_semantic_sha256
            != freeze["candidate_semantic_sha256"]
            or overlay_result.artifact_manifest_sha256
            != freeze["overlay_artifact_manifest_sha256"]
            or overlay_result.overlay_semantic_sha256
            != freeze["overlay_semantic_sha256"]
            or overlay_result.parent_candidate_semantic_sha256
            != candidate_result.candidate_semantic_sha256):
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev artifact identity 漂移")
    overlay_index = load_w02_morphology_overlay_index(overlay_root)
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
    overlay_operations = 0
    queried_spans = 0
    generalized_candidates = 0
    max_generalized = 0
    evaluation_count = 0
    ranking_cache = W02MorphologyRankingCache.empty()
    try:
        with open_w02_candidate_predictor(candidate_root) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
            for observation, evaluation in iter_w02_dev_pairs(parent, dev):
                base, operations = predict_w02_dev_observation(
                    candidate_index, observation)
                spans = _requested_spans(evaluation)
                successor = predict_w02_morphology_successor(
                    overlay_index, observation, base,
                    requested_spans=spans, ranking_cache=ranking_cache)
                dimension, family, passed, evidence_sha = _evaluate_pair(
                    observation, evaluation, successor.prediction,
                    dimension_by_key)
                if dimension not in rows or family not in family_counts:
                    raise W02MorphologySuccessorDevCalibrationError(
                        "successor dev dimension/family 漂移")
                rows[dimension].append((passed, evidence_sha))
                family_counts[family] += 1
                base_operations += operations + 8
                overlay_operations += successor.logic_operations
                queried_spans += len(spans)
                generalized_candidates += successor.generalized_candidate_count
                max_generalized = max(
                    max_generalized, successor.generalized_candidate_count)
                evaluation_count += 1
                total = (overlay_index.logic_operations + base_operations
                         + overlay_operations)
                if total > freeze["resource_budget"]["max_logic_operations"]:
                    raise W02MorphologySuccessorDevCalibrationError(
                        "successor dev logic resource stop")
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
                "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_DEV_CALIBRATION_REPORT"),
            "artifact_version": W02_MORPH_SUCCESSOR_DEV_REPORT_VERSION,
            "base_logic_operations": base_operations,
            "candidate_artifact_manifest_sha256":
                freeze["candidate_artifact_manifest_sha256"],
            "candidate_index_row_count": candidate_index.row_count,
            "candidate_index_semantic_sha256": candidate_index.semantic_sha256,
            "candidate_semantic_sha256": freeze["candidate_semantic_sha256"],
            "code_freeze_sha256": freeze["code_freeze_sha256"],
            "compile_freeze_sha256": freeze["compile_freeze_sha256"],
            "dev_input_commitment": freeze["dev_input_commitment"],
            "dimension_results": dimensions,
            "evaluator_record_reads": evaluation_count,
            "family_counts": family_counts,
            "formal_dev_calibration_runs": 1 if formal else 0,
            "formal_private_evaluation_runs": 0,
            "formal_successor_transform_runs": 1,
            "formal_training_runs": 1,
            "generalized_candidate_count": generalized_candidates,
            "language_capability_mastered": 0,
            "language_readiness": 0,
            "logic_operations": (
                overlay_index.logic_operations + base_operations
                + overlay_operations),
            "max_generalized_candidates_per_observation": max_generalized,
            "next_action": (
                "W02_SHADOW_AUDIT" if formal and status == "PASS"
                else "W02_SUCCESSOR_DEV_FAILED_STOP" if formal
                else "W02_SUCCESSOR_DEV_CALIBRATION_FREEZE"),
            "observation_reads": evaluation_count,
            "old_dev_failure_report_file_sha256":
                freeze["old_dev_failure_report_file_sha256"],
            "overlay_artifact_manifest_sha256":
                freeze["overlay_artifact_manifest_sha256"],
            "overlay_inference_logic_operations": overlay_operations,
            "overlay_rule_row_count": overlay_index.row_count,
            "overlay_semantic_sha256": overlay_index.semantic_sha256,
            "private_family_registered": 0,
            "private_payload_reads": 0,
            "queried_span_count": queried_spans,
            "ranking_cache_entry_count": len(ranking_cache.values),
            "ranking_cache_hit_count": ranking_cache.hit_count,
            "ranking_cache_miss_count": ranking_cache.miss_count,
            "release_key": "PH2-D03-V2",
            "run_id": 1 if formal else 0,
            "run_scope": "FORMAL" if formal else "DEVELOPMENT_PREFLIGHT",
            "shadow_started": 0,
            "source_count": source_count,
            "source_identity_sha256": source_sha,
            "stage_key": "W-02",
            "status": status,
            "successor_transform_logic_operations":
                overlay_index.logic_operations,
            "teacher_calls": 0,
            "transport_bytes_read": input_bytes,
            "validated_layout_count": len(W02_DEV_LAYOUT_PATHS),
            "zero_write_audit": {
                "candidate_writes": 0,
                "companion_writes": 0,
                "core_writes": 0,
                "dev_owner_writes": 0,
                "evidence_writes": 0,
                "memory_writes": 0,
                "overlay_writes": 0,
                "use_writes": 0,
            },
        }
    finally:
        ranking_cache.close()
    if (_tree_sha256(candidate_root) != before_candidate
            or _tree_sha256(overlay_root) != before_overlay
            or _tree_sha256(dev.root) != before_dev):
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev calibration 产生非授权写入")
    _assert_expected_report(report, freeze)
    validate_v2_safe_report(report)
    return report


def run_w02_morphology_successor_dev_preflight(
        repository_root: str | Path,
        dev_root: str | Path,
        candidate_artifact_root: str | Path,
        overlay_artifact_root: str | Path,
        ) -> dict[str, object]:
    """在 freeze 前运行可重复开发预演，不增加正式次数。"""
    return _run_calibration(
        Path(repository_root).resolve(), dev_root, candidate_artifact_root,
        overlay_artifact_root, formal=False)


def run_w02_morphology_successor_dev_calibration(
        repository_root: str | Path,
        dev_root: str | Path,
        candidate_artifact_root: str | Path,
        overlay_artifact_root: str | Path,
        *,
        run_id: int = 1,
        ) -> dict[str, object]:
    """在公开 freeze 后执行唯一一次正式 successor dev calibration。"""
    if run_id != 1:
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev formal run_id 固定为 1")
    return _run_calibration(
        Path(repository_root).resolve(), dev_root, candidate_artifact_root,
        overlay_artifact_root, formal=True)


def publish_w02_morphology_successor_dev_calibration_report(
        repository_root: str | Path,
        external_report: str | Path,
        ) -> Path:
    """回读 Git 外正式报告并不可覆盖发布安全公开投影。"""
    repository = Path(repository_root).resolve()
    value = read_canonical_object(external_report)
    validate_v2_safe_report(value)
    freeze_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH)
    freeze = read_w02_morphology_successor_dev_calibration_freeze(repository)
    freeze_size, freeze_sha = _sha256_file(freeze_path)
    if (value.get("artifact_version")
            != W02_MORPH_SUCCESSOR_DEV_REPORT_VERSION
            or value.get("run_scope") != "FORMAL"
            or value.get("run_id") != 1
            or value.get("formal_dev_calibration_runs") != 1
            or value.get("formal_private_evaluation_runs") != 0
            or value.get("private_payload_reads") != 0
            or value.get("teacher_calls") != 0
            or value.get("code_freeze_sha256")
            != freeze["code_freeze_sha256"]):
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev 正式报告状态非法")
    _assert_expected_report(value, freeze)
    public = dict(value)
    public["dev_freeze_file_sha256"] = freeze_sha
    public["dev_freeze_size_bytes"] = freeze_size
    validate_v2_safe_report(public)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_DEV_REPORT_PATH).parts)
    write_immutable_json(public, target)
    if target.read_bytes() != canonical_json_bytes(public) + b"\n":
        raise W02MorphologySuccessorDevCalibrationError(
            "successor dev report 发布字节漂移")
    return target


__all__ = [
    "W02_MORPH_SUCCESSOR_DEV_CODE_PATHS",
    "W02_MORPH_SUCCESSOR_DEV_EXPECTED_COUNTS",
    "W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH",
    "W02_MORPH_SUCCESSOR_DEV_REPORT_PATH",
    "W02MorphologySuccessorDevCalibrationError",
    "build_w02_morphology_successor_dev_calibration_freeze",
    "publish_w02_morphology_successor_dev_calibration_freeze",
    "publish_w02_morphology_successor_dev_calibration_report",
    "read_w02_morphology_successor_dev_calibration_freeze",
    "run_w02_morphology_successor_dev_calibration",
    "run_w02_morphology_successor_dev_preflight",
]
