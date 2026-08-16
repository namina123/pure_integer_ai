"""冻结 recovery-v9 GIMP 正式评测分母、维度与唯一运行合同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_runtime_gate import (
    NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND,
    NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS,
    V8_CANDIDATE_PACK_MANIFEST_SHA256,
    V8_CANDIDATE_PROGRAM_SHA256,
    V9_RUNTIME_GATE_QUERY_COUNT,
    V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE,
    V9_RUNTIME_SHAPES_SHA256,
    V9_SOURCE_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_V1")
NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_STATUS = (
    "GIMP_FULL_DENOMINATOR_AND_SIX_DIMENSIONS_FROZEN_LABEL_BLIND_NOT_RUN")

NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER = (
    "ORTHOGRAPHIC_ATOM_TRANSFER",
    "SOURCE_CONDITIONED_LEXICAL_TRANSFER",
    "IDENTITY_PRESERVATION",
    "STRUCTURE_AND_GENERATION_INTEGRITY",
    "RUNTIME_INDEXED_REFERENCE_EQUIVALENCE",
    "END_TO_END_COVERAGE",
)

NORMALIZATION_RECOVERY_V9_DIMENSIONS = {
    "ORTHOGRAPHIC_ATOM_TRANSFER": {
        "authorized_changed_exact_count_min": 1,
        "bearing": 1,
        "single_han_frozen_inventory_count": 793,
        "wrong_count_max": 0,
    },
    "SOURCE_CONDITIONED_LEXICAL_TRANSFER": {
        "authorized_changed_exact_count_min": 1,
        "bearing": 1,
        "official_source_exact_condition_required": 1,
        "unconditioned_execution_count_max": 0,
        "wrong_count_max": 0,
    },
    "IDENTITY_PRESERVATION": {
        "bearing": 1,
        "false_change_count_max": 0,
        "frozen_identity_inventory_count": 669,
        "identity_veto_exact_count_min": 1,
    },
    "STRUCTURE_AND_GENERATION_INTEGRITY": {
        "bearing": 1,
        "committed_structure_bearing_exact_count_min": 1,
        "exception_count_max": 0,
        "generation_hard_conjunct": 1,
        "partial_commit_count_max": 0,
        "structure_mismatch_count_max": 0,
    },
    "RUNTIME_INDEXED_REFERENCE_EQUIVALENCE": {
        "all_inputs_execute_twice": 1,
        "bearing": 1,
        "exception_count_max": 0,
        "indexed_reference_mismatch_count_max": 0,
        "production_enabled_must_equal": 0,
    },
    "END_TO_END_COVERAGE": {
        "bearing": 1,
        "changed_exact_count_min": 2,
        "exact_unknown_wrong_must_equal_full_inventory": 1,
        "full_frozen_inventory_count": 9_264,
        "wrong_count_max": 0,
    },
}

V9_SOURCE_CENSUS_SHA256 = (
    "5c66430011da0eb338379c45f0dbe2bb13ba012d0335fcdc79ce15a69685499c")
V9_PAIR_IDENTITIES_SHA256 = (
    "c593307e18f3aa79c7f74487cfb153921b81bce1d23029e34b2dfd987305b5f1")
V9_RUNTIME_GATE_SHA256 = (
    "ceb2c8abf740a8d6d2e4e0d5aef2452e3a7800f3c7cc4b287286ed8e77822207")

_EXPECTED_BUCKETS = {
    "contains_han_both_count": 8_940,
    "equal_length_count": 4_579,
    "evaluation_eligible_count": 8_924,
    "identity_count": 669,
    "input_conflict_count": 113,
    "nonidentity_count": 8_595,
    "single_han_difference_count": 793,
    "structure_equal_count": 9_248,
    "structure_unequal_count": 16,
    "variable_length_count": 4_685,
}
_EXPECTED_DOMAIN_COUNTS = {
    "po": 5_982,
    "po-libgimp": 668,
    "po-plug-ins": 1_988,
    "po-python": 223,
    "po-script-fu": 333,
    "po-tags": 2,
    "po-tips": 27,
    "po-windows-installer": 41,
}


def _sha256(payload: bytes) -> str:
    """返回commitment或固定输入的SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise BroadQaExternalDataError(f"v9 commitment {label} 非法")
    return value


def _canonical_json(path: Path, *, expected_sha256: str,
                    label: str) -> dict[str, object]:
    """读取一份规范单行JSON并核对固定SHA。"""
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v9 commitment {label} 不可读") from error
    if (_sha256(encoded) != _sha_value(expected_sha256, label=label)
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(f"v9 commitment {label} identity漂移")
    return stored


def _source_inputs(
        source_pack_dir: str | Path,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """只读source manifest与aggregate census，不打开其他payload。"""
    root = Path(source_pack_dir).resolve()
    manifest = _canonical_json(
        root / "manifest.json",
        expected_sha256=V9_SOURCE_PACK_MANIFEST_SHA256,
        label="source manifest")
    census = _canonical_json(
        root / "source-census.jsonl",
        expected_sha256=V9_SOURCE_CENSUS_SHA256,
        label="source census")
    summary = census.get("parser_summary")
    files = manifest.get("files")
    pair_files = [item for item in files if isinstance(item, dict)
                  and item.get("relative_path") == "pair-identities.jsonl"] if (
                      isinstance(files, list)) else []
    census_files = [item for item in files if isinstance(item, dict)
                    and item.get("relative_path") == "source-census.jsonl"] if (
                        isinstance(files, list)) else []
    if (manifest.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_PACK_V1"
            or manifest.get("status")
            != "GIMP_RAW_AND_LABEL_FREE_IDENTITY_FROZEN_NOT_FORMAL"
            or manifest.get("production_enabled") != 0
            or manifest.get("mastery_claimed") != 0
            or manifest.get("label_or_translation_surface_published") != 0
            or len(pair_files) != 1
            or pair_files[0].get("record_count") != 9_264
            or pair_files[0].get("sha256") != V9_PAIR_IDENTITIES_SHA256
            or len(census_files) != 1
            or census_files[0].get("record_count") != 1
            or census_files[0].get("sha256") != V9_SOURCE_CENSUS_SHA256
            or census.get("record_kind")
            != "V9_GIMP_SOURCE_PACK_CENSUS_V1"
            or census.get("label_free_identity_count") != 9_264
            or census.get("runtime_shape_count") != 9_264
            or not isinstance(summary, dict)):
        raise BroadQaExternalDataError("v9 commitment source boundary漂移")
    buckets = {
        "contains_han_both_count": summary.get("contains_han_both_count"),
        "equal_length_count": summary.get("equal_length_pair_count"),
        "evaluation_eligible_count": summary.get(
            "v9_evaluation_eligible_pair_count"),
        "identity_count": summary.get("identity_pair_count"),
        "input_conflict_count": summary.get("input_conflict_count"),
        "nonidentity_count": summary.get("nonidentity_pair_count"),
        "single_han_difference_count": summary.get(
            "single_han_difference_count"),
        "structure_equal_count": summary.get("structure_equal_count"),
        "structure_unequal_count": (
            summary.get("plain_pair_count", 0)
            - summary.get("structure_equal_count", 0)),
        "variable_length_count": summary.get("variable_length_pair_count"),
    }
    if (buckets != _EXPECTED_BUCKETS
            or summary.get("domain_pair_counts") != _EXPECTED_DOMAIN_COUNTS
            or summary.get("plain_pair_count") != 9_264):
        raise BroadQaExternalDataError("v9 commitment denominator aggregate漂移")
    return manifest, census


def _runtime_gate(value: str | Path) -> dict[str, object]:
    """只读已封存gate aggregate，不重读candidate或重跑workload。"""
    report = _canonical_json(
        Path(value).resolve() / "runtime-gate.json",
        expected_sha256=V9_RUNTIME_GATE_SHA256,
        label="runtime gate")
    profile = report.get("profile")
    aggregate = profile.get("aggregate") if isinstance(profile, dict) else None
    budget = profile.get("budget") if isinstance(profile, dict) else None
    if (report.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND
            or report.get("status")
            != NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS
            or report.get("source_pack_manifest_sha256")
            != V9_SOURCE_PACK_MANIFEST_SHA256
            or report.get("source_runtime_shapes_sha256")
            != V9_RUNTIME_SHAPES_SHA256
            or report.get("candidate_pack_manifest_sha256")
            != V8_CANDIDATE_PACK_MANIFEST_SHA256
            or report.get("candidate_program_sha256")
            != V8_CANDIDATE_PROGRAM_SHA256
            or report.get("formal_guard_write_count") != 0
            or report.get("formal_label_read_count") != 0
            or report.get("source_raw_archive_read_count") != 0
            or report.get("source_translation_surface_read_count") != 0
            or not isinstance(profile, dict)
            or profile.get("gate_outcome") != "PASS"
            or not isinstance(aggregate, dict)
            or not isinstance(budget, dict)
            or aggregate.get("query_count") != V9_RUNTIME_GATE_QUERY_COUNT
            or aggregate.get("total_wall_ns", 0)
            >= V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE
            or any(aggregate.get(name) != 0 for name in (
                "exception_count", "indexed_reference_mismatch_count",
                "indexed_repeat_mismatch_count", "partial_commit_count",
                "production_enabled_count", "structure_mismatch_count"))
            or budget.get("query_count") != V9_RUNTIME_GATE_QUERY_COUNT
            or budget.get("total_wall_ns_max_exclusive")
            != V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE):
        raise BroadQaExternalDataError("v9 commitment runtime gate漂移")
    return report


def build_normalization_recovery_v9_evaluation_commitment(
        *, source_manifest: dict[str, object],
        source_census: dict[str, object],
        runtime_gate: dict[str, object],
        ) -> dict[str, object]:
    """从固定aggregate输入冻结GIMP全分母正式合同。"""
    if (source_manifest.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_PACK_V1"
            or source_census.get("record_kind")
            != "V9_GIMP_SOURCE_PACK_CENSUS_V1"
            or runtime_gate.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND):
        raise BroadQaExternalDataError("v9 commitment build输入漂移")
    denominator = {
        "aggregate_buckets": _EXPECTED_BUCKETS,
        "domain_pair_counts": _EXPECTED_DOMAIN_COUNTS,
        "eligible_subset_cannot_replace_full_denominator": 1,
        "identity_artifact": {
            "record_count": 9_264,
            "relative_path": "pair-identities.jsonl",
            "sha256": V9_PAIR_IDENTITIES_SHA256,
        },
        "label_blind": 1,
        "record_count": 9_264,
        "source_census_sha256": V9_SOURCE_CENSUS_SHA256,
        "source_pack_manifest_sha256": V9_SOURCE_PACK_MANIFEST_SHA256,
    }
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_KIND,
        "candidate_or_code_read_count": 0,
        "denominator": denominator,
        "dimension_order": list(NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER),
        "dimensions": NORMALIZATION_RECOVERY_V9_DIMENSIONS,
        "format_version": 1,
        "formal_contract": {
            "candidate_applicability_cannot_shrink_denominator": 1,
            "candidate_code_family_freeze_required_before_label_read": 1,
            "denominator_or_threshold_change_after_publish_allowed": 0,
            "formal_run_count_max": 1,
            "guard_write_required_before_archive_or_label_read": 1,
            "host_candidate_and_publication_write_isolation_required": 1,
            "individual_label_publication_allowed": 0,
            "missing_required_evidence_outcome": "NE",
            "overall_rule": "FAIL_DOMINATES_NE_DOMINATES_PASS",
            "retry_after_any_terminal_or_post_guard_exception_allowed": 0,
            "wrong_committed_output_outcome": "FAIL",
        },
        "gimp_identity_raw_or_translation_read_count": 0,
        "judgements": ["EXACT", "UNKNOWN", "WRONG"],
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_gate_manifest_read_count": 1,
        "runtime_gate_sha256": V9_RUNTIME_GATE_SHA256,
        "source_census_read_count": 1,
        "source_manifest_read_count": 1,
        "status": NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_STATUS,
        "teacher_api_llm_call_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式commitment工作根位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v9 commitment run root必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析输入输出并限制其仍位于K盘run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"v9 commitment {label}越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个artifact根是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _derive(
        source_pack_dir: Path, runtime_gate_dir: Path,
        ) -> dict[str, object]:
    """严格读取三个aggregate输入并重建commitment。"""
    manifest, census = _source_inputs(source_pack_dir)
    gate = _runtime_gate(runtime_gate_dir)
    return build_normalization_recovery_v9_evaluation_commitment(
        source_manifest=manifest, source_census=census, runtime_gate=gate)


def publish_normalization_recovery_v9_evaluation_commitment(
        *, run_root: str | Path,
        source_pack_dir: str | Path,
        runtime_gate_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布标签盲v9正式commitment。"""
    root = _require_k_root(run_root)
    source = _within(root, source_pack_dir, label="source pack")
    gate = _within(root, runtime_gate_dir, label="runtime gate")
    target = _within(root, target_dir, label="target")
    paths = (source, gate, target)
    if (target.exists() or not source.is_dir() or not gate.is_dir()
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError("v9 commitment path非法")
    commitment = _derive(source, gate)
    target.mkdir()
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(commitment))
    return {**commitment, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v9_evaluation_commitment(
        source_dir: str | Path, *,
        source_pack_dir: str | Path,
        runtime_gate_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """严格回读v9 commitment并重建全部确定字段。"""
    stored = _canonical_json(
        Path(source_dir).resolve() / "manifest.json",
        expected_sha256=expected_manifest_sha256,
        label="commitment")
    expected = _derive(
        Path(source_pack_dir).resolve(), Path(runtime_gate_dir).resolve())
    if not strict_json_equal(stored, expected):
        raise BroadQaExternalDataError("v9 commitment fields漂移")
    return {**stored, "manifest_sha256": expected_manifest_sha256}


__all__ = [
    "NORMALIZATION_RECOVERY_V9_DIMENSIONS",
    "NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER",
    "NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_KIND",
    "NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_STATUS",
    "V9_RUNTIME_GATE_SHA256",
    "build_normalization_recovery_v9_evaluation_commitment",
    "publish_normalization_recovery_v9_evaluation_commitment",
    "read_normalization_recovery_v9_evaluation_commitment",
]
