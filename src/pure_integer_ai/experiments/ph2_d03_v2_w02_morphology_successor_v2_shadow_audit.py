"""PH2-D03-V2 W-02 morphology successor V2 的无标签 shadow audit。"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _hash_value,
    _sha256_file,
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
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_shadow_audit import (
    W02_SHADOW_LAYOUTS,
    W02ShadowInputRoot,
    _audit_prediction,
    _gate,
    _light_prediction,
    _select_shadow_spans,
    _shadow_identity,
    iter_w02_shadow_records,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN,
    W02MorphologySuccessorV2Cache,
    predict_w02_morphology_successor_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_dev_calibration import (
    W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_PATH,
    W02_MORPH_SUCCESSOR_V2_DEV_REPORT_PATH,
    read_w02_morphology_successor_v2_dev_calibration_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
)


W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-SHADOW-AUDIT-FREEZE-V1")
W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-SHADOW-AUDIT-REPORT-V1")
W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_shadow_audit_freeze_v1.json")
W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_shadow_audit_report_v1.json")
W02_MORPH_SUCCESSOR_V2_SHADOW_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v2_shadow_audit.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_v2_shadow_audit.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v2_shadow_audit.py",
)
W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS = {
    "base_logic_operations": 5_478_480,
    "combined_added_candidate_count": 197_892,
    "full_route_observation_count": 4_497,
    "light_observation_count": 54_009,
    "logic_operations": 7_855_671,
    "max_combined_added_candidates_per_observation": 48,
    "max_v1_generalized_candidates_per_observation": 40,
    "max_v2_edge_candidates_per_observation": 8,
    "max_v2_edge_candidates_per_requested_span": 8,
    "observation_reads": 58_506,
    "queried_span_count": 8_994,
    "source_count": 50_322,
    "transport_bytes_read": 11_860_231,
    "v1_generalized_candidate_count": 167_002,
    "v1_overlay_inference_logic_operations": 256_503,
    "v1_overlay_ranking_cache_entry_count": 217,
    "v1_overlay_ranking_cache_hit_count": 8_777,
    "v1_overlay_ranking_cache_miss_count": 217,
    "v1_overlay_rule_row_count": 47_975,
    "v1_successor_transform_logic_operations": 1_574_251,
    "v2_edge_candidate_count": 30_890,
    "v2_overlay_inference_logic_operations": 541_826,
    "v2_overlay_ranking_cache_entry_count": 4_230,
    "v2_overlay_ranking_cache_hit_count": 4_764,
    "v2_overlay_ranking_cache_miss_count": 4_230,
    "v2_overlay_rule_row_count": 383,
    "v2_successor_transform_logic_operations": 4_611,
}
W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES = (
    ("W02-SHADOW-V2-CARRIER-ROUNDTRIP", 58_506),
    ("W02-SHADOW-V2-BOUNDARY-UNICODE", 58_506),
    ("W02-SHADOW-V2-ROUTED-EXACT", 4_497),
    ("W02-SHADOW-V2-V1-DUAL-SPAN", 4_497),
    ("W02-SHADOW-V2-EDGE-BOUNDED-RETENTION", 4_497),
)


# object-model: exception
class W02MorphologySuccessorV2ShadowAuditError(RuntimeError):
    """V2 shadow 输入、冻结、运行或公开投影不满足严格合同。"""


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(repository) or not target.is_file()):
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow repository file 非法")
    return target


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    for relative in W02_MORPH_SUCCESSOR_V2_SHADOW_CODE_PATHS:
        size, digest = _sha256_file(_repository_file(repository, relative))
        rows.append({"repository_file": relative, "sha256": digest,
                     "size_bytes": size})
    return rows, _hash_value(rows)


def _dependency_state(repository: Path) -> dict[str, object]:
    parent = read_w02_compile_freeze(repository)
    dev_freeze = read_w02_morphology_successor_v2_dev_calibration_freeze(
        repository)
    dev_freeze_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_V2_DEV_FREEZE_PATH)
    dev_report_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_V2_DEV_REPORT_PATH)
    dev_report = read_canonical_object(dev_report_path)
    if (dev_report.get("status") != "PASS"
            or dev_report.get("formal_dev_calibration_runs") != 1
            or dev_report.get("formal_private_evaluation_runs") != 0
            or dev_report.get("private_payload_reads") != 0
            or dev_report.get("teacher_calls") != 0
            or dev_report.get("dev_freeze_file_sha256")
            != _sha256_file(dev_freeze_path)[1]
            or dev_report.get("code_freeze_sha256")
            != dev_freeze.get("code_freeze_sha256")
            or dev_report.get("v1_candidates_retained") != 1
            or dev_report.get("max_v2_edge_candidates_per_requested_span")
            != W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN):
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow parent dev PASS 漂移")
    shadow_files = [
        _shadow_identity(parent, key).to_dict() for key in W02_SHADOW_LAYOUTS
    ]
    return {
        "compile_freeze_sha256": parent.sha256(),
        "dev_freeze_file_sha256": _sha256_file(dev_freeze_path)[1],
        "dev_report": dev_report,
        "dev_report_file_sha256": _sha256_file(dev_report_path)[1],
        "shadow_input_commitment": _hash_value(shadow_files),
        "shadow_input_files": shadow_files,
    }


def _development_identity(repository: Path) -> dict[str, object]:
    dependency = _dependency_state(repository)
    dev = dependency["dev_report"]
    code_rows, code_sha = _code_rows(repository)
    return {
        "candidate_artifact_manifest_sha256":
            dev["candidate_artifact_manifest_sha256"],
        "candidate_semantic_sha256": dev["candidate_semantic_sha256"],
        "code_files": code_rows,
        "code_freeze_sha256": code_sha,
        "compile_freeze_sha256": dependency["compile_freeze_sha256"],
        "dev_freeze_file_sha256": dependency["dev_freeze_file_sha256"],
        "dev_pass_report_file_sha256": dependency["dev_report_file_sha256"],
        "resource_budget": {"max_logic_operations": 9_000_000,
                            "max_records": 100_000},
        "shadow_input_commitment": dependency["shadow_input_commitment"],
        "shadow_input_files": dependency["shadow_input_files"],
        "v1_overlay_artifact_manifest_sha256":
            dev["v1_overlay_artifact_manifest_sha256"],
        "v1_overlay_semantic_sha256": dev["v1_overlay_semantic_sha256"],
        "v2_overlay_artifact_manifest_sha256":
            dev["v2_overlay_artifact_manifest_sha256"],
        "v2_overlay_semantic_sha256": dev["v2_overlay_semantic_sha256"],
    }


def build_w02_morphology_successor_v2_shadow_audit_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """冻结 V2 无标签 shadow 输入、五门审计和资源上界。"""
    repository = Path(repository_root).resolve()
    if not W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS:
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow exact counts 尚未冻结")
    identity = _development_identity(repository)
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_AUDIT_FREEZE"),
        "artifact_version": W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_VERSION,
        **identity,
        "expected_counts": dict(W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS),
        "expected_gates": [
            {"denominator": denominator, "failed": 0, "gate_key": name,
             "ne": 0, "numerator": denominator, "status": "PASS"}
            for name, denominator in W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES
        ],
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "formal_shadow_audit_runs": 0,
        "formal_successor_transform_runs": 1,
        "formal_successor_v2_transform_runs": 1,
        "formal_training_runs": 1,
        "label_reads": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": "W02_FORMAL_MORPHOLOGY_SUCCESSOR_V2_SHADOW_AUDIT",
        "private_family_registered": 0,
        "private_payload_reads": 0,
        "probe_policy": {
            "all_observations_carrier_boundary_unicode": 1,
            "all_routed_observations_full_exact": 1,
            "routed_overlay_spans": "FIRST_EXACT_AND_FIRST_NONEXACT_WIDTH_2_1_3",
            "threshold_reduction": 0,
            "v1_candidate_retention_required": 1,
            "v2_max_edge_candidates_per_span":
                W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN,
        },
        "release_key": "PH2-D03-V2",
        "shadow_started": 0,
        "stage_key": "W-02",
        "status": (
            "W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_AUDIT_FREEZE_COMPLETE"),
        "teacher_calls": 0,
    }


def publish_w02_morphology_successor_v2_shadow_audit_freeze(
        repository_root: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_v2_shadow_audit_freeze(repository)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    return target


def read_w02_morphology_successor_v2_shadow_audit_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    target = _repository_file(
        repository, W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_PATH)
    value = read_canonical_object(target)
    if value != build_w02_morphology_successor_v2_shadow_audit_freeze(
            repository):
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow freeze 与 live identity 漂移")
    return value


def _morphology_identity(item: object) -> tuple[object, ...]:
    return tuple(getattr(item, name) for name in (
        "start", "end", "form", "lemma", "upos", "feats_json",
        "support_count"))


def _audit_v2_extension(
        v1: object,
        v2: object,
        requested_spans: tuple[tuple[int, int], ...],
        ) -> tuple[bool, int]:
    before = {_morphology_identity(item)
              for item in v1.prediction.morphology_candidates}
    after = {_morphology_identity(item)
             for item in v2.prediction.morphology_candidates}
    added = after - before
    counts = {span: 0 for span in requested_spans}
    valid_spans = True
    for identity in added:
        span = (identity[0], identity[1])
        if span not in counts:
            valid_spans = False
            continue
        counts[span] += 1
    maximum = max(counts.values(), default=0)
    ok = (before.issubset(after)
          and len(added) == v2.edge_candidate_count
          and valid_spans
          and maximum <= W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN)
    return ok, maximum


def _expected_gate_projection() -> list[dict[str, object]]:
    return [
        {"denominator": denominator, "failed": 0, "gate_key": name,
         "ne": 0, "numerator": denominator, "status": "PASS"}
        for name, denominator in W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES
    ]


def _gate_projection(report: dict[str, object]) -> list[dict[str, object]]:
    gates = report.get("audit_results")
    if not isinstance(gates, list):
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow gates 非 list")
    return [{
        "denominator": row.get("denominator"),
        "failed": row.get("failed"),
        "gate_key": row.get("gate_key"),
        "ne": row.get("ne"),
        "numerator": row.get("numerator"),
        "status": row.get("status"),
    } for row in gates if isinstance(row, dict)]


def _assert_preflight(report: dict[str, object]) -> None:
    zero_write = report.get("zero_write_audit")
    if (_gate_projection(report) != _expected_gate_projection()
            or report.get("status") != "PASS"
            or report.get("label_reads") != 0
            or report.get("logic_operations") > 9_000_000
            or report.get("max_v2_edge_candidates_per_requested_span")
            > W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN
            or not isinstance(zero_write, dict)
            or any(value != 0 for value in zero_write.values())):
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow preflight gate 未闭合")


def _assert_expected(report: dict[str, object], freeze: dict[str, object]) -> None:
    for key, expected in freeze["expected_counts"].items():
        if report.get(key) != expected:
            raise W02MorphologySuccessorV2ShadowAuditError(
                f"successor V2 shadow expected count 漂移: {key}")
    if _gate_projection(report) != freeze["expected_gates"]:
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow audit gate 未全 PASS")


def _run_shadow(
        repository: Path,
        shadow_root: str | Path,
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        *,
        formal: bool,
        ) -> dict[str, object]:
    freeze = (read_w02_morphology_successor_v2_shadow_audit_freeze(repository)
              if formal else _development_identity(repository))
    parent = read_w02_compile_freeze(repository)
    shadow = W02ShadowInputRoot(Path(shadow_root))
    candidate_root = Path(candidate_artifact_root).resolve()
    v1_root = Path(v1_overlay_artifact_root).resolve()
    v2_root = Path(v2_overlay_artifact_root).resolve()
    before = (_tree_sha256(shadow.root), _tree_sha256(candidate_root),
              _tree_sha256(v1_root), _tree_sha256(v2_root))
    candidate_result = read_w02_candidate_artifact(candidate_root)
    v1_result = read_w02_morphology_overlay_artifact(v1_root)
    v2_result = read_w02_morphology_successor_v2_overlay_artifact(v2_root)
    if (candidate_result.artifact_manifest_sha256
            != freeze["candidate_artifact_manifest_sha256"]
            or candidate_result.candidate_semantic_sha256
            != freeze["candidate_semantic_sha256"]
            or v1_result.artifact_manifest_sha256
            != freeze["v1_overlay_artifact_manifest_sha256"]
            or v1_result.overlay_semantic_sha256
            != freeze["v1_overlay_semantic_sha256"]
            or v2_result.artifact_manifest_sha256
            != freeze["v2_overlay_artifact_manifest_sha256"]
            or v2_result.semantic_sha256
            != freeze["v2_overlay_semantic_sha256"]):
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow artifact identity 漂移")
    source_digests = []
    source_count = 0
    for record in iter_w02_shadow_records(parent, shadow, "SHADOW_SOURCE"):
        if not isinstance(record, SourceRefRecord):
            raise W02MorphologySuccessorV2ShadowAuditError(
                "successor V2 shadow SourceRef 类型错误")
        source_digests.append(_hash_value(record.stable_key.to_list()))
        source_count += 1
    v1_index = load_w02_morphology_overlay_index(v1_root)
    v2_index = load_w02_morphology_successor_v2_overlay_index(v2_root)
    if v1_index.dataset_keys != v2_index.dataset_keys:
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow route identity 漂移")
    counts = {
        "base_logic_operations": 0,
        "combined_added_candidate_count": 0,
        "full_route_observation_count": 0,
        "light_observation_count": 0,
        "max_combined_added_candidates_per_observation": 0,
        "max_v1_generalized_candidates_per_observation": 0,
        "max_v2_edge_candidates_per_observation": 0,
        "max_v2_edge_candidates_per_requested_span": 0,
        "observation_reads": 0,
        "queried_span_count": 0,
        "v1_generalized_candidate_count": 0,
        "v1_overlay_inference_logic_operations": 0,
        "v2_edge_candidate_count": 0,
        "v2_overlay_inference_logic_operations": 0,
    }
    passed = {name: 0 for name, _ in
              W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES}
    evidence = {name: [] for name, _ in
                W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES}
    v1_cache = W02MorphologyRankingCache.empty()
    v2_cache = W02MorphologySuccessorV2Cache.empty()
    try:
        with open_w02_candidate_predictor(candidate_root) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
            for layout_key in (
                    "SHADOW_TRAIN_OBSERVATION", "SHADOW_DEV_OBSERVATION"):
                for record in iter_w02_shadow_records(parent, shadow, layout_key):
                    if not isinstance(record, ObservationRecord):
                        raise W02MorphologySuccessorV2ShadowAuditError(
                            "successor V2 shadow Observation 类型错误")
                    routed = record.dataset_key.components in v1_index.dataset_keys
                    if routed:
                        base, operations = predict_w02_dev_observation(
                            candidate_index, record)
                        operations += 8
                        counts["full_route_observation_count"] += 1
                    else:
                        base, operations = _light_prediction(
                            candidate_index, record)
                        counts["light_observation_count"] += 1
                    counts["base_logic_operations"] += operations
                    counts["observation_reads"] += 1
                    carrier_ok, boundary_ok, morphology_ok, digest = (
                        _audit_prediction(
                            record, base, require_morphology=routed))
                    for index in (0, 1):
                        gate_name = W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES[
                            index][0]
                        evidence[gate_name].append(digest)
                    passed[W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES[0][0]] += int(
                        carrier_ok)
                    passed[W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES[1][0]] += int(
                        boundary_ok)
                    if not routed:
                        continue
                    exact_gate = W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES[2][0]
                    evidence[exact_gate].append(digest)
                    passed[exact_gate] += int(morphology_ok)
                    spans = _select_shadow_spans(base, v1_index.max_form_length)
                    v1 = predict_w02_morphology_successor(
                        v1_index, record, base, requested_spans=spans,
                        ranking_cache=v1_cache)
                    base_keys = {_morphology_identity(item)
                                 for item in base.morphology_candidates}
                    v1_keys = {_morphology_identity(item)
                               for item in v1.prediction.morphology_candidates}
                    v1_ok = (base_keys.issubset(v1_keys)
                             and v1.generalized_candidate_count <= 40)
                    v1_digest = _hash_value({
                        "base_prediction_sha256": _hash_value(base.to_dict()),
                        "generalized_candidate_count":
                            v1.generalized_candidate_count,
                        "observation_key": list(record.stable_key.components),
                        "requested_spans": [list(span) for span in spans],
                        "v1_ok": int(v1_ok),
                        "v1_prediction_sha256":
                            _hash_value(v1.prediction.to_dict()),
                    })
                    v1_gate = W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES[3][0]
                    evidence[v1_gate].append(v1_digest)
                    passed[v1_gate] += int(v1_ok)
                    v2 = predict_w02_morphology_successor_v2(
                        v2_index, record, v1, requested_spans=spans,
                        cache=v2_cache)
                    v2_ok, maximum = _audit_v2_extension(v1, v2, spans)
                    v2_digest = _hash_value({
                        "edge_candidate_count": v2.edge_candidate_count,
                        "max_edge_candidates_per_span": maximum,
                        "observation_key": list(record.stable_key.components),
                        "requested_spans": [list(span) for span in spans],
                        "v1_prediction_sha256":
                            _hash_value(v1.prediction.to_dict()),
                        "v2_ok": int(v2_ok),
                        "v2_prediction_sha256":
                            _hash_value(v2.prediction.to_dict()),
                    })
                    v2_gate = W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES[4][0]
                    evidence[v2_gate].append(v2_digest)
                    passed[v2_gate] += int(v2_ok)
                    counts["queried_span_count"] += len(spans)
                    counts["v1_generalized_candidate_count"] += (
                        v1.generalized_candidate_count)
                    counts["v1_overlay_inference_logic_operations"] += (
                        v1.logic_operations)
                    counts["v2_edge_candidate_count"] += v2.edge_candidate_count
                    counts["v2_overlay_inference_logic_operations"] += (
                        v2.logic_operations)
                    combined = (v1.generalized_candidate_count
                                + v2.edge_candidate_count)
                    counts["combined_added_candidate_count"] += combined
                    counts["max_combined_added_candidates_per_observation"] = max(
                        counts["max_combined_added_candidates_per_observation"],
                        combined)
                    counts["max_v1_generalized_candidates_per_observation"] = max(
                        counts["max_v1_generalized_candidates_per_observation"],
                        v1.generalized_candidate_count)
                    counts["max_v2_edge_candidates_per_observation"] = max(
                        counts["max_v2_edge_candidates_per_observation"],
                        v2.edge_candidate_count)
                    counts["max_v2_edge_candidates_per_requested_span"] = max(
                        counts["max_v2_edge_candidates_per_requested_span"],
                        maximum)
        audit_results = [
            _gate(name, denominator, passed[name], evidence[name])
            for name, denominator in
            W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES
        ]
        logic_operations = (
            v1_index.logic_operations + v2_index.logic_operations
            + counts["base_logic_operations"]
            + counts["v1_overlay_inference_logic_operations"]
            + counts["v2_overlay_inference_logic_operations"])
        report = {
            "artifact_kind": (
                "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_AUDIT_REPORT"),
            "artifact_version": W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_VERSION,
            "audit_results": audit_results,
            **counts,
            "candidate_artifact_manifest_sha256":
                freeze["candidate_artifact_manifest_sha256"],
            "candidate_semantic_sha256": freeze["candidate_semantic_sha256"],
            "code_freeze_sha256": freeze["code_freeze_sha256"],
            "compile_freeze_sha256": freeze["compile_freeze_sha256"],
            "dev_pass_report_file_sha256":
                freeze["dev_pass_report_file_sha256"],
            "formal_dev_calibration_runs": 1,
            "formal_private_evaluation_runs": 0,
            "formal_shadow_audit_runs": 1 if formal else 0,
            "formal_successor_transform_runs": 1,
            "formal_successor_v2_transform_runs": 1,
            "formal_training_runs": 1,
            "label_reads": 0,
            "language_capability_mastered": 0,
            "language_readiness": 0,
            "logic_operations": logic_operations,
            "next_action": (
                "W02_SUCCESSOR_V2_PRIVATE_FAMILY_REGISTRATION_FREEZE"
                if formal and all(row["status"] == "PASS"
                                  for row in audit_results)
                else "W02_SUCCESSOR_V2_SHADOW_FAILED_STOP" if formal
                else "W02_SUCCESSOR_V2_SHADOW_AUDIT_FREEZE"),
            "private_family_registered": 0,
            "private_payload_reads": 0,
            "release_key": "PH2-D03-V2",
            "run_id": 1 if formal else 0,
            "run_scope": "FORMAL" if formal else "DEVELOPMENT_PREFLIGHT",
            "shadow_input_commitment": freeze["shadow_input_commitment"],
            "shadow_started": 1 if formal else 0,
            "source_count": source_count,
            "source_identity_sha256": _hash_value(source_digests),
            "stage_key": "W-02",
            "status": ("PASS" if all(row["status"] == "PASS"
                                     for row in audit_results) else "FAIL"),
            "teacher_calls": 0,
            "transport_bytes_read": sum(
                _shadow_identity(parent, key).transport_size_bytes
                for key in W02_SHADOW_LAYOUTS),
            "v1_overlay_artifact_manifest_sha256":
                freeze["v1_overlay_artifact_manifest_sha256"],
            "v1_overlay_ranking_cache_entry_count": len(v1_cache.values),
            "v1_overlay_ranking_cache_hit_count": v1_cache.hit_count,
            "v1_overlay_ranking_cache_miss_count": v1_cache.miss_count,
            "v1_overlay_rule_row_count": v1_index.row_count,
            "v1_overlay_semantic_sha256": v1_index.semantic_sha256,
            "v1_successor_transform_logic_operations": v1_index.logic_operations,
            "v2_overlay_artifact_manifest_sha256":
                freeze["v2_overlay_artifact_manifest_sha256"],
            "v2_overlay_ranking_cache_entry_count": len(v2_cache.values),
            "v2_overlay_ranking_cache_hit_count": v2_cache.hit_count,
            "v2_overlay_ranking_cache_miss_count": v2_cache.miss_count,
            "v2_overlay_rule_row_count": v2_index.row_count,
            "v2_overlay_semantic_sha256": v2_index.semantic_sha256,
            "v2_successor_transform_logic_operations": v2_index.logic_operations,
            "zero_write_audit": {
                "candidate_writes": 0,
                "companion_writes": 0,
                "core_writes": 0,
                "evidence_writes": 0,
                "memory_writes": 0,
                "shadow_owner_writes": 0,
                "use_writes": 0,
                "v1_overlay_writes": 0,
                "v2_overlay_writes": 0,
            },
        }
    finally:
        v1_cache.close()
        v2_cache.close()
    after = (_tree_sha256(shadow.root), _tree_sha256(candidate_root),
             _tree_sha256(v1_root), _tree_sha256(v2_root))
    if after != before:
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow audit 产生非授权写入")
    _assert_preflight(report)
    if formal:
        _assert_expected(report, freeze)
    validate_v2_safe_report(report)
    return report


def run_w02_morphology_successor_v2_shadow_preflight(
        repository_root: str | Path,
        shadow_root: str | Path,
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        ) -> dict[str, object]:
    return _run_shadow(
        Path(repository_root).resolve(), shadow_root, candidate_artifact_root,
        v1_overlay_artifact_root, v2_overlay_artifact_root, formal=False)


def run_w02_morphology_successor_v2_shadow_audit(
        repository_root: str | Path,
        shadow_root: str | Path,
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        *,
        run_id: int = 1,
        ) -> dict[str, object]:
    if run_id != 1:
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow formal run_id 固定为 1")
    return _run_shadow(
        Path(repository_root).resolve(), shadow_root, candidate_artifact_root,
        v1_overlay_artifact_root, v2_overlay_artifact_root, formal=True)


def publish_w02_morphology_successor_v2_shadow_audit_report(
        repository_root: str | Path,
        external_report: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = read_canonical_object(external_report)
    validate_v2_safe_report(value)
    freeze_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_PATH)
    freeze = read_w02_morphology_successor_v2_shadow_audit_freeze(repository)
    freeze_size, freeze_sha = _sha256_file(freeze_path)
    if (value.get("artifact_version")
            != W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_VERSION
            or value.get("run_scope") != "FORMAL"
            or value.get("run_id") != 1
            or value.get("formal_shadow_audit_runs") != 1
            or value.get("formal_private_evaluation_runs") != 0
            or value.get("private_payload_reads") != 0
            or value.get("label_reads") != 0
            or value.get("teacher_calls") != 0
            or value.get("code_freeze_sha256")
            != freeze["code_freeze_sha256"]):
        raise W02MorphologySuccessorV2ShadowAuditError(
            "successor V2 shadow formal report 状态非法")
    _assert_preflight(value)
    _assert_expected(value, freeze)
    public = dict(value)
    public["shadow_freeze_file_sha256"] = freeze_sha
    public["shadow_freeze_size_bytes"] = freeze_size
    validate_v2_safe_report(public)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_PATH).parts)
    write_immutable_json(public, target)
    return target


__all__ = [
    "W02_MORPH_SUCCESSOR_V2_SHADOW_CODE_PATHS",
    "W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS",
    "W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_PATH",
    "W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_PATH",
    "W02MorphologySuccessorV2ShadowAuditError",
    "W02ShadowInputRoot",
    "build_w02_morphology_successor_v2_shadow_audit_freeze",
    "publish_w02_morphology_successor_v2_shadow_audit_freeze",
    "publish_w02_morphology_successor_v2_shadow_audit_report",
    "read_w02_morphology_successor_v2_shadow_audit_freeze",
    "run_w02_morphology_successor_v2_shadow_audit",
    "run_w02_morphology_successor_v2_shadow_preflight",
]
