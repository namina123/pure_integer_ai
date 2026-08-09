"""PH2-D03-V2 W-02 morphology successor 的无标签 shadow audit。"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path, PurePosixPath
from typing import Iterator
import unicodedata

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02_CAPABILITY_OOV_BOUNDARY_LATTICE,
    W02_CAPABILITY_UNICODE_ANALYSIS,
    W02CandidatePrediction,
    W02UnicodeUnit,
    boundary_lattice,
    generate_with_carrier_rules,
    observe_w02_carrier,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02FileFreeze,
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
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_dev_calibration import (
    W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH,
    W02_MORPH_SUCCESSOR_DEV_REPORT_PATH,
    read_w02_morphology_successor_dev_calibration_freeze,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    parse_canonical_json_bytes,
)


W02_MORPH_SUCCESSOR_SHADOW_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-SHADOW-AUDIT-FREEZE-V1")
W02_MORPH_SUCCESSOR_SHADOW_REPORT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-SHADOW-AUDIT-REPORT-V1")
W02_MORPH_SUCCESSOR_SHADOW_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_shadow_audit_freeze_v1.json")
W02_MORPH_SUCCESSOR_SHADOW_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_shadow_audit_report_v1.json")
W02_MORPH_SUCCESSOR_SHADOW_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_shadow_audit.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_shadow_audit.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_shadow_audit.py",
)
W02_SHADOW_LAYOUTS = {
    "SHADOW_SOURCE": "source/source_refs.jsonl.gz",
    "SHADOW_TRAIN_OBSERVATION": "observations/train.jsonl.gz",
    "SHADOW_DEV_OBSERVATION": "observations/dev.jsonl.gz",
}
W02_SHADOW_EXPECTED_COUNTS = {
    "base_logic_operations": 5_478_480,
    "full_route_observation_count": 4_497,
    "generalized_candidate_count": 167_002,
    "light_observation_count": 54_009,
    "logic_operations": 7_309_234,
    "max_generalized_candidates_per_observation": 40,
    "observation_reads": 58_506,
    "overlay_inference_logic_operations": 256_503,
    "queried_span_count": 8_994,
    "ranking_cache_entry_count": 217,
    "ranking_cache_hit_count": 8_777,
    "ranking_cache_miss_count": 217,
    "source_count": 50_322,
    "successor_transform_logic_operations": 1_574_251,
    "transport_bytes_read": 11_860_231,
}
W02_SHADOW_EXPECTED_GATES = (
    ("W02-SHADOW-CARRIER-ROUNDTRIP", 58_506),
    ("W02-SHADOW-BOUNDARY-UNICODE", 58_506),
    ("W02-SHADOW-ROUTED-EXACT", 4_497),
    ("W02-SHADOW-OVERLAY-DUAL-SPAN", 4_497),
)


# object-model: exception
class W02MorphologySuccessorShadowAuditError(RuntimeError):
    """Shadow 输入、预测、冻结或公开报告不满足严格合同。"""


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(repository) or not target.is_file()):
        raise W02MorphologySuccessorShadowAuditError(
            "shadow repository file 非法")
    return target


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    for relative in W02_MORPH_SUCCESSOR_SHADOW_CODE_PATHS:
        size, digest = _sha256_file(_repository_file(repository, relative))
        rows.append({"repository_file": relative, "sha256": digest,
                     "size_bytes": size})
    return rows, _hash_value(rows)


def _shadow_identity(parent: object, layout_key: str) -> W02FileFreeze:
    if layout_key not in W02_SHADOW_LAYOUTS:
        raise W02MorphologySuccessorShadowAuditError(
            "shadow layout 未注册")
    matches = tuple(item for item in parent.files
                    if item.layout_key == layout_key)
    if len(matches) != 1:
        raise W02MorphologySuccessorShadowAuditError(
            "shadow layout freeze 不唯一")
    item = matches[0]
    expected_split = {
        "SHADOW_SOURCE": "",
        "SHADOW_TRAIN_OBSERVATION": "train",
        "SHADOW_DEV_OBSERVATION": "dev",
    }[layout_key]
    expected_kind = "source_ref" if layout_key == "SHADOW_SOURCE" else "observation"
    if (item.root_key != "SHADOW_AUDIT_ROOT"
            or item.record_kind != expected_kind
            or item.split != expected_split
            or item.storage_relative_path != W02_SHADOW_LAYOUTS[layout_key]):
        raise W02MorphologySuccessorShadowAuditError(
            "shadow layout 与 compile freeze 漂移")
    return item


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02ShadowInputRoot:
    """只允许指向冻结的 shadow-audit owner 根。"""

    root: Path

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        object.__setattr__(self, "root", root)
        if not root.is_dir() or root.name != "shadow-audit" or root.is_symlink():
            raise W02MorphologySuccessorShadowAuditError(
                "shadow root 身份非法")


def _safe_shadow_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise W02MorphologySuccessorShadowAuditError(
                "shadow 输入不得经过 symlink")
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W02MorphologySuccessorShadowAuditError(
            "shadow 输入文件缺失或逃逸")
    return target


def iter_w02_shadow_records(
        parent: object,
        shadow: W02ShadowInputRoot,
        layout_key: str,
        ) -> Iterator[object]:
    """逐条读取 shadow gzip，并在 EOF 闭合 transport/content 身份。"""
    identity = _shadow_identity(parent, layout_key)
    target = _safe_shadow_file(shadow.root, W02_SHADOW_LAYOUTS[layout_key])
    size, digest = _sha256_file(target)
    if size != identity.transport_size_bytes or digest != identity.transport_sha256:
        raise W02MorphologySuccessorShadowAuditError(
            "shadow transport identity 漂移")
    content_digest = hashlib.sha256()
    content_size = 0
    count = 0
    first_key: tuple[int, ...] | None = None
    last_key: tuple[int, ...] | None = None
    previous_key: tuple[int, ...] | None = None
    try:
        with target.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise W02MorphologySuccessorShadowAuditError(
                            f"shadow JSONL 第 {line_number} 行换行非法")
                    content_digest.update(line)
                    content_size += len(line)
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = validate_v2_record(value)
                    if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                        raise W02MorphologySuccessorShadowAuditError(
                            "shadow record kind 漂移")
                    if identity.split and getattr(record, "split", "") != identity.split:
                        raise W02MorphologySuccessorShadowAuditError(
                            "shadow split 漂移")
                    key = record.stable_key.components
                    if previous_key is not None and key <= previous_key:
                        raise W02MorphologySuccessorShadowAuditError(
                            "shadow stable key 未严格排序")
                    previous_key = key
                    first_key = key if first_key is None else first_key
                    last_key = key
                    count += 1
                    yield record
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, W02MorphologySuccessorShadowAuditError):
            raise
        raise W02MorphologySuccessorShadowAuditError(
            "shadow gzip/JSONL 读取失败") from error
    if (count != identity.record_count
            or content_size != identity.content_size_bytes
            or content_digest.hexdigest() != identity.content_sha256
            or first_key != identity.first_record_key
            or last_key != identity.last_record_key):
        raise W02MorphologySuccessorShadowAuditError(
            "shadow content identity 漂移")


def _light_prediction(index: object, observation: ObservationRecord) -> tuple[W02CandidatePrediction, int]:
    """全量 shadow 只跑 carrier/boundary/Unicode，不扫描无关词形 span。"""
    observed = observe_w02_carrier(observation)
    rules = index.carrier_rules.get(observed.carrier_kind, ())
    generation = generate_with_carrier_rules(
        rules, carrier_kind=observed.carrier_kind, surface=observed.surface)
    if W02_CAPABILITY_OOV_BOUNDARY_LATTICE in index.capabilities and index.oov_lengths:
        points = boundary_lattice(
            observed.surface, observed_unit_lengths=index.oov_lengths)
    else:
        points = (0,) if not observed.surface else (0, len(observed.surface))
    unicode_units: tuple[W02UnicodeUnit, ...] = ()
    if W02_CAPABILITY_UNICODE_ANALYSIS in index.capabilities:
        unicode_units = tuple(W02UnicodeUnit(
            ord(char), unicodedata.category(char), unicodedata.combining(char))
            for char in observed.surface)
    status = {"GENERATED": "PREDICTED", "AMBIGUOUS": "AMBIGUOUS",
              "UNKNOWN": "UNKNOWN"}[generation.status]
    prediction = W02CandidatePrediction(
        observed.observation_key, status, generation, points,
        unicode_units, (), index.capabilities)
    operations = (len(observed.surface) + len(rules) + len(points)
                  + len(unicode_units) + 8)
    return prediction, operations


def _select_shadow_spans(
        base: W02CandidatePrediction,
        max_form_length: int,
        ) -> tuple[tuple[int, int], ...]:
    """固定选择一个 exact span 和一个非 exact 邻域 span，不使用标签。"""
    exact = sorted({(item.start, item.end)
                    for item in base.morphology_candidates})
    points = base.boundary_lattice
    possible: set[tuple[int, int]] = set()
    for width in (2, 1, 3):
        for position in range(max(0, len(points) - width)):
            start = points[position]
            end = points[position + width]
            if (0 <= start < end <= len(base.generation.surface)
                    and end - start <= max_form_length):
                possible.add((start, end))
    exact_set = set(exact)
    nonexact = next((span for span in sorted(possible)
                     if span not in exact_set), None)
    exact_span = exact[0] if exact else None
    spans = tuple(sorted({span for span in (exact_span, nonexact)
                          if span is not None}))
    if len(spans) != 2:
        raise W02MorphologySuccessorShadowAuditError(
            "shadow dual-span probe 无法闭合")
    return spans


def _audit_prediction(
        observation: ObservationRecord,
        prediction: W02CandidatePrediction,
        *,
        require_morphology: bool,
        ) -> tuple[bool, bool, bool, str]:
    language = observation.typed_payload.to_value()["language_payload"]
    carrier_ok = (
        prediction.status == "PREDICTED"
        and prediction.generation.carrier_serialization
        == language["carrier_serialization"]
        and prediction.generation.content_span_start
        == language["content_span_start"]
        and prediction.generation.content_span_end
        == language["content_span_end"])
    surface = prediction.generation.surface
    points = prediction.boundary_lattice
    boundary_unicode_ok = (
        tuple(sorted(set(points))) == points
        and 0 in points and len(surface) in points
        and len(prediction.unicode_units) == len(surface)
        and all((unit.code_point, unit.category, unit.combining_class) == (
            ord(char), unicodedata.category(char), unicodedata.combining(char))
            for unit, char in zip(prediction.unicode_units, surface, strict=True)))
    morphology_ok = (not require_morphology or all(
        item.start in points and item.end in points
        and surface[item.start:item.end] == item.form
        and 0 <= item.start < item.end <= len(surface)
        for item in prediction.morphology_candidates))
    evidence = _hash_value({
        "boundary_unicode_ok": int(boundary_unicode_ok),
        "carrier_ok": int(carrier_ok),
        "morphology_ok": int(morphology_ok),
        "observation_key": list(observation.stable_key.components),
        "prediction_sha256": _hash_value(prediction.to_dict()),
    })
    return carrier_ok, boundary_unicode_ok, morphology_ok, evidence


def _gate(name: str, denominator: int, passed: int, digests: list[str]) -> dict[str, object]:
    failed = denominator - passed
    return {
        "denominator": denominator,
        "evidence_sha256": _hash_value(digests),
        "failed": failed,
        "gate_key": name,
        "ne": 0,
        "numerator": passed,
        "status": "PASS" if failed == 0 and denominator > 0 else "FAIL",
    }


def _dependency_state(repository: Path) -> dict[str, object]:
    parent = read_w02_compile_freeze(repository)
    dev_freeze = read_w02_morphology_successor_dev_calibration_freeze(repository)
    dev_freeze_path = _repository_file(repository, W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH)
    dev_report_path = _repository_file(repository, W02_MORPH_SUCCESSOR_DEV_REPORT_PATH)
    dev_report = read_canonical_object(dev_report_path)
    if (dev_report.get("status") != "PASS"
            or dev_report.get("formal_dev_calibration_runs") != 1
            or dev_report.get("formal_private_evaluation_runs") != 0
            or dev_report.get("private_payload_reads") != 0
            or dev_report.get("teacher_calls") != 0
            or dev_report.get("dev_freeze_file_sha256")
            != _sha256_file(dev_freeze_path)[1]
            or dev_report.get("code_freeze_sha256")
            != dev_freeze.get("code_freeze_sha256")):
        raise W02MorphologySuccessorShadowAuditError(
            "shadow parent dev PASS 漂移")
    shadow_files = [
        _shadow_identity(parent, key).to_dict() for key in W02_SHADOW_LAYOUTS
    ]
    return {
        "compile_freeze_sha256": parent.sha256(),
        "dev_report": dev_report,
        "dev_report_file_sha256": _sha256_file(dev_report_path)[1],
        "dev_freeze_file_sha256": _sha256_file(dev_freeze_path)[1],
        "shadow_input_commitment": _hash_value(shadow_files),
        "shadow_input_files": shadow_files,
    }


def build_w02_morphology_successor_shadow_audit_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """冻结无标签 shadow 输入、分解式全量审计和资源门。"""
    repository = Path(repository_root).resolve()
    dependency = _dependency_state(repository)
    dev = dependency["dev_report"]
    code_rows, code_sha = _code_rows(repository)
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_SHADOW_AUDIT_FREEZE"),
        "artifact_version": W02_MORPH_SUCCESSOR_SHADOW_FREEZE_VERSION,
        "candidate_artifact_manifest_sha256":
            dev["candidate_artifact_manifest_sha256"],
        "candidate_semantic_sha256": dev["candidate_semantic_sha256"],
        "code_files": code_rows,
        "code_freeze_sha256": code_sha,
        "compile_freeze_sha256": dependency["compile_freeze_sha256"],
        "dev_freeze_file_sha256": dependency["dev_freeze_file_sha256"],
        "dev_pass_report_file_sha256": dependency["dev_report_file_sha256"],
        "expected_counts": dict(W02_SHADOW_EXPECTED_COUNTS),
        "expected_gates": [
            {"denominator": denominator, "failed": 0, "gate_key": name,
             "ne": 0, "numerator": denominator, "status": "PASS"}
            for name, denominator in W02_SHADOW_EXPECTED_GATES
        ],
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "formal_shadow_audit_runs": 0,
        "formal_successor_transform_runs": 1,
        "formal_training_runs": 1,
        "label_reads": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": "W02_FORMAL_MORPHOLOGY_SUCCESSOR_SHADOW_AUDIT",
        "overlay_artifact_manifest_sha256":
            dev["overlay_artifact_manifest_sha256"],
        "overlay_semantic_sha256": dev["overlay_semantic_sha256"],
        "private_family_registered": 0,
        "private_payload_reads": 0,
        "probe_policy": {
            "all_observations_carrier_boundary_unicode": 1,
            "all_routed_observations_full_exact": 1,
            "routed_overlay_spans": "FIRST_EXACT_AND_FIRST_NONEXACT_WIDTH_2_1_3",
            "threshold_reduction": 0,
        },
        "release_key": "PH2-D03-V2",
        "resource_budget": {"max_logic_operations": 9_000_000,
                            "max_records": 100_000},
        "shadow_input_commitment": dependency["shadow_input_commitment"],
        "shadow_input_files": dependency["shadow_input_files"],
        "shadow_started": 0,
        "stage_key": "W-02",
        "status": "W02_MORPHOLOGY_SUCCESSOR_SHADOW_AUDIT_FREEZE_COMPLETE",
        "teacher_calls": 0,
    }


def publish_w02_morphology_successor_shadow_audit_freeze(
        repository_root: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_shadow_audit_freeze(repository)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_SHADOW_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    return target


def read_w02_morphology_successor_shadow_audit_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    target = _repository_file(repository, W02_MORPH_SUCCESSOR_SHADOW_FREEZE_PATH)
    value = read_canonical_object(target)
    if value != build_w02_morphology_successor_shadow_audit_freeze(repository):
        raise W02MorphologySuccessorShadowAuditError(
            "shadow freeze 与 live identity 漂移")
    return value


def _assert_expected(report: dict[str, object], freeze: dict[str, object]) -> None:
    for key, expected in freeze["expected_counts"].items():
        if report.get(key) != expected:
            raise W02MorphologySuccessorShadowAuditError(
                f"shadow expected count 漂移: {key}")
    gates = report.get("audit_results")
    if not isinstance(gates, list):
        raise W02MorphologySuccessorShadowAuditError("shadow gates 非 list")
    projection = [{
        "denominator": row.get("denominator"), "failed": row.get("failed"),
        "gate_key": row.get("gate_key"), "ne": row.get("ne"),
        "numerator": row.get("numerator"), "status": row.get("status"),
    } for row in gates if isinstance(row, dict)]
    if projection != freeze["expected_gates"]:
        raise W02MorphologySuccessorShadowAuditError(
            "shadow audit gate 未全 PASS")


def _run_shadow(
        repository: Path,
        shadow_root: str | Path,
        candidate_artifact_root: str | Path,
        overlay_artifact_root: str | Path,
        *,
        formal: bool,
        ) -> dict[str, object]:
    freeze = (read_w02_morphology_successor_shadow_audit_freeze(repository)
              if formal else
              build_w02_morphology_successor_shadow_audit_freeze(repository))
    parent = read_w02_compile_freeze(repository)
    shadow = W02ShadowInputRoot(Path(shadow_root))
    candidate_root = Path(candidate_artifact_root).resolve()
    overlay_root = Path(overlay_artifact_root).resolve()
    before = (_tree_sha256(shadow.root), _tree_sha256(candidate_root),
              _tree_sha256(overlay_root))
    candidate_result = read_w02_candidate_artifact(candidate_root)
    overlay_result = read_w02_morphology_overlay_artifact(overlay_root)
    if (candidate_result.artifact_manifest_sha256
            != freeze["candidate_artifact_manifest_sha256"]
            or candidate_result.candidate_semantic_sha256
            != freeze["candidate_semantic_sha256"]
            or overlay_result.artifact_manifest_sha256
            != freeze["overlay_artifact_manifest_sha256"]
            or overlay_result.overlay_semantic_sha256
            != freeze["overlay_semantic_sha256"]):
        raise W02MorphologySuccessorShadowAuditError(
            "shadow artifact identity 漂移")
    source_digests = []
    source_count = 0
    for record in iter_w02_shadow_records(parent, shadow, "SHADOW_SOURCE"):
        if not isinstance(record, SourceRefRecord):
            raise W02MorphologySuccessorShadowAuditError(
                "shadow SourceRef 类型错误")
        source_digests.append(_hash_value(record.stable_key.to_list()))
        source_count += 1
    overlay_index = load_w02_morphology_overlay_index(overlay_root)
    counts = {
        "base_logic_operations": 0,
        "full_route_observation_count": 0,
        "generalized_candidate_count": 0,
        "light_observation_count": 0,
        "max_generalized_candidates_per_observation": 0,
        "observation_reads": 0,
        "overlay_inference_logic_operations": 0,
        "queried_span_count": 0,
    }
    passed = {name: 0 for name, _ in W02_SHADOW_EXPECTED_GATES}
    evidence = {name: [] for name, _ in W02_SHADOW_EXPECTED_GATES}
    cache = W02MorphologyRankingCache.empty()
    try:
        with open_w02_candidate_predictor(candidate_root) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
            for layout_key in (
                    "SHADOW_TRAIN_OBSERVATION", "SHADOW_DEV_OBSERVATION"):
                for record in iter_w02_shadow_records(parent, shadow, layout_key):
                    if not isinstance(record, ObservationRecord):
                        raise W02MorphologySuccessorShadowAuditError(
                            "shadow Observation 类型错误")
                    routed = record.dataset_key.components in overlay_index.dataset_keys
                    if routed:
                        base, operations = predict_w02_dev_observation(
                            candidate_index, record)
                        operations += 8
                        counts["full_route_observation_count"] += 1
                    else:
                        base, operations = _light_prediction(candidate_index, record)
                        counts["light_observation_count"] += 1
                    counts["base_logic_operations"] += operations
                    counts["observation_reads"] += 1
                    carrier_ok, boundary_ok, morphology_ok, digest = (
                        _audit_prediction(record, base,
                                          require_morphology=routed))
                    evidence[W02_SHADOW_EXPECTED_GATES[0][0]].append(digest)
                    evidence[W02_SHADOW_EXPECTED_GATES[1][0]].append(digest)
                    passed[W02_SHADOW_EXPECTED_GATES[0][0]] += int(carrier_ok)
                    passed[W02_SHADOW_EXPECTED_GATES[1][0]] += int(boundary_ok)
                    if not routed:
                        continue
                    evidence[W02_SHADOW_EXPECTED_GATES[2][0]].append(digest)
                    passed[W02_SHADOW_EXPECTED_GATES[2][0]] += int(morphology_ok)
                    spans = _select_shadow_spans(
                        base, overlay_index.max_form_length)
                    successor = predict_w02_morphology_successor(
                        overlay_index, record, base, requested_spans=spans,
                        ranking_cache=cache)
                    base_keys = {canonical_json_bytes(item.to_dict())
                                 for item in base.morphology_candidates}
                    successor_keys = {canonical_json_bytes(item.to_dict())
                                      for item in successor.prediction.morphology_candidates}
                    overlay_ok = (base_keys.issubset(successor_keys)
                                  and successor.generalized_candidate_count <= 40)
                    overlay_digest = _hash_value({
                        "base_prediction_sha256": _hash_value(base.to_dict()),
                        "generalized_candidate_count":
                            successor.generalized_candidate_count,
                        "observation_key": list(record.stable_key.components),
                        "overlay_ok": int(overlay_ok),
                        "requested_spans": [list(span) for span in spans],
                        "successor_prediction_sha256":
                            _hash_value(successor.prediction.to_dict()),
                    })
                    evidence[W02_SHADOW_EXPECTED_GATES[3][0]].append(
                        overlay_digest)
                    passed[W02_SHADOW_EXPECTED_GATES[3][0]] += int(overlay_ok)
                    counts["overlay_inference_logic_operations"] += (
                        successor.logic_operations)
                    counts["queried_span_count"] += len(spans)
                    counts["generalized_candidate_count"] += (
                        successor.generalized_candidate_count)
                    counts["max_generalized_candidates_per_observation"] = max(
                        counts["max_generalized_candidates_per_observation"],
                        successor.generalized_candidate_count)
        audit_results = [
            _gate(name, denominator, passed[name], evidence[name])
            for name, denominator in W02_SHADOW_EXPECTED_GATES
        ]
        logic_operations = (overlay_index.logic_operations
                            + counts["base_logic_operations"]
                            + counts["overlay_inference_logic_operations"])
        report = {
            "artifact_kind": (
                "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_SHADOW_AUDIT_REPORT"),
            "artifact_version": W02_MORPH_SUCCESSOR_SHADOW_REPORT_VERSION,
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
            "formal_training_runs": 1,
            "label_reads": 0,
            "language_capability_mastered": 0,
            "language_readiness": 0,
            "logic_operations": logic_operations,
            "next_action": (
                "W02_PRIVATE_FAMILY_REGISTRATION_FREEZE"
                if formal and all(row["status"] == "PASS"
                                  for row in audit_results)
                else "W02_SHADOW_FAILED_STOP" if formal
                else "W02_SHADOW_AUDIT_FREEZE"),
            "overlay_artifact_manifest_sha256":
                freeze["overlay_artifact_manifest_sha256"],
            "overlay_semantic_sha256": freeze["overlay_semantic_sha256"],
            "private_family_registered": 0,
            "private_payload_reads": 0,
            "ranking_cache_entry_count": len(cache.values),
            "ranking_cache_hit_count": cache.hit_count,
            "ranking_cache_miss_count": cache.miss_count,
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
            "successor_transform_logic_operations":
                overlay_index.logic_operations,
            "teacher_calls": 0,
            "transport_bytes_read": sum(
                _shadow_identity(parent, key).transport_size_bytes
                for key in W02_SHADOW_LAYOUTS),
            "zero_write_audit": {
                "candidate_writes": 0, "companion_writes": 0,
                "core_writes": 0, "dev_owner_writes": 0,
                "evidence_writes": 0, "memory_writes": 0,
                "overlay_writes": 0, "shadow_owner_writes": 0,
                "use_writes": 0,
            },
        }
    finally:
        cache.close()
    after = (_tree_sha256(shadow.root), _tree_sha256(candidate_root),
             _tree_sha256(overlay_root))
    if after != before:
        raise W02MorphologySuccessorShadowAuditError(
            "shadow audit 产生非授权写入")
    if report["logic_operations"] > freeze["resource_budget"]["max_logic_operations"]:
        raise W02MorphologySuccessorShadowAuditError(
            "shadow logic resource stop")
    _assert_expected(report, freeze)
    validate_v2_safe_report(report)
    return report


def run_w02_morphology_successor_shadow_preflight(
        repository_root: str | Path, shadow_root: str | Path,
        candidate_artifact_root: str | Path,
        overlay_artifact_root: str | Path,
        ) -> dict[str, object]:
    return _run_shadow(
        Path(repository_root).resolve(), shadow_root, candidate_artifact_root,
        overlay_artifact_root, formal=False)


def run_w02_morphology_successor_shadow_audit(
        repository_root: str | Path, shadow_root: str | Path,
        candidate_artifact_root: str | Path,
        overlay_artifact_root: str | Path, *, run_id: int = 1,
        ) -> dict[str, object]:
    if run_id != 1:
        raise W02MorphologySuccessorShadowAuditError(
            "shadow formal run_id 固定为 1")
    return _run_shadow(
        Path(repository_root).resolve(), shadow_root, candidate_artifact_root,
        overlay_artifact_root, formal=True)


def publish_w02_morphology_successor_shadow_audit_report(
        repository_root: str | Path, external_report: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = read_canonical_object(external_report)
    validate_v2_safe_report(value)
    freeze_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_SHADOW_FREEZE_PATH)
    freeze = read_w02_morphology_successor_shadow_audit_freeze(repository)
    freeze_size, freeze_sha = _sha256_file(freeze_path)
    if (value.get("artifact_version") != W02_MORPH_SUCCESSOR_SHADOW_REPORT_VERSION
            or value.get("run_scope") != "FORMAL" or value.get("run_id") != 1
            or value.get("formal_shadow_audit_runs") != 1
            or value.get("formal_private_evaluation_runs") != 0
            or value.get("private_payload_reads") != 0
            or value.get("label_reads") != 0
            or value.get("teacher_calls") != 0
            or value.get("code_freeze_sha256") != freeze["code_freeze_sha256"]):
        raise W02MorphologySuccessorShadowAuditError(
            "shadow formal report 状态非法")
    _assert_expected(value, freeze)
    public = dict(value)
    public["shadow_freeze_file_sha256"] = freeze_sha
    public["shadow_freeze_size_bytes"] = freeze_size
    validate_v2_safe_report(public)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_SHADOW_REPORT_PATH).parts)
    write_immutable_json(public, target)
    return target


__all__ = [
    "W02_MORPH_SUCCESSOR_SHADOW_CODE_PATHS",
    "W02_MORPH_SUCCESSOR_SHADOW_FREEZE_PATH",
    "W02_MORPH_SUCCESSOR_SHADOW_REPORT_PATH",
    "W02MorphologySuccessorShadowAuditError",
    "W02ShadowInputRoot",
    "build_w02_morphology_successor_shadow_audit_freeze",
    "iter_w02_shadow_records",
    "publish_w02_morphology_successor_shadow_audit_freeze",
    "publish_w02_morphology_successor_shadow_audit_report",
    "read_w02_morphology_successor_shadow_audit_freeze",
    "run_w02_morphology_successor_shadow_audit",
    "run_w02_morphology_successor_shadow_preflight",
]
