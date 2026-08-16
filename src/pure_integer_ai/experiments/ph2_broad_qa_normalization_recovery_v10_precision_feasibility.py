"""发布 recovery-v10 precision-first TRAIN feasibility artifact。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_sources import (
    read_opencc_unique_t2s_routes,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate import (
    build_normalization_recovery_v8_candidate_index,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate import (
    compile_normalization_recovery_v10_precision_candidate,
    compile_normalization_recovery_v10_precision_candidate_v2,
    derive_normalization_recovery_v10_precision_preflight,
    derive_normalization_recovery_v10_precision_source_loso_audit,
    derive_normalization_recovery_v10_precision_training_audit,
    derive_normalization_recovery_v10_precision_v2_preflight,
    derive_normalization_recovery_v10_precision_v2_training_audit,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_V1")
NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_STATUS = (
    "TRAIN_ONLY_ZERO_WRONG_PRECISION_FEASIBILITY_PASS_NOT_FORMAL")
NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_V2_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_V2")
NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_V2_STATUS = (
    "TRAIN_LOSO_SOURCE_ONLY_ZERO_WRONG_FEASIBILITY_PASS_NOT_FORMAL")

_OUTPUT_FILES = (
    ("candidate-program.json", "V10_PRECISION_CANDIDATE_PROGRAM"),
    ("preflight.json", "V10_PRECISION_LABEL_BLIND_PREFLIGHT"),
    ("training-audit.json", "V10_PRECISION_TRAIN_AGGREGATE_AUDIT"),
)
_V2_OUTPUT_FILES = (
    ("candidate-program.json", "V10_PRECISION_V2_CANDIDATE_PROGRAM"),
    ("preflight.json", "V10_PRECISION_V2_LABEL_BLIND_PREFLIGHT"),
    ("training-audit.json", "V10_PRECISION_V2_TRAIN_AGGREGATE_AUDIT"),
    ("source-loso-audit.json", "V10_PRECISION_V2_SOURCE_FAMILY_LOSO_AUDIT"),
)
_OBSERVATION_FILES = (
    "qbittorrent-observations.jsonl",
    "stellarium-observations.jsonl",
    "keepassxc-observations.jsonl",
)


def _sha256(payload: bytes) -> str:
    """返回 manifest 或文件的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"v10 feasibility {label} 非法")
    return value


def _require_k_root(value: str | Path) -> Path:
    """要求显式、已存在的 K 盘 run root。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v10 feasibility run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析路径并拒绝逃出唯一 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"v10 feasibility {label} 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _read_manifest(
        root: Path, *, expected_sha256: str, label: str,
        ) -> dict[str, object]:
    """回读规范 manifest 并核验调用方冻结的外部身份。"""
    try:
        payload = (root / "manifest.json").read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v10 feasibility {label} manifest 不可读") from error
    if (_sha256(payload) != expected_sha256
            or not isinstance(value, dict)
            or canonical_json_line(value) != payload
            or not isinstance(value.get("files"), list)):
        raise BroadQaExternalDataError(
            f"v10 feasibility {label} manifest 漂移")
    return value


def _file_record(
        manifest: dict[str, object], relative_path: str, *, label: str,
        ) -> dict[str, object]:
    """从 manifest 选择唯一文件承诺。"""
    records = [
        item for item in manifest["files"]
        if isinstance(item, dict)
        and item.get("relative_path") == relative_path
    ]
    if len(records) != 1:
        raise BroadQaExternalDataError(
            f"v10 feasibility {label} file commitment 缺失")
    record = records[0]
    if (type(record.get("bytes")) is not int or record["bytes"] < 0
            or _sha_value(record.get("sha256"), label=f"{label} file SHA")
            != record["sha256"]):
        raise BroadQaExternalDataError(
            f"v10 feasibility {label} file commitment 漂移")
    return record


def _read_committed_json(
        root: Path, manifest: dict[str, object], relative_path: str, *,
        label: str,
        ) -> dict[str, object]:
    """回读一份被 manifest 承诺的规范 JSON。"""
    record = _file_record(manifest, relative_path, label=label)
    try:
        payload = (root / relative_path).read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v10 feasibility {label} JSON 不可读") from error
    if (len(payload) != record["bytes"]
            or _sha256(payload) != record["sha256"]
            or not isinstance(value, dict)
            or canonical_json_line(value) != payload):
        raise BroadQaExternalDataError(
            f"v10 feasibility {label} JSON 漂移")
    return value


def _read_committed_jsonl(
        root: Path, manifest: dict[str, object], relative_path: str, *,
        label: str,
        ) -> tuple[dict[str, object], ...]:
    """流式回读被 manifest 承诺的规范 JSONL。"""
    record = _file_record(manifest, relative_path, label=label)
    digest = hashlib.sha256()
    size = 0
    values = []
    try:
        with (root / relative_path).open("rb") as handle:
            for line in handle:
                digest.update(line)
                size += len(line)
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v10 feasibility {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v10 feasibility {label} JSONL 不可读") from error
    expected_count = record.get("record_count")
    if (size != record["bytes"] or digest.hexdigest() != record["sha256"]
            or (expected_count is not None
                and (type(expected_count) is not int
                     or expected_count != len(values)))):
        raise BroadQaExternalDataError(
            f"v10 feasibility {label} JSONL identity 漂移")
    return tuple(values)


def _surface(observation: dict[str, object], role: str) -> str:
    """从 Qt/gettext 统一 Observation 提取唯一 locale 表面。"""
    value = observation.get(role)
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            "v10 feasibility observation locale record 漂移")
    strings = [item for item in (
        value.get("translation"), value.get("msgstr"))
        if isinstance(item, str)]
    if len(strings) != 1 or not strings[0]:
        raise BroadQaExternalDataError(
            "v10 feasibility observation surface 漂移")
    return strings[0]


def _observation_material(
        outputs: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """把冻结 Observation 投影为 compile safety 与 TRAIN audit cases。"""
    safety = []
    cases = []
    for item in outputs:
        source = item.get("official_source_text")
        family = item.get("source_family")
        if not isinstance(source, str) or not source or not isinstance(family, str):
            raise BroadQaExternalDataError(
                "v10 feasibility observation source/family 漂移")
        input_text = _surface(item, "zh_hant")
        output_text = _surface(item, "zh_hans")
        safety.append({
            "input_text": input_text,
            "official_source_text": source,
            "output_text": output_text,
            "source_family": family,
        })
        cases.append({
            "expected_output": output_text,
            "input_text": input_text,
            "official_source_text": source,
            "source_family": family,
            "structure_tokens": list(localization_structure_tokens(input_text)),
        })
    return tuple(safety), tuple(cases)


def _artifact(
        relative_path: str, role: str, payload: bytes,
        ) -> dict[str, object]:
    """形成一份输出文件承诺。"""
    return {
        "bytes": len(payload),
        "relative_path": relative_path,
        "role": role,
        "sha256": _sha256(payload),
    }


def load_normalization_recovery_v10_precision_material(
        *,
        base_candidate_dir: Path,
        expected_base_candidate_manifest_sha256: str,
        protocol_dir: Path,
        expected_protocol_manifest_sha256: str,
        observation_dir: Path,
        expected_observation_manifest_sha256: str,
        opencc_source_pack_dir: Path,
        expected_opencc_source_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, str],
            dict[str, int],
        ]:
    """从四份冻结 artifact 读取 v10 可重派生的最小 TRAIN material。"""
    base_manifest = _read_manifest(
        base_candidate_dir,
        expected_sha256=expected_base_candidate_manifest_sha256,
        label="base candidate")
    base_candidate = _read_committed_json(
        base_candidate_dir, base_manifest, "candidate-program.json",
        label="base candidate program")
    build_normalization_recovery_v8_candidate_index(base_candidate)
    protocol_manifest = _read_manifest(
        protocol_dir, expected_sha256=expected_protocol_manifest_sha256,
        label="training protocol")
    controls = _read_committed_jsonl(
        protocol_dir, protocol_manifest, "exact-input-control.jsonl",
        label="exact-input controls")
    observation_manifest = _read_manifest(
        observation_dir, expected_sha256=expected_observation_manifest_sha256,
        label="observation pack")
    observations = tuple(
        item for name in _OBSERVATION_FILES
        for item in _read_committed_jsonl(
            observation_dir, observation_manifest, name,
            label=f"observation {name}"))
    expected_observations = observation_manifest.get("summary", {}).get(
        "observation_count") if isinstance(
            observation_manifest.get("summary"), dict) else None
    if (type(expected_observations) is not int
            or expected_observations != len(observations)):
        raise BroadQaExternalDataError(
            "v10 feasibility observation denominator 漂移")
    safety, cases = _observation_material(observations)
    try:
        opencc_manifest_payload = (
            opencc_source_pack_dir / "manifest.json").read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 feasibility OpenCC manifest 不可读") from error
    if _sha256(opencc_manifest_payload) != expected_opencc_source_manifest_sha256:
        raise BroadQaExternalDataError(
            "v10 feasibility OpenCC manifest identity 漂移")
    opencc_routes, opencc_census = read_opencc_unique_t2s_routes(
        opencc_source_pack_dir)
    return base_candidate, controls, safety, cases, opencc_routes, opencc_census


def _derive(
        *,
        base_candidate_dir: Path,
        expected_base_candidate_manifest_sha256: str,
        protocol_dir: Path,
        expected_protocol_manifest_sha256: str,
        observation_dir: Path,
        expected_observation_manifest_sha256: str,
        opencc_source_pack_dir: Path,
        expected_opencc_source_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, bytes],
        ]:
    """从四份冻结 TRAIN 输入重派生 candidate、preflight 与 audit。"""
    base_manifest = _read_manifest(
        base_candidate_dir,
        expected_sha256=expected_base_candidate_manifest_sha256,
        label="base candidate")
    base_candidate = _read_committed_json(
        base_candidate_dir, base_manifest, "candidate-program.json",
        label="base candidate program")
    build_normalization_recovery_v8_candidate_index(base_candidate)

    protocol_manifest = _read_manifest(
        protocol_dir, expected_sha256=expected_protocol_manifest_sha256,
        label="training protocol")
    controls = _read_committed_jsonl(
        protocol_dir, protocol_manifest, "exact-input-control.jsonl",
        label="exact-input controls")

    observation_manifest = _read_manifest(
        observation_dir, expected_sha256=expected_observation_manifest_sha256,
        label="observation pack")
    observations = tuple(
        item for name in _OBSERVATION_FILES
        for item in _read_committed_jsonl(
            observation_dir, observation_manifest, name,
            label=f"observation {name}"))
    expected_observations = observation_manifest.get("summary", {}).get(
        "observation_count") if isinstance(
            observation_manifest.get("summary"), dict) else None
    if (type(expected_observations) is not int
            or expected_observations != len(observations)):
        raise BroadQaExternalDataError(
            "v10 feasibility observation denominator 漂移")
    safety, cases = _observation_material(observations)

    opencc_manifest_payload = (
        opencc_source_pack_dir / "manifest.json").read_bytes()
    if _sha256(opencc_manifest_payload) != expected_opencc_source_manifest_sha256:
        raise BroadQaExternalDataError(
            "v10 feasibility OpenCC manifest identity 漂移")
    opencc_routes, opencc_census = read_opencc_unique_t2s_routes(
        opencc_source_pack_dir)

    candidate = compile_normalization_recovery_v10_precision_candidate(
        base_candidate=base_candidate,
        exact_input_controls=controls,
        safety_observations=safety,
        training_protocol_manifest_sha256=(
            expected_protocol_manifest_sha256),
        observation_pack_manifest_sha256=(
            expected_observation_manifest_sha256),
        opencc_routes=opencc_routes,
        opencc_source_pack_manifest_sha256=(
            expected_opencc_source_manifest_sha256),
    )
    preflight = derive_normalization_recovery_v10_precision_preflight(
        candidate)
    audit = derive_normalization_recovery_v10_precision_training_audit(
        candidate, cases)
    if (preflight["failure_count"] != 0
            or preflight["indexed_reference_mismatch_count"] != 0
            or audit["training_outcome"]
            != "PASS_ZERO_WRONG_NONZERO_CHANGED_EXACT"
            or audit["facility_outcome"] != "PASS"):
        raise BroadQaExternalDataError(
            "v10 feasibility precision hard gate 未通过")
    values = {
        "candidate-program.json": candidate,
        "preflight.json": preflight,
        "training-audit.json": audit,
    }
    payloads = {
        name: canonical_json_line(value) for name, value in values.items()}
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_KIND,
        "base_candidate_manifest_sha256": (
            expected_base_candidate_manifest_sha256),
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "files": [
            _artifact(name, role, payloads[name])
            for name, role in _OUTPUT_FILES],
        "format_version": 1,
        "formal_or_evaluation_payload_read_count": 0,
        "mastery_claimed": 0,
        "observation_pack_manifest_sha256": (
            expected_observation_manifest_sha256),
        "opencc_source_pack_manifest_sha256": (
            expected_opencc_source_manifest_sha256),
        "opencc_unique_route_count": opencc_census["unique_route_count"],
        "preflight_sha256": preflight["preflight_sha256"],
        "production_enabled": 0,
        "rule_counts": candidate["rule_counts"],
        "status": NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_audit_sha256": audit["training_audit_sha256"],
        "training_case_count": audit["case_count"],
        "training_outcomes": audit["outcomes"],
        "training_protocol_manifest_sha256": (
            expected_protocol_manifest_sha256),
    }
    return manifest, candidate, preflight, audit, payloads


def publish_normalization_recovery_v10_precision_feasibility(
        *, run_root: str | Path,
        base_candidate_dir: str | Path,
        expected_base_candidate_manifest_sha256: str,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        observation_dir: str | Path,
        expected_observation_manifest_sha256: str,
        opencc_source_pack_dir: str | Path,
        expected_opencc_source_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 K 盘 TRAIN-only precision feasibility。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=label) for value, label in (
        (base_candidate_dir, "base_candidate_dir"),
        (protocol_dir, "protocol_dir"),
        (observation_dir, "observation_dir"),
        (opencc_source_pack_dir, "opencc_source_pack_dir"),
        (target_dir, "target_dir"),
    ))
    if (any(not path.is_dir() for path in paths[:-1])
            or paths[-1].exists()
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError(
            "v10 feasibility artifact path 非法")
    shas = tuple(_sha_value(value, label="input SHA") for value in (
        expected_base_candidate_manifest_sha256,
        expected_protocol_manifest_sha256,
        expected_observation_manifest_sha256,
        expected_opencc_source_manifest_sha256,
    ))
    manifest, _candidate, _preflight, _audit, payloads = _derive(
        base_candidate_dir=paths[0],
        expected_base_candidate_manifest_sha256=shas[0],
        protocol_dir=paths[1],
        expected_protocol_manifest_sha256=shas[1],
        observation_dir=paths[2],
        expected_observation_manifest_sha256=shas[2],
        opencc_source_pack_dir=paths[3],
        expected_opencc_source_manifest_sha256=shas[3],
    )
    target = paths[-1]
    target.mkdir()
    for name, _role in _OUTPUT_FILES:
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v10_precision_feasibility(
        feasibility_dir: str | Path, *,
        base_candidate_dir: str | Path,
        expected_base_candidate_manifest_sha256: str,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        observation_dir: str | Path,
        expected_observation_manifest_sha256: str,
        opencc_source_pack_dir: str | Path,
        expected_opencc_source_manifest_sha256: str,
        expected_feasibility_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, object],
            dict[str, object], dict[str, object],
        ]:
    """从四份冻结输入重派生并严格回读 feasibility artifact。"""
    roots = tuple(Path(value).resolve() for value in (
        feasibility_dir, base_candidate_dir, protocol_dir,
        observation_dir, opencc_source_pack_dir))
    if (any(not path.is_dir() for path in roots)
            or any(_overlap(left, right)
                   for index, left in enumerate(roots)
                   for right in roots[index + 1:])):
        raise BroadQaExternalDataError(
            "v10 feasibility strict roots 混淆")
    shas = tuple(_sha_value(value, label="expected SHA") for value in (
        expected_base_candidate_manifest_sha256,
        expected_protocol_manifest_sha256,
        expected_observation_manifest_sha256,
        expected_opencc_source_manifest_sha256,
        expected_feasibility_manifest_sha256,
    ))
    expected, candidate, preflight, audit, payloads = _derive(
        base_candidate_dir=roots[1],
        expected_base_candidate_manifest_sha256=shas[0],
        protocol_dir=roots[2],
        expected_protocol_manifest_sha256=shas[1],
        observation_dir=roots[3],
        expected_observation_manifest_sha256=shas[2],
        opencc_source_pack_dir=roots[4],
        expected_opencc_source_manifest_sha256=shas[3],
    )
    try:
        encoded = (roots[0] / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 feasibility stored manifest 不可读") from error
    if (_sha256(encoded) != shas[4]
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored != expected):
        raise BroadQaExternalDataError(
            "v10 feasibility stored manifest 漂移")
    for name, _role in _OUTPUT_FILES:
        try:
            payload = (roots[0] / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v10 feasibility stored {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v10 feasibility stored {name} 重派生漂移")
    return {**stored, "manifest_sha256": shas[4]}, candidate, preflight, audit


def _derive_v2(
        *,
        predecessor_feasibility_dir: Path,
        expected_predecessor_feasibility_manifest_sha256: str,
        base_candidate_dir: Path,
        expected_base_candidate_manifest_sha256: str,
        protocol_dir: Path,
        expected_protocol_manifest_sha256: str,
        observation_dir: Path,
        expected_observation_manifest_sha256: str,
        opencc_source_pack_dir: Path,
        expected_opencc_source_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object],
            dict[str, object], dict[str, object], dict[str, bytes],
        ]:
    """重派生 source-only commit candidate、全量 TRAIN 与三方向 LOSO。"""
    _read_manifest(
        predecessor_feasibility_dir,
        expected_sha256=expected_predecessor_feasibility_manifest_sha256,
        label="predecessor feasibility")
    (base_candidate, controls, safety, cases,
     opencc_routes, opencc_census) = (
        load_normalization_recovery_v10_precision_material(
            base_candidate_dir=base_candidate_dir,
            expected_base_candidate_manifest_sha256=(
                expected_base_candidate_manifest_sha256),
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=(
                expected_protocol_manifest_sha256),
            observation_dir=observation_dir,
            expected_observation_manifest_sha256=(
                expected_observation_manifest_sha256),
            opencc_source_pack_dir=opencc_source_pack_dir,
            expected_opencc_source_manifest_sha256=(
                expected_opencc_source_manifest_sha256),
        ))
    candidate = compile_normalization_recovery_v10_precision_candidate_v2(
        base_candidate=base_candidate,
        exact_input_controls=controls,
        safety_observations=safety,
        training_protocol_manifest_sha256=(
            expected_protocol_manifest_sha256),
        observation_pack_manifest_sha256=(
            expected_observation_manifest_sha256),
        opencc_routes=opencc_routes,
        opencc_source_pack_manifest_sha256=(
            expected_opencc_source_manifest_sha256),
    )
    preflight = derive_normalization_recovery_v10_precision_v2_preflight(
        candidate)
    audit = derive_normalization_recovery_v10_precision_v2_training_audit(
        candidate, cases)
    loso = derive_normalization_recovery_v10_precision_source_loso_audit(
        base_candidate=base_candidate, safety_observations=safety)
    if (preflight["failure_count"] != 0
            or preflight["indexed_reference_mismatch_count"] != 0
            or audit["training_outcome"]
            != "PASS_ZERO_WRONG_NONZERO_CHANGED_EXACT"
            or audit["facility_outcome"] != "PASS"
            or loso["status"] != "PASS_ZERO_WRONG_NONZERO_EXACT"):
        raise BroadQaExternalDataError(
            "v10 feasibility v2 hard gate 未通过")
    values = {
        "candidate-program.json": candidate,
        "preflight.json": preflight,
        "source-loso-audit.json": loso,
        "training-audit.json": audit,
    }
    payloads = {
        name: canonical_json_line(value) for name, value in values.items()}
    manifest = {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_V2_KIND),
        "base_candidate_manifest_sha256": (
            expected_base_candidate_manifest_sha256),
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "files": [
            _artifact(name, role, payloads[name])
            for name, role in _V2_OUTPUT_FILES],
        "formal_or_evaluation_payload_read_count": 0,
        "format_version": 2,
        "mastery_claimed": 0,
        "observation_pack_manifest_sha256": (
            expected_observation_manifest_sha256),
        "opencc_source_pack_manifest_sha256": (
            expected_opencc_source_manifest_sha256),
        "opencc_unique_route_count": opencc_census["unique_route_count"],
        "predecessor_feasibility_manifest_sha256": (
            expected_predecessor_feasibility_manifest_sha256),
        "preflight_sha256": preflight["preflight_sha256"],
        "production_enabled": 0,
        "rule_counts": candidate["rule_counts"],
        "source_loso_audit_sha256": loso["loso_audit_sha256"],
        "source_loso_outcomes": loso["outcomes"],
        "status": NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_V2_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_audit_sha256": audit["training_audit_sha256"],
        "training_case_count": audit["case_count"],
        "training_outcomes": audit["outcomes"],
        "training_protocol_manifest_sha256": (
            expected_protocol_manifest_sha256),
    }
    return manifest, candidate, preflight, audit, loso, payloads


def publish_normalization_recovery_v10_precision_feasibility_v2(
        *, run_root: str | Path,
        predecessor_feasibility_dir: str | Path,
        expected_predecessor_feasibility_manifest_sha256: str,
        base_candidate_dir: str | Path,
        expected_base_candidate_manifest_sha256: str,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        observation_dir: str | Path,
        expected_observation_manifest_sha256: str,
        opencc_source_pack_dir: str | Path,
        expected_opencc_source_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布经 family-holdout 收口的 v2 feasibility。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=label) for value, label in (
        (predecessor_feasibility_dir, "predecessor_feasibility_dir"),
        (base_candidate_dir, "base_candidate_dir"),
        (protocol_dir, "protocol_dir"),
        (observation_dir, "observation_dir"),
        (opencc_source_pack_dir, "opencc_source_pack_dir"),
        (target_dir, "target_dir"),
    ))
    if (any(not path.is_dir() for path in paths[:-1])
            or paths[-1].exists()
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError(
            "v10 feasibility v2 artifact path 非法")
    shas = tuple(_sha_value(value, label="v2 input SHA") for value in (
        expected_predecessor_feasibility_manifest_sha256,
        expected_base_candidate_manifest_sha256,
        expected_protocol_manifest_sha256,
        expected_observation_manifest_sha256,
        expected_opencc_source_manifest_sha256,
    ))
    manifest, _candidate, _preflight, _audit, _loso, payloads = _derive_v2(
        predecessor_feasibility_dir=paths[0],
        expected_predecessor_feasibility_manifest_sha256=shas[0],
        base_candidate_dir=paths[1],
        expected_base_candidate_manifest_sha256=shas[1],
        protocol_dir=paths[2],
        expected_protocol_manifest_sha256=shas[2],
        observation_dir=paths[3],
        expected_observation_manifest_sha256=shas[3],
        opencc_source_pack_dir=paths[4],
        expected_opencc_source_manifest_sha256=shas[4],
    )
    target = paths[-1]
    target.mkdir()
    for name, _role in _V2_OUTPUT_FILES:
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v10_precision_feasibility_v2(
        feasibility_dir: str | Path, *,
        predecessor_feasibility_dir: str | Path,
        expected_predecessor_feasibility_manifest_sha256: str,
        base_candidate_dir: str | Path,
        expected_base_candidate_manifest_sha256: str,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        observation_dir: str | Path,
        expected_observation_manifest_sha256: str,
        opencc_source_pack_dir: str | Path,
        expected_opencc_source_manifest_sha256: str,
        expected_feasibility_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object],
            dict[str, object], dict[str, object],
        ]:
    """重派生并严格回读 v2 candidate、TRAIN audit 与 LOSO。"""
    roots = tuple(Path(value).resolve() for value in (
        feasibility_dir, predecessor_feasibility_dir, base_candidate_dir,
        protocol_dir, observation_dir, opencc_source_pack_dir))
    if (any(not path.is_dir() for path in roots)
            or any(_overlap(left, right)
                   for index, left in enumerate(roots)
                   for right in roots[index + 1:])):
        raise BroadQaExternalDataError(
            "v10 feasibility v2 strict roots 混淆")
    shas = tuple(_sha_value(value, label="v2 expected SHA") for value in (
        expected_predecessor_feasibility_manifest_sha256,
        expected_base_candidate_manifest_sha256,
        expected_protocol_manifest_sha256,
        expected_observation_manifest_sha256,
        expected_opencc_source_manifest_sha256,
        expected_feasibility_manifest_sha256,
    ))
    expected, candidate, preflight, audit, loso, payloads = _derive_v2(
        predecessor_feasibility_dir=roots[1],
        expected_predecessor_feasibility_manifest_sha256=shas[0],
        base_candidate_dir=roots[2],
        expected_base_candidate_manifest_sha256=shas[1],
        protocol_dir=roots[3],
        expected_protocol_manifest_sha256=shas[2],
        observation_dir=roots[4],
        expected_observation_manifest_sha256=shas[3],
        opencc_source_pack_dir=roots[5],
        expected_opencc_source_manifest_sha256=shas[4],
    )
    try:
        encoded = (roots[0] / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 feasibility v2 stored manifest 不可读") from error
    if (_sha256(encoded) != shas[5]
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored != expected):
        raise BroadQaExternalDataError(
            "v10 feasibility v2 stored manifest 漂移")
    for name, _role in _V2_OUTPUT_FILES:
        try:
            payload = (roots[0] / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v10 feasibility v2 stored {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v10 feasibility v2 stored {name} 重派生漂移")
    return (
        {**stored, "manifest_sha256": shas[5]},
        candidate, preflight, audit, loso,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_KIND",
    "NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_STATUS",
    "NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_V2_KIND",
    "NORMALIZATION_RECOVERY_V10_PRECISION_FEASIBILITY_V2_STATUS",
    "load_normalization_recovery_v10_precision_material",
    "publish_normalization_recovery_v10_precision_feasibility",
    "publish_normalization_recovery_v10_precision_feasibility_v2",
    "read_normalization_recovery_v10_precision_feasibility",
    "read_normalization_recovery_v10_precision_feasibility_v2",
]
