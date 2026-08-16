"""发布 recovery-v10 source-only 候选与完整 TRAIN runtime shape pack。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_record_id,
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate import (
    NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_KIND,
    NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_STATUS,
    derive_normalization_recovery_v10_precision_v2_preflight,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_feasibility import (
    load_normalization_recovery_v10_precision_material,
    read_normalization_recovery_v10_precision_feasibility_v2,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_V1")
NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_STATUS = (
    "TRAIN_SOURCE_ONLY_CANDIDATE_PACK_RUNTIME_NOT_RUN_NOT_FORMAL")
NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_SHAPE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_SHAPE_V1")

V10_PRECISION_FEASIBILITY_V1_MANIFEST_SHA256 = (
    "31897f9d5f10cb949d85d010a75fa34c2b943d2677d502950f6df383a64d5212")
V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256 = (
    "9b22d7896f6d86695405490e1ff82393ffde0cdc974944bd24b687c52fdc7522")
V10_PRECISION_BASE_CANDIDATE_MANIFEST_SHA256 = (
    "f3c1a011a05afbb3307e7a9c308077a5c990e093800df7fc1a292a221cfc02f6")
V10_PRECISION_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "210d73d97c353c0225c3b1f97c67020bad5b1e762a4cb171a9d2e4589621c7d3")
V10_PRECISION_OBSERVATION_PACK_MANIFEST_SHA256 = (
    "99ab49c0605be76b2206746330969a071d8b6deed83f3aa454610a99546ddf65")
V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256 = (
    "189f42097dc059218be337231d340a4265d2783b64c2fb884892db0caf8af94c")
V10_PRECISION_CANDIDATE_PROGRAM_SHA256 = (
    "16669ddcc65ae40934cd43529d081b02a6bb9dd324667957fd635b30a3f23cb8")
V10_PRECISION_PREFLIGHT_SHA256 = (
    "5556ab978e569d84af975d593cbd0c1a9b73e4b94c2ed79018bb4d26d7217c99")
V10_PRECISION_TRAINING_AUDIT_SHA256 = (
    "e5bf010c2cb2f1621499aa53adea1740dcc3034736583f9c0848fb88caab1d7a")
V10_PRECISION_SOURCE_LOSO_AUDIT_SHA256 = (
    "6c579cc7d0d8fbf7545bfa122824b071041dd04343be6b789b65e62afdb5c813")
V10_PRECISION_RUNTIME_QUERY_COUNT = 33_179
V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT = 45

_FILES = (
    ("candidate-program.json", "V10_PRECISION_V2_CANDIDATE_PROGRAM"),
    ("preflight.json", "V10_PRECISION_V2_PREFLIGHT"),
    ("runtime-shapes.jsonl", "V10_PRECISION_TRAIN_RUNTIME_SHAPES"),
)


def _sha256(payload: bytes) -> str:
    """返回文件、query roster 或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(
            f"v10 precision candidate pack {label} 非法")
    return value


def _require_k_root(value: str | Path) -> Path:
    """要求显式、已存在的 K 盘 run root。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v10 precision candidate pack run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析路径并拒绝逃出唯一 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v10 precision candidate pack {label} 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _artifact(
        name: str, role: str, payload: bytes, *,
        record_count: int | None = None,
        ) -> dict[str, object]:
    """形成一份输出文件承诺。"""
    value = {
        "bytes": len(payload),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }
    if record_count is not None:
        value["record_count"] = record_count
    return value


def _file_record(
        manifest: dict[str, object], relative_path: str,
        ) -> dict[str, object]:
    """从 manifest 选择唯一文件承诺。"""
    files = manifest.get("files")
    matches = [
        item for item in files
        if isinstance(item, dict)
        and item.get("relative_path") == relative_path
    ] if isinstance(files, list) else []
    if len(matches) != 1:
        raise BroadQaExternalDataError(
            "v10 precision candidate pack file commitment 漂移")
    record = matches[0]
    if (type(record.get("bytes")) is not int or record["bytes"] < 0
            or _sha_value(record.get("sha256"), label="file SHA")
            != record["sha256"]):
        raise BroadQaExternalDataError(
            "v10 precision candidate pack file record 漂移")
    return record


def _read_committed_json(
        root: Path, manifest: dict[str, object], relative_path: str,
        ) -> dict[str, object]:
    """读取一份被 pack manifest 承诺的规范 JSON。"""
    record = _file_record(manifest, relative_path)
    try:
        payload = (root / relative_path).read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v10 precision candidate pack {relative_path} 不可读") from error
    if (len(payload) != record["bytes"]
            or _sha256(payload) != record["sha256"]
            or not isinstance(value, dict)
            or canonical_json_line(value) != payload):
        raise BroadQaExternalDataError(
            f"v10 precision candidate pack {relative_path} 漂移")
    return value


def _read_runtime_shapes(
        root: Path, manifest: dict[str, object],
        ) -> tuple[tuple[dict[str, object], ...], bytes]:
    """流式读取无 label runtime shapes，并核验每条自绑定身份。"""
    record = _file_record(manifest, "runtime-shapes.jsonl")
    digest = hashlib.sha256()
    payload_parts = []
    shapes = []
    try:
        with (root / "runtime-shapes.jsonl").open("rb") as handle:
            for ordinal, line in enumerate(handle):
                digest.update(line)
                payload_parts.append(line)
                item = json.loads(line)
                if (not isinstance(item, dict)
                        or canonical_json_line(item) != line
                        or set(item) != {
                            "expected_output_included", "format_version",
                            "ordinal", "query", "record_kind", "shape_id",
                            "source_family_included", "training_query_only",
                        }
                        or item.get("ordinal") != ordinal
                        or item.get("record_kind")
                        != NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_SHAPE_KIND
                        or item.get("expected_output_included") != 0
                        or item.get("source_family_included") != 0
                        or item.get("training_query_only") != 1
                        or item.get("format_version") != 1):
                    raise BroadQaExternalDataError(
                        "v10 precision candidate pack runtime shape 漂移")
                query = item.get("query")
                if not isinstance(query, dict) or set(query) != {
                        "input_text", "official_source_text",
                        "structure_tokens"}:
                    raise BroadQaExternalDataError(
                        "v10 precision candidate pack runtime query schema 漂移")
                input_text = query.get("input_text")
                source = query.get("official_source_text")
                tokens = query.get("structure_tokens")
                identity = {
                    "ordinal": ordinal,
                    "query": query,
                    "record_kind": item["record_kind"],
                }
                if (not isinstance(input_text, str) or not input_text
                        or not isinstance(source, str) or not source
                        or not isinstance(tokens, list)
                        or any(not isinstance(token, str) or not token
                               for token in tokens)
                        or tuple(tokens)
                        != localization_structure_tokens(input_text)
                        or item.get("shape_id")
                        != localization_record_id(identity)):
                    raise BroadQaExternalDataError(
                        "v10 precision candidate pack runtime query 漂移")
                shapes.append(item)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 precision candidate pack runtime shapes 不可读") from error
    payload = b"".join(payload_parts)
    if (len(payload) != record["bytes"]
            or digest.hexdigest() != record["sha256"]
            or record.get("record_count") != len(shapes)):
        raise BroadQaExternalDataError(
            "v10 precision candidate pack runtime shape identity 漂移")
    return tuple(shapes), payload


def derive_normalization_recovery_v10_precision_runtime_shapes(
        cases: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从 TRAIN case 只投影 query，永久排除 expected output 与 family。"""
    if (not isinstance(cases, tuple)
            or len(cases) != V10_PRECISION_RUNTIME_QUERY_COUNT):
        raise BroadQaExternalDataError(
            "v10 precision candidate pack runtime denominator 漂移")
    shapes = []
    for ordinal, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BroadQaExternalDataError(
                "v10 precision candidate pack TRAIN case 非对象")
        input_text = case.get("input_text")
        source = case.get("official_source_text")
        tokens = case.get("structure_tokens")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(source, str) or not source
                or not isinstance(tokens, list)
                or any(not isinstance(item, str) or not item for item in tokens)
                or tuple(tokens) != localization_structure_tokens(input_text)):
            raise BroadQaExternalDataError(
                "v10 precision candidate pack TRAIN query 漂移")
        query = {
            "input_text": input_text,
            "official_source_text": source,
            "structure_tokens": tokens,
        }
        identity = {
            "ordinal": ordinal,
            "query": query,
            "record_kind": (
                NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_SHAPE_KIND),
        }
        shapes.append({
            **identity,
            "expected_output_included": 0,
            "format_version": 1,
            "shape_id": localization_record_id(identity),
            "source_family_included": 0,
            "training_query_only": 1,
        })
    return tuple(shapes)


def _derive(
        *, feasibility_dir: Path,
        predecessor_feasibility_dir: Path,
        base_candidate_dir: Path,
        protocol_dir: Path,
        observation_dir: Path,
        opencc_source_pack_dir: Path,
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object],
            tuple[dict[str, object], ...], dict[str, bytes],
        ]:
    """严格重派生 v2 feasibility，并形成无 label runtime roster。"""
    feasibility, candidate, preflight, audit, loso = (
        read_normalization_recovery_v10_precision_feasibility_v2(
            feasibility_dir,
            predecessor_feasibility_dir=predecessor_feasibility_dir,
            expected_predecessor_feasibility_manifest_sha256=(
                V10_PRECISION_FEASIBILITY_V1_MANIFEST_SHA256),
            base_candidate_dir=base_candidate_dir,
            expected_base_candidate_manifest_sha256=(
                V10_PRECISION_BASE_CANDIDATE_MANIFEST_SHA256),
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=(
                V10_PRECISION_TRAINING_PROTOCOL_MANIFEST_SHA256),
            observation_dir=observation_dir,
            expected_observation_manifest_sha256=(
                V10_PRECISION_OBSERVATION_PACK_MANIFEST_SHA256),
            opencc_source_pack_dir=opencc_source_pack_dir,
            expected_opencc_source_manifest_sha256=(
                V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256),
            expected_feasibility_manifest_sha256=(
                V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256),
        ))
    (_base, _controls, _safety, cases, _routes, _census) = (
        load_normalization_recovery_v10_precision_material(
            base_candidate_dir=base_candidate_dir,
            expected_base_candidate_manifest_sha256=(
                V10_PRECISION_BASE_CANDIDATE_MANIFEST_SHA256),
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=(
                V10_PRECISION_TRAINING_PROTOCOL_MANIFEST_SHA256),
            observation_dir=observation_dir,
            expected_observation_manifest_sha256=(
                V10_PRECISION_OBSERVATION_PACK_MANIFEST_SHA256),
            opencc_source_pack_dir=opencc_source_pack_dir,
            expected_opencc_source_manifest_sha256=(
                V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256),
        ))
    if (feasibility.get("manifest_sha256")
            != V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256
            or candidate.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_KIND
            or candidate.get("status")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_STATUS
            or candidate.get("candidate_program_sha256")
            != V10_PRECISION_CANDIDATE_PROGRAM_SHA256
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0
            or preflight.get("preflight_sha256")
            != V10_PRECISION_PREFLIGHT_SHA256
            or preflight.get("failure_count") != 0
            or audit.get("training_audit_sha256")
            != V10_PRECISION_TRAINING_AUDIT_SHA256
            or audit.get("case_count") != V10_PRECISION_RUNTIME_QUERY_COUNT
            or audit.get("outcomes") != {
                "EXACT": V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT,
                "UNKNOWN": (V10_PRECISION_RUNTIME_QUERY_COUNT
                            - V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT),
                "WRONG": 0,
                "MISMATCH": 0,
            }
            or loso.get("loso_audit_sha256")
            != V10_PRECISION_SOURCE_LOSO_AUDIT_SHA256
            or loso.get("status") != "PASS_ZERO_WRONG_NONZERO_EXACT"):
        raise BroadQaExternalDataError(
            "v10 precision candidate pack feasibility state 漂移")
    shapes = derive_normalization_recovery_v10_precision_runtime_shapes(cases)
    shape_payload = b"".join(canonical_json_line(item) for item in shapes)
    query_payload = b"".join(
        canonical_json_line(item["query"]) for item in shapes)
    payloads = {
        "candidate-program.json": canonical_json_line(candidate),
        "preflight.json": canonical_json_line(preflight),
        "runtime-shapes.jsonl": shape_payload,
    }
    manifest = {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_KIND),
        "base_candidate_manifest_sha256": (
            V10_PRECISION_BASE_CANDIDATE_MANIFEST_SHA256),
        "candidate_program_sha256": V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
        "expected_source_commit_count": (
            V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT),
        "files": [
            _artifact(
                name, role, payloads[name],
                record_count=(len(shapes)
                              if name == "runtime-shapes.jsonl" else None),
            )
            for name, role in _FILES
        ],
        "formal_or_evaluation_payload_read_count": 0,
        "format_version": 1,
        "mastery_claimed": 0,
        "observation_pack_manifest_sha256": (
            V10_PRECISION_OBSERVATION_PACK_MANIFEST_SHA256),
        "opencc_source_pack_manifest_sha256": (
            V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256),
        "predecessor_feasibility_manifest_sha256": (
            V10_PRECISION_FEASIBILITY_V1_MANIFEST_SHA256),
        "preflight_sha256": V10_PRECISION_PREFLIGHT_SHA256,
        "production_enabled": 0,
        "query_roster_bytes": len(query_payload),
        "query_roster_sha256": _sha256(query_payload),
        "rule_counts": candidate["rule_counts"],
        "runtime_shape_count": len(shapes),
        "runtime_shapes_sha256": _sha256(shape_payload),
        "source_loso_audit_sha256": V10_PRECISION_SOURCE_LOSO_AUDIT_SHA256,
        "status": NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_audit_sha256": V10_PRECISION_TRAINING_AUDIT_SHA256,
        "training_protocol_manifest_sha256": (
            V10_PRECISION_TRAINING_PROTOCOL_MANIFEST_SHA256),
        "v2_feasibility_manifest_sha256": (
            V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256),
    }
    return manifest, candidate, preflight, shapes, payloads


def publish_normalization_recovery_v10_precision_candidate_pack(
        *, run_root: str | Path,
        feasibility_dir: str | Path,
        predecessor_feasibility_dir: str | Path,
        base_candidate_dir: str | Path,
        protocol_dir: str | Path,
        observation_dir: str | Path,
        opencc_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 source-only 候选与 33,179 条 runtime shape。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=label) for value, label in (
        (feasibility_dir, "feasibility_dir"),
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
            "v10 precision candidate pack path 非法")
    manifest, _candidate, _preflight, _shapes, payloads = _derive(
        feasibility_dir=paths[0],
        predecessor_feasibility_dir=paths[1],
        base_candidate_dir=paths[2],
        protocol_dir=paths[3],
        observation_dir=paths[4],
        opencc_source_pack_dir=paths[5],
    )
    target = paths[-1]
    target.mkdir()
    for name, _role in _FILES:
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v10_precision_candidate_pack_runtime_payload(
        source_dir: str | Path, *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object],
            tuple[dict[str, object], ...],
        ]:
    """只从自包含 pack 回读 runtime payload，不重读上游 TRAIN material。"""
    root = Path(source_dir).resolve()
    expected_sha = _sha_value(
        expected_manifest_sha256, label="runtime manifest SHA")
    try:
        physical = {item.name for item in root.iterdir()}
        encoded = (root / "manifest.json").read_bytes()
        manifest = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 precision candidate pack runtime payload 不可读") from error
    if (physical != {"manifest.json", *[name for name, _role in _FILES]}
            or _sha256(encoded) != expected_sha
            or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded
            or manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_KIND
            or manifest.get("status")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_STATUS
            or manifest.get("candidate_program_sha256")
            != V10_PRECISION_CANDIDATE_PROGRAM_SHA256
            or manifest.get("v2_feasibility_manifest_sha256")
            != V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256
            or manifest.get("runtime_shape_count")
            != V10_PRECISION_RUNTIME_QUERY_COUNT
            or manifest.get("expected_source_commit_count")
            != V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT
            or manifest.get("production_enabled") != 0
            or manifest.get("mastery_claimed") != 0
            or manifest.get("formal_or_evaluation_payload_read_count") != 0
            or manifest.get("teacher_api_llm_call_count") != 0):
        raise BroadQaExternalDataError(
            "v10 precision candidate pack runtime manifest 漂移")
    candidate = _read_committed_json(
        root, manifest, "candidate-program.json")
    preflight = _read_committed_json(root, manifest, "preflight.json")
    shapes, shape_payload = _read_runtime_shapes(root, manifest)
    derived_preflight = (
        derive_normalization_recovery_v10_precision_v2_preflight(candidate))
    query_payload = b"".join(
        canonical_json_line(item["query"]) for item in shapes)
    if (candidate.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_KIND
            or candidate.get("status")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_STATUS
            or candidate.get("candidate_program_sha256")
            != V10_PRECISION_CANDIDATE_PROGRAM_SHA256
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0
            or preflight != derived_preflight
            or preflight.get("preflight_sha256")
            != V10_PRECISION_PREFLIGHT_SHA256
            or preflight.get("failure_count") != 0
            or manifest.get("runtime_shapes_sha256")
            != _sha256(shape_payload)
            or manifest.get("query_roster_bytes") != len(query_payload)
            or manifest.get("query_roster_sha256") != _sha256(query_payload)):
        raise BroadQaExternalDataError(
            "v10 precision candidate pack runtime payload 漂移")
    return (
        {**manifest, "manifest_sha256": expected_sha},
        candidate,
        preflight,
        shapes,
    )


def read_normalization_recovery_v10_precision_candidate_pack(
        source_dir: str | Path, *,
        feasibility_dir: str | Path,
        predecessor_feasibility_dir: str | Path,
        base_candidate_dir: str | Path,
        protocol_dir: str | Path,
        observation_dir: str | Path,
        opencc_source_pack_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object],
            tuple[dict[str, object], ...],
        ]:
    """重派生并严格回读 candidate、preflight 与 runtime shapes。"""
    root = Path(source_dir).resolve()
    expected_sha = _sha_value(
        expected_manifest_sha256, label="expected manifest SHA")
    expected, candidate, preflight, shapes, payloads = _derive(
        feasibility_dir=Path(feasibility_dir).resolve(),
        predecessor_feasibility_dir=Path(
            predecessor_feasibility_dir).resolve(),
        base_candidate_dir=Path(base_candidate_dir).resolve(),
        protocol_dir=Path(protocol_dir).resolve(),
        observation_dir=Path(observation_dir).resolve(),
        opencc_source_pack_dir=Path(opencc_source_pack_dir).resolve(),
    )
    try:
        physical = {item.name for item in root.iterdir()}
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 precision candidate pack 不可读") from error
    if (physical != {"manifest.json", *[name for name, _role in _FILES]}
            or _sha256(encoded) != expected_sha
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored != expected):
        raise BroadQaExternalDataError(
            "v10 precision candidate pack manifest 漂移")
    for name, _role in _FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v10 precision candidate pack {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v10 precision candidate pack {name} 重派生漂移")
    return (
        {**stored, "manifest_sha256": expected_sha},
        candidate,
        preflight,
        shapes,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_STATUS",
    "NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_SHAPE_KIND",
    "V10_PRECISION_CANDIDATE_PROGRAM_SHA256",
    "V10_PRECISION_FEASIBILITY_V1_MANIFEST_SHA256",
    "V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256",
    "V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT",
    "V10_PRECISION_RUNTIME_QUERY_COUNT",
    "derive_normalization_recovery_v10_precision_runtime_shapes",
    "publish_normalization_recovery_v10_precision_candidate_pack",
    "read_normalization_recovery_v10_precision_candidate_pack",
    "read_normalization_recovery_v10_precision_candidate_pack_runtime_payload",
]
