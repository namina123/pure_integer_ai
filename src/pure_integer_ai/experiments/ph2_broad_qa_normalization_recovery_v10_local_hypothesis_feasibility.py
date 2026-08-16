"""发布 recovery-v10 local hypothesis projection+LOSO 的TRAIN-only NE证据。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_sources import (
    read_opencc_unique_t2s_routes,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_loso import (
    derive_normalization_recovery_v10_local_hypothesis_loso,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_projection import (
    derive_normalization_recovery_v10_local_hypothesis_projection,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack import (
    V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
    V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256,
    V10_PRECISION_OBSERVATION_PACK_MANIFEST_SHA256,
    V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_feasibility import (
    load_normalization_recovery_v10_precision_observations,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_FEASIBILITY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_FEASIBILITY_V1")
NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_FEASIBILITY_STATUS = (
    "TRAIN_ONLY_NE_PREDECESSOR_ONLY_NO_AUTHORIZATION_NOT_FORMAL")

_OUTPUT_FILES = (
    ("projection-summary.json", "LOCAL_HYPOTHESIS_PROJECTION_SUMMARY"),
    ("loso-audit.json", "LOCAL_HYPOTHESIS_LOSO_AGGREGATE_AUDIT"),
    ("survivors.jsonl", "LOCAL_HYPOTHESIS_PREDECESSOR_COVERED_SURVIVORS"),
)


def _sha256(payload: bytes) -> str:
    """返回输入、输出、源码或manifest的SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(
            f"v10 local hypothesis feasibility {label} 非法")
    return value


def _require_k_root(value: str | Path) -> Path:
    """要求显式、已存在的K盘run root。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析路径并拒绝逃出唯一K盘run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v10 local hypothesis feasibility {label} 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个artifact根是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _file_record(
        manifest: dict[str, object], relative_path: str,
        ) -> dict[str, object]:
    """从manifest选择唯一文件承诺。"""
    files = manifest.get("files")
    matches = [
        item for item in files
        if isinstance(item, dict)
        and item.get("relative_path") == relative_path
    ] if isinstance(files, list) else []
    if len(matches) != 1:
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility file commitment 漂移")
    record = matches[0]
    if (type(record.get("bytes")) is not int or record["bytes"] < 0
            or _sha_value(record.get("sha256"), label="file SHA")
            != record["sha256"]):
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility file record 漂移")
    return record


def _candidate_source_rules(
        feasibility_dir: Path,
        ) -> tuple[dict[str, object], ...]:
    """从固定v2 feasibility直接读取自绑定candidate的7条source规则。"""
    try:
        manifest_payload = (feasibility_dir / "manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility v2 manifest 不可读") from error
    if (_sha256(manifest_payload)
            != V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256
            or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != manifest_payload
            or manifest.get("candidate_program_sha256")
            != V10_PRECISION_CANDIDATE_PROGRAM_SHA256
            or manifest.get("production_enabled") != 0
            or manifest.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility v2 manifest 漂移")
    candidate_record = _file_record(manifest, "candidate-program.json")
    try:
        candidate_payload = (
            feasibility_dir / "candidate-program.json").read_bytes()
        candidate = json.loads(candidate_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility candidate 不可读") from error
    if (len(candidate_payload) != candidate_record["bytes"]
            or _sha256(candidate_payload) != candidate_record["sha256"]
            or not isinstance(candidate, dict)
            or canonical_json_line(candidate) != candidate_payload
            or candidate.get("candidate_program_sha256")
            != V10_PRECISION_CANDIDATE_PROGRAM_SHA256
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility candidate 漂移")
    inventories = candidate.get("inventories")
    rules = inventories.get("source_conditioned_rules") if isinstance(
        inventories, dict) else None
    if (not isinstance(rules, list) or len(rules) != 7
            or any(not isinstance(item, dict) for item in rules)):
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility source rule inventory 漂移")
    return tuple(rules)


def _artifact(
        name: str, role: str, payload: bytes, *, record_count: int | None,
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


def normalization_recovery_v10_local_hypothesis_code_files(
        ) -> list[dict[str, object]]:
    """承诺合同、projection、LOSO与artifact reader的公开源码字节。"""
    directory = Path(__file__).resolve().parent
    names = (
        "ph2_broad_qa_normalization_phrase_learning.py",
        "ph2_broad_qa_normalization_recovery_v5_localization_structure.py",
        "ph2_broad_qa_normalization_recovery_v10_local_hypothesis_contract.py",
        "ph2_broad_qa_normalization_recovery_v10_local_hypothesis_projection.py",
        "ph2_broad_qa_normalization_recovery_v10_local_hypothesis_loso.py",
        "ph2_broad_qa_normalization_recovery_v10_local_hypothesis_feasibility.py",
        "ph2_broad_qa_normalization_recovery_v10_precision_feasibility.py",
    )
    return [{
        "bytes": len(payload),
        "relative_path": f"src/pure_integer_ai/experiments/{name}",
        "sha256": _sha256(payload),
    } for name in names for payload in [(directory / name).read_bytes()]]


def _derive(
        *, observation_dir: Path,
        opencc_source_pack_dir: Path,
        feasibility_dir: Path,
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object],
            tuple[dict[str, object], ...], dict[str, bytes],
        ]:
    """从固定TRAIN输入重派生projection、LOSO与predecessor coverage。"""
    _observation_manifest, observations = (
        load_normalization_recovery_v10_precision_observations(
            observation_dir=observation_dir,
            expected_observation_manifest_sha256=(
                V10_PRECISION_OBSERVATION_PACK_MANIFEST_SHA256),
        ))
    try:
        opencc_manifest_payload = (
            opencc_source_pack_dir / "manifest.json").read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility OpenCC manifest 不可读") from error
    if (_sha256(opencc_manifest_payload)
            != V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256):
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility OpenCC identity 漂移")
    routes, opencc_census = read_opencc_unique_t2s_routes(
        opencc_source_pack_dir)
    predecessor_rules = _candidate_source_rules(feasibility_dir)
    projection_records, projection_summary = (
        derive_normalization_recovery_v10_local_hypothesis_projection(
            observations=observations,
            opencc_routes=routes,
            opencc_source_pack_manifest_sha256=(
                V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256),
        ))
    audit, survivors = derive_normalization_recovery_v10_local_hypothesis_loso(
        observations=observations,
        projection_records=projection_records,
        predecessor_source_rules=predecessor_rules,
    )
    projection_payload = b"".join(
        canonical_json_line(item) for item in projection_records)
    survivor_payload = b"".join(
        canonical_json_line(item) for item in survivors)
    if (projection_summary.get("authorization_count") != 0
            or projection_summary.get("observation_count") != 33_179
            or projection_summary.get("formal_or_evaluation_payload_read_count")
            != 0
            or audit.get("outcome") != "NE_PREDECESSOR_ONLY_SURVIVORS"
            or audit.get("outcomes", {}).get("WRONG") != 0
            or audit.get("survivor_count") != len(survivors)
            or audit.get("novel_survivor_count") != 0
            or audit.get("authorization_rule_count") != 0
            or audit.get("predecessor_rule_count") != len(predecessor_rules)
            or any(item.get("predecessor_covered") != 1
                   for item in survivors)):
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility NE gate 未闭合")
    payloads = {
        "projection-summary.json": canonical_json_line(projection_summary),
        "loso-audit.json": canonical_json_line(audit),
        "survivors.jsonl": survivor_payload,
    }
    manifest = {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_FEASIBILITY_KIND),
        "authorization_rule_count": 0,
        "candidate_program_sha256": V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
        "code_files": normalization_recovery_v10_local_hypothesis_code_files(),
        "files": [
            _artifact(
                name, role, payloads[name],
                record_count=(len(survivors)
                              if name == "survivors.jsonl" else None),
            )
            for name, role in _OUTPUT_FILES
        ],
        "formal_or_evaluation_payload_read_count": 0,
        "format_version": 1,
        "loso_audit_sha256": audit["loso_audit_sha256"],
        "loso_outcome": audit["outcome"],
        "mastery_claimed": 0,
        "novel_survivor_count": audit["novel_survivor_count"],
        "observation_pack_manifest_sha256": (
            V10_PRECISION_OBSERVATION_PACK_MANIFEST_SHA256),
        "opencc_source_pack_manifest_sha256": (
            V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256),
        "opencc_unique_route_count": opencc_census["unique_route_count"],
        "predecessor_rule_count": len(predecessor_rules),
        "production_enabled": 0,
        "projection_record_count": len(projection_records),
        "projection_records_bytes": len(projection_payload),
        "projection_records_sha256": _sha256(projection_payload),
        "projection_summary_sha256": projection_summary[
            "projection_summary_sha256"],
        "status": NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_FEASIBILITY_STATUS,
        "survivor_count": len(survivors),
        "survivors_sha256": _sha256(survivor_payload),
        "teacher_api_llm_call_count": 0,
        "v2_feasibility_manifest_sha256": (
            V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256),
    }
    return manifest, projection_summary, audit, survivors, payloads


def publish_normalization_recovery_v10_local_hypothesis_feasibility(
        *, run_root: str | Path,
        observation_dir: str | Path,
        opencc_source_pack_dir: str | Path,
        feasibility_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布projection+LOSO的紧凑NE feasibility artifact。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=label) for value, label in (
        (observation_dir, "observation_dir"),
        (opencc_source_pack_dir, "opencc_source_pack_dir"),
        (feasibility_dir, "feasibility_dir"),
        (target_dir, "target_dir"),
    ))
    if (any(not path.is_dir() for path in paths[:-1])
            or paths[-1].exists()
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility path 非法")
    manifest, _summary, _audit, _survivors, payloads = _derive(
        observation_dir=paths[0],
        opencc_source_pack_dir=paths[1],
        feasibility_dir=paths[2],
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


def read_normalization_recovery_v10_local_hypothesis_feasibility(
        source_dir: str | Path, *,
        observation_dir: str | Path,
        opencc_source_pack_dir: str | Path,
        feasibility_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object],
            tuple[dict[str, object], ...],
        ]:
    """重派生并逐字节回读projection summary、LOSO与survivor trace。"""
    root = Path(source_dir).resolve()
    expected_sha = _sha_value(
        expected_manifest_sha256, label="expected manifest SHA")
    expected, summary, audit, survivors, payloads = _derive(
        observation_dir=Path(observation_dir).resolve(),
        opencc_source_pack_dir=Path(opencc_source_pack_dir).resolve(),
        feasibility_dir=Path(feasibility_dir).resolve(),
    )
    try:
        physical = {item.name for item in root.iterdir()}
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility artifact 不可读") from error
    if (physical != {"manifest.json", *[name for name, _role in _OUTPUT_FILES]}
            or _sha256(encoded) != expected_sha
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored != expected):
        raise BroadQaExternalDataError(
            "v10 local hypothesis feasibility manifest 漂移")
    for name, _role in _OUTPUT_FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v10 local hypothesis feasibility {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v10 local hypothesis feasibility {name} 重派生漂移")
    return (
        {**stored, "manifest_sha256": expected_sha},
        summary,
        audit,
        survivors,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_FEASIBILITY_KIND",
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_FEASIBILITY_STATUS",
    "normalization_recovery_v10_local_hypothesis_code_files",
    "publish_normalization_recovery_v10_local_hypothesis_feasibility",
    "read_normalization_recovery_v10_local_hypothesis_feasibility",
]
