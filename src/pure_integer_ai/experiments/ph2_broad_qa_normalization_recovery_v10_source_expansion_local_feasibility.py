"""发布 recovery-v10 五 family local projection+LOSO 的TRAIN-only证据。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_feasibility import (
    _candidate_source_rules,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack import (
    V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
    V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256,
    V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_five_family_audit import (
    read_normalization_recovery_v10_five_family_audit_aggregate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_local_loso import (
    derive_normalization_recovery_v10_source_expansion_local_loso,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_local_projection import (
    V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES,
    derive_normalization_recovery_v10_source_expansion_local_projection,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_observation_pack import (
    V10_FIVE_FAMILY_AUDIT_MANIFEST_SHA256,
    V10_SOURCE_EXPANSION_OBSERVATION_FILES,
    read_normalization_recovery_v10_source_expansion_observation_aggregate,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


V10_SOURCE_EXPANSION_OBSERVATION_MANIFEST_SHA256 = (
    "50f1feabce731bf2bc78bad85a5507c67fa0e6d8e7ddafee6de6757ec8abeafe")
V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_LOCAL_FEASIBILITY_V1")
V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_NE_STATUS = (
    "TRAIN_ONLY_NE_NO_NOVEL_FIVE_FAMILY_LOCAL_AUTHORIZATION_NOT_FORMAL")
V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_FAIL_STATUS = (
    "TRAIN_ONLY_FAIL_WRONG_NONZERO_FIVE_FAMILY_LOCAL_AUTHORIZATION_NOT_FORMAL")
V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_PASS_STATUS = (
    "TRAIN_ONLY_PASS_NOVEL_FIVE_FAMILY_LOCAL_AUTHORIZATION_NOT_FORMAL")

_OUTPUT_FILES = (
    ("projection-summary.json", "FIVE_FAMILY_LOCAL_PROJECTION_SUMMARY"),
    ("loso-audit.json", "FIVE_FAMILY_LOCAL_LOSO_AGGREGATE_AUDIT"),
    ("survivors.jsonl", "FIVE_FAMILY_LOCAL_LOSO_SURVIVORS"),
)


def _sha256(payload: bytes) -> str:
    """返回输入、输出、源码或manifest的SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式、已存在的K盘run root。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析路径并拒绝逃出唯一K盘run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v10 expanded local feasibility {label} 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个artifact根是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _artifact(
        name: str,
        role: str,
        payload: bytes,
        *,
        record_count: int | None,
        ) -> dict[str, object]:
    """形成一份输出文件commitment。"""
    value = {
        "bytes": len(payload),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }
    if record_count is not None:
        value["record_count"] = record_count
    return value


def normalization_recovery_v10_source_expansion_local_code_files(
        ) -> list[dict[str, object]]:
    """承诺共享算法与五family revision的公开源码字节。"""
    directory = Path(__file__).resolve().parent
    names = (
        "ph2_broad_qa_normalization_phrase_learning.py",
        "ph2_broad_qa_normalization_recovery_v5_localization_structure.py",
        "ph2_broad_qa_normalization_recovery_v10_local_hypothesis_loso.py",
        "ph2_broad_qa_normalization_recovery_v10_local_hypothesis_projection.py",
        "ph2_broad_qa_normalization_recovery_v10_source_expansion_five_family_audit.py",
        "ph2_broad_qa_normalization_recovery_v10_source_expansion_local_projection.py",
        "ph2_broad_qa_normalization_recovery_v10_source_expansion_local_loso.py",
        "ph2_broad_qa_normalization_recovery_v10_source_expansion_local_feasibility.py",
        "ph2_broad_qa_normalization_recovery_v10_source_expansion_observation_pack.py",
    )
    return [{
        "bytes": len(payload),
        "relative_path": f"src/pure_integer_ai/experiments/{name}",
        "sha256": _sha256(payload),
    } for name in names for payload in [(directory / name).read_bytes()]]


def _status(outcome: str, *, novel_survivor_count: int) -> str:
    """按预先冻结的WRONG优先、novel次之门选择能力状态。"""
    if outcome == "FAIL_WRONG_NONZERO":
        return V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_FAIL_STATUS
    if (outcome == "PASS_ZERO_WRONG_NOVEL_FIVE_DIRECTION_SURVIVOR"
            and novel_survivor_count > 0):
        return V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_PASS_STATUS
    return V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_NE_STATUS


def _derive(
        *,
        observation_dir: Path,
        five_family_audit_dir: Path,
        opencc_source_pack_dir: Path,
        predecessor_feasibility_dir: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            tuple[dict[str, object], ...],
            dict[str, bytes],
        ]:
    """从四份固定TRAIN输入重派生projection、collision gate与五方向LOSO。"""
    observation_manifest, observation_outputs = (
        read_normalization_recovery_v10_source_expansion_observation_aggregate(
            observation_dir,
            expected_manifest_sha256=(
                V10_SOURCE_EXPANSION_OBSERVATION_MANIFEST_SHA256),
        ))
    observations = tuple(
        item
        for _family, name in V10_SOURCE_EXPANSION_OBSERVATION_FILES
        for item in observation_outputs[name]
    )
    audit_manifest, audit_outputs = (
        read_normalization_recovery_v10_five_family_audit_aggregate(
            five_family_audit_dir,
            expected_manifest_sha256=V10_FIVE_FAMILY_AUDIT_MANIFEST_SHA256,
        ))
    collision_records = audit_outputs["source-input-collisions.jsonl"]
    if (len(collision_records) != 33
            or any(item.get("cross_family_output_conflict") != 1
                   for item in collision_records)):
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility collision ledger 漂移")
    collisions = tuple(sorted(
        str(item["source_input_sha256"]) for item in collision_records))
    try:
        opencc_manifest_payload = (
            opencc_source_pack_dir / "manifest.json").read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility OpenCC manifest 不可读") from error
    if (_sha256(opencc_manifest_payload)
            != V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256):
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility OpenCC identity 漂移")
    routes, opencc_census = read_opencc_unique_t2s_routes(
        opencc_source_pack_dir)
    predecessor_rules = _candidate_source_rules(predecessor_feasibility_dir)
    projection_records, projection_summary = (
        derive_normalization_recovery_v10_source_expansion_local_projection(
            observations=observations,
            opencc_routes=routes,
            opencc_source_pack_manifest_sha256=(
                V10_PRECISION_OPENCC_SOURCE_PACK_MANIFEST_SHA256),
        ))
    loso_audit, survivors = (
        derive_normalization_recovery_v10_source_expansion_local_loso(
            observations=observations,
            projection_records=projection_records,
            predecessor_source_rules=predecessor_rules,
            collision_source_input_sha256s=collisions,
        ))
    projection_payload = b"".join(
        canonical_json_line(item) for item in projection_records)
    survivor_payload = b"".join(
        canonical_json_line(item) for item in survivors)
    observation_summary = observation_manifest.get("summary")
    if (not isinstance(observation_summary, dict)
            or projection_summary.get("authorization_count") != 0
            or projection_summary.get("observation_count") != 37_939
            or projection_summary.get("observation_count")
            != observation_summary.get("observation_count")
            or projection_summary.get("formal_or_evaluation_payload_read_count")
            != 0
            or loso_audit.get("direction_count") != 5
            or loso_audit.get("collision_ledger_count") != 33
            or loso_audit.get("collision_matched_count") != 33
            or loso_audit.get("survivor_count") != len(survivors)
            or loso_audit.get("authorization_rule_count") != 0
            or loso_audit.get("predecessor_rule_count") != len(
                predecessor_rules)):
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility gate 未闭合")
    status = _status(
        str(loso_audit["outcome"]),
        novel_survivor_count=int(loso_audit["novel_survivor_count"]),
    )
    payloads = {
        "projection-summary.json": canonical_json_line(projection_summary),
        "loso-audit.json": canonical_json_line(loso_audit),
        "survivors.jsonl": survivor_payload,
    }
    manifest = {
        "artifact_kind": V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_KIND,
        "authorization_rule_count": 0,
        "candidate_program_sha256": V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
        "code_files": (
            normalization_recovery_v10_source_expansion_local_code_files()),
        "collision_ledger_count": len(collisions),
        "collision_ledger_sha256": next(
            item["sha256"] for item in audit_manifest["files"]
            if item["relative_path"] == "source-input-collisions.jsonl"),
        "collision_vetoed_hypothesis_count": loso_audit[
            "collision_vetoed_hypothesis_count"],
        "collision_vetoed_projection_record_count": loso_audit[
            "collision_vetoed_projection_record_count"],
        "files": [
            _artifact(
                name,
                role,
                payloads[name],
                record_count=(len(survivors)
                              if name == "survivors.jsonl" else None),
            )
            for name, role in _OUTPUT_FILES
        ],
        "five_family_audit_manifest_sha256": (
            V10_FIVE_FAMILY_AUDIT_MANIFEST_SHA256),
        "formal_or_evaluation_payload_read_count": 0,
        "format_version": 1,
        "loso_audit_sha256": loso_audit["loso_audit_sha256"],
        "loso_outcome": loso_audit["outcome"],
        "loso_outcomes": loso_audit["outcomes"],
        "mastery_claimed": 0,
        "novel_survivor_count": loso_audit["novel_survivor_count"],
        "observation_pack_manifest_sha256": (
            V10_SOURCE_EXPANSION_OBSERVATION_MANIFEST_SHA256),
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
        "status": status,
        "survivor_count": len(survivors),
        "survivors_sha256": _sha256(survivor_payload),
        "teacher_api_llm_call_count": 0,
        "v2_feasibility_manifest_sha256": (
            V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256),
    }
    return manifest, projection_summary, loso_audit, survivors, payloads


def publish_normalization_recovery_v10_source_expansion_local_feasibility(
        *,
        run_root: str | Path,
        observation_dir: str | Path,
        five_family_audit_dir: str | Path,
        opencc_source_pack_dir: str | Path,
        predecessor_feasibility_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布五family projection+LOSO紧凑feasibility artifact。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=label) for value, label in (
        (observation_dir, "observation_dir"),
        (five_family_audit_dir, "five_family_audit_dir"),
        (opencc_source_pack_dir, "opencc_source_pack_dir"),
        (predecessor_feasibility_dir, "predecessor_feasibility_dir"),
        (target_dir, "target_dir"),
    ))
    if (any(not path.is_dir() for path in paths[:-1])
            or paths[-1].exists()
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility path 非法")
    manifest, _summary, _audit, _survivors, payloads = _derive(
        observation_dir=paths[0],
        five_family_audit_dir=paths[1],
        opencc_source_pack_dir=paths[2],
        predecessor_feasibility_dir=paths[3],
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


def read_normalization_recovery_v10_source_expansion_local_feasibility(
        source_dir: str | Path,
        *,
        observation_dir: str | Path,
        five_family_audit_dir: str | Path,
        opencc_source_pack_dir: str | Path,
        predecessor_feasibility_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            tuple[dict[str, object], ...],
        ]:
    """重派生并逐字节回读projection summary、LOSO与survivor trace。"""
    root = Path(source_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_KIND):
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility manifest identity 漂移")
    expected, summary, audit, survivors, payloads = _derive(
        observation_dir=Path(observation_dir).resolve(),
        five_family_audit_dir=Path(five_family_audit_dir).resolve(),
        opencc_source_pack_dir=Path(opencc_source_pack_dir).resolve(),
        predecessor_feasibility_dir=Path(
            predecessor_feasibility_dir).resolve(),
    )
    try:
        stored_payloads = {
            name: (root / name).read_bytes() for name, _role in _OUTPUT_FILES
        }
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility output 不可读") from error
    if stored_payloads != payloads or stored != expected:
        raise BroadQaExternalDataError(
            "v10 expanded local feasibility records/fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, summary, audit, survivors


__all__ = [
    "V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_FAIL_STATUS",
    "V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_KIND",
    "V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_NE_STATUS",
    "V10_SOURCE_EXPANSION_LOCAL_FEASIBILITY_PASS_STATUS",
    "publish_normalization_recovery_v10_source_expansion_local_feasibility",
    "read_normalization_recovery_v10_source_expansion_local_feasibility",
]
