"""发布并严格回读 recovery-v6 TRAIN-only audit artifact。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_protocol import (
    read_normalization_recovery_v5_learner_input,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_rule_pack import (
    read_normalization_recovery_v6_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_training_audit_records import (
    derive_normalization_recovery_v6_training_audit,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_V1")
NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_STATUS = (
    "TRAIN_ONLY_COMPLETE_NOT_FORMAL_NOT_DEPLOYED")

AUDIT_FILES = (
    ("runtime-audit.jsonl", "V6_FULL_PACK_RUNTIME_FACILITY", "case_id"),
    ("loso-audit.jsonl", "V6_FOUR_SOURCE_LOSO_CAPABILITY", "loso_id"),
)


def _sha256(payload: bytes) -> str:
    """返回 artifact 或文件 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(value, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(value, list):
        return (len(value) == len(expected)
                and all(_strict_equal(left, right)
                        for left, right in zip(value, expected)))
    return value == expected


def _require_k_root(value: str | Path) -> Path:
    """要求显式 v6 audit 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v6 audit root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析 artifact 路径并拒绝逃出显式 K 盘根。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或存在包含关系。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _read_manifest_only(
        root: Path,
        *,
        expected_manifest_sha256: str,
        label: str,
        ) -> dict[str, object]:
    """只打开 sealed manifest 并核验外部 SHA 与规范编码。"""
    expected_sha = _sha_value(
        expected_manifest_sha256, label=f"v6 audit expected {label}")
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v6 audit {label} manifest 不可读") from error
    if (_sha256(encoded) != expected_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            f"v6 audit {label} manifest identity/encoding 漂移")
    return stored


def _read_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    """严格解析一个 compact canonical JSONL reference。"""
    values = []
    try:
        for line in payload.splitlines(keepends=True):
            value = json.loads(line)
            if not isinstance(value, dict) or canonical_json_line(value) != line:
                raise BroadQaExternalDataError(f"{label} JSONL 非规范")
            values.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} JSONL 不可读") from error
    if not values:
        raise BroadQaExternalDataError(f"{label} JSONL 为空")
    return tuple(values)


def _read_simulation_reference(
        root: Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...]]:
    """只读 simulation manifest 与 16/4 条 compact reference 文件。"""
    manifest = _read_manifest_only(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
        label="successor simulation",
    )
    file_by_name = {
        str(item.get("relative_path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict)}
    outputs = []
    for name in ("strategy-family-results.jsonl", "strategy-results.jsonl"):
        commitment = file_by_name.get(name)
        if (not isinstance(commitment, dict)
                or type(commitment.get("bytes")) is not int
                or type(commitment.get("record_count")) is not int):
            raise BroadQaExternalDataError(
                "v6 audit simulation compact commitment 缺失")
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v6 audit simulation {name} 不可读") from error
        values = _read_jsonl(payload, label=f"simulation {name}")
        if (len(payload) != commitment["bytes"]
                or len(values) != commitment["record_count"]
                or _sha256(payload) != commitment.get("sha256")):
            raise BroadQaExternalDataError(
                "v6 audit simulation compact file identity 漂移")
        outputs.append(values)
    return ({**manifest, "manifest_sha256": expected_manifest_sha256},
            outputs[0], outputs[1])


def _payload(values: tuple[dict[str, object], ...]) -> bytes:
    """把 audit records 编码为规范 JSONL。"""
    return b"".join(canonical_json_line(item) for item in values)


def _artifact(
        *,
        name: str,
        role: str,
        values: tuple[dict[str, object], ...],
        payload: bytes,
        ) -> dict[str, object]:
    """构造一个 v6 audit 文件承诺。"""
    return {
        "bytes": len(payload),
        "record_count": len(values),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _derive(
        *,
        protocol_dir: Path,
        expected_protocol_manifest_sha256: str,
        predecessor_pack_dir: Path,
        expected_predecessor_pack_manifest_sha256: str,
        pack_dir: Path,
        expected_pack_manifest_sha256: str,
        denominator_audit_dir: Path,
        expected_denominator_audit_manifest_sha256: str,
        simulation_dir: Path,
        expected_simulation_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]],
                   dict[str, bytes]]:
    """严格回读 TRAIN/pack 与 compact references 后重派生 audit。"""
    protocol_values = read_normalization_recovery_v5_learner_input(
        protocol_dir,
        expected_manifest_sha256=expected_protocol_manifest_sha256,
    )
    protocol_manifest, observations, fragments, _groups, _work = protocol_values
    pack_manifest, pack_outputs = read_normalization_recovery_v6_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        predecessor_pack_dir=predecessor_pack_dir,
        expected_predecessor_pack_manifest_sha256=(
            expected_predecessor_pack_manifest_sha256),
        expected_pack_manifest_sha256=expected_pack_manifest_sha256,
    )
    denominator_manifest = _read_manifest_only(
        denominator_audit_dir,
        expected_manifest_sha256=(
            expected_denominator_audit_manifest_sha256),
        label="v5 denominator audit",
    )
    simulation_manifest, family_records, strategy_records = (
        _read_simulation_reference(
            simulation_dir,
            expected_manifest_sha256=expected_simulation_manifest_sha256,
        ))
    runtime, loso, summary = derive_normalization_recovery_v6_training_audit(
        protocol_manifest=protocol_manifest,
        observations=observations,
        fragments=fragments,
        pack_manifest=pack_manifest,
        pack_outputs=pack_outputs,
        audit_manifest_sha256=expected_denominator_audit_manifest_sha256,
        audit_manifest=denominator_manifest,
        simulation_manifest_sha256=expected_simulation_manifest_sha256,
        simulation_manifest=simulation_manifest,
        simulation_family_records=family_records,
        simulation_strategy_records=strategy_records,
    )
    outputs = {
        "runtime-audit.jsonl": runtime,
        "loso-audit.jsonl": loso,
    }
    payloads = {name: _payload(outputs[name])
                for name, _role, _identity in AUDIT_FILES}
    files = [_artifact(
        name=name, role=role, values=outputs[name], payload=payloads[name])
        for name, role, _identity in AUDIT_FILES]
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_KIND,
        "candidate_pack_read_count": 0,
        "denominator_audit_manifest_only_read_count": 1,
        "denominator_audit_manifest_sha256": (
            expected_denominator_audit_manifest_sha256),
        "denominator_audit_non_manifest_read_count": 0,
        "evaluation_commitment_read_count": 0,
        "evaluation_payload_read_count": 0,
        "files": files,
        "formal_run_count": 0,
        "format_version": 1,
        "mastery_claimed": 0,
        "pack_manifest_sha256": expected_pack_manifest_sha256,
        "predecessor_rule_pack_manifest_sha256": (
            expected_predecessor_pack_manifest_sha256),
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": expected_protocol_manifest_sha256,
        "reserve_identity_read_count": 0,
        "reserve_payload_read_count": 0,
        "simulation_case_file_read_count": 0,
        "simulation_compact_file_read_count": 2,
        "simulation_manifest_sha256": expected_simulation_manifest_sha256,
        "status": NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
    }
    return manifest, outputs, payloads


def publish_normalization_recovery_v6_training_audit(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        predecessor_pack_dir: str | Path,
        expected_predecessor_pack_manifest_sha256: str,
        pack_dir: str | Path,
        expected_pack_manifest_sha256: str,
        denominator_audit_dir: str | Path,
        expected_denominator_audit_manifest_sha256: str,
        simulation_dir: str | Path,
        expected_simulation_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 v6 TRAIN-only audit，并以 manifest-last 封口。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=label) for value, label in (
        (protocol_dir, "protocol_dir"),
        (predecessor_pack_dir, "predecessor_pack_dir"),
        (pack_dir, "pack_dir"),
        (denominator_audit_dir, "denominator_audit_dir"),
        (simulation_dir, "simulation_dir"),
        (target_dir, "target_dir"),
    ))
    protocol_root, predecessor_root, pack_root, denominator_root, simulation_root, target = paths
    shas = tuple(_sha_value(value, label=label) for value, label in (
        (expected_protocol_manifest_sha256, "v6 audit protocol manifest"),
        (expected_predecessor_pack_manifest_sha256, "v6 audit predecessor pack"),
        (expected_pack_manifest_sha256, "v6 audit pack"),
        (expected_denominator_audit_manifest_sha256, "v6 audit denominator"),
        (expected_simulation_manifest_sha256, "v6 audit simulation"),
    ))
    roots = paths
    if (target.exists()
            or any(not path.is_dir() for path in roots[:-1])
            or any(_overlap(left, right)
                   for index, left in enumerate(roots)
                   for right in roots[index + 1:])):
        raise BroadQaExternalDataError(
            "v6 audit 输入缺失、artifact 混淆或 target 已存在")
    manifest, outputs, payloads = _derive(
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=shas[0],
        predecessor_pack_dir=predecessor_root,
        expected_predecessor_pack_manifest_sha256=shas[1],
        pack_dir=pack_root,
        expected_pack_manifest_sha256=shas[2],
        denominator_audit_dir=denominator_root,
        expected_denominator_audit_manifest_sha256=shas[3],
        simulation_dir=simulation_root,
        expected_simulation_manifest_sha256=shas[4],
    )
    target.mkdir(parents=True)
    for name, _role, identity_key in AUDIT_FILES:
        identities = [str(item[identity_key]) for item in outputs[name]]
        if len(set(identities)) != len(identities):
            raise BroadQaExternalDataError(f"v6 audit {name} identity 重复")
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v6_training_audit(
        audit_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        predecessor_pack_dir: str | Path,
        expected_predecessor_pack_manifest_sha256: str,
        pack_dir: str | Path,
        expected_pack_manifest_sha256: str,
        denominator_audit_dir: str | Path,
        expected_denominator_audit_manifest_sha256: str,
        simulation_dir: str | Path,
        expected_simulation_manifest_sha256: str,
        expected_audit_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """以六个外部 SHA 重派生并严格回读完整 v6 audit。"""
    root = Path(audit_dir).resolve()
    input_roots = tuple(Path(value).resolve() for value in (
        protocol_dir, predecessor_pack_dir, pack_dir,
        denominator_audit_dir, simulation_dir))
    roots = (root, *input_roots)
    if any(_overlap(left, right)
           for index, left in enumerate(roots)
           for right in roots[index + 1:]):
        raise BroadQaExternalDataError("v6 audit artifact 根混淆")
    expected, outputs, payloads = _derive(
        protocol_dir=input_roots[0],
        expected_protocol_manifest_sha256=_sha_value(
            expected_protocol_manifest_sha256, label="v6 expected protocol"),
        predecessor_pack_dir=input_roots[1],
        expected_predecessor_pack_manifest_sha256=_sha_value(
            expected_predecessor_pack_manifest_sha256,
            label="v6 expected predecessor pack"),
        pack_dir=input_roots[2],
        expected_pack_manifest_sha256=_sha_value(
            expected_pack_manifest_sha256, label="v6 expected pack"),
        denominator_audit_dir=input_roots[3],
        expected_denominator_audit_manifest_sha256=_sha_value(
            expected_denominator_audit_manifest_sha256,
            label="v6 expected denominator"),
        simulation_dir=input_roots[4],
        expected_simulation_manifest_sha256=_sha_value(
            expected_simulation_manifest_sha256,
            label="v6 expected simulation"),
    )
    audit_sha = _sha_value(
        expected_audit_manifest_sha256, label="v6 expected audit")
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v6 audit manifest 不可读") from error
    if (_sha256(encoded) != audit_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not _strict_equal(stored, expected)):
        raise BroadQaExternalDataError(
            "v6 audit manifest identity/encoding/material 漂移")
    for name, _role, _identity in AUDIT_FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(f"v6 audit {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v6 audit {name} 与 TRAIN 重派生漂移")
    return ({**stored, "manifest_sha256": audit_sha}, outputs)


__all__ = [
    "AUDIT_FILES",
    "NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_STATUS",
    "publish_normalization_recovery_v6_training_audit",
    "read_normalization_recovery_v6_training_audit",
]
