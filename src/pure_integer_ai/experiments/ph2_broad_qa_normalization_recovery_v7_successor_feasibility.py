"""发布 recovery-v7 三项 TRAIN-only successor feasibility artifact。

publisher 只读取既有四来源 TRAIN protocol/rule pack、v6 TRAIN audit manifest 与
v7 VLC commitment manifest。VLC/Qt source、identity roster、translation label、
candidate 和 formal artifact 均不可读；输出只含 surface hash 与不可执行合同。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_successor_feasibility_records import (
    derive_context_scoped_local_projections,
    derive_identity_inputs,
    derive_source_policy_replay_projections,
    derive_variable_structure_projections,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_V1")
NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_STATUS = (
    "TRAIN_ONLY_FEASIBILITY_COMPLETE_NOT_LEARNER_NOT_RUNTIME")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V5_RULE_PACK_MANIFEST_SHA256 = (
    "d5d930508b891d489704084c82a512a1c31d66ab82f646bf016a32ac31cdb144")
V6_TRAINING_AUDIT_MANIFEST_SHA256 = (
    "385efb20b38f5a5580326d1ac21064f33f795db45d01acaff62fbd406338cfbc")
V7_EVALUATION_COMMITMENT_MANIFEST_SHA256 = (
    "a406598a134a0390e101419518f81bf9877a415e8b4b060c4982be0e1844a8d4")

_INPUT_FILES = {
    "observations": (
        "train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS"),
    "target_rules": (
        "target-phrase-rules.jsonl", "LEARNED_TARGET_PHRASE_RULES"),
    "evidence": ("evidence.jsonl", "LEARNED_SCOPED_PHRASE_EVIDENCE"),
    "conflicts": ("conflict-ledger.jsonl", "LEARNED_CONFLICT_LEDGER"),
    "identities": (
        "identity-observations.jsonl", "IDENTITY_PRESERVATION_AUDIT_BUCKET"),
}
_OUTPUT_FILES = (
    ("variable-structure-projections.jsonl",
     "VARIABLE_STRUCTURE_TRANSFER_FEASIBILITY"),
    ("context-scoped-local-projections.jsonl",
     "CONTEXT_SCOPED_LOCAL_TRANSFER_FEASIBILITY"),
    ("source-policy-replay-projections.jsonl",
     "SOURCE_POLICY_REPLAY_FEASIBILITY"),
)


def _sha256(payload: bytes) -> str:
    """返回文件或规范 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """复用严格 JSON 比较并保持本模块错误边界。"""
    return strict_json_equal(value, expected)


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v7 feasibility root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制一个输入或输出路径位于 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"v7 feasibility {label} path 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个物理目录是否互为祖先或相同。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _read_manifest(
        directory: Path,
        *,
        expected_sha256: str,
        label: str,
        ) -> dict[str, object]:
    """读取一个规范 manifest，并核对固定 SHA。"""
    try:
        payload = (directory / "manifest.json").read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v7 feasibility {label} manifest 不可读") from error
    if (not isinstance(value, dict)
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or _sha256(payload) != expected_sha256
            or canonical_json_line(value) != payload):
        raise BroadQaExternalDataError(
            f"v7 feasibility {label} manifest identity 漂移")
    return value


def _artifact_from_manifest(
        manifest: dict[str, object],
        *,
        relative_path: str,
        role: str,
        ) -> dict[str, object]:
    """从 sealed manifest 取得唯一输入文件承诺。"""
    files = manifest.get("files")
    if not isinstance(files, list):
        raise BroadQaExternalDataError("v7 feasibility input files 非列表")
    matches = [item for item in files if isinstance(item, dict)
               and item.get("relative_path") == relative_path
               and item.get("role") == role]
    if len(matches) != 1:
        raise BroadQaExternalDataError(
            f"v7 feasibility input artifact {relative_path} 漂移")
    value = matches[0]
    if (type(value.get("bytes")) is not int or value["bytes"] <= 0
            or type(value.get("record_count")) is not int
            or value["record_count"] <= 0
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64):
        raise BroadQaExternalDataError(
            f"v7 feasibility input artifact {relative_path} 非法")
    return value


def _iter_jsonl(
        path: Path,
        *,
        artifact: dict[str, object],
        label: str,
        ) -> Iterator[dict[str, object]]:
    """流式读取规范 JSONL，并在 EOF 核对完整物理承诺。"""
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    try:
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                byte_count += len(line)
                record_count += 1
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise BroadQaExternalDataError(
                        f"v7 feasibility {label} JSONL 非法") from error
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v7 feasibility {label} JSONL 非规范")
                yield value
    except OSError as error:
        raise BroadQaExternalDataError(
            f"v7 feasibility {label} 不可读") from error
    if (byte_count != artifact["bytes"]
            or record_count != artifact["record_count"]
            or digest.hexdigest() != artifact["sha256"]):
        raise BroadQaExternalDataError(
            f"v7 feasibility {label} 物理 identity 漂移")


def _validate_inputs(
        *,
        protocol: dict[str, object],
        rule_pack: dict[str, object],
        audit: dict[str, object],
        commitment: dict[str, object],
        ) -> None:
    """核验 TRAIN PASS、VLC 分母先冻结与零 held-out payload read。"""
    audit_summary = audit.get("summary")
    denominator = commitment.get("denominator")
    if (protocol.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"
            or protocol.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or protocol.get("evaluation_or_held_out_payload_read_count") != 0
            or rule_pack.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_RULE_PACK_V1"
            or rule_pack.get("status")
            != "FROZEN_NOT_EVALUATED_NOT_DEPLOYED"
            or rule_pack.get("protocol_manifest_sha256")
            != V5_TRAINING_PROTOCOL_MANIFEST_SHA256
            or audit.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_V1"
            or audit.get("status")
            != "TRAIN_ONLY_COMPLETE_NOT_FORMAL_NOT_DEPLOYED"
            or not isinstance(audit_summary, dict)
            or audit_summary.get("audit_outcome")
            != "FACILITY_PASS_CAPABILITY_PASS"
            or audit_summary.get("facility_failure_count") != 0
            or audit_summary.get("identity_false_change_count") != 0
            or commitment.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1"
            or commitment.get("status")
            != "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE"
            or commitment.get("source_non_manifest_file_read_count") != 0
            or commitment.get("training_source_read_count") != 0
            or not isinstance(denominator, dict)
            or denominator.get("record_count") != 3_656
            or denominator.get("label_blind") != 1
            or protocol.get("production_enabled") != 0
            or rule_pack.get("production_enabled") != 0
            or audit.get("production_enabled") != 0
            or commitment.get("production_enabled") != 0):
        raise BroadQaExternalDataError(
            "v7 feasibility sealed input contract 漂移")


def _derive(
        *,
        protocol_dir: Path,
        rule_pack_dir: Path,
        protocol: dict[str, object],
        rule_pack: dict[str, object],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """流式派生三份 successor feasibility records 与 summary。"""
    artifacts = {
        "observations": _artifact_from_manifest(
            protocol,
            relative_path=_INPUT_FILES["observations"][0],
            role=_INPUT_FILES["observations"][1],
        ),
        **{
            name: _artifact_from_manifest(
                rule_pack,
                relative_path=_INPUT_FILES[name][0],
                role=_INPUT_FILES[name][1],
            )
            for name in ("target_rules", "evidence", "conflicts", "identities")
        },
    }
    variable, variable_summary = derive_variable_structure_projections(
        _iter_jsonl(
            protocol_dir / _INPUT_FILES["observations"][0],
            artifact=artifacts["observations"],
            label="pair observations",
        ))
    identity_inputs = derive_identity_inputs(_iter_jsonl(
        rule_pack_dir / _INPUT_FILES["identities"][0],
        artifact=artifacts["identities"],
        label="identity observations",
    ))
    conflicts = tuple(_iter_jsonl(
        rule_pack_dir / _INPUT_FILES["conflicts"][0],
        artifact=artifacts["conflicts"],
        label="conflict ledger",
    ))
    source_replay, source_summary, conflict_inputs = (
        derive_source_policy_replay_projections(conflicts))
    target_rules = tuple(_iter_jsonl(
        rule_pack_dir / _INPUT_FILES["target_rules"][0],
        artifact=artifacts["target_rules"],
        label="target phrase rules",
    ))
    context, context_summary = derive_context_scoped_local_projections(
        target_rules=target_rules,
        evidence=_iter_jsonl(
            rule_pack_dir / _INPUT_FILES["evidence"][0],
            artifact=artifacts["evidence"],
            label="scoped evidence",
        ),
        identity_inputs=identity_inputs,
        conflict_inputs=conflict_inputs,
    )
    outputs = {
        _OUTPUT_FILES[0][0]: variable,
        _OUTPUT_FILES[1][0]: context,
        _OUTPUT_FILES[2][0]: source_replay,
    }
    return outputs, {
        "context_scoped_local_transfer": context_summary,
        "implementation_order": [
            "VARIABLE_STRUCTURE_TRANSFER",
            "CONTEXT_SCOPED_LOCAL_TRANSFER",
            "SOURCE_POLICY_REPLAY",
        ],
        "overall_outcome": (
            "FEASIBILITY_CONFIRMED_NARROW_OR_PARTIAL_IMPLEMENTATION_REQUIRED"),
        "source_policy_replay": source_summary,
        "variable_structure_transfer": variable_summary,
    }


def _write_jsonl(
        path: Path,
        values: tuple[dict[str, object], ...],
        ) -> None:
    """独占写入一份规范 feasibility JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_output_jsonl(
        path: Path,
        *,
        label: str,
        ) -> tuple[dict[str, object], ...]:
    """读取已发布 output，并拒绝任何非规范行。"""
    try:
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
        values = tuple(json.loads(line) for line in lines)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v7 feasibility output {label} 不可读") from error
    if (not lines or b"".join(lines) != payload
            or any(not isinstance(item, dict) for item in values)
            or b"".join(canonical_json_line(item) for item in values)
            != payload):
        raise BroadQaExternalDataError(
            f"v7 feasibility output {label} 非规范")
    return values


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """构造一个 output 物理文件承诺。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造零 held-out payload read、零执行的 feasibility manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_KIND,
        "candidate_family_formal_run_count": 0,
        "files": files,
        "format_version": 1,
        "held_out_boundary": {
            "consumed_qt_individual_or_derivative_read_count": 0,
            "vlc_commitment_manifest_read_count": 1,
            "vlc_identity_raw_or_translation_read_count": 0,
        },
        "inputs": {
            "v5_rule_pack_manifest_sha256": V5_RULE_PACK_MANIFEST_SHA256,
            "v5_training_protocol_manifest_sha256": (
                V5_TRAINING_PROTOCOL_MANIFEST_SHA256),
            "v6_training_audit_manifest_sha256": (
                V6_TRAINING_AUDIT_MANIFEST_SHA256),
            "v7_evaluation_commitment_manifest_sha256": (
                V7_EVALUATION_COMMITMENT_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_surface_published_in_projection": 0,
    }


def _input_state(
        *,
        protocol_dir: Path,
        rule_pack_dir: Path,
        audit_dir: Path,
        commitment_dir: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ]:
    """读取并核验四个 sealed input manifests。"""
    protocol = _read_manifest(
        protocol_dir,
        expected_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        label="v5 protocol",
    )
    rule_pack = _read_manifest(
        rule_pack_dir,
        expected_sha256=V5_RULE_PACK_MANIFEST_SHA256,
        label="v5 rule pack",
    )
    audit = _read_manifest(
        audit_dir,
        expected_sha256=V6_TRAINING_AUDIT_MANIFEST_SHA256,
        label="v6 training audit",
    )
    commitment = _read_manifest(
        commitment_dir,
        expected_sha256=V7_EVALUATION_COMMITMENT_MANIFEST_SHA256,
        label="v7 commitment",
    )
    _validate_inputs(
        protocol=protocol,
        rule_pack=rule_pack,
        audit=audit,
        commitment=commitment,
    )
    return protocol, rule_pack, audit, commitment


def publish_normalization_recovery_v7_successor_feasibility(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        rule_pack_dir: str | Path,
        training_audit_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布三项 TRAIN-only successor feasibility artifact。"""
    root = _require_k_root(run_root)
    protocol_dir = _within(
        root, training_protocol_dir, label="training protocol")
    pack_dir = _within(root, rule_pack_dir, label="rule pack")
    audit_dir = _within(root, training_audit_dir, label="training audit")
    commitment_dir = _within(
        root, evaluation_commitment_dir, label="evaluation commitment")
    target = _within(root, target_dir, label="target")
    inputs = (protocol_dir, pack_dir, audit_dir, commitment_dir)
    if (any(not path.is_dir() for path in inputs)
            or any(_overlap(target, path) for path in inputs)
            or target.exists()):
        raise BroadQaExternalDataError(
            "v7 successor feasibility input/target path 非法")
    protocol, rule_pack, _audit, _commitment = _input_state(
        protocol_dir=protocol_dir,
        rule_pack_dir=pack_dir,
        audit_dir=audit_dir,
        commitment_dir=commitment_dir,
    )
    outputs, summary = _derive(
        protocol_dir=protocol_dir,
        rule_pack_dir=pack_dir,
        protocol=protocol,
        rule_pack=rule_pack,
    )
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(files=files, summary=summary)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v7_successor_feasibility(
        feasibility_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        rule_pack_dir: str | Path,
        training_audit_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 sealed TRAIN inputs 重派生并严格回读 feasibility artifact。"""
    root = Path(feasibility_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 successor feasibility manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 successor feasibility manifest identity 漂移")
    protocol_dir = Path(training_protocol_dir).resolve()
    pack_dir = Path(rule_pack_dir).resolve()
    audit_dir = Path(training_audit_dir).resolve()
    commitment_dir = Path(evaluation_commitment_dir).resolve()
    protocol, rule_pack, _audit, _commitment = _input_state(
        protocol_dir=protocol_dir,
        rule_pack_dir=pack_dir,
        audit_dir=audit_dir,
        commitment_dir=commitment_dir,
    )
    expected_outputs, summary = _derive(
        protocol_dir=protocol_dir,
        rule_pack_dir=pack_dir,
        protocol=protocol,
        rule_pack=rule_pack,
    )
    stored_outputs = {
        name: _read_output_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    if any(not _strict_equal(stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 successor feasibility records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected_manifest = _manifest(files=files, summary=summary)
    if not _strict_equal(stored, expected_manifest):
        raise BroadQaExternalDataError(
            "v7 successor feasibility manifest 字段漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            stored_outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_KIND",
    "NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_STATUS",
    "publish_normalization_recovery_v7_successor_feasibility",
    "read_normalization_recovery_v7_successor_feasibility",
]
