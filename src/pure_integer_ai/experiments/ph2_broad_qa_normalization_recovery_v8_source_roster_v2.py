"""发布并严格回读 recovery-v8 TRAIN source roster v2。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit import (
    read_normalization_recovery_v8_source_content_aggregate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster import (
    read_normalization_recovery_v8_source_roster,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster_v2_records import (
    NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_CENSUS_KIND,
    derive_normalization_recovery_v8_source_roster_v2,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_ARTIFACT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_TRAIN_SOURCE_ROSTER_V2")
NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_STATUS = (
    "V2_REPLACES_ZERO_ACTIVE_BITCOIN_KEEPASSXC_LOCALE_UNREAD")

V8_SOURCE_ROSTER_V1_MANIFEST_SHA256 = (
    "0fcc981b23f5d1f7c052f80e37d07b27dd7fa61c5db7ec036884e25d5493b9fc")
V8_SOURCE_CONTENT_AUDIT_MANIFEST_SHA256 = (
    "cdcb7170a49475d9d8ee5c76732b1116d59a4c5cd5eab71b58e063dbf23ea588")

_OUTPUT_FILES = (
    ("source-roster-v2.jsonl", "V8_TRAIN_SOURCE_ROSTER_V2"),
    ("source-census-v2.jsonl", "V8_TRAIN_SOURCE_ROSTER_CENSUS_V2"),
)


def _sha256(payload: bytes) -> str:
    """返回artifact或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v8 roster v2 run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 roster v2 {label} 越出run root") from error
    return path


def _content_records(
        audit_dir: Path,
        ) -> tuple[dict[str, object], ...]:
    """只读已封存content aggregate，不重开31份source blob。"""
    manifest, outputs = (
        read_normalization_recovery_v8_source_content_aggregate(
            audit_dir,
            expected_manifest_sha256=(
                V8_SOURCE_CONTENT_AUDIT_MANIFEST_SHA256),
        ))
    if (manifest.get("summary", {}).get("content_pass_count") != 2
            or manifest.get("summary", {}).get("content_rejected_count") != 1):
        raise BroadQaExternalDataError(
            "v8 roster v2 content manifest fields 漂移")
    return outputs["source-content.jsonl"]


def _inputs(
        *,
        roster_dir: Path,
        content_audit_dir: Path,
        ) -> tuple[
            tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """严格回读v1 roster与compact content aggregate。"""
    _manifest_value, outputs = read_normalization_recovery_v8_source_roster(
        roster_dir,
        expected_manifest_sha256=V8_SOURCE_ROSTER_V1_MANIFEST_SHA256,
    )
    return outputs["source-roster.jsonl"], _content_records(
        content_audit_dir)


def _outputs(
        *,
        roster_dir: Path,
        content_audit_dir: Path,
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """由两个sealed predecessor重派生v2 records/census。"""
    v1, content = _inputs(
        roster_dir=roster_dir, content_audit_dir=content_audit_dir)
    records, summary = derive_normalization_recovery_v8_source_roster_v2(
        v1_roster=v1, content_records=content)
    census = ({
        **summary,
        "format_version": 1,
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_CENSUS_KIND,
    },)
    return {
        _OUTPUT_FILES[0][0]: records,
        _OUTPUT_FILES[1][0]: census,
    }, summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _stored_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """读取本artifact规范JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v8 roster v2 {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v8 roster v2 {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成v2输出文件commitment。"""
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
    """构造KeePassXC locale unread的v2 roster manifest。"""
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_ARTIFACT_KIND),
        "files": files,
        "format_version": 1,
        "inputs": {
            "v1_source_roster_manifest_sha256": (
                V8_SOURCE_ROSTER_V1_MANIFEST_SHA256),
            "v1_source_content_audit_manifest_sha256": (
                V8_SOURCE_CONTENT_AUDIT_MANIFEST_SHA256),
        },
        "keepassxc_locale_blob_content_read_count": 0,
        "mastery_claimed": 0,
        "predecessor_content_non_manifest_read_count": 2,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
    }


def publish_normalization_recovery_v8_source_roster_v2(
        *,
        run_root: str | Path,
        v1_roster_dir: str | Path,
        content_audit_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布v8 source roster v2。"""
    root = _require_k_root(run_root)
    roster = _within(root, v1_roster_dir, label="v1 roster")
    content = _within(root, content_audit_dir, label="content audit")
    target = _within(root, target_dir, label="target")
    if (not roster.is_dir() or not content.is_dir() or target.exists()
            or target == roster or target == content):
        raise BroadQaExternalDataError("v8 roster v2 input/target path 非法")
    outputs, summary = _outputs(
        roster_dir=roster, content_audit_dir=content)
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(files=files, summary=summary)
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v8_source_roster_v2(
        source_dir: str | Path,
        *,
        v1_roster_dir: str | Path,
        content_audit_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生v2 roster并拒绝records/manifest同步篡改。"""
    root = Path(source_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 roster v2 manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v8 roster v2 manifest identity 漂移")
    expected_outputs, summary = _outputs(
        roster_dir=Path(v1_roster_dir).resolve(),
        content_audit_dir=Path(content_audit_dir).resolve(),
    )
    stored_outputs = {
        name: _stored_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError("v8 roster v2 records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary)
    if stored != expected:
        raise BroadQaExternalDataError("v8 roster v2 fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_ARTIFACT_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_V2_STATUS",
    "V8_SOURCE_CONTENT_AUDIT_MANIFEST_SHA256",
    "V8_SOURCE_ROSTER_V1_MANIFEST_SHA256",
    "publish_normalization_recovery_v8_source_roster_v2",
    "read_normalization_recovery_v8_source_roster_v2",
]
