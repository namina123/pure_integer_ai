"""发布并严格回读 recovery-v8 TRAIN source roster commitment。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster_records import (
    NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_CENSUS_KIND,
    derive_normalization_recovery_v8_source_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_ARTIFACT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_TRAIN_SOURCE_ROSTER_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_STATUS = (
    "NEW_TRAIN_SOURCE_TREE_LICENSE_PATH_FROZEN_LOCALE_UNREAD")

_OUTPUT_FILES = (
    ("source-roster.jsonl", "V8_TRAIN_SOURCE_ROSTER"),
    ("source-census.jsonl", "V8_TRAIN_SOURCE_ROSTER_CENSUS"),
)

_QT_SOURCE_MANIFEST_SHA256 = (
    "8e31bbd0f00ec643f725b8a6b09d4d5d3e189805f71b3c69905b4914aa7a1340")
_AUDACITY_AGGREGATE_SHA256 = (
    "7e4618f7652d751da6e402ae75d71da400c1ece386b72cb2adea513e643050bd")
_AUDACITY_SOURCE_MANIFEST_SHA256 = (
    "64cfb20d34aa4bb4597e84fd325dfcd8b86659602010a07bc9b99e94882e86d4")
_VLC_COMMITMENT_MANIFEST_SHA256 = (
    "a406598a134a0390e101419518f81bf9877a415e8b4b060c4982be0e1844a8d4")


def _sha256(payload: bytes) -> str:
    """返回 artifact 或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 位于已存在的 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v8 source roster run root 必须在 K 盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制目标仍位于本次 K 盘 run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 source roster {label} 越出 run root") from error
    return path


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格读取规范 JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v8 source roster {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v8 source roster {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成一个输出文件 commitment。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """重派生 roster 与无表面 census。"""
    roster, census = derive_normalization_recovery_v8_source_roster()
    census_record = {
        **census,
        "format_version": 1,
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_CENSUS_KIND,
    }
    return {
        _OUTPUT_FILES[0][0]: roster,
        _OUTPUT_FILES[1][0]: (census_record,),
    }, census


def _manifest(
        *,
        files: list[dict[str, object]],
        census: dict[str, object],
        ) -> dict[str, object]:
    """构造只承诺 tree/license/path 的 v8 roster manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_ARTIFACT_KIND,
        "consumed_source_exclusion": {
            "audacity_aggregate_sha256": _AUDACITY_AGGREGATE_SHA256,
            "audacity_individual_or_translation_read_count": 0,
            "audacity_source_manifest_sha256": (
                _AUDACITY_SOURCE_MANIFEST_SHA256),
            "firefox_evaluation_or_reserve_read_count": 0,
            "qt_individual_or_derivative_read_count": 0,
            "qt_source_manifest_sha256": _QT_SOURCE_MANIFEST_SHA256,
        },
        "files": files,
        "format_version": 1,
        "locale_blob_content_read_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_STATUS,
        "summary": census,
        "teacher_api_llm_call_count": 0,
        "vlc_final_commitment": {
            "individual_identity_raw_or_translation_read_count": 0,
            "manifest_sha256": _VLC_COMMITMENT_MANIFEST_SHA256,
        },
    }


def publish_normalization_recovery_v8_source_roster(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 v8 TRAIN source roster commitment。"""
    root = _require_k_root(run_root)
    target = _within(root, target_dir, label="target")
    if target.exists():
        raise BroadQaExternalDataError("v8 source roster target 已存在")
    outputs, census = _outputs()
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(files=files, census=census)
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v8_source_roster(
        source_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生 roster 并拒绝 records/manifest 同步篡改。"""
    root = Path(source_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 source roster manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v8 source roster manifest identity 漂移")
    expected_outputs, census = _outputs()
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError("v8 source roster records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, census=census)
    if stored != expected:
        raise BroadQaExternalDataError("v8 source roster fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_ARTIFACT_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_ROSTER_STATUS",
    "publish_normalization_recovery_v8_source_roster",
    "read_normalization_recovery_v8_source_roster",
]
