"""发布并严格回读 recovery-v10 独立来源扩充名册。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster_records import (
    V10_SOURCE_EXPANSION_CENSUS_RECORD_KIND,
    derive_normalization_recovery_v10_source_expansion_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


V10_SOURCE_EXPANSION_ROSTER_ARTIFACT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_ROSTER_V1")
V10_SOURCE_EXPANSION_ROSTER_STATUS = (
    "TWO_TRAIN_CANDIDATES_FROZEN_LOCALE_UNREAD_FORMAL_RESERVE_PRESERVED")
V10_LOCAL_HYPOTHESIS_FEASIBILITY_MANIFEST_SHA256 = (
    "9f655ee0eca8a94f7bc0ce0ec92292ef89e21a21a3667f0465013f9d04a10f2f")
V8_OBSERVATION_PACK_MANIFEST_SHA256 = (
    "99ab49c0605be76b2206746330969a071d8b6deed83f3aa454610a99546ddf65")

_OUTPUT_FILES = (
    ("source-candidates.jsonl", "V10_SOURCE_EXPANSION_CANDIDATES"),
    ("source-exclusions.jsonl", "V10_SOURCE_EXPANSION_EXCLUSIONS"),
    ("source-census.jsonl", "V10_SOURCE_EXPANSION_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回 artifact 或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 位于已存在的 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v10 source roster run root 必须在 K 盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制目标仍位于本次 K 盘 run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v10 source roster {label} 越出 run root") from error
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
                        f"v10 source roster {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v10 source roster {label} 不可读") from error
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
    """重派生候选、排除账与 census。"""
    candidates, exclusions, census = (
        derive_normalization_recovery_v10_source_expansion_roster())
    census_record = {
        **census,
        "format_version": 1,
        "record_kind": V10_SOURCE_EXPANSION_CENSUS_RECORD_KIND,
    }
    return {
        _OUTPUT_FILES[0][0]: candidates,
        _OUTPUT_FILES[1][0]: exclusions,
        _OUTPUT_FILES[2][0]: (census_record,),
    }, census


def _manifest(
        *,
        files: list[dict[str, object]],
        census: dict[str, object],
        ) -> dict[str, object]:
    """构造只承诺来源元数据与决策账的 manifest。"""
    return {
        "artifact_kind": V10_SOURCE_EXPANSION_ROSTER_ARTIFACT_KIND,
        "files": files,
        "format_version": 1,
        "formal_or_evaluation_payload_read_count": 0,
        "inputs": {
            "v10_local_hypothesis_feasibility_manifest_sha256": (
                V10_LOCAL_HYPOTHESIS_FEASIBILITY_MANIFEST_SHA256),
            "v8_observation_pack_manifest_sha256": (
                V8_OBSERVATION_PACK_MANIFEST_SHA256),
        },
        "locale_blob_content_read_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "source_pack_published_count": 0,
        "status": V10_SOURCE_EXPANSION_ROSTER_STATUS,
        "summary": census,
        "teacher_api_llm_call_count": 0,
    }


def publish_normalization_recovery_v10_source_expansion_roster(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 v10 独立来源扩充名册。"""
    root = _require_k_root(run_root)
    target = _within(root, target_dir, label="target")
    if target.exists():
        raise BroadQaExternalDataError("v10 source roster target 已存在")
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


def read_normalization_recovery_v10_source_expansion_roster(
        source_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生名册并拒绝 records/manifest 同步篡改。"""
    root = Path(source_dir).resolve()
    expected_names = {"manifest.json", *[name for name, _ in _OUTPUT_FILES]}
    try:
        physical = tuple(root.iterdir())
    except OSError as error:
        raise BroadQaExternalDataError("v10 source roster artifact 不可读") from error
    if ({item.name for item in physical} != expected_names
            or any(item.is_dir() for item in physical)):
        raise BroadQaExternalDataError(
            "v10 source roster physical inventory 漂移")
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v10 source roster manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v10 source roster manifest identity 漂移")
    expected_outputs, census = _outputs()
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError("v10 source roster records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, census=census)
    if stored != expected:
        raise BroadQaExternalDataError("v10 source roster fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "V10_LOCAL_HYPOTHESIS_FEASIBILITY_MANIFEST_SHA256",
    "V10_SOURCE_EXPANSION_ROSTER_ARTIFACT_KIND",
    "V10_SOURCE_EXPANSION_ROSTER_STATUS",
    "V8_OBSERVATION_PACK_MANIFEST_SHA256",
    "publish_normalization_recovery_v10_source_expansion_roster",
    "read_normalization_recovery_v10_source_expansion_roster",
]
