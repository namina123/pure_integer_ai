"""发布并严格回读 recovery-v10 新 TRAIN 来源内容可行性。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_content_records import (
    V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES,
    V10_SOURCE_EXPANSION_ROSTER_MANIFEST_SHA256,
    derive_normalization_recovery_v10_source_expansion_content,
    read_normalization_recovery_v10_source_expansion_content_state,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster import (
    V8_OBSERVATION_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


V10_SOURCE_EXPANSION_CONTENT_ARTIFACT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_CONTENT_V1")
V10_SOURCE_EXPANSION_CONTENT_STATUS = (
    "TWO_TRAIN_SOURCE_CONTENT_AND_PREDECESSOR_OVERLAP_PASS_NOT_SOURCE_PACK")


def _sha256(payload: bytes) -> str:
    """返回 artifact 或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 位于已存在的 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v10 source content run root 必须在 K 盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入输出仍位于本次 K 盘 run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v10 source content {label} 越出 run root") from error
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个已解析路径是否互为祖先。"""
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


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
                        f"v10 source content {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v10 source content {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成一个 aggregate 输出文件 commitment。"""
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
    """构造 aggregate-only 内容可行性 manifest。"""
    return {
        "artifact_kind": V10_SOURCE_EXPANSION_CONTENT_ARTIFACT_KIND,
        "files": files,
        "formal_or_evaluation_payload_read_count": 0,
        "format_version": 1,
        "inputs": {
            "v10_source_expansion_roster_manifest_sha256": (
                V10_SOURCE_EXPANSION_ROSTER_MANIFEST_SHA256),
            "v8_observation_pack_manifest_sha256": (
                V8_OBSERVATION_PACK_MANIFEST_SHA256),
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "source_pack_published_count": 0,
        "status": V10_SOURCE_EXPANSION_CONTENT_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_rule_published_count": 0,
    }


def _state(
        *,
        roster_dir: Path,
        predecessor_observation_dir: Path,
        mixxx_source_root: Path,
        mumble_source_root: Path,
        ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """向 records 层转发固定两家来源路径。"""
    return read_normalization_recovery_v10_source_expansion_content_state(
        roster_dir=roster_dir,
        predecessor_observation_dir=predecessor_observation_dir,
        source_roots={
            "MIXXX_PROJECT": mixxx_source_root,
            "MUMBLE_PROJECT": mumble_source_root,
        },
    )


def publish_normalization_recovery_v10_source_expansion_content(
        *,
        run_root: str | Path,
        roster_dir: str | Path,
        predecessor_observation_dir: str | Path,
        mixxx_source_root: str | Path,
        mumble_source_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布两家新 TRAIN 来源的 aggregate 内容可行性。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=str(index)) for index, value in
                  enumerate((
                      roster_dir,
                      predecessor_observation_dir,
                      mixxx_source_root,
                      mumble_source_root,
                      target_dir,
                  )))
    roster, predecessor, mixxx, mumble, target = paths
    if (target.exists()
            or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v10 source content input/target path 非法")
    outputs, summary = derive_normalization_recovery_v10_source_expansion_content(
        *_state(
            roster_dir=roster,
            predecessor_observation_dir=predecessor,
            mixxx_source_root=mixxx,
            mumble_source_root=mumble,
        ))
    target.mkdir()
    files = []
    for name, role in V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(files=files, summary=summary)
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v10_source_expansion_content(
        source_dir: str | Path,
        *,
        roster_dir: str | Path,
        predecessor_observation_dir: str | Path,
        mixxx_source_root: str | Path,
        mumble_source_root: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生内容可行性并拒绝 records/manifest 同步篡改。"""
    root = Path(source_dir).resolve()
    expected_names = {
        "manifest.json",
        *[name for name, _ in V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES],
    }
    try:
        physical = tuple(root.iterdir())
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 source content artifact 不可读") from error
    if ({item.name for item in physical} != expected_names
            or any(item.is_dir() for item in physical)):
        raise BroadQaExternalDataError(
            "v10 source content physical inventory 漂移")
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 source content manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v10 source content manifest identity 漂移")
    expected_outputs, summary = (
        derive_normalization_recovery_v10_source_expansion_content(
            *_state(
                roster_dir=Path(roster_dir).resolve(),
                predecessor_observation_dir=Path(
                    predecessor_observation_dir).resolve(),
                mixxx_source_root=Path(mixxx_source_root).resolve(),
                mumble_source_root=Path(mumble_source_root).resolve(),
            )))
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError(
            "v10 source content records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary)
    if stored != expected:
        raise BroadQaExternalDataError(
            "v10 source content fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "V10_SOURCE_EXPANSION_CONTENT_ARTIFACT_KIND",
    "V10_SOURCE_EXPANSION_CONTENT_STATUS",
    "publish_normalization_recovery_v10_source_expansion_content",
    "read_normalization_recovery_v10_source_expansion_content",
]
