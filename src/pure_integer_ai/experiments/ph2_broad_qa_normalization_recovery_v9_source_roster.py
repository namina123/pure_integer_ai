"""发布并严格回读 recovery-v9 GIMP 标签盲 source roster。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_roster_records import (
    NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_CENSUS_KIND,
    derive_normalization_recovery_v9_source_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_ARTIFACT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_EVALUATION_SOURCE_ROSTER_V1")
NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_STATUS = (
    "GIMP_METADATA_LICENSE_AND_EIGHT_DOMAIN_PATHS_FROZEN_LOCALE_UNREAD")

_OUTPUT_FILES = (
    ("source-roster.jsonl", "V9_GIMP_EVALUATION_SOURCE_ROSTER"),
    ("source-census.jsonl", "V9_GIMP_EVALUATION_SOURCE_ROSTER_CENSUS"),
)
_LICENSE_EXPECTED = {
    "COPYING": {
        "bytes": 35_151,
        "git_blob_sha1": "e60008693e017bec1b4eb49c84be3898e26fcf2a",
        "sha256": (
            "e79e9c8a0c85d735ff98185918ec94ed7d175efc377012787aebcf3b80f0d90b"),
    },
    "LICENSE": {
        "bytes": 2_823,
        "git_blob_sha1": "eb43828b995bd169913e2032aed4201194562d70",
        "sha256": (
            "0986a9c943105de194155dd1897a58b81f941c9045e361b5477c958cfbaf7b0a"),
    },
}


def _sha256(payload: bytes) -> str:
    """返回 artifact 或文件 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    """按 Git `blob <bytes>\\0<payload>` 规则复算 blob identity。"""
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 位于已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v9 source roster run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入输出仍位于本次 K 盘 run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v9 source roster {label} 越出run root") from error
    return path


def _verify_license_root(root: Path) -> None:
    """核验已读取许可正文与冻结 bytes/blob/SHA 完全一致。"""
    if not root.is_dir():
        raise BroadQaExternalDataError("v9 source roster license root 不存在")
    files = {item.name for item in root.iterdir() if item.is_file()}
    if files != set(_LICENSE_EXPECTED) or any(
            item.is_dir() for item in root.iterdir()):
        raise BroadQaExternalDataError("v9 source roster license inventory 漂移")
    for name, expected in _LICENSE_EXPECTED.items():
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                "v9 source roster license 不可读") from error
        if (len(payload) != expected["bytes"]
                or _git_blob_sha1(payload) != expected["git_blob_sha1"]
                or _sha256(payload) != expected["sha256"]):
            raise BroadQaExternalDataError(
                "v9 source roster license identity 漂移")


def _outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """派生标签盲 roster 与 census 输出。"""
    records, census = derive_normalization_recovery_v9_source_roster()
    census_record = ({
        **census,
        "format_version": 1,
        "record_kind": NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_CENSUS_KIND,
    },)
    return {
        _OUTPUT_FILES[0][0]: records,
        _OUTPUT_FILES[1][0]: census_record,
    }, census


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """读取并核验规范 JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v9 source roster {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v9 source roster {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成一个 roster 输出文件 commitment。"""
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
        census: dict[str, object],
        ) -> dict[str, object]:
    """构造 locale 正文未读的 v9 source roster manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_ARTIFACT_KIND,
        "files": files,
        "format_version": 1,
        "label_or_translation_read_count": 0,
        "license_file_content_read_count": 2,
        "locale_blob_content_read_count": 0,
        "mastery_claimed": 0,
        "predecessor_lineage": {
            "v8_train_source_roster_v2_manifest_sha256": (
                "60c801a6e3b41adf59f06f0ebbfbccc030a5dfdcc1807012ca6bfc5e51e1f68a"),
            "v8_vlc_formal_failure_sha256": (
                "e30a3d2379c00b598d45190cdd2b85ccab97e53195e3eb32558a1938fd6ddbec"),
            "v8_vlc_formal_status": "NE_NO_RECEIPT_NO_RERUN",
            "v8_vlc_source_consumed": 1,
        },
        "production_enabled": 0,
        "runtime_program_published": 0,
        "selection_contract": {
            "all_discovered_complete_domain_pairs_selected": 1,
            "metadata_selected_before_locale_open": 1,
            "prior_source_repository_overlap_count": 0,
            "repository_independent": 1,
        },
        "status": NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_STATUS,
        "summary": census,
        "surface_published": 0,
        "teacher_api_llm_call_count": 0,
    }


def publish_normalization_recovery_v9_source_roster(
        *,
        run_root: str | Path,
        license_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """核验许可正文后不可覆盖发布 v9 标签盲 source roster。"""
    root = _require_k_root(run_root)
    license_path = _within(root, license_root, label="license root")
    target = _within(root, target_dir, label="target")
    if target.exists() or target == license_path:
        raise BroadQaExternalDataError("v9 source roster target 非法")
    _verify_license_root(license_path)
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


def read_normalization_recovery_v9_source_roster(
        source_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生 v9 roster 并拒绝 records/manifest 同步篡改。"""
    root = Path(source_dir).resolve()
    expected_names = {"manifest.json", *[name for name, _role in _OUTPUT_FILES]}
    try:
        physical_names = {item.name for item in root.iterdir()}
    except OSError as error:
        raise BroadQaExternalDataError(
            "v9 source roster artifact 不可读") from error
    if physical_names != expected_names or any(
            item.is_dir() for item in root.iterdir()):
        raise BroadQaExternalDataError("v9 source roster physical inventory 漂移")
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v9 source roster manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v9 source roster manifest identity 漂移")
    expected_outputs, census = _outputs()
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError("v9 source roster records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, census=census)
    if stored != expected:
        raise BroadQaExternalDataError("v9 source roster fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_ARTIFACT_KIND",
    "NORMALIZATION_RECOVERY_V9_SOURCE_ROSTER_STATUS",
    "publish_normalization_recovery_v9_source_roster",
    "read_normalization_recovery_v9_source_roster",
]
