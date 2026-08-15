"""发布 recovery-v8 roster-v2 的aggregate content feasibility。

qBittorrent与Stellarium只继承sealed v1 aggregate；本轮只读取新加入的
KeePassXC固定license/locale blob。pair surface仅在内存存在，不写artifact。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit import (
    read_normalization_recovery_v8_source_content_aggregate,
    read_normalization_recovery_v8_source_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_family_records import (
    derive_normalization_recovery_v8_source_family_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster_v2 import (
    read_normalization_recovery_v8_source_roster_v2,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_AUDIT_V2_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_FEASIBILITY_V2")
NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_RECORD_V2_KIND = (
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_FEASIBILITY_RECORD_V2")
NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_CENSUS_V2_KIND = (
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_FEASIBILITY_CENSUS_V2")
NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_V2_PASS_STATUS = (
    "CONTENT_FEASIBILITY_V2_3_PASS_NOT_SOURCE_PACK")
NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_V2_REJECTED_STATUS = (
    "CONTENT_FEASIBILITY_V2_REPLACEMENT_REJECTED_NOT_SOURCE_PACK")

V8_SOURCE_ROSTER_V2_MANIFEST_SHA256 = (
    "60c801a6e3b41adf59f06f0ebbfbccc030a5dfdcc1807012ca6bfc5e51e1f68a")
V8_SOURCE_CONTENT_V1_MANIFEST_SHA256 = (
    "cdcb7170a49475d9d8ee5c76732b1116d59a4c5cd5eab71b58e063dbf23ea588")

_OUTPUT_FILES = (
    ("source-content-v2.jsonl", "V8_SOURCE_CONTENT_FEASIBILITY_V2"),
    ("source-content-census-v2.jsonl", "V8_SOURCE_CONTENT_CENSUS_V2"),
)
_V2_FAMILIES = (
    "KEEPASSXC_PROJECT",
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
)
_INHERITED_FAMILIES = (
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
)
_CONTENT_PASS = "PASS_NONZERO_ACTIVE_COMMON_PAIR"
_CONTENT_REJECTED = "REJECTED_ZERO_ACTIVE_COMMON_PAIR"


def _sha256(payload: bytes) -> str:
    """返回artifact、record或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v8 source content v2 run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 source content v2 {label} 越出run root") from error
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


def _inherited_record(
        roster: dict[str, object],
        predecessor: dict[str, object],
        ) -> dict[str, object]:
    """把已封存PASS aggregate绑定到未变的v2 family identity。"""
    family = str(roster.get("source_family"))
    if (family not in _INHERITED_FAMILIES
            or predecessor.get("source_family") != family
            or predecessor.get("content_outcome") != _CONTENT_PASS
            or roster.get("content_feasibility_outcome") != _CONTENT_PASS
            or roster.get("selection_status") != "INHERITED_V1_CONTENT_PASS"
            or predecessor.get("license_expression")
            != roster.get("license", {}).get("expression")
            or predecessor.get("source_policy_scope")
            != roster.get("source_policy_scope")
            or predecessor.get("license_file_read_count")
            != len(roster.get("license", {}).get("files", []))
            or predecessor.get("locale_file_read_count")
            != roster.get("locale_file_count")):
        raise BroadQaExternalDataError(
            "v8 source content v2 inherited aggregate 漂移")
    summary = predecessor.get("parser_summary")
    if (not isinstance(summary, dict)
            or summary.get("content_outcome") != _CONTENT_PASS):
        raise BroadQaExternalDataError(
            "v8 source content v2 inherited parser summary 漂移")
    return {
        "content_blob_read_this_revision_count": 0,
        "content_inheritance": "SEALED_V1_AGGREGATE",
        "content_outcome": _CONTENT_PASS,
        "format_version": 1,
        "license_expression": predecessor["license_expression"],
        "license_file_count": predecessor["license_file_read_count"],
        "locale_file_count": predecessor["locale_file_read_count"],
        "pair_surface_published": 0,
        "parser_summary": summary,
        "predecessor_content_record_sha256": _sha256(
            canonical_json_line(predecessor)),
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_RECORD_V2_KIND,
        "source_family": family,
        "source_pack_published": 0,
        "source_policy_scope": predecessor["source_policy_scope"],
        "source_roster_revision": 2,
        "transient_pair_count": predecessor["transient_pair_count"],
    }


def _replacement_record(
        roster: dict[str, object],
        source_root: Path,
        ) -> dict[str, object]:
    """逐blob核验并派生KeePassXC aggregate，不发布pair surface。"""
    if (roster.get("source_family") != "KEEPASSXC_PROJECT"
            or roster.get("selection_status")
            != "SELECTED_V2_REPLACEMENT_TREE_LICENSE_PATH_FROZEN"
            or roster.get("content_feasibility_outcome")
            != "NOT_READ_ROSTER_V2_REPLACEMENT"
            or roster.get("locale_blob_content_read_count") != 0):
        raise BroadQaExternalDataError(
            "v8 source content v2 replacement roster 漂移")
    payloads = read_normalization_recovery_v8_source_payloads(
        roster, source_root)
    locale_paths = tuple(str(item["relative_path"])
                         for item in roster["locale_files"])
    _files, pairs, summary = (
        derive_normalization_recovery_v8_source_family_records(
            roster, payloads))
    return {
        "content_blob_read_this_revision_count": len(payloads),
        "content_inheritance": "REPLACEMENT_BLOB_DERIVED_V2",
        "content_outcome": summary["content_outcome"],
        "format_version": 1,
        "license_expression": roster["license"]["expression"],
        "license_file_count": len(roster["license"]["files"]),
        "locale_file_count": len(locale_paths),
        "pair_surface_published": 0,
        "parser_summary": summary,
        "predecessor_content_record_sha256": "",
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_RECORD_V2_KIND,
        "source_family": "KEEPASSXC_PROJECT",
        "source_pack_published": 0,
        "source_policy_scope": roster["source_policy_scope"],
        "source_roster_revision": 2,
        "transient_pair_count": len(pairs),
    }


def _derive(
        roster: tuple[dict[str, object], ...],
        predecessor: tuple[dict[str, object], ...],
        keepassxc_source_root: Path,
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """组合两家sealed aggregate与一家replacement content结果。"""
    by_roster = {str(item.get("source_family")): item for item in roster}
    by_predecessor = {
        str(item.get("source_family")): item for item in predecessor}
    if (set(by_roster) != set(_V2_FAMILIES)
            or set(by_predecessor) != {
                "BITCOIN_CORE_PROJECT",
                "QBITTORRENT_PROJECT",
                "STELLARIUM_PROJECT",
            }
            or by_predecessor["BITCOIN_CORE_PROJECT"].get("content_outcome")
            != _CONTENT_REJECTED
            or by_predecessor["BITCOIN_CORE_PROJECT"].get(
                "transient_pair_count") != 0):
        raise BroadQaExternalDataError(
            "v8 source content v2 predecessor inventory 漂移")
    records = [
        _inherited_record(by_roster[family], by_predecessor[family])
        for family in _INHERITED_FAMILIES
    ]
    records.append(_replacement_record(
        by_roster["KEEPASSXC_PROJECT"], keepassxc_source_root))
    records.sort(key=lambda item: str(item["source_family"]))
    summary = {
        "content_pass_count": sum(
            item["content_outcome"] == _CONTENT_PASS for item in records),
        "content_rejected_count": sum(
            item["content_outcome"] == _CONTENT_REJECTED for item in records),
        "inherited_content_family_count": 2,
        "pair_surface_published": 0,
        "predecessor_content_file_read_count": 2,
        "replacement_blob_read_count": int(
            records[0]["content_blob_read_this_revision_count"]),
        "replacement_content_family_count": 1,
        "selected_license_file_count": sum(
            len(item["license"]["files"]) for item in roster),
        "selected_locale_file_count": sum(
            int(item["locale_file_count"]) for item in roster),
        "selected_source_family_count": len(records),
        "source_pack_published_count": 0,
        "structure_equal_count": sum(
            int(item["parser_summary"]["structure_equal_count"])
            for item in records),
        "transient_pair_count": sum(
            int(item["transient_pair_count"]) for item in records),
        "v8_training_eligible_pair_count": sum(
            int(item["parser_summary"]["v8_training_eligible_pair_count"])
            for item in records),
    }
    census = ({
        **summary,
        "format_version": 1,
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_CENSUS_V2_KIND,
    },)
    return {
        _OUTPUT_FILES[0][0]: tuple(records),
        _OUTPUT_FILES[1][0]: census,
    }, summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格读取本artifact规范JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v8 source content v2 {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v8 source content v2 {label} 不可读") from error
    return tuple(values)


def read_normalization_recovery_v8_source_content_aggregate_v2(
        audit_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """只按sealed commitments回读v2 aggregate，不重开source blob。"""
    root = Path(audit_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 source content v2 aggregate manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_AUDIT_V2_KIND):
        raise BroadQaExternalDataError(
            "v8 source content v2 aggregate manifest identity 漂移")
    files = {str(item.get("relative_path")): item
             for item in stored.get("files", [])
             if isinstance(item, dict)}
    if set(files) != {name for name, _role in _OUTPUT_FILES}:
        raise BroadQaExternalDataError(
            "v8 source content v2 aggregate file inventory 漂移")
    outputs = {}
    for name, role in _OUTPUT_FILES:
        values = _read_jsonl(root / name, label=role)
        if files[name] != _artifact(
                root / name, role=role, count=len(values)):
            raise BroadQaExternalDataError(
                "v8 source content v2 aggregate file identity 漂移")
        outputs[name] = values
    census = outputs["source-content-census-v2.jsonl"]
    if (len(outputs["source-content-v2.jsonl"]) != 3
            or len(census) != 1
            or stored.get("summary") != {
                key: value for key, value in census[0].items()
                if key not in {"format_version", "record_kind"}
            }):
        raise BroadQaExternalDataError(
            "v8 source content v2 aggregate census 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, outputs


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成aggregate输出文件commitment。"""
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
    """构造三家均PASS或replacement拒绝的v2 aggregate manifest。"""
    status = (
        NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_V2_PASS_STATUS
        if summary["content_pass_count"] == 3
        and summary["content_rejected_count"] == 0
        else NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_V2_REJECTED_STATUS)
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_AUDIT_V2_KIND,
        "consumed_or_final_source_read_count": 0,
        "files": files,
        "format_version": 1,
        "inputs": {
            "v1_source_content_manifest_sha256": (
                V8_SOURCE_CONTENT_V1_MANIFEST_SHA256),
            "v2_source_roster_manifest_sha256": (
                V8_SOURCE_ROSTER_V2_MANIFEST_SHA256),
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": status,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
    }


def _state(
        *,
        v2_roster_dir: Path,
        v1_roster_dir: Path,
        v1_content_audit_dir: Path,
        keepassxc_source_root: Path,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            Path,
        ]:
    """严格回读两个sealed predecessor并核对replacement source root。"""
    _roster_manifest, roster_outputs = (
        read_normalization_recovery_v8_source_roster_v2(
            v2_roster_dir,
            v1_roster_dir=v1_roster_dir,
            content_audit_dir=v1_content_audit_dir,
            expected_manifest_sha256=V8_SOURCE_ROSTER_V2_MANIFEST_SHA256,
        ))
    _content_manifest, content_outputs = (
        read_normalization_recovery_v8_source_content_aggregate(
            v1_content_audit_dir,
            expected_manifest_sha256=V8_SOURCE_CONTENT_V1_MANIFEST_SHA256,
        ))
    if not keepassxc_source_root.is_dir():
        raise BroadQaExternalDataError(
            "v8 source content v2 KeePassXC source root 不存在")
    return (
        roster_outputs["source-roster-v2.jsonl"],
        content_outputs["source-content.jsonl"],
        keepassxc_source_root,
    )


def publish_normalization_recovery_v8_source_content_audit_v2(
        *,
        run_root: str | Path,
        v2_roster_dir: str | Path,
        v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path,
        keepassxc_source_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布roster-v2的aggregate content feasibility。"""
    root = _require_k_root(run_root)
    paths = tuple(
        _within(root, value, label=str(index))
        for index, value in enumerate((
            v2_roster_dir,
            v1_roster_dir,
            v1_content_audit_dir,
            keepassxc_source_root,
            target_dir,
        )))
    v2_roster, v1_roster, v1_content, keepassxc, target = paths
    if (target.exists()
            or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v8 source content v2 input/target path 非法")
    state = _state(
        v2_roster_dir=v2_roster,
        v1_roster_dir=v1_roster,
        v1_content_audit_dir=v1_content,
        keepassxc_source_root=keepassxc,
    )
    outputs, summary = _derive(*state)
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


def read_normalization_recovery_v8_source_content_audit_v2(
        audit_dir: str | Path,
        *,
        v2_roster_dir: str | Path,
        v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path,
        keepassxc_source_root: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生v2 aggregate，并拒绝records/manifest同步篡改。"""
    root = Path(audit_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 source content v2 manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v8 source content v2 manifest identity 漂移")
    state = _state(
        v2_roster_dir=Path(v2_roster_dir).resolve(),
        v1_roster_dir=Path(v1_roster_dir).resolve(),
        v1_content_audit_dir=Path(v1_content_audit_dir).resolve(),
        keepassxc_source_root=Path(keepassxc_source_root).resolve(),
    )
    expected_outputs, summary = _derive(*state)
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError(
            "v8 source content v2 records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary)
    if stored != expected:
        raise BroadQaExternalDataError(
            "v8 source content v2 fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_AUDIT_V2_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_V2_PASS_STATUS",
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_V2_REJECTED_STATUS",
    "V8_SOURCE_CONTENT_V1_MANIFEST_SHA256",
    "V8_SOURCE_ROSTER_V2_MANIFEST_SHA256",
    "publish_normalization_recovery_v8_source_content_audit_v2",
    "read_normalization_recovery_v8_source_content_aggregate_v2",
    "read_normalization_recovery_v8_source_content_audit_v2",
]
