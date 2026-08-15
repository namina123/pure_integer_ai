"""发布 recovery-v8 新 TRAIN source 的aggregate content feasibility。

audit严格回读v1 roster，逐blob核对license/locale文件，再调用共享Qt TS或
gettext parser。输出只含aggregate census；pair surface只存在于本次内存，
不写artifact，也不形成TRAIN protocol、candidate或runtime。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
    sha256_hex,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster import (
    read_normalization_recovery_v8_source_roster,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_family_records import (
    derive_normalization_recovery_v8_source_family_records,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_FEASIBILITY_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_FEASIBILITY_RECORD_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_FEASIBILITY_CENSUS_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_STATUS = (
    "CONTENT_FEASIBILITY_2_PASS_1_REJECTED_NOT_SOURCE_PACK")

V8_SOURCE_ROSTER_MANIFEST_SHA256 = (
    "0fcc981b23f5d1f7c052f80e37d07b27dd7fa61c5db7ec036884e25d5493b9fc")

_OUTPUT_FILES = (
    ("source-content.jsonl", "V8_SOURCE_CONTENT_FEASIBILITY"),
    ("source-content-census.jsonl", "V8_SOURCE_CONTENT_CENSUS"),
)
_FAMILIES = (
    "BITCOIN_CORE_PROJECT",
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
)


def _sha256(payload: bytes) -> str:
    """返回artifact或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在的K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v8 source content run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 source content {label} 越出run root") from error
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


def _read_payloads(
        record: dict[str, object],
        root: Path,
        ) -> dict[str, bytes]:
    """按roster逐blob读取，并核对bytes/Git SHA-1/primary license SHA。"""
    license_value = record.get("license")
    locale_files = record.get("locale_files")
    if (not isinstance(license_value, dict)
            or not isinstance(license_value.get("files"), list)
            or not isinstance(locale_files, list)):
        raise BroadQaExternalDataError(
            "v8 source content roster file inventory 漂移")
    items = tuple(license_value["files"] + locale_files)
    payloads = {}
    for item in items:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v8 source content blob record 非对象")
        relative = item.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise BroadQaExternalDataError(
                "v8 source content relative path 漂移")
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise BroadQaExternalDataError(
                "v8 source content blob path 越界") from error
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                "v8 source content blob 不可读") from error
        if (len(payload) != item.get("bytes")
                or git_blob_sha1(payload) != item.get("git_blob_sha1")):
            raise BroadQaExternalDataError(
                "v8 source content blob identity 漂移")
        payloads[relative] = payload
    primary = license_value["files"][0]
    primary_payload = payloads[str(primary["relative_path"])]
    if (len(primary_payload) != license_value.get("primary_bytes")
            or sha256_hex(primary_payload)
            != license_value.get("primary_sha256")):
        raise BroadQaExternalDataError(
            "v8 source content primary license 漂移")
    return payloads


def read_normalization_recovery_v8_source_payloads(
        record: dict[str, object],
        source_root: str | Path,
        ) -> dict[str, bytes]:
    """按冻结roster身份读取单个source family的全部blob。"""
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise BroadQaExternalDataError(
            "v8 source content source root 不存在")
    return _read_payloads(record, root)


def _derive_family(
        record: dict[str, object],
        payloads: dict[str, bytes],
        ) -> dict[str, object]:
    """按family调用共享parser，并只返回aggregate content record。"""
    family = str(record["source_family"])
    license_expression = str(record["license"]["expression"])
    locale_paths = tuple(str(item["relative_path"])
                         for item in record["locale_files"])
    _files, pairs, summary = (
        derive_normalization_recovery_v8_source_family_records(
            record, payloads))
    return {
        "content_outcome": summary["content_outcome"],
        "format_version": 1,
        "license_expression": license_expression,
        "license_file_read_count": len(record["license"]["files"]),
        "locale_file_read_count": len(locale_paths),
        "pair_surface_published": 0,
        "parser_summary": summary,
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_RECORD_KIND,
        "source_family": family,
        "source_pack_published": 0,
        "source_policy_scope": record["source_policy_scope"],
        "transient_pair_count": len(pairs),
    }


def _derive(
        roster: tuple[dict[str, object], ...],
        roots: dict[str, Path],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """逐family核对source blobs并形成aggregate-only outputs。"""
    by_family = {str(item["source_family"]): item for item in roster}
    if (set(by_family) != set(_FAMILIES)
            or set(roots) != set(_FAMILIES)):
        raise BroadQaExternalDataError(
            "v8 source content family inventory 漂移")
    records = []
    census = Counter()
    for family in _FAMILIES:
        payloads = _read_payloads(by_family[family], roots[family])
        record = _derive_family(by_family[family], payloads)
        records.append(record)
        summary = record["parser_summary"]
        census["content_pass_count"] += int(
            record["content_outcome"] == "PASS_NONZERO_ACTIVE_COMMON_PAIR")
        census["content_rejected_count"] += int(
            record["content_outcome"] == "REJECTED_ZERO_ACTIVE_COMMON_PAIR")
        census["license_file_read_count"] += int(
            record["license_file_read_count"])
        census["locale_file_read_count"] += int(
            record["locale_file_read_count"])
        census["transient_pair_count"] += int(record["transient_pair_count"])
        census["structure_equal_count"] += int(
            summary["structure_equal_count"])
        census["v8_training_eligible_pair_count"] += int(
            summary["v8_training_eligible_pair_count"])
    records.sort(key=lambda item: str(item["source_family"]))
    summary = {
        "content_pass_count": census["content_pass_count"],
        "content_rejected_count": census["content_rejected_count"],
        "license_file_read_count": census["license_file_read_count"],
        "locale_file_read_count": census["locale_file_read_count"],
        "pair_surface_published": 0,
        "selected_source_family_count": len(records),
        "source_pack_published_count": 0,
        "structure_equal_count": census["structure_equal_count"],
        "transient_pair_count": census["transient_pair_count"],
        "v8_training_eligible_pair_count": census[
            "v8_training_eligible_pair_count"],
    }
    census_record = {
        **summary,
        "format_version": 1,
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_CENSUS_KIND,
    }
    return {
        _OUTPUT_FILES[0][0]: tuple(records),
        _OUTPUT_FILES[1][0]: (census_record,),
    }, summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格读取规范JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v8 source content {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v8 source content {label} 不可读") from error
    return tuple(values)


def read_normalization_recovery_v8_source_content_aggregate(
        audit_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """只按sealed commitments回读v1 aggregate，不重开source blob。"""
    root = Path(audit_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 source content aggregate manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_AUDIT_KIND):
        raise BroadQaExternalDataError(
            "v8 source content aggregate manifest identity 漂移")
    files = {str(item.get("relative_path")): item
             for item in stored.get("files", [])
             if isinstance(item, dict)}
    if set(files) != {name for name, _role in _OUTPUT_FILES}:
        raise BroadQaExternalDataError(
            "v8 source content aggregate file inventory 漂移")
    outputs = {}
    for name, role in _OUTPUT_FILES:
        values = _read_jsonl(root / name, label=role)
        if files[name] != _artifact(
                root / name, role=role, count=len(values)):
            raise BroadQaExternalDataError(
                "v8 source content aggregate file identity 漂移")
        outputs[name] = values
    census = outputs["source-content-census.jsonl"]
    if (len(outputs["source-content.jsonl"]) != 3
            or len(census) != 1
            or stored.get("summary") != {
                key: value for key, value in census[0].items()
                if key not in {"format_version", "record_kind"}
            }):
        raise BroadQaExternalDataError(
            "v8 source content aggregate census 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, outputs


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成一个aggregate输出文件commitment。"""
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
    """构造只发布aggregate content结果的manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_AUDIT_KIND,
        "consumed_or_final_source_read_count": 0,
        "files": files,
        "format_version": 1,
        "inputs": {
            "v8_source_roster_manifest_sha256": (
                V8_SOURCE_ROSTER_MANIFEST_SHA256),
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
    }


def _state(
        *,
        roster_dir: Path,
        roots: dict[str, Path],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, Path]]:
    """严格回读roster并核对三个source root存在。"""
    _manifest_value, outputs = read_normalization_recovery_v8_source_roster(
        roster_dir,
        expected_manifest_sha256=V8_SOURCE_ROSTER_MANIFEST_SHA256,
    )
    roster = outputs["source-roster.jsonl"]
    if any(not root.is_dir() for root in roots.values()):
        raise BroadQaExternalDataError(
            "v8 source content source root 不存在")
    return roster, roots


def publish_normalization_recovery_v8_source_content_audit(
        *,
        run_root: str | Path,
        roster_dir: str | Path,
        bitcoin_source_root: str | Path,
        qbittorrent_source_root: str | Path,
        stellarium_source_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布三家source的aggregate content feasibility。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=str(index)) for index, value in
                  enumerate((
                      roster_dir, bitcoin_source_root,
                      qbittorrent_source_root, stellarium_source_root,
                      target_dir)))
    roster_path, bitcoin, qbit, stellarium, target = paths
    if (target.exists() or not roster_path.is_dir()
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v8 source content input/target path 非法")
    roots = {
        "BITCOIN_CORE_PROJECT": bitcoin,
        "QBITTORRENT_PROJECT": qbit,
        "STELLARIUM_PROJECT": stellarium,
    }
    roster, verified_roots = _state(
        roster_dir=roster_path, roots=roots)
    outputs, summary = _derive(roster, verified_roots)
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


def read_normalization_recovery_v8_source_content_audit(
        audit_dir: str | Path,
        *,
        roster_dir: str | Path,
        bitcoin_source_root: str | Path,
        qbittorrent_source_root: str | Path,
        stellarium_source_root: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生aggregate，并拒绝records/manifest同步篡改。"""
    root = Path(audit_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 source content manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v8 source content manifest identity 漂移")
    roots = {
        "BITCOIN_CORE_PROJECT": Path(bitcoin_source_root).resolve(),
        "QBITTORRENT_PROJECT": Path(qbittorrent_source_root).resolve(),
        "STELLARIUM_PROJECT": Path(stellarium_source_root).resolve(),
    }
    roster, verified_roots = _state(
        roster_dir=Path(roster_dir).resolve(), roots=roots)
    expected_outputs, summary = _derive(roster, verified_roots)
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError(
            "v8 source content records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary)
    if stored != expected:
        raise BroadQaExternalDataError(
            "v8 source content fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_CONTENT_STATUS",
    "V8_SOURCE_ROSTER_MANIFEST_SHA256",
    "publish_normalization_recovery_v8_source_content_audit",
    "read_normalization_recovery_v8_source_content_aggregate",
    "read_normalization_recovery_v8_source_content_audit",
    "read_normalization_recovery_v8_source_payloads",
]
