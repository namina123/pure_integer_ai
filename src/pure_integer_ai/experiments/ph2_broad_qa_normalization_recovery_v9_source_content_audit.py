"""发布 recovery-v9 GIMP aggregate-only 内容可行性审计。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_gettext_source_records import (
    derive_normalization_recovery_v9_gettext_source_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_roster import (
    read_normalization_recovery_v9_source_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_CONTENT_FEASIBILITY_V1")
NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_CONTENT_FEASIBILITY_RECORD_V1")
NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_CONTENT_FEASIBILITY_CENSUS_V1")
NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_PASS_STATUS = (
    "GIMP_EIGHT_DOMAIN_CONTENT_PASS_NOT_SOURCE_PACK_NOT_FORMAL")
NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_REJECTED_STATUS = (
    "GIMP_CONTENT_REJECTED_NOT_SOURCE_PACK_NOT_FORMAL")

V9_SOURCE_ROSTER_MANIFEST_SHA256 = (
    "fd036c301ed901a861c7e58b62359b30dc2ed98a9f836afc76453730112f92d8")

_OUTPUT_FILES = (
    ("source-content.jsonl", "V9_GIMP_SOURCE_CONTENT_FEASIBILITY"),
    ("source-content-census.jsonl", "V9_GIMP_SOURCE_CONTENT_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回 artifact、record 或 manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 位于已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v9 source content run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入输出仍位于本次 K 盘 run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v9 source content {label} 越出run root") from error
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
    """按 roster 精确读取 16 个 locale blob 并拒绝额外文件。"""
    locale_files = record.get("locale_files")
    if not isinstance(locale_files, list) or len(locale_files) != 16:
        raise BroadQaExternalDataError("v9 source content locale roster 漂移")
    expected = {str(item.get("relative_path")): item
                for item in locale_files if isinstance(item, dict)}
    physical = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    if len(expected) != 16 or physical != set(expected):
        raise BroadQaExternalDataError(
            "v9 source content physical inventory 漂移")
    payloads = {}
    for relative, item in expected.items():
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
            payload = path.read_bytes()
        except (ValueError, OSError) as error:
            raise BroadQaExternalDataError(
                "v9 source content blob 不可读") from error
        if (len(payload) != item.get("bytes")
                or git_blob_sha1(payload) != item.get("git_blob_sha1")):
            raise BroadQaExternalDataError(
                "v9 source content blob identity 漂移")
        payloads[relative] = payload
    return payloads


def _pair_specs(record: dict[str, object]) -> tuple[dict[str, object], ...]:
    """从冻结 roster 派生八个 domain 的简繁 parser specs。"""
    by_domain: dict[str, dict[str, str]] = {}
    for item in record["locale_files"]:
        domain = str(item["domain"])
        locale = str(item["locale"])
        path = str(item["relative_path"])
        if locale in by_domain.setdefault(domain, {}):
            raise BroadQaExternalDataError("v9 source content locale 重复")
        by_domain[domain][locale] = path
    if len(by_domain) != 8 or any(
            set(values) != {"zh_CN", "zh_TW"}
            for values in by_domain.values()):
        raise BroadQaExternalDataError("v9 source content domain pair 漂移")
    return tuple({
        "domain": domain,
        "zh_Hans": {
            "expected_language": "zh_CN",
            "relative_path": values["zh_CN"],
        },
        "zh_Hant": {
            "expected_language": "zh_TW",
            "relative_path": values["zh_TW"],
        },
    } for domain, values in sorted(by_domain.items()))


def _content_record(
        record: dict[str, object],
        source_root: Path,
        ) -> dict[str, object]:
    """重解析 GIMP locale 并仅返回 aggregate content record。"""
    if (record.get("source_family") != "GIMP_PROJECT"
            or record.get("locale_blob_content_read_count") != 0
            or record.get("label_or_translation_read_count") != 0
            or record.get("locale_pair_count") != 8
            or record.get("license", {}).get("expression")
            != "GPL-3.0-or-later"):
        raise BroadQaExternalDataError("v9 source content roster 字段漂移")
    payloads = _read_payloads(record, source_root)
    source_files, pairs, summary = (
        derive_normalization_recovery_v9_gettext_source_records(
            source_family="GIMP_PROJECT",
            source_policy_scope=str(record["source_policy_scope"]),
            license_expression="GPL-3.0-or-later",
            pair_specs=_pair_specs(record),
            files=payloads,
        ))
    identity_roster = [{
        "pair_id": item["pair_id"],
        "source_identity": item["source_identity"],
    } for item in pairs]
    return {
        "content_outcome": summary["content_outcome"],
        "format_version": 1,
        "individual_label_print_count": 0,
        "label_pair_surface_published": 0,
        "license_expression": "GPL-3.0-or-later",
        "locale_blob_read_count": len(payloads),
        "locale_file_commitment_sha256": _sha256(
            canonical_json_bytes(source_files)),
        "pair_identity_roster_sha256": _sha256(
            canonical_json_bytes(identity_roster)),
        "parser_identity": "POLIB_1_2_0_GETTEXT_PO_V9_OBSOLETE_STRICT_V1",
        "parser_summary": summary,
        "record_kind": NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_RECORD_KIND,
        "source_family": "GIMP_PROJECT",
        "source_pack_published": 0,
        "source_policy_scope": record["source_policy_scope"],
        "source_roster_manifest_sha256": V9_SOURCE_ROSTER_MANIFEST_SHA256,
        "transient_pair_count": len(pairs),
    }


def _derive(
        roster: tuple[dict[str, object], ...],
        source_root: Path,
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """由唯一 GIMP roster record 派生 aggregate artifact 输出。"""
    if len(roster) != 1:
        raise BroadQaExternalDataError("v9 source content roster 分母漂移")
    record = _content_record(roster[0], source_root)
    summary = {
        "content_pass_count": int(
            record["content_outcome"] == "PASS_NONZERO_ACTIVE_COMMON_PAIR"),
        "content_rejected_count": int(
            record["content_outcome"] != "PASS_NONZERO_ACTIVE_COMMON_PAIR"),
        "individual_label_print_count": 0,
        "label_pair_surface_published": 0,
        "locale_blob_read_count": record["locale_blob_read_count"],
        "plain_pair_count": record["parser_summary"]["plain_pair_count"],
        "selected_source_family_count": 1,
        "source_pack_published_count": 0,
        "structure_equal_count": record[
            "parser_summary"]["structure_equal_count"],
        "v9_evaluation_eligible_pair_count": record[
            "parser_summary"]["v9_evaluation_eligible_pair_count"],
    }
    census = ({
        **summary,
        "format_version": 1,
        "record_kind": NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_CENSUS_KIND,
    },)
    return {
        _OUTPUT_FILES[0][0]: (record,),
        _OUTPUT_FILES[1][0]: census,
    }, summary


def _state(
        roster_dir: Path,
        source_root: Path,
        ) -> tuple[tuple[dict[str, object], ...], Path]:
    """严格回读 source roster 并核对 locale root 存在。"""
    _manifest, outputs = read_normalization_recovery_v9_source_roster(
        roster_dir,
        expected_manifest_sha256=V9_SOURCE_ROSTER_MANIFEST_SHA256,
    )
    if not source_root.is_dir():
        raise BroadQaExternalDataError("v9 source content source root 不存在")
    return outputs["source-roster.jsonl"], source_root


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """读取并核验本 artifact 的规范 JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v9 source content {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v9 source content {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成 aggregate 输出文件 commitment。"""
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
    """构造 GIMP content PASS 或 rejected manifest。"""
    status = (
        NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_PASS_STATUS
        if summary["content_pass_count"] == 1
        and summary["content_rejected_count"] == 0
        else NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_REJECTED_STATUS)
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_AUDIT_KIND,
        "files": files,
        "format_version": 1,
        "individual_label_print_count": 0,
        "inputs": {
            "source_roster_manifest_sha256": V9_SOURCE_ROSTER_MANIFEST_SHA256,
        },
        "label_pair_surface_published": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": status,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
    }


def publish_normalization_recovery_v9_source_content_audit(
        *,
        run_root: str | Path,
        roster_dir: str | Path,
        source_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 GIMP aggregate-only content feasibility。"""
    root = _require_k_root(run_root)
    roster = _within(root, roster_dir, label="roster")
    source = _within(root, source_root, label="source")
    target = _within(root, target_dir, label="target")
    if (target.exists() or not roster.is_dir() or not source.is_dir()
            or _overlap(target, roster) or _overlap(target, source)):
        raise BroadQaExternalDataError("v9 source content input/target 非法")
    outputs, summary = _derive(*_state(roster, source))
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


def read_normalization_recovery_v9_source_content_aggregate(
        audit_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """只按封存文件回读 aggregate，不重新打开 locale blob。"""
    root = Path(audit_dir).resolve()
    expected_names = {"manifest.json", *[name for name, _role in _OUTPUT_FILES]}
    try:
        physical_names = {item.name for item in root.iterdir()}
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v9 source content aggregate 不可读") from error
    if (physical_names != expected_names
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_AUDIT_KIND):
        raise BroadQaExternalDataError(
            "v9 source content aggregate identity 漂移")
    by_name = {str(item.get("relative_path")): item
               for item in stored.get("files", [])
               if isinstance(item, dict)}
    if set(by_name) != {name for name, _role in _OUTPUT_FILES}:
        raise BroadQaExternalDataError(
            "v9 source content aggregate file inventory 漂移")
    outputs = {}
    for name, role in _OUTPUT_FILES:
        values = _read_jsonl(root / name, label=role)
        if by_name[name] != _artifact(
                root / name, role=role, count=len(values)):
            raise BroadQaExternalDataError(
                "v9 source content aggregate file identity 漂移")
        outputs[name] = values
    census = outputs["source-content-census.jsonl"]
    if (len(outputs["source-content.jsonl"]) != 1 or len(census) != 1
            or stored.get("summary") != {
                key: value for key, value in census[0].items()
                if key not in {"format_version", "record_kind"}
            }):
        raise BroadQaExternalDataError(
            "v9 source content aggregate census 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, outputs


def read_normalization_recovery_v9_source_content_audit(
        audit_dir: str | Path,
        *,
        roster_dir: str | Path,
        source_root: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重解析 locale 并拒绝 aggregate/manifest 同步篡改。"""
    root = Path(audit_dir).resolve()
    aggregate, stored_outputs = (
        read_normalization_recovery_v9_source_content_aggregate(
            root, expected_manifest_sha256=expected_manifest_sha256))
    expected_outputs, summary = _derive(*_state(
        Path(roster_dir).resolve(), Path(source_root).resolve()))
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError("v9 source content records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary)
    stored = dict(aggregate)
    stored.pop("manifest_sha256")
    if stored != expected:
        raise BroadQaExternalDataError("v9 source content fields 漂移")
    return aggregate, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_PASS_STATUS",
    "NORMALIZATION_RECOVERY_V9_SOURCE_CONTENT_REJECTED_STATUS",
    "V9_SOURCE_ROSTER_MANIFEST_SHA256",
    "publish_normalization_recovery_v9_source_content_audit",
    "read_normalization_recovery_v9_source_content_aggregate",
    "read_normalization_recovery_v9_source_content_audit",
]
