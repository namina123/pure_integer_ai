"""发布并严格回读单个 recovery-v8 TRAIN source family pack。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit import (
    read_normalization_recovery_v8_source_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit_v2 import (
    read_normalization_recovery_v8_source_content_aggregate_v2,
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


NORMALIZATION_RECOVERY_V8_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_TRAIN_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_PACK_STATUS = (
    "SEALED_TRAIN_SOURCE_ONLY_NOT_LEARNED")
NORMALIZATION_RECOVERY_V8_SOURCE_PACK_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V8_TRAIN_SOURCE_PACK_CENSUS_V1")

V8_SOURCE_ROSTER_V2_MANIFEST_SHA256 = (
    "60c801a6e3b41adf59f06f0ebbfbccc030a5dfdcc1807012ca6bfc5e51e1f68a")
V8_SOURCE_CONTENT_V2_MANIFEST_SHA256 = (
    "e938a02f99d3ac22ee778d720130b09a50c021abf8bee4e1deb6affcfad2e3fe")

_SOURCE_FAMILIES = (
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
    "KEEPASSXC_PROJECT",
)
_OUTPUT_FILES = (
    ("source-files.jsonl", "V8_SOURCE_FILE_RECORDS"),
    ("translation-pairs.jsonl", "V8_TRANSLATION_PAIR_RECORDS"),
    ("source-census.jsonl", "V8_SOURCE_PACK_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回artifact、record或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v8 source pack run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 source pack {label} 越出run root") from error
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


def _record_sha256(value: dict[str, object]) -> str:
    """形成predecessor record的规范commitment。"""
    return _sha256(canonical_json_line(value))


def _state(
        *,
        source_family: str,
        v2_roster_dir: Path,
        v1_roster_dir: Path,
        v1_content_audit_dir: Path,
        v2_content_audit_dir: Path,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """严格回读roster-v2与content-v2，并选择唯一family记录。"""
    if source_family not in _SOURCE_FAMILIES:
        raise BroadQaExternalDataError(
            "v8 source pack family 未支持")
    _roster_manifest, roster_outputs = (
        read_normalization_recovery_v8_source_roster_v2(
            v2_roster_dir,
            v1_roster_dir=v1_roster_dir,
            content_audit_dir=v1_content_audit_dir,
            expected_manifest_sha256=V8_SOURCE_ROSTER_V2_MANIFEST_SHA256,
        ))
    _content_manifest, content_outputs = (
        read_normalization_recovery_v8_source_content_aggregate_v2(
            v2_content_audit_dir,
            expected_manifest_sha256=V8_SOURCE_CONTENT_V2_MANIFEST_SHA256,
        ))
    roster = {str(item.get("source_family")): item
              for item in roster_outputs["source-roster-v2.jsonl"]}
    content = {str(item.get("source_family")): item
               for item in content_outputs["source-content-v2.jsonl"]}
    if (set(roster) != set(_SOURCE_FAMILIES)
            or set(content) != set(_SOURCE_FAMILIES)):
        raise BroadQaExternalDataError(
            "v8 source pack predecessor inventory 漂移")
    return roster[source_family], content[source_family]


def _validate_content_record(
        roster: dict[str, object],
        content: dict[str, object],
        *,
        parser_summary: dict[str, object],
        pair_count: int,
        ) -> None:
    """要求完整重派生结果与sealed content-v2 aggregate逐字段一致。"""
    if (content.get("content_outcome")
            != "PASS_NONZERO_ACTIVE_COMMON_PAIR"
            or content.get("source_family") != roster.get("source_family")
            or content.get("source_policy_scope")
            != roster.get("source_policy_scope")
            or content.get("license_expression")
            != roster.get("license", {}).get("expression")
            or content.get("license_file_count")
            != len(roster.get("license", {}).get("files", []))
            or content.get("locale_file_count")
            != roster.get("locale_file_count")
            or content.get("transient_pair_count") != pair_count
            or content.get("parser_summary") != parser_summary):
        raise BroadQaExternalDataError(
            "v8 source pack content-v2 aggregate 漂移")


def _derive(
        *,
        roster: dict[str, object],
        content: dict[str, object],
        source_root: Path,
        ) -> tuple[
            dict[str, bytes],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
            dict[str, object],
        ]:
    """逐blob核验并派生完整source files、pairs与census。"""
    payloads = read_normalization_recovery_v8_source_payloads(
        roster, source_root)
    file_records, pairs, parser_summary = (
        derive_normalization_recovery_v8_source_family_records(
            roster, payloads))
    _validate_content_record(
        roster, content,
        parser_summary=parser_summary,
        pair_count=len(pairs),
    )
    license_paths = {
        str(item["relative_path"]) for item in roster["license"]["files"]}
    locale_paths = {
        str(item["relative_path"]) for item in roster["locale_files"]}
    if (set(payloads) != license_paths.union(locale_paths)
            or len(file_records) != len(locale_paths)):
        raise BroadQaExternalDataError(
            "v8 source pack raw/parser inventory 漂移")
    census = {
        "excluded_or_ineligible_pair_count": (
            len(pairs)
            - int(parser_summary["v8_training_eligible_pair_count"])),
        "format_version": 1,
        "identity_pair_count": int(parser_summary["identity_pair_count"]),
        "license_file_count": len(license_paths),
        "locale_file_count": len(locale_paths),
        "pair_record_count": len(pairs),
        "pair_surface_public_git_count": 0,
        "raw_blob_count": len(payloads),
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_PACK_CENSUS_KIND,
        "source_family": roster["source_family"],
        "source_file_record_count": len(file_records),
        "source_pack_family_vote_count": 1,
        "structure_equal_pair_count": int(
            parser_summary["structure_equal_count"]),
        "structure_unequal_pair_count": (
            len(pairs) - int(parser_summary["structure_equal_count"])),
        "v8_training_eligible_pair_count": int(
            parser_summary["v8_training_eligible_pair_count"]),
    }
    return payloads, file_records, pairs, census, parser_summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格读取规范、非空JSONL。"""
    values = []
    try:
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BroadQaExternalDataError(
                    f"v8 source pack {label} record 非对象")
            values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v8 source pack {label} 不可读") from error
    if (not lines or b"".join(lines) != payload
            or b"".join(canonical_json_line(item) for item in values)
            != payload):
        raise BroadQaExternalDataError(
            f"v8 source pack {label} JSONL 非规范")
    return tuple(values)


def _artifact(
        root: Path,
        path: Path,
        *,
        role: str,
        count: int,
        ) -> dict[str, object]:
    """形成支持嵌套raw路径的物理文件commitment。"""
    payload = path.read_bytes()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise BroadQaExternalDataError(
            "v8 source pack artifact path 越界") from error
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": relative,
        "role": role,
        "sha256": _sha256(payload),
    }


def _raw_target(target: Path, relative: str) -> Path:
    """把roster相对路径映射到pack内raw目录并拒绝路径穿越。"""
    path = (target / "raw" / Path(relative)).resolve()
    try:
        path.relative_to(target / "raw")
    except ValueError as error:
        raise BroadQaExternalDataError(
            "v8 source pack raw path 越界") from error
    return path


def _raw_artifacts(
        target: Path,
        roster: dict[str, object],
        payloads: dict[str, bytes],
        ) -> list[dict[str, object]]:
    """不可覆盖保存固定raw blobs并形成逐文件commitment。"""
    license_paths = {
        str(item["relative_path"]) for item in roster["license"]["files"]}
    files = []
    for relative in sorted(payloads):
        path = _raw_target(target, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payloads[relative])
        role = (
            "V8_SOURCE_RAW_LICENSE_BLOB" if relative in license_paths
            else "V8_SOURCE_RAW_LOCALE_BLOB")
        files.append(_artifact(target, path, role=role, count=0))
    return files


def _require_physical_inventory(
        root: Path,
        payloads: dict[str, bytes],
        ) -> None:
    """拒绝pack内任何manifest、records或roster raw之外的额外文件。"""
    expected = {
        "manifest.json",
        *(name for name, _role in _OUTPUT_FILES),
        *(f"raw/{Path(relative).as_posix()}" for relative in payloads),
    }
    try:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*") if path.is_file()
        }
    except OSError as error:
        raise BroadQaExternalDataError(
            "v8 source pack physical inventory 不可读") from error
    if actual != expected:
        raise BroadQaExternalDataError(
            "v8 source pack physical inventory 漂移")


def _manifest(
        *,
        roster: dict[str, object],
        content: dict[str, object],
        files: list[dict[str, object]],
        census: dict[str, object],
        parser_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造单family、未训练且不增加family vote的source-pack manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_SOURCE_PACK_KIND,
        "files": files,
        "format_version": 1,
        "inputs": {
            "content_record_sha256": _record_sha256(content),
            "source_content_v2_manifest_sha256": (
                V8_SOURCE_CONTENT_V2_MANIFEST_SHA256),
            "source_roster_v2_manifest_sha256": (
                V8_SOURCE_ROSTER_V2_MANIFEST_SHA256),
            "source_roster_record_sha256": _record_sha256(roster),
        },
        "license": roster["license"],
        "mastery_claimed": 0,
        "observation_pack_published": 0,
        "parser_identity": roster["parser_identity"],
        "parser_summary": parser_summary,
        "production_enabled": 0,
        "raw_source": {
            "commit": roster["commit"],
            "commit_date": roster["commit_date"],
            "repository": roster["repository"],
            "root_tree": roster["root_tree"],
        },
        "source_census": census,
        "source_family": roster["source_family"],
        "source_family_vote_count": 1,
        "source_policy_scope": roster["source_policy_scope"],
        "status": NORMALIZATION_RECOVERY_V8_SOURCE_PACK_STATUS,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
        "training_read_count": 0,
    }


def publish_normalization_recovery_v8_source_pack(
        *,
        run_root: str | Path,
        source_family: str,
        v2_roster_dir: str | Path,
        v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path,
        v2_content_audit_dir: str | Path,
        source_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布一个v8 source family pack。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=str(index)) for index, value in
                  enumerate((
                      v2_roster_dir,
                      v1_roster_dir,
                      v1_content_audit_dir,
                      v2_content_audit_dir,
                      source_root,
                      target_dir,
                  )))
    v2_roster, v1_roster, v1_content, v2_content, source, target = paths
    if (target.exists()
            or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v8 source pack input/target path 非法")
    roster, content = _state(
        source_family=source_family,
        v2_roster_dir=v2_roster,
        v1_roster_dir=v1_roster,
        v1_content_audit_dir=v1_content,
        v2_content_audit_dir=v2_content,
    )
    payloads, source_files, pairs, census, parser_summary = _derive(
        roster=roster, content=content, source_root=source)
    target.mkdir()
    files = _raw_artifacts(target, roster, payloads)
    outputs = {
        _OUTPUT_FILES[0][0]: source_files,
        _OUTPUT_FILES[1][0]: pairs,
        _OUTPUT_FILES[2][0]: (census,),
    }
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(
            target, path, role=role, count=len(outputs[name])))
    manifest = _manifest(
        roster=roster,
        content=content,
        files=files,
        census=census,
        parser_summary=parser_summary,
    )
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v8_source_pack(
        source_pack_dir: str | Path,
        *,
        v2_roster_dir: str | Path,
        v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path,
        v2_content_audit_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从pack内raw blobs严格重派生全部records与manifest。"""
    root = Path(source_pack_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 source pack manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V8_SOURCE_PACK_KIND):
        raise BroadQaExternalDataError(
            "v8 source pack manifest identity 漂移")
    family = stored.get("source_family")
    if not isinstance(family, str):
        raise BroadQaExternalDataError(
            "v8 source pack family identity 漂移")
    roster, content = _state(
        source_family=family,
        v2_roster_dir=Path(v2_roster_dir).resolve(),
        v1_roster_dir=Path(v1_roster_dir).resolve(),
        v1_content_audit_dir=Path(v1_content_audit_dir).resolve(),
        v2_content_audit_dir=Path(v2_content_audit_dir).resolve(),
    )
    payloads, source_files, pairs, census, parser_summary = _derive(
        roster=roster,
        content=content,
        source_root=root / "raw",
    )
    _require_physical_inventory(root, payloads)
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    expected_outputs = {
        _OUTPUT_FILES[0][0]: source_files,
        _OUTPUT_FILES[1][0]: pairs,
        _OUTPUT_FILES[2][0]: (census,),
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError(
            "v8 source pack records 漂移")
    files = []
    license_paths = {
        str(item["relative_path"]) for item in roster["license"]["files"]}
    for relative in sorted(payloads):
        raw_path = _raw_target(root, relative)
        role = (
            "V8_SOURCE_RAW_LICENSE_BLOB" if relative in license_paths
            else "V8_SOURCE_RAW_LOCALE_BLOB")
        files.append(_artifact(root, raw_path, role=role, count=0))
    for name, role in _OUTPUT_FILES:
        files.append(_artifact(
            root,
            root / name,
            role=role,
            count=len(expected_outputs[name]),
        ))
    expected = _manifest(
        roster=roster,
        content=content,
        files=files,
        census=census,
        parser_summary=parser_summary,
    )
    if stored != expected:
        raise BroadQaExternalDataError(
            "v8 source pack fields 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded)},
        source_files,
        pairs,
        census,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V8_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_PACK_STATUS",
    "V8_SOURCE_CONTENT_V2_MANIFEST_SHA256",
    "V8_SOURCE_ROSTER_V2_MANIFEST_SHA256",
    "publish_normalization_recovery_v8_source_pack",
    "read_normalization_recovery_v8_source_pack",
]
