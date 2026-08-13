"""冻结 OpenCC t2s 依赖来源，不把词典存在表述为系统学习结果。"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path

import opencc

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_DEPENDENCY_SOURCE_PACK_V1")
NORMALIZATION_SOURCE_PACK_STATUS = "DEPENDENCY_SOURCE_ONLY_NOT_LEARNED"
NORMALIZATION_SOURCE_PACKAGE = "opencc-python-reimplemented"
NORMALIZATION_SOURCE_VERSION = "0.1.7"
NORMALIZATION_SOURCE_UPSTREAM_URL = (
    "https://github.com/yichen0831/opencc-python")
NORMALIZATION_SOURCE_LICENSE_ID = "Apache-2.0"
NORMALIZATION_SOURCE_FILES = {
    "LICENSE.txt": {
        "role": "PACKAGE_LICENSE",
        "sha256": (
            "18f3fedf9eac72e6054260489c764c1e54f63cf2b16055523f561ffeec7bc908"),
    },
    "config/t2s.json": {
        "role": "CONVERSION_CONFIGURATION",
        "sha256": (
            "3848b420a86a7c5f79a796321029db6f9aaa534bc3a7a67b90fc66744d3030d0"),
    },
    "dictionary/TSCharacters.txt": {
        "role": "CHARACTER_MAPPING_DICTIONARY",
        "sha256": (
            "6b5a0a799bea2bb22c001f635eaa3fc2904310f0c08addbff275477a80ecf09a"),
    },
    "dictionary/TSPhrases.txt": {
        "role": "PHRASE_MAPPING_DICTIONARY",
        "sha256": (
            "b2ef895dd4953b4bb77fc8ef8d26a2a9ca6d43a760ed9a1d767672cfafa6324f"),
    },
}


def _sha256(payload: bytes) -> str:
    """返回小型依赖文件的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _installed_source_paths() -> dict[str, Path]:
    """定位已安装固定版本的配置、词典和 dist-info 许可。"""
    try:
        distribution = metadata.distribution(NORMALIZATION_SOURCE_PACKAGE)
    except metadata.PackageNotFoundError as error:
        raise BroadQaExternalDataError("OpenCC dependency 未安装") from error
    if distribution.version != NORMALIZATION_SOURCE_VERSION:
        raise BroadQaExternalDataError("OpenCC dependency 版本漂移")
    package_root = Path(opencc.__file__).resolve().parent
    files = distribution.files or ()
    license_entries = tuple(
        item for item in files
        if item.name == "LICENSE.txt"
        and item.parent.name.endswith(".dist-info"))
    if len(license_entries) != 1:
        raise BroadQaExternalDataError("OpenCC dependency license 不唯一")
    return {
        "LICENSE.txt": Path(distribution.locate_file(
            license_entries[0])).resolve(),
        "config/t2s.json": package_root / "config" / "t2s.json",
        "dictionary/TSCharacters.txt": (
            package_root / "dictionary" / "TSCharacters.txt"),
        "dictionary/TSPhrases.txt": (
            package_root / "dictionary" / "TSPhrases.txt"),
    }


def _parse_dictionary(payload: bytes, *, label: str) -> dict[str, int]:
    """按 OpenCC 0.1.7 的 UTF-8、单 tab、非空键值合同审计词典。"""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BroadQaExternalDataError(
            f"normalization source {label} 非 UTF-8") from error
    lines = text.splitlines()
    if not lines or not text.endswith("\n"):
        raise BroadQaExternalDataError(
            f"normalization source {label} 为空或截断")
    keys = []
    value_count = 0
    for ordinal, line in enumerate(lines, start=1):
        if line.count("\t") != 1:
            raise BroadQaExternalDataError(
                f"normalization source {label} 第 {ordinal} 行不是单 tab")
        key, value = line.split("\t")
        values = value.split(" ")
        if (not key or not value or any(not item for item in values)):
            raise BroadQaExternalDataError(
                f"normalization source {label} 第 {ordinal} 行键值为空")
        keys.append(key)
        value_count += len(values)
    if len(set(keys)) != len(keys):
        raise BroadQaExternalDataError(
            f"normalization source {label} 存在重复 key")
    return {
        "entry_count": len(keys),
        "maximum_key_codepoint_count": max(map(len, keys)),
        "minimum_key_codepoint_count": min(map(len, keys)),
        "value_variant_count": value_count,
    }


def inspect_normalization_source_payloads(
        payloads: dict[str, bytes],
        ) -> dict[str, object]:
    """结构化核验配置和词典，并返回可写入 manifest 的解析统计。"""
    if (not isinstance(payloads, dict)
            or set(payloads) != set(NORMALIZATION_SOURCE_FILES)
            or any(not isinstance(value, bytes) for value in payloads.values())):
        raise BroadQaExternalDataError("normalization source payload 集合漂移")
    try:
        config = json.loads(payloads["config/t2s.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization source t2s config 非法") from error
    expected_config = {
        "name": "Traditional Chinese to Simplified Chinese",
        "segmentation": {
            "type": "mmseg",
            "dict": {"type": "txt", "file": "TSPhrases.txt"},
        },
        "conversion_chain": [{
            "dict": {
                "type": "group",
                "dicts": [
                    {"type": "txt", "file": "TSPhrases.txt"},
                    {"type": "txt", "file": "TSCharacters.txt"},
                ],
            },
        }],
    }
    if config != expected_config:
        raise BroadQaExternalDataError(
            "normalization source t2s config 解析顺序漂移")
    dictionaries = {
        name: _parse_dictionary(payloads[name], label=name)
        for name in (
            "dictionary/TSCharacters.txt",
            "dictionary/TSPhrases.txt",
        )
    }
    try:
        license_text = payloads["LICENSE.txt"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise BroadQaExternalDataError(
            "normalization source license 非 UTF-8") from error
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise BroadQaExternalDataError("normalization source license 漂移")
    return {
        "configuration": "t2s",
        "conversion_dictionary_order": [
            "TSPhrases.txt", "TSCharacters.txt"],
        "dictionary_parser": "UTF8_SINGLE_TAB_FIRST_VALUE_V1",
        "dictionary_statistics": dictionaries,
        "matching_contract": "LONGEST_LEFT_TO_RIGHT_GROUP_FIRST_MATCH_V1",
        "multiple_mapping_policy": "FIRST_SPACE_SEPARATED_VALUE",
        "segmentation_dictionary": "TSPhrases.txt",
        "segmentation_type": "mmseg",
    }


def publish_normalization_source_pack(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """从已安装固定依赖不可覆盖发布来源 pack，不产生学习状态。"""
    root = Path(run_root).resolve()
    target = Path(target_dir).resolve()
    if not root.is_dir() or not target.is_relative_to(root):
        raise BroadQaExternalDataError(
            "normalization source target 必须位于有效 run root")
    if target.exists():
        raise BroadQaExternalDataError("normalization source target 已存在")
    paths = _installed_source_paths()
    try:
        payloads = {name: path.read_bytes() for name, path in paths.items()}
    except OSError as error:
        raise BroadQaExternalDataError(
            "normalization source dependency 文件不可读") from error
    for name, identity in NORMALIZATION_SOURCE_FILES.items():
        if _sha256(payloads[name]) != identity["sha256"]:
            raise BroadQaExternalDataError(
                f"normalization source {name} SHA 漂移")
    parsing_contract = inspect_normalization_source_payloads(payloads)
    target.mkdir(parents=True)
    file_records = []
    for name in sorted(payloads):
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payloads[name])
        file_records.append({
            "bytes": len(payloads[name]),
            "relative_path": name,
            "role": NORMALIZATION_SOURCE_FILES[name]["role"],
            "sha256": NORMALIZATION_SOURCE_FILES[name]["sha256"],
        })
    manifest = {
        "artifact_kind": NORMALIZATION_SOURCE_PACK_KIND,
        "contrastive_non_equivalence_label_count": 0,
        "file_count": len(file_records),
        "files": file_records,
        "format_version": 1,
        "learner_read_count": 0,
        "license_id": NORMALIZATION_SOURCE_LICENSE_ID,
        "package_name": NORMALIZATION_SOURCE_PACKAGE,
        "package_version": NORMALIZATION_SOURCE_VERSION,
        "parsing_contract": parsing_contract,
        "production_enabled": 0,
        "rules_written": 0,
        "semantic_labels_written": 0,
        "status": NORMALIZATION_SOURCE_PACK_STATUS,
        "upstream_url": NORMALIZATION_SOURCE_UPSTREAM_URL,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
    }


def read_normalization_source_pack(
        target_dir: str | Path,
        ) -> dict[str, object]:
    """严格回读来源 pack、每份物理文件和结构化解析合同。"""
    root = Path(target_dir).resolve()
    manifest_path = root / "manifest.json"
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization source manifest 不可读") from error
    expected = {
        "artifact_kind", "contrastive_non_equivalence_label_count",
        "file_count", "files", "format_version", "learner_read_count",
        "license_id", "package_name", "package_version",
        "parsing_contract", "production_enabled", "rules_written",
        "semantic_labels_written", "status", "upstream_url",
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected
            or canonical_json_line(manifest) != payload
            or manifest["artifact_kind"] != NORMALIZATION_SOURCE_PACK_KIND
            or type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
            or manifest["package_name"] != NORMALIZATION_SOURCE_PACKAGE
            or manifest["package_version"] != NORMALIZATION_SOURCE_VERSION
            or manifest["upstream_url"] != NORMALIZATION_SOURCE_UPSTREAM_URL
            or manifest["license_id"] != NORMALIZATION_SOURCE_LICENSE_ID
            or manifest["status"] != NORMALIZATION_SOURCE_PACK_STATUS
            or any(type(manifest[name]) is not int or manifest[name] != 0
                   for name in (
                       "contrastive_non_equivalence_label_count",
                       "learner_read_count", "production_enabled",
                       "rules_written", "semantic_labels_written"))
            or type(manifest["file_count"]) is not int
            or manifest["file_count"] != len(NORMALIZATION_SOURCE_FILES)
            or not isinstance(manifest["files"], list)):
        raise BroadQaExternalDataError(
            "normalization source manifest 漂移")
    file_records = manifest["files"]
    if [item.get("relative_path") for item in file_records] != sorted(
            NORMALIZATION_SOURCE_FILES):
        raise BroadQaExternalDataError(
            "normalization source file inventory 漂移")
    payloads = {}
    for item in file_records:
        name = item.get("relative_path") if isinstance(item, dict) else None
        expected_identity = NORMALIZATION_SOURCE_FILES.get(name)
        path = (root / name).resolve() if isinstance(name, str) else root
        try:
            file_payload = path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                "normalization source file 不可读") from error
        if (set(item) != {"bytes", "relative_path", "role", "sha256"}
                or expected_identity is None or not path.is_relative_to(root)
                or item["role"] != expected_identity["role"]
                or item["sha256"] != expected_identity["sha256"]
                or type(item["bytes"]) is not int
                or item["bytes"] != len(file_payload)
                or item["sha256"] != _sha256(file_payload)):
            raise BroadQaExternalDataError(
                "normalization source file commitment 漂移")
        payloads[name] = file_payload
    if inspect_normalization_source_payloads(
            payloads) != manifest["parsing_contract"]:
        raise BroadQaExternalDataError(
            "normalization source parsing contract 漂移")
    return {
        **manifest,
        "manifest_sha256": _sha256(payload),
    }


def main(argv: list[str] | None = None) -> int:
    """发布或回读固定 OpenCC normalization 依赖来源 pack。"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--run-root", required=True)
    publish.add_argument("--target-dir", required=True)
    read = subparsers.add_parser("read")
    read.add_argument("--target-dir", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "publish":
        report = publish_normalization_source_pack(
            run_root=arguments.run_root,
            target_dir=arguments.target_dir,
        )
    else:
        report = read_normalization_source_pack(arguments.target_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_SOURCE_FILES",
    "NORMALIZATION_SOURCE_LICENSE_ID",
    "NORMALIZATION_SOURCE_PACKAGE",
    "NORMALIZATION_SOURCE_PACK_KIND",
    "NORMALIZATION_SOURCE_PACK_STATUS",
    "NORMALIZATION_SOURCE_UPSTREAM_URL",
    "NORMALIZATION_SOURCE_VERSION",
    "inspect_normalization_source_payloads",
    "publish_normalization_source_pack",
    "read_normalization_source_pack",
]
