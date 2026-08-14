"""从固定 VS Code 本地化 archive 派生严格来源与对齐记录。

本模块只处理传入的 archive bytes。它使用标准 JSON parser 的重复键拒绝模式，
按 locale 文件相对路径和完整 JSON key path 对齐 zh-Hant/zh-Hans，并保留原文、
文件 identity、结构 token 与许可边界；不读取路径、训练、评测或 reserve。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import stat
import zipfile

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


VSCODE_SOURCE_FILE_RECORD_KIND = "VSCODE_LOCALIZATION_SOURCE_FILE_V1"
VSCODE_TRANSLATION_PAIR_RECORD_KIND = "VSCODE_LOCALIZATION_PAIR_V1"

VSCODE_LOCALE_ROOTS = {
    "zh_Hans": "i18n/vscode-language-pack-zh-hans",
    "zh_Hant": "i18n/vscode-language-pack-zh-hant",
}
VSCODE_ROOT_FILES = ("LICENSE.md", "README.md")
VSCODE_TRANSLATION_PREFIX = "translations/"
VSCODE_PAIR_TEXT_SCALAR_MAX = 320
VSCODE_ARCHIVE_FILE_COUNT_MAX = 512
VSCODE_ARCHIVE_UNCOMPRESSED_BYTES_MAX = 64 * 1024 * 1024

_HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_STRUCTURE_TOKEN = re.compile(
    r"\{\d+(?::[^{}\n]+)?\}"
    r"|\{[A-Za-z_][A-Za-z0-9_]*\}"
    r"|\$\([^\r\n()]+\)"
    r"|%\d*\$?[A-Za-z]"
    r"|</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>"
    r"|&[A-Za-z0-9#]+;"
    r"|`+"
    r"|\\[nrt]"
)


def _sha256(payload: bytes) -> str:
    """返回来源或规范记录的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    """按 Git blob 编码重算 SHA-1。"""
    header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload).hexdigest()


def _record_id(identity: dict[str, object]) -> str:
    """从完整来源 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _archive_files(archive_payload: bytes) -> dict[str, bytes]:
    """严格读取只含根许可说明与两棵 locale tree 的 Git archive。"""
    if not isinstance(archive_payload, bytes):
        raise BroadQaExternalDataError("VS Code archive payload 非 bytes")
    files: dict[str, bytes] = {}
    locale_prefixes = tuple(
        f"{value}/" for value in VSCODE_LOCALE_ROOTS.values())
    total_uncompressed_bytes = 0
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
            inventory = archive.infolist()
            if len(inventory) > VSCODE_ARCHIVE_FILE_COUNT_MAX:
                raise BroadQaExternalDataError(
                    "VS Code archive member 数超过预算")
            for info in inventory:
                path = PurePosixPath(info.filename)
                name = path.as_posix()
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if (path.is_absolute() or "\\" in name
                        or any(part in ("", ".", "..") for part in path.parts)
                        or info.flag_bits & 0x1
                        or stat.S_ISLNK(unix_mode)):
                    raise BroadQaExternalDataError(
                        "VS Code archive member path 非法")
                if info.is_dir():
                    if (name != "i18n"
                            and not any(name == root or name.startswith(
                                f"{root}/")
                                for root in VSCODE_LOCALE_ROOTS.values())):
                        raise BroadQaExternalDataError(
                            "VS Code archive directory 越界")
                    continue
                total_uncompressed_bytes += info.file_size
                if (info.file_size < 0
                        or total_uncompressed_bytes
                        > VSCODE_ARCHIVE_UNCOMPRESSED_BYTES_MAX):
                    raise BroadQaExternalDataError(
                        "VS Code archive 解压体积超过预算")
                if (name not in VSCODE_ROOT_FILES
                        and not name.startswith(locale_prefixes)):
                    raise BroadQaExternalDataError(
                        "VS Code archive 含越界来源文件")
                if name in files:
                    raise BroadQaExternalDataError(
                        "VS Code archive member identity 重复")
                files[name] = archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise BroadQaExternalDataError("VS Code archive 非法") from error
    if any(name not in files for name in VSCODE_ROOT_FILES):
        raise BroadQaExternalDataError("VS Code archive 根来源不完整")
    locale_paths = {}
    for locale, root in VSCODE_LOCALE_ROOTS.items():
        prefix = f"{root}/"
        locale_paths[locale] = {
            name[len(prefix):] for name in files if name.startswith(prefix)}
        if not locale_paths[locale]:
            raise BroadQaExternalDataError(
                "VS Code archive locale tree 为空")
    if locale_paths["zh_Hans"] != locale_paths["zh_Hant"]:
        raise BroadQaExternalDataError(
            "VS Code locale 文件 inventory 未对齐")
    translation_paths = {
        name for name in locale_paths["zh_Hans"]
        if name.startswith(VSCODE_TRANSLATION_PREFIX)
        and name.endswith(".json")
    }
    if not translation_paths:
        raise BroadQaExternalDataError(
            "VS Code archive translation JSON 为空")
    return files


def _source_file_records(
        files: dict[str, bytes],
        ) -> tuple[dict[str, object], ...]:
    """为 archive 中每个 Git blob 形成来源记录。"""
    values = []
    for relative_path, payload in sorted(files.items()):
        locale = ""
        locale_relative_path = relative_path
        for candidate, root in VSCODE_LOCALE_ROOTS.items():
            prefix = f"{root}/"
            if relative_path.startswith(prefix):
                locale = candidate
                locale_relative_path = relative_path[len(prefix):]
                break
        identity = {
            "git_blob_sha1": _git_blob_sha1(payload),
            "relative_path": relative_path,
            "sha256": _sha256(payload),
        }
        values.append({
            **identity,
            "bytes": len(payload),
            "file_id": _record_id(identity),
            "format_version": 1,
            "locale": locale,
            "locale_relative_path": locale_relative_path,
            "record_kind": VSCODE_SOURCE_FILE_RECORD_KIND,
            "role": (
                "TRANSLATION_JSON"
                if locale and locale_relative_path.startswith(
                    VSCODE_TRANSLATION_PREFIX)
                and locale_relative_path.endswith(".json")
                else "SOURCE_AUXILIARY"),
        })
    return tuple(values)


def _reject_duplicate_keys(
        pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
    """构造 JSON object，并拒绝任意层级的重复 key。"""
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise BroadQaExternalDataError(
                "VS Code translation JSON 含重复 key")
        value[key] = item
    return value


def _translation_values(
        payload: bytes,
        *,
        label: str,
        ) -> dict[tuple[str, ...], str]:
    """严格解析一个 locale JSON，并展平完整 contents key path。"""
    try:
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"VS Code {label} translation JSON 非法") from error
    if (not isinstance(value, dict)
            or tuple(value) != ("", "version", "contents")
            or value.get("version") != "1.0.0"
            or not isinstance(value.get(""), list)
            or any(not isinstance(item, str) for item in value[""])
            or not isinstance(value.get("contents"), dict)):
        raise BroadQaExternalDataError(
            f"VS Code {label} translation JSON schema 漂移")
    flattened: dict[tuple[str, ...], str] = {}

    def visit(item: object, path: tuple[str, ...]) -> None:
        if isinstance(item, dict):
            if not item:
                raise BroadQaExternalDataError(
                    f"VS Code {label} contents 含空 object")
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise BroadQaExternalDataError(
                        f"VS Code {label} contents key 非法")
                visit(child, path + (key,))
            return
        if not isinstance(item, str) or not path or path in flattened:
            raise BroadQaExternalDataError(
                f"VS Code {label} contents leaf 非法")
        flattened[path] = item

    visit(value["contents"], ())
    if not flattened:
        raise BroadQaExternalDataError(
            f"VS Code {label} contents 为空")
    return flattened


def _structure_tokens(value: str) -> tuple[str, ...]:
    """提取占位符、codicon、标记与转义结构。"""
    tokens = []
    for match in _STRUCTURE_TOKEN.finditer(value):
        token = match.group()
        if token.startswith("<"):
            body = token[1:-1].strip()
            closing = body.startswith("/")
            self_closing = body.endswith("/")
            name = body.lstrip("/").rstrip("/").split(None, 1)[0].lower()
            prefix = "HTML_CLOSE:" if closing else (
                "HTML_SELF:" if self_closing else "HTML_OPEN:")
            tokens.append(prefix + name)
        else:
            tokens.append(token)
    return tuple(tokens)


def _translation_pairs(
        files: dict[str, bytes],
        file_records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按 locale 相对路径与完整 JSON key path 派生翻译对。"""
    file_by_path = {
        str(item["relative_path"]): item for item in file_records}
    hans_root = VSCODE_LOCALE_ROOTS["zh_Hans"]
    hant_root = VSCODE_LOCALE_ROOTS["zh_Hant"]
    translation_paths = sorted({
        name[len(hans_root) + 1:]
        for name in files
        if name.startswith(f"{hans_root}/{VSCODE_TRANSLATION_PREFIX}")
        and name.endswith(".json")
    })
    values = []
    for translation_path in translation_paths:
        hans_path = f"{hans_root}/{translation_path}"
        hant_path = f"{hant_root}/{translation_path}"
        hans = _translation_values(files[hans_path], label=hans_path)
        hant = _translation_values(files[hant_path], label=hant_path)
        if set(hans) != set(hant):
            raise BroadQaExternalDataError(
                "VS Code translation JSON key inventory 未对齐")
        for json_path in sorted(hans):
            hans_text = hans[json_path]
            hant_text = hant[json_path]
            hans_tokens = _structure_tokens(hans_text)
            hant_tokens = _structure_tokens(hant_text)
            identity = {
                "json_path": list(json_path),
                "translation_relative_path": translation_path,
                "zh_hans_file_id": file_by_path[hans_path]["file_id"],
                "zh_hant_file_id": file_by_path[hant_path]["file_id"],
            }
            contains_han_both = int(
                bool(_HAN.search(hans_text)) and bool(_HAN.search(hant_text)))
            structure_equal = int(hans_tokens == hant_tokens)
            within_limit = int(
                len(hans_text) <= VSCODE_PAIR_TEXT_SCALAR_MAX
                and len(hant_text) <= VSCODE_PAIR_TEXT_SCALAR_MAX)
            values.append({
                **identity,
                "contains_han_both": contains_han_both,
                "equal_length": int(len(hans_text) == len(hant_text)),
                "format_version": 1,
                "identity_preservation": int(hans_text == hant_text),
                "json_path_sha256": _sha256(canonical_json_bytes(
                    list(json_path))),
                "pair_id": _record_id(identity),
                "record_kind": VSCODE_TRANSLATION_PAIR_RECORD_KIND,
                "structure_equal": structure_equal,
                "training_eligible": int(
                    contains_han_both and structure_equal and within_limit),
                "within_scalar_limit": within_limit,
                "zh_hans_structure_tokens": list(hans_tokens),
                "zh_hans_text": hans_text,
                "zh_hans_text_sha256": _sha256(
                    hans_text.encode("utf-8")),
                "zh_hant_structure_tokens": list(hant_tokens),
                "zh_hant_text": hant_text,
                "zh_hant_text_sha256": _sha256(
                    hant_text.encode("utf-8")),
            })
    if (not values or len({item["pair_id"] for item in values})
            != len(values)):
        raise BroadQaExternalDataError(
            "VS Code translation pair identity 非法")
    return tuple(values)


def _summary(
        file_records: tuple[dict[str, object], ...],
        pairs: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """汇总完整文件、JSON leaf 与训练资格库存。"""
    counts = Counter()
    for item in pairs:
        if item["contains_han_both"]:
            counts["both_han_pair_count"] += 1
        if item["identity_preservation"]:
            counts["all_identity_pair_count"] += 1
            if item["contains_han_both"]:
                counts["both_han_identity_pair_count"] += 1
        else:
            counts["all_nonidentity_pair_count"] += 1
            counts[
                "all_equal_length_nonidentity_pair_count"
                if item["equal_length"]
                else "all_variable_length_nonidentity_pair_count"] += 1
            if item["contains_han_both"]:
                counts["both_han_nonidentity_pair_count"] += 1
                counts[
                    "both_han_equal_length_nonidentity_pair_count"
                    if item["equal_length"]
                    else "both_han_variable_length_nonidentity_pair_count"
                ] += 1
                counts[
                    "both_han_structure_equal_nonidentity_pair_count"
                    if item["structure_equal"]
                    else "both_han_structure_mismatch_nonidentity_pair_count"
                ] += 1
                if item["structure_equal"] and not item["within_scalar_limit"]:
                    counts["both_han_structure_equal_over_limit_count"] += 1
        if item["training_eligible"]:
            counts["training_eligible_pair_count"] += 1
    locale_file_counts = {
        locale: sum(item["locale"] == locale for item in file_records)
        for locale in sorted(VSCODE_LOCALE_ROOTS)
    }
    return {
        "all_equal_length_nonidentity_pair_count": counts[
            "all_equal_length_nonidentity_pair_count"],
        "all_identity_pair_count": counts["all_identity_pair_count"],
        "all_nonidentity_pair_count": counts[
            "all_nonidentity_pair_count"],
        "all_variable_length_nonidentity_pair_count": counts[
            "all_variable_length_nonidentity_pair_count"],
        "archive_file_count": len(file_records),
        "both_han_equal_length_nonidentity_pair_count": counts[
            "both_han_equal_length_nonidentity_pair_count"],
        "both_han_identity_pair_count": counts[
            "both_han_identity_pair_count"],
        "both_han_nonidentity_pair_count": counts[
            "both_han_nonidentity_pair_count"],
        "both_han_pair_count": counts["both_han_pair_count"],
        "both_han_structure_equal_nonidentity_pair_count": counts[
            "both_han_structure_equal_nonidentity_pair_count"],
        "both_han_structure_equal_over_limit_count": counts[
            "both_han_structure_equal_over_limit_count"],
        "both_han_structure_mismatch_nonidentity_pair_count": counts[
            "both_han_structure_mismatch_nonidentity_pair_count"],
        "both_han_variable_length_nonidentity_pair_count": counts[
            "both_han_variable_length_nonidentity_pair_count"],
        "locale_file_counts": locale_file_counts,
        "pair_count": len(pairs),
        "training_eligible_pair_count": counts[
            "training_eligible_pair_count"],
        "translation_json_file_count": sum(
            item["locale"] == "zh_Hans"
            and item["role"] == "TRANSLATION_JSON"
            for item in file_records),
    }


def parse_normalization_recovery_v4_vscode_archive(
        archive_payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从 archive bytes 派生来源文件、翻译对与 census。"""
    files = _archive_files(archive_payload)
    file_records = _source_file_records(files)
    pairs = _translation_pairs(files, file_records)
    return file_records, pairs, _summary(file_records, pairs)


__all__ = [
    "VSCODE_LOCALE_ROOTS",
    "VSCODE_PAIR_TEXT_SCALAR_MAX",
    "VSCODE_SOURCE_FILE_RECORD_KIND",
    "VSCODE_TRANSLATION_PAIR_RECORD_KIND",
    "parse_normalization_recovery_v4_vscode_archive",
]
