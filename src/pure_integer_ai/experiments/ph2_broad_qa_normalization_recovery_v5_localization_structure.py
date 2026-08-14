"""recovery-v5 本地化来源共享的结构与物理边界。

模块只处理调用方传入的 bytes 和字符串，不打开路径，也不接触训练、评测、
candidate 或生产状态。PO/TS adapter 共享同一套 archive 安全检查、来源 identity、
结构 token 与 pair 特征，避免不同格式各自发明“结构相等”的含义。
"""
from __future__ import annotations

import hashlib
import io
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


_HAN_SCALAR = re.compile(r"^[\u3400-\u9fff\uf900-\ufaff]$")
_HAN_ANYWHERE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_STRUCTURE_TOKEN = re.compile(
    r"%(?:[A-Z][A-Z0-9_]*|L?\d+|n|"
    r"[-+#0 ]*\d*(?:\.\d+)?[diouxXeEfFgGcrs%])"
    r"|\$\[[^\]\n]+\]"
    r"|\$\([^\r\n()]+\)"
    r"|\{[^{}\n]+\}"
    r"|\[(?:/?(?:b|i|u|s|code|url|img|color|font|font_size|hint|kbd|"
    r"center|right|fill|indent|ol|ul|table|cell|p|br)(?:=[^\]\n]*)?)\]"
    r"|</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>"
    r"|&[A-Za-z0-9#]+;"
    r"|`+"
    r"|\\[nrt]",
    re.IGNORECASE,
)


def sha256_hex(payload: bytes) -> str:
    """返回来源或规范 artifact 的 SHA-256。"""
    if not isinstance(payload, bytes):
        raise BroadQaExternalDataError("localization hash payload 非 bytes")
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    """按 Git blob 编码重算 SHA-1。"""
    if not isinstance(payload, bytes):
        raise BroadQaExternalDataError("localization blob payload 非 bytes")
    header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload).hexdigest()


def localization_record_id(identity: dict[str, object]) -> str:
    """从完整结构化来源 identity 形成稳定记录 id。"""
    if not isinstance(identity, dict) or not identity:
        raise BroadQaExternalDataError("localization record identity 非法")
    return sha256_hex(canonical_json_bytes(identity))


def strict_json_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值，并严格区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(strict_json_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(strict_json_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def read_exact_localization_zip(
        archive_payload: bytes,
        *,
        expected_files: tuple[str, ...],
        allowed_directories: tuple[str, ...] = (),
        label: str,
        member_count_max: int,
        uncompressed_bytes_max: int,
        ) -> dict[str, bytes]:
    """严格读取文件清单已冻结的有界 ZIP，不接受越界或隐式成员。"""
    if (not isinstance(archive_payload, bytes)
            or not isinstance(expected_files, tuple)
            or not expected_files
            or len(set(expected_files)) != len(expected_files)
            or any(not isinstance(item, str) or not item
                   for item in expected_files)
            or not isinstance(allowed_directories, tuple)
            or len(set(allowed_directories)) != len(allowed_directories)
            or any(not isinstance(item, str) or not item
                   for item in allowed_directories)
            or type(member_count_max) is not int
            or member_count_max < len(expected_files)
            or type(uncompressed_bytes_max) is not int
            or uncompressed_bytes_max <= 0):
        raise BroadQaExternalDataError(f"{label} ZIP contract 非法")
    expected = set(expected_files)
    directories = {item.rstrip("/") for item in allowed_directories}
    files: dict[str, bytes] = {}
    total_uncompressed = 0
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
            inventory = archive.infolist()
            if len(inventory) > member_count_max:
                raise BroadQaExternalDataError(
                    f"{label} ZIP member 数超过预算")
            for info in inventory:
                path = PurePosixPath(info.filename)
                name = path.as_posix().rstrip("/")
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if (path.is_absolute() or "\\" in info.filename
                        or any(part in ("", ".", "..")
                               for part in path.parts)
                        or info.flag_bits & 0x1
                        or stat.S_ISLNK(unix_mode)):
                    raise BroadQaExternalDataError(
                        f"{label} ZIP member path 非法")
                if info.is_dir():
                    if name not in directories:
                        raise BroadQaExternalDataError(
                            f"{label} ZIP directory 越界")
                    continue
                if name not in expected or name in files:
                    raise BroadQaExternalDataError(
                        f"{label} ZIP file inventory 漂移")
                if info.file_size < 0:
                    raise BroadQaExternalDataError(
                        f"{label} ZIP member size 非法")
                total_uncompressed += info.file_size
                if total_uncompressed > uncompressed_bytes_max:
                    raise BroadQaExternalDataError(
                        f"{label} ZIP 解压体积超过预算")
                payload = archive.read(info)
                if len(payload) != info.file_size:
                    raise BroadQaExternalDataError(
                        f"{label} ZIP member bytes 漂移")
                files[name] = payload
    except BroadQaExternalDataError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise BroadQaExternalDataError(f"{label} ZIP 非法") from error
    if set(files) != expected:
        raise BroadQaExternalDataError(f"{label} ZIP 来源不完整")
    return files


def localization_structure_tokens(value: str) -> tuple[str, ...]:
    """提取占位符、嵌入标记、实体和显式转义的有序结构。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError("localization structure text 非字符串")
    tokens = []
    for match in _STRUCTURE_TOKEN.finditer(value):
        token = match.group()
        if token.startswith("[") and not token.startswith("$["):
            body = token[1:-1]
            closing = body.startswith("/")
            name = body.lstrip("/").split("=", 1)[0].lower()
            tokens.append(
                ("BBCODE_CLOSE:" if closing else "BBCODE_OPEN:") + name)
        elif token.startswith("<"):
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


def localization_pair_features(
        zh_hant_text: str,
        zh_hans_text: str,
        *,
        scalar_limit: int,
        ) -> dict[str, object]:
    """形成简繁 pair 的结构、长度、identity 与单 Han 差异事实。"""
    if (not isinstance(zh_hant_text, str)
            or not isinstance(zh_hans_text, str)
            or type(scalar_limit) is not int
            or scalar_limit <= 0):
        raise BroadQaExternalDataError("localization pair feature 输入非法")
    hant_tokens = localization_structure_tokens(zh_hant_text)
    hans_tokens = localization_structure_tokens(zh_hans_text)
    equal_length = int(len(zh_hant_text) == len(zh_hans_text))
    differences = (
        tuple((left, right) for left, right in zip(
            zh_hant_text, zh_hans_text) if left != right)
        if equal_length else ())
    single_han_difference = int(
        len(differences) == 1
        and bool(_HAN_SCALAR.fullmatch(differences[0][0]))
        and bool(_HAN_SCALAR.fullmatch(differences[0][1])))
    structure_equal = int(hant_tokens == hans_tokens)
    within_scalar_limit = int(
        len(zh_hant_text) <= scalar_limit
        and len(zh_hans_text) <= scalar_limit)
    return {
        "contains_han_both": int(
            bool(_HAN_ANYWHERE.search(zh_hant_text))
            and bool(_HAN_ANYWHERE.search(zh_hans_text))),
        "equal_length": equal_length,
        "identity_preservation": int(zh_hant_text == zh_hans_text),
        "single_han_difference": single_han_difference,
        "structure_equal": structure_equal,
        "training_eligible": int(
            structure_equal == 1 and within_scalar_limit == 1),
        "within_scalar_limit": within_scalar_limit,
        "zh_hans_structure_tokens": list(hans_tokens),
        "zh_hant_structure_tokens": list(hant_tokens),
    }


__all__ = [
    "git_blob_sha1",
    "localization_pair_features",
    "localization_record_id",
    "localization_structure_tokens",
    "read_exact_localization_zip",
    "sha256_hex",
    "strict_json_equal",
]
