"""冻结并严格解析独立 ICU 简繁 transliteration 来源。

本模块只保存 ICU release-77-1 的来源身份和规则结构。它不读取 OpenCC
development pack，不运行评测，也不把 ICU 规则存在表述为系统能力。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_ICU_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_EVALUATION_SOURCE_PACK_V1")
NORMALIZATION_ICU_SOURCE_STATUS = (
    "INDEPENDENT_SOURCE_FROZEN_NOT_SELECTED_NOT_EVALUATED")
NORMALIZATION_ICU_RELEASE = "release-77-1"
NORMALIZATION_ICU_COMMIT = "457157a92aa053e632cc7fcfd0e12f8a943b2d11"
NORMALIZATION_ICU_BLOB_SHA = "95ec2976342052df0d03797ba156c61c357a49ce"
NORMALIZATION_ICU_LICENSE_ID = "Unicode-3.0"
NORMALIZATION_ICU_RULE_PATH = "icu4c/source/data/translit/Hans_Hant.txt"
NORMALIZATION_ICU_RULE_URL = (
    "https://raw.githubusercontent.com/unicode-org/icu/release-77-1/"
    "icu4c/source/data/translit/Hans_Hant.txt")
NORMALIZATION_ICU_LICENSE_URL = (
    "https://raw.githubusercontent.com/unicode-org/icu/release-77-1/LICENSE")
NORMALIZATION_ICU_RULE_SHA256 = (
    "9f5945be3b0b286df3a2874203496e6ecd3ec6c95200edd4208f7d1830511ce2")
NORMALIZATION_ICU_LICENSE_SHA256 = (
    "451167c55c0fa447cc2d5632714f5e3c567fe4f1e1badefab2c1333852198aca")
NORMALIZATION_ICU_RULE_BYTES = 55_681
NORMALIZATION_ICU_LICENSE_BYTES = 26_487
NORMALIZATION_ICU_VARIABLE_KIND = "ICU_TRANSLITERATION_VARIABLE_V1"
NORMALIZATION_ICU_RULE_KIND = "ICU_TRANSLITERATION_RULE_V1"
NORMALIZATION_ICU_ARROWS = ("←", "→", "↔")


def _sha256(payload: bytes) -> str:
    """返回来源或规范 artifact 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _physical_lines(payload: bytes) -> tuple[dict[str, object], ...]:
    """切分带 BOM 的 LF 文件并保留每条物理行承诺。"""
    if not payload.startswith(b"\xef\xbb\xbf"):
        raise BroadQaExternalDataError("ICU normalization source 缺少 UTF-8 BOM")
    encoded_lines = payload.splitlines(keepends=True)
    if (not encoded_lines or any(not line.endswith(b"\n")
                                 for line in encoded_lines)
            or any(line.endswith(b"\r\n") for line in encoded_lines)):
        raise BroadQaExternalDataError(
            "ICU normalization source 必须是完整 LF 物理行")
    records = []
    byte_start = 0
    for ordinal, encoded_line in enumerate(encoded_lines, start=1):
        try:
            text = encoded_line.decode(
                "utf-8-sig" if ordinal == 1 else "utf-8")
        except UnicodeDecodeError as error:
            raise BroadQaExternalDataError(
                "ICU normalization source 非 UTF-8") from error
        if "\r" in text or not text.endswith("\n"):
            raise BroadQaExternalDataError(
                "ICU normalization source 行结尾漂移")
        byte_end = byte_start + len(encoded_line)
        records.append({
            "byte_end": byte_end,
            "byte_start": byte_start,
            "encoded": encoded_line,
            "line_ordinal": ordinal,
            "line_sha256": _sha256(encoded_line),
            "text": text[:-1],
        })
        byte_start = byte_end
    if byte_start != len(payload):
        raise BroadQaExternalDataError(
            "ICU normalization source 物理行覆盖漂移")
    return tuple(records)


def _statement_chunks(
        lines: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按注释和分号形成 statement，支持跨物理行且失败关闭。"""
    chunks = []
    active_lines = []
    active_parts = []
    for line in lines:
        text = str(line["text"])
        body, _marker, _comment = text.partition("#")
        content = body.strip()
        if not content:
            if active_lines:
                active_lines.append(line)
            continue
        if "'" in content or "\\" in content:
            raise BroadQaExternalDataError(
                "ICU normalization source 出现未支持的 quoted/escaped syntax")
        if content.count(";") > 1 or (";" in content
                                      and not content.endswith(";")):
            raise BroadQaExternalDataError(
                "ICU normalization source statement 分号位置非法")
        active_lines.append(line)
        active_parts.append(content)
        if content.endswith(";"):
            normalized = "".join(active_parts)[:-1].strip()
            if not normalized:
                raise BroadQaExternalDataError(
                    "ICU normalization source 空 statement")
            physical = [{
                "byte_end": int(item["byte_end"]),
                "byte_start": int(item["byte_start"]),
                "line_ordinal": int(item["line_ordinal"]),
                "line_sha256": str(item["line_sha256"]),
            } for item in active_lines]
            chunks.append({
                "byte_end": int(active_lines[-1]["byte_end"]),
                "byte_start": int(active_lines[0]["byte_start"]),
                "line_end_ordinal": int(active_lines[-1]["line_ordinal"]),
                "line_start_ordinal": int(active_lines[0]["line_ordinal"]),
                "normalized_statement": normalized,
                "physical_lines": physical,
                "statement_sha256": _sha256(b"".join(
                    item["encoded"] for item in active_lines)),
            })
            active_lines = []
            active_parts = []
    if active_lines or active_parts:
        raise BroadQaExternalDataError(
            "ICU normalization source 存在未闭合 statement")
    return tuple(chunks)


def _parse_variable(chunk: dict[str, object]) -> dict[str, object]:
    """解析本来源使用的简单 UnicodeSet 变量定义。"""
    statement = str(chunk["normalized_statement"])
    if statement.count("=") != 1:
        raise BroadQaExternalDataError("ICU normalization variable 非单一赋值")
    name, expression = (value.strip() for value in statement.split("="))
    if (not name.startswith("$") or not name[1:].isidentifier()
            or len(expression) < 2 or expression[0] != "["
            or expression[-1] != "]" or any(
                token in expression[1:-1] for token in "-^&$[]{}")):
        raise BroadQaExternalDataError(
            "ICU normalization variable syntax 未支持")
    members = list(expression[1:-1])
    if not members or len(set(members)) != len(members):
        raise BroadQaExternalDataError(
            "ICU normalization variable members 非法")
    return {
        **{key: chunk[key] for key in (
            "byte_end", "byte_start", "line_end_ordinal",
            "line_start_ordinal", "physical_lines", "statement_sha256")},
        "expression": expression,
        "format_version": 1,
        "members": members,
        "record_kind": NORMALIZATION_ICU_VARIABLE_KIND,
        "variable_name": name,
    }


def _split_context(value: str) -> tuple[str, str, str]:
    """把 ICU input side 分成 ante context、key 和 post context。"""
    if value.count("{") > 1 or value.count("}") > 1:
        raise BroadQaExternalDataError("ICU normalization context 重复")
    if "{" not in value and "}" not in value:
        return "", value.strip(), ""
    if "{" in value and "}" in value:
        if value.index("{") > value.index("}"):
            raise BroadQaExternalDataError("ICU normalization context 顺序非法")
        ante, remainder = value.split("{", 1)
        key, post = remainder.split("}", 1)
        return ante.strip(), key.strip(), post.strip()
    if "{" in value:
        ante, key = value.split("{", 1)
        return ante.strip(), key.strip(), ""
    key, post = value.split("}", 1)
    return "", key.strip(), post.strip()


def _variable_references(value: str) -> tuple[str, ...]:
    """从有限 ICU expression 中提取完整 variable 名称。"""
    references = []
    position = 0
    while "$" in value[position:]:
        position = value.index("$", position)
        end = position + 1
        while end < len(value) and (
                value[end].isalnum() or value[end] == "_"):
            end += 1
        reference = value[position:end]
        if reference == "$":
            raise BroadQaExternalDataError(
                "ICU normalization variable reference 非法")
        references.append(reference)
        position = end
    return tuple(sorted(set(references)))


def _validate_expression(value: str, *, allow_context: bool) -> None:
    """核验当前来源实际使用的有限规则表达式。"""
    if not value:
        raise BroadQaExternalDataError("ICU normalization rule side 为空")
    forbidden = "()|*+?@&:"
    if any(token in value for token in forbidden):
        raise BroadQaExternalDataError(
            "ICU normalization rule 出现未支持表达式")
    if not allow_context and any(token in value for token in "{}"):
        raise BroadQaExternalDataError(
            "ICU normalization output side 不得包含 context")
    cursor = 0
    while cursor < len(value):
        character = value[cursor]
        if character == "$":
            cursor += 1
            start = cursor
            while cursor < len(value) and (
                    value[cursor].isalnum() or value[cursor] == "_"):
                cursor += 1
            if cursor == start:
                raise BroadQaExternalDataError(
                    "ICU normalization variable reference 非法")
            continue
        if character == "[":
            end = value.find("]", cursor + 1)
            if end < 0 or any(token in value[cursor + 1:end]
                              for token in "-^&$[{}"):
                raise BroadQaExternalDataError(
                    "ICU normalization UnicodeSet syntax 未支持")
            cursor = end + 1
            continue
        if character == "]":
            raise BroadQaExternalDataError(
                "ICU normalization UnicodeSet 未配对")
        cursor += 1


def _parse_rule(
        chunk: dict[str, object],
        *,
        variables: dict[str, tuple[str, ...]],
        ) -> dict[str, object]:
    """解析一条规则并显式投影 ICU reverse/T2S 资格。"""
    statement = str(chunk["normalized_statement"])
    present = [arrow for arrow in NORMALIZATION_ICU_ARROWS
               if arrow in statement]
    if len(present) != 1 or statement.count(present[0]) != 1:
        raise BroadQaExternalDataError(
            "ICU normalization rule arrow 非唯一")
    arrow = present[0]
    left, right = (value.strip() for value in statement.split(arrow))
    _validate_expression(left, allow_context=True)
    _validate_expression(right, allow_context=False)
    ante, key, post = _split_context(left)
    if not key:
        raise BroadQaExternalDataError("ICU normalization rule key 为空")
    referenced = tuple(sorted(set(
        _variable_references(left) + _variable_references(right))))
    # 当前固定来源只出现完整变量名；独立核验防止把未知变量当普通文本。
    if any(reference not in variables for reference in referenced):
        raise BroadQaExternalDataError(
            "ICU normalization rule 引用未知 variable")
    has_context = bool(ante or post)
    t2s_eligible = int(arrow in ("←", "↔") and not has_context)
    return {
        **{key_name: chunk[key_name] for key_name in (
            "byte_end", "byte_start", "line_end_ordinal",
            "line_start_ordinal", "physical_lines", "statement_sha256")},
        "ante_context": ante,
        "arrow": arrow,
        "format_version": 1,
        "forward_input_key": key,
        "forward_output": right,
        "has_context": int(has_context),
        "post_context": post,
        "record_kind": NORMALIZATION_ICU_RULE_KIND,
        "referenced_variables": list(referenced),
        "t2s_expected_output": key if t2s_eligible else "",
        "t2s_input": right if t2s_eligible else "",
        "t2s_reverse_eligible": t2s_eligible,
    }


def parse_normalization_icu_source(
        payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """严格解析 ICU 变量和规则，并返回确定性结构统计。"""
    if not isinstance(payload, bytes):
        raise BroadQaExternalDataError("ICU normalization payload 非 bytes")
    lines = _physical_lines(payload)
    chunks = _statement_chunks(lines)
    variables = []
    rules = []
    variable_members: dict[str, tuple[str, ...]] = {}
    for chunk in chunks:
        statement = str(chunk["normalized_statement"])
        if statement.startswith("$") and "=" in statement:
            record = _parse_variable(chunk)
            name = str(record["variable_name"])
            if name in variable_members:
                raise BroadQaExternalDataError(
                    "ICU normalization variable 重复")
            variable_members[name] = tuple(record["members"])
            variables.append(record)
        else:
            rules.append(_parse_rule(chunk, variables=variable_members))
    if not variables or not rules:
        raise BroadQaExternalDataError("ICU normalization source 结构为空")
    rule_ids = [str(value["statement_sha256"]) for value in rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise BroadQaExternalDataError("ICU normalization rule identity 重复")
    arrow_counts = Counter(str(value["arrow"]) for value in rules)
    eligible = [value for value in rules if value["t2s_reverse_eligible"] == 1]
    inputs = [str(value["t2s_input"]) for value in eligible]
    if len(set(inputs)) != len(inputs):
        raise BroadQaExternalDataError(
            "ICU normalization reverse input 存在冲突或重复")
    summary = {
        "arrow_counts": {
            arrow: arrow_counts[arrow]
            for arrow in NORMALIZATION_ICU_ARROWS
        },
        "context_rule_count": sum(
            int(value["has_context"]) for value in rules),
        "forward_only_excluded_count": sum(
            value["arrow"] == "→" for value in rules),
        "physical_line_count": len(lines),
        "rule_count": len(rules),
        "statement_count": len(chunks),
        "t2s_reverse_eligible_count": len(eligible),
        "t2s_reverse_identity_count": sum(
            value["t2s_input"] == value["t2s_expected_output"]
            for value in eligible),
        "t2s_reverse_non_identity_count": sum(
            value["t2s_input"] != value["t2s_expected_output"]
            for value in eligible),
        "variable_count": len(variables),
    }
    return tuple(variables), tuple(rules), summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    """严格回读规范 JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        "ICU normalization JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "ICU normalization JSONL 不可读") from error
    return tuple(values)


def publish_normalization_icu_source_pack(
        *,
        run_root: str | Path,
        rule_source_path: str | Path,
        license_source_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """从显式 K 盘来源不可覆盖发布 ICU 独立 source pack。"""
    root = Path(run_root).resolve()
    rule_source = Path(rule_source_path).resolve()
    license_source = Path(license_source_path).resolve()
    target = Path(target_dir).resolve()
    if (not root.is_dir() or not rule_source.is_relative_to(root)
            or not license_source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "ICU normalization source/target 必须位于有效 run root")
    if target.exists():
        raise BroadQaExternalDataError("ICU normalization source target 已存在")
    try:
        rule_payload = rule_source.read_bytes()
        license_payload = license_source.read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "ICU normalization source 文件不可读") from error
    if (len(rule_payload) != NORMALIZATION_ICU_RULE_BYTES
            or _sha256(rule_payload) != NORMALIZATION_ICU_RULE_SHA256
            or len(license_payload) != NORMALIZATION_ICU_LICENSE_BYTES
            or _sha256(license_payload) != NORMALIZATION_ICU_LICENSE_SHA256):
        raise BroadQaExternalDataError("ICU normalization source identity 漂移")
    try:
        license_text = license_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BroadQaExternalDataError(
            "ICU normalization license 非 UTF-8") from error
    if ("UNICODE LICENSE V3" not in license_text
            or "SPDX-License-Identifier: Unicode-3.0" not in license_text):
        raise BroadQaExternalDataError("ICU normalization license 文本漂移")
    variables, rules, summary = parse_normalization_icu_source(rule_payload)
    target.mkdir(parents=True)
    rule_path = target / "Hans_Hant.txt"
    license_path = target / "LICENSE"
    variable_path = target / "variables.jsonl"
    rules_path = target / "rules.jsonl"
    rule_path.write_bytes(rule_payload)
    license_path.write_bytes(license_payload)
    _write_jsonl(variable_path, variables)
    _write_jsonl(rules_path, rules)
    files = []
    for path, role, count in (
            (license_path, "LICENSE", 0),
            (rule_path, "ICU_TRANSLITERATION_SOURCE", 0),
            (rules_path, "PARSED_RULE_RECORDS", len(rules)),
            (variable_path, "PARSED_VARIABLE_RECORDS", len(variables))):
        files.append({
            "bytes": path.stat().st_size,
            "record_count": count,
            "relative_path": path.relative_to(target).as_posix(),
            "role": role,
            "sha256": _sha256(path.read_bytes()),
        })
    manifest = {
        "artifact_kind": NORMALIZATION_ICU_SOURCE_PACK_KIND,
        "evaluation_record_count": 0,
        "evaluation_run_count": 0,
        "files": files,
        "format_version": 1,
        "learned_pack_read_count": 0,
        "license_id": NORMALIZATION_ICU_LICENSE_ID,
        "license_url": NORMALIZATION_ICU_LICENSE_URL,
        "production_enabled": 0,
        "release": NORMALIZATION_ICU_RELEASE,
        "repository_commit": NORMALIZATION_ICU_COMMIT,
        "rule_blob_sha": NORMALIZATION_ICU_BLOB_SHA,
        "rule_path": NORMALIZATION_ICU_RULE_PATH,
        "rule_url": NORMALIZATION_ICU_RULE_URL,
        "status": NORMALIZATION_ICU_SOURCE_STATUS,
        "summary": summary,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_icu_source_pack(
        target_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """严格回读所有物理文件并从原始 ICU 字节重派生 records。"""
    root = Path(target_dir).resolve()
    manifest_path = root / "manifest.json"
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "ICU normalization source manifest 不可读") from error
    expected = {
        "artifact_kind", "evaluation_record_count", "evaluation_run_count",
        "files", "format_version", "learned_pack_read_count", "license_id",
        "license_url", "production_enabled", "release", "repository_commit",
        "rule_blob_sha", "rule_path", "rule_url", "status", "summary",
    }
    fixed = {
        "artifact_kind": NORMALIZATION_ICU_SOURCE_PACK_KIND,
        "evaluation_record_count": 0,
        "evaluation_run_count": 0,
        "format_version": 1,
        "learned_pack_read_count": 0,
        "license_id": NORMALIZATION_ICU_LICENSE_ID,
        "license_url": NORMALIZATION_ICU_LICENSE_URL,
        "production_enabled": 0,
        "release": NORMALIZATION_ICU_RELEASE,
        "repository_commit": NORMALIZATION_ICU_COMMIT,
        "rule_blob_sha": NORMALIZATION_ICU_BLOB_SHA,
        "rule_path": NORMALIZATION_ICU_RULE_PATH,
        "rule_url": NORMALIZATION_ICU_RULE_URL,
        "status": NORMALIZATION_ICU_SOURCE_STATUS,
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected
            or canonical_json_line(manifest) != payload
            or any(not _strict_equal(manifest[key], value)
                   for key, value in fixed.items())
            or not isinstance(manifest["files"], list)):
        raise BroadQaExternalDataError(
            "ICU normalization source manifest 漂移")
    expected_paths = ["LICENSE", "Hans_Hant.txt", "rules.jsonl", "variables.jsonl"]
    if [item.get("relative_path") for item in manifest["files"]] != expected_paths:
        raise BroadQaExternalDataError("ICU normalization file inventory 漂移")
    for item in manifest["files"]:
        path = (root / item.get("relative_path", "")).resolve()
        try:
            file_payload = path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                "ICU normalization source artifact 不可读") from error
        if (not path.is_relative_to(root)
                or set(item) != {"bytes", "record_count", "relative_path",
                                 "role", "sha256"}
                or type(item["bytes"]) is not int
                or type(item["record_count"]) is not int
                or item["bytes"] != len(file_payload)
                or item["sha256"] != _sha256(file_payload)):
            raise BroadQaExternalDataError(
                "ICU normalization source artifact commitment 漂移")
    rule_payload = (root / "Hans_Hant.txt").read_bytes()
    license_payload = (root / "LICENSE").read_bytes()
    if (len(rule_payload) != NORMALIZATION_ICU_RULE_BYTES
            or _sha256(rule_payload) != NORMALIZATION_ICU_RULE_SHA256
            or len(license_payload) != NORMALIZATION_ICU_LICENSE_BYTES
            or _sha256(license_payload) != NORMALIZATION_ICU_LICENSE_SHA256):
        raise BroadQaExternalDataError("ICU normalization frozen source 漂移")
    derived_variables, derived_rules, summary = parse_normalization_icu_source(
        rule_payload)
    stored_variables = _read_jsonl(root / "variables.jsonl")
    stored_rules = _read_jsonl(root / "rules.jsonl")
    if (stored_variables != derived_variables or stored_rules != derived_rules
            or not _strict_equal(manifest["summary"], summary)
            or manifest["files"][2]["record_count"] != len(derived_rules)
            or manifest["files"][3]["record_count"] != len(derived_variables)):
        raise BroadQaExternalDataError(
            "ICU normalization parsed records/source 漂移")
    return ({**manifest, "manifest_sha256": _sha256(payload)},
            stored_variables, stored_rules)


def main(argv: list[str] | None = None) -> int:
    """发布或严格回读 ICU normalization source pack。"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--run-root", required=True)
    publish.add_argument("--rule-source-path", required=True)
    publish.add_argument("--license-source-path", required=True)
    publish.add_argument("--target-dir", required=True)
    read = subparsers.add_parser("read")
    read.add_argument("--target-dir", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "publish":
        report = publish_normalization_icu_source_pack(
            run_root=arguments.run_root,
            rule_source_path=arguments.rule_source_path,
            license_source_path=arguments.license_source_path,
            target_dir=arguments.target_dir,
        )
    else:
        report, _, _ = read_normalization_icu_source_pack(
            arguments.target_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_ICU_ARROWS",
    "NORMALIZATION_ICU_LICENSE_SHA256",
    "NORMALIZATION_ICU_RULE_SHA256",
    "NORMALIZATION_ICU_SOURCE_PACK_KIND",
    "NORMALIZATION_ICU_SOURCE_STATUS",
    "parse_normalization_icu_source",
    "publish_normalization_icu_source_pack",
    "read_normalization_icu_source_pack",
]
