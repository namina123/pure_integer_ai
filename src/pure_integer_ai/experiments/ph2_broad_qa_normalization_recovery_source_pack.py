"""冻结 normalization recovery v2 的 Firefox 中文本地化独立来源。

本模块使用官方 Fluent parser 解析固定 Git commit 的 ``zh-CN`` 与
``zh-TW``。source pack 只保存来源、结构和 provenance，不读取 learner、
既有 formal item、reserve 或 candidate，也不运行能力评测。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import PurePosixPath, Path
import zipfile

from fluent.syntax import FluentParser, ast

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_EVALUATION_SOURCE_PACK_V2")
NORMALIZATION_RECOVERY_SOURCE_STATUS = (
    "INDEPENDENT_FIREFOX_SOURCE_FROZEN_NOT_SELECTED_NOT_EVALUATED")
FIREFOX_PATTERN_PAIR_RECORD_KIND = "FIREFOX_L10N_PATTERN_PAIR_V1"

FIREFOX_L10N_REPOSITORY_URL = (
    "https://github.com/mozilla-l10n/firefox-l10n")
FIREFOX_L10N_COMMIT = "c176c2ccb293868c0e900c5f40b4506261542710"
FIREFOX_L10N_COMMIT_DATE = "2026-08-14T05:04:24Z"
FIREFOX_L10N_ROOT_TREE = "b93695fb11502984ca0b2a973dc0302acbe13c0d"
FIREFOX_L10N_LOCALE_TREES = {
    "zh-CN": "ca7ae8b68ff16452afcef62bcae31ee1bc865680",
    "zh-TW": "82b5213389c835a7d69423da6ebe85e480c4f408",
}
FIREFOX_L10N_ARCHIVE_NAME = (
    "firefox-l10n-c176c2ccb293868c-raw-v2.zip")
FIREFOX_L10N_ARCHIVE_BYTES = 1_788_589
FIREFOX_L10N_ARCHIVE_SHA256 = (
    "659d1cc432becf07b5d9d32f1c25c6c0b4d7a6f82b1934795f9ce735bd49346a")
FIREFOX_L10N_LICENSE_ID = "MPL-2.0"
FIREFOX_L10N_LICENSE_URL = (
    "https://raw.githubusercontent.com/mozilla-l10n/firefox-l10n/"
    f"{FIREFOX_L10N_COMMIT}/LICENSE")
FIREFOX_L10N_LICENSE_BLOB_SHA1 = (
    "a612ad9813b006ce81d1ee438dd784da99a54007")
FIREFOX_L10N_LICENSE_BYTES = 16_725
FIREFOX_L10N_LICENSE_SHA256 = (
    "1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5")
FIREFOX_L10N_ARCHIVE_FILE_COUNT = 728
FIREFOX_L10N_FTL_FILE_COUNTS = {"zh-CN": 242, "zh-TW": 243}
FIREFOX_L10N_PATTERN_COUNTS = {"zh-CN": 15_036, "zh-TW": 15_171}
FIREFOX_L10N_COMMON_PATTERN_COUNT = 15_036
FIREFOX_L10N_STRUCTURE_EQUAL_COUNT = 14_762
FIREFOX_L10N_PLAIN_PAIR_COUNT = 12_336

FLUENT_SYNTAX_VERSION = "0.19.0"
FLUENT_SYNTAX_WHEEL_SHA256 = (
    "b352b3475fac6c6ed5f06527921f432aac073d764445508ee5218aeccc7cc5c4")
TYPING_EXTENSIONS_VERSION = "4.12.2"
TYPING_EXTENSIONS_WHEEL_SHA256 = (
    "04e5ca0351e0f3f85c6853954072df659d0d13fac324d0072316b67d7794700d")

_LOCALES = ("zh-CN", "zh-TW")


def _sha256(payload: bytes) -> str:
    """返回来源或规范 artifact 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _git_object_sha1(kind: str, payload: bytes) -> str:
    """按 Git 对象编码计算 blob/tree SHA-1。"""
    header = kind.encode("ascii") + b" " + str(len(payload)).encode("ascii")
    return hashlib.sha1(header + b"\0" + payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _safe_member_name(value: str) -> str:
    """要求 ZIP member 为受限、安全的 POSIX 相对路径。"""
    if not isinstance(value, str) or not value or "\\" in value:
        raise BroadQaExternalDataError("Firefox archive member path 非法")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BroadQaExternalDataError("Firefox archive member path 逃逸")
    if path.parts[0] not in ("LICENSE", *_LOCALES):
        raise BroadQaExternalDataError("Firefox archive 含越界根成员")
    if path.parts[0] == "LICENSE" and len(path.parts) != 1:
        raise BroadQaExternalDataError("Firefox LICENSE member 非法")
    return path.as_posix()


def _archive_files(
        archive_payload: bytes,
        ) -> dict[str, tuple[bytes, int]]:
    """读取固定 archive，并保留 Git executable mode。"""
    if not isinstance(archive_payload, bytes):
        raise BroadQaExternalDataError("Firefox archive payload 非 bytes")
    files: dict[str, tuple[bytes, int]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
            for info in archive.infolist():
                name = _safe_member_name(info.filename.rstrip("/"))
                if info.flag_bits & 0x1:
                    raise BroadQaExternalDataError(
                        "Firefox archive 不得含加密 member")
                if info.is_dir():
                    continue
                if name in files:
                    raise BroadQaExternalDataError(
                        "Firefox archive member 重复")
                mode = (info.external_attr >> 16) & 0xFFFF
                files[name] = (archive.read(info), int(bool(mode & 0o111)))
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise BroadQaExternalDataError("Firefox archive 非法") from error
    if "LICENSE" not in files or any(
            not any(name.startswith(locale + "/") for name in files)
            for locale in _LOCALES):
        raise BroadQaExternalDataError("Firefox archive 来源根不完整")
    return files


def _tree_sha1(
        files: dict[str, tuple[bytes, int]],
        prefix: str,
        ) -> str:
    """从 archive member 重算指定 Git tree，不信任 ZIP 自报。"""
    direct: dict[str, tuple[bytes, int]] = {}
    directories = set()
    for path, value in files.items():
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):]
        if "/" in rest:
            directories.add(rest.split("/", 1)[0])
        elif rest:
            direct[rest] = value
    entries: list[tuple[bytes, bytes, bytes]] = []
    for name, (payload, executable) in direct.items():
        mode = b"100755" if executable else b"100644"
        entries.append((name.encode("utf-8"), mode, bytes.fromhex(
            _git_object_sha1("blob", payload))))
    for name in directories:
        child_sha = _tree_sha1(files, prefix + name + "/")
        entries.append(((name + "/").encode("utf-8"), b"40000",
                        bytes.fromhex(child_sha)))
    entries.sort(key=lambda item: item[0])
    payload = b"".join(
        mode + b" " + (name[:-1] if name.endswith(b"/") else name)
        + b"\0" + object_id
        for name, mode, object_id in entries)
    return _git_object_sha1("tree", payload)


def _ast_value(value: object, *, mask_text: bool) -> object:
    """移除 parser span，并可屏蔽自然语言文本以得到结构 identity。"""
    if isinstance(value, dict):
        node_type = value.get("type")
        result = {}
        for key in sorted(value):
            if key == "span":
                continue
            if mask_text and node_type == "TextElement" and key == "value":
                result[key] = "<TEXT>"
            else:
                result[key] = _ast_value(value[key], mask_text=mask_text)
        return result
    if isinstance(value, list):
        return [_ast_value(item, mask_text=mask_text) for item in value]
    return value


def _plain_text(pattern: ast.Pattern) -> str | None:
    """仅为纯 TextElement pattern 返回原始表面，不展平 Placeable。"""
    if not all(isinstance(item, ast.TextElement) for item in pattern.elements):
        return None
    return "".join(item.value for item in pattern.elements)


def _byte_offsets(text: str) -> tuple[int, ...]:
    """构造字符 offset 到 UTF-8 byte offset 的确定性映射。"""
    offsets = [0]
    total = 0
    for item in text:
        total += len(item.encode("utf-8"))
        offsets.append(total)
    return tuple(offsets)


def _locale_patterns(
        files: dict[str, tuple[bytes, int]],
        locale: str,
        ) -> tuple[dict[tuple[str, str, str, str], dict[str, object]], dict[str, int]]:
    """结构化解析单个 locale 的全部 Fluent message、term 与 attribute。"""
    parser = FluentParser(with_spans=True)
    patterns: dict[tuple[str, str, str, str], dict[str, object]] = {}
    counters: Counter[str] = Counter()
    prefix = locale + "/"
    for path in sorted(name for name in files
                       if name.startswith(prefix) and name.endswith(".ftl")):
        payload, _ = files[path]
        if payload.startswith(b"\xef\xbb\xbf"):
            raise BroadQaExternalDataError("Firefox FTL 不得含 UTF-8 BOM")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BroadQaExternalDataError("Firefox FTL 非 UTF-8") from error
        resource = parser.parse(text)
        offsets = _byte_offsets(text)
        counters["ftl_file_count"] += 1
        counters["ftl_bytes"] += len(payload)
        relative = path[len(prefix):]
        file_sha = _sha256(payload)
        file_blob = _git_object_sha1("blob", payload)
        for entry in resource.body:
            counters[type(entry).__name__] += 1
            if isinstance(entry, ast.Junk):
                raise BroadQaExternalDataError(
                    f"Firefox FTL parser Junk: {path}")
            if not isinstance(entry, (ast.Message, ast.Term)):
                continue
            entry_kind = "TERM" if isinstance(entry, ast.Term) else "MESSAGE"
            message_id = entry.id.name
            values = []
            if entry.value is not None:
                values.append(("", entry.value))
            values.extend((item.id.name, item.value)
                          for item in entry.attributes)
            for attribute_id, pattern in values:
                key = (relative, entry_kind, message_id, attribute_id)
                if key in patterns:
                    raise BroadQaExternalDataError(
                        "Firefox FTL pattern identity 重复")
                start = pattern.span.start
                end = pattern.span.end
                byte_start = offsets[start]
                byte_end = offsets[end]
                source_slice = payload[byte_start:byte_end]
                if source_slice != text[start:end].encode("utf-8"):
                    raise BroadQaExternalDataError(
                        "Firefox FTL span/byte 映射漂移")
                encoded_ast = canonical_json_bytes(_ast_value(
                    pattern.to_json(), mask_text=False))
                encoded_skeleton = canonical_json_bytes(_ast_value(
                    pattern.to_json(), mask_text=True))
                patterns[key] = {
                    "attribute_id": attribute_id,
                    "byte_end": byte_end,
                    "byte_start": byte_start,
                    "entry_kind": entry_kind,
                    "file_git_blob_sha1": file_blob,
                    "file_sha256": file_sha,
                    "locale": locale,
                    "message_id": message_id,
                    "pattern_ast_sha256": _sha256(encoded_ast),
                    "pattern_skeleton_sha256": _sha256(encoded_skeleton),
                    "relative_path": relative,
                    "source_slice_sha256": _sha256(source_slice),
                    "surface_text": _plain_text(pattern),
                }
                counters["pattern_count"] += 1
                if patterns[key]["surface_text"] is not None:
                    counters["plain_pattern_count"] += 1
    return patterns, dict(sorted(counters.items()))


def parse_normalization_recovery_firefox_archive(
        archive_payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """解析固定 Firefox archive，并生成文件与跨 locale pattern 对齐记录。"""
    files = _archive_files(archive_payload)
    file_records = tuple({
        "bytes": len(payload),
        "executable": executable,
        "git_blob_sha1": _git_object_sha1("blob", payload),
        "relative_path": path,
        "sha256": _sha256(payload),
    } for path, (payload, executable) in sorted(files.items()))
    locale_patterns = {}
    locale_summaries = {}
    locale_trees = {}
    for locale in _LOCALES:
        patterns, summary = _locale_patterns(files, locale)
        locale_patterns[locale] = patterns
        locale_summaries[locale] = summary
        locale_trees[locale] = _tree_sha1(files, locale + "/")
    common = sorted(set(locale_patterns["zh-CN"]).intersection(
        locale_patterns["zh-TW"]))
    pair_records = []
    structure_equal_count = 0
    plain_pair_count = 0
    for key in common:
        cn = locale_patterns["zh-CN"][key]
        tw = locale_patterns["zh-TW"][key]
        structure_equal = int(
            cn["pattern_skeleton_sha256"] == tw["pattern_skeleton_sha256"])
        plain_pair = int(structure_equal == 1
                         and cn["surface_text"] is not None
                         and tw["surface_text"] is not None)
        structure_equal_count += structure_equal
        plain_pair_count += plain_pair
        identity = "\0".join(key)
        pair_records.append({
            "attribute_id": key[3],
            "entry_kind": key[1],
            "format_version": 1,
            "message_id": key[2],
            "pair_id": _sha256(
                ("FIREFOX_L10N_PATTERN_PAIR_V1\0" + identity).encode("utf-8")),
            "plain_pair_eligible": plain_pair,
            "record_kind": FIREFOX_PATTERN_PAIR_RECORD_KIND,
            "relative_path": key[0],
            "structure_equal": structure_equal,
            "zh_cn": cn,
            "zh_tw": tw,
        })
    license_payload, _ = files["LICENSE"]
    summary = {
        "archive_file_count": len(file_records),
        "common_pattern_count": len(pair_records),
        "license_bytes": len(license_payload),
        "license_git_blob_sha1": _git_object_sha1("blob", license_payload),
        "license_sha256": _sha256(license_payload),
        "locale_summaries": locale_summaries,
        "locale_trees": locale_trees,
        "plain_pair_count": plain_pair_count,
        "structure_equal_count": structure_equal_count,
    }
    return file_records, tuple(pair_records), summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格回读规范 JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(f"{label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} JSONL 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """返回 manifest 文件身份。"""
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
        parser_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造零训练、零评测读取的恢复来源 manifest。"""
    return {
        "archive_acquisition": {
            "command_contract": (
                "git -c core.autocrlf=false archive --format=zip "
                "<commit> LICENSE zh-CN zh-TW"),
            "commit": FIREFOX_L10N_COMMIT,
            "commit_date": FIREFOX_L10N_COMMIT_DATE,
            "repository_url": FIREFOX_L10N_REPOSITORY_URL,
            "root_tree": FIREFOX_L10N_ROOT_TREE,
        },
        "artifact_kind": NORMALIZATION_RECOVERY_SOURCE_PACK_KIND,
        "candidate_pack_read_count": 0,
        "evaluation_run_count": 0,
        "files": files,
        "format_version": 2,
        "learned_pack_read_count": 0,
        "license": {
            "attribution": "Mozilla contributors; fixed Firefox l10n commit",
            "blob_sha1": FIREFOX_L10N_LICENSE_BLOB_SHA1,
            "license_id": FIREFOX_L10N_LICENSE_ID,
            "license_url": FIREFOX_L10N_LICENSE_URL,
            "sha256": FIREFOX_L10N_LICENSE_SHA256,
        },
        "mastery_claimed": 0,
        "parser": {
            "fluent_syntax_version": FLUENT_SYNTAX_VERSION,
            "fluent_syntax_wheel_sha256": FLUENT_SYNTAX_WHEEL_SHA256,
            "typing_extensions_version": TYPING_EXTENSIONS_VERSION,
            "typing_extensions_wheel_sha256": (
                TYPING_EXTENSIONS_WHEEL_SHA256),
        },
        "parser_summary": parser_summary,
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "recovery_training_source_read_count": 0,
        "reserve_payload_read_count": 0,
        "source_roster": {
            "formal_source_keys": ["MOZILLA_FIREFOX_L10N"],
            "wikidata_lexemes": (
                "EXCLUDED_SAME_WIKIMEDIA_ECOSYSTEM_AS_CONSUMED_MEDIAWIKI"),
            "wiktionary": (
                "EXCLUDED_SAME_WIKIMEDIA_ECOSYSTEM_AS_CONSUMED_MEDIAWIKI"),
        },
        "status": NORMALIZATION_RECOVERY_SOURCE_STATUS,
        "teacher_api_llm_call_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery source run root 必须是 K 盘目录")
    return root


def _validate_official_source(
        archive_payload: bytes,
        summary: dict[str, object],
        ) -> None:
    """核验固定 archive、Git tree、许可和完整 parser 规模。"""
    if (len(archive_payload) != FIREFOX_L10N_ARCHIVE_BYTES
            or _sha256(archive_payload) != FIREFOX_L10N_ARCHIVE_SHA256
            or summary.get("archive_file_count")
            != FIREFOX_L10N_ARCHIVE_FILE_COUNT
            or summary.get("license_bytes") != FIREFOX_L10N_LICENSE_BYTES
            or summary.get("license_sha256") != FIREFOX_L10N_LICENSE_SHA256
            or summary.get("license_git_blob_sha1")
            != FIREFOX_L10N_LICENSE_BLOB_SHA1
            or not _strict_equal(
                summary.get("locale_trees"), FIREFOX_L10N_LOCALE_TREES)
            or summary.get("common_pattern_count")
            != FIREFOX_L10N_COMMON_PATTERN_COUNT
            or summary.get("structure_equal_count")
            != FIREFOX_L10N_STRUCTURE_EQUAL_COUNT
            or summary.get("plain_pair_count")
            != FIREFOX_L10N_PLAIN_PAIR_COUNT):
        raise BroadQaExternalDataError("Firefox official source identity 漂移")
    locale_summaries = summary.get("locale_summaries")
    if not isinstance(locale_summaries, dict):
        raise BroadQaExternalDataError("Firefox locale summary 缺失")
    for locale in _LOCALES:
        item = locale_summaries.get(locale)
        if (not isinstance(item, dict)
                or item.get("ftl_file_count")
                != FIREFOX_L10N_FTL_FILE_COUNTS[locale]
                or item.get("pattern_count")
                != FIREFOX_L10N_PATTERN_COUNTS[locale]):
            raise BroadQaExternalDataError("Firefox parser inventory 漂移")


def publish_normalization_recovery_source_pack(
        *,
        run_root: str | Path,
        archive_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 Firefox recovery evaluation source pack。"""
    root = _require_k_root(run_root)
    source = Path(archive_path).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_file() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError("normalization recovery source path 越界")
    if target.exists():
        raise BroadQaExternalDataError("normalization recovery source target 已存在")
    archive_payload = source.read_bytes()
    file_records, pair_records, summary = (
        parse_normalization_recovery_firefox_archive(archive_payload))
    _validate_official_source(archive_payload, summary)
    target.mkdir(parents=True)
    archive_target = target / FIREFOX_L10N_ARCHIVE_NAME
    with archive_target.open("xb") as handle:
        handle.write(archive_payload)
    file_path = target / "source-files.jsonl"
    pair_path = target / "pattern-pairs.jsonl"
    _write_jsonl(file_path, file_records)
    _write_jsonl(pair_path, pair_records)
    files = [
        _artifact(archive_target, role="FIREFOX_L10N_RAW_ARCHIVE", count=0),
        _artifact(file_path, role="FIREFOX_L10N_SOURCE_FILES",
                  count=len(file_records)),
        _artifact(pair_path, role="FIREFOX_L10N_PATTERN_PAIRS",
                  count=len(pair_records)),
    ]
    manifest = _manifest(files=files, parser_summary=summary)
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_recovery_source_pack(
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从固定 Git archive 重派生并严格回读 Firefox source pack。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded_manifest = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded_manifest)
        archive_payload = (root / FIREFOX_L10N_ARCHIVE_NAME).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery source pack 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization recovery source manifest 非规范")
    derived_files, derived_pairs, summary = (
        parse_normalization_recovery_firefox_archive(archive_payload))
    _validate_official_source(archive_payload, summary)
    stored_files = _read_jsonl(
        root / "source-files.jsonl", label="Firefox source files")
    stored_pairs = _read_jsonl(
        root / "pattern-pairs.jsonl", label="Firefox pattern pairs")
    if (not _strict_equal(stored_files, derived_files)
            or not _strict_equal(stored_pairs, derived_pairs)):
        raise BroadQaExternalDataError(
            "normalization recovery records/source 漂移")
    files = [
        _artifact(root / FIREFOX_L10N_ARCHIVE_NAME,
                  role="FIREFOX_L10N_RAW_ARCHIVE", count=0),
        _artifact(root / "source-files.jsonl",
                  role="FIREFOX_L10N_SOURCE_FILES", count=len(derived_files)),
        _artifact(root / "pattern-pairs.jsonl",
                  role="FIREFOX_L10N_PATTERN_PAIRS", count=len(derived_pairs)),
    ]
    expected = _manifest(files=files, parser_summary=summary)
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery source manifest 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded_manifest)},
        derived_files,
        derived_pairs,
    )


__all__ = [
    "FIREFOX_L10N_ARCHIVE_BYTES",
    "FIREFOX_L10N_ARCHIVE_NAME",
    "FIREFOX_L10N_ARCHIVE_SHA256",
    "FIREFOX_L10N_COMMIT",
    "FIREFOX_L10N_LICENSE_BYTES",
    "FIREFOX_L10N_LICENSE_SHA256",
    "FIREFOX_L10N_LOCALE_TREES",
    "NORMALIZATION_RECOVERY_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_SOURCE_STATUS",
    "parse_normalization_recovery_firefox_archive",
    "publish_normalization_recovery_source_pack",
    "read_normalization_recovery_source_pack",
]
