"""派生 recovery-v7 VS Code 官方英文 source binding 可行性。

TypeScript AST 只负责提取 ``module + localization key + English message``。
本模块把它与冻结的本地化 pair 对齐，并只发布哈希承诺、源码 blob 身份与
aggregate；英文、中文和源码表面始终只存在于派生内存中。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
from zipfile import BadZipFile, ZipFile

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


VSCODE_ENGLISH_SOURCE_FILE_KIND = (
    "NORMALIZATION_RECOVERY_V7_VSCODE_ENGLISH_SOURCE_FILE_V1")
VSCODE_ENGLISH_MAIN_BINDING_KIND = (
    "NORMALIZATION_RECOVERY_V7_VSCODE_ENGLISH_MAIN_BINDING_V1")
VSCODE_ENGLISH_PROPOSAL_BINDING_KIND = (
    "NORMALIZATION_RECOVERY_V7_VSCODE_ENGLISH_PROPOSAL_BINDING_V1")
VSCODE_ENGLISH_BINDING_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V7_VSCODE_ENGLISH_BINDING_CENSUS_V1")

MAIN_TRANSLATION_PATH = "translations/main.i18n.json"
BINDING_UNIQUE = "UNIQUE_OFFICIAL_ENGLISH_SOURCE_BINDING"
BINDING_CONFLICT = "CONFLICTING_OFFICIAL_ENGLISH_SOURCE_BINDING"
BINDING_MISSING = "MISSING_OFFICIAL_ENGLISH_SOURCE_BINDING"
BINDING_OUTSIDE_SELECTION = "OUTSIDE_PINNED_MAIN_SOURCE_SELECTION"
BINDING_NOT_VSCODE = "NOT_VSCODE_PROPOSAL"

_ARCHIVE_MEMBER_MAX = 30_000
_ARCHIVE_UNCOMPRESSED_MAX = 512 * 1024 * 1024
_SELECTED_FILE_MAX = 16 * 1024 * 1024


def _sha256(payload: bytes) -> str:
    """返回表面、记录或集合的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    """返回逐字节源码的 Git blob SHA-1。"""
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\x00"
    return hashlib.sha1(prefix + payload).hexdigest()


def _record_id(value: dict[str, object]) -> str:
    """从完整规范 identity 构造稳定记录 id。"""
    return _sha256(canonical_json_bytes(value))


def _text_sha256(value: str) -> str:
    """哈希内存文本，不把文本写入派生 artifact。"""
    return _sha256(value.encode("utf-8"))


def inspect_vscode_typescript_source(
        *,
        archive_path: str | Path,
        source_root: str | Path,
        archive_prefix: str,
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """安全核对 source ZIP 与展开的 ``src/**/*.ts|tsx`` 逐字节一致。"""
    archive_file = Path(archive_path).resolve()
    root = Path(source_root).resolve()
    if (not archive_file.is_file() or not root.is_dir()
            or not archive_prefix or not archive_prefix.endswith("/")):
        raise BroadQaExternalDataError(
            "VS Code English source archive/root 非法")
    selected: list[tuple[object, str]] = []
    names = set()
    selected_casefold = set()
    total_bytes = 0
    unselected_symlink_count = 0
    try:
        with ZipFile(archive_file) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > _ARCHIVE_MEMBER_MAX:
                raise BroadQaExternalDataError(
                    "VS Code English source archive member 数非法")
            for info in infos:
                name = info.filename
                pure = PurePosixPath(name)
                mode = (info.external_attr >> 16) & 0o170000
                if (not name or "\\" in name or pure.is_absolute()
                        or ".." in pure.parts or ":" in name
                        or name in names):
                    raise BroadQaExternalDataError(
                        "VS Code English source archive member 非法")
                names.add(name)
                total_bytes += info.file_size
                if total_bytes > _ARCHIVE_UNCOMPRESSED_MAX:
                    raise BroadQaExternalDataError(
                        "VS Code English source archive 解压预算超限")
                allowed_mode = mode in {
                    0, stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK}
                if not allowed_mode:
                    raise BroadQaExternalDataError(
                        "VS Code English source archive 文件类型非法")
                relative = name[len(archive_prefix):] \
                    if name.startswith(archive_prefix) else ""
                is_selected = (
                    relative.startswith("src/")
                    and relative.endswith((".ts", ".tsx"))
                    and not name.endswith("/"))
                if not is_selected:
                    unselected_symlink_count += int(mode == stat.S_IFLNK)
                    continue
                if (mode == stat.S_IFLNK
                        or info.file_size > _SELECTED_FILE_MAX):
                    raise BroadQaExternalDataError(
                        "VS Code English selected source member 非法")
                folded = relative.casefold()
                if folded in selected_casefold:
                    raise BroadQaExternalDataError(
                        "VS Code English selected source path 冲突")
                selected_casefold.add(folded)
                selected.append((info, relative))
            if not selected:
                raise BroadQaExternalDataError(
                    "VS Code English selected source 为空")
            disk_paths = set()
            for path in root.rglob("*"):
                if path.suffix not in {".ts", ".tsx"}:
                    continue
                if path.is_symlink() or not path.is_file():
                    raise BroadQaExternalDataError(
                        "VS Code English source-root 文件类型非法")
                disk_paths.add(path.relative_to(root).as_posix())
            expected_paths = {relative for _info, relative in selected}
            if disk_paths != expected_paths:
                raise BroadQaExternalDataError(
                    "VS Code English source-root inventory 漂移")
            records = []
            for info, relative in sorted(selected, key=lambda item: item[1]):
                archive_payload = archive.read(info)
                disk_payload = (root / PurePosixPath(relative)).read_bytes()
                if archive_payload != disk_payload:
                    raise BroadQaExternalDataError(
                        "VS Code English source-root bytes 漂移")
                try:
                    archive_payload.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise BroadQaExternalDataError(
                        "VS Code English selected source UTF-8 非法") from error
                module = relative[4:].rsplit(".", 1)[0]
                identity = {
                    "bytes": len(archive_payload),
                    "git_blob_sha1": _git_blob_sha1(archive_payload),
                    "module_path_sha256": _text_sha256(module),
                    "relative_path_sha256": _text_sha256(relative),
                    "sha256": _sha256(archive_payload),
                }
                records.append({
                    **identity,
                    "module": module,
                    "record_id": _record_id(identity),
                    "relative_path": relative,
                })
    except BroadQaExternalDataError:
        raise
    except (OSError, BadZipFile, KeyError, RuntimeError) as error:
        raise BroadQaExternalDataError(
            "VS Code English source archive 不可读") from error
    public_inventory = [{
        key: value for key, value in item.items()
        if key not in {"module", "relative_path"}
    } for item in records]
    summary = {
        "archive_member_count": len(infos),
        "archive_uncompressed_bytes": total_bytes,
        "selected_source_bytes": sum(
            int(item["bytes"]) for item in records),
        "selected_source_file_count": len(records),
        "selected_source_inventory_sha256": _sha256(
            canonical_json_bytes(public_inventory)),
        "unselected_symlink_count": unselected_symlink_count,
    }
    return tuple(records), summary


def run_vscode_typescript_ast_extractor(
        *,
        source_root: str | Path,
        parser_root: str | Path,
        source_files: tuple[dict[str, object], ...],
        node_executable: str | Path = "node",
        extractor_path: str | Path | None = None,
        timeout_seconds: int = 600,
        ) -> dict[str, object]:
    """以无 shell 子进程运行固定 TypeScript AST 提取器。"""
    root = Path(source_root).resolve()
    parser = Path(parser_root).resolve()
    extractor = Path(extractor_path).resolve() if extractor_path else (
        Path(__file__).with_name(
            "ph2_broad_qa_normalization_recovery_v7_"
            "vscode_english_binding_ast.mjs").resolve())
    node = shutil.which(str(node_executable))
    relative_paths = [str(item.get("relative_path", ""))
                      for item in source_files]
    if (not root.is_dir() or not parser.is_dir() or not extractor.is_file()
            or node is None or not relative_paths
            or len(set(relative_paths)) != len(relative_paths)
            or type(timeout_seconds) is not int or timeout_seconds <= 0):
        raise BroadQaExternalDataError(
            "VS Code English AST extractor input 非法")
    request = json.dumps(
        {"relative_paths": relative_paths},
        ensure_ascii=True, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    try:
        completed = subprocess.run(
            [node, str(extractor), str(parser), str(root)],
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BroadQaExternalDataError(
            "VS Code English AST extractor 执行失败") from error
    if completed.returncode != 0 or completed.stderr:
        raise BroadQaExternalDataError(
            "VS Code English AST extractor 返回失败")
    try:
        result = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "VS Code English AST extractor 输出非法") from error
    _validate_ast_result(result, source_files=source_files)
    return result


def _validate_ast_result(
        value: object,
        *,
        source_files: tuple[dict[str, object], ...],
        ) -> None:
    """核验 AST 输出 schema、路径、计数与 occurrence identity。"""
    keys = {
        "binding_key_count", "bindings", "call_count", "file_count",
        "parse_diagnostics", "typescript_version", "unsupported",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise BroadQaExternalDataError("VS Code English AST schema 漂移")
    paths = {str(item.get("relative_path")): str(item.get("module"))
             for item in source_files}
    bindings = value["bindings"]
    diagnostics = value["parse_diagnostics"]
    unsupported = value["unsupported"]
    integers = (
        value["binding_key_count"], value["call_count"], value["file_count"])
    if (any(type(item) is not int or item < 0 for item in integers)
            or value["file_count"] != len(source_files)
            or not isinstance(value["typescript_version"], str)
            or not value["typescript_version"]
            or not isinstance(bindings, list)
            or not isinstance(diagnostics, list)
            or not isinstance(unsupported, list)
            or value["call_count"] != len(bindings) + len(unsupported)):
        raise BroadQaExternalDataError("VS Code English AST counts 非法")
    seen = set()
    binding_keys = set()
    for item in bindings:
        required = {
            "callee", "key", "message", "module", "position",
            "relative_path",
        }
        if (not isinstance(item, dict) or set(item) != required
                or item["callee"] not in {"localize", "localize2"}
                or any(not isinstance(item[key], str) or not item[key]
                       for key in ("key", "message", "module",
                                   "relative_path"))
                or type(item["position"]) is not int
                or item["position"] < 0
                or paths.get(item["relative_path"]) != item["module"]):
            raise BroadQaExternalDataError(
                "VS Code English AST binding 非法")
        identity = (
            item["relative_path"], item["position"], item["callee"],
            item["key"], item["message"])
        if identity in seen:
            raise BroadQaExternalDataError(
                "VS Code English AST occurrence 重复")
        seen.add(identity)
        binding_keys.add((item["module"], item["key"]))
    if value["binding_key_count"] != len(binding_keys):
        raise BroadQaExternalDataError(
            "VS Code English AST binding key census 漂移")
    for rows, label in ((diagnostics, "diagnostic"),
                        (unsupported, "unsupported")):
        for item in rows:
            if (not isinstance(item, dict)
                    or not isinstance(item.get("relative_path"), str)
                    or item["relative_path"] not in paths
                    or type(item.get("position")) is not int):
                raise BroadQaExternalDataError(
                    f"VS Code English AST {label} 非法")


def _public_source_files(
        values: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """移除源码路径表面，只保留文件与模块哈希 identity。"""
    output = []
    for item in values:
        identity = {
            "bytes": item["bytes"],
            "git_blob_sha1": item["git_blob_sha1"],
            "module_path_sha256": item["module_path_sha256"],
            "relative_path_sha256": item["relative_path_sha256"],
            "sha256": item["sha256"],
        }
        output.append({
            **identity,
            "format_version": 1,
            "record_id": item["record_id"],
            "record_kind": VSCODE_ENGLISH_SOURCE_FILE_KIND,
        })
    output.sort(key=lambda item: str(item["record_id"]))
    return tuple(output)


def transient_vscode_english_source_by_pair(
        *,
        source_files: tuple[dict[str, object], ...],
        ast_result: dict[str, object],
        vscode_pairs: tuple[dict[str, object], ...],
        ) -> dict[str, str]:
    """返回 main pair 到唯一官方英文 message 的内存映射。"""
    _validate_ast_result(ast_result, source_files=source_files)
    if ast_result["parse_diagnostics"] or ast_result["unsupported"]:
        raise BroadQaExternalDataError(
            "VS Code English AST 存在 diagnostic 或动态 localize")
    calls: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in ast_result["bindings"]:
        calls[(str(item["module"]), str(item["key"]))].add(
            str(item["message"]))
    values = {}
    for pair in vscode_pairs:
        if (not isinstance(pair, dict)
                or pair.get("training_eligible") != 1
                or pair.get("translation_relative_path")
                != MAIN_TRANSLATION_PATH):
            continue
        pair_id = pair.get("pair_id")
        path = pair.get("json_path")
        if (not isinstance(pair_id, str) or len(pair_id) != 64
                or not isinstance(path, list) or len(path) != 2
                or any(not isinstance(item, str) or not item for item in path)):
            raise BroadQaExternalDataError(
                "VS Code English transient pair identity 非法")
        messages = calls.get((path[0], path[1]), set())
        if len(messages) == 1:
            values[pair_id] = next(iter(messages))
    return values


def derive_vscode_english_binding_feasibility(
        *,
        source_files: tuple[dict[str, object], ...],
        ast_result: dict[str, object],
        vscode_pairs: tuple[dict[str, object], ...],
        proposals: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """对齐官方英文 AST 与本地化 pair，并形成非授权 feasibility。"""
    _validate_ast_result(ast_result, source_files=source_files)
    if ast_result["parse_diagnostics"] or ast_result["unsupported"]:
        raise BroadQaExternalDataError(
            "VS Code English AST 存在 diagnostic 或动态 localize")
    source_by_path = {
        str(item["relative_path"]): item for item in source_files}
    if len(source_by_path) != len(source_files):
        raise BroadQaExternalDataError(
            "VS Code English source file identity 重复")
    calls: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in ast_result["bindings"]:
        calls[(str(item["module"]), str(item["key"]))].append(item)
    eligible = []
    pair_by_id = {}
    for pair in vscode_pairs:
        eligible_flag = pair.get("training_eligible") \
            if isinstance(pair, dict) else None
        if (not isinstance(pair, dict) or type(eligible_flag) is not int
                or eligible_flag not in {0, 1}):
            raise BroadQaExternalDataError(
                "VS Code English localization pair 非法")
        pair_id = pair.get("pair_id")
        json_path_sha256 = pair.get("json_path_sha256")
        if (not isinstance(pair_id, str) or len(pair_id) != 64
                or not isinstance(json_path_sha256, str)
                or len(json_path_sha256) != 64
                or pair_id in pair_by_id):
            raise BroadQaExternalDataError(
                "VS Code English localization pair identity 非法")
        pair_by_id[pair_id] = pair
        if pair["training_eligible"] == 1:
            eligible.append(pair)
    main_pairs = [item for item in eligible
                  if item.get("translation_relative_path")
                  == MAIN_TRANSLATION_PATH]
    extension_pairs = [item for item in eligible
                       if item.get("translation_relative_path")
                       != MAIN_TRANSLATION_PATH]
    main_records = []
    outcome_counts = Counter()
    for pair in sorted(main_pairs, key=lambda item: str(item["pair_id"])):
        json_path = pair.get("json_path")
        if (not isinstance(json_path, list) or len(json_path) != 2
                or any(not isinstance(item, str) or not item
                       for item in json_path)):
            raise BroadQaExternalDataError(
                "VS Code English main localization path 非法")
        module, key = json_path
        occurrences = calls.get((module, key), [])
        messages = sorted({str(item["message"]) for item in occurrences})
        source_ids = sorted({
            str(source_by_path[str(item["relative_path"])]["record_id"])
            for item in occurrences})
        outcome = BINDING_MISSING if not messages else (
            BINDING_UNIQUE if len(messages) == 1 else BINDING_CONFLICT)
        outcome_counts[outcome] += 1
        identity = {
            "localization_key_sha256": _text_sha256(key),
            "module_path_sha256": _text_sha256(module),
            "pair_id": pair["pair_id"],
            "pair_json_path_sha256": pair["json_path_sha256"],
        }
        main_records.append({
            **identity,
            "binding_outcome": outcome,
            "english_message_sha256s": [
                _text_sha256(item) for item in messages],
            "format_version": 1,
            "localize_call_occurrence_count": len(occurrences),
            "record_id": _record_id(identity),
            "record_kind": VSCODE_ENGLISH_MAIN_BINDING_KIND,
            "source_file_ids": source_ids,
            "source_or_message_surface_published": 0,
        })
    main_records.sort(key=lambda item: str(item["record_id"]))
    main_by_pair = {str(item["pair_id"]): item for item in main_records}
    proposal_records = []
    proposal_outcomes = Counter()
    for proposal in proposals:
        family = proposal.get("held_out_source_family")
        pair_id = proposal.get("source_pair_id")
        observation_id = proposal.get("held_out_observation_id")
        if (not isinstance(family, str) or not family
                or not isinstance(pair_id, str) or len(pair_id) != 64
                or not isinstance(observation_id, str) or not observation_id):
            raise BroadQaExternalDataError(
                "VS Code English proposal identity 非法")
        if family == VSCODE_SOURCE_FAMILY:
            binding = main_by_pair.get(pair_id)
            if binding is None:
                raise BroadQaExternalDataError(
                    "VS Code English proposal 未绑定 main pair")
            outcome = str(binding["binding_outcome"])
            binding_record_id = str(binding["record_id"])
        else:
            outcome = BINDING_NOT_VSCODE
            binding_record_id = ""
        proposal_outcomes[outcome] += 1
        identity = {
            "held_out_observation_id_sha256": _text_sha256(observation_id),
            "held_out_source_family": family,
            "source_pair_id": pair_id,
        }
        proposal_records.append({
            **identity,
            "binding_outcome": outcome,
            "binding_record_id": binding_record_id,
            "format_version": 1,
            "record_id": _record_id(identity),
            "record_kind": VSCODE_ENGLISH_PROPOSAL_BINDING_KIND,
            "source_binding_authorizes_transformation": 0,
        })
    proposal_records.sort(key=lambda item: str(item["record_id"]))
    census = ({
        "binding_conflict_count": outcome_counts[BINDING_CONFLICT],
        "binding_missing_count": outcome_counts[BINDING_MISSING],
        "binding_unique_count": outcome_counts[BINDING_UNIQUE],
        "eligible_pair_count": len(main_pairs),
        "format_version": 1,
        "outside_source_selection_count": 0,
        "record_kind": VSCODE_ENGLISH_BINDING_CENSUS_KIND,
        "scope": "MAIN_I18N_PINNED_SRC_TYPESCRIPT",
    }, {
        "binding_conflict_count": 0,
        "binding_missing_count": 0,
        "binding_unique_count": 0,
        "eligible_pair_count": len(extension_pairs),
        "format_version": 1,
        "outside_source_selection_count": len(extension_pairs),
        "record_kind": VSCODE_ENGLISH_BINDING_CENSUS_KIND,
        "scope": "EXTENSION_I18N_OUTSIDE_PINNED_SRC_SELECTION",
    })
    census = tuple({
        **item,
        "record_id": _record_id({
            "eligible_pair_count": item["eligible_pair_count"],
            "scope": item["scope"],
        }),
    } for item in census)
    public_sources = _public_source_files(source_files)
    summary = {
        "ast_binding_key_count": ast_result["binding_key_count"],
        "ast_call_count": ast_result["call_count"],
        "ast_parse_diagnostic_count": 0,
        "ast_unsupported_count": 0,
        "capability_outcome": (
            "PASS_OFFICIAL_ENGLISH_SOURCE_BINDING_FEASIBILITY_"
            "NOT_TRANSFORMATION_AUTHORIZATION"),
        "eligible_extension_pair_count": len(extension_pairs),
        "eligible_main_pair_count": len(main_pairs),
        "main_binding_conflict_count": outcome_counts[BINDING_CONFLICT],
        "main_binding_missing_count": outcome_counts[BINDING_MISSING],
        "main_binding_unique_count": outcome_counts[BINDING_UNIQUE],
        "main_binding_record_set_sha256": _sha256(canonical_json_bytes(
            [item["record_id"] for item in main_records])),
        "proposal_binding_outcome_counts": {
            key: proposal_outcomes[key] for key in sorted(proposal_outcomes)},
        "proposal_count": len(proposal_records),
        "proposal_record_set_sha256": _sha256(canonical_json_bytes(
            [item["record_id"] for item in proposal_records])),
        "source_binding_authorizes_transformation": 0,
        "source_file_record_count": len(public_sources),
        "source_file_record_set_sha256": _sha256(canonical_json_bytes(
            [item["record_id"] for item in public_sources])),
        "source_or_message_surface_published": 0,
        "typescript_version": ast_result["typescript_version"],
    }
    return (
        public_sources,
        tuple(main_records),
        tuple(proposal_records),
        census,
        summary,
    )


__all__ = [
    "BINDING_CONFLICT",
    "BINDING_MISSING",
    "BINDING_NOT_VSCODE",
    "BINDING_OUTSIDE_SELECTION",
    "BINDING_UNIQUE",
    "MAIN_TRANSLATION_PATH",
    "VSCODE_ENGLISH_BINDING_CENSUS_KIND",
    "VSCODE_ENGLISH_MAIN_BINDING_KIND",
    "VSCODE_ENGLISH_PROPOSAL_BINDING_KIND",
    "VSCODE_ENGLISH_SOURCE_FILE_KIND",
    "derive_vscode_english_binding_feasibility",
    "inspect_vscode_typescript_source",
    "run_vscode_typescript_ast_extractor",
    "transient_vscode_english_source_by_pair",
]
