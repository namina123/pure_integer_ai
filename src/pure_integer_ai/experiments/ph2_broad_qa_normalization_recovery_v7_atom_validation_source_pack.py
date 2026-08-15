"""不可覆盖发布 Audacity atom-validation held-out 来源 pack。

publisher 从固定 partial Git checkout 请求三份已冻结 blob；selection 在 blob 读取前
已经由代码常量固定。pack 保存 K 盘 raw source 与无 translation 的 identity roster，
不生成 validation label JSONL，也不运行 proposal、authorization 或 scoring。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import polib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_source_records import (
    AUDACITY_LICENSE_PATH,
    AUDACITY_SOURCE_FILES,
    AUDACITY_TRANSLATION_LICENSE_EXPRESSION,
    parse_audacity_atom_validation_files,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "AUDACITY_ATOM_VALIDATION_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS = (
    "INDEPENDENT_ATOM_VALIDATION_SOURCE_FROZEN_"
    "LABEL_JSONL_NOT_MATERIALIZED")

AUDACITY_REPOSITORY_URL = "https://github.com/audacity/audacity.git"
AUDACITY_BRANCH = "master"
AUDACITY_COMMIT = "2f42f1c968ad15b5ab871f3bdf56249bd311a84e"
AUDACITY_ROOT_TREE = "c2de742ab2243273921c2dc9237d411c09cbc2cc"
AUDACITY_COMMIT_DATE = "2026-08-14T11:10:34+02:00"
AUDACITY_SOURCE_BLOBS = {
    "LICENSE.txt": "da463be6f109f2b2ba8540b84694c59cfe8d2c9f",
    "au3/locale/zh_CN.po": "01a2dbabd9893789c9a8f5287f3764af2c3fe875",
    "au3/locale/zh_TW.po": "8f457e59383404605da2a5956a92823d07dacce6",
}
AUDACITY_LICENSE_BYTES = 73_540
AUDACITY_LICENSE_SHA256 = (
    "1580ffb4a0c6bbb716324c645682964120eae418b9a1c51842cda140343cb139")
AUDACITY_SOURCE_FAMILY = "AUDACITY_PROJECT"
AUDACITY_SOURCE_POLICY_SCOPE = "AUDACITY_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"
POLIB_VERSION = "1.2.0"

_RAW_ROLE_BY_PATH = {
    "LICENSE.txt": "AUDACITY_LICENSE_TEXT",
    "au3/locale/zh_CN.po": "AUDACITY_ZH_CN_TRANSLATION_PO",
    "au3/locale/zh_TW.po": "AUDACITY_ZH_TW_TRANSLATION_PO",
}
_OUTPUT_FILES = (
    ("source-files.jsonl", "AUDACITY_ATOM_VALIDATION_SOURCE_FILES"),
    ("evaluation-inventory.identity.jsonl",
     "AUDACITY_ATOM_VALIDATION_IDENTITY_WITHOUT_LABELS"),
)


def _sha256(payload: bytes) -> str:
    """返回固定来源文件或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "Audacity atom-validation run root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入或输出位于显式 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"Audacity atom-validation {label} path 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个目录是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _git_text(root: Path, *arguments: str) -> str:
    """执行只读 Git 命令并返回严格 UTF-8 文本。"""
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            timeout=120,
        )
        return completed.stdout.decode("utf-8", errors="strict").strip()
    except (OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, UnicodeDecodeError) as error:
        raise BroadQaExternalDataError(
            "Audacity atom-validation Git identity 不可读") from error


def _git_blob(root: Path, relative_path: str) -> bytes:
    """从固定 commit 请求一份已冻结 Git blob。"""
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), "cat-file", "blob",
             f"{AUDACITY_COMMIT}:{relative_path}"),
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        raise BroadQaExternalDataError(
            "Audacity atom-validation Git blob 不可读") from error
    return completed.stdout


def _git_source_material(
        checkout_root: Path,
        ) -> tuple[dict[str, object], dict[str, bytes]]:
    """核对固定 checkout identity，并读取三份预选 blob。"""
    if not checkout_root.is_dir():
        raise BroadQaExternalDataError(
            "Audacity atom-validation checkout 不存在")
    identity = {
        "branch": AUDACITY_BRANCH,
        "commit": _git_text(checkout_root, "rev-parse", "HEAD"),
        "commit_date": _git_text(
            checkout_root, "show", "-s", "--format=%cI", "HEAD"),
        "remote": _git_text(
            checkout_root, "config", "--get", "remote.origin.url"),
        "root_tree": _git_text(
            checkout_root, "rev-parse", "HEAD^{tree}"),
    }
    if (identity["commit"] != AUDACITY_COMMIT
            or identity["commit_date"] != AUDACITY_COMMIT_DATE
            or identity["root_tree"] != AUDACITY_ROOT_TREE
            or identity["remote"] not in {
                AUDACITY_REPOSITORY_URL,
                AUDACITY_REPOSITORY_URL.removesuffix(".git"),
            }):
        raise BroadQaExternalDataError(
            "Audacity atom-validation checkout identity 漂移")
    identity["remote"] = AUDACITY_REPOSITORY_URL
    for relative_path, expected_blob in AUDACITY_SOURCE_BLOBS.items():
        if _git_text(
                checkout_root, "rev-parse",
                f"{AUDACITY_COMMIT}:{relative_path}") != expected_blob:
            raise BroadQaExternalDataError(
                "Audacity atom-validation source blob identity 漂移")
    files = {
        relative_path: _git_blob(checkout_root, relative_path)
        for relative_path in AUDACITY_SOURCE_FILES}
    license_payload = files[AUDACITY_LICENSE_PATH]
    if (len(license_payload) != AUDACITY_LICENSE_BYTES
            or _sha256(license_payload) != AUDACITY_LICENSE_SHA256):
        raise BroadQaExternalDataError(
            "Audacity atom-validation license identity 漂移")
    return identity, files


def _validate_frozen_raw_files(files: dict[str, bytes]) -> None:
    """核对 stored raw 仍是固定 commit 的三份 Git blob。"""
    if (set(files) != set(AUDACITY_SOURCE_BLOBS)
            or any(git_blob_sha1(files[path]) != expected_blob
                   for path, expected_blob in AUDACITY_SOURCE_BLOBS.items())):
        raise BroadQaExternalDataError(
            "Audacity atom-validation frozen raw blob 漂移")
    license_payload = files[AUDACITY_LICENSE_PATH]
    if (len(license_payload) != AUDACITY_LICENSE_BYTES
            or _sha256(license_payload) != AUDACITY_LICENSE_SHA256):
        raise BroadQaExternalDataError(
            "Audacity atom-validation license identity 漂移")


def _write_jsonl(
        path: Path,
        values: tuple[dict[str, object], ...],
        ) -> None:
    """独占写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(
        path: Path,
        *,
        label: str,
        ) -> tuple[dict[str, object], ...]:
    """读取规范 JSONL，并拒绝空行与非 object。"""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            f"Audacity atom-validation {label} 不可读") from error
    values = []
    for line in payload.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BroadQaExternalDataError(
                f"Audacity atom-validation {label} JSONL 非法") from error
        if not isinstance(value, dict) or canonical_json_line(value) != line:
            raise BroadQaExternalDataError(
                f"Audacity atom-validation {label} 非规范")
        values.append(value)
    if not values:
        raise BroadQaExternalDataError(
            f"Audacity atom-validation {label} 为空")
    return tuple(values)


def _artifact(
        path: Path,
        *,
        role: str,
        count: int,
        ) -> dict[str, object]:
    """形成一份物理文件 commitment。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _raw_artifact(root: Path, relative_path: str) -> dict[str, object]:
    """形成保留相对路径的 raw source commitment。"""
    path = root / relative_path
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": 0,
        "relative_path": relative_path,
        "role": _RAW_ROLE_BY_PATH[relative_path],
        "sha256": _sha256(payload),
    }


def _inventory_identity(
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """移除 zh-CN/zh-TW surface，只保留冻结分母来源 identity。"""
    values = tuple({
        "format_version": 1,
        "pair_id": item["pair_id"],
        "record_kind": "AUDACITY_ATOM_VALIDATION_IDENTITY_WITHOUT_LABEL_V1",
        "source_identity": item["source_identity"],
        "source_identity_sha256": item["source_identity_sha256"],
        "zh_hans_source_file_id": item["zh_hans"]["source_file_id"],
        "zh_hant_source_file_id": item["zh_hant"]["source_file_id"],
    } for item in pairs)
    if (not values or len({item["pair_id"] for item in values})
            != len(values)):
        raise BroadQaExternalDataError(
            "Audacity atom-validation identity roster 非法")
    return values


def _manifest(
        *,
        files: list[dict[str, object]],
        git_identity: dict[str, object],
        parser_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造 Audacity atom-validation source pack manifest。"""
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND),
        "files": files,
        "format_version": 1,
        "git_acquisition": git_identity,
        "license": {
            "license_bytes": AUDACITY_LICENSE_BYTES,
            "license_git_blob_sha1": AUDACITY_SOURCE_BLOBS["LICENSE.txt"],
            "license_sha256": AUDACITY_LICENSE_SHA256,
            "translation_file_default_expression": (
                AUDACITY_TRANSLATION_LICENSE_EXPRESSION),
        },
        "mastery_claimed": 0,
        "parser": {
            "identity": "POLIB_FIXED_COMMON_SOURCE_IDENTITY_V1",
            "polib_version": POLIB_VERSION,
        },
        "parser_summary": parser_summary,
        "production_enabled": 0,
        "selection_boundary": {
            "individual_translation_or_pair_read_before_selection": 0,
            "selection": (
                "ALL_COMMON_SINGULAR_NONFUZZY_NONOBSOLETE_"
                "NONEMPTY_MSGCTXT_MSGID_MSGID_PLURAL_IDENTITIES"),
            "source_selected_before_translation_blob_read": 1,
            "structure_equal_required_for_denominator": 0,
        },
        "source_family": AUDACITY_SOURCE_FAMILY,
        "source_policy_scope": AUDACITY_SOURCE_POLICY_SCOPE,
        "status": NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_exclusion": {
            "derivative_message_or_pair_allowed_in_v7_train": 0,
            "exclusion_granularity": "WHOLE_SOURCE_PACK_AND_ALL_DERIVATIVES",
        },
        "validation_state": {
            "candidate_or_runtime_read_count": 0,
            "formal_label_jsonl_materialized": 0,
            "individual_translation_surface_published_in_jsonl": 0,
            "raw_translation_surface_stored_on_k": 1,
            "validation_run_count": 0,
        },
    }


def publish_audacity_atom_validation_source_pack(
        *,
        run_root: str | Path,
        checkout_root: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """从固定 Git checkout 不可覆盖发布 Audacity validation source。"""
    root = _require_k_root(run_root)
    checkout = _within(root, checkout_root, label="checkout")
    target = _within(root, target_dir, label="target")
    if (not checkout.is_dir() or target.exists()
            or _overlap(checkout, target)):
        raise BroadQaExternalDataError(
            "Audacity atom-validation input/target path 非法")
    if polib.__version__ != POLIB_VERSION:
        raise BroadQaExternalDataError(
            "Audacity atom-validation polib 版本漂移")
    git_identity, raw_files = _git_source_material(checkout)
    _validate_frozen_raw_files(raw_files)
    source_records, pairs, summary = parse_audacity_atom_validation_files(
        raw_files)
    inventory = _inventory_identity(pairs)
    target.mkdir()
    for relative_path, payload in raw_files.items():
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payload)
    source_path = target / _OUTPUT_FILES[0][0]
    inventory_path = target / _OUTPUT_FILES[1][0]
    _write_jsonl(source_path, source_records)
    _write_jsonl(inventory_path, inventory)
    files = [
        _raw_artifact(target, relative_path)
        for relative_path in AUDACITY_SOURCE_FILES]
    files.extend((
        _artifact(
            source_path, role=_OUTPUT_FILES[0][1],
            count=len(source_records)),
        _artifact(
            inventory_path, role=_OUTPUT_FILES[1][1],
            count=len(inventory)),
    ))
    manifest = _manifest(
        files=files,
        git_identity=git_identity,
        parser_summary=summary,
    )
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_audacity_atom_validation_source_pack(
        source_pack_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从 pack 内 raw files 重派生并严格回读 source 与 identity。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
        raw_files = {
            relative_path: (root / relative_path).read_bytes()
            for relative_path in AUDACITY_SOURCE_FILES}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "Audacity atom-validation source pack 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "Audacity atom-validation manifest identity 漂移")
    _validate_frozen_raw_files(raw_files)
    source_records, pairs, summary = parse_audacity_atom_validation_files(
        raw_files)
    inventory = _inventory_identity(pairs)
    stored_source = _read_jsonl(
        root / _OUTPUT_FILES[0][0], label=_OUTPUT_FILES[0][1])
    stored_inventory = _read_jsonl(
        root / _OUTPUT_FILES[1][0], label=_OUTPUT_FILES[1][1])
    if (not strict_json_equal(stored_source, source_records)
            or not strict_json_equal(stored_inventory, inventory)):
        raise BroadQaExternalDataError(
            "Audacity atom-validation records/source 漂移")
    files = [
        _raw_artifact(root, relative_path)
        for relative_path in AUDACITY_SOURCE_FILES]
    files.extend((
        _artifact(
            root / _OUTPUT_FILES[0][0], role=_OUTPUT_FILES[0][1],
            count=len(source_records)),
        _artifact(
            root / _OUTPUT_FILES[1][0], role=_OUTPUT_FILES[1][1],
            count=len(inventory)),
    ))
    expected = _manifest(
        files=files,
        git_identity={
            "branch": AUDACITY_BRANCH,
            "commit": AUDACITY_COMMIT,
            "commit_date": AUDACITY_COMMIT_DATE,
            "remote": AUDACITY_REPOSITORY_URL,
            "root_tree": AUDACITY_ROOT_TREE,
        },
        parser_summary=summary,
    )
    if not strict_json_equal(stored, expected):
        raise BroadQaExternalDataError(
            "Audacity atom-validation manifest fields 漂移")
    return (
        {**stored, "manifest_sha256": expected_manifest_sha256},
        source_records,
        inventory,
    )


__all__ = [
    "AUDACITY_COMMIT",
    "AUDACITY_COMMIT_DATE",
    "AUDACITY_REPOSITORY_URL",
    "AUDACITY_ROOT_TREE",
    "AUDACITY_SOURCE_BLOBS",
    "AUDACITY_SOURCE_FAMILY",
    "AUDACITY_SOURCE_POLICY_SCOPE",
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS",
    "publish_audacity_atom_validation_source_pack",
    "read_audacity_atom_validation_source_pack",
]
