"""DLG-05 v4 owner handoff 的 metadata-only 边界。

候选侧只读取 owner receipt 的公开元数据，以及 label 文件的路径、大小和物理
链接属性；绝不打开、解压、哈希或解析 label payload。receipt 必须绑定当前 K 盘
source bundle、freeze、中文投影和一次性状态 ``SEALED_UNREAD``。当前旧 v4 family 是
synthetic fixture，本模块会在读取 receipt 或 label 前拒绝它；external source capsule 的
runtime artifact reader 建立前，任何 owner handoff 都保持关闭。本模块不创建 owner receipt、
不创建 guard、不执行 selection/formal，也不改变任何运行状态。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.conversation_heldout_v4_family import (
    ConversationHeldOutV4Family,
)
from pure_integer_ai.experiments.conversation_heldout_v4_metadata_audit import (
    ConversationHeldOutV4MetadataAudit,
    audit_v4_family_artifacts,
)


# object-model: exception
class ConversationHeldOutV4OwnerHandoffError(RuntimeError):
    """owner receipt 缺失、漂移、越界或 label-late 边界不闭合。"""


_REPARSE_POINT = 0x0400
_RECEIPT_NAME = "owner_receipt.json"
_RECEIPT_SCHEMA = "dlg05-v4-owner-receipt-v1"
_SEALED_UNREAD = "SEALED_UNREAD"


def _fail(message: str) -> None:
    raise ConversationHeldOutV4OwnerHandoffError(message)


def _strict_int_tuple(
        value: object, *, label: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验 receipt 中的非负整数 tuple/list，不接受 bool 或浮点。"""
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple) or (not value and not allow_empty):
        _fail(f"{label} 必须是{'非空' if not allow_empty else ''}整数序列")
    assert_int(*value, _where=label)
    if any(type(item) is not int or item < 0 for item in value):
        _fail(f"{label} 必须是非负严格整数")
    return value


def _digest(value: object, *, label: str) -> tuple[int, ...]:
    """解析固定 32 字节 SHA-256 十六进制元数据。"""
    if not isinstance(value, str):
        _fail(f"{label} 必须是十六进制字符串")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        _fail(f"{label} 不是合法十六进制")
        raise AssertionError from exc
    if len(raw) != 32:
        _fail(f"{label} 必须是 SHA-256")
    return tuple(raw)


def _positive(value: object, *, label: str) -> int:
    """校验 receipt 计数和大小。"""
    if type(value) is not int or value <= 0:
        _fail(f"{label} 必须是正严格整数")
    return value


def _zero_one(value: object, *, label: str) -> int:
    """校验一次性状态门使用 0/1 整数。"""
    if type(value) is not int or value not in (0, 1):
        _fail(f"{label} 必须是 0/1")
    return value


def _canonical_json(value: object) -> bytes:
    """生成 receipt 自身完整性校验使用的规范 JSON 字节。"""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒绝 JSON 重复字段，避免 receipt 解析出现语义覆盖。"""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"receipt JSON 字段重复：{key}")
        result[key] = value
    return result


def _read_receipt(path: Path) -> dict[str, object]:
    """只读解析 canonical owner receipt，不触碰 label 路径。"""
    try:
        payload = path.read_bytes()
        if not payload.endswith(b"\n"):
            _fail("owner receipt 必须以单个换行结束")
        value = json.loads(
            payload[:-1].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except ConversationHeldOutV4OwnerHandoffError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _fail("owner receipt 不是规范 JSON")
        raise AssertionError from exc
    if not isinstance(value, dict):
        _fail("owner receipt 顶层必须是 object")
    if _canonical_json(value) != payload:
        _fail("owner receipt 不是确定性 canonical JSON")
    declared = value.get("document_sha256")
    if not isinstance(declared, str):
        _fail("owner receipt 缺少 document_sha256")
    without_digest = dict(value)
    without_digest.pop("document_sha256", None)
    actual = hashlib.sha256(_canonical_json(without_digest)).hexdigest()
    if declared != actual:
        _fail("owner receipt document SHA 漂移")
    return value


def _root(root: str | Path, *, require_k_drive: bool, label: str) -> Path:
    """解析 owner/source 根并拒绝链接或 reparse root。"""
    raw_path = Path(root)
    if raw_path.is_symlink():
        _fail(f"{label} 不存在或是链接")
    try:
        raw_stat = os.stat(raw_path, follow_symlinks=False)
    except OSError as exc:
        _fail(f"{label} 不存在或不可读取")
        raise AssertionError from exc
    if getattr(raw_stat, "st_file_attributes", 0) & _REPARSE_POINT:
        _fail(f"{label} 不得是 reparse point")
    path = raw_path.resolve()
    if require_k_drive and path.drive.upper() != "K:":
        _fail(f"{label} 必须位于 K 盘")
    if not path.is_dir() or path.is_symlink():
        _fail(f"{label} 不存在或是链接")
    stat = os.stat(path, follow_symlinks=False)
    if getattr(stat, "st_file_attributes", 0) & _REPARSE_POINT:
        _fail(f"{label} 不得是 reparse point")
    return path


def _plain_file(path: Path, *, label: str, read: bool) -> os.stat_result:
    """核验普通单硬链接文件；read=False 时只读取文件系统元数据。"""
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} 不是普通文件")
    stat = os.stat(path, follow_symlinks=False)
    if getattr(stat, "st_nlink", 1) != 1:
        _fail(f"{label} 硬链接数必须为 1")
    if getattr(stat, "st_file_attributes", 0) & _REPARSE_POINT:
        _fail(f"{label} 不得是 reparse point")
    if read:
        return stat
    # 保留显式参数，调用方可审计本函数绝不打开文件。
    return stat


def _relative_label_path(value: object) -> PurePosixPath:
    """把 receipt label path 限制为 owner root 内的 POSIX 相对路径。"""
    if not isinstance(value, str) or not value:
        _fail("label_payload_path 必须是非空字符串")
    if "\\" in value or ":" in value:
        _fail("label_payload_path 不得含 Windows 绝对路径")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        _fail("label_payload_path 必须是安全相对路径")
    if relative == PurePosixPath(_RECEIPT_NAME):
        _fail("label payload 不得覆盖 owner receipt")
    return relative


def _file_set(root: Path) -> tuple[Path, ...]:
    """只枚举 owner 根文件名，不读取文件内容。"""
    result: list[Path] = []
    for path in root.rglob("*"):
        stat = os.stat(path, follow_symlinks=False)
        if getattr(stat, "st_file_attributes", 0) & _REPARSE_POINT:
            _fail("owner root 含 reparse point")
        if path.is_dir():
            if path.is_symlink():
                _fail("owner root 含目录链接")
            continue
        _plain_file(path, label="owner root file", read=False)
        result.append(path.relative_to(root))
    return tuple(sorted(result))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4OwnerMetadata:
    """候选侧读取到的 owner 元数据；不包含 label payload。"""

    schema_version: int
    owner_namespace: tuple[int, ...]
    bundle_payload_sha256: tuple[int, ...]
    bundle_payload_size: int
    bundle_index: int
    case_count: int
    turn_count: int
    source_count: int
    candidate_count: int
    label_payload_path: str
    label_payload_size: int
    label_payload_sha256: tuple[int, ...]
    projection_markdown_sha256: tuple[int, ...]
    projection_html_sha256: tuple[int, ...]
    status: str
    labels_read: int
    selection_run_count: int
    formal_run: int

    def __post_init__(self) -> None:
        """核验 metadata 只允许 sealed/unread、零执行的 owner 交接状态。"""
        if self.schema_version != 1:
            _fail("owner metadata schema version 非法")
        _strict_int_tuple(self.owner_namespace, label="owner namespace")
        for name, value in (
                ("bundle_payload_size", self.bundle_payload_size),
                ("bundle_index", self.bundle_index),
                ("case_count", self.case_count),
                ("turn_count", self.turn_count),
                ("source_count", self.source_count),
                ("candidate_count", self.candidate_count),
                ("label_payload_size", self.label_payload_size)):
            _positive(value, label=name)
        for name, value in (
                ("bundle_payload_sha256", self.bundle_payload_sha256),
                ("label_payload_sha256", self.label_payload_sha256),
                ("projection_markdown_sha256", self.projection_markdown_sha256),
                ("projection_html_sha256", self.projection_html_sha256)):
            if not isinstance(value, tuple) or len(value) != 32:
                _fail(f"{name} 必须是 32 字节 SHA-256")
        if self.status != _SEALED_UNREAD:
            _fail("owner metadata 必须保持 SEALED_UNREAD")
        if self.labels_read != 0 or self.selection_run_count != 0 or self.formal_run != 0:
            _fail("owner metadata 在 handoff 前不得读取 label 或运行 formal")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含路径文本的纯整数 metadata 身份。"""
        return (
            self.schema_version, *self.owner_namespace,
            self.bundle_payload_size, self.bundle_index,
            self.case_count, self.turn_count, self.source_count,
            self.candidate_count, self.label_payload_size,
            *self.bundle_payload_sha256,
            *self.label_payload_sha256,
            *self.projection_markdown_sha256,
            *self.projection_html_sha256,
            self.labels_read, self.selection_run_count, self.formal_run,
        )


def _parse_metadata(document: dict[str, object]) -> ConversationHeldOutV4OwnerMetadata:
    """从已校验 JSON 构造 typed owner metadata，并拒绝未知字段。"""
    expected = {
        "bundle_index", "bundle_payload_sha256", "bundle_payload_size",
        "case_count", "candidate_count", "document_sha256", "formal_run",
        "label_payload_path", "label_payload_sha256", "label_payload_size",
        "labels_read", "owner_namespace", "projection", "schema",
        "selection_run_count", "source_count", "status", "turn_count",
    }
    if set(document) != expected:
        _fail("owner receipt 字段集合不精确")
    projection = document["projection"]
    if not isinstance(projection, dict) or set(projection) != {
            "html_sha256", "markdown_sha256"}:
        _fail("owner receipt projection 字段不精确")
    if document["schema"] != _RECEIPT_SCHEMA:
        _fail("owner receipt schema 不匹配")
    return ConversationHeldOutV4OwnerMetadata(
        1,
        _strict_int_tuple(document["owner_namespace"], label="owner namespace"),
        _digest(document["bundle_payload_sha256"], label="bundle payload SHA"),
        _positive(document["bundle_payload_size"], label="bundle payload size"),
        _positive(document["bundle_index"], label="bundle index"),
        _positive(document["case_count"], label="case count"),
        _positive(document["turn_count"], label="turn count"),
        _positive(document["source_count"], label="source count"),
        _positive(document["candidate_count"], label="candidate count"),
        str(_relative_label_path(document["label_payload_path"])),
        _positive(document["label_payload_size"], label="label payload size"),
        _digest(document["label_payload_sha256"], label="label payload SHA"),
        _digest(projection["markdown_sha256"], label="projection Markdown SHA"),
        _digest(projection["html_sha256"], label="projection HTML SHA"),
        document["status"],
        _zero_one(document["labels_read"], label="labels_read"),
        _zero_one(document["selection_run_count"], label="selection_run_count"),
        _zero_one(document["formal_run"], label="formal_run"),
    )


def _projection_digests(source_root: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """读取公开投影字节并返回其 SHA；不读取 owner label。"""
    paths = (
        source_root / "projection" / "dlg05_v4_reading.md",
        source_root / "projection" / "dlg05_v4_reading.html",
    )
    result = []
    for path in paths:
        _plain_file(path, label="source projection", read=True)
        result.append(tuple(hashlib.sha256(path.read_bytes()).digest()))
    return result[0], result[1]


def read_v4_owner_metadata(
        owner_root: str | Path,
        source_root: str | Path,
        *,
        require_k_drive: bool = True,
        family: ConversationHeldOutV4Family | None = None,
        ) -> ConversationHeldOutV4OwnerMetadata:
    """只读验证 owner handoff；label 文件只读取 stat，不读取内容。"""
    owner = _root(owner_root, require_k_drive=require_k_drive, label="owner root")
    source = _root(source_root, require_k_drive=require_k_drive, label="source root")
    if not isinstance(family, ConversationHeldOutV4Family):
        _fail("owner handoff 不得默认回落到 synthetic family")
    audit: ConversationHeldOutV4MetadataAudit = audit_v4_family_artifacts(
        source, require_k_drive=False, family=family)
    if not audit.source_qualified:
        _fail("synthetic fixture 不得进入 owner handoff")
    receipt_path = owner / _RECEIPT_NAME
    _plain_file(receipt_path, label="owner receipt", read=True)
    document = _read_receipt(receipt_path)
    metadata = _parse_metadata(document)
    projection_md, projection_html = _projection_digests(source)
    if metadata.bundle_payload_sha256 != family.bundle.payload_sha256:
        _fail("owner receipt bundle SHA 与 source bundle 不一致")
    if (metadata.bundle_payload_size != family.bundle.payload_size
            or metadata.bundle_index != family.bundle.index
            or metadata.case_count != 6
            or metadata.turn_count != audit.turn_count
            or metadata.source_count != audit.source_count
            or metadata.candidate_count != audit.candidate_count):
        _fail("owner receipt bundle 计数或身份不一致")
    if (metadata.projection_markdown_sha256 != projection_md
            or metadata.projection_html_sha256 != projection_html):
        _fail("owner receipt projection SHA 与公开投影不一致")
    label_relative = PurePosixPath(metadata.label_payload_path)
    label_path = owner.joinpath(*label_relative.parts)
    owner_resolved = owner.resolve()
    if label_path.resolve() != owner_resolved / Path(*label_relative.parts):
        _fail("label payload 路径越界")
    label_stat = _plain_file(label_path, label="label payload", read=False)
    if label_stat.st_size != metadata.label_payload_size:
        _fail("label payload 大小与 owner receipt 不一致")
    files = _file_set(owner)
    expected_files = tuple(sorted((Path(_RECEIPT_NAME), Path(*label_relative.parts))))
    if files != expected_files:
        _fail("owner root 文件闭包不精确")
    return metadata


__all__ = [
    "ConversationHeldOutV4OwnerHandoffError",
    "ConversationHeldOutV4OwnerMetadata",
    "read_v4_owner_metadata",
]
