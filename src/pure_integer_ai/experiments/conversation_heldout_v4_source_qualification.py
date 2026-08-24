"""DLG-05 v4 外部来源声明的只读、无标签机械审计边界。

本模块只验证调用方带入的 anchor、source capsule、来源选择、许可声明与训练来源声明在
字节和结构上的对应关系。它不生成 receipt 或 inventory，不运行 candidate runtime/R03，
不读取 label、private、formal 或 owner 内容。普通数据结构不能证明人员/组织独立性，
也不能验证声明中的谱系、官方许可或存储快照不可变性；因此任何成功结果均保持未资格化。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    ConversationHeldOutV4RuntimeSourceCapsule,
    V4_RUNTIME_FAMILY_KEY,
    V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    ConversationHeldOutV4ExternalCapsuleError,
    read_v4_external_input_capsule,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    decode_integer_tuple,
    encode_integer_tuple,
)


V4_SOURCE_QUALIFICATION_RECEIPT_KIND = (
    "DLG05_V4_SOURCE_QUALIFICATION_RECEIPT_V1")
V4_SOURCE_QUALIFICATION_RECEIPT_SCHEMA = (
    "dlg05-v4-source-qualification-receipt-v1")
V4_SOURCE_QUALIFICATION_MANIFEST_SCHEMA = (
    "dlg05-v4-source-qualification-manifest-v1")
V4_TRAINING_PROVENANCE_KIND = "DLG05_V4_TRAINING_PROVENANCE_INVENTORY_V1"
V4_TRAINING_PROVENANCE_SCHEMA = "dlg05-v4-training-provenance-inventory-v1"
V4_TRAINING_PROVENANCE_MANIFEST_SCHEMA = (
    "dlg05-v4-training-provenance-manifest-v1")
V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS = (
    "OUT_OF_BAND_ATTESTATION_CLAIM_CONSISTENT_UNQUALIFIED")
V4_TEST_ONLY_QUALIFICATION_STATUS = (
    "TEST_ONLY_SOURCE_QUALIFICATION_MECHANICS_ONLY")
V4_SOURCE_QUALIFICATION_AUDIT_SCOPE = (
    "BOUNDED_MECHANICAL_CLAIM_CONSISTENCY_ONLY")
V4_SELECTION_RULE = "ALL_FROZEN_POPULATION_V1"
V4_TRAINING_COVERAGE = "ALL_ACTIVE_TRAINING_SOURCES_AND_DERIVATIVES"
V4_TRAINING_UNIVERSE_NONEMPTY = "NONEMPTY_ACTIVE_TRAINING_UNIVERSE_V1"
V4_TRAINING_UNIVERSE_EMPTY = "FROZEN_EMPTY_TRAINING_UNIVERSE_V1"
V4_TRAINING_EXCLUSION_RULE = "WHOLE_SOURCE_AND_DERIVATIVES"
V4_FILES_SCOPE = "PRE_MANIFEST_PAYLOAD_FILES_ONLY"
V4_MAX_TRAINING_PROVENANCE_ENTRIES = 4096
V4_MAX_TRAINING_PROVENANCE_PAYLOAD_BYTES = 4 * 1024 * 1024
V4_MAX_EXTERNAL_CLOSURE_MANIFEST_BYTES = 64 * 1024

_QUALIFICATION_RECEIPT_FILE = Path("qualification_receipt.json")
_QUALIFICATION_INTS_FILE = Path("qualification.canonical.ints")
_TRAINING_PROVENANCE_FILE = Path("training_provenance.json")
_TRAINING_INTS_FILE = Path("training_provenance.canonical.ints")
_MANIFEST_FILE = Path("manifest.json")
_REPARSE_POINT = 0x0400


# object-model: exception
class ConversationHeldOutV4SourceQualificationError(RuntimeError):
    """来源资格 artifact、外部锚点、来源选择或训练不相交条件不完整。"""


def _fail(message: str) -> None:
    """统一产生本模块的 fail-closed 错误。"""
    raise ConversationHeldOutV4SourceQualificationError(message)


def _sha256(payload: bytes) -> tuple[int, ...]:
    """把字节的完整 SHA-256 表示为严格整数 tuple。"""
    return tuple(hashlib.sha256(payload).digest())


def _digest_hex(value: tuple[int, ...]) -> str:
    """验证并导出 SHA-256 的小写十六进制表示。"""
    _validate_digest(value, label="digest")
    return bytes(value).hex()


def _validate_digest(value: object, *, label: str) -> tuple[int, ...]:
    """拒绝截断、bool、浮点或越界的 SHA-256 整数序列。"""
    if (not isinstance(value, tuple) or len(value) != 32
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        _fail(f"{label} 必须是完整 SHA-256 整数 tuple")
    return value


def _digest_from_document(value: Any, *, label: str) -> tuple[int, ...]:
    """从 canonical JSON 的小写 SHA-256 字段恢复严格整数 tuple。"""
    if (not isinstance(value, str) or len(value) != 64
            or value != value.lower()):
        _fail(f"{label} 必须是小写 SHA-256")
    try:
        digest = tuple(bytes.fromhex(value))
    except ValueError as exc:
        _fail(f"{label} SHA-256 非法")
        raise AssertionError from exc
    return _validate_digest(digest, label=label)


def _strict_int_tuple(
        value: Any, *, label: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """从 JSON 读取无隐式转换的严格整数列表。"""
    if (not isinstance(value, list) or (not allow_empty and not value)
            or any(type(item) is not int for item in value)):
        _fail(f"{label} 必须是{'可空' if allow_empty else '非空'}严格整数列表")
    return tuple(value)


def _protocol_key_from_document(value: Any, *, label: str) -> ProtocolKey:
    """从完整整数列表重建 ProtocolKey，不接受字符串身份。"""
    try:
        return ProtocolKey(_strict_int_tuple(value, label=label))
    except (TypeError, ValueError) as exc:
        _fail(f"{label} ProtocolKey 非法")
        raise AssertionError from exc


def _source_ref_from_document(value: Any, *, label: str) -> SourceRef:
    """从完整稳定键重建 SourceRef，避免用摘要替代来源本体。"""
    try:
        return SourceRef.from_stable_key(_strict_int_tuple(value, label=label))
    except (TypeError, ValueError) as exc:
        _fail(f"{label} SourceRef 非法")
        raise AssertionError from exc


def _exact_fields(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    """拒绝未知或缺失字段，避免把 label/private 内容夹入只读资格协议。"""
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{label} 字段集合漂移")
    return value


def _encoded_key_sha256(value: tuple[int, ...]) -> tuple[int, ...]:
    """对完整整数稳定键的规范 codec 字节求 SHA，不以缩略键替代身份。"""
    return _sha256(encode_integer_tuple(value))


def _scalars_sha256(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """对来源 URI、许可或归属的精确 UTF-8 scalar 文本求 SHA。"""
    if not isinstance(value, tuple) or not value:
        _fail(f"{label} 不得为空")
    try:
        payload = "".join(chr(item) for item in value).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"{label} scalar 非法")
        raise AssertionError from exc
    return _sha256(payload)


def _is_reparse(path: Path) -> bool:
    """在 resolve 前检查 Windows reparse attribute，其他平台保持 false。"""
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(stat, "st_file_attributes", 0) & _REPARSE_POINT)


def _existing_normal_directory(path: Path, *, label: str) -> Path:
    """逐级拒绝 link/reparse 后解析已存在的普通目录。"""
    if not path.is_absolute() or ".." in path.parts:
        _fail(f"{label} 必须是不含 .. 的绝对路径")
    for current in (path, *path.parents):
        if current.is_symlink() or _is_reparse(current):
            _fail(f"{label} 含链接或 reparse point")
        if not current.exists():
            _fail(f"{label} 不存在")
        if current == current.parent:
            break
    try:
        resolved = path.resolve()
    except OSError as exc:
        _fail(f"{label} 无法解析")
        raise AssertionError from exc
    if (not resolved.is_dir() or resolved.is_symlink()
            or _is_reparse(resolved)):
        _fail(f"{label} 不是普通目录")
    return resolved


def _read_root(
        root: str | Path, *, require_k_drive: bool, label: str,
        ) -> Path:
    """在读取任何资格文件前验证 K 盘根、绝对路径和物理目录边界。"""
    if type(require_k_drive) is not bool:
        raise TypeError("require_k_drive 必须是 bool")
    target = _existing_normal_directory(Path(root), label=label)
    if require_k_drive and target.drive.upper() != "K:":
        _fail(f"{label} 必须位于 K 盘")
    return target


def _require_disjoint(roots: tuple[Path, ...]) -> None:
    """拒绝 source、qualification、training 三根任意相同或嵌套。"""
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if (left == right or left.is_relative_to(right)
                    or right.is_relative_to(left)):
                _fail("source/qualification/training roots 不得重叠或嵌套")


def _plain_file(root: Path, relative: Path) -> Path:
    """返回 root 内唯一普通单硬链接文件，拒绝目录、链接和路径逃逸。"""
    if relative.is_absolute() or relative.parent != Path("."):
        _fail("资格 artifact 文件不允许子目录")
    path = root / relative
    if not path.is_file() or path.is_symlink() or _is_reparse(path):
        _fail(f"资格 artifact 文件不是普通文件: {relative.name}")
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        _fail(f"资格 artifact 文件不可读取: {relative.name}")
        raise AssertionError from exc
    if getattr(stat, "st_nlink", 1) != 1:
        _fail(f"资格 artifact 文件硬链接数必须为 1: {relative.name}")
    try:
        resolved = path.resolve()
    except OSError as exc:
        _fail(f"资格 artifact 文件无法解析: {relative.name}")
        raise AssertionError from exc
    if not resolved.is_relative_to(root):
        _fail(f"资格 artifact 文件路径越界: {relative.name}")
    return path


def _require_file_set(root: Path, expected: frozenset[Path]) -> None:
    """要求 root 恰有声明的三文件闭包，不允许缓存、投影或额外元数据。"""
    try:
        children = tuple(root.iterdir())
    except OSError as exc:
        _fail("资格 artifact root 不可枚举")
        raise AssertionError from exc
    found: set[Path] = set()
    for child in children:
        relative = child.relative_to(root)
        if (child.is_dir() or child.is_symlink() or _is_reparse(child)
                or not child.is_file()):
            _fail("资格 artifact root 含目录、链接、reparse 或非普通文件")
        _plain_file(root, relative)
        found.add(relative)
    if frozenset(found) != expected:
        _fail("资格 artifact 文件闭包不精确")


def _read_plain_bytes(
        root: Path, relative: Path, *, max_bytes: int | None = None,
        ) -> bytes:
    """在读取前后复核普通文件门，并先用 stat 拒绝超出小型 transport 预算的文件。"""
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise TypeError("资格 artifact max_bytes 必须为正严格整数或 None")
    path = _plain_file(root, relative)
    try:
        size_before = os.stat(path, follow_symlinks=False).st_size
    except OSError as exc:
        _fail(f"资格 artifact 文件无法 stat: {relative.name}")
        raise AssertionError from exc
    if type(size_before) is not int or size_before < 0:
        _fail(f"资格 artifact 文件尺寸非法: {relative.name}")
    if max_bytes is not None and size_before > max_bytes:
        _fail(f"资格 artifact 文件超过小型 transport 容量: {relative.name}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail(f"资格 artifact 文件读取失败: {relative.name}")
        raise AssertionError from exc
    _plain_file(root, relative)
    try:
        size_after = os.stat(path, follow_symlinks=False).st_size
    except OSError as exc:
        _fail(f"资格 artifact 文件读取后无法 stat: {relative.name}")
        raise AssertionError from exc
    if size_after != len(payload):
        _fail(f"资格 artifact 文件读取期间尺寸漂移: {relative.name}")
    if not payload:
        _fail(f"资格 artifact 文件不能为空: {relative.name}")
    if max_bytes is not None and len(payload) > max_bytes:
        _fail(f"资格 artifact 文件超过小型 transport 容量: {relative.name}")
    return payload


def _closure_manifest_document(
        *, artifact_kind: str, schema: str, body_file: Path, ints_file: Path,
        body_payload: bytes, ints_payload: bytes,
        ) -> dict[str, Any]:
    """重建 manifest 前两个 payload 的规范描述符，不给 manifest 伪造自哈希。"""
    return {
        "artifact_kind": artifact_kind,
        "files": {
            body_file.name: {
                "sha256": hashlib.sha256(body_payload).hexdigest(),
                "size": len(body_payload),
            },
            ints_file.name: {
                "sha256": hashlib.sha256(ints_payload).hexdigest(),
                "size": len(ints_payload),
            },
        },
        "files_scope": V4_FILES_SCOPE,
        "format_version": 1,
        "schema": schema,
    }


def _read_external_closure(
        root: Path, *, body_file: Path, ints_file: Path,
        artifact_kind: str, schema: str, label: str,
        max_body_bytes: int | None = None,
        max_ints_bytes: int | None = None,
        ) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    """只读验证一个外部 receipt/inventory 的 JSON、整数镜像和 manifest-last 闭包。"""
    expected_files = frozenset({body_file, ints_file, _MANIFEST_FILE})
    _require_file_set(root, expected_files)
    manifest_payload = _read_plain_bytes(
        root, _MANIFEST_FILE,
        max_bytes=V4_MAX_EXTERNAL_CLOSURE_MANIFEST_BYTES)
    body_payload = _read_plain_bytes(root, body_file, max_bytes=max_body_bytes)
    ints_payload = _read_plain_bytes(root, ints_file, max_bytes=max_ints_bytes)
    try:
        manifest = parse_canonical_json_bytes(manifest_payload, require_object=True)
        document = parse_canonical_json_bytes(body_payload, require_object=True)
    except DatasetContractError as exc:
        _fail(f"{label} JSON 不是规范 canonical JSON")
        raise AssertionError from exc
    _exact_fields(manifest, frozenset({
        "artifact_kind", "files", "files_scope", "format_version", "schema",
    }), label=f"{label} manifest")
    if (manifest["artifact_kind"] != artifact_kind
            or manifest["schema"] != schema
            or manifest["format_version"] != 1
            or manifest["files_scope"] != V4_FILES_SCOPE):
        _fail(f"{label} manifest kind/schema/version 非法")
    files = _exact_fields(manifest["files"], frozenset({
        body_file.name, ints_file.name,
    }), label=f"{label} manifest files")
    for relative, payload in ((body_file, body_payload), (ints_file, ints_payload)):
        entry = _exact_fields(files[relative.name], frozenset({"sha256", "size"}),
                              label=f"{label} manifest files.{relative.name}")
        if (type(entry["size"]) is not int or entry["size"] != len(payload)
                or entry["sha256"] != hashlib.sha256(payload).hexdigest()):
            _fail(f"{label} manifest 文件身份漂移: {relative.name}")
    try:
        decoded = decode_integer_tuple(ints_payload)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail(f"{label} canonical ints 非法")
        raise AssertionError from exc
    if decoded != tuple(body_payload) or encode_integer_tuple(decoded) != ints_payload:
        _fail(f"{label} canonical ints 与 JSON payload 漂移")
    expected_manifest = canonical_json_bytes(_closure_manifest_document(
        artifact_kind=artifact_kind,
        schema=schema,
        body_file=body_file,
        ints_file=ints_file,
        body_payload=body_payload,
        ints_payload=ints_payload,
    ))
    if manifest_payload != expected_manifest:
        _fail(f"{label} manifest 不是当前 payload 的规范闭包")
    _require_file_set(root, expected_files)
    return document, body_payload, ints_payload, manifest_payload


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4QualificationTrustAnchor:
    """调用方以 out-of-band 通道带入的 receipt/inventory 声明锚点。

    本结构不是签名、受保护 trust root 或身份认证凭据；其值可以约束当前审计输入，
    不能令本模块自行认定签发者为独立第三方。
    """

    qualifier_key: ProtocolKey
    training_inventory_authority_key: ProtocolKey
    qualification_receipt_sha256: tuple[int, ...]
    training_inventory_manifest_sha256: tuple[int, ...]
    trust_anchor_key: ProtocolKey

    def __post_init__(self) -> None:
        """确保 anchor 自身只有外部身份和完整 digest，不夹带来源正文或标签。"""
        for label, value in (
                ("qualifier key", self.qualifier_key),
                ("training inventory authority key",
                 self.training_inventory_authority_key),
                ("trust anchor key", self.trust_anchor_key)):
            if not isinstance(value, ProtocolKey):
                raise TypeError(f"v4 qualification trust anchor {label} 类型错误")
        _validate_digest(self.qualification_receipt_sha256,
                         label="qualification receipt sha256")
        _validate_digest(self.training_inventory_manifest_sha256,
                         label="training inventory manifest sha256")
        if len({
                self.qualifier_key,
                self.training_inventory_authority_key,
                self.trust_anchor_key,
        }) != 3:
            _fail("qualification trust anchor 三个 authority key 必须不同")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4TrainingProvenanceEntry:
    """训练来源清单的一条 SourceRef/content/lineage 三重身份。"""

    source_ref: SourceRef
    content_sha256: tuple[int, ...]
    lineage_key: ProtocolKey

    def __post_init__(self) -> None:
        """拒绝摘要替代 SourceRef、缺 lineage 或不完整内容 hash。"""
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("training provenance source_ref 类型错误")
        _validate_digest(self.content_sha256, label="training provenance content sha256")
        if not isinstance(self.lineage_key, ProtocolKey):
            raise TypeError("training provenance lineage_key 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回三重不相交审计使用的完整整数身份。"""
        result: list[int] = []
        for value in (
                self.source_ref.stable_key(), self.content_sha256,
                self.lineage_key.components):
            result.extend((len(value), *value))
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _QualificationSourceIdentity:
    """qualification receipt 中一条全人口来源的外部 lineage 声明。"""

    source_ref: SourceRef
    content_sha256: tuple[int, ...]
    lineage_key: ProtocolKey

    def __post_init__(self) -> None:
        """要求可重建 SourceRef、完整内容 SHA 和明确 lineage key。"""
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("qualification source source_ref 类型错误")
        _validate_digest(self.content_sha256,
                         label="qualification source content sha256")
        if not isinstance(self.lineage_key, ProtocolKey):
            raise TypeError("qualification source lineage_key 类型错误")

    def sort_key(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """给 source table 一个与平台无关的完整整数排序键。"""
        return (
            self.source_ref.stable_key(), self.content_sha256,
            self.lineage_key.components,
        )

    def document(self) -> dict[str, Any]:
        """导出选择表的完整、非摘要来源身份。"""
        return {
            "content_sha256": _digest_hex(self.content_sha256),
            "lineage_key": list(self.lineage_key.components),
            "source_ref_key": list(self.source_ref.stable_key()),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4SourceQualificationAudit:
    """只读机械审计结果，不授予来源、runtime、owner、formal 或能力资格。"""

    capsule: ConversationHeldOutV4RuntimeSourceCapsule
    qualification_receipt_sha256: tuple[int, ...]
    qualification_manifest_sha256: tuple[int, ...]
    training_inventory_manifest_sha256: tuple[int, ...]
    qualification_status: str
    audit_scope: str
    source_count: int
    turn_count: int

    def __post_init__(self) -> None:
        """固定外部 capsule、artifact 身份和明确未资格化的审计边界。"""
        if (not isinstance(self.capsule, ConversationHeldOutV4RuntimeSourceCapsule)
                or self.capsule.origin != V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL):
            _fail("source qualification audit 必须绑定 external runtime capsule")
        for label, value in (
                ("qualification receipt sha256",
                 self.qualification_receipt_sha256),
                ("qualification manifest sha256",
                 self.qualification_manifest_sha256),
                ("training inventory manifest sha256",
                 self.training_inventory_manifest_sha256)):
            _validate_digest(value, label=label)
        if self.qualification_status not in {
                V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS,
                V4_TEST_ONLY_QUALIFICATION_STATUS}:
            _fail("source qualification audit 状态非法")
        if self.audit_scope != V4_SOURCE_QUALIFICATION_AUDIT_SCOPE:
            _fail("source qualification audit scope 非法")
        if (type(self.source_count) is not int or self.source_count <= 0
                or type(self.turn_count) is not int or self.turn_count <= 0):
            _fail("qualified source source/turn count 必须为正严格整数")


def _source_records(
        capsule: ConversationHeldOutV4RuntimeSourceCapsule,
        ) -> tuple[Any, ...]:
    """聚合 capsule 全部唯一来源，拒绝同一 SourceRef 的不同 source record。"""
    records: dict[SourceRef, Any] = {}
    for runtime_input in capsule.inputs:
        for record in runtime_input.source_records:
            existing = records.get(record.source)
            if existing is not None and existing != record:
                _fail("同一 SourceRef 在 capsule 中不得对应不同 source record")
            records[record.source] = record
    if not records:
        _fail("external capsule 缺少 source record")
    return tuple(sorted(records.values(), key=lambda item: item.source.stable_key()))


def _capsule_binding_document(
        capsule: ConversationHeldOutV4RuntimeSourceCapsule,
        records: tuple[Any, ...],
        ) -> dict[str, Any]:
    """构造 qualification receipt 必须逐项相等的 capsule 身份绑定。"""
    if capsule.external_producer_key is None:
        _fail("external capsule 缺 producer key")
    if capsule.external_producer_declaration is None:
        _fail("external capsule 缺 producer declaration")
    turn_key = []
    for runtime_input in capsule.inputs:
        value = runtime_input.stable_key()
        turn_key.extend((len(value), *value))
    dependencies = capsule.dependencies
    return {
        "dependencies": {
            "artifact_sha256": _digest_hex(dependencies.artifact_sha256),
            "document_sha256": _digest_hex(dependencies.document_sha256),
            "inventory_sha256": _digest_hex(dependencies.inventory_sha256),
        },
        "family_key": list(V4_RUNTIME_FAMILY_KEY.components),
        "input_stable_key_sha256": _digest_hex(
            _encoded_key_sha256(capsule.stable_key())),
        "manifest_sha256": _digest_hex(capsule.manifest_sha256),
        "origin": capsule.origin,
        "producer_declaration_sha256": _digest_hex(_sha256(
            capsule.external_producer_declaration.encode("utf-8"))),
        "producer_key": list(capsule.external_producer_key.components),
        "source_count": len(records),
        "turn_count": len(capsule.inputs),
        "turn_table_sha256": _digest_hex(_encoded_key_sha256(tuple(turn_key))),
    }


def _parse_source_identity(value: Any, *, label: str) -> _QualificationSourceIdentity:
    """从全人口或训练表读取 SourceRef/content/lineage 的完整三重身份。"""
    _exact_fields(value, frozenset({
        "content_sha256", "lineage_key", "source_ref_key",
    }), label=label)
    return _QualificationSourceIdentity(
        _source_ref_from_document(value["source_ref_key"], label=f"{label}.source"),
        _digest_from_document(value["content_sha256"], label=f"{label}.content"),
        _protocol_key_from_document(value["lineage_key"], label=f"{label}.lineage"),
    )


def _parse_population(
        selection: Any, records: tuple[Any, ...],
        ) -> tuple[_QualificationSourceIdentity, ...]:
    """验证选择表覆盖 capsule 全部来源；lineage 仍是 receipt 的外部声明。"""
    _exact_fields(selection, frozenset({
        "population", "population_count", "population_roster_sha256",
        "selected_source_count", "selected_source_table_sha256",
        "selection_policy_sha256", "selection_rule",
    }), label="qualification selection")
    if selection["selection_rule"] != V4_SELECTION_RULE:
        _fail("qualification selection rule 非法")
    if (type(selection["population_count"]) is not int
            or type(selection["selected_source_count"]) is not int):
        _fail("qualification selection count 必须是严格整数")
    _digest_from_document(selection["selection_policy_sha256"],
                          label="qualification selection policy")
    raw_population = selection["population"]
    if not isinstance(raw_population, list) or not raw_population:
        _fail("qualification population 必须是非空列表")
    population = tuple(_parse_source_identity(
        item, label=f"qualification population[{index}]")
        for index, item in enumerate(raw_population))
    if tuple(item.sort_key() for item in population) != tuple(sorted(
            item.sort_key() for item in population)):
        _fail("qualification population 必须按完整 source identity 排序")
    if len({item.source_ref for item in population}) != len(population):
        _fail("qualification population SourceRef 不得重复")
    if (selection["population_count"] != len(population)
            or selection["selected_source_count"] != len(population)):
        _fail("qualification population count 漂移")
    population_payload = canonical_json_bytes({
        "population": [item.document() for item in population],
    })
    population_sha = _digest_hex(_sha256(population_payload))
    if (selection["population_roster_sha256"] != population_sha
            or selection["selected_source_table_sha256"] != population_sha):
        _fail("qualification population roster/table SHA 漂移")
    record_map = {record.source: record for record in records}
    if tuple(item.source_ref for item in population) != tuple(record_map):
        _fail("ALL_FROZEN_POPULATION 未精确覆盖 capsule 全部来源")
    for item in population:
        if item.content_sha256 != record_map[item.source_ref].content_sha256:
            _fail("qualification population content SHA 与 capsule 不一致")
    return population


def _parse_license_review(
        value: Any, *, population: tuple[_QualificationSourceIdentity, ...],
        records: tuple[Any, ...],
        ) -> None:
    """验证每个来源的许可/归属声明摘要与 capsule 绑定，不回读官方证据。"""
    _exact_fields(value, frozenset({
        "policy_sha256", "records", "reviewed_source_count",
    }), label="qualification license review")
    _digest_from_document(value["policy_sha256"],
                          label="qualification license policy")
    if type(value["reviewed_source_count"]) is not int:
        _fail("qualification reviewed source count 必须是严格整数")
    raw_records = value["records"]
    if not isinstance(raw_records, list):
        _fail("qualification license records 必须是列表")
    if value["reviewed_source_count"] != len(raw_records):
        _fail("qualification reviewed source count 漂移")
    expected_fields = frozenset({
        "allowed", "attribution_scalars_sha256", "content_sha256",
        "license_scalars_sha256", "lineage_key",
        "official_license_evidence_sha256",
        "official_license_evidence_uri_sha256", "source_record_stable_key_sha256",
        "source_ref_key", "source_uri_sha256",
    })
    parsed: list[_QualificationSourceIdentity] = []
    record_map = {record.source: record for record in records}
    population_map = {item.source_ref: item for item in population}
    for index, item in enumerate(raw_records):
        _exact_fields(item, expected_fields,
                      label=f"qualification license records[{index}]")
        identity = _QualificationSourceIdentity(
            _source_ref_from_document(item["source_ref_key"],
                                      label=f"qualification license[{index}].source"),
            _digest_from_document(item["content_sha256"],
                                  label=f"qualification license[{index}].content"),
            _protocol_key_from_document(item["lineage_key"],
                                        label=f"qualification license[{index}].lineage"),
        )
        if type(item["allowed"]) is not int or item["allowed"] != 1:
            _fail("qualification license record 必须明确 allowed=1")
        for field in (
                "source_record_stable_key_sha256", "source_uri_sha256",
                "license_scalars_sha256", "attribution_scalars_sha256",
                "official_license_evidence_uri_sha256",
                "official_license_evidence_sha256"):
            _digest_from_document(item[field],
                                  label=f"qualification license[{index}].{field}")
        record = record_map.get(identity.source_ref)
        if record is None or population_map.get(identity.source_ref) != identity:
            _fail("qualification license record 未绑定全人口 source identity")
        expected = {
            "source_record_stable_key_sha256": _digest_hex(
                _encoded_key_sha256(record.stable_key())),
            "source_uri_sha256": _digest_hex(_scalars_sha256(
                record.source_uri_scalars, label="source URI")),
            "license_scalars_sha256": _digest_hex(_scalars_sha256(
                record.license_scalars, label="source license")),
            "attribution_scalars_sha256": _digest_hex(_scalars_sha256(
                record.attribution_scalars, label="source attribution")),
        }
        for field, expected_value in expected.items():
            if item[field] != expected_value:
                _fail(f"qualification license record {field} 与 capsule 不一致")
        parsed.append(identity)
    if tuple(item.sort_key() for item in parsed) != tuple(sorted(
            item.sort_key() for item in parsed)):
        _fail("qualification license records 必须按完整 source identity 排序")
    if tuple(parsed) != population:
        _fail("qualification license records 必须精确覆盖全人口")


def _parse_training_inventory(
        value: dict[str, Any],
        ) -> tuple[tuple[ConversationHeldOutV4TrainingProvenanceEntry, ...], ProtocolKey]:
    """验证有界训练来源 inventory 的覆盖声明、冻结宇宙和排序条目。

    该 JSON transport 明确只支持小型机械审计，不能承担真实全量训练来源清单；
    超过容量时必须拒绝，后续需改为 K 盘有序分片与流式归并。
    """
    _exact_fields(value, frozenset({
        "artifact_kind", "coverage", "entries", "format_version",
        "inventory_authority_key", "inventory_scope_sha256", "schema",
        "source_count", "training_universe",
    }), label="training provenance")
    if (value["artifact_kind"] != V4_TRAINING_PROVENANCE_KIND
            or value["schema"] != V4_TRAINING_PROVENANCE_SCHEMA
            or value["format_version"] != 1
            or value["coverage"] != V4_TRAINING_COVERAGE):
        _fail("training provenance kind/schema/coverage 非法")
    if type(value["source_count"]) is not int or value["source_count"] < 0:
        _fail("training provenance source_count 非法")
    _digest_from_document(value["inventory_scope_sha256"],
                          label="training provenance inventory scope")
    authority = _protocol_key_from_document(value["inventory_authority_key"],
                                             label="training provenance authority")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list):
        _fail("training provenance entries 必须是列表")
    if len(raw_entries) > V4_MAX_TRAINING_PROVENANCE_ENTRIES:
        _fail("training provenance 超过小型 transport 条目容量")
    entries = tuple(ConversationHeldOutV4TrainingProvenanceEntry(
        item.source_ref, item.content_sha256, item.lineage_key)
        for item in (_parse_source_identity(
            entry, label=f"training provenance entries[{index}]")
            for index, entry in enumerate(raw_entries)))
    if value["source_count"] != len(entries):
        _fail("training provenance source_count 漂移")
    if tuple((item.source_ref.stable_key(), item.content_sha256,
              item.lineage_key.components) for item in entries) != tuple(sorted(
                  (item.source_ref.stable_key(), item.content_sha256,
                   item.lineage_key.components) for item in entries)):
        _fail("training provenance entries 必须按完整三重身份排序")
    if len({item.source_ref for item in entries}) != len(entries):
        _fail("training provenance SourceRef 不得重复")
    if entries and value["training_universe"] != V4_TRAINING_UNIVERSE_NONEMPTY:
        _fail("非空 training provenance 必须声明非空冻结训练宇宙")
    if not entries and value["training_universe"] != V4_TRAINING_UNIVERSE_EMPTY:
        _fail("空 training provenance 必须显式声明冻结空训练宇宙")
    return entries, authority


def _parse_authorities(
        value: Any, capsule: ConversationHeldOutV4RuntimeSourceCapsule,
        ) -> tuple[ProtocolKey, ProtocolKey, ProtocolKey]:
    """验证 receipt 的四方 authority 声明彼此不同且 producer 绑定 capsule。"""
    expected_fields = frozenset({
        "producer_equals_qualifier", "producer_equals_selector",
        "producer_equals_training_inventory_authority", "producer_key",
        "qualifier_equals_training_inventory_authority", "qualifier_key",
        "selector_equals_qualifier", "selector_equals_training_inventory_authority",
        "selector_key", "training_inventory_authority_key",
    })
    _exact_fields(value, expected_fields, label="qualification authorities")
    if capsule.external_producer_key is None:
        _fail("external capsule 缺 producer key")
    producer = _protocol_key_from_document(value["producer_key"],
                                           label="qualification producer")
    selector = _protocol_key_from_document(value["selector_key"],
                                           label="qualification selector")
    qualifier = _protocol_key_from_document(value["qualifier_key"],
                                            label="qualification qualifier")
    training_authority = _protocol_key_from_document(
        value["training_inventory_authority_key"],
        label="qualification training authority")
    if producer != capsule.external_producer_key:
        _fail("qualification producer key 与 capsule 不一致")
    for field in expected_fields - {
            "producer_key", "selector_key", "qualifier_key",
            "training_inventory_authority_key"}:
        if type(value[field]) is not int or value[field] != 0:
            _fail(f"qualification authority {field} 必须为 0")
    if len({producer, selector, qualifier, training_authority}) != 4:
        _fail("qualification producer/selector/qualifier/training authority 必须两两不同")
    return selector, qualifier, training_authority


def _validate_disjointness(
        value: Any, *, population: tuple[_QualificationSourceIdentity, ...],
        training_entries: tuple[ConversationHeldOutV4TrainingProvenanceEntry, ...],
        training_manifest_sha256: tuple[int, ...],
        ) -> None:
    """计算可绑定的 SourceRef/内容交集，并保守检查 receipt 自述 lineage 交集。

    capsule 尚未把 lineage 编为可验证本体，故 lineage 只在两张外部声明表之间比较；
    它可阻断冲突，不能证明全部派生链确已穷尽。
    """
    _exact_fields(value, frozenset({
        "content_sha256_intersection_count", "coverage", "exclusion_rule",
        "lineage_intersection_count", "source_ref_intersection_count",
        "training_inventory_manifest_sha256",
    }), label="qualification training disjointness")
    if (value["coverage"] != V4_TRAINING_COVERAGE
            or value["exclusion_rule"] != V4_TRAINING_EXCLUSION_RULE
            or value["training_inventory_manifest_sha256"]
            != _digest_hex(training_manifest_sha256)):
        _fail("qualification training disjointness binding 非法")
    candidate_source_refs = {item.source_ref.stable_key() for item in population}
    candidate_content = {item.content_sha256 for item in population}
    candidate_lineages = {item.lineage_key.components for item in population}
    training_source_refs = {
        item.source_ref.stable_key() for item in training_entries}
    training_content = {item.content_sha256 for item in training_entries}
    training_lineages = {item.lineage_key.components for item in training_entries}
    actual_counts = {
        "source_ref_intersection_count": len(
            candidate_source_refs & training_source_refs),
        "content_sha256_intersection_count": len(
            candidate_content & training_content),
        "lineage_intersection_count": len(
            candidate_lineages & training_lineages),
    }
    for field, actual in actual_counts.items():
        if type(value[field]) is not int or value[field] != 0 or actual != 0:
            _fail(f"qualification training disjointness {field} 非零或声明漂移")


def _parse_qualification_receipt(
        value: dict[str, Any], *, capsule: ConversationHeldOutV4RuntimeSourceCapsule,
        records: tuple[Any, ...],
        training_entries: tuple[ConversationHeldOutV4TrainingProvenanceEntry, ...],
        training_authority: ProtocolKey,
        training_manifest_sha256: tuple[int, ...],
        ) -> tuple[ProtocolKey, ProtocolKey, ProtocolKey]:
    """验证外部 receipt 的机械绑定；不把其中任何文字升级为外部事实。"""
    _exact_fields(value, frozenset({
        "artifact_kind", "candidate_result_read_count", "capsule", "format_version",
        "independent_authorities", "label_or_formal_read_count", "license_review",
        "qualification_status", "schema", "selection", "training_disjointness",
    }), label="qualification receipt")
    if (value["artifact_kind"] != V4_SOURCE_QUALIFICATION_RECEIPT_KIND
            or value["schema"] != V4_SOURCE_QUALIFICATION_RECEIPT_SCHEMA
            or value["format_version"] != 1
            or value["qualification_status"]
            != V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS):
        _fail("qualification receipt kind/schema/status 非法")
    if (type(value["candidate_result_read_count"]) is not int
            or value["candidate_result_read_count"] != 0
            or type(value["label_or_formal_read_count"]) is not int
            or value["label_or_formal_read_count"] != 0):
        _fail("qualification receipt 不得读取 candidate result、label 或 formal")
    expected_capsule = _capsule_binding_document(capsule, records)
    if value["capsule"] != expected_capsule:
        _fail("qualification receipt capsule binding 漂移")
    selector, qualifier, declared_training_authority = _parse_authorities(
        value["independent_authorities"], capsule)
    if declared_training_authority != training_authority:
        _fail("qualification receipt training authority 与 inventory 不一致")
    population = _parse_population(value["selection"], records)
    _parse_license_review(value["license_review"], population=population,
                          records=records)
    _validate_disjointness(
        value["training_disjointness"], population=population,
        training_entries=training_entries,
        training_manifest_sha256=training_manifest_sha256)
    return selector, qualifier, declared_training_authority


def _validate_trust_anchor(
        trust_anchor: ConversationHeldOutV4QualificationTrustAnchor,
        *, selector: ProtocolKey, qualifier: ProtocolKey,
        training_authority: ProtocolKey,
        producer: ProtocolKey, receipt_payload: bytes,
        training_manifest_payload: bytes,
        ) -> None:
    """把 receipt/inventory 与调用方传入的 anchor 逐项对齐，不认证该 anchor。"""
    if not isinstance(trust_anchor, ConversationHeldOutV4QualificationTrustAnchor):
        raise TypeError("trust_anchor 必须是 ConversationHeldOutV4QualificationTrustAnchor")
    if trust_anchor.qualifier_key != qualifier:
        _fail("trust anchor qualifier key 与 receipt 不一致")
    if trust_anchor.training_inventory_authority_key != training_authority:
        _fail("trust anchor training authority 与 receipt 不一致")
    if trust_anchor.qualification_receipt_sha256 != _sha256(receipt_payload):
        _fail("trust anchor qualification receipt SHA 不一致")
    if trust_anchor.training_inventory_manifest_sha256 != _sha256(
            training_manifest_payload):
        _fail("trust anchor training manifest SHA 不一致")
    if trust_anchor.trust_anchor_key in {
            producer, selector, qualifier, training_authority}:
        _fail("trust anchor key 不得与 producer/selector/qualifier/training authority 相同")


def _read_capsule(
        root: Path, *, require_k_drive: bool,
        ) -> ConversationHeldOutV4RuntimeSourceCapsule:
    """只通过 R02 reader 导入 external capsule，不能手工构造来源 origin。"""
    try:
        capsule = read_v4_external_input_capsule(
            root, require_k_drive=require_k_drive)
    except ConversationHeldOutV4ExternalCapsuleError as exc:
        _fail(f"external source capsule 不可验证: {exc}")
        raise AssertionError from exc
    if capsule.origin != V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL or not capsule.is_external:
        _fail("source qualification 只接受 external source capsule")
    return capsule


def _audit_status_for_transport(*, require_k_drive: bool) -> str:
    """返回 transport 状态；K 盘隔离不能把普通声明升级为外部资格。"""
    if type(require_k_drive) is not bool:
        raise TypeError("require_k_drive 必须是 bool")
    return (V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS if require_k_drive
            else V4_TEST_ONLY_QUALIFICATION_STATUS)


def audit_v4_independent_source_qualification(
        source_capsule_root: str | Path,
        qualification_root: str | Path,
        training_provenance_root: str | Path,
        trust_anchor: ConversationHeldOutV4QualificationTrustAnchor,
        *,
        require_k_drive: bool = True,
        ) -> ConversationHeldOutV4SourceQualificationAudit:
    """只读审计来源/训练/许可声明的机械一致性，任何成功都不授予资格。

    K 盘只是大运行数据隔离边界，不能证明外部身份、输入谱系、官方许可或快照不可变。
    ``require_k_drive=False`` 只用于 test transport；两种模式的结果都没有 owner、formal、
    held-out 或能力含义。
    """
    source_root = _read_root(
        source_capsule_root, require_k_drive=require_k_drive,
        label="source capsule root")
    qualification = _read_root(
        qualification_root, require_k_drive=require_k_drive,
        label="qualification root")
    training = _read_root(
        training_provenance_root, require_k_drive=require_k_drive,
        label="training provenance root")
    _require_disjoint((source_root, qualification, training))
    capsule = _read_capsule(source_root, require_k_drive=require_k_drive)
    records = _source_records(capsule)
    qualification_document, receipt_payload, _receipt_ints, qualification_manifest = (
        _read_external_closure(
            qualification,
            body_file=_QUALIFICATION_RECEIPT_FILE,
            ints_file=_QUALIFICATION_INTS_FILE,
            artifact_kind=V4_SOURCE_QUALIFICATION_RECEIPT_KIND,
            schema=V4_SOURCE_QUALIFICATION_MANIFEST_SCHEMA,
            label="qualification receipt"))
    training_document, _training_payload, _training_ints, training_manifest = (
        _read_external_closure(
            training,
            body_file=_TRAINING_PROVENANCE_FILE,
            ints_file=_TRAINING_INTS_FILE,
            artifact_kind=V4_TRAINING_PROVENANCE_KIND,
            schema=V4_TRAINING_PROVENANCE_MANIFEST_SCHEMA,
            label="training provenance",
            max_body_bytes=V4_MAX_TRAINING_PROVENANCE_PAYLOAD_BYTES,
            max_ints_bytes=V4_MAX_TRAINING_PROVENANCE_PAYLOAD_BYTES))
    training_entries, training_authority = _parse_training_inventory(
        training_document)
    selector, qualifier, declared_training_authority = _parse_qualification_receipt(
        qualification_document,
        capsule=capsule,
        records=records,
        training_entries=training_entries,
        training_authority=training_authority,
        training_manifest_sha256=_sha256(training_manifest))
    if declared_training_authority != training_authority:
        _fail("qualification receipt training authority 漂移")
    if capsule.external_producer_key is None:
        _fail("external capsule 缺 producer key")
    _validate_trust_anchor(
        trust_anchor,
        selector=selector,
        qualifier=qualifier,
        training_authority=training_authority,
        producer=capsule.external_producer_key,
        receipt_payload=receipt_payload,
        training_manifest_payload=training_manifest)

    # 二次回读仅尽力发现审计期间漂移；真正的不可变性需由外部快照/ACL 提供。
    capsule_after = _read_capsule(source_root, require_k_drive=require_k_drive)
    qualification_after = _read_external_closure(
        qualification,
        body_file=_QUALIFICATION_RECEIPT_FILE,
        ints_file=_QUALIFICATION_INTS_FILE,
        artifact_kind=V4_SOURCE_QUALIFICATION_RECEIPT_KIND,
        schema=V4_SOURCE_QUALIFICATION_MANIFEST_SCHEMA,
        label="qualification receipt")
    training_after = _read_external_closure(
        training,
        body_file=_TRAINING_PROVENANCE_FILE,
        ints_file=_TRAINING_INTS_FILE,
        artifact_kind=V4_TRAINING_PROVENANCE_KIND,
        schema=V4_TRAINING_PROVENANCE_MANIFEST_SCHEMA,
        label="training provenance",
        max_body_bytes=V4_MAX_TRAINING_PROVENANCE_PAYLOAD_BYTES,
        max_ints_bytes=V4_MAX_TRAINING_PROVENANCE_PAYLOAD_BYTES)
    if (capsule_after != capsule
            or qualification_after != (
                qualification_document, receipt_payload, _receipt_ints,
                qualification_manifest)
            or training_after != (
                training_document, _training_payload, _training_ints,
                training_manifest)):
        _fail("source/qualification/training artifact 在 audit 期间漂移")
    return ConversationHeldOutV4SourceQualificationAudit(
        capsule,
        _sha256(receipt_payload),
        _sha256(qualification_manifest),
        _sha256(training_manifest),
        _audit_status_for_transport(require_k_drive=require_k_drive),
        V4_SOURCE_QUALIFICATION_AUDIT_SCOPE,
        len(records),
        len(capsule.inputs),
    )


__all__ = [
    "ConversationHeldOutV4QualificationTrustAnchor",
    "ConversationHeldOutV4SourceQualificationAudit",
    "ConversationHeldOutV4SourceQualificationError",
    "ConversationHeldOutV4TrainingProvenanceEntry",
    "V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS",
    "V4_MAX_EXTERNAL_CLOSURE_MANIFEST_BYTES",
    "V4_MAX_TRAINING_PROVENANCE_ENTRIES",
    "V4_MAX_TRAINING_PROVENANCE_PAYLOAD_BYTES",
    "V4_SELECTION_RULE",
    "V4_SOURCE_QUALIFICATION_MANIFEST_SCHEMA",
    "V4_SOURCE_QUALIFICATION_RECEIPT_KIND",
    "V4_SOURCE_QUALIFICATION_RECEIPT_SCHEMA",
    "V4_SOURCE_QUALIFICATION_AUDIT_SCOPE",
    "V4_TEST_ONLY_QUALIFICATION_STATUS",
    "V4_TRAINING_COVERAGE",
    "V4_TRAINING_EXCLUSION_RULE",
    "V4_TRAINING_PROVENANCE_MANIFEST_SCHEMA",
    "V4_TRAINING_PROVENANCE_KIND",
    "V4_TRAINING_PROVENANCE_SCHEMA",
    "V4_TRAINING_UNIVERSE_EMPTY",
    "V4_TRAINING_UNIVERSE_NONEMPTY",
    "audit_v4_independent_source_qualification",
]
