"""DLG-05 v4 P2-A 的有界 typed provenance stream catalog。

本模块只核验已显式声明的 K run root 内 P0 分帧 node/leaf shard。它不做 parent
DAG join、三路投影、merge、resume 或发布；因此返回的 catalog 只说明输入的物理及
规范编码机械闭合，不能说明来源资格、训练覆盖或任何能力结果。

每个实际读取都由 K 边界打开已复核的二进制 handle，再转交给 P0 reader。catalog 不
收集全部 node、leaf 或 parent 关系；跨 shard 排序只保留当前的一个逻辑键。leaf 到 node
snapshot 的 SourceRef endpoint 必须由后续外排 DAG join 在磁盘上核验，不能在这里以
内存字典替代。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import BinaryIO

from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4LineageNode,
    ConversationHeldOutV4ProvenanceError,
    ConversationHeldOutV4ProvenanceLeaf,
)
from pure_integer_ai.cognition.shared.unicode_representation import (
    validate_unicode_scalars,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerFramedStreamError,
    IntegerFramedStreamFooter,
    IntegerFramedStreamReader,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    KRunRoot,
    capture_plain_file_identity,
    open_plain_binary,
    relative_path,
    require_plain_file_identity,
)


V4_PROVENANCE_STREAM_KIND_NODE = 1
V4_PROVENANCE_STREAM_KIND_LEAF = 2
V4_PROVENANCE_CATALOG_BUDGET_SCHEMA = 1
V4_PROVENANCE_STREAM_IDENTITY_SCHEMA = 1
V4_PROVENANCE_STREAM_CATALOG_SCHEMA = 1

_STREAM_KINDS = frozenset({
    V4_PROVENANCE_STREAM_KIND_NODE,
    V4_PROVENANCE_STREAM_KIND_LEAF,
})
_SHA256_SIZE = hashlib.sha256().digest_size


# object-model: exception
class ConversationHeldOutV4ProvenanceScalableError(RuntimeError):
    """P2-A catalog 输入不闭合、超预算或在读取期间发生物理漂移。"""


def _fail(message: str) -> None:
    """统一产生 fail-closed 的 P2-A catalog 错误。"""
    raise ConversationHeldOutV4ProvenanceScalableError(message)


def _require_nonnegative_int(value: int, *, label: str) -> int:
    """拒绝 bool 和隐式数值，固定所有 catalog 资源预算为严格整数。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _require_positive_int(value: int, *, label: str) -> int:
    """固定必须能容纳至少一个元数据项的正整数预算。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def _require_relative_path(value: Path, *, label: str) -> Path:
    """拒绝裸字符串、根路径和向上逃逸，保留声明的相对 shard 路径。"""
    if not isinstance(value, Path):
        raise TypeError(f"{label} 必须是 Path")
    if (value.is_absolute() or value.drive or value.root or not value.parts
            or ".." in value.parts
            or any(part in {"", ".", ".."} for part in value.parts)):
        raise ValueError(f"{label} 必须是非空且不含 .. 的相对 Path")
    return value


def _relative_identity(value: Path) -> str:
    """返回宿主文件系统语义下用于拒绝重复声明的相对路径身份。"""
    return os.path.normcase(str(value))


def _relative_path_integer_key(value: Path) -> tuple[int, ...]:
    """把相对路径转为规范 POSIX 分隔的 Unicode scalar key，绝不包含 root。"""
    _require_relative_path(value, label="catalog identity relative path")
    text = os.path.normcase(value.as_posix())
    if os.sep == "\\":
        text = text.replace("\\", "/")
    scalars = tuple(ord(character) for character in text)
    try:
        return validate_unicode_scalars(scalars)
    except (TypeError, ValueError) as exc:
        raise ValueError("catalog identity relative path 含非法 Unicode scalar") from exc


def _require_sha256(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """校验 catalog 保存的完整物理 SHA-256 字节 tuple。"""
    try:
        strict_integer_tuple(value, label=label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是严格整数 tuple") from exc
    if len(value) != _SHA256_SIZE or any(item < 0 or item > 255 for item in value):
        raise ValueError(f"{label} 必须是完整 SHA-256 字节 tuple")
    return value


def _kind_label(kind: int) -> str:
    """把固定 typed stream tag 映射为仅用于诊断的稳定名称。"""
    if kind == V4_PROVENANCE_STREAM_KIND_NODE:
        return "node"
    if kind == V4_PROVENANCE_STREAM_KIND_LEAF:
        return "leaf"
    raise AssertionError("未注册的 stream kind")


# object-model: resource_owner; representation=protocol; interop=pending
class _BoundedHashingBinaryReader:
    """接管一个 K-boundary 句柄，并在同一 P0 解码遍历中计算物理身份。"""

    __slots__ = ("_byte_count", "_digest", "_max_bytes", "_stream")

    def __init__(self, stream: BinaryIO, *, max_bytes: int) -> None:
        """固定物理读取上限；超过上限时在读取该字节后立即 fail closed。"""
        self._byte_count = 0
        self._digest = hashlib.sha256()
        self._max_bytes = _require_nonnegative_int(
            max_bytes, label="catalog physical stream read limit")
        self._stream = stream

    @property
    def byte_count(self) -> int:
        """返回已经通过同一打开 handle 读取的物理字节数。"""
        return self._byte_count

    @property
    def sha256(self) -> tuple[int, ...]:
        """返回当前已读取物理字节的完整 SHA-256，不替代 P0 payload footer。"""
        return tuple(self._digest.digest())

    def read(self, size: int = -1) -> bytes:
        """有界转发读取并把实际 bytes 纳入物理 digest，绝不缓存完整文件。"""
        if type(size) is not int:
            raise TypeError("catalog physical stream read size 必须是严格整数")
        remaining_with_sentinel = self._max_bytes - self._byte_count + 1
        requested = min(
            size if size >= 0 else remaining_with_sentinel,
            remaining_with_sentinel,
        )
        payload = self._stream.read(requested)
        if not isinstance(payload, bytes):
            return payload
        next_count = self._byte_count + len(payload)
        if next_count > self._max_bytes:
            _fail("catalog shard 物理字节超过预算")
        self._digest.update(payload)
        self._byte_count = next_count
        return payload

    def close(self) -> None:
        """释放唯一被 P0 reader 接管的底层 K-boundary handle。"""
        self._stream.close()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCatalogBudget:
    """P2-A catalog 的小型元数据和单 shard 流式读取上限。"""

    max_total_shards: int
    max_physical_bytes_per_shard: int
    max_total_physical_bytes: int
    max_frame_bytes: int
    max_records_per_stream: int
    max_total_payload_bytes_per_stream: int

    def __post_init__(self) -> None:
        """固定 shard、物理字节和 P0 reader 的全部显式预算。"""
        _require_positive_int(
            self.max_total_shards, label="catalog max_total_shards")
        _require_positive_int(
            self.max_physical_bytes_per_shard,
            label="catalog max_physical_bytes_per_shard",
        )
        _require_positive_int(
            self.max_total_physical_bytes,
            label="catalog max_total_physical_bytes",
        )
        _require_nonnegative_int(
            self.max_frame_bytes, label="catalog max_frame_bytes")
        _require_nonnegative_int(
            self.max_records_per_stream,
            label="catalog max_records_per_stream",
        )
        _require_nonnegative_int(
            self.max_total_payload_bytes_per_stream,
            label="catalog max_total_payload_bytes_per_stream",
        )

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含路径或 capability 的固定字段规范预算 identity。"""
        return (
            V4_PROVENANCE_CATALOG_BUDGET_SCHEMA,
            self.max_total_shards,
            self.max_physical_bytes_per_shard,
            self.max_total_physical_bytes,
            self.max_frame_bytes,
            self.max_records_per_stream,
            self.max_total_payload_bytes_per_stream,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回可由后续 receipt/cursor 绑定的完整 schema-backed 预算键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCatalogInput:
    """P2-A 的显式 K capability、分类型有序 shard 清单和读取预算。"""

    root: KRunRoot
    node_relative_paths: tuple[Path, ...]
    leaf_relative_paths: tuple[Path, ...]
    budget: ConversationHeldOutV4ProvenanceCatalogBudget

    def __post_init__(self) -> None:
        """拒绝裸路径 root、未分型清单、空 node 输入和超出小 catalog 的声明。"""
        if not isinstance(self.root, KRunRoot):
            raise TypeError("catalog root 必须是 KRunRoot capability")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceCatalogBudget):
            raise TypeError("catalog budget 类型错误")
        if not isinstance(self.node_relative_paths, tuple):
            raise TypeError("catalog node_relative_paths 必须是 Path tuple")
        if not isinstance(self.leaf_relative_paths, tuple):
            raise TypeError("catalog leaf_relative_paths 必须是 Path tuple")
        if not self.node_relative_paths:
            _fail("catalog 至少需要一个 node shard")
        for label, values in (
                ("catalog node_relative_paths", self.node_relative_paths),
                ("catalog leaf_relative_paths", self.leaf_relative_paths)):
            for value in values:
                _require_relative_path(value, label=label)
        total_shards = len(self.node_relative_paths) + len(self.leaf_relative_paths)
        if total_shards > self.budget.max_total_shards:
            _fail("catalog shard 数超过预算")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceStreamIdentity:
    """一个已完整消费、typed 且物理身份固定的 sealed P0 shard。"""

    stream_kind: int
    relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical_byte_count: int
    physical_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """保存相对定位、完整 P0 footer 与不替代 footer 的物理 SHA/size。"""
        if type(self.stream_kind) is not int or self.stream_kind not in _STREAM_KINDS:
            raise ValueError("stream identity kind 未注册")
        _require_relative_path(self.relative_path, label="stream identity path")
        if not isinstance(self.p0_footer, IntegerFramedStreamFooter):
            raise TypeError("stream identity p0_footer 类型错误")
        _require_positive_int(
            self.physical_byte_count,
            label="stream identity physical_byte_count",
        )
        _require_sha256(
            self.physical_sha256,
            label="stream identity physical_sha256",
        )

    def integer_stream(self) -> tuple[int, ...]:
        """返回仅含 typed 相对路径、完整 footer 与物理身份的规范整数键。"""
        result = [V4_PROVENANCE_STREAM_IDENTITY_SCHEMA, self.stream_kind]
        for item in (
                _relative_path_integer_key(self.relative_path),
                self.p0_footer.integer_tuple(),
                self.physical_sha256):
            pack_key(result, item)
        result.append(self.physical_byte_count)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 KRunRoot 绝对路径的完整 stream identity 键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceStreamCatalog:
    """P2-A 的有界、分型且不包含全量 record 的输入 catalog。"""

    root: KRunRoot
    node_streams: tuple[ConversationHeldOutV4ProvenanceStreamIdentity, ...]
    leaf_streams: tuple[ConversationHeldOutV4ProvenanceStreamIdentity, ...]
    total_physical_byte_count: int
    budget: ConversationHeldOutV4ProvenanceCatalogBudget

    def __post_init__(self) -> None:
        """复核 typed stream 分离、路径唯一、物理字节和冻结预算的可重算性。"""
        if not isinstance(self.root, KRunRoot):
            raise TypeError("catalog root 必须是 KRunRoot capability")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceCatalogBudget):
            raise TypeError("catalog budget 类型错误")
        if not isinstance(self.node_streams, tuple):
            raise TypeError("catalog node_streams 必须是 stream identity tuple")
        if not isinstance(self.leaf_streams, tuple):
            raise TypeError("catalog leaf_streams 必须是 stream identity tuple")
        if not self.node_streams:
            _fail("catalog 至少需要一个 node stream")
        for expected_kind, label, values in (
                (V4_PROVENANCE_STREAM_KIND_NODE,
                 "catalog node_streams", self.node_streams),
                (V4_PROVENANCE_STREAM_KIND_LEAF,
                 "catalog leaf_streams", self.leaf_streams)):
            if any(not isinstance(item, ConversationHeldOutV4ProvenanceStreamIdentity)
                   for item in values):
                raise TypeError(f"{label} 含非 stream identity")
            if any(item.stream_kind != expected_kind for item in values):
                _fail(f"{label} 含错误的 typed stream identity")
        all_streams = self.node_streams + self.leaf_streams
        path_identities = tuple(
            _relative_identity(item.relative_path) for item in all_streams)
        if len(set(path_identities)) != len(path_identities):
            _fail("catalog node/leaf stream path 不得重复")
        _require_nonnegative_int(
            self.total_physical_byte_count,
            label="catalog total_physical_byte_count",
        )
        if self.total_physical_byte_count != sum(
                item.physical_byte_count for item in all_streams):
            _fail("catalog total_physical_byte_count 与 shard 身份不一致")
        if len(all_streams) > self.budget.max_total_shards:
            _fail("catalog stream 数超过冻结预算")
        if self.total_physical_byte_count > self.budget.max_total_physical_bytes:
            _fail("catalog 物理字节超过冻结预算")
        for item in all_streams:
            if item.physical_byte_count > self.budget.max_physical_bytes_per_shard:
                _fail("catalog shard 物理字节超过冻结预算")
            if item.p0_footer.record_count > self.budget.max_records_per_stream:
                _fail("catalog shard record 数超过冻结预算")
            if (item.p0_footer.total_payload_bytes
                    > self.budget.max_total_payload_bytes_per_stream):
                _fail("catalog shard payload 字节超过冻结预算")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 root 绝对路径的完整 catalog schema-backed 整数 identity。"""
        result = [V4_PROVENANCE_STREAM_CATALOG_SCHEMA]
        pack_key(result, self.budget.integer_stream())
        result.append(len(self.node_streams))
        for item in self.node_streams:
            pack_key(result, item.integer_stream())
        result.append(len(self.leaf_streams))
        for item in self.leaf_streams:
            pack_key(result, item.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回可供后续计划绑定的无 root canonical catalog identity。"""
        return self.integer_stream()


def _validate_input_path_closure(
        value: ConversationHeldOutV4ProvenanceCatalogInput,
        ) -> None:
    """逐项核验显式清单的词法闭包，拒绝重复、裸目录或 root 外定位。"""
    seen: set[str] = set()
    for kind, paths in (
            (V4_PROVENANCE_STREAM_KIND_NODE, value.node_relative_paths),
            (V4_PROVENANCE_STREAM_KIND_LEAF, value.leaf_relative_paths)):
        kind_label = _kind_label(kind)
        for index, relative in enumerate(paths):
            label = f"catalog {kind_label} shard[{index}]"
            _require_relative_path(relative, label=label)
            try:
                absolute = relative_path(value.root, relative, label=label)
            except KRunBoundaryError as exc:
                raise ConversationHeldOutV4ProvenanceScalableError(
                    f"{label} 路径不在可用 K run root 内") from exc
            identity = os.path.normcase(str(absolute))
            if identity in seen:
                _fail("catalog node/leaf shard path 不得重复")
            seen.add(identity)


def _decode_sort_key(
        record: tuple[int, ...], *, stream_kind: int, label: str,
        ) -> tuple[int, ...]:
    """把当前 P0 record 严格解码为对应 P1 本体并返回其排序身份。"""
    try:
        if stream_kind == V4_PROVENANCE_STREAM_KIND_NODE:
            node = ConversationHeldOutV4LineageNode.from_integer_stream(record)
            return node.node_key.components
        if stream_kind == V4_PROVENANCE_STREAM_KIND_LEAF:
            leaf = ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(record)
            return leaf.stable_key()
    except (ConversationHeldOutV4ProvenanceError, AssertionError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ProvenanceScalableError(
            f"{label} P1 {_kind_label(stream_kind)} record 非法") from exc
    raise AssertionError("未注册的 stream kind")


def _validate_stream_order(
        previous_key: tuple[int, ...] | None,
        current_key: tuple[int, ...],
        *, label: str, is_first_record: bool,
        ) -> tuple[int, ...]:
    """以严格递增的 typed identity 同时拒绝 shard 内及跨 shard 重复或倒序。"""
    if previous_key is not None and current_key <= previous_key:
        boundary = "跨 shard" if is_first_record else "shard 内"
        _fail(f"{label} {boundary} typed identity 未严格递增")
    return current_key


def _read_sealed_typed_stream(
        value: ConversationHeldOutV4ProvenanceCatalogInput,
        *,
        stream_kind: int,
        relative: Path,
        index: int,
        total_physical_before: int,
        previous_global_key: tuple[int, ...] | None,
        ) -> tuple[
            ConversationHeldOutV4ProvenanceStreamIdentity,
            int,
            tuple[int, ...] | None,
        ]:
    """经 K-boundary handle 流式验证一个 sealed P0 stream，不聚合其 records。

    P0 reader 只接管 ``open_plain_binary`` 已打开的 handle，绝不在 P2 内以 path 再次
    打开文件。物理 SHA 在这一次解码读取中同步累计，读取前后只捕获/复核 O(1) 文件
    identity；该机械复核不替代受控 ACL 或不可变 snapshot 证明。
    """
    kind_label = _kind_label(stream_kind)
    label = f"catalog {kind_label} shard[{index}]"
    remaining_total = (
        value.budget.max_total_physical_bytes - total_physical_before)
    if remaining_total < 0:
        _fail("catalog 累计物理字节已超过预算")
    physical_limit = min(
        value.budget.max_physical_bytes_per_shard,
        remaining_total,
    )
    try:
        physical_identity = capture_plain_file_identity(
            value.root,
            relative,
            label=label,
        )
        if physical_identity.byte_count > physical_limit:
            _fail("catalog shard 物理字节超过预算")
        path = require_plain_file_identity(
            value.root,
            relative,
            physical_identity,
            label=label,
        )
        current_global_key = previous_global_key
        record_count = 0
        with open_plain_binary(
                value.root,
                relative,
                label=label,
                expected_identity=physical_identity,
        ) as stream:
            hashing_stream = _BoundedHashingBinaryReader(
                stream,
                max_bytes=physical_limit,
            )
            with IntegerFramedStreamReader.from_open_binary(
                    hashing_stream,
                    path=path,
                    max_frame_bytes=value.budget.max_frame_bytes,
                    max_record_count=value.budget.max_records_per_stream,
                    max_total_payload_bytes=(
                        value.budget.max_total_payload_bytes_per_stream),
            ) as reader:
                for record in reader:
                    current_key = _decode_sort_key(
                        record,
                        stream_kind=stream_kind,
                        label=f"{label} record[{record_count}]",
                    )
                    current_global_key = _validate_stream_order(
                        current_global_key,
                        current_key,
                        label=f"{label} record[{record_count}]",
                        is_first_record=record_count == 0,
                    )
                    record_count += 1
                footer = reader.finish()
                if (footer.record_count != record_count
                        or footer.total_payload_bytes
                        != reader.total_payload_bytes):
                    _fail(f"{label} P0 footer 与流式计数不一致")
        physical_from_handle_byte_count = hashing_stream.byte_count
        physical_from_handle_sha256 = hashing_stream.sha256
        require_plain_file_identity(
            value.root,
            relative,
            physical_identity,
            label=label,
        )
    except ConversationHeldOutV4ProvenanceScalableError:
        raise
    except (KRunBoundaryError, IntegerFramedStreamError, OSError, ValueError,
            TypeError) as exc:
        raise ConversationHeldOutV4ProvenanceScalableError(
            f"{label} 未通过 sealed P0 typed stream 复核") from exc
    if physical_from_handle_byte_count != physical_identity.byte_count:
        _fail(f"{label} 打开 handle 的物理字节数与捕获 identity 漂移")
    total_physical_after = (
        total_physical_before + physical_from_handle_byte_count)
    if total_physical_after > value.budget.max_total_physical_bytes:
        _fail("catalog 累计物理字节超过预算")
    identity = ConversationHeldOutV4ProvenanceStreamIdentity(
        stream_kind,
        relative,
        footer,
        physical_from_handle_byte_count,
        physical_from_handle_sha256,
    )
    return identity, total_physical_after, current_global_key


def build_v4_provenance_stream_catalog(
        value: ConversationHeldOutV4ProvenanceCatalogInput,
        ) -> ConversationHeldOutV4ProvenanceStreamCatalog:
    """流式验证 P2-A node/leaf shard catalog 并返回最小物理输入身份。

    ``node_relative_paths`` 与 ``leaf_relative_paths`` 的 tuple 顺序就是固定的逻辑
    shard 顺序，调用方不得依赖路径名排序或目录扫描。每个 node stream 按完整 node key
    严格递增，每个 leaf stream 按完整 leaf key 严格递增；同一 typed identity 跨 shard
    重复同样拒绝。P2-A 故意不检查 leaf 到 node snapshot 的 SourceRef endpoint，后续
    DAG join 必须以有界外排方式完成该检查。
    """
    if not isinstance(value, ConversationHeldOutV4ProvenanceCatalogInput):
        raise TypeError("catalog input 必须是 ConversationHeldOutV4ProvenanceCatalogInput")
    _validate_input_path_closure(value)
    total_physical_bytes = 0
    node_streams: list[ConversationHeldOutV4ProvenanceStreamIdentity] = []
    leaf_streams: list[ConversationHeldOutV4ProvenanceStreamIdentity] = []
    node_previous_key: tuple[int, ...] | None = None
    leaf_previous_key: tuple[int, ...] | None = None
    for index, relative in enumerate(value.node_relative_paths):
        identity, total_physical_bytes, node_previous_key = (
            _read_sealed_typed_stream(
                value,
                stream_kind=V4_PROVENANCE_STREAM_KIND_NODE,
                relative=relative,
                index=index,
                total_physical_before=total_physical_bytes,
                previous_global_key=node_previous_key,
            )
        )
        node_streams.append(identity)
    for index, relative in enumerate(value.leaf_relative_paths):
        identity, total_physical_bytes, leaf_previous_key = (
            _read_sealed_typed_stream(
                value,
                stream_kind=V4_PROVENANCE_STREAM_KIND_LEAF,
                relative=relative,
                index=index,
                total_physical_before=total_physical_bytes,
                previous_global_key=leaf_previous_key,
            )
        )
        leaf_streams.append(identity)
    return ConversationHeldOutV4ProvenanceStreamCatalog(
        value.root,
        tuple(node_streams),
        tuple(leaf_streams),
        total_physical_bytes,
        value.budget,
    )


def revalidate_v4_provenance_stream_catalog(
        value: ConversationHeldOutV4ProvenanceStreamCatalog,
        ) -> ConversationHeldOutV4ProvenanceStreamCatalog:
    """按 catalog 自带 capability、typed 路径和冻结预算完整重读并拒绝任何 identity 漂移。

    此入口不接受可替换的裸路径或新预算。它只重建同形输入，重新流式验证全部 P0/P1
    shard，随后要求无 root canonical stable key 逐整数一致；因此 resume 或阶段间复用
    不能悄悄增大预算、替换 footer/物理身份或改变 shard 顺序。
    """
    if not isinstance(value, ConversationHeldOutV4ProvenanceStreamCatalog):
        raise TypeError("catalog revalidation 必须接收 ConversationHeldOutV4ProvenanceStreamCatalog")
    rebuilt = build_v4_provenance_stream_catalog(
        ConversationHeldOutV4ProvenanceCatalogInput(
            value.root,
            tuple(item.relative_path for item in value.node_streams),
            tuple(item.relative_path for item in value.leaf_streams),
            value.budget,
        ))
    if rebuilt.stable_key() != value.stable_key():
        _fail("catalog revalidation stable key 与冻结输入不一致")
    return rebuilt


__all__ = [
    "ConversationHeldOutV4ProvenanceCatalogBudget",
    "ConversationHeldOutV4ProvenanceCatalogInput",
    "ConversationHeldOutV4ProvenanceScalableError",
    "ConversationHeldOutV4ProvenanceStreamCatalog",
    "ConversationHeldOutV4ProvenanceStreamIdentity",
    "V4_PROVENANCE_CATALOG_BUDGET_SCHEMA",
    "V4_PROVENANCE_STREAM_CATALOG_SCHEMA",
    "V4_PROVENANCE_STREAM_IDENTITY_SCHEMA",
    "V4_PROVENANCE_STREAM_KIND_LEAF",
    "V4_PROVENANCE_STREAM_KIND_NODE",
    "build_v4_provenance_stream_catalog",
    "revalidate_v4_provenance_stream_catalog",
]
