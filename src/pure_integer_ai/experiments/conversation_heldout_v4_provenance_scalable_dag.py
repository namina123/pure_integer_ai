"""DLG-05 v4 R04b P2-B1 的有界直接谱系闭合。

本模块只消费 P2-A 已封存的 typed catalog，并在一个与 source root 完全不相交的空
K run root 内构造未发布的 P0 流。它完成直接 parent endpoint 与 leaf endpoint 的精确
闭合；不计算传递祖先、不物化投影、不合并训练/held-out，也不产生 cursor、receipt 或
manifest。

每次再次读取 catalog 或本模块上游 P0 流，都会比较完整 footer、物理字节数和 SHA-256。
因此可观察到的输入替换、截断或重编码只能导致 fail-closed 的残片，不能返回 B1 结果。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import BinaryIO

from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4LineageNode,
    ConversationHeldOutV4ProvenanceError,
    ConversationHeldOutV4ProvenanceLeaf,
    ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_LINEAGE_NODE_SCHEMA,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable import (
    ConversationHeldOutV4ProvenanceStreamCatalog,
    ConversationHeldOutV4ProvenanceStreamIdentity,
    revalidate_v4_provenance_stream_catalog,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerFramedStreamBudgetExceeded,
    IntegerFramedStreamFooter,
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
    IntegerStreamReader,
    encoded_integer_tuple_size,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.integer_external_sort import (
    IntegerExternalSortBudget,
    IntegerExternalSortBudgetExceeded,
    IntegerExternalSortError,
    IntegerExternalSortInputIdentity,
    IntegerExternalSortResult,
    external_sort_sealed_integer_records,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunFileDigest,
    KRunRoot,
    capture_plain_file_identity,
    create_new_run_root,
    open_exclusive_binary,
    open_plain_binary,
    require_disjoint_run_roots,
    require_fresh_empty_run_root,
    require_plain_file_identity,
)


V4_PROVENANCE_DAG_BUDGET_SCHEMA = 1
V4_PROVENANCE_DAG_STREAM_DESCRIPTOR_SCHEMA = 1
V4_PROVENANCE_DAG_RESULT_SCHEMA = 1

V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT = 1
V4_PROVENANCE_DAG_STREAM_PARENT_REQUEST = 2
V4_PROVENANCE_DAG_STREAM_SORTED_PARENT_REQUEST = 3
V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE = 4
V4_PROVENANCE_DAG_STREAM_SORTED_LEAF = 5
V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT = 6

V4_PROVENANCE_DAG_NODE_ENDPOINT_RECORD_SCHEMA = 1
V4_PROVENANCE_DAG_PARENT_REQUEST_RECORD_SCHEMA = 1
V4_PROVENANCE_DAG_DIRECT_EDGE_RECORD_SCHEMA = 1
V4_PROVENANCE_DAG_LEAF_ENDPOINT_RECORD_SCHEMA = 1

_STREAM_KINDS = frozenset({
    V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT,
    V4_PROVENANCE_DAG_STREAM_PARENT_REQUEST,
    V4_PROVENANCE_DAG_STREAM_SORTED_PARENT_REQUEST,
    V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE,
    V4_PROVENANCE_DAG_STREAM_SORTED_LEAF,
    V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT,
})
_STAGE_NODE_REPLAY = "stage-01-node-replay"
_STAGE_PARENT_SORT = "stage-02-parent-sort"
_STAGE_DIRECT_EDGE = "stage-03-direct-edge"
_STAGE_LEAF_SORT = "stage-04-leaf-sort"
_STAGE_LEAF_ENDPOINT = "stage-05-leaf-endpoint"
_NODE_ENDPOINT_FILE = Path("node-endpoints.pifrs")
_PARENT_REQUEST_FILE = Path("parent-requests.pifrs")
_SORTED_PARENT_REQUEST_FILE = Path("parent-requests-by-parent.pifrs")
_DIRECT_EDGE_FILE = Path("direct-edges.pifrs")
_SORTED_LEAF_FILE = Path("leaves-by-lineage-node.pifrs")
_LEAF_ENDPOINT_FILE = Path("leaf-endpoints.pifrs")
_EXTERNAL_SORT_BUDGET_FIELD_COUNT = 16


# object-model: exception
class ConversationHeldOutV4ProvenanceScalableDagError(RuntimeError):
    """B1 直接 DAG 或 leaf endpoint 不能在冻结预算内机械闭合。"""


# object-model: exception
class ConversationHeldOutV4ProvenanceScalableDagBudgetExceeded(
        ConversationHeldOutV4ProvenanceScalableDagError):
    """B1 的显式 records、physical、handle 或累计物化预算被超过。"""


def _fail(message: str) -> None:
    """统一产生 B1 的领域 fail-closed 错误。"""
    raise ConversationHeldOutV4ProvenanceScalableDagError(message)


def _budget_fail(message: str) -> None:
    """统一产生预算耗尽错误，避免普通输入错误被误标为资源不足。"""
    raise ConversationHeldOutV4ProvenanceScalableDagBudgetExceeded(message)


def _require_positive_int(value: int, *, label: str) -> int:
    """固定预算与格式字段为拒绝 bool 的正严格整数。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def _require_nonnegative_int(value: int, *, label: str) -> int:
    """固定累计统计值为拒绝 bool 的非负严格整数。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _require_stage_name(value: str) -> str:
    """限制逻辑阶段名为可稳定传给外排且不含路径语义的 ASCII 标识。"""
    if not isinstance(value, str) or not value:
        raise TypeError("logical_stage_name 必须是非空 str")
    if len(value) > 56:
        raise ValueError("logical_stage_name 超过长度上限")
    if not value[0].isascii() or not value[0].isalnum():
        raise ValueError("logical_stage_name 必须以 ASCII 字母或数字开始")
    if any(not character.isascii() or not (
            character.isalnum() or character in {"-", "_", "."})
           for character in value):
        raise ValueError("logical_stage_name 只能包含 ASCII 字母、数字、-、_、.")
    return value


def _relative_path_key(value: Path, *, label: str) -> tuple[int, ...]:
    """把固定 work 相对路径映射为不包含本机 root 的 ASCII 整数键。"""
    if (not isinstance(value, Path) or value.is_absolute() or value.drive
            or value.root or not value.parts or ".." in value.parts
            or any(part in {"", ".", ".."} for part in value.parts)):
        raise ValueError(f"{label} 必须是非空且不含 .. 的相对 Path")
    text = value.as_posix()
    if any(not character.isascii() for character in text):
        raise ValueError(f"{label} 必须只含 ASCII")
    return tuple(ord(character) for character in text)


def _digest_integer_key(value: KRunFileDigest) -> tuple[int, ...]:
    """展开完整物理字节和 SHA-256，不以摘要替代 P0 footer 本体。"""
    if not isinstance(value, KRunFileDigest):
        raise TypeError("physical 必须是 KRunFileDigest")
    return (value.byte_count, *value.sha256)


def _external_sort_budget_integer_key(
        value: IntegerExternalSortBudget,
        ) -> tuple[int, ...]:
    """稳定展开通用外排的所有固定预算字段，避免把配置丢出结果 identity。"""
    if not isinstance(value, IntegerExternalSortBudget):
        raise TypeError("external_sort_budget 必须是 IntegerExternalSortBudget")
    result = (
        value.max_input_file_count,
        value.max_input_physical_bytes,
        value.max_input_record_count,
        value.max_input_payload_bytes,
        value.max_record_payload_bytes,
        value.max_batch_record_count,
        value.max_batch_payload_bytes,
        value.max_batch_sort_key_bytes,
        value.max_temporary_run_count,
        value.max_temporary_record_count,
        value.max_temporary_payload_bytes,
        value.max_temporary_physical_bytes,
        value.max_output_physical_bytes,
        value.merge_fan_in,
        value.max_open_files,
        value.max_merge_pass_count,
    )
    if len(result) != _EXTERNAL_SORT_BUDGET_FIELD_COUNT:
        raise AssertionError("external sort budget 字段数漂移")
    return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceDagBudget:
    """B1 重放、join、输出 P0 与外排的全量有限资源合同。"""

    max_node_count: int
    max_leaf_count: int
    max_parent_request_count: int
    max_direct_edge_count: int
    max_parents_per_node: int
    max_leaves_per_node: int
    max_record_payload_bytes: int
    max_stream_record_count: int
    max_stream_payload_bytes: int
    max_stream_physical_bytes: int
    max_total_materialized_physical_bytes: int
    max_open_files: int
    external_sort_budget: IntegerExternalSortBudget

    def __post_init__(self) -> None:
        """拒绝未显式限定的 P1 展开、P0 stream、join group 与外排配置。"""
        values = (
            ("max_node_count", self.max_node_count),
            ("max_leaf_count", self.max_leaf_count),
            ("max_parent_request_count", self.max_parent_request_count),
            ("max_direct_edge_count", self.max_direct_edge_count),
            ("max_parents_per_node", self.max_parents_per_node),
            ("max_leaves_per_node", self.max_leaves_per_node),
            ("max_record_payload_bytes", self.max_record_payload_bytes),
            ("max_stream_record_count", self.max_stream_record_count),
            ("max_stream_payload_bytes", self.max_stream_payload_bytes),
            ("max_stream_physical_bytes", self.max_stream_physical_bytes),
            ("max_total_materialized_physical_bytes",
             self.max_total_materialized_physical_bytes),
            ("max_open_files", self.max_open_files),
        )
        for label, item in values:
            _require_positive_int(item, label=label)
        if not isinstance(self.external_sort_budget, IntegerExternalSortBudget):
            raise TypeError("external_sort_budget 必须是 IntegerExternalSortBudget")
        if self.max_open_files < 3:
            raise ValueError("max_open_files 至少覆盖两个 reader 和一个 writer")
        if self.external_sort_budget.max_open_files > self.max_open_files:
            raise ValueError("external_sort_budget max_open_files 不得超过 B1 上限")
        if (self.external_sort_budget.max_output_physical_bytes
                > self.max_stream_physical_bytes):
            raise ValueError("external sort output 物理预算不得超过 B1 stream 上限")
        if self.max_record_payload_bytes > self.max_stream_payload_bytes:
            raise ValueError("单 record payload 不得超过 stream payload 预算")
        if self.max_node_count > self.max_stream_record_count:
            raise ValueError("node 数预算不得超过 stream record 预算")
        if self.max_parent_request_count > self.max_stream_record_count:
            raise ValueError("parent request 数预算不得超过 stream record 预算")
        if self.max_direct_edge_count > self.max_stream_record_count:
            raise ValueError("direct edge 数预算不得超过 stream record 预算")
        if self.max_leaf_count > self.max_stream_record_count:
            raise ValueError("leaf 数预算不得超过 stream record 预算")

    def integer_stream(self) -> tuple[int, ...]:
        """返回完整 B1 预算，不携带工作目录或环境身份。"""
        result = [V4_PROVENANCE_DAG_BUDGET_SCHEMA]
        result.extend((
            self.max_node_count,
            self.max_leaf_count,
            self.max_parent_request_count,
            self.max_direct_edge_count,
            self.max_parents_per_node,
            self.max_leaves_per_node,
            self.max_record_payload_bytes,
            self.max_stream_record_count,
            self.max_stream_payload_bytes,
            self.max_stream_physical_bytes,
            self.max_total_materialized_physical_bytes,
            self.max_open_files,
        ))
        pack_key(result, _external_sort_budget_integer_key(self.external_sort_budget))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回后续闭包或资源阶段可绑定的 schema-backed 预算键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceDagStreamDescriptor:
    """一个 B1 未发布 P0 输出的相对定位、footer 与物理身份。"""

    stream_kind: int
    work_relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """保留完整 P0/physical 本体且拒绝绝对工作根泄漏。"""
        if type(self.stream_kind) is not int or self.stream_kind not in _STREAM_KINDS:
            raise ValueError("B1 stream_kind 未注册")
        _relative_path_key(
            self.work_relative_path,
            label="B1 stream work_relative_path",
        )
        if not isinstance(self.p0_footer, IntegerFramedStreamFooter):
            raise TypeError("B1 stream p0_footer 类型错误")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("B1 stream physical 类型错误")
        if self.physical.byte_count <= 0:
            raise ValueError("B1 P0 stream 物理字节必须为正")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 K root 的完整 stream descriptor 整数 identity。"""
        result = [V4_PROVENANCE_DAG_STREAM_DESCRIPTOR_SCHEMA, self.stream_kind]
        for item in (
                _relative_path_key(
                    self.work_relative_path,
                    label="B1 stream work_relative_path"),
                self.p0_footer.integer_tuple(),
                _digest_integer_key(self.physical)):
            pack_key(result, item)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回可供后续 P2 切片比对的无绝对路径身份。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceDirectDagResult:
    """B1 的未发布 P0 descriptors、计数和预算，不声明资格或能力结论。"""

    logical_stage_name: str
    catalog_stable_key: tuple[int, ...]
    node_endpoint_stream: ConversationHeldOutV4ProvenanceDagStreamDescriptor
    parent_request_stream: ConversationHeldOutV4ProvenanceDagStreamDescriptor
    sorted_parent_request_stream: ConversationHeldOutV4ProvenanceDagStreamDescriptor
    direct_edge_stream: ConversationHeldOutV4ProvenanceDagStreamDescriptor
    sorted_leaf_stream: ConversationHeldOutV4ProvenanceDagStreamDescriptor
    leaf_endpoint_stream: ConversationHeldOutV4ProvenanceDagStreamDescriptor
    node_count: int
    parent_request_count: int
    direct_edge_count: int
    leaf_count: int
    leaf_endpoint_count: int
    materialized_physical_bytes: int
    budget: ConversationHeldOutV4ProvenanceDagBudget

    def __post_init__(self) -> None:
        """绑定每类输出、精确计数和实际物化量，禁止结果含 capability。"""
        _require_stage_name(self.logical_stage_name)
        try:
            strict_integer_tuple(
                self.catalog_stable_key,
                label="B1 catalog_stable_key",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("B1 catalog_stable_key 必须是完整严格整数 tuple") from exc
        expected = (
            (V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT,
             self.node_endpoint_stream),
            (V4_PROVENANCE_DAG_STREAM_PARENT_REQUEST,
             self.parent_request_stream),
            (V4_PROVENANCE_DAG_STREAM_SORTED_PARENT_REQUEST,
             self.sorted_parent_request_stream),
            (V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE,
             self.direct_edge_stream),
            (V4_PROVENANCE_DAG_STREAM_SORTED_LEAF,
             self.sorted_leaf_stream),
            (V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT,
             self.leaf_endpoint_stream),
        )
        for kind, descriptor in expected:
            if not isinstance(descriptor, ConversationHeldOutV4ProvenanceDagStreamDescriptor):
                raise TypeError("B1 result 含非 stream descriptor")
            if descriptor.stream_kind != kind:
                raise ValueError("B1 result stream descriptor 类型错位")
        values = (
            ("node_count", self.node_count),
            ("parent_request_count", self.parent_request_count),
            ("direct_edge_count", self.direct_edge_count),
            ("leaf_count", self.leaf_count),
            ("leaf_endpoint_count", self.leaf_endpoint_count),
            ("materialized_physical_bytes", self.materialized_physical_bytes),
        )
        for label, item in values:
            _require_nonnegative_int(item, label=f"B1 {label}")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceDagBudget):
            raise TypeError("B1 result budget 类型错误")
        if self.node_count != self.node_endpoint_stream.p0_footer.record_count:
            raise ValueError("B1 node_count 与 endpoint footer 不一致")
        if self.parent_request_count != self.parent_request_stream.p0_footer.record_count:
            raise ValueError("B1 parent request count 与 footer 不一致")
        if (self.parent_request_count
                != self.sorted_parent_request_stream.p0_footer.record_count):
            raise ValueError("B1 sorted parent request footer 不一致")
        if self.direct_edge_count != self.direct_edge_stream.p0_footer.record_count:
            raise ValueError("B1 direct edge count 与 footer 不一致")
        if self.leaf_count != self.sorted_leaf_stream.p0_footer.record_count:
            raise ValueError("B1 leaf count 与 sorted footer 不一致")
        if self.leaf_endpoint_count != self.leaf_endpoint_stream.p0_footer.record_count:
            raise ValueError("B1 leaf endpoint count 与 footer 不一致")
        if self.direct_edge_count != self.parent_request_count:
            raise ValueError("B1 每个 parent request 必须有一个 direct edge")
        if self.leaf_endpoint_count != self.leaf_count:
            raise ValueError("B1 每个 leaf 必须有一个 endpoint")
        if self.node_count > self.budget.max_node_count:
            raise ValueError("B1 node_count 超过冻结预算")
        if self.parent_request_count > self.budget.max_parent_request_count:
            raise ValueError("B1 parent_request_count 超过冻结预算")
        if self.direct_edge_count > self.budget.max_direct_edge_count:
            raise ValueError("B1 direct_edge_count 超过冻结预算")
        if self.leaf_count > self.budget.max_leaf_count:
            raise ValueError("B1 leaf_count 超过冻结预算")
        if self.materialized_physical_bytes > self.budget.max_total_materialized_physical_bytes:
            raise ValueError("B1 materialized_physical_bytes 超过冻结预算")

    def integer_stream(self) -> tuple[int, ...]:
        """返回 B1 未发布结果的完整、无 root、schema-backed 整数 identity。"""
        result = [V4_PROVENANCE_DAG_RESULT_SCHEMA]
        pack_key(result, tuple(ord(item) for item in self.logical_stage_name))
        pack_key(result, self.catalog_stable_key)
        for descriptor in (
                self.node_endpoint_stream,
                self.parent_request_stream,
                self.sorted_parent_request_stream,
                self.direct_edge_stream,
                self.sorted_leaf_stream,
                self.leaf_endpoint_stream):
            pack_key(result, descriptor.integer_stream())
        result.extend((
            self.node_count,
            self.parent_request_count,
            self.direct_edge_count,
            self.leaf_count,
            self.leaf_endpoint_count,
            self.materialized_physical_bytes,
        ))
        pack_key(result, self.budget.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回可由后续 B2/B3 绑定而不泄漏 K root 的结果 identity。"""
        return self.integer_stream()


# object-model: resource_owner; representation=protocol; interop=pending
class _BoundedDigestingBinaryHandle:
    """以单一已核验 handle 流式累计有限 physical SHA-256。"""

    __slots__ = ("_byte_count", "_digest", "_max_bytes", "_stream")

    def __init__(self, stream: BinaryIO, *, max_bytes: int) -> None:
        """接管 capability 打开的唯一二进制 handle，不缓存完整 P0 文件。"""
        _require_positive_int(max_bytes, label="B1 physical max_bytes")
        self._byte_count = 0
        self._digest = hashlib.sha256()
        self._max_bytes = max_bytes
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        """在 P0 解码前限制并摘要每次底层成功返回的读取字节。"""
        if type(size) is not int:
            raise TypeError("B1 physical read size 必须是严格整数")
        remaining_with_sentinel = self._max_bytes - self._byte_count + 1
        requested = min(
            size if size >= 0 else remaining_with_sentinel,
            remaining_with_sentinel,
        )
        payload = self._stream.read(requested)
        if isinstance(payload, bytes):
            self._accept(payload)
        return payload

    def write(self, data: bytes | bytearray | memoryview) -> int:
        """只将底层确认写出的前缀纳入 physical identity。"""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("B1 physical write data 必须是 bytes-like")
        if len(data) > self._max_bytes - self._byte_count:
            _budget_fail("B1 P0 stream 物理字节超过预算")
        written = self._stream.write(data)
        if type(written) is int and 0 < written <= len(data):
            self._accept(bytes(memoryview(data)[:written]))
        return written

    def flush(self) -> None:
        """透传 P0 seal 所需 flush，不将其伪造为物理写入。"""
        self._stream.flush()

    def close(self) -> None:
        """关闭唯一底层 handle；P0 reader/writer 负责其正常生命周期。"""
        self._stream.close()

    def physical(self) -> KRunFileDigest:
        """返回同一 handle 已成功处理的有限 physical identity。"""
        return KRunFileDigest(
            self._byte_count,
            tuple(self._digest.digest()),
        )

    def _accept(self, payload: bytes) -> None:
        """在任何数据进入 P0 前强制单 stream physical 预算。"""
        if len(payload) > self._max_bytes - self._byte_count:
            _budget_fail("B1 P0 stream 物理字节超过预算")
        self._digest.update(payload)
        self._byte_count += len(payload)


# object-model: resource_owner; representation=protocol; interop=pending
class _VerifiedP0Reader:
    """按 descriptor 重放一个 P0 stream，并在 finish 时复核 physical identity。"""

    __slots__ = (
        "_descriptor", "_file_identity", "_finished", "_handle", "_reader",
        "_relative", "_root",
    )

    def __init__(
            self,
            root: KRunRoot,
            relative: Path,
            descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor
            | ConversationHeldOutV4ProvenanceStreamIdentity,
            *,
            budget: ConversationHeldOutV4ProvenanceDagBudget,
            label: str,
            ) -> None:
        """以读前/读后文件身份和同一 handle SHA 防止上游 P0 静默漂移。"""
        expected_footer, expected_physical = _expected_p0_identity(descriptor)
        if expected_physical.byte_count > budget.max_stream_physical_bytes:
            _budget_fail(f"{label} physical bytes 超过 B1 stream 预算")
        self._descriptor = descriptor
        self._relative = relative
        self._root = root
        self._finished = False
        self._file_identity = capture_plain_file_identity(
            root,
            relative,
            label=f"{label} capture",
        )
        if self._file_identity.byte_count != expected_physical.byte_count:
            _fail(f"{label} 文件字节数与冻结 descriptor 漂移")
        stream = open_plain_binary(
            root,
            relative,
            label=label,
            expected_identity=self._file_identity,
        )
        self._handle = _BoundedDigestingBinaryHandle(
            stream,
            max_bytes=expected_physical.byte_count,
        )
        try:
            self._reader = IntegerFramedStreamReader.from_open_binary(
                self._handle,
                path=relative,
                max_frame_bytes=budget.max_record_payload_bytes,
                max_record_count=budget.max_stream_record_count,
                max_total_payload_bytes=budget.max_stream_payload_bytes,
            )
        except BaseException:
            try:
                self._handle.close()
            except OSError:
                pass
            raise
        if not isinstance(expected_footer, IntegerFramedStreamFooter):
            raise AssertionError("B1 expected footer 类型漂移")

    def __enter__(self) -> "_VerifiedP0Reader":
        """返回单记录 cursor；调用方必须在正常路径显式 finish。"""
        return self

    def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
            ) -> None:
        """异常路径只关闭不可信残余输入，不把未完整读取视为已验证。"""
        self.close()

    def read_record(self) -> tuple[int, ...] | None:
        """读取一条当前 P0 record；任何 codec 失败都由底层立即关闭 handle。"""
        return self._reader.read_record()

    def finish(self) -> IntegerFramedStreamFooter:
        """完整消费并逐字段比较 footer、physical bytes/SHA 与冻结 descriptor。"""
        if self._finished:
            _fail("B1 verified P0 reader 不得重复 finish")
        footer = self._reader.finish()
        physical = self._handle.physical()
        expected_footer, expected_physical = _expected_p0_identity(self._descriptor)
        require_plain_file_identity(
            self._root,
            self._relative,
            self._file_identity,
            label="B1 P0 post-read",
        )
        if footer != expected_footer:
            _fail("B1 P0 footer 与冻结 descriptor 漂移")
        if physical != expected_physical:
            _fail("B1 P0 physical bytes 或 SHA-256 与冻结 descriptor 漂移")
        self._finished = True
        return footer

    def close(self) -> None:
        """结束 reader 生命周期；已由 footer 自动关闭时此操作保持幂等。"""
        self._reader.close()


# object-model: resource_owner; representation=protocol; interop=pending
class _BoundedP0Writer:
    """只通过 K capability 排他创建一个有计数、payload 与 physical 上限的 P0 stream。"""

    __slots__ = (
        "_budget", "_handle", "_payload_bytes", "_record_count", "_relative",
        "_root", "_sealed", "_writer", "_work_relative_path", "_stream_kind",
    )

    def __init__(
            self,
            root: KRunRoot,
            relative: Path,
            *,
            work_relative_path: Path,
            stream_kind: int,
            budget: ConversationHeldOutV4ProvenanceDagBudget,
            label: str,
            ) -> None:
        """建立一个新文件；任何失败残片不删除、不覆盖且不能返回 descriptor。"""
        self._budget = budget
        self._relative = relative
        self._root = root
        self._work_relative_path = work_relative_path
        self._stream_kind = stream_kind
        self._record_count = 0
        self._payload_bytes = 0
        self._sealed = False
        stream = open_exclusive_binary(root, relative, label=label)
        self._handle = _BoundedDigestingBinaryHandle(
            stream,
            max_bytes=budget.max_stream_physical_bytes,
        )
        try:
            self._writer = IntegerFramedStreamWriter.from_open_binary(
                self._handle,
                path=relative,
            )
        except BaseException:
            try:
                self._handle.close()
            except OSError:
                pass
            raise

    def append(self, record: tuple[int, ...]) -> None:
        """在写入前预留 record/payload 预算，确保 P0 不接收越限条目。"""
        if self._sealed:
            _fail("B1 P0 writer 已 seal")
        try:
            strict_integer_tuple(record, label="B1 output record")
            payload_bytes = encoded_integer_tuple_size(record)
        except (TypeError, ValueError) as exc:
            _fail("B1 output record 不是规范整数 tuple")
            raise AssertionError from exc
        if payload_bytes > self._budget.max_record_payload_bytes:
            _budget_fail("B1 单 record payload 超过预算")
        if self._record_count >= self._budget.max_stream_record_count:
            _budget_fail("B1 P0 stream record 数超过预算")
        if payload_bytes > self._budget.max_stream_payload_bytes - self._payload_bytes:
            _budget_fail("B1 P0 stream payload 超过预算")
        self._writer.append(record)
        self._record_count += 1
        self._payload_bytes += payload_bytes

    def seal(self) -> ConversationHeldOutV4ProvenanceDagStreamDescriptor:
        """唯一封存 stream，并以同一 handle 的实际 SHA 形成未发布 descriptor。"""
        if self._sealed:
            _fail("B1 P0 writer 不得重复 seal")
        footer = self._writer.seal()
        physical = self._handle.physical()
        identity = capture_plain_file_identity(
            self._root,
            self._relative,
            label="B1 P0 output identity",
        )
        if (footer.record_count != self._record_count
                or footer.total_payload_bytes != self._payload_bytes):
            _fail("B1 P0 output footer 与流式计数不一致")
        if physical.byte_count != identity.byte_count:
            _fail("B1 P0 output handle 与路径物理字节漂移")
        self._sealed = True
        descriptor = ConversationHeldOutV4ProvenanceDagStreamDescriptor(
            self._stream_kind,
            self._work_relative_path,
            footer,
            physical,
        )
        # 写侧同一 handle 的摘要不能替代读回的封存/physical 核验。
        with _VerifiedP0Reader(
                self._root,
                self._relative,
                descriptor,
                budget=self._budget,
                label="B1 P0 output verification",
        ) as verifier:
            verifier.finish()
        return descriptor

    def close(self) -> None:
        """异常路径关闭 P0 writer；未 seal 残片保持不可发布。"""
        self._writer.close()


def _expected_p0_identity(
        descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor
        | ConversationHeldOutV4ProvenanceStreamIdentity,
        ) -> tuple[IntegerFramedStreamFooter, KRunFileDigest]:
    """从 catalog 或 B1 descriptor 取得同形完整 P0/physical identity。"""
    if isinstance(descriptor, ConversationHeldOutV4ProvenanceDagStreamDescriptor):
        return descriptor.p0_footer, descriptor.physical
    if isinstance(descriptor, ConversationHeldOutV4ProvenanceStreamIdentity):
        return (
            descriptor.p0_footer,
            KRunFileDigest(
                descriptor.physical_byte_count,
                descriptor.physical_sha256,
            ),
        )
    raise TypeError("B1 descriptor 类型错误")


def _work_relative(stage: str, filename: Path) -> Path:
    """构造固定 stage sibling 内的无绝对路径 descriptor 定位。"""
    _require_stage_name(stage)
    _relative_path_key(filename, label="B1 stage filename")
    return Path(stage) / filename


def _create_stage_root(
        work_run_root: KRunRoot, stage: str,
        ) -> KRunRoot:
    """排他创建一个新的 work sibling stage root，并二次确认它仍为空。"""
    _require_stage_name(stage)
    stage_root = create_new_run_root(
        work_run_root.path / stage,
        require_k_drive=not work_run_root.test_transport,
        label=f"B1 {stage}",
    )
    require_fresh_empty_run_root(stage_root, label=f"B1 {stage}")
    return stage_root


def _reserve_materialized(
        current: int,
        maximum_next: int,
        *,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        label: str,
        ) -> None:
    """在创建下一 stage 前保守预留其最大物理物化量，避免事后越过总预算。"""
    _require_nonnegative_int(current, label="B1 materialized current")
    _require_nonnegative_int(maximum_next, label="B1 materialized maximum_next")
    if maximum_next > budget.max_total_materialized_physical_bytes - current:
        _budget_fail(f"{label} 将超过 B1 累计物化物理字节预算")


def _add_materialized(
        current: int,
        actual: int,
        *,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        label: str,
        ) -> int:
    """登记一个已成功 stage 的实际物化字节，并再次拒绝任何内部预算漂移。"""
    _require_nonnegative_int(current, label="B1 materialized current")
    _require_nonnegative_int(actual, label="B1 materialized actual")
    if actual > budget.max_total_materialized_physical_bytes - current:
        _budget_fail(f"{label} 实际物化物理字节超过 B1 预算")
    return current + actual


def _read_protocol_key(
        reader: IntegerStreamReader, *, label: str,
        ) -> ProtocolKey:
    """从完整 framed key 恢复 ProtocolKey，禁止任意摘要或截断代替节点本体。"""
    try:
        return ProtocolKey(reader.read_key(label=label))
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail(f"{label} ProtocolKey 非法")
        raise AssertionError from exc


def encode_v4_provenance_node_endpoint_record(
        node_key: ProtocolKey,
        snapshot: ConversationHeldOutV4SnapshotIdentity,
        ) -> tuple[int, ...]:
    """编码 node key 与完整 snapshot 的 B1 endpoint record。"""
    if not isinstance(node_key, ProtocolKey):
        raise TypeError("node endpoint node_key 必须是 ProtocolKey")
    if not isinstance(snapshot, ConversationHeldOutV4SnapshotIdentity):
        raise TypeError("node endpoint snapshot 类型错误")
    result = [V4_PROVENANCE_DAG_NODE_ENDPOINT_RECORD_SCHEMA]
    pack_key(result, node_key.components)
    pack_key(result, snapshot.integer_stream())
    return tuple(result)


def decode_v4_provenance_node_endpoint_record(
        record: tuple[int, ...],
        ) -> tuple[ProtocolKey, ConversationHeldOutV4SnapshotIdentity]:
    """严格恢复 node endpoint，不接受丢失 snapshot 字段或未知尾部。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label="node endpoint schema")
        if schema != V4_PROVENANCE_DAG_NODE_ENDPOINT_RECORD_SCHEMA:
            _fail("node endpoint schema 不兼容")
        node_key = _read_protocol_key(reader, label="node endpoint node_key")
        snapshot = ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
            reader.read_key(label="node endpoint snapshot"))
        reader.finish()
        return node_key, snapshot
    except ConversationHeldOutV4ProvenanceScalableDagError:
        raise
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail("node endpoint record 非法")
        raise AssertionError from exc


def encode_v4_provenance_parent_request_record(
        parent_node_key: ProtocolKey,
        child_node_key: ProtocolKey,
        ) -> tuple[int, ...]:
    """编码完整 parent/child key 对，供外排后按 parent endpoint join。"""
    if not isinstance(parent_node_key, ProtocolKey):
        raise TypeError("parent request parent_node_key 必须是 ProtocolKey")
    if not isinstance(child_node_key, ProtocolKey):
        raise TypeError("parent request child_node_key 必须是 ProtocolKey")
    result = [V4_PROVENANCE_DAG_PARENT_REQUEST_RECORD_SCHEMA]
    pack_key(result, parent_node_key.components)
    pack_key(result, child_node_key.components)
    return tuple(result)


def decode_v4_provenance_parent_request_record(
        record: tuple[int, ...],
        ) -> tuple[ProtocolKey, ProtocolKey]:
    """恢复严格完整 parent/child request，不允许 hash 或 URI 替代节点键。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label="parent request schema")
        if schema != V4_PROVENANCE_DAG_PARENT_REQUEST_RECORD_SCHEMA:
            _fail("parent request schema 不兼容")
        parent = _read_protocol_key(reader, label="parent request parent")
        child = _read_protocol_key(reader, label="parent request child")
        reader.finish()
        return parent, child
    except ConversationHeldOutV4ProvenanceScalableDagError:
        raise
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail("parent request record 非法")
        raise AssertionError from exc


def encode_v4_provenance_direct_edge_record(
        child_node_key: ProtocolKey,
        parent_node_key: ProtocolKey,
        parent_snapshot: ConversationHeldOutV4SnapshotIdentity,
        ) -> tuple[int, ...]:
    """编码 direct edge，并保留完整 parent node endpoint snapshot 本体。"""
    if not isinstance(child_node_key, ProtocolKey):
        raise TypeError("direct edge child_node_key 必须是 ProtocolKey")
    if not isinstance(parent_node_key, ProtocolKey):
        raise TypeError("direct edge parent_node_key 必须是 ProtocolKey")
    if not isinstance(parent_snapshot, ConversationHeldOutV4SnapshotIdentity):
        raise TypeError("direct edge parent_snapshot 类型错误")
    result = [V4_PROVENANCE_DAG_DIRECT_EDGE_RECORD_SCHEMA]
    pack_key(result, child_node_key.components)
    pack_key(result, parent_node_key.components)
    pack_key(result, parent_snapshot.integer_stream())
    return tuple(result)


def decode_v4_provenance_direct_edge_record(
        record: tuple[int, ...],
        ) -> tuple[ProtocolKey, ProtocolKey, ConversationHeldOutV4SnapshotIdentity]:
    """恢复 direct edge 的 child、parent 和完整 parent snapshot。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label="direct edge schema")
        if schema != V4_PROVENANCE_DAG_DIRECT_EDGE_RECORD_SCHEMA:
            _fail("direct edge schema 不兼容")
        child = _read_protocol_key(reader, label="direct edge child")
        parent = _read_protocol_key(reader, label="direct edge parent")
        snapshot = ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
            reader.read_key(label="direct edge parent snapshot"))
        reader.finish()
        return child, parent, snapshot
    except ConversationHeldOutV4ProvenanceScalableDagError:
        raise
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail("direct edge record 非法")
        raise AssertionError from exc


def encode_v4_provenance_leaf_endpoint_record(
        leaf: ConversationHeldOutV4ProvenanceLeaf,
        snapshot: ConversationHeldOutV4SnapshotIdentity,
        ) -> tuple[int, ...]:
    """编码完整 leaf 与其已核对 SourceRef 的完整 endpoint snapshot。"""
    if not isinstance(leaf, ConversationHeldOutV4ProvenanceLeaf):
        raise TypeError("leaf endpoint leaf 类型错误")
    if not isinstance(snapshot, ConversationHeldOutV4SnapshotIdentity):
        raise TypeError("leaf endpoint snapshot 类型错误")
    result = [V4_PROVENANCE_DAG_LEAF_ENDPOINT_RECORD_SCHEMA]
    pack_key(result, leaf.integer_stream())
    pack_key(result, snapshot.integer_stream())
    return tuple(result)


def decode_v4_provenance_leaf_endpoint_record(
        record: tuple[int, ...],
        ) -> tuple[ConversationHeldOutV4ProvenanceLeaf,
                   ConversationHeldOutV4SnapshotIdentity]:
    """恢复 leaf endpoint，并再次要求 leaf SourceRef 等于 snapshot SourceRef。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label="leaf endpoint schema")
        if schema != V4_PROVENANCE_DAG_LEAF_ENDPOINT_RECORD_SCHEMA:
            _fail("leaf endpoint schema 不兼容")
        leaf = ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(
            reader.read_key(label="leaf endpoint leaf"))
        snapshot = ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
            reader.read_key(label="leaf endpoint snapshot"))
        reader.finish()
    except ConversationHeldOutV4ProvenanceScalableDagError:
        raise
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail("leaf endpoint record 非法")
        raise AssertionError from exc
    if leaf.source_ref.stable_key() != snapshot.source_ref.stable_key():
        _fail("leaf endpoint SourceRef 与 snapshot SourceRef 不一致")
    return leaf, snapshot


def _decode_lineage_node(record: tuple[int, ...], *, label: str) -> ConversationHeldOutV4LineageNode:
    """以 P1 typed decoder 重新验证 catalog node record，不接受只按表面字段取键。"""
    try:
        return ConversationHeldOutV4LineageNode.from_integer_stream(record)
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail(f"{label} P1 lineage node record 非法")
        raise AssertionError from exc


def _bounded_parent_count_from_node_record(
        record: tuple[int, ...], *, label: str,
        ) -> int:
    """在 P1 构造完整 parent tuple 前读取其声明长度，先施加 B1 显式 group 上限。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label=f"{label} schema")
        if schema != V4_PROVENANCE_LINEAGE_NODE_SCHEMA:
            _fail(f"{label} P1 lineage node schema 不兼容")
        reader.read_key(label=f"{label} node key")
        reader.read_key(label=f"{label} snapshot")
        return reader.read_nonnegative(label=f"{label} parent count")
    except ConversationHeldOutV4ProvenanceScalableDagError:
        raise
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail(f"{label} P1 lineage node header 非法")
        raise AssertionError from exc


def _decode_leaf(record: tuple[int, ...], *, label: str) -> ConversationHeldOutV4ProvenanceLeaf:
    """以 P1 typed decoder 重新验证 catalog leaf record 的完整来源与节点键。"""
    try:
        return ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(record)
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail(f"{label} P1 provenance leaf record 非法")
        raise AssertionError from exc


def _replay_nodes(
        catalog: ConversationHeldOutV4ProvenanceStreamCatalog,
        stage_root: KRunRoot,
        *,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        ) -> tuple[
            ConversationHeldOutV4ProvenanceDagStreamDescriptor,
            ConversationHeldOutV4ProvenanceDagStreamDescriptor,
            int,
            int,
        ]:
    """流式重放 nodes，物化 endpoint 与尚未排序的 parent request 两个 P0 流。"""
    endpoint_writer = _BoundedP0Writer(
        stage_root,
        _NODE_ENDPOINT_FILE,
        work_relative_path=_work_relative(_STAGE_NODE_REPLAY, _NODE_ENDPOINT_FILE),
        stream_kind=V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT,
        budget=budget,
        label="B1 node endpoint output",
    )
    request_writer = _BoundedP0Writer(
        stage_root,
        _PARENT_REQUEST_FILE,
        work_relative_path=_work_relative(_STAGE_NODE_REPLAY, _PARENT_REQUEST_FILE),
        stream_kind=V4_PROVENANCE_DAG_STREAM_PARENT_REQUEST,
        budget=budget,
        label="B1 parent request output",
    )
    node_count = 0
    request_count = 0
    previous_node_key: tuple[int, ...] | None = None
    try:
        for stream_index, stream_identity in enumerate(catalog.node_streams):
            with _VerifiedP0Reader(
                    catalog.root,
                    stream_identity.relative_path,
                    stream_identity,
                    budget=budget,
                    label=f"B1 catalog node[{stream_index}]",
            ) as reader:
                while True:
                    raw = reader.read_record()
                    if raw is None:
                        break
                    parent_count = _bounded_parent_count_from_node_record(
                        raw,
                        label=f"B1 catalog node[{stream_index}]",
                    )
                    if parent_count > budget.max_parents_per_node:
                        _budget_fail("B1 单 node parent 数超过预算")
                    node = _decode_lineage_node(
                        raw,
                        label=f"B1 catalog node[{stream_index}]",
                    )
                    if node_count >= budget.max_node_count:
                        _budget_fail("B1 node 数超过预算")
                    if len(node.parent_node_keys) != parent_count:
                        _fail("B1 P1 node parent count 重放漂移")
                    if (previous_node_key is not None
                            and node.node_key.components <= previous_node_key):
                        _fail("B1 catalog node key 未严格递增")
                    previous_node_key = node.node_key.components
                    endpoint_writer.append(
                        encode_v4_provenance_node_endpoint_record(
                            node.node_key,
                            node.snapshot,
                        ))
                    node_count += 1
                    for parent_node_key in node.parent_node_keys:
                        if request_count >= budget.max_parent_request_count:
                            _budget_fail("B1 parent request 数超过预算")
                        request_writer.append(
                            encode_v4_provenance_parent_request_record(
                                parent_node_key,
                                node.node_key,
                            ))
                        request_count += 1
                reader.finish()
        endpoint_descriptor = endpoint_writer.seal()
        request_descriptor = request_writer.seal()
    finally:
        endpoint_writer.close()
        request_writer.close()
    if endpoint_descriptor.p0_footer.record_count != node_count:
        _fail("B1 node endpoint footer 与 node 计数不一致")
    if request_descriptor.p0_footer.record_count != request_count:
        _fail("B1 parent request footer 与 request 计数不一致")
    return endpoint_descriptor, request_descriptor, node_count, request_count


def _verify_external_sort_inputs(
        result: IntegerExternalSortResult,
        expected: tuple[
            tuple[Path, IntegerFramedStreamFooter, KRunFileDigest], ...],
        *,
        label: str,
        ) -> None:
    """要求外排在同一次消费中读取的全部 P0 identity 精确等于冻结描述符。"""
    if not isinstance(result, IntegerExternalSortResult):
        raise TypeError("B1 external sort result 类型错误")
    actual = result.identity.input_identities
    if len(actual) != len(expected):
        _fail(f"{label} external sort 输入数与冻结 descriptor 不一致")
    for index, (item, expected_item) in enumerate(zip(actual, expected, strict=True)):
        relative, footer, physical = expected_item
        if not isinstance(item, IntegerExternalSortInputIdentity):
            _fail(f"{label} external sort input[{index}] 类型错误")
        if (item.relative_path != relative
                or item.footer != footer
                or item.physical != physical):
            _fail(f"{label} external sort input[{index}] identity 漂移")


def _descriptor_from_external_sort(
        result: IntegerExternalSortResult,
        *,
        stream_kind: int,
        stage: str,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        ) -> ConversationHeldOutV4ProvenanceDagStreamDescriptor:
    """把通用外排已验证 output 收敛为 B1 无绝对路径 descriptor。"""
    physical = result.identity.output_physical
    if physical.byte_count > budget.max_stream_physical_bytes:
        _budget_fail("B1 external sort output 物理字节超过 stream 预算")
    if result.identity.output_footer.record_count > budget.max_stream_record_count:
        _budget_fail("B1 external sort output record 数超过 stream 预算")
    if result.identity.output_footer.total_payload_bytes > budget.max_stream_payload_bytes:
        _budget_fail("B1 external sort output payload 超过 stream 预算")
    return ConversationHeldOutV4ProvenanceDagStreamDescriptor(
        stream_kind,
        _work_relative(stage, result.identity.output_relative_path),
        result.identity.output_footer,
        physical,
    )


def _sort_parent_requests(
        source_root: KRunRoot,
        source_descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        work_root: KRunRoot,
        *,
        logical_stage_name: str,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        ) -> tuple[ConversationHeldOutV4ProvenanceDagStreamDescriptor, int]:
    """以 parent key 有界外排 request；同一 parent 的稳定次序由上游流顺序固定。"""
    result = external_sort_sealed_integer_records(
        source_root,
        (_PARENT_REQUEST_FILE,),
        work_root,
        output_relative_path=_SORTED_PARENT_REQUEST_FILE,
        logical_stage_name=f"{logical_stage_name}.parent-request.v1",
        sort_key=lambda record: decode_v4_provenance_parent_request_record(
            record)[0].components,
        budget=budget.external_sort_budget,
    )
    _verify_external_sort_inputs(
        result,
        ((_PARENT_REQUEST_FILE, source_descriptor.p0_footer,
          source_descriptor.physical),),
        label="B1 parent request sort",
    )
    descriptor = _descriptor_from_external_sort(
        result,
        stream_kind=V4_PROVENANCE_DAG_STREAM_SORTED_PARENT_REQUEST,
        stage=_STAGE_PARENT_SORT,
        budget=budget,
    )
    return descriptor, result.temporary_physical_bytes


def _join_direct_edges(
        endpoint_root: KRunRoot,
        endpoint_descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        request_root: KRunRoot,
        request_descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        output_root: KRunRoot,
        *,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        ) -> tuple[ConversationHeldOutV4ProvenanceDagStreamDescriptor, int]:
    """以两个 cursor merge join 每个已排序 parent request 到完整 node endpoint。"""
    writer = _BoundedP0Writer(
        output_root,
        _DIRECT_EDGE_FILE,
        work_relative_path=_work_relative(_STAGE_DIRECT_EDGE, _DIRECT_EDGE_FILE),
        stream_kind=V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE,
        budget=budget,
        label="B1 direct edge output",
    )
    edge_count = 0
    previous_request_key: tuple[int, ...] | None = None
    try:
        with _VerifiedP0Reader(
                endpoint_root,
                _NODE_ENDPOINT_FILE,
                endpoint_descriptor,
                budget=budget,
                label="B1 node endpoint input",
        ) as endpoint_reader, _VerifiedP0Reader(
                request_root,
                _SORTED_PARENT_REQUEST_FILE,
                request_descriptor,
                budget=budget,
                label="B1 sorted parent request input",
        ) as request_reader:
            endpoint: tuple[ProtocolKey, ConversationHeldOutV4SnapshotIdentity] | None = None
            previous_endpoint_key: tuple[int, ...] | None = None

            def advance_endpoint() -> tuple[
                    ProtocolKey, ConversationHeldOutV4SnapshotIdentity] | None:
                """读取一个 node endpoint；每次只保留当前 key 与完整 snapshot。"""
                nonlocal previous_endpoint_key
                raw = endpoint_reader.read_record()
                if raw is None:
                    return None
                decoded = decode_v4_provenance_node_endpoint_record(raw)
                if (previous_endpoint_key is not None
                        and decoded[0].components <= previous_endpoint_key):
                    _fail("B1 node endpoint key 未严格递增")
                previous_endpoint_key = decoded[0].components
                return decoded

            endpoint = advance_endpoint()
            while True:
                raw_request = request_reader.read_record()
                if raw_request is None:
                    break
                parent_key, child_key = decode_v4_provenance_parent_request_record(
                    raw_request)
                if (previous_request_key is not None
                        and parent_key.components < previous_request_key):
                    _fail("B1 sorted parent request key 倒序")
                previous_request_key = parent_key.components
                while (endpoint is not None
                       and endpoint[0].components < parent_key.components):
                    endpoint = advance_endpoint()
                if endpoint is None or endpoint[0].components != parent_key.components:
                    _fail("B1 parent request 指向未知 node endpoint")
                if edge_count >= budget.max_direct_edge_count:
                    _budget_fail("B1 direct edge 数超过预算")
                writer.append(encode_v4_provenance_direct_edge_record(
                    child_key,
                    parent_key,
                    endpoint[1],
                ))
                edge_count += 1
            while endpoint is not None:
                endpoint = advance_endpoint()
            request_reader.finish()
            endpoint_reader.finish()
        descriptor = writer.seal()
    finally:
        writer.close()
    return descriptor, edge_count


def _write_empty_sorted_leaf(
        root: KRunRoot,
        *,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        ) -> ConversationHeldOutV4ProvenanceDagStreamDescriptor:
    """在 catalog 没有 leaf shard 时仍产出一个封存空 P0 typed 输出。"""
    writer = _BoundedP0Writer(
        root,
        _SORTED_LEAF_FILE,
        work_relative_path=_work_relative(_STAGE_LEAF_SORT, _SORTED_LEAF_FILE),
        stream_kind=V4_PROVENANCE_DAG_STREAM_SORTED_LEAF,
        budget=budget,
        label="B1 empty sorted leaf output",
    )
    try:
        return writer.seal()
    finally:
        writer.close()


def _sort_leaves(
        catalog: ConversationHeldOutV4ProvenanceStreamCatalog,
        work_root: KRunRoot,
        *,
        logical_stage_name: str,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        ) -> tuple[ConversationHeldOutV4ProvenanceDagStreamDescriptor, int]:
    """以 lineage node key 有界外排 leaves，供后续与 node endpoint 流式 join。"""
    if not catalog.leaf_streams:
        return _write_empty_sorted_leaf(work_root, budget=budget), 0
    input_paths = tuple(item.relative_path for item in catalog.leaf_streams)
    result = external_sort_sealed_integer_records(
        catalog.root,
        input_paths,
        work_root,
        output_relative_path=_SORTED_LEAF_FILE,
        logical_stage_name=f"{logical_stage_name}.leaf-node.v1",
        sort_key=lambda record: _decode_leaf(
            record,
            label="B1 leaf external sort",
        ).lineage_node_key.components,
        budget=budget.external_sort_budget,
    )
    expected = tuple(
        (item.relative_path, item.p0_footer, KRunFileDigest(
            item.physical_byte_count,
            item.physical_sha256,
        ))
        for item in catalog.leaf_streams
    )
    _verify_external_sort_inputs(result, expected, label="B1 leaf sort")
    descriptor = _descriptor_from_external_sort(
        result,
        stream_kind=V4_PROVENANCE_DAG_STREAM_SORTED_LEAF,
        stage=_STAGE_LEAF_SORT,
        budget=budget,
    )
    return descriptor, result.temporary_physical_bytes


def _join_leaf_endpoints(
        endpoint_root: KRunRoot,
        endpoint_descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        leaf_root: KRunRoot,
        leaf_descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        output_root: KRunRoot,
        *,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        ) -> tuple[ConversationHeldOutV4ProvenanceDagStreamDescriptor, int]:
    """按 lineage node merge join leaves，并精确比较完整十一整数 SourceRef。"""
    writer = _BoundedP0Writer(
        output_root,
        _LEAF_ENDPOINT_FILE,
        work_relative_path=_work_relative(_STAGE_LEAF_ENDPOINT, _LEAF_ENDPOINT_FILE),
        stream_kind=V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT,
        budget=budget,
        label="B1 leaf endpoint output",
    )
    leaf_count = 0
    current_leaf_node_key: tuple[int, ...] | None = None
    current_leaf_group_count = 0
    previous_leaf_node_key: tuple[int, ...] | None = None
    try:
        with _VerifiedP0Reader(
                endpoint_root,
                _NODE_ENDPOINT_FILE,
                endpoint_descriptor,
                budget=budget,
                label="B1 node endpoint leaf join input",
        ) as endpoint_reader, _VerifiedP0Reader(
                leaf_root,
                _SORTED_LEAF_FILE,
                leaf_descriptor,
                budget=budget,
                label="B1 sorted leaf input",
        ) as leaf_reader:
            endpoint: tuple[ProtocolKey, ConversationHeldOutV4SnapshotIdentity] | None = None
            previous_endpoint_key: tuple[int, ...] | None = None

            def advance_endpoint() -> tuple[
                    ProtocolKey, ConversationHeldOutV4SnapshotIdentity] | None:
                """仅保留 leaf join 当前 node endpoint，避免 materialize 全部 node 图。"""
                nonlocal previous_endpoint_key
                raw = endpoint_reader.read_record()
                if raw is None:
                    return None
                decoded = decode_v4_provenance_node_endpoint_record(raw)
                if (previous_endpoint_key is not None
                        and decoded[0].components <= previous_endpoint_key):
                    _fail("B1 node endpoint key 未严格递增")
                previous_endpoint_key = decoded[0].components
                return decoded

            endpoint = advance_endpoint()
            while True:
                raw_leaf = leaf_reader.read_record()
                if raw_leaf is None:
                    break
                leaf = _decode_leaf(raw_leaf, label="B1 sorted leaf")
                node_key = leaf.lineage_node_key.components
                if (previous_leaf_node_key is not None
                        and node_key < previous_leaf_node_key):
                    _fail("B1 sorted leaf lineage node key 倒序")
                previous_leaf_node_key = node_key
                if current_leaf_node_key != node_key:
                    current_leaf_node_key = node_key
                    current_leaf_group_count = 0
                if current_leaf_group_count >= budget.max_leaves_per_node:
                    _budget_fail("B1 同一 lineage node leaf group 超过预算")
                current_leaf_group_count += 1
                if leaf_count >= budget.max_leaf_count:
                    _budget_fail("B1 leaf 数超过预算")
                while (endpoint is not None
                       and endpoint[0].components < node_key):
                    endpoint = advance_endpoint()
                if endpoint is None or endpoint[0].components != node_key:
                    _fail("B1 leaf 指向未知 node endpoint")
                if leaf.source_ref.stable_key() != endpoint[1].source_ref.stable_key():
                    _fail("B1 leaf SourceRef 与 node snapshot SourceRef 不一致")
                writer.append(encode_v4_provenance_leaf_endpoint_record(
                    leaf,
                    endpoint[1],
                ))
                leaf_count += 1
            while endpoint is not None:
                endpoint = advance_endpoint()
            leaf_reader.finish()
            endpoint_reader.finish()
        descriptor = writer.seal()
    finally:
        writer.close()
    return descriptor, leaf_count


def build_v4_provenance_direct_dag(
        catalog: ConversationHeldOutV4ProvenanceStreamCatalog,
        work_run_root: KRunRoot,
        *,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        logical_stage_name: str,
        ) -> ConversationHeldOutV4ProvenanceDirectDagResult:
    """执行 P2-B1：重放直接 node/parent/leaf endpoint，不发布任何 artifact。

    调用方必须提供 P2-A 已冻结 catalog 和一个不同、非嵌套且空的 K work root。本入口先
    完整 revalidate catalog；随后每个外排和 join 又重比相关 P0 descriptor。所有输出均为
    新建 stage sibling 的 sealed stream，失败不会覆盖或删除任何已有对象。
    """
    if not isinstance(catalog, ConversationHeldOutV4ProvenanceStreamCatalog):
        raise TypeError("B1 catalog 必须是 ConversationHeldOutV4ProvenanceStreamCatalog")
    if not isinstance(work_run_root, KRunRoot):
        raise TypeError("B1 work_run_root 必须是 KRunRoot")
    if not isinstance(budget, ConversationHeldOutV4ProvenanceDagBudget):
        raise TypeError("B1 budget 必须是 ConversationHeldOutV4ProvenanceDagBudget")
    _require_stage_name(logical_stage_name)

    frozen_catalog = revalidate_v4_provenance_stream_catalog(catalog)
    require_disjoint_run_roots(
        frozen_catalog.root,
        work_run_root,
        label="B1 catalog/work roots",
    )
    require_fresh_empty_run_root(work_run_root, label="B1 work_run_root")

    materialized_physical_bytes = 0

    _reserve_materialized(
        materialized_physical_bytes,
        budget.max_stream_physical_bytes * 2,
        budget=budget,
        label="B1 node replay",
    )
    node_stage = _create_stage_root(work_run_root, _STAGE_NODE_REPLAY)
    node_endpoint, parent_request, node_count, parent_request_count = _replay_nodes(
        frozen_catalog,
        node_stage,
        budget=budget,
    )
    materialized_physical_bytes = _add_materialized(
        materialized_physical_bytes,
        node_endpoint.physical.byte_count + parent_request.physical.byte_count,
        budget=budget,
        label="B1 node replay",
    )

    parent_sort_reservation = (
        budget.external_sort_budget.max_temporary_physical_bytes
        + budget.external_sort_budget.max_output_physical_bytes
    )
    _reserve_materialized(
        materialized_physical_bytes,
        parent_sort_reservation,
        budget=budget,
        label="B1 parent request sort",
    )
    parent_sort_stage = _create_stage_root(work_run_root, _STAGE_PARENT_SORT)
    try:
        sorted_parent_request, parent_sort_temporary_bytes = _sort_parent_requests(
            node_stage,
            parent_request,
            parent_sort_stage,
            logical_stage_name=logical_stage_name,
            budget=budget,
        )
    except (IntegerExternalSortBudgetExceeded,
            IntegerFramedStreamBudgetExceeded) as exc:
        _budget_fail("B1 parent request external sort 超过冻结预算")
        raise AssertionError from exc
    except IntegerExternalSortError as exc:
        _fail("B1 parent request external sort 失败")
        raise AssertionError from exc
    if sorted_parent_request.p0_footer.record_count != parent_request_count:
        _fail("B1 sorted parent request 数与 replay 不一致")
    materialized_physical_bytes = _add_materialized(
        materialized_physical_bytes,
        parent_sort_temporary_bytes + sorted_parent_request.physical.byte_count,
        budget=budget,
        label="B1 parent request sort",
    )

    _reserve_materialized(
        materialized_physical_bytes,
        budget.max_stream_physical_bytes,
        budget=budget,
        label="B1 direct edge join",
    )
    direct_edge_stage = _create_stage_root(work_run_root, _STAGE_DIRECT_EDGE)
    direct_edge, direct_edge_count = _join_direct_edges(
        node_stage,
        node_endpoint,
        parent_sort_stage,
        sorted_parent_request,
        direct_edge_stage,
        budget=budget,
    )
    materialized_physical_bytes = _add_materialized(
        materialized_physical_bytes,
        direct_edge.physical.byte_count,
        budget=budget,
        label="B1 direct edge join",
    )

    leaf_sort_reservation = (
        budget.max_stream_physical_bytes
        if not frozen_catalog.leaf_streams
        else budget.external_sort_budget.max_temporary_physical_bytes
        + budget.external_sort_budget.max_output_physical_bytes
    )
    _reserve_materialized(
        materialized_physical_bytes,
        leaf_sort_reservation,
        budget=budget,
        label="B1 leaf sort",
    )
    leaf_sort_stage = _create_stage_root(work_run_root, _STAGE_LEAF_SORT)
    try:
        sorted_leaf, leaf_sort_temporary_bytes = _sort_leaves(
            frozen_catalog,
            leaf_sort_stage,
            logical_stage_name=logical_stage_name,
            budget=budget,
        )
    except (IntegerExternalSortBudgetExceeded,
            IntegerFramedStreamBudgetExceeded) as exc:
        _budget_fail("B1 leaf external sort 超过冻结预算")
        raise AssertionError from exc
    except IntegerExternalSortError as exc:
        _fail("B1 leaf external sort 失败")
        raise AssertionError from exc
    leaf_count = sorted_leaf.p0_footer.record_count
    if leaf_count > budget.max_leaf_count:
        _budget_fail("B1 sorted leaf 数超过预算")
    materialized_physical_bytes = _add_materialized(
        materialized_physical_bytes,
        leaf_sort_temporary_bytes + sorted_leaf.physical.byte_count,
        budget=budget,
        label="B1 leaf sort",
    )

    _reserve_materialized(
        materialized_physical_bytes,
        budget.max_stream_physical_bytes,
        budget=budget,
        label="B1 leaf endpoint join",
    )
    leaf_endpoint_stage = _create_stage_root(work_run_root, _STAGE_LEAF_ENDPOINT)
    leaf_endpoint, leaf_endpoint_count = _join_leaf_endpoints(
        node_stage,
        node_endpoint,
        leaf_sort_stage,
        sorted_leaf,
        leaf_endpoint_stage,
        budget=budget,
    )
    materialized_physical_bytes = _add_materialized(
        materialized_physical_bytes,
        leaf_endpoint.physical.byte_count,
        budget=budget,
        label="B1 leaf endpoint join",
    )

    return ConversationHeldOutV4ProvenanceDirectDagResult(
        logical_stage_name,
        frozen_catalog.stable_key(),
        node_endpoint,
        parent_request,
        sorted_parent_request,
        direct_edge,
        sorted_leaf,
        leaf_endpoint,
        node_count,
        parent_request_count,
        direct_edge_count,
        leaf_count,
        leaf_endpoint_count,
        materialized_physical_bytes,
        budget,
    )


def _canonical_b1_record(
        record: tuple[int, ...],
        *, stream_kind: int,
        ) -> tuple[
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[int, ...] | None,
        ]:
    """解码一条 B1 输出，并返回规范重编码、固定排序键和可选受限 group 键。"""
    if stream_kind == V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT:
        node_key, snapshot = decode_v4_provenance_node_endpoint_record(record)
        return (
            encode_v4_provenance_node_endpoint_record(node_key, snapshot),
            (node_key.components,),
            None,
        )
    if stream_kind == V4_PROVENANCE_DAG_STREAM_PARENT_REQUEST:
        parent_key, child_key = decode_v4_provenance_parent_request_record(record)
        return (
            encode_v4_provenance_parent_request_record(parent_key, child_key),
            (child_key.components, parent_key.components),
            child_key.components,
        )
    if stream_kind == V4_PROVENANCE_DAG_STREAM_SORTED_PARENT_REQUEST:
        parent_key, child_key = decode_v4_provenance_parent_request_record(record)
        return (
            encode_v4_provenance_parent_request_record(parent_key, child_key),
            (parent_key.components, child_key.components),
            parent_key.components,
        )
    if stream_kind == V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE:
        child_key, parent_key, snapshot = decode_v4_provenance_direct_edge_record(
            record)
        return (
            encode_v4_provenance_direct_edge_record(
                child_key, parent_key, snapshot),
            (parent_key.components, child_key.components),
            None,
        )
    if stream_kind == V4_PROVENANCE_DAG_STREAM_SORTED_LEAF:
        leaf = _decode_leaf(record, label="B1 revalidation sorted leaf")
        return (
            leaf.integer_stream(),
            (leaf.lineage_node_key.components, leaf.stable_key()),
            leaf.lineage_node_key.components,
        )
    if stream_kind == V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT:
        leaf, snapshot = decode_v4_provenance_leaf_endpoint_record(record)
        return (
            encode_v4_provenance_leaf_endpoint_record(leaf, snapshot),
            (leaf.lineage_node_key.components, leaf.stable_key()),
            leaf.lineage_node_key.components,
        )
    raise AssertionError("B1 revalidation stream_kind 未注册")


def _revalidate_b1_descriptor(
        work_root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        *,
        expected_kind: int,
        expected_path: Path,
        expected_count: int,
        group_limit: int | None,
        budget: ConversationHeldOutV4ProvenanceDagBudget,
        label: str,
        ) -> int:
    """以单一只读 P0 handle 完整回放一条 B1 descriptor。"""
    if descriptor.stream_kind != expected_kind:
        _fail(f"{label} stream_kind 与 B1 result 槽位不一致")
    if descriptor.work_relative_path != expected_path:
        _fail(f"{label} work_relative_path 与 B1 固定 stage 不一致")
    if group_limit is not None:
        _require_positive_int(group_limit, label=f"{label} group_limit")
    count = 0
    previous_sort_key: tuple[tuple[int, ...], ...] | None = None
    current_group_key: tuple[int, ...] | None = None
    current_group_count = 0
    with _VerifiedP0Reader(
            work_root,
            descriptor.work_relative_path,
            descriptor,
            budget=budget,
            label=f"{label} revalidation",
    ) as reader:
        while True:
            record = reader.read_record()
            if record is None:
                break
            canonical, sort_key, group_key = _canonical_b1_record(
                record, stream_kind=expected_kind)
            if canonical != record:
                _fail(f"{label} record 不是规范完整编码")
            if previous_sort_key is not None and sort_key <= previous_sort_key:
                _fail(f"{label} record sort key 未严格递增")
            previous_sort_key = sort_key
            if group_limit is not None:
                if group_key is None:
                    raise AssertionError("B1 revalidation group key 缺失")
                if group_key != current_group_key:
                    current_group_key = group_key
                    current_group_count = 0
                if current_group_count >= group_limit:
                    _budget_fail(f"{label} group 超过 B1 预算")
                current_group_count += 1
            if count >= budget.max_stream_record_count:
                _budget_fail(f"{label} record 数超过 B1 stream 预算")
            count += 1
        footer = reader.finish()
    if footer.record_count != count or count != expected_count:
        _fail(f"{label} record count 与 B1 result 或 footer 不一致")
    return count


def revalidate_v4_provenance_direct_dag_result(
        result: ConversationHeldOutV4ProvenanceDirectDagResult,
        work_root: KRunRoot,
        *,
        catalog: ConversationHeldOutV4ProvenanceStreamCatalog,
        ) -> ConversationHeldOutV4ProvenanceDirectDagResult:
    """只读重验 B1 catalog 到 six sealed descriptor 的固定传递链。

    ``work_root`` 可为初始 create-new capability 或稍后以
    ``open_existing_run_root`` 取得的只读 capability。本入口不创建目录、不外排，也不
    重跑 DAG；它只重验 catalog、catalog-to-result stable key、B1 固定路径的六条 P0
    descriptor，以及每条输出记录的 canonical 编码、排序、计数和 group 预算。
    """
    if not isinstance(result, ConversationHeldOutV4ProvenanceDirectDagResult):
        raise TypeError("B1 revalidation result 类型错误")
    if not isinstance(work_root, KRunRoot):
        raise TypeError("B1 revalidation work_root 必须是 KRunRoot")
    if not isinstance(catalog, ConversationHeldOutV4ProvenanceStreamCatalog):
        raise TypeError("B1 revalidation catalog 类型错误")
    frozen_catalog = revalidate_v4_provenance_stream_catalog(catalog)
    require_disjoint_run_roots(
        frozen_catalog.root,
        work_root,
        label="B1 revalidation catalog/work roots",
    )
    if result.catalog_stable_key != frozen_catalog.stable_key():
        _fail("B1 revalidation catalog stable key 与 result 不一致")
    stable_key = result.stable_key()
    try:
        strict_integer_tuple(stable_key, label="B1 revalidation result stable_key")
    except (TypeError, ValueError) as exc:
        _fail("B1 revalidation result stable_key 非法")
        raise AssertionError from exc

    descriptor_specs = (
        (
            result.node_endpoint_stream,
            V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT,
            _work_relative(_STAGE_NODE_REPLAY, _NODE_ENDPOINT_FILE),
            result.node_count,
            None,
            "B1 node endpoint",
        ),
        (
            result.parent_request_stream,
            V4_PROVENANCE_DAG_STREAM_PARENT_REQUEST,
            _work_relative(_STAGE_NODE_REPLAY, _PARENT_REQUEST_FILE),
            result.parent_request_count,
            result.budget.max_parents_per_node,
            "B1 parent request",
        ),
        (
            result.sorted_parent_request_stream,
            V4_PROVENANCE_DAG_STREAM_SORTED_PARENT_REQUEST,
            _work_relative(_STAGE_PARENT_SORT, _SORTED_PARENT_REQUEST_FILE),
            result.parent_request_count,
            None,
            "B1 sorted parent request",
        ),
        (
            result.direct_edge_stream,
            V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE,
            _work_relative(_STAGE_DIRECT_EDGE, _DIRECT_EDGE_FILE),
            result.direct_edge_count,
            None,
            "B1 direct edge",
        ),
        (
            result.sorted_leaf_stream,
            V4_PROVENANCE_DAG_STREAM_SORTED_LEAF,
            _work_relative(_STAGE_LEAF_SORT, _SORTED_LEAF_FILE),
            result.leaf_count,
            result.budget.max_leaves_per_node,
            "B1 sorted leaf",
        ),
        (
            result.leaf_endpoint_stream,
            V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT,
            _work_relative(_STAGE_LEAF_ENDPOINT, _LEAF_ENDPOINT_FILE),
            result.leaf_endpoint_count,
            result.budget.max_leaves_per_node,
            "B1 leaf endpoint",
        ),
    )
    total_output_physical_bytes = 0
    for descriptor, kind, path, count, group_limit, label in descriptor_specs:
        _revalidate_b1_descriptor(
            work_root,
            descriptor,
            expected_kind=kind,
            expected_path=path,
            expected_count=count,
            group_limit=group_limit,
            budget=result.budget,
            label=label,
        )
        total_output_physical_bytes = _add_materialized(
            total_output_physical_bytes,
            descriptor.physical.byte_count,
            budget=result.budget,
            label="B1 revalidation sealed output total",
        )
    if total_output_physical_bytes > result.materialized_physical_bytes:
        _fail("B1 revalidation result materialized bytes 小于 sealed outputs")
    return result


__all__ = [
    "ConversationHeldOutV4ProvenanceDagBudget",
    "ConversationHeldOutV4ProvenanceDirectDagResult",
    "ConversationHeldOutV4ProvenanceScalableDagBudgetExceeded",
    "ConversationHeldOutV4ProvenanceScalableDagError",
    "ConversationHeldOutV4ProvenanceDagStreamDescriptor",
    "V4_PROVENANCE_DAG_BUDGET_SCHEMA",
    "V4_PROVENANCE_DAG_DIRECT_EDGE_RECORD_SCHEMA",
    "V4_PROVENANCE_DAG_LEAF_ENDPOINT_RECORD_SCHEMA",
    "V4_PROVENANCE_DAG_NODE_ENDPOINT_RECORD_SCHEMA",
    "V4_PROVENANCE_DAG_PARENT_REQUEST_RECORD_SCHEMA",
    "V4_PROVENANCE_DAG_RESULT_SCHEMA",
    "V4_PROVENANCE_DAG_STREAM_DESCRIPTOR_SCHEMA",
    "V4_PROVENANCE_DAG_STREAM_DIRECT_EDGE",
    "V4_PROVENANCE_DAG_STREAM_LEAF_ENDPOINT",
    "V4_PROVENANCE_DAG_STREAM_NODE_ENDPOINT",
    "V4_PROVENANCE_DAG_STREAM_PARENT_REQUEST",
    "V4_PROVENANCE_DAG_STREAM_SORTED_LEAF",
    "V4_PROVENANCE_DAG_STREAM_SORTED_PARENT_REQUEST",
    "build_v4_provenance_direct_dag",
    "decode_v4_provenance_direct_edge_record",
    "decode_v4_provenance_leaf_endpoint_record",
    "decode_v4_provenance_node_endpoint_record",
    "decode_v4_provenance_parent_request_record",
    "encode_v4_provenance_direct_edge_record",
    "encode_v4_provenance_leaf_endpoint_record",
    "encode_v4_provenance_node_endpoint_record",
    "encode_v4_provenance_parent_request_record",
    "revalidate_v4_provenance_direct_dag_result",
]
