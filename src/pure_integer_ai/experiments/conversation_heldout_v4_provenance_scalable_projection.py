"""DLG-05 v4 R04b P2-B3 的三路可扩展 leaf provenance 投影。

本模块只消费同一 create-new K work capability 内已封存的 P2-B1/B2 结果。它先逐条
重验 B1/B2 的全部 P0 descriptor，再分别物化并外排 SourceRef、内容 SHA-256 与谱系
投影。三条流彼此不合并，也不产出 duplicate/conflict verdict、cursor、receipt、manifest
或任何发布物。

每次真实消费上游 P0 stream 都重比 footer、物理字节数、SHA-256 与路径身份。因此可观察
到的替换、截断或重编码只能使本切片 fail closed；不能返回 B3 result。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import BinaryIO, Callable, Iterable

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4ProvenanceError,
    ConversationHeldOutV4ProvenanceLeaf,
    ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_SIDE_HELD_OUT,
    V4_PROVENANCE_SIDE_TRAINING,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_closure import (
    ConversationHeldOutV4ProvenanceClosureBudget,
    ConversationHeldOutV4ProvenanceClosureResult,
    ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
    ConversationHeldOutV4ProvenanceScalableClosureError,
    build_v4_provenance_transitive_closure,
    decode_v4_provenance_closure_pair_record,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_dag import (
    ConversationHeldOutV4ProvenanceDagStreamDescriptor,
    ConversationHeldOutV4ProvenanceDirectDagResult,
    ConversationHeldOutV4ProvenanceScalableDagError,
    decode_v4_provenance_leaf_endpoint_record,
    decode_v4_provenance_node_endpoint_record,
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
    open_existing_run_root,
    open_exclusive_binary,
    open_plain_binary,
    require_created_run_root,
    require_fresh_empty_run_root,
    require_plain_file_identity,
)


V4_PROVENANCE_PROJECTION_BUDGET_SCHEMA = 1
V4_PROVENANCE_PROJECTION_STREAM_DESCRIPTOR_SCHEMA = 1
V4_PROVENANCE_PROJECTION_RESULT_SCHEMA = 1

V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF = 1
V4_PROVENANCE_PROJECTION_STREAM_CONTENT = 2
V4_PROVENANCE_PROJECTION_STREAM_LINEAGE = 3

V4_PROVENANCE_PROJECTION_SOURCE_REF_RECORD_SCHEMA = 1
V4_PROVENANCE_PROJECTION_CONTENT_RECORD_SCHEMA = 1
V4_PROVENANCE_PROJECTION_LINEAGE_RECORD_SCHEMA = 1

V4_PROVENANCE_PROJECTION_NAMESPACE_LINEAGE_NODE = 1

_STREAM_KINDS = frozenset({
    V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF,
    V4_PROVENANCE_PROJECTION_STREAM_CONTENT,
    V4_PROVENANCE_PROJECTION_STREAM_LINEAGE,
})
_LINEAGE_NAMESPACES = frozenset({
    V4_PROVENANCE_PROJECTION_NAMESPACE_LINEAGE_NODE,
})

_STAGE_SOURCE_REF_MATERIALIZE = "stage-14-projection-source-ref"
_STAGE_SOURCE_REF_SORT = "stage-15-projection-source-ref-sort"
_STAGE_CONTENT_MATERIALIZE = "stage-16-projection-content"
_STAGE_CONTENT_SORT = "stage-17-projection-content-sort"
_STAGE_LINEAGE_MATERIALIZE = "stage-18-projection-lineage"
_STAGE_LINEAGE_SORT = "stage-19-projection-lineage-sort"

_SOURCE_REF_RAW_FILE = Path("source-ref-raw.pifrs")
_SOURCE_REF_SORTED_FILE = Path("source-ref-sorted.pifrs")
_CONTENT_RAW_FILE = Path("content-raw.pifrs")
_CONTENT_SORTED_FILE = Path("content-sorted.pifrs")
_LINEAGE_RAW_FILE = Path("lineage-raw.pifrs")
_LINEAGE_SORTED_FILE = Path("lineage-sorted.pifrs")


# object-model: exception
class ConversationHeldOutV4ProvenanceScalableProjectionError(RuntimeError):
    """B3 三路 provenance 投影无法在冻结边界内机械闭合。"""


# object-model: exception
class ConversationHeldOutV4ProvenanceScalableProjectionBudgetExceeded(
        ConversationHeldOutV4ProvenanceScalableProjectionError):
    """B3 的 records、group、physical、handle 或累计物化预算被超过。"""


def _fail(message: str) -> None:
    """统一产生 B3 的领域 fail-closed 错误。"""
    raise ConversationHeldOutV4ProvenanceScalableProjectionError(message)


def _budget_fail(message: str) -> None:
    """统一产生 B3 预算耗尽，避免把结构损坏伪装为资源问题。"""
    raise ConversationHeldOutV4ProvenanceScalableProjectionBudgetExceeded(message)


def _require_positive_int(value: int, *, label: str) -> int:
    """拒绝 bool、零和负数的固定资源或格式字段。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def _require_nonnegative_int(value: int, *, label: str) -> int:
    """拒绝 bool 和负数的累计统计字段。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _require_stage_name(value: str) -> str:
    """约束逻辑阶段名为无路径语义、可稳定外排的 ASCII 标识。"""
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
    """把受限 work 相对路径表示为不携带本机 root 的 ASCII 整数键。"""
    if (not isinstance(value, Path) or value.is_absolute() or value.drive
            or value.root or not value.parts or ".." in value.parts
            or any(part in {"", ".", ".."} for part in value.parts)):
        raise ValueError(f"{label} 必须是非空且不含 .. 的相对 Path")
    text = value.as_posix()
    if any(not character.isascii() for character in text):
        raise ValueError(f"{label} 必须只含 ASCII")
    return tuple(ord(character) for character in text)


def _digest_key(value: KRunFileDigest) -> tuple[int, ...]:
    """展开物理字节数和完整 SHA-256，避免以摘要替代 descriptor 本体。"""
    if not isinstance(value, KRunFileDigest):
        raise TypeError("physical 必须是 KRunFileDigest")
    return (value.byte_count, *value.sha256)


def _packed_sort_key(*parts: tuple[int, ...]) -> tuple[int, ...]:
    """以长度分帧连接排序字段，避免相邻完整键产生歧义。"""
    result: list[int] = []
    for part in parts:
        pack_key(result, part)
    return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceProjectionBudget:
    """B3 三路物化、lineage group、P0 与外排的完整有限资源合同。"""

    max_node_count: int
    max_leaf_count: int
    max_known_pair_count: int
    max_leaves_per_node: int
    max_ancestors_per_node: int
    max_lineage_projection_record_count: int
    max_record_payload_bytes: int
    max_stream_record_count: int
    max_stream_payload_bytes: int
    max_stream_physical_bytes: int
    max_total_materialized_physical_bytes: int
    max_open_files: int
    external_sort_budget: IntegerExternalSortBudget
    closure_replay_budget: ConversationHeldOutV4ProvenanceClosureBudget

    def __post_init__(self) -> None:
        """拒绝无界 node、leaf、ancestor group、stream、外排与物化配置。"""
        for label, value in (
                ("max_node_count", self.max_node_count),
                ("max_leaf_count", self.max_leaf_count),
                ("max_known_pair_count", self.max_known_pair_count),
                ("max_leaves_per_node", self.max_leaves_per_node),
                ("max_ancestors_per_node", self.max_ancestors_per_node),
                ("max_lineage_projection_record_count",
                 self.max_lineage_projection_record_count),
                ("max_record_payload_bytes", self.max_record_payload_bytes),
                ("max_stream_record_count", self.max_stream_record_count),
                ("max_stream_payload_bytes", self.max_stream_payload_bytes),
                ("max_stream_physical_bytes", self.max_stream_physical_bytes),
                ("max_total_materialized_physical_bytes",
                 self.max_total_materialized_physical_bytes),
                ("max_open_files", self.max_open_files)):
            _require_positive_int(value, label=label)
        if not isinstance(self.external_sort_budget, IntegerExternalSortBudget):
            raise TypeError("B3 external_sort_budget 类型错误")
        if not isinstance(self.closure_replay_budget,
                          ConversationHeldOutV4ProvenanceClosureBudget):
            raise TypeError("B3 closure_replay_budget 类型错误")
        if self.max_open_files < 4:
            raise ValueError("B3 max_open_files 至少覆盖三个 reader 和一个 writer")
        if self.external_sort_budget.max_open_files > self.max_open_files:
            raise ValueError("B3 external sort 打开句柄预算不得超过 B3 上限")
        if (self.external_sort_budget.max_output_physical_bytes
                > self.max_stream_physical_bytes):
            raise ValueError("B3 external sort 输出不得超过 stream 物理预算")
        if self.max_record_payload_bytes > self.max_stream_payload_bytes:
            raise ValueError("B3 单 record payload 不得超过 stream payload 预算")
        if any(value > self.max_stream_record_count for value in (
                self.max_node_count,
                self.max_leaf_count,
                self.max_known_pair_count,
                self.max_lineage_projection_record_count)):
            raise ValueError("B3 单类 record 预算不得超过单 stream record 预算")
        if self.max_lineage_projection_record_count < self.max_leaf_count:
            raise ValueError("B3 lineage 投影预算至少覆盖每 leaf 的 self endpoint")
        replay = self.closure_replay_budget
        if replay.max_node_count > self.max_node_count:
            raise ValueError("B3 closure replay node 预算不得超过 B3 上限")
        if replay.max_known_pair_count > self.max_known_pair_count:
            raise ValueError("B3 closure replay known pair 预算不得超过 B3 上限")
        if replay.max_record_payload_bytes > self.max_record_payload_bytes:
            raise ValueError("B3 closure replay record payload 不得超过 B3 上限")
        if replay.max_stream_record_count > self.max_stream_record_count:
            raise ValueError("B3 closure replay stream record 不得超过 B3 上限")
        if replay.max_stream_payload_bytes > self.max_stream_payload_bytes:
            raise ValueError("B3 closure replay stream payload 不得超过 B3 上限")
        if replay.max_stream_physical_bytes > self.max_stream_physical_bytes:
            raise ValueError("B3 closure replay stream physical 不得超过 B3 上限")
        if replay.max_open_files > self.max_open_files:
            raise ValueError("B3 closure replay open handle 不得超过 B3 上限")
        if (replay.max_total_materialized_physical_bytes
                > self.max_total_materialized_physical_bytes):
            raise ValueError("B3 closure replay total materialized 不得超过 B3 上限")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 K root 的完整 B3 资源身份。"""
        result = [V4_PROVENANCE_PROJECTION_BUDGET_SCHEMA]
        result.extend((
            self.max_node_count,
            self.max_leaf_count,
            self.max_known_pair_count,
            self.max_leaves_per_node,
            self.max_ancestors_per_node,
            self.max_lineage_projection_record_count,
            self.max_record_payload_bytes,
            self.max_stream_record_count,
            self.max_stream_payload_bytes,
            self.max_stream_physical_bytes,
            self.max_total_materialized_physical_bytes,
            self.max_open_files,
        ))
        pack_key(result, self.external_sort_budget.integer_stream())
        pack_key(result, self.closure_replay_budget.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 schema-backed、未丢字段的预算稳定键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceProjectionStreamDescriptor:
    """一个 B3 已排序、未发布 P0 projection 的相对位置与物理身份。"""

    stream_kind: int
    work_relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """固定 projection 类型、相对路径、footer 与完整物理 SHA-256。"""
        if type(self.stream_kind) is not int or self.stream_kind not in _STREAM_KINDS:
            raise ValueError("B3 stream_kind 未注册")
        _relative_path_key(self.work_relative_path, label="B3 work_relative_path")
        if not isinstance(self.p0_footer, IntegerFramedStreamFooter):
            raise TypeError("B3 p0_footer 类型错误")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("B3 physical 类型错误")
        if self.physical.byte_count <= 0:
            raise ValueError("B3 sealed P0 stream 物理字节必须为正")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不携带 K root 的完整 stream descriptor 整数表示。"""
        result = [V4_PROVENANCE_PROJECTION_STREAM_DESCRIPTOR_SCHEMA,
                  self.stream_kind]
        for value in (
                _relative_path_key(self.work_relative_path,
                                   label="B3 work_relative_path"),
                self.p0_footer.integer_tuple(),
                _digest_key(self.physical)):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 P2-C 可绑定的 descriptor 完整稳定键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceProjectionResult:
    """B3 的三份已排序 leaf projection；不表达交叉结论或来源资格。"""

    logical_stage_name: str
    direct_dag_stable_key: tuple[int, ...]
    closure_stable_key: tuple[int, ...]
    source_ref_projection_stream: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor
    content_projection_stream: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor
    lineage_projection_stream: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor
    source_ref_projection_count: int
    content_projection_count: int
    lineage_projection_count: int
    materialized_physical_bytes: int
    budget: ConversationHeldOutV4ProvenanceProjectionBudget

    def __post_init__(self) -> None:
        """绑定两级上游 identity、三路 sorted descriptor 和实际物化统计。"""
        _require_stage_name(self.logical_stage_name)
        for label, value in (
                ("B3 direct_dag_stable_key", self.direct_dag_stable_key),
                ("B3 closure_stable_key", self.closure_stable_key)):
            try:
                strict_integer_tuple(value, label=label)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} 必须是完整严格整数 tuple") from exc
        for descriptor, kind in (
                (self.source_ref_projection_stream,
                 V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF),
                (self.content_projection_stream,
                 V4_PROVENANCE_PROJECTION_STREAM_CONTENT),
                (self.lineage_projection_stream,
                 V4_PROVENANCE_PROJECTION_STREAM_LINEAGE)):
            if not isinstance(descriptor,
                              ConversationHeldOutV4ProvenanceProjectionStreamDescriptor):
                raise TypeError("B3 result 含非 projection stream descriptor")
            if descriptor.stream_kind != kind:
                raise ValueError("B3 result stream descriptor 类型错位")
        for label, value in (
                ("source_ref_projection_count", self.source_ref_projection_count),
                ("content_projection_count", self.content_projection_count),
                ("lineage_projection_count", self.lineage_projection_count),
                ("materialized_physical_bytes",
                 self.materialized_physical_bytes)):
            _require_nonnegative_int(value, label=f"B3 {label}")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceProjectionBudget):
            raise TypeError("B3 result budget 类型错误")
        if (self.source_ref_projection_count
                != self.source_ref_projection_stream.p0_footer.record_count):
            raise ValueError("B3 SourceRef count 与 footer 不一致")
        if (self.content_projection_count
                != self.content_projection_stream.p0_footer.record_count):
            raise ValueError("B3 content count 与 footer 不一致")
        if (self.lineage_projection_count
                != self.lineage_projection_stream.p0_footer.record_count):
            raise ValueError("B3 lineage count 与 footer 不一致")
        if self.source_ref_projection_count != self.content_projection_count:
            raise ValueError("B3 SourceRef/content projection count 必须相等")
        if self.source_ref_projection_count > self.budget.max_leaf_count:
            raise ValueError("B3 leaf projection count 超过预算")
        if self.lineage_projection_count > self.budget.max_lineage_projection_record_count:
            raise ValueError("B3 lineage projection count 超过预算")
        if self.materialized_physical_bytes > self.budget.max_total_materialized_physical_bytes:
            raise ValueError("B3 materialized physical bytes 超过预算")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 K root 的 B3 完整结果 identity。"""
        result = [V4_PROVENANCE_PROJECTION_RESULT_SCHEMA]
        pack_key(result, tuple(ord(item) for item in self.logical_stage_name))
        pack_key(result, self.direct_dag_stable_key)
        pack_key(result, self.closure_stable_key)
        for descriptor in (
                self.source_ref_projection_stream,
                self.content_projection_stream,
                self.lineage_projection_stream):
            pack_key(result, descriptor.integer_stream())
        result.extend((
            self.source_ref_projection_count,
            self.content_projection_count,
            self.lineage_projection_count,
            self.materialized_physical_bytes,
        ))
        pack_key(result, self.budget.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回后续只读 merge 层可绑定的完整 B3 identity。"""
        return self.integer_stream()


# object-model: resource_owner; representation=protocol; interop=pending
class _DigestingBinaryHandle:
    """在同一 P0 handle 生命周期中统计物理 SHA-256，绝不缓存整文件。"""

    __slots__ = ("_byte_count", "_digest", "_max_bytes", "_stream")

    def __init__(self, stream: BinaryIO, *, max_bytes: int) -> None:
        """接管 K boundary 已打开的 handle 并固定本次物理 I/O 上限。"""
        self._byte_count = 0
        self._digest = hashlib.sha256()
        self._max_bytes = _require_positive_int(
            max_bytes, label="B3 physical max_bytes")
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        """按实际读取字节累计 SHA，并在 P0 footer 前阻止超限读取。"""
        if type(size) is not int:
            raise TypeError("B3 physical read size 必须是严格整数")
        remaining_with_sentinel = self._max_bytes - self._byte_count + 1
        requested = min(
            size if size >= 0 else remaining_with_sentinel,
            remaining_with_sentinel,
        )
        result = self._stream.read(requested)
        if isinstance(result, bytes):
            self._accept(result)
        return result

    def write(self, data: bytes | bytearray | memoryview) -> int:
        """只累计底层已确认写出的前缀，部分写入交给 P0 writer 失败处理。"""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("B3 physical write data 必须是 bytes-like")
        if len(data) > self._max_bytes - self._byte_count:
            _budget_fail("B3 P0 stream physical bytes 超过预算")
        written = self._stream.write(data)
        if type(written) is int and 0 < written <= len(data):
            self._accept(bytes(memoryview(data)[:written]))
        return written

    def flush(self) -> None:
        """透传 P0 seal 的 flush，不把它伪造成物理写入。"""
        self._stream.flush()

    def close(self) -> None:
        """关闭唯一持有的底层 handle。"""
        self._stream.close()

    def physical(self) -> KRunFileDigest:
        """返回本 handle 已成功处理字节的完整物理身份。"""
        return KRunFileDigest(self._byte_count, tuple(self._digest.digest()))

    def _accept(self, value: bytes) -> None:
        """在更新 SHA 前拒绝任何超过冻结物理预算的实际 I/O。"""
        if len(value) > self._max_bytes - self._byte_count:
            _budget_fail("B3 P0 stream physical bytes 超过预算")
        self._digest.update(value)
        self._byte_count += len(value)


# object-model: resource_owner; representation=protocol; interop=pending
class _VerifiedP0Reader:
    """完整消费一个 descriptor，并在 finish 时复核 footer、物理和路径身份。"""

    __slots__ = (
        "_expected_footer", "_expected_physical", "_file_identity", "_finished",
        "_handle", "_reader", "_relative", "_root",
    )

    def __init__(
            self,
            root: KRunRoot,
            relative: Path,
            *,
            footer: IntegerFramedStreamFooter,
            physical: KRunFileDigest,
            budget: ConversationHeldOutV4ProvenanceProjectionBudget,
            label: str,
            ) -> None:
        """以 capture/open/post-read 三段边界固定 descriptor 的实际消费。"""
        if physical.byte_count > budget.max_stream_physical_bytes:
            _budget_fail(f"{label} physical bytes 超过 B3 stream 预算")
        self._expected_footer = footer
        self._expected_physical = physical
        self._root = root
        self._relative = relative
        self._finished = False
        self._file_identity = capture_plain_file_identity(
            root, relative, label=f"{label} capture")
        if self._file_identity.byte_count != physical.byte_count:
            _fail(f"{label} 文件字节数与冻结 descriptor 漂移")
        stream = open_plain_binary(
            root,
            relative,
            label=label,
            expected_identity=self._file_identity,
        )
        self._handle = _DigestingBinaryHandle(stream, max_bytes=physical.byte_count)
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

    def __enter__(self) -> "_VerifiedP0Reader":
        """返回唯一 P0 cursor；正常路径必须以 finish 结束。"""
        return self

    def __exit__(self, exc_type: object, exc_value: object,
                 traceback: object) -> None:
        """异常路径只关闭尚未可信的输入 handle。"""
        self.close()

    def read_record(self) -> tuple[int, ...] | None:
        """读取下一帧；codec 错误自然阻断当前 descriptor。"""
        return self._reader.read_record()

    def finish(self) -> IntegerFramedStreamFooter:
        """完整消费后逐字段比较 footer、物理 SHA 与读后路径身份。"""
        if self._finished:
            _fail("B3 verified P0 reader 不得重复 finish")
        footer = self._reader.finish()
        physical = self._handle.physical()
        require_plain_file_identity(
            self._root,
            self._relative,
            self._file_identity,
            label="B3 P0 post-read",
        )
        if footer != self._expected_footer:
            _fail("B3 P0 footer 与冻结 descriptor 漂移")
        if physical != self._expected_physical:
            _fail("B3 P0 physical bytes 或 SHA-256 与冻结 descriptor 漂移")
        self._finished = True
        return footer

    def close(self) -> None:
        """关闭 P0 cursor；正常 finish 后保持幂等。"""
        self._reader.close()


def _descriptor_footer_physical(
        value: ConversationHeldOutV4ProvenanceDagStreamDescriptor
        | ConversationHeldOutV4ProvenanceClosureStreamDescriptor
        | ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        ) -> tuple[IntegerFramedStreamFooter, KRunFileDigest]:
    """从 B1/B2/B3 descriptor 提取完整 P0 footer 与物理身份。"""
    if isinstance(value, (
            ConversationHeldOutV4ProvenanceDagStreamDescriptor,
            ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
            ConversationHeldOutV4ProvenanceProjectionStreamDescriptor)):
        return value.p0_footer, value.physical
    raise TypeError("B3 stream descriptor 类型错误")


def _verify_descriptor(
        root: KRunRoot,
        relative: Path,
        descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor
        | ConversationHeldOutV4ProvenanceClosureStreamDescriptor
        | ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        *,
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        label: str,
        ) -> None:
    """逐条消费未解释的 P0 descriptor，锁定全部上游物理身份。"""
    footer, physical = _descriptor_footer_physical(descriptor)
    with _VerifiedP0Reader(
            root,
            relative,
            footer=footer,
            physical=physical,
            budget=budget,
            label=label,
    ) as reader:
        reader.finish()


def _write_stream(
        root: KRunRoot,
        relative: Path,
        *,
        stage: str,
        stream_kind: int,
        records: Iterable[tuple[int, ...]],
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        label: str,
        ) -> ConversationHeldOutV4ProvenanceProjectionStreamDescriptor:
    """排他写出一个 record/payload/physical 有界的 sealed B3 P0 stream。"""
    _require_stage_name(stage)
    if type(stream_kind) is not int or stream_kind not in _STREAM_KINDS:
        raise ValueError("B3 output stream_kind 未注册")
    stream = open_exclusive_binary(root, relative, label=label)
    handle = _DigestingBinaryHandle(
        stream,
        max_bytes=budget.max_stream_physical_bytes,
    )
    writer = IntegerFramedStreamWriter.from_open_binary(handle, path=relative)
    count = 0
    payload_bytes = 0
    try:
        for record in records:
            strict_integer_tuple(record, label="B3 output record")
            encoded_size = encoded_integer_tuple_size(record)
            if encoded_size > budget.max_record_payload_bytes:
                _budget_fail("B3 单 record payload 超过预算")
            if count >= budget.max_stream_record_count:
                _budget_fail("B3 P0 stream record 数超过预算")
            if encoded_size > budget.max_stream_payload_bytes - payload_bytes:
                _budget_fail("B3 P0 stream payload 超过预算")
            writer.append(record)
            count += 1
            payload_bytes += encoded_size
        footer = writer.seal()
    finally:
        writer.close()
    physical = handle.physical()
    identity = capture_plain_file_identity(root, relative, label=f"{label} identity")
    if (footer.record_count != count
            or footer.total_payload_bytes != payload_bytes
            or physical.byte_count != identity.byte_count):
        _fail(f"{label} 写入后的 P0 footer 或 physical 身份漂移")
    descriptor = ConversationHeldOutV4ProvenanceProjectionStreamDescriptor(
        stream_kind,
        Path(stage) / relative,
        footer,
        physical,
    )
    _verify_descriptor(
        root,
        relative,
        descriptor,
        budget=budget,
        label=f"{label} verification",
    )
    return descriptor


def _new_stage(work_root: KRunRoot, stage: str) -> KRunRoot:
    """在原始 create-new run 中排他创建一个空的 B3 sibling stage capability。"""
    _require_stage_name(stage)
    result = create_new_run_root(
        work_root.path / stage,
        require_k_drive=not work_root.test_transport,
        label=f"B3 {stage}",
    )
    require_fresh_empty_run_root(result, label=f"B3 {stage}")
    return result


def _stage_local_input(
        work_root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor
        | ConversationHeldOutV4ProvenanceClosureStreamDescriptor
        | ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        *, label: str,
        ) -> tuple[KRunRoot, Path]:
    """从 root-relative descriptor 恢复单层 stage capability 与安全文件名。"""
    relative = descriptor.work_relative_path
    _relative_path_key(relative, label=f"{label} descriptor path")
    if len(relative.parts) != 2:
        _fail(f"{label} descriptor 必须定位到单层 stage sibling")
    stage, filename = relative.parts
    stage_root = open_existing_run_root(
        work_root.path / stage,
        require_k_drive=not work_root.test_transport,
        label=f"{label} source stage",
    )
    return stage_root, Path(filename)


def _reserve_materialized(
        current: int,
        maximum_next: int,
        *,
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        label: str,
        ) -> None:
    """在创建下一 stage 前保守预留可能写入的物理字节。"""
    _require_nonnegative_int(current, label="B3 materialized current")
    _require_positive_int(maximum_next, label="B3 materialized maximum_next")
    if maximum_next > budget.max_total_materialized_physical_bytes - current:
        _budget_fail(f"{label} 可能物化量超过 B3 总预算")


def _add_materialized(
        current: int,
        addition: int,
        *,
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        label: str,
        ) -> int:
    """把已实际写入的 B3 stream/temporary bytes 纳入累计硬上限。"""
    _require_nonnegative_int(current, label="B3 materialized current")
    _require_nonnegative_int(addition, label="B3 materialized addition")
    if addition > budget.max_total_materialized_physical_bytes - current:
        _budget_fail(f"{label} 实际物化量超过 B3 总预算")
    return current + addition


def _read_protocol_key(reader: IntegerStreamReader, *, label: str) -> ProtocolKey:
    """从完整长度分帧键恢复 ProtocolKey，拒绝摘要或截断替代。"""
    try:
        return ProtocolKey(reader.read_key(label=label))
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail(f"{label} ProtocolKey 非法")
        raise AssertionError from exc


def _read_source_ref(reader: IntegerStreamReader, *, label: str) -> SourceRef:
    """从完整十一整数 SourceRef 键恢复一等来源本体。"""
    try:
        return SourceRef.from_stable_key(reader.read_key(label=label))
    except (AssertionError, IntegerCodecError, TypeError, ValueError) as exc:
        _fail(f"{label} SourceRef 非法")
        raise AssertionError from exc


def _read_leaf(reader: IntegerStreamReader, *, label: str) -> ConversationHeldOutV4ProvenanceLeaf:
    """恢复完整 leaf，不允许调用方拼接投影中的残缺 leaf。"""
    try:
        return ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(
            reader.read_key(label=label))
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail(f"{label} provenance leaf 非法")
        raise AssertionError from exc


def encode_v4_provenance_source_ref_projection_record(
        source_ref: SourceRef,
        side: int,
        leaf: ConversationHeldOutV4ProvenanceLeaf,
        ) -> tuple[int, ...]:
    """编码 `(full SourceRef, side, full leaf)`，不折叠跨 side 或同源 leaf。"""
    if not isinstance(source_ref, SourceRef):
        raise TypeError("B3 SourceRef projection source_ref 必须是 SourceRef")
    if not isinstance(leaf, ConversationHeldOutV4ProvenanceLeaf):
        raise TypeError("B3 SourceRef projection leaf 类型错误")
    if type(side) is not int or side != leaf.side:
        raise ValueError("B3 SourceRef projection side 必须等于 leaf side")
    if source_ref.stable_key() != leaf.source_ref.stable_key():
        raise ValueError("B3 SourceRef projection source_ref 必须等于 leaf SourceRef")
    result = [V4_PROVENANCE_PROJECTION_SOURCE_REF_RECORD_SCHEMA]
    pack_key(result, source_ref.stable_key())
    result.append(side)
    pack_key(result, leaf.integer_stream())
    return tuple(result)


def decode_v4_provenance_source_ref_projection_record(
        record: tuple[int, ...],
        ) -> tuple[SourceRef, int, ConversationHeldOutV4ProvenanceLeaf]:
    """严格恢复 SourceRef projection，并重验其与 full leaf 的闭合。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label="B3 SourceRef projection schema")
        if schema != V4_PROVENANCE_PROJECTION_SOURCE_REF_RECORD_SCHEMA:
            _fail("B3 SourceRef projection schema 不兼容")
        source_ref = _read_source_ref(reader, label="B3 SourceRef projection source_ref")
        side = reader.read_positive(label="B3 SourceRef projection side")
        leaf = _read_leaf(reader, label="B3 SourceRef projection leaf")
        reader.finish()
    except ConversationHeldOutV4ProvenanceScalableProjectionError:
        raise
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail("B3 SourceRef projection record 非法")
        raise AssertionError from exc
    if side != leaf.side or source_ref.stable_key() != leaf.source_ref.stable_key():
        _fail("B3 SourceRef projection 与 leaf 不闭合")
    return source_ref, side, leaf


def encode_v4_provenance_content_projection_record(
        content_sha256: tuple[int, ...],
        side: int,
        leaf: ConversationHeldOutV4ProvenanceLeaf,
        ) -> tuple[int, ...]:
    """编码 `(full content SHA-256, side, full leaf)`，不进行内容去重或 verdict。"""
    if not isinstance(leaf, ConversationHeldOutV4ProvenanceLeaf):
        raise TypeError("B3 content projection leaf 类型错误")
    try:
        strict_integer_tuple(content_sha256, label="B3 content projection SHA-256")
    except (TypeError, ValueError) as exc:
        raise ValueError("B3 content projection SHA-256 必须是严格整数 tuple") from exc
    if content_sha256 != leaf.content_sha256:
        raise ValueError("B3 content projection SHA-256 必须等于 leaf content")
    if type(side) is not int or side != leaf.side:
        raise ValueError("B3 content projection side 必须等于 leaf side")
    result = [V4_PROVENANCE_PROJECTION_CONTENT_RECORD_SCHEMA]
    pack_key(result, content_sha256)
    result.append(side)
    pack_key(result, leaf.integer_stream())
    return tuple(result)


def decode_v4_provenance_content_projection_record(
        record: tuple[int, ...],
        ) -> tuple[tuple[int, ...], int, ConversationHeldOutV4ProvenanceLeaf]:
    """严格恢复 content projection，并重验内容 SHA 和 side 都来自 full leaf。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label="B3 content projection schema")
        if schema != V4_PROVENANCE_PROJECTION_CONTENT_RECORD_SCHEMA:
            _fail("B3 content projection schema 不兼容")
        content_sha256 = reader.read_key(label="B3 content projection SHA-256")
        side = reader.read_positive(label="B3 content projection side")
        leaf = _read_leaf(reader, label="B3 content projection leaf")
        reader.finish()
    except ConversationHeldOutV4ProvenanceScalableProjectionError:
        raise
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail("B3 content projection record 非法")
        raise AssertionError from exc
    if content_sha256 != leaf.content_sha256 or side != leaf.side:
        _fail("B3 content projection 与 leaf 不闭合")
    return content_sha256, side, leaf


def encode_v4_provenance_lineage_projection_record(
        namespace_tag: int,
        ancestor_node_key: ProtocolKey,
        ancestor_snapshot: ConversationHeldOutV4SnapshotIdentity,
        side: int,
        leaf: ConversationHeldOutV4ProvenanceLeaf,
        ) -> tuple[int, ...]:
    """编码完整 lineage 项，self 与 strict ancestor 均显式保留 node/snapshot 本体。"""
    if type(namespace_tag) is not int or namespace_tag not in _LINEAGE_NAMESPACES:
        raise ValueError("B3 lineage namespace_tag 未注册")
    if not isinstance(ancestor_node_key, ProtocolKey):
        raise TypeError("B3 lineage ancestor_node_key 必须是 ProtocolKey")
    if not isinstance(ancestor_snapshot, ConversationHeldOutV4SnapshotIdentity):
        raise TypeError("B3 lineage ancestor_snapshot 类型错误")
    if not isinstance(leaf, ConversationHeldOutV4ProvenanceLeaf):
        raise TypeError("B3 lineage leaf 类型错误")
    if type(side) is not int or side != leaf.side:
        raise ValueError("B3 lineage side 必须等于 leaf side")
    result = [V4_PROVENANCE_PROJECTION_LINEAGE_RECORD_SCHEMA, namespace_tag]
    pack_key(result, ancestor_node_key.components)
    pack_key(result, ancestor_snapshot.integer_stream())
    result.append(side)
    pack_key(result, leaf.integer_stream())
    return tuple(result)


def decode_v4_provenance_lineage_projection_record(
        record: tuple[int, ...],
        ) -> tuple[
            int,
            ProtocolKey,
            ConversationHeldOutV4SnapshotIdentity,
            int,
            ConversationHeldOutV4ProvenanceLeaf,
        ]:
    """严格恢复完整 lineage projection，不允许 URI/digest 替代 ancestor 本体。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label="B3 lineage projection schema")
        if schema != V4_PROVENANCE_PROJECTION_LINEAGE_RECORD_SCHEMA:
            _fail("B3 lineage projection schema 不兼容")
        namespace_tag = reader.read_positive(label="B3 lineage namespace_tag")
        if namespace_tag not in _LINEAGE_NAMESPACES:
            _fail("B3 lineage namespace_tag 未注册")
        ancestor_node_key = _read_protocol_key(
            reader, label="B3 lineage ancestor_node_key")
        ancestor_snapshot = ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
            reader.read_key(label="B3 lineage ancestor_snapshot"))
        side = reader.read_positive(label="B3 lineage side")
        leaf = _read_leaf(reader, label="B3 lineage leaf")
        reader.finish()
    except ConversationHeldOutV4ProvenanceScalableProjectionError:
        raise
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail("B3 lineage projection record 非法")
        raise AssertionError from exc
    if side != leaf.side:
        _fail("B3 lineage projection side 与 leaf 不闭合")
    return namespace_tag, ancestor_node_key, ancestor_snapshot, side, leaf


def _source_ref_projection_sort_key(record: tuple[int, ...]) -> tuple[int, ...]:
    """按完整 SourceRef、side、full leaf 排序，保持所有 leaf 而不是合并集合。"""
    source_ref, side, leaf = decode_v4_provenance_source_ref_projection_record(record)
    return _packed_sort_key(
        source_ref.stable_key(),
        (side,),
        leaf.integer_stream(),
    )


def _content_projection_sort_key(record: tuple[int, ...]) -> tuple[int, ...]:
    """按完整 content SHA-256、side、full leaf 排序，不给出任何重复结论。"""
    content_sha256, side, leaf = decode_v4_provenance_content_projection_record(record)
    return _packed_sort_key(content_sha256, (side,), leaf.integer_stream())


def _lineage_projection_sort_key(record: tuple[int, ...]) -> tuple[int, ...]:
    """按 namespace、完整 ancestor node/snapshot、side、full leaf 排序。"""
    namespace_tag, ancestor, snapshot, side, leaf = (
        decode_v4_provenance_lineage_projection_record(record))
    return _packed_sort_key(
        (namespace_tag,),
        ancestor.components,
        snapshot.integer_stream(),
        (side,),
        leaf.integer_stream(),
    )


def _verify_b1_result(
        work_root: KRunRoot,
        result: ConversationHeldOutV4ProvenanceDirectDagResult,
        *, budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        ) -> None:
    """重放 B1 全部 six descriptors，不能因 B3 只用其中两条流而遗漏漂移。"""
    for label, descriptor in (
            ("node endpoint", result.node_endpoint_stream),
            ("parent request", result.parent_request_stream),
            ("sorted parent request", result.sorted_parent_request_stream),
            ("direct edge", result.direct_edge_stream),
            ("sorted leaf", result.sorted_leaf_stream),
            ("leaf endpoint", result.leaf_endpoint_stream)):
        _verify_descriptor(
            work_root,
            descriptor.work_relative_path,
            descriptor,
            budget=budget,
            label=f"B3 B1 {label} revalidation",
        )


def _verify_b2_result(
        work_root: KRunRoot,
        result: ConversationHeldOutV4ProvenanceClosureResult,
        *, budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        ) -> None:
    """重放 B2 全部 three descriptors，空 frontier 也不能跳过身份核验。"""
    for label, descriptor in (
            ("edge by child", result.edge_by_child_stream),
            ("known ancestor", result.known_ancestor_stream),
            ("empty frontier", result.empty_frontier_stream)):
        _verify_descriptor(
            work_root,
            descriptor.work_relative_path,
            descriptor,
            budget=budget,
            label=f"B3 B2 {label} revalidation",
        )


def _sort_one_stream(
        work_root: KRunRoot,
        source: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        output_root: KRunRoot,
        *,
        stage: str,
        output_relative: Path,
        stream_kind: int,
        logical_stage_name: str,
        sort_key: Callable[[tuple[int, ...]], tuple[int, ...]],
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        label: str,
        ) -> tuple[ConversationHeldOutV4ProvenanceProjectionStreamDescriptor, int]:
    """对一条独立 raw projection 外排，并回比实际消费的输入 descriptor。"""
    source_root, source_relative = _stage_local_input(
        work_root, source, label=label)
    try:
        result = external_sort_sealed_integer_records(
            source_root,
            (source_relative,),
            output_root,
            output_relative_path=output_relative,
            logical_stage_name=logical_stage_name,
            sort_key=sort_key,
            budget=budget.external_sort_budget,
        )
    except (IntegerExternalSortBudgetExceeded,
            IntegerFramedStreamBudgetExceeded) as exc:
        _budget_fail(f"{label} external sort 超过冻结预算")
        raise AssertionError from exc
    except IntegerExternalSortError as exc:
        _fail(f"{label} external sort 失败")
        raise AssertionError from exc
    if not isinstance(result, IntegerExternalSortResult):
        raise TypeError("B3 external sort result 类型错误")
    identities = result.identity.input_identities
    if (len(identities) != 1
            or not isinstance(identities[0], IntegerExternalSortInputIdentity)
            or identities[0].relative_path != source_relative
            or identities[0].footer != source.p0_footer
            or identities[0].physical != source.physical):
        _fail(f"{label} external sort input identity 漂移")
    descriptor = ConversationHeldOutV4ProvenanceProjectionStreamDescriptor(
        stream_kind,
        Path(stage) / output_relative,
        result.identity.output_footer,
        result.identity.output_physical,
    )
    _verify_descriptor(
        output_root,
        output_relative,
        descriptor,
        budget=budget,
        label=f"{label} output verification",
    )
    return descriptor, result.temporary_physical_bytes


def _verify_replayed_known_ancestor(
        work_root: KRunRoot,
        claimed: ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        replayed: ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        *,
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        ) -> int:
    """逐条比较 claimed B2 known 与仅由 B1 direct DAG 重放得到的 strict closure。

    P0 descriptor 的 SHA 只能证明调用方提供的字节未在本次读取中变化，不能证明其中 pair
    真从 B1 DAG 推导。故 B3 用其冻结的 replay budget 在独立 namespace 重跑 B2，再在
    `(descendant, ancestor, full snapshot)` 的规范 record 级别完全相等；一条真实 endpoint
    组成但不存在祖先关系的伪造 pair 也会被拒绝。
    """
    claimed_root, claimed_relative = _stage_local_input(
        work_root, claimed, label="B3 claimed known ancestor")
    replayed_root, replayed_relative = _stage_local_input(
        work_root, replayed, label="B3 replayed known ancestor")
    count = 0
    with _VerifiedP0Reader(
            claimed_root,
            claimed_relative,
            footer=claimed.p0_footer,
            physical=claimed.physical,
            budget=budget,
            label="B3 claimed known ancestor input",
    ) as claimed_reader, _VerifiedP0Reader(
            replayed_root,
            replayed_relative,
            footer=replayed.p0_footer,
            physical=replayed.physical,
            budget=budget,
            label="B3 replayed known ancestor input",
    ) as replayed_reader:
        previous_key: tuple[int, ...] | None = None
        while True:
            claimed_record = claimed_reader.read_record()
            replayed_record = replayed_reader.read_record()
            if claimed_record is None or replayed_record is None:
                if claimed_record != replayed_record:
                    _fail("B3 B2 known 与 B1 replay closure record 数不一致")
                break
            try:
                descendant, ancestor, _snapshot = (
                    decode_v4_provenance_closure_pair_record(claimed_record))
                decode_v4_provenance_closure_pair_record(replayed_record)
            except (ConversationHeldOutV4ProvenanceScalableClosureError,
                    ConversationHeldOutV4ProvenanceError, IntegerCodecError,
                    TypeError, ValueError) as exc:
                _fail("B3 claimed/replayed closure pair record 非法")
                raise AssertionError from exc
            pair_key = _packed_sort_key(descendant.components, ancestor.components)
            if previous_key is not None and pair_key <= previous_key:
                _fail("B3 claimed known ancestor key 未严格递增")
            previous_key = pair_key
            if descendant.components == ancestor.components:
                _fail("B3 claimed known ancestor 含 self pair")
            if count >= budget.max_known_pair_count:
                _budget_fail("B3 claimed known ancestor pair 数超过预算")
            if claimed_record != replayed_record:
                _fail("B3 B2 known 与 B1 replay closure 不一致")
            count += 1
        claimed_reader.finish()
        replayed_reader.finish()
    return count


def _materialize_leaf_projection(
        work_root: KRunRoot,
        leaf_endpoint: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        output_root: KRunRoot,
        *,
        output_relative: Path,
        stage: str,
        stream_kind: int,
        record_factory: Callable[[ConversationHeldOutV4ProvenanceLeaf], tuple[int, ...]],
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        label: str,
        ) -> tuple[ConversationHeldOutV4ProvenanceProjectionStreamDescriptor, int]:
    """独立重读 B1 leaf endpoint，物化单一路径但不解释其他 projection。"""
    source_root, source_relative = _stage_local_input(
        work_root, leaf_endpoint, label=f"{label} leaf endpoint")
    count = 0

    def records() -> Iterable[tuple[int, ...]]:
        """在唯一 descriptor reader 生命周期中产出一条 projection 的完整 leaf records。"""
        nonlocal count
        with _VerifiedP0Reader(
                source_root,
                source_relative,
                footer=leaf_endpoint.p0_footer,
                physical=leaf_endpoint.physical,
                budget=budget,
                label=f"{label} leaf endpoint input",
        ) as reader:
            while True:
                raw = reader.read_record()
                if raw is None:
                    break
                try:
                    leaf, snapshot = decode_v4_provenance_leaf_endpoint_record(raw)
                except (ConversationHeldOutV4ProvenanceScalableDagError,
                        ConversationHeldOutV4ProvenanceError, IntegerCodecError,
                        TypeError, ValueError) as exc:
                    _fail(f"{label} leaf endpoint record 非法")
                    raise AssertionError from exc
                if leaf.source_ref.stable_key() != snapshot.source_ref.stable_key():
                    _fail(f"{label} leaf endpoint SourceRef 与 snapshot 不一致")
                if count >= budget.max_leaf_count:
                    _budget_fail(f"{label} leaf 数超过 B3 预算")
                count += 1
                yield record_factory(leaf)
            reader.finish()

    descriptor = _write_stream(
        output_root,
        output_relative,
        stage=stage,
        stream_kind=stream_kind,
        records=records(),
        budget=budget,
        label=f"{label} output",
    )
    return descriptor, count


def _lineage_projection_stream(
        work_root: KRunRoot,
        node_endpoint: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        leaf_endpoint: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        known_ancestor: ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        output_root: KRunRoot,
        *,
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        ) -> tuple[
            ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
            tuple[int, int, int, int],
        ]:
    """流式写出每 leaf self endpoint 和 B2 的全部 strict ancestor lineage records。"""
    node_root, node_relative = _stage_local_input(
        work_root, node_endpoint, label="B3 lineage node endpoint")
    leaf_root, leaf_relative = _stage_local_input(
        work_root, leaf_endpoint, label="B3 lineage leaf endpoint")
    known_root, known_relative = _stage_local_input(
        work_root, known_ancestor, label="B3 lineage known ancestor")
    counts = [0, 0, 0, 0]

    def records() -> Iterable[tuple[int, ...]]:
        """只缓存当前 node 的 leaf/ancestor groups，绝不 materialize 全图或全 leaf 集。"""
        previous_node_key: tuple[int, ...] | None = None
        previous_leaf_node_key: tuple[int, ...] | None = None
        previous_known_key: tuple[int, ...] | None = None
        with _VerifiedP0Reader(
                node_root,
                node_relative,
                footer=node_endpoint.p0_footer,
                physical=node_endpoint.physical,
                budget=budget,
                label="B3 lineage node endpoint input",
        ) as node_reader, _VerifiedP0Reader(
                leaf_root,
                leaf_relative,
                footer=leaf_endpoint.p0_footer,
                physical=leaf_endpoint.physical,
                budget=budget,
                label="B3 lineage leaf endpoint input",
        ) as leaf_reader, _VerifiedP0Reader(
                known_root,
                known_relative,
                footer=known_ancestor.p0_footer,
                physical=known_ancestor.physical,
                budget=budget,
                label="B3 lineage strict ancestor input",
        ) as known_reader:

            def next_leaf() -> tuple[
                    ConversationHeldOutV4ProvenanceLeaf,
                    ConversationHeldOutV4SnapshotIdentity,
                ] | None:
                """读取按 lineage node 非递减排序的 leaf endpoint，并保留当前一条。"""
                nonlocal previous_leaf_node_key
                raw = leaf_reader.read_record()
                if raw is None:
                    return None
                try:
                    leaf, snapshot = decode_v4_provenance_leaf_endpoint_record(raw)
                except (ConversationHeldOutV4ProvenanceScalableDagError,
                        ConversationHeldOutV4ProvenanceError, IntegerCodecError,
                        TypeError, ValueError) as exc:
                    _fail("B3 lineage leaf endpoint record 非法")
                    raise AssertionError from exc
                node_key = leaf.lineage_node_key.components
                if (previous_leaf_node_key is not None
                        and node_key < previous_leaf_node_key):
                    _fail("B3 lineage leaf endpoint node key 倒序")
                previous_leaf_node_key = node_key
                if counts[1] >= budget.max_leaf_count:
                    _budget_fail("B3 lineage leaf 数超过预算")
                counts[1] += 1
                return leaf, snapshot

            def next_known() -> tuple[
                    ProtocolKey,
                    ProtocolKey,
                    ConversationHeldOutV4SnapshotIdentity,
                ] | None:
                """读取严格递增的 B2 `(descendant, ancestor)`，拒绝 self 或重复 pair。"""
                nonlocal previous_known_key
                raw = known_reader.read_record()
                if raw is None:
                    return None
                try:
                    descendant, ancestor, snapshot = (
                        decode_v4_provenance_closure_pair_record(raw))
                except (ConversationHeldOutV4ProvenanceScalableClosureError,
                        ConversationHeldOutV4ProvenanceError, IntegerCodecError,
                        TypeError, ValueError) as exc:
                    _fail("B3 lineage strict ancestor record 非法")
                    raise AssertionError from exc
                key = _packed_sort_key(descendant.components, ancestor.components)
                if previous_known_key is not None and key <= previous_known_key:
                    _fail("B3 lineage strict ancestor key 未严格递增")
                previous_known_key = key
                if descendant.components == ancestor.components:
                    _fail("B3 lineage strict ancestor 含 self pair")
                if counts[2] >= budget.max_known_pair_count:
                    _budget_fail("B3 lineage strict ancestor pair 数超过预算")
                counts[2] += 1
                return descendant, ancestor, snapshot

            def emit(
                    ancestor: ProtocolKey,
                    snapshot: ConversationHeldOutV4SnapshotIdentity,
                    leaf: ConversationHeldOutV4ProvenanceLeaf,
                    ) -> tuple[int, ...]:
                """在 yield 前计入 lineage 乘积预算，并保留完整 namespace/node/snapshot/leaf。"""
                if counts[3] >= budget.max_lineage_projection_record_count:
                    _budget_fail("B3 lineage projection record 数超过预算")
                counts[3] += 1
                return encode_v4_provenance_lineage_projection_record(
                    V4_PROVENANCE_PROJECTION_NAMESPACE_LINEAGE_NODE,
                    ancestor,
                    snapshot,
                    leaf.side,
                    leaf,
                )

            pending_leaf = next_leaf()
            pending_known = next_known()
            while True:
                raw_node = node_reader.read_record()
                if raw_node is None:
                    break
                try:
                    node_key, node_snapshot = decode_v4_provenance_node_endpoint_record(
                        raw_node)
                except (ConversationHeldOutV4ProvenanceScalableDagError,
                        ConversationHeldOutV4ProvenanceError, IntegerCodecError,
                        TypeError, ValueError) as exc:
                    _fail("B3 lineage node endpoint record 非法")
                    raise AssertionError from exc
                if (previous_node_key is not None
                        and node_key.components <= previous_node_key):
                    _fail("B3 lineage node endpoint key 未严格递增")
                previous_node_key = node_key.components
                if counts[0] >= budget.max_node_count:
                    _budget_fail("B3 lineage node 数超过预算")
                counts[0] += 1

                if (pending_leaf is not None
                        and pending_leaf[0].lineage_node_key.components
                        < node_key.components):
                    _fail("B3 lineage leaf 指向未知 node endpoint")
                leaves: list[ConversationHeldOutV4ProvenanceLeaf] = []
                while (pending_leaf is not None
                       and pending_leaf[0].lineage_node_key.components
                       == node_key.components):
                    leaf, leaf_snapshot = pending_leaf
                    if leaf_snapshot != node_snapshot:
                        _fail("B3 lineage leaf endpoint snapshot 与 node endpoint 不一致")
                    if len(leaves) >= budget.max_leaves_per_node:
                        _budget_fail("B3 lineage 同一 node leaf group 超过预算")
                    leaves.append(leaf)
                    pending_leaf = next_leaf()

                if (pending_known is not None
                        and pending_known[0].components < node_key.components):
                    _fail("B3 lineage strict ancestor 指向未知 descendant node")
                ancestors: list[
                    tuple[ProtocolKey, ConversationHeldOutV4SnapshotIdentity]
                ] = []
                while (pending_known is not None
                       and pending_known[0].components == node_key.components):
                    _descendant, ancestor, ancestor_snapshot = pending_known
                    if len(ancestors) >= budget.max_ancestors_per_node:
                        _budget_fail("B3 lineage 同一 descendant ancestor group 超过预算")
                    ancestors.append((ancestor, ancestor_snapshot))
                    pending_known = next_known()

                for leaf in leaves:
                    yield emit(node_key, node_snapshot, leaf)
                for ancestor, ancestor_snapshot in ancestors:
                    for leaf in leaves:
                        yield emit(ancestor, ancestor_snapshot, leaf)

            if pending_leaf is not None:
                _fail("B3 lineage leaf 指向未知 node endpoint")
            if pending_known is not None:
                _fail("B3 lineage strict ancestor 指向未知 descendant node")
            node_reader.finish()
            leaf_reader.finish()
            known_reader.finish()

    descriptor = _write_stream(
        output_root,
        _LINEAGE_RAW_FILE,
        stage=_STAGE_LINEAGE_MATERIALIZE,
        stream_kind=V4_PROVENANCE_PROJECTION_STREAM_LINEAGE,
        records=records(),
        budget=budget,
        label="B3 lineage raw output",
    )
    return descriptor, (counts[0], counts[1], counts[2], counts[3])


def build_v4_provenance_leaf_projections(
        direct_dag_result: ConversationHeldOutV4ProvenanceDirectDagResult,
        closure_result: ConversationHeldOutV4ProvenanceClosureResult,
        work_run_root: KRunRoot,
        *,
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        logical_stage_name: str,
        ) -> ConversationHeldOutV4ProvenanceProjectionResult:
    """执行 P2-B3 的三路 leaf projection，不进入任何 cross-side 判定或发布。

    ``work_run_root`` 必须是 B1 原始 ``create_new_run_root`` capability；B3 允许其中已有
    B1/B2 stage，但独立 closure replay 与六个 projection stage 都必须此前不存在。B3 在创建
    任何新 stage 前重验 B1/B2 全部 descriptor；每次实际读 leaf/node/known 时又单独复核同一
    P0 identity。
    """
    if not isinstance(direct_dag_result,
                      ConversationHeldOutV4ProvenanceDirectDagResult):
        raise TypeError("B3 direct_dag_result 类型错误")
    if not isinstance(closure_result,
                      ConversationHeldOutV4ProvenanceClosureResult):
        raise TypeError("B3 closure_result 类型错误")
    if not isinstance(work_run_root, KRunRoot):
        raise TypeError("B3 work_run_root 必须是 KRunRoot")
    if not isinstance(budget, ConversationHeldOutV4ProvenanceProjectionBudget):
        raise TypeError("B3 budget 类型错误")
    _require_stage_name(logical_stage_name)
    require_created_run_root(work_run_root, label="B3 work_run_root")
    if closure_result.direct_dag_stable_key != direct_dag_result.stable_key():
        _fail("B3 B2 direct_dag_stable_key 与 B1 result 不一致")
    if closure_result.node_count != direct_dag_result.node_count:
        _fail("B3 B2/B1 node_count 不一致")
    if closure_result.direct_edge_count != direct_dag_result.direct_edge_count:
        _fail("B3 B2/B1 direct_edge_count 不一致")
    if (closure_result.edge_by_child_stream.p0_footer.record_count
            != closure_result.direct_edge_count):
        _fail("B3 B2 edge-by-child footer 与 direct_edge_count 不一致")
    _verify_b1_result(work_run_root, direct_dag_result, budget=budget)
    _verify_b2_result(work_run_root, closure_result, budget=budget)
    if direct_dag_result.node_count > budget.max_node_count:
        _budget_fail("B3 B1 node_count 超过预算")
    if direct_dag_result.leaf_count > budget.max_leaf_count:
        _budget_fail("B3 B1 leaf_count 超过预算")
    if closure_result.known_pair_count > budget.max_known_pair_count:
        _budget_fail("B3 B2 known_pair_count 超过预算")

    materialized = 0
    sort_reservation = (
        budget.external_sort_budget.max_temporary_physical_bytes
        + budget.external_sort_budget.max_output_physical_bytes)

    # descriptor SHA 只能绑定本次读取的字节，不能证明 calling code 提供的 B2 pair 真由
    # B1 DAG 推导。用 B3 自己冻结的预算在独立 stage namespace 重放 B2，并逐 record 比对
    # claimed known；这比仅检查 pair 两端 endpoint 都存在更强，能拒绝真实 endpoint 组成的
    # 虚假祖先关系。
    _reserve_materialized(
        materialized,
        budget.closure_replay_budget.max_total_materialized_physical_bytes,
        budget=budget,
        label="B3 independent closure replay",
    )
    replayed_closure = build_v4_provenance_transitive_closure(
        direct_dag_result,
        work_run_root,
        budget=budget.closure_replay_budget,
        logical_stage_name="b3-closure-replay.v1",
        stage_namespace="b3replay",
    )
    materialized = _add_materialized(
        materialized,
        replayed_closure.materialized_physical_bytes,
        budget=budget,
        label="B3 independent closure replay",
    )
    if replayed_closure.node_count != direct_dag_result.node_count:
        _fail("B3 replay closure node count 与 B1 result 不一致")
    if replayed_closure.direct_edge_count != direct_dag_result.direct_edge_count:
        _fail("B3 replay closure direct edge count 与 B1 result 不一致")
    replayed_known_pair_count = _verify_replayed_known_ancestor(
        work_run_root,
        closure_result.known_ancestor_stream,
        replayed_closure.known_ancestor_stream,
        budget=budget,
    )
    if replayed_known_pair_count != closure_result.known_pair_count:
        _fail("B3 claimed known ancestor count 与 B2 result 不一致")

    _reserve_materialized(
        materialized,
        budget.max_stream_physical_bytes,
        budget=budget,
        label="B3 SourceRef raw",
    )
    source_ref_raw_root = _new_stage(work_run_root, _STAGE_SOURCE_REF_MATERIALIZE)
    source_ref_raw, source_ref_count = _materialize_leaf_projection(
        work_run_root,
        direct_dag_result.leaf_endpoint_stream,
        source_ref_raw_root,
        output_relative=_SOURCE_REF_RAW_FILE,
        stage=_STAGE_SOURCE_REF_MATERIALIZE,
        stream_kind=V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF,
        record_factory=lambda leaf: encode_v4_provenance_source_ref_projection_record(
            leaf.source_ref, leaf.side, leaf),
        budget=budget,
        label="B3 SourceRef raw",
    )
    materialized = _add_materialized(
        materialized,
        source_ref_raw.physical.byte_count,
        budget=budget,
        label="B3 SourceRef raw",
    )
    if source_ref_count != direct_dag_result.leaf_count:
        _fail("B3 SourceRef leaf count 与 B1 result 不一致")

    _reserve_materialized(
        materialized,
        sort_reservation,
        budget=budget,
        label="B3 SourceRef sort",
    )
    source_ref_sort_root = _new_stage(work_run_root, _STAGE_SOURCE_REF_SORT)
    source_ref_projection, temporary_bytes = _sort_one_stream(
        work_run_root,
        source_ref_raw,
        source_ref_sort_root,
        stage=_STAGE_SOURCE_REF_SORT,
        output_relative=_SOURCE_REF_SORTED_FILE,
        stream_kind=V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF,
        logical_stage_name=f"{logical_stage_name}.source-ref.v1",
        sort_key=_source_ref_projection_sort_key,
        budget=budget,
        label="B3 SourceRef sort",
    )
    materialized = _add_materialized(
        materialized,
        temporary_bytes + source_ref_projection.physical.byte_count,
        budget=budget,
        label="B3 SourceRef sort",
    )
    if source_ref_projection.p0_footer.record_count != source_ref_count:
        _fail("B3 SourceRef sorted footer 与 raw count 不一致")

    _reserve_materialized(
        materialized,
        budget.max_stream_physical_bytes,
        budget=budget,
        label="B3 content raw",
    )
    content_raw_root = _new_stage(work_run_root, _STAGE_CONTENT_MATERIALIZE)
    content_raw, content_count = _materialize_leaf_projection(
        work_run_root,
        direct_dag_result.leaf_endpoint_stream,
        content_raw_root,
        output_relative=_CONTENT_RAW_FILE,
        stage=_STAGE_CONTENT_MATERIALIZE,
        stream_kind=V4_PROVENANCE_PROJECTION_STREAM_CONTENT,
        record_factory=lambda leaf: encode_v4_provenance_content_projection_record(
            leaf.content_sha256, leaf.side, leaf),
        budget=budget,
        label="B3 content raw",
    )
    materialized = _add_materialized(
        materialized,
        content_raw.physical.byte_count,
        budget=budget,
        label="B3 content raw",
    )
    if content_count != direct_dag_result.leaf_count:
        _fail("B3 content leaf count 与 B1 result 不一致")

    _reserve_materialized(
        materialized,
        sort_reservation,
        budget=budget,
        label="B3 content sort",
    )
    content_sort_root = _new_stage(work_run_root, _STAGE_CONTENT_SORT)
    content_projection, temporary_bytes = _sort_one_stream(
        work_run_root,
        content_raw,
        content_sort_root,
        stage=_STAGE_CONTENT_SORT,
        output_relative=_CONTENT_SORTED_FILE,
        stream_kind=V4_PROVENANCE_PROJECTION_STREAM_CONTENT,
        logical_stage_name=f"{logical_stage_name}.content.v1",
        sort_key=_content_projection_sort_key,
        budget=budget,
        label="B3 content sort",
    )
    materialized = _add_materialized(
        materialized,
        temporary_bytes + content_projection.physical.byte_count,
        budget=budget,
        label="B3 content sort",
    )
    if content_projection.p0_footer.record_count != content_count:
        _fail("B3 content sorted footer 与 raw count 不一致")

    _reserve_materialized(
        materialized,
        budget.max_stream_physical_bytes,
        budget=budget,
        label="B3 lineage raw",
    )
    lineage_raw_root = _new_stage(work_run_root, _STAGE_LINEAGE_MATERIALIZE)
    lineage_raw, lineage_input_counts = _lineage_projection_stream(
        work_run_root,
        direct_dag_result.node_endpoint_stream,
        direct_dag_result.leaf_endpoint_stream,
        closure_result.known_ancestor_stream,
        lineage_raw_root,
        budget=budget,
    )
    materialized = _add_materialized(
        materialized,
        lineage_raw.physical.byte_count,
        budget=budget,
        label="B3 lineage raw",
    )
    node_count, lineage_leaf_count, known_pair_count, lineage_count = (
        lineage_input_counts)
    if node_count != direct_dag_result.node_count:
        _fail("B3 lineage node count 与 B1 result 不一致")
    if lineage_leaf_count != direct_dag_result.leaf_count:
        _fail("B3 lineage leaf count 与 B1 result 不一致")
    if known_pair_count != closure_result.known_pair_count:
        _fail("B3 lineage known pair count 与 B2 result 不一致")
    if lineage_raw.p0_footer.record_count != lineage_count:
        _fail("B3 lineage raw footer 与 emitted count 不一致")

    _reserve_materialized(
        materialized,
        sort_reservation,
        budget=budget,
        label="B3 lineage sort",
    )
    lineage_sort_root = _new_stage(work_run_root, _STAGE_LINEAGE_SORT)
    lineage_projection, temporary_bytes = _sort_one_stream(
        work_run_root,
        lineage_raw,
        lineage_sort_root,
        stage=_STAGE_LINEAGE_SORT,
        output_relative=_LINEAGE_SORTED_FILE,
        stream_kind=V4_PROVENANCE_PROJECTION_STREAM_LINEAGE,
        logical_stage_name=f"{logical_stage_name}.lineage.v1",
        sort_key=_lineage_projection_sort_key,
        budget=budget,
        label="B3 lineage sort",
    )
    materialized = _add_materialized(
        materialized,
        temporary_bytes + lineage_projection.physical.byte_count,
        budget=budget,
        label="B3 lineage sort",
    )
    if lineage_projection.p0_footer.record_count != lineage_count:
        _fail("B3 lineage sorted footer 与 raw count 不一致")

    return ConversationHeldOutV4ProvenanceProjectionResult(
        logical_stage_name,
        direct_dag_result.stable_key(),
        closure_result.stable_key(),
        source_ref_projection,
        content_projection,
        lineage_projection,
        source_ref_count,
        content_count,
        lineage_count,
        materialized,
        budget,
    )


def _revalidate_projection_stream(
        work_root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        *,
        expected_kind: int,
        expected_side: int,
        expected_count: int,
        budget: ConversationHeldOutV4ProvenanceProjectionBudget,
        decode_record: Callable[[tuple[int, ...]], tuple[object, ...]],
        sort_key: Callable[[tuple[int, ...]], tuple[int, ...]],
        side_index: int,
        label: str,
        ) -> int:
    """完整只读回放一条已封存 B3 stream，并重验其 projection 语义。"""
    if descriptor.stream_kind != expected_kind:
        _fail(f"{label} stream_kind 与 result 槽位不一致")
    source_root, source_relative = _stage_local_input(
        work_root, descriptor, label=label)
    count = 0
    previous_key: tuple[int, ...] | None = None
    with _VerifiedP0Reader(
            source_root,
            source_relative,
            footer=descriptor.p0_footer,
            physical=descriptor.physical,
            budget=budget,
            label=f"{label} revalidation",
    ) as reader:
        while True:
            record = reader.read_record()
            if record is None:
                break
            decoded = decode_record(record)
            if decoded[side_index] != expected_side:
                _fail(f"{label} record side 与 expected_side 不一致")
            current_key = sort_key(record)
            if previous_key is not None and current_key <= previous_key:
                _fail(f"{label} record sort key 未严格递增")
            previous_key = current_key
            if count >= budget.max_stream_record_count:
                _budget_fail(f"{label} record 数超过 B3 stream 预算")
            count += 1
        footer = reader.finish()
    if footer.record_count != count or count != expected_count:
        _fail(f"{label} record count 与 result 或 footer 不一致")
    return count


def revalidate_v4_provenance_leaf_projections(
        result: ConversationHeldOutV4ProvenanceProjectionResult,
        work_root: KRunRoot,
        *,
        expected_side: int,
        ) -> ConversationHeldOutV4ProvenanceProjectionResult:
    """只读复验 P2-B3 的三路 sealed projection，并返回同一冻结 result。

    此入口不重建投影、不创建 temporary storage，也不读取 B1/B2 或其他域。它只接受原始
    work capability 可是原始 create-new 或后续以 ``open_existing_run_root`` 打开的只读
    capability；逐条回放 SourceRef、content 和 lineage 三个已排序 P0 descriptor，复核
    实际 footer、物理 SHA-256、路径身份、record side、排序和 result count。
    成功仅说明这份 B3 结果在指定 side 下仍可机械读取；不产生训练或资格结论。
    """
    if not isinstance(result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("B3 revalidation result 类型错误")
    if not isinstance(work_root, KRunRoot):
        raise TypeError("B3 revalidation work_root 必须是 KRunRoot")
    if type(expected_side) is not int or expected_side not in {
            V4_PROVENANCE_SIDE_TRAINING, V4_PROVENANCE_SIDE_HELD_OUT}:
        raise ValueError("B3 revalidation expected_side 未注册")

    stable_key = result.stable_key()
    try:
        strict_integer_tuple(stable_key, label="B3 revalidation result stable_key")
    except (TypeError, ValueError) as exc:
        _fail("B3 revalidation result stable_key 非法")
        raise AssertionError from exc

    source_count = _revalidate_projection_stream(
        work_root,
        result.source_ref_projection_stream,
        expected_kind=V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF,
        expected_side=expected_side,
        expected_count=result.source_ref_projection_count,
        budget=result.budget,
        decode_record=decode_v4_provenance_source_ref_projection_record,
        sort_key=_source_ref_projection_sort_key,
        side_index=1,
        label="B3 SourceRef projection",
    )
    content_count = _revalidate_projection_stream(
        work_root,
        result.content_projection_stream,
        expected_kind=V4_PROVENANCE_PROJECTION_STREAM_CONTENT,
        expected_side=expected_side,
        expected_count=result.content_projection_count,
        budget=result.budget,
        decode_record=decode_v4_provenance_content_projection_record,
        sort_key=_content_projection_sort_key,
        side_index=1,
        label="B3 content projection",
    )
    lineage_count = _revalidate_projection_stream(
        work_root,
        result.lineage_projection_stream,
        expected_kind=V4_PROVENANCE_PROJECTION_STREAM_LINEAGE,
        expected_side=expected_side,
        expected_count=result.lineage_projection_count,
        budget=result.budget,
        decode_record=decode_v4_provenance_lineage_projection_record,
        sort_key=_lineage_projection_sort_key,
        side_index=3,
        label="B3 lineage projection",
    )
    if source_count != content_count:
        _fail("B3 revalidation SourceRef/content count 不一致")
    if lineage_count < source_count:
        _fail("B3 revalidation lineage count 不足 self endpoint")
    return result


__all__ = [
    "ConversationHeldOutV4ProvenanceProjectionBudget",
    "ConversationHeldOutV4ProvenanceProjectionResult",
    "ConversationHeldOutV4ProvenanceProjectionStreamDescriptor",
    "ConversationHeldOutV4ProvenanceScalableProjectionBudgetExceeded",
    "ConversationHeldOutV4ProvenanceScalableProjectionError",
    "V4_PROVENANCE_PROJECTION_BUDGET_SCHEMA",
    "V4_PROVENANCE_PROJECTION_CONTENT_RECORD_SCHEMA",
    "V4_PROVENANCE_PROJECTION_LINEAGE_RECORD_SCHEMA",
    "V4_PROVENANCE_PROJECTION_NAMESPACE_LINEAGE_NODE",
    "V4_PROVENANCE_PROJECTION_RESULT_SCHEMA",
    "V4_PROVENANCE_PROJECTION_SOURCE_REF_RECORD_SCHEMA",
    "V4_PROVENANCE_PROJECTION_STREAM_CONTENT",
    "V4_PROVENANCE_PROJECTION_STREAM_DESCRIPTOR_SCHEMA",
    "V4_PROVENANCE_PROJECTION_STREAM_LINEAGE",
    "V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF",
    "build_v4_provenance_leaf_projections",
    "decode_v4_provenance_content_projection_record",
    "decode_v4_provenance_lineage_projection_record",
    "decode_v4_provenance_source_ref_projection_record",
    "encode_v4_provenance_content_projection_record",
    "encode_v4_provenance_lineage_projection_record",
    "encode_v4_provenance_source_ref_projection_record",
    "revalidate_v4_provenance_leaf_projections",
]
