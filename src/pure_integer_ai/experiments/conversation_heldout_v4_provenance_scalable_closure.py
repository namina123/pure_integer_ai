"""DLG-05 v4 R04b P2-B2 的有界外部传递祖先闭包。

本模块只消费已完成的 P2-B1 direct DAG result，并在同一个、由排他创建 capability
持有的 K run 内新建 stage sibling。它只产生严格祖先 pair 的未发布 P0 stream；self
endpoint 仍由 B3 从 node/leaf endpoint 显式加入，不能把 self pair 当作传递闭包，也不能
用 direct edge 冒充任意祖先。

所有读写使用 K boundary 已核验 handle 与 P0 framed stream。任何输入漂移、未知格式、
环、重复 pair 的 snapshot 冲突或资源预算耗尽均 fail closed，不执行删除、覆盖、resume、
receipt、manifest 或跨 side 判定。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import BinaryIO, Iterable

from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4ProvenanceError,
    ConversationHeldOutV4SnapshotIdentity,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_dag import (
    ConversationHeldOutV4ProvenanceDagStreamDescriptor,
    ConversationHeldOutV4ProvenanceDirectDagResult,
    decode_v4_provenance_direct_edge_record,
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
    KRunBoundaryError,
    KRunFileDigest,
    KRunFileIdentity,
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


V4_PROVENANCE_CLOSURE_BUDGET_SCHEMA = 1
V4_PROVENANCE_CLOSURE_STREAM_DESCRIPTOR_SCHEMA = 1
V4_PROVENANCE_CLOSURE_RESULT_SCHEMA = 1
V4_PROVENANCE_CLOSURE_PAIR_RECORD_SCHEMA = 1

V4_PROVENANCE_CLOSURE_STREAM_INITIAL_PAIR = 1
V4_PROVENANCE_CLOSURE_STREAM_INITIAL_PAIR_SORTED = 2
V4_PROVENANCE_CLOSURE_STREAM_EDGE_BY_CHILD = 3
V4_PROVENANCE_CLOSURE_STREAM_FRONTIER_BY_ANCESTOR = 4
V4_PROVENANCE_CLOSURE_STREAM_CANDIDATE = 5
V4_PROVENANCE_CLOSURE_STREAM_CANDIDATE_SORTED = 6
V4_PROVENANCE_CLOSURE_STREAM_KNOWN = 7
V4_PROVENANCE_CLOSURE_STREAM_FRONTIER = 8

_STREAM_KINDS = frozenset({
    V4_PROVENANCE_CLOSURE_STREAM_INITIAL_PAIR,
    V4_PROVENANCE_CLOSURE_STREAM_INITIAL_PAIR_SORTED,
    V4_PROVENANCE_CLOSURE_STREAM_EDGE_BY_CHILD,
    V4_PROVENANCE_CLOSURE_STREAM_FRONTIER_BY_ANCESTOR,
    V4_PROVENANCE_CLOSURE_STREAM_CANDIDATE,
    V4_PROVENANCE_CLOSURE_STREAM_CANDIDATE_SORTED,
    V4_PROVENANCE_CLOSURE_STREAM_KNOWN,
    V4_PROVENANCE_CLOSURE_STREAM_FRONTIER,
})

_STAGE_INITIAL_PAIR = "stage-06-closure-initial-pair"
_STAGE_INITIAL_SORT = "stage-07-closure-initial-sort"
_STAGE_INITIAL_UNION = "stage-08-closure-initial-union"
_STAGE_EDGE_BY_CHILD = "stage-09-closure-edge-by-child"
_INITIAL_PAIR_FILE = Path("initial-pairs.pifrs")
_INITIAL_PAIR_SORTED_FILE = Path("initial-pairs-sorted.pifrs")
_EDGE_BY_CHILD_FILE = Path("direct-edges-by-child.pifrs")
_KNOWN_FILE = Path("known.pifrs")
_FRONTIER_FILE = Path("frontier.pifrs")
_FRONTIER_BY_ANCESTOR_FILE = Path("frontier-by-ancestor.pifrs")
_CANDIDATE_FILE = Path("candidates.pifrs")
_CANDIDATE_SORTED_FILE = Path("candidates-sorted.pifrs")


# object-model: exception
class ConversationHeldOutV4ProvenanceScalableClosureError(RuntimeError):
    """B2 的上游 identity、DAG closure 或有界物化合同不成立。"""


# object-model: exception
class ConversationHeldOutV4ProvenanceScalableClosureBudgetExceeded(
        ConversationHeldOutV4ProvenanceScalableClosureError):
    """B2 的 pair、轮数、I/O 或物化资源超过冻结上限。"""


def _fail(message: str) -> None:
    """统一产生 B2 fail-closed 错误。"""
    raise ConversationHeldOutV4ProvenanceScalableClosureError(message)


def _budget_fail(message: str) -> None:
    """区分资源耗尽与上游本体/identity 不一致。"""
    raise ConversationHeldOutV4ProvenanceScalableClosureBudgetExceeded(message)


def _require_positive_int(value: int, *, label: str) -> int:
    """拒绝 bool、零和隐式数值，固定上限/格式字段。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def _require_nonnegative_int(value: int, *, label: str) -> int:
    """固定结果计数和物化统计为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _require_stage_name(value: str) -> str:
    """限制 stage 标识为无路径语义的稳定 ASCII 名称。"""
    if not isinstance(value, str) or not value or len(value) > 72:
        raise ValueError("B2 stage 名必须是长度受限的非空 str")
    if (not value[0].isascii() or not value[0].isalnum()
            or any(not item.isascii() or not (
                item.isalnum() or item in {"-", "_", "."})
                   for item in value)):
        raise ValueError("B2 stage 名只能含 ASCII 字母、数字、-、_、.")
    return value


def _require_stage_namespace(value: str | None) -> str:
    """校验可选 B2 replay namespace，防止独立审计 stage 与原 closure stage 冲突。"""
    if value is None:
        return ""
    _require_stage_name(value)
    if len(value) > 18:
        raise ValueError("B2 stage_namespace 超过长度上限")
    return value


def _stage_name(stage_namespace: str, stage: str) -> str:
    """以短 namespace 派生此前不存在的 B2 sibling stage 名，而不改变默认命名。"""
    _require_stage_name(stage)
    if not stage_namespace:
        return stage
    result = f"{stage_namespace}-{stage}"
    _require_stage_name(result)
    return result


def _relative_path_key(value: Path, *, label: str) -> tuple[int, ...]:
    """把固定的 work 相对路径映射为不含本机 root 的 ASCII scalar key。"""
    if (not isinstance(value, Path) or value.is_absolute() or value.drive
            or value.root or not value.parts or ".." in value.parts
            or any(part in {"", ".", ".."} for part in value.parts)):
        raise ValueError(f"{label} 必须是非空且不含 .. 的相对 Path")
    text = value.as_posix()
    if any(not item.isascii() for item in text):
        raise ValueError(f"{label} 必须只含 ASCII")
    return tuple(ord(item) for item in text)


def _digest_key(value: KRunFileDigest) -> tuple[int, ...]:
    """完整展开 physical size/SHA；摘要不能替代 P0 footer。"""
    if not isinstance(value, KRunFileDigest):
        raise TypeError("B2 physical 必须是 KRunFileDigest")
    return (value.byte_count, *value.sha256)


def _packed_sort_key(*parts: tuple[int, ...]) -> tuple[int, ...]:
    """以长度前缀拼接可变长度整数 key，避免裸连接的排序歧义。"""
    result: list[int] = []
    for part in parts:
        strict_integer_tuple(part, label="B2 sort key part")
        pack_key(result, part)
    return tuple(result)


def _external_sort_budget_key(value: IntegerExternalSortBudget) -> tuple[int, ...]:
    """要求 B2 identity 完整绑定通用外排配置。"""
    if not isinstance(value, IntegerExternalSortBudget):
        raise TypeError("B2 external_sort_budget 类型错误")
    return value.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceClosureBudget:
    """B2 传递闭包、P0 stage 与外排的全部显式资源预算。"""

    max_node_count: int
    max_direct_edge_count: int
    max_known_pair_count: int
    max_candidate_pair_count_per_round: int
    max_closure_round_count: int
    max_parents_per_node: int
    max_record_payload_bytes: int
    max_stream_record_count: int
    max_stream_payload_bytes: int
    max_stream_physical_bytes: int
    max_total_materialized_physical_bytes: int
    max_open_files: int
    external_sort_budget: IntegerExternalSortBudget

    def __post_init__(self) -> None:
        """拒绝任何无上限的图展开、流、round 或外排配置。"""
        for label, value in (
                ("max_node_count", self.max_node_count),
                ("max_direct_edge_count", self.max_direct_edge_count),
                ("max_known_pair_count", self.max_known_pair_count),
                ("max_candidate_pair_count_per_round",
                 self.max_candidate_pair_count_per_round),
                ("max_closure_round_count", self.max_closure_round_count),
                ("max_parents_per_node", self.max_parents_per_node),
                ("max_record_payload_bytes", self.max_record_payload_bytes),
                ("max_stream_record_count", self.max_stream_record_count),
                ("max_stream_payload_bytes", self.max_stream_payload_bytes),
                ("max_stream_physical_bytes", self.max_stream_physical_bytes),
                ("max_total_materialized_physical_bytes",
                 self.max_total_materialized_physical_bytes),
                ("max_open_files", self.max_open_files)):
            _require_positive_int(value, label=label)
        if not isinstance(self.external_sort_budget, IntegerExternalSortBudget):
            raise TypeError("B2 external_sort_budget 类型错误")
        if self.max_open_files < 3:
            raise ValueError("B2 max_open_files 至少覆盖两个 reader 和一个 writer")
        if self.external_sort_budget.max_open_files > self.max_open_files:
            raise ValueError("B2 external sort 打开句柄预算不得超过 B2 上限")
        if (self.external_sort_budget.max_output_physical_bytes
                > self.max_stream_physical_bytes):
            raise ValueError("B2 external sort 输出不得超过 stream 物理预算")
        if self.max_record_payload_bytes > self.max_stream_payload_bytes:
            raise ValueError("B2 单 record payload 不得超过 stream payload 预算")
        if any(value > self.max_stream_record_count for value in (
                self.max_node_count,
                self.max_direct_edge_count,
                self.max_known_pair_count,
                self.max_candidate_pair_count_per_round)):
            raise ValueError("B2 单类 record 预算不得超过单 stream record 预算")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 K root 的完整 B2 资源身份。"""
        result = [V4_PROVENANCE_CLOSURE_BUDGET_SCHEMA]
        result.extend((
            self.max_node_count,
            self.max_direct_edge_count,
            self.max_known_pair_count,
            self.max_candidate_pair_count_per_round,
            self.max_closure_round_count,
            self.max_parents_per_node,
            self.max_record_payload_bytes,
            self.max_stream_record_count,
            self.max_stream_payload_bytes,
            self.max_stream_physical_bytes,
            self.max_total_materialized_physical_bytes,
            self.max_open_files,
        ))
        pack_key(result, _external_sort_budget_key(self.external_sort_budget))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 schema-backed 的完整可审计 B2 budget key。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceClosureStreamDescriptor:
    """B2 一个已 seal、未发布 P0 输出的相对定位和完整物理身份。"""

    stream_kind: int
    work_relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """固定类型、相对路径、footer 和 physical SHA，不泄露 K root。"""
        if type(self.stream_kind) is not int or self.stream_kind not in _STREAM_KINDS:
            raise ValueError("B2 stream_kind 未注册")
        _relative_path_key(self.work_relative_path, label="B2 work_relative_path")
        if not isinstance(self.p0_footer, IntegerFramedStreamFooter):
            raise TypeError("B2 p0_footer 类型错误")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("B2 physical 类型错误")
        if self.physical.byte_count <= 0:
            raise ValueError("B2 sealed P0 stream 物理字节必须为正")

    def integer_stream(self) -> tuple[int, ...]:
        """返回 descriptor 的完整无 root 规范整数表示。"""
        result = [V4_PROVENANCE_CLOSURE_STREAM_DESCRIPTOR_SCHEMA, self.stream_kind]
        for value in (
                _relative_path_key(self.work_relative_path,
                                   label="B2 work_relative_path"),
                self.p0_footer.integer_tuple(),
                _digest_key(self.physical)):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回可被 B3/P2-C 绑定的完整 descriptor key。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceClosureResult:
    """B2 闭合结果；只说明严格祖先机械闭合，绝不声明训练资格。"""

    logical_stage_name: str
    direct_dag_stable_key: tuple[int, ...]
    edge_by_child_stream: ConversationHeldOutV4ProvenanceClosureStreamDescriptor
    known_ancestor_stream: ConversationHeldOutV4ProvenanceClosureStreamDescriptor
    empty_frontier_stream: ConversationHeldOutV4ProvenanceClosureStreamDescriptor
    node_count: int
    direct_edge_count: int
    known_pair_count: int
    closure_round_count: int
    materialized_physical_bytes: int
    budget: ConversationHeldOutV4ProvenanceClosureBudget

    def __post_init__(self) -> None:
        """绑定最终严格祖先、空 frontier、B1 identity 与全部冻结统计。"""
        _require_stage_name(self.logical_stage_name)
        strict_integer_tuple(self.direct_dag_stable_key,
                             label="B2 direct_dag_stable_key")
        for descriptor, kind in (
                (self.edge_by_child_stream,
                 V4_PROVENANCE_CLOSURE_STREAM_EDGE_BY_CHILD),
                (self.known_ancestor_stream,
                 V4_PROVENANCE_CLOSURE_STREAM_KNOWN),
                (self.empty_frontier_stream,
                 V4_PROVENANCE_CLOSURE_STREAM_FRONTIER)):
            if not isinstance(descriptor,
                              ConversationHeldOutV4ProvenanceClosureStreamDescriptor):
                raise TypeError("B2 result 含非 closure stream descriptor")
            if descriptor.stream_kind != kind:
                raise ValueError("B2 result stream descriptor 类型错位")
        for label, value in (
                ("node_count", self.node_count),
                ("direct_edge_count", self.direct_edge_count),
                ("known_pair_count", self.known_pair_count),
                ("closure_round_count", self.closure_round_count),
                ("materialized_physical_bytes",
                 self.materialized_physical_bytes)):
            _require_nonnegative_int(value, label=f"B2 {label}")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceClosureBudget):
            raise TypeError("B2 result budget 类型错误")
        if self.node_count > self.budget.max_node_count:
            raise ValueError("B2 node_count 超过冻结预算")
        if self.direct_edge_count > self.budget.max_direct_edge_count:
            raise ValueError("B2 direct_edge_count 超过冻结预算")
        if self.known_pair_count != self.known_ancestor_stream.p0_footer.record_count:
            raise ValueError("B2 known_pair_count 与 footer 不一致")
        if self.empty_frontier_stream.p0_footer.record_count != 0:
            raise ValueError("B2 只有 empty frontier 才能返回 closure result")
        if self.closure_round_count > self.budget.max_closure_round_count:
            raise ValueError("B2 closure_round_count 超过冻结预算")
        if self.materialized_physical_bytes > self.budget.max_total_materialized_physical_bytes:
            raise ValueError("B2 materialized physical bytes 超过冻结预算")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 K root 的 B2 closure result 身份。"""
        result = [V4_PROVENANCE_CLOSURE_RESULT_SCHEMA]
        pack_key(result, tuple(ord(item) for item in self.logical_stage_name))
        pack_key(result, self.direct_dag_stable_key)
        for descriptor in (
                self.edge_by_child_stream,
                self.known_ancestor_stream,
                self.empty_frontier_stream):
            pack_key(result, descriptor.integer_stream())
        result.extend((
            self.node_count,
            self.direct_edge_count,
            self.known_pair_count,
            self.closure_round_count,
            self.materialized_physical_bytes,
        ))
        pack_key(result, self.budget.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 B3/P2-C 可绑定的完全 schema-backed closure identity。"""
        return self.integer_stream()


# object-model: resource_owner; representation=protocol; interop=pending
class _DigestingBinaryHandle:
    """在 P0 同一 handle 读写期间统计 physical SHA，不缓存整个文件。"""

    __slots__ = ("_byte_count", "_digest", "_max_bytes", "_stream")

    def __init__(self, stream: BinaryIO, *, max_bytes: int) -> None:
        """接管已由 K boundary 打开的 handle，并固定物理字节预算。"""
        self._byte_count = 0
        self._digest = hashlib.sha256()
        self._max_bytes = _require_positive_int(
            max_bytes, label="B2 physical max_bytes")
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        """按底层成功读取的实际字节累计 SHA，并在 P0 前拦截超限。"""
        if type(size) is not int:
            raise TypeError("B2 physical read size 必须是严格整数")
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
        """只累计底层确认写出的前缀，部分写入由 P0 writer 处理为失败。"""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("B2 physical write data 必须是 bytes-like")
        if len(data) > self._max_bytes - self._byte_count:
            _budget_fail("B2 P0 stream physical bytes 超过预算")
        written = self._stream.write(data)
        if type(written) is int and 0 < written <= len(data):
            self._accept(bytes(memoryview(data)[:written]))
        return written

    def flush(self) -> None:
        """透传 P0 seal 所需 flush，不把它伪造为物理写入。"""
        self._stream.flush()

    def close(self) -> None:
        """关闭唯一持有的底层 handle。"""
        self._stream.close()

    def physical(self) -> KRunFileDigest:
        """返回同一 handle 已成功处理字节的完整物理身份。"""
        return KRunFileDigest(self._byte_count, tuple(self._digest.digest()))

    def _accept(self, value: bytes) -> None:
        """在更新 SHA 前拒绝超过物理预算的实际 I/O。"""
        if len(value) > self._max_bytes - self._byte_count:
            _budget_fail("B2 P0 stream physical bytes 超过预算")
        self._digest.update(value)
        self._byte_count += len(value)


# object-model: resource_owner; representation=protocol; interop=pending
class _VerifiedP0Reader:
    """完整消费一个冻结 P0 descriptor，并在 finish 时复核所有物理身份。"""

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
            budget: ConversationHeldOutV4ProvenanceClosureBudget,
            label: str,
            ) -> None:
        """以 capture/open/post-read 三段边界固定一个 descriptor 的实际消费。"""
        if physical.byte_count > budget.max_stream_physical_bytes:
            _budget_fail(f"{label} physical bytes 超过 B2 stream 预算")
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
        """返回逐条 P0 cursor，调用方必须在正常路径调用 finish。"""
        return self

    def __exit__(self, exc_type: object, exc_value: object,
                 traceback: object) -> None:
        """异常路径只关闭未可信的剩余输入。"""
        self.close()

    def read_record(self) -> tuple[int, ...] | None:
        """读取当前 P0 record；底层 codec 失败即关闭 handle。"""
        return self._reader.read_record()

    def finish(self) -> IntegerFramedStreamFooter:
        """完全消费并比较 footer、handle SHA/size 与读后路径身份。"""
        if self._finished:
            _fail("B2 verified P0 reader 不得重复 finish")
        footer = self._reader.finish()
        physical = self._handle.physical()
        require_plain_file_identity(
            self._root,
            self._relative,
            self._file_identity,
            label="B2 P0 post-read",
        )
        if footer != self._expected_footer:
            _fail("B2 P0 footer 与冻结 descriptor 漂移")
        if physical != self._expected_physical:
            _fail("B2 P0 physical bytes 或 SHA-256 与冻结 descriptor 漂移")
        self._finished = True
        return footer

    def close(self) -> None:
        """关闭 P0 cursor；footer 成功后的 close 保持幂等。"""
        self._reader.close()


def _descriptor_footer_physical(
        value: ConversationHeldOutV4ProvenanceDagStreamDescriptor
        | ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        ) -> tuple[IntegerFramedStreamFooter, KRunFileDigest]:
    """从 B1/B2 stream descriptor 提取完整 P0 footer 与 physical 本体。"""
    if isinstance(value, ConversationHeldOutV4ProvenanceDagStreamDescriptor):
        return value.p0_footer, value.physical
    if isinstance(value, ConversationHeldOutV4ProvenanceClosureStreamDescriptor):
        return value.p0_footer, value.physical
    raise TypeError("B2 stream descriptor 类型错误")


def _verify_descriptor(
        root: KRunRoot,
        relative: Path,
        descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor
        | ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        *,
        budget: ConversationHeldOutV4ProvenanceClosureBudget,
        label: str,
        ) -> None:
    """逐条消费一个未解释字段的 P0 descriptor，以锁定所有上游物理身份。"""
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
        budget: ConversationHeldOutV4ProvenanceClosureBudget,
        label: str,
        ) -> ConversationHeldOutV4ProvenanceClosureStreamDescriptor:
    """排他写出一个有 record/payload/physical 上限的 sealed B2 P0 stream。"""
    _require_stage_name(stage)
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
            strict_integer_tuple(record, label="B2 output record")
            encoded_size = encoded_integer_tuple_size(record)
            if encoded_size > budget.max_record_payload_bytes:
                _budget_fail("B2 单 record payload 超过预算")
            if count >= budget.max_stream_record_count:
                _budget_fail("B2 P0 stream record 数超过预算")
            if encoded_size > budget.max_stream_payload_bytes - payload_bytes:
                _budget_fail("B2 P0 stream payload 超过预算")
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
    descriptor = ConversationHeldOutV4ProvenanceClosureStreamDescriptor(
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
    """在已创建的多阶段 work run 内排他新建一个空 sibling stage capability。"""
    _require_stage_name(stage)
    result = create_new_run_root(
        work_root.path / stage,
        require_k_drive=not work_root.test_transport,
        label=f"B2 {stage}",
    )
    require_fresh_empty_run_root(result, label=f"B2 {stage}")
    return result


def _stage_local_input(
        work_root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceDagStreamDescriptor
        | ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        *, label: str,
        ) -> tuple[KRunRoot, Path]:
    """从结果中的 root-relative descriptor 恢复其单层 stage capability 与文件名。"""
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
        current: int, maximum_next: int,
        *, budget: ConversationHeldOutV4ProvenanceClosureBudget,
        label: str,
        ) -> None:
    """在创建下一 stage 前保守预留它可能产生的物理字节。"""
    _require_nonnegative_int(current, label="B2 materialized current")
    _require_nonnegative_int(maximum_next, label="B2 materialized maximum_next")
    if maximum_next > budget.max_total_materialized_physical_bytes - current:
        _budget_fail(f"{label} 将超过 B2 累计物化物理字节预算")


def _add_materialized(
        current: int, actual: int,
        *, budget: ConversationHeldOutV4ProvenanceClosureBudget,
        label: str,
        ) -> int:
    """登记已完成 stage 的实际物化量，并再次防止内部预算漂移。"""
    _require_nonnegative_int(actual, label="B2 materialized actual")
    if actual > budget.max_total_materialized_physical_bytes - current:
        _budget_fail(f"{label} 实际物化物理字节超过预算")
    return current + actual


def _read_protocol_key(reader: IntegerStreamReader, *, label: str) -> ProtocolKey:
    """从完整 P0 record 恢复 ProtocolKey，拒绝摘要/截断代替本体。"""
    try:
        return ProtocolKey(reader.read_key(label=label))
    except (IntegerCodecError, TypeError, ValueError) as exc:
        _fail(f"{label} ProtocolKey 非法")
        raise AssertionError from exc


def encode_v4_provenance_closure_pair_record(
        descendant_node_key: ProtocolKey,
        ancestor_node_key: ProtocolKey,
        ancestor_snapshot: ConversationHeldOutV4SnapshotIdentity,
        ) -> tuple[int, ...]:
    """编码严格祖先 pair，并完整保留 ancestor snapshot 本体。"""
    if not isinstance(descendant_node_key, ProtocolKey):
        raise TypeError("B2 descendant_node_key 必须是 ProtocolKey")
    if not isinstance(ancestor_node_key, ProtocolKey):
        raise TypeError("B2 ancestor_node_key 必须是 ProtocolKey")
    if not isinstance(ancestor_snapshot, ConversationHeldOutV4SnapshotIdentity):
        raise TypeError("B2 ancestor_snapshot 类型错误")
    result = [V4_PROVENANCE_CLOSURE_PAIR_RECORD_SCHEMA]
    pack_key(result, descendant_node_key.components)
    pack_key(result, ancestor_node_key.components)
    pack_key(result, ancestor_snapshot.integer_stream())
    return tuple(result)


def decode_v4_provenance_closure_pair_record(
        record: tuple[int, ...],
        ) -> tuple[ProtocolKey, ProtocolKey, ConversationHeldOutV4SnapshotIdentity]:
    """严格恢复 B2 pair，拒绝尾字段、错误 schema 或缺失 snapshot 本体。"""
    try:
        reader = IntegerStreamReader(record)
        schema = reader.read_positive(label="B2 closure pair schema")
        if schema != V4_PROVENANCE_CLOSURE_PAIR_RECORD_SCHEMA:
            _fail("B2 closure pair schema 不兼容")
        descendant = _read_protocol_key(reader, label="B2 closure descendant")
        ancestor = _read_protocol_key(reader, label="B2 closure ancestor")
        snapshot = ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
            reader.read_key(label="B2 closure ancestor snapshot"))
        reader.finish()
        return descendant, ancestor, snapshot
    except ConversationHeldOutV4ProvenanceScalableClosureError:
        raise
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail("B2 closure pair record 非法")
        raise AssertionError from exc


def _pair_key(
        value: tuple[ProtocolKey, ProtocolKey, ConversationHeldOutV4SnapshotIdentity],
        ) -> tuple[int, ...]:
    """返回闭包 union 所用的完整 `(descendant, ancestor)` canonical key。"""
    return _packed_sort_key(value[0].components, value[1].components)


def _pair_sort_key(record: tuple[int, ...]) -> tuple[int, ...]:
    """供外排按 `(descendant, ancestor)` 排序，snapshot 不参与去重键。"""
    decoded = decode_v4_provenance_closure_pair_record(record)
    return _pair_key(decoded)


def _frontier_ancestor_sort_key(record: tuple[int, ...]) -> tuple[int, ...]:
    """供传播 join 按 `(ancestor, descendant)` 排序 frontier。"""
    descendant, ancestor, _snapshot = decode_v4_provenance_closure_pair_record(record)
    return _packed_sort_key(ancestor.components, descendant.components)


def _edge_child_sort_key(record: tuple[int, ...]) -> tuple[int, ...]:
    """供传播 join 按 `(child, parent)` 排序 B1 direct edge。"""
    child, parent, _snapshot = decode_v4_provenance_direct_edge_record(record)
    return _packed_sort_key(child.components, parent.components)


def _sort_one_stream(
        work_root: KRunRoot,
        source: ConversationHeldOutV4ProvenanceDagStreamDescriptor
        | ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        output_root: KRunRoot,
        *,
        stage: str,
        output_relative: Path,
        stream_kind: int,
        logical_stage_name: str,
        sort_key: object,
        budget: ConversationHeldOutV4ProvenanceClosureBudget,
        label: str,
        ) -> tuple[ConversationHeldOutV4ProvenanceClosureStreamDescriptor, int]:
    """将单一冻结 P0 stream 外排到新 sibling，并回比外排实际读取的 identity。"""
    source_root, source_relative = _stage_local_input(work_root, source, label=label)
    footer, physical = _descriptor_footer_physical(source)
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
    identities = result.identity.input_identities
    if (len(identities) != 1
            or not isinstance(identities[0], IntegerExternalSortInputIdentity)
            or identities[0].relative_path != source_relative
            or identities[0].footer != footer
            or identities[0].physical != physical):
        _fail(f"{label} external sort input identity 漂移")
    descriptor = ConversationHeldOutV4ProvenanceClosureStreamDescriptor(
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


def _initial_pair_stream(
        work_root: KRunRoot,
        direct_edge: ConversationHeldOutV4ProvenanceDagStreamDescriptor,
        output_root: KRunRoot,
        *,
        stage: str,
        budget: ConversationHeldOutV4ProvenanceClosureBudget,
        ) -> tuple[ConversationHeldOutV4ProvenanceClosureStreamDescriptor, int]:
    """从 B1 direct edge 流式物化 strict direct ancestor pair，拒绝 self edge。"""
    source_root, source_relative = _stage_local_input(
        work_root, direct_edge, label="B2 direct edge")

    def iter_pairs():
        """在唯一 reader 生命周期中按 direct edge 顺序产生初始 strict ancestor pair。"""
        count = 0
        with _VerifiedP0Reader(
                source_root,
                source_relative,
                footer=direct_edge.p0_footer,
                physical=direct_edge.physical,
                budget=budget,
                label="B2 initial direct edge input",
        ) as reader:
            while True:
                raw = reader.read_record()
                if raw is None:
                    break
                child, parent, snapshot = decode_v4_provenance_direct_edge_record(raw)
                if child.components == parent.components:
                    _fail("B2 direct edge 产生 self pair，检测到谱系环")
                if count >= budget.max_direct_edge_count:
                    _budget_fail("B2 direct edge 数超过预算")
                count += 1
                yield encode_v4_provenance_closure_pair_record(
                    child, parent, snapshot)
            reader.finish()

    descriptor = _write_stream(
        output_root,
        _INITIAL_PAIR_FILE,
        stage=stage,
        stream_kind=V4_PROVENANCE_CLOSURE_STREAM_INITIAL_PAIR,
        records=iter_pairs(),
        budget=budget,
        label="B2 initial pair output",
    )
    return descriptor, descriptor.p0_footer.record_count


def _initial_union(
        work_root: KRunRoot,
        source: ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        output_root: KRunRoot,
        *,
        stage: str,
        budget: ConversationHeldOutV4ProvenanceClosureBudget,
        ) -> tuple[
            ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
            ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
            int,
        ]:
    """去重初始 direct pair，冻结同 pair snapshot 一致性，并同时写 known/frontier。"""
    source_root, source_relative = _stage_local_input(
        work_root, source, label="B2 initial sorted pair")

    def unique_pairs():
        """以排序输入逐条折叠同 pair，不缓存全量 known。"""
        previous: tuple[ProtocolKey, ProtocolKey, ConversationHeldOutV4SnapshotIdentity] | None = None
        emitted_count = 0
        with _VerifiedP0Reader(
                source_root,
                source_relative,
                footer=source.p0_footer,
                physical=source.physical,
                budget=budget,
                label="B2 initial sorted pair input",
        ) as reader:
            while True:
                raw = reader.read_record()
                if raw is None:
                    break
                current = decode_v4_provenance_closure_pair_record(raw)
                key = _pair_key(current)
                if current[0].components == current[1].components:
                    _fail("B2 initial pair 检测到谱系环")
                if previous is not None:
                    previous_key = _pair_key(previous)
                    if key < previous_key:
                        _fail("B2 initial sorted pair key 倒序")
                    if key == previous_key:
                        if current[2] != previous[2]:
                            _fail("B2 同 pair ancestor snapshot 冲突")
                        continue
                previous = current
                if emitted_count >= budget.max_known_pair_count:
                    _budget_fail("B2 initial known pair 数超过预算")
                emitted_count += 1
                yield encode_v4_provenance_closure_pair_record(*current)
            reader.finish()

    # 同一 generator 不能同时供两个 writer。先 materialize canonical known，再由其 P0
    # descriptor 流式复制到 frontier，保持每个阶段只依赖 sealed 输入。
    known = _write_stream(
        output_root,
        _KNOWN_FILE,
        stage=stage,
        stream_kind=V4_PROVENANCE_CLOSURE_STREAM_KNOWN,
        records=unique_pairs(),
        budget=budget,
        label="B2 initial known output",
    )
    known_root, known_relative = _stage_local_input(
        work_root,
        known,
        label="B2 initial known copy",
    )

    def copy_known():
        """从刚 seal 的 known 逐条复制为首轮 frontier，不在内存中保留 pair 集。"""
        with _VerifiedP0Reader(
                known_root,
                known_relative,
                footer=known.p0_footer,
                physical=known.physical,
                budget=budget,
                label="B2 initial known input",
        ) as reader:
            while True:
                record = reader.read_record()
                if record is None:
                    break
                yield record
            reader.finish()

    frontier = _write_stream(
        output_root,
        _FRONTIER_FILE,
        stage=stage,
        stream_kind=V4_PROVENANCE_CLOSURE_STREAM_FRONTIER,
        records=copy_known(),
        budget=budget,
        label="B2 initial frontier output",
    )
    return known, frontier, known.p0_footer.record_count


def _read_edge_group(
        reader: _VerifiedP0Reader,
        pending: tuple[ProtocolKey, ProtocolKey,
                       ConversationHeldOutV4SnapshotIdentity] | None,
        *, budget: ConversationHeldOutV4ProvenanceClosureBudget,
        ) -> tuple[
            tuple[ProtocolKey, tuple[tuple[ProtocolKey,
                                           ConversationHeldOutV4SnapshotIdentity], ...]]
            | None,
            tuple[ProtocolKey, ProtocolKey,
                  ConversationHeldOutV4SnapshotIdentity] | None,
        ]:
    """读取一个按 child 排序的 direct-edge parent group，仅缓存该 node 的 parent 集。"""
    first = pending
    if first is None:
        raw = reader.read_record()
        if raw is None:
            return None, None
        first = decode_v4_provenance_direct_edge_record(raw)
    child, parent, snapshot = first
    group: list[tuple[ProtocolKey, ConversationHeldOutV4SnapshotIdentity]] = [
        (parent, snapshot)]
    previous_parent = parent.components
    next_pending = None
    while True:
        raw = reader.read_record()
        if raw is None:
            break
        candidate = decode_v4_provenance_direct_edge_record(raw)
        candidate_child, candidate_parent, candidate_snapshot = candidate
        if candidate_child.components < child.components:
            _fail("B2 edge-by-child key 倒序")
        if candidate_child.components != child.components:
            next_pending = candidate
            break
        if candidate_parent.components <= previous_parent:
            _fail("B2 edge-by-child parent key 未严格递增")
        if len(group) >= budget.max_parents_per_node:
            _budget_fail("B2 同一 child parent group 超过预算")
        group.append((candidate_parent, candidate_snapshot))
        previous_parent = candidate_parent.components
    if len(group) > budget.max_parents_per_node:
        _budget_fail("B2 同一 child parent group 超过预算")
    return (child, tuple(group)), next_pending


def _propagate_frontier(
        work_root: KRunRoot,
        frontier: ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        edge_by_child: ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        output_root: KRunRoot,
        *, stage: str, budget: ConversationHeldOutV4ProvenanceClosureBudget,
        ) -> ConversationHeldOutV4ProvenanceClosureStreamDescriptor:
    """以 frontier ancestor 与 direct-edge child 的流式 group join 产生下一轮候选。"""
    frontier_root, frontier_relative = _stage_local_input(
        work_root, frontier, label="B2 frontier-by-ancestor")
    edge_root, edge_relative = _stage_local_input(
        work_root, edge_by_child, label="B2 edge-by-child")

    def candidates():
        """仅缓存当前 child 的 parent group，逐 frontier pair 生成 candidate。"""
        candidate_count = 0
        previous_frontier_key: tuple[int, ...] | None = None
        with _VerifiedP0Reader(
                frontier_root,
                frontier_relative,
                footer=frontier.p0_footer,
                physical=frontier.physical,
                budget=budget,
                label="B2 frontier propagation input",
        ) as frontier_reader, _VerifiedP0Reader(
                edge_root,
                edge_relative,
                footer=edge_by_child.p0_footer,
                physical=edge_by_child.physical,
                budget=budget,
                label="B2 edge propagation input",
        ) as edge_reader:
            edge_group, pending = _read_edge_group(edge_reader, None, budget=budget)
            while True:
                raw = frontier_reader.read_record()
                if raw is None:
                    break
                descendant, ancestor, _snapshot = (
                    decode_v4_provenance_closure_pair_record(raw))
                frontier_key = _packed_sort_key(
                    ancestor.components, descendant.components)
                if (previous_frontier_key is not None
                        and frontier_key <= previous_frontier_key):
                    _fail("B2 frontier-by-ancestor key 未严格递增")
                previous_frontier_key = frontier_key
                while (edge_group is not None
                       and edge_group[0].components < ancestor.components):
                    edge_group, pending = _read_edge_group(
                        edge_reader, pending, budget=budget)
                if (edge_group is None
                        or edge_group[0].components != ancestor.components):
                    continue
                for parent, parent_snapshot in edge_group[1]:
                    if descendant.components == parent.components:
                        _fail("B2 新 pair 的 descendant 等于 ancestor，检测到谱系环")
                    if candidate_count >= budget.max_candidate_pair_count_per_round:
                        _budget_fail("B2 单轮 candidate pair 超过预算")
                    candidate_count += 1
                    yield encode_v4_provenance_closure_pair_record(
                        descendant, parent, parent_snapshot)
            frontier_reader.finish()
            edge_reader.finish()

    return _write_stream(
        output_root,
        _CANDIDATE_FILE,
        stage=stage,
        stream_kind=V4_PROVENANCE_CLOSURE_STREAM_CANDIDATE,
        records=candidates(),
        budget=budget,
        label="B2 candidate output",
    )


def _next_pair(
        reader: _VerifiedP0Reader,
        *, previous_key: tuple[int, ...] | None, label: str,
        ) -> tuple[
            tuple[ProtocolKey, ProtocolKey, ConversationHeldOutV4SnapshotIdentity] | None,
            tuple[int, ...] | None,
        ]:
    """读取 known 的一条严格递增 pair，不允许其中出现重复键。"""
    raw = reader.read_record()
    if raw is None:
        return None, previous_key
    value = decode_v4_provenance_closure_pair_record(raw)
    key = _pair_key(value)
    if previous_key is not None and key <= previous_key:
        _fail(f"{label} key 未严格递增")
    return value, key


def _next_candidate_group(
        reader: _VerifiedP0Reader,
        pending: tuple[ProtocolKey, ProtocolKey,
                       ConversationHeldOutV4SnapshotIdentity] | None,
        *, previous_key: tuple[int, ...] | None, label: str,
        ) -> tuple[
            tuple[ProtocolKey, ProtocolKey, ConversationHeldOutV4SnapshotIdentity] | None,
            tuple[ProtocolKey, ProtocolKey,
                  ConversationHeldOutV4SnapshotIdentity] | None,
            tuple[int, ...] | None,
        ]:
    """折叠 candidate 的同 pair group，要求所有路径带回同一完整 snapshot。"""
    first = pending
    if first is None:
        raw = reader.read_record()
        if raw is None:
            return None, None, previous_key
        first = decode_v4_provenance_closure_pair_record(raw)
    key = _pair_key(first)
    if previous_key is not None and key <= previous_key:
        _fail(f"{label} key 未严格递增")
    next_pending = None
    while True:
        raw = reader.read_record()
        if raw is None:
            break
        current = decode_v4_provenance_closure_pair_record(raw)
        current_key = _pair_key(current)
        if current_key < key:
            _fail(f"{label} key 倒序")
        if current_key != key:
            next_pending = current
            break
        if current[2] != first[2]:
            _fail("B2 同 candidate pair ancestor snapshot 冲突")
    return first, next_pending, key


def _union_known_and_candidates(
        work_root: KRunRoot,
        known: ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        candidate: ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
        output_root: KRunRoot,
        *, stage: str, budget: ConversationHeldOutV4ProvenanceClosureBudget,
        ) -> tuple[
            ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
            ConversationHeldOutV4ProvenanceClosureStreamDescriptor,
            int,
        ]:
    """merge-union known/candidate，并只把新 pair 写入下一轮 frontier。"""
    known_root, known_relative = _stage_local_input(work_root, known, label="B2 known")
    candidate_root, candidate_relative = _stage_local_input(
        work_root, candidate, label="B2 candidate sorted")

    def merged_records(*, frontier_only: bool):
        """以相同 deterministic merge 生成 known 或仅生成此次新增 frontier。"""
        known_previous: tuple[int, ...] | None = None
        candidate_previous: tuple[int, ...] | None = None
        emitted_known = 0
        with _VerifiedP0Reader(
                known_root,
                known_relative,
                footer=known.p0_footer,
                physical=known.physical,
                budget=budget,
                label="B2 known union input",
        ) as known_reader, _VerifiedP0Reader(
                candidate_root,
                candidate_relative,
                footer=candidate.p0_footer,
                physical=candidate.physical,
                budget=budget,
                label="B2 candidate union input",
        ) as candidate_reader:
            known_value, known_previous = _next_pair(
                known_reader, previous_key=known_previous, label="B2 known")
            candidate_value, pending_candidate, candidate_previous = _next_candidate_group(
                candidate_reader,
                None,
                previous_key=candidate_previous,
                label="B2 candidate",
            )
            while known_value is not None or candidate_value is not None:
                known_key = _pair_key(known_value) if known_value is not None else None
                candidate_key = (
                    _pair_key(candidate_value)
                    if candidate_value is not None else None)
                is_new = False
                if (candidate_value is None
                        or (known_value is not None and known_key < candidate_key)):
                    output = known_value
                    known_value, known_previous = _next_pair(
                        known_reader,
                        previous_key=known_previous,
                        label="B2 known",
                    )
                elif (known_value is None or candidate_key < known_key):
                    output = candidate_value
                    if output[0].components == output[1].components:
                        _fail("B2 candidate 产生 self pair，检测到谱系环")
                    is_new = True
                    candidate_value, pending_candidate, candidate_previous = (
                        _next_candidate_group(
                            candidate_reader,
                            pending_candidate,
                            previous_key=candidate_previous,
                            label="B2 candidate",
                        ))
                else:
                    if known_value[2] != candidate_value[2]:
                        _fail("B2 known/candidate 同 pair ancestor snapshot 冲突")
                    output = known_value
                    known_value, known_previous = _next_pair(
                        known_reader,
                        previous_key=known_previous,
                        label="B2 known",
                    )
                    candidate_value, pending_candidate, candidate_previous = (
                        _next_candidate_group(
                            candidate_reader,
                            pending_candidate,
                            previous_key=candidate_previous,
                            label="B2 candidate",
                        ))
                if not frontier_only:
                    if emitted_known >= budget.max_known_pair_count:
                        _budget_fail("B2 known pair 数超过预算")
                    emitted_known += 1
                    yield encode_v4_provenance_closure_pair_record(*output)
                elif is_new:
                    yield encode_v4_provenance_closure_pair_record(*output)
            known_reader.finish()
            candidate_reader.finish()

    known_output = _write_stream(
        output_root,
        _KNOWN_FILE,
        stage=stage,
        stream_kind=V4_PROVENANCE_CLOSURE_STREAM_KNOWN,
        records=merged_records(frontier_only=False),
        budget=budget,
        label="B2 next known output",
    )
    frontier_output = _write_stream(
        output_root,
        _FRONTIER_FILE,
        stage=stage,
        stream_kind=V4_PROVENANCE_CLOSURE_STREAM_FRONTIER,
        records=merged_records(frontier_only=True),
        budget=budget,
        label="B2 next frontier output",
    )
    return known_output, frontier_output, known_output.p0_footer.record_count


def _verify_b1_result(
        work_root: KRunRoot,
        result: ConversationHeldOutV4ProvenanceDirectDagResult,
        *, budget: ConversationHeldOutV4ProvenanceClosureBudget,
        ) -> None:
    """重放 B1 的所有 six descriptors，避免只验证所需 direct edge 而遗漏上游漂移。"""
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
            label=f"B2 B1 {label} revalidation",
        )


def build_v4_provenance_transitive_closure(
        direct_dag_result: ConversationHeldOutV4ProvenanceDirectDagResult,
        work_run_root: KRunRoot,
        *, budget: ConversationHeldOutV4ProvenanceClosureBudget,
        logical_stage_name: str,
        stage_namespace: str | None = None,
        ) -> ConversationHeldOutV4ProvenanceClosureResult:
    """执行 P2-B2 的 strict transitive ancestor closure，不发布 provenance 结论。

    ``work_run_root`` 必须是 B1 原始 create-new capability；B2 允许其中已有 B1 stage，
    但每个新 B2 stage 都必须此前不存在。可选 ``stage_namespace`` 只供同一 B1 输入的独立
    闭包重放使用，保持默认 stage 名不变且阻止与已存在 closure 输出混写。初始 strict direct
    pair 进入 known/frontier，随后仅以 frontier 的 ancestor 查 direct edge child，直到新
    frontier 为空。self pair 是真实环信号而非可接受的 self closure，B3 应从 endpoint 明确
    处理 self 投影。
    """
    if not isinstance(direct_dag_result,
                      ConversationHeldOutV4ProvenanceDirectDagResult):
        raise TypeError("B2 direct_dag_result 类型错误")
    if not isinstance(work_run_root, KRunRoot):
        raise TypeError("B2 work_run_root 必须是 KRunRoot")
    if not isinstance(budget, ConversationHeldOutV4ProvenanceClosureBudget):
        raise TypeError("B2 budget 类型错误")
    _require_stage_name(logical_stage_name)
    namespace = _require_stage_namespace(stage_namespace)
    require_created_run_root(work_run_root, label="B2 work_run_root")
    _verify_b1_result(work_run_root, direct_dag_result, budget=budget)
    if direct_dag_result.node_count > budget.max_node_count:
        _budget_fail("B2 B1 node_count 超过预算")
    if direct_dag_result.direct_edge_count > budget.max_direct_edge_count:
        _budget_fail("B2 B1 direct_edge_count 超过预算")

    initial_pair_stage = _stage_name(namespace, _STAGE_INITIAL_PAIR)
    initial_sort_stage = _stage_name(namespace, _STAGE_INITIAL_SORT)
    initial_union_stage = _stage_name(namespace, _STAGE_INITIAL_UNION)
    edge_by_child_stage = _stage_name(namespace, _STAGE_EDGE_BY_CHILD)

    materialized = 0
    _reserve_materialized(
        materialized,
        budget.max_stream_physical_bytes,
        budget=budget,
        label="B2 initial pair",
    )
    initial_root = _new_stage(work_run_root, initial_pair_stage)
    initial_pair, direct_edge_count = _initial_pair_stream(
        work_run_root,
        direct_dag_result.direct_edge_stream,
        initial_root,
        stage=initial_pair_stage,
        budget=budget,
    )
    materialized = _add_materialized(
        materialized,
        initial_pair.physical.byte_count,
        budget=budget,
        label="B2 initial pair",
    )
    if direct_edge_count != direct_dag_result.direct_edge_count:
        _fail("B2 direct edge count 与 B1 result 不一致")

    sort_reservation = (
        budget.external_sort_budget.max_temporary_physical_bytes
        + budget.external_sort_budget.max_output_physical_bytes)
    _reserve_materialized(
        materialized,
        sort_reservation,
        budget=budget,
        label="B2 initial pair sort",
    )
    initial_sort_root = _new_stage(work_run_root, initial_sort_stage)
    initial_sorted, temporary_bytes = _sort_one_stream(
        work_run_root,
        initial_pair,
        initial_sort_root,
        stage=initial_sort_stage,
        output_relative=_INITIAL_PAIR_SORTED_FILE,
        stream_kind=V4_PROVENANCE_CLOSURE_STREAM_INITIAL_PAIR_SORTED,
        logical_stage_name=f"{logical_stage_name}.initial-pair.v1",
        sort_key=_pair_sort_key,
        budget=budget,
        label="B2 initial pair sort",
    )
    materialized = _add_materialized(
        materialized,
        temporary_bytes + initial_sorted.physical.byte_count,
        budget=budget,
        label="B2 initial pair sort",
    )

    _reserve_materialized(
        materialized,
        budget.max_stream_physical_bytes * 2,
        budget=budget,
        label="B2 initial union",
    )
    initial_union_root = _new_stage(work_run_root, initial_union_stage)
    known, frontier, known_pair_count = _initial_union(
        work_run_root,
        initial_sorted,
        initial_union_root,
        stage=initial_union_stage,
        budget=budget,
    )
    materialized = _add_materialized(
        materialized,
        known.physical.byte_count + frontier.physical.byte_count,
        budget=budget,
        label="B2 initial union",
    )
    if known_pair_count > budget.max_known_pair_count:
        _budget_fail("B2 initial known pair 数超过预算")

    _reserve_materialized(
        materialized,
        sort_reservation,
        budget=budget,
        label="B2 edge-by-child sort",
    )
    edge_stage = _new_stage(work_run_root, edge_by_child_stage)
    edge_by_child, temporary_bytes = _sort_one_stream(
        work_run_root,
        direct_dag_result.direct_edge_stream,
        edge_stage,
        stage=edge_by_child_stage,
        output_relative=_EDGE_BY_CHILD_FILE,
        stream_kind=V4_PROVENANCE_CLOSURE_STREAM_EDGE_BY_CHILD,
        logical_stage_name=f"{logical_stage_name}.edge-by-child.v1",
        sort_key=_edge_child_sort_key,
        budget=budget,
        label="B2 edge-by-child sort",
    )
    materialized = _add_materialized(
        materialized,
        temporary_bytes + edge_by_child.physical.byte_count,
        budget=budget,
        label="B2 edge-by-child sort",
    )
    if edge_by_child.p0_footer.record_count != direct_edge_count:
        _fail("B2 edge-by-child count 与 direct edge 不一致")

    round_count = 0
    while frontier.p0_footer.record_count:
        if round_count >= budget.max_closure_round_count:
            _budget_fail("B2 closure round 数超过预算")
        round_count += 1
        round_token = f"r{round_count:04d}"
        frontier_sort_stage = _stage_name(
            namespace, f"stage-10-closure-frontier-sort-{round_token}")
        candidate_stage = _stage_name(
            namespace, f"stage-11-closure-candidate-{round_token}")
        candidate_sort_stage = _stage_name(
            namespace, f"stage-12-closure-candidate-sort-{round_token}")
        union_stage = _stage_name(
            namespace, f"stage-13-closure-union-{round_token}")

        _reserve_materialized(
            materialized,
            sort_reservation,
            budget=budget,
            label=f"B2 {round_token} frontier sort",
        )
        frontier_sort_root = _new_stage(work_run_root, frontier_sort_stage)
        frontier_by_ancestor, temporary_bytes = _sort_one_stream(
            work_run_root,
            frontier,
            frontier_sort_root,
            stage=frontier_sort_stage,
            output_relative=_FRONTIER_BY_ANCESTOR_FILE,
            stream_kind=V4_PROVENANCE_CLOSURE_STREAM_FRONTIER_BY_ANCESTOR,
            logical_stage_name=f"{logical_stage_name}.frontier-by-ancestor.{round_token}",
            sort_key=_frontier_ancestor_sort_key,
            budget=budget,
            label=f"B2 {round_token} frontier sort",
        )
        materialized = _add_materialized(
            materialized,
            temporary_bytes + frontier_by_ancestor.physical.byte_count,
            budget=budget,
            label=f"B2 {round_token} frontier sort",
        )

        _reserve_materialized(
            materialized,
            budget.max_stream_physical_bytes,
            budget=budget,
            label=f"B2 {round_token} candidate",
        )
        candidate_root = _new_stage(work_run_root, candidate_stage)
        candidate = _propagate_frontier(
            work_run_root,
            frontier_by_ancestor,
            edge_by_child,
            candidate_root,
            stage=candidate_stage,
            budget=budget,
        )
        materialized = _add_materialized(
            materialized,
            candidate.physical.byte_count,
            budget=budget,
            label=f"B2 {round_token} candidate",
        )

        _reserve_materialized(
            materialized,
            sort_reservation,
            budget=budget,
            label=f"B2 {round_token} candidate sort",
        )
        candidate_sort_root = _new_stage(work_run_root, candidate_sort_stage)
        candidate_sorted, temporary_bytes = _sort_one_stream(
            work_run_root,
            candidate,
            candidate_sort_root,
            stage=candidate_sort_stage,
            output_relative=_CANDIDATE_SORTED_FILE,
            stream_kind=V4_PROVENANCE_CLOSURE_STREAM_CANDIDATE_SORTED,
            logical_stage_name=f"{logical_stage_name}.candidate.{round_token}",
            sort_key=_pair_sort_key,
            budget=budget,
            label=f"B2 {round_token} candidate sort",
        )
        materialized = _add_materialized(
            materialized,
            temporary_bytes + candidate_sorted.physical.byte_count,
            budget=budget,
            label=f"B2 {round_token} candidate sort",
        )

        _reserve_materialized(
            materialized,
            budget.max_stream_physical_bytes * 2,
            budget=budget,
            label=f"B2 {round_token} union",
        )
        union_root = _new_stage(work_run_root, union_stage)
        known, frontier, known_pair_count = _union_known_and_candidates(
            work_run_root,
            known,
            candidate_sorted,
            union_root,
            stage=union_stage,
            budget=budget,
        )
        materialized = _add_materialized(
            materialized,
            known.physical.byte_count + frontier.physical.byte_count,
            budget=budget,
            label=f"B2 {round_token} union",
        )

    return ConversationHeldOutV4ProvenanceClosureResult(
        logical_stage_name,
        direct_dag_result.stable_key(),
        edge_by_child,
        known,
        frontier,
        direct_dag_result.node_count,
        direct_edge_count,
        known_pair_count,
        round_count,
        materialized,
        budget,
    )


__all__ = [
    "ConversationHeldOutV4ProvenanceClosureBudget",
    "ConversationHeldOutV4ProvenanceClosureResult",
    "ConversationHeldOutV4ProvenanceClosureStreamDescriptor",
    "ConversationHeldOutV4ProvenanceScalableClosureBudgetExceeded",
    "ConversationHeldOutV4ProvenanceScalableClosureError",
    "V4_PROVENANCE_CLOSURE_BUDGET_SCHEMA",
    "V4_PROVENANCE_CLOSURE_PAIR_RECORD_SCHEMA",
    "V4_PROVENANCE_CLOSURE_RESULT_SCHEMA",
    "V4_PROVENANCE_CLOSURE_STREAM_CANDIDATE",
    "V4_PROVENANCE_CLOSURE_STREAM_CANDIDATE_SORTED",
    "V4_PROVENANCE_CLOSURE_STREAM_DESCRIPTOR_SCHEMA",
    "V4_PROVENANCE_CLOSURE_STREAM_EDGE_BY_CHILD",
    "V4_PROVENANCE_CLOSURE_STREAM_FRONTIER",
    "V4_PROVENANCE_CLOSURE_STREAM_FRONTIER_BY_ANCESTOR",
    "V4_PROVENANCE_CLOSURE_STREAM_INITIAL_PAIR",
    "V4_PROVENANCE_CLOSURE_STREAM_INITIAL_PAIR_SORTED",
    "V4_PROVENANCE_CLOSURE_STREAM_KNOWN",
    "build_v4_provenance_transitive_closure",
    "decode_v4_provenance_closure_pair_record",
    "encode_v4_provenance_closure_pair_record",
]
