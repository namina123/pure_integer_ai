"""DLG-05 v4 R04b P2-C 的跨侧精确 provenance merge 基础设施。

本模块只接受 B3 已排序 P0 projection，并以两 reader/一 writer 流式比较完整
SourceRef、content SHA-256 或 lineage 本体。它不接入 active ingest，不知道 runtime、
owner、private/formal、JSON 或 SQLite；跨侧交集是 fail-closed 机械事件，不是资格结论。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import BinaryIO

from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4ProvenanceError,
    V4_PROVENANCE_SIDE_HELD_OUT,
    V4_PROVENANCE_SIDE_TRAINING,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_projection import (
    ConversationHeldOutV4ProvenanceProjectionResult,
    ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
    V4_PROVENANCE_PROJECTION_STREAM_CONTENT,
    V4_PROVENANCE_PROJECTION_STREAM_LINEAGE,
    V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF,
    decode_v4_provenance_content_projection_record,
    decode_v4_provenance_lineage_projection_record,
    decode_v4_provenance_source_ref_projection_record,
    encode_v4_provenance_content_projection_record,
    encode_v4_provenance_lineage_projection_record,
    encode_v4_provenance_source_ref_projection_record,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerFramedStreamBudgetExceeded,
    IntegerFramedStreamFooter,
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
    encode_integer_tuple,
    encoded_integer_tuple_size,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunFileDigest,
    KRunRoot,
    capture_plain_file_identity,
    create_new_run_root,
    is_created_run_root,
    open_existing_run_root,
    open_exclusive_binary,
    open_plain_binary,
    publish_manifest_last,
    require_exact_file_closure,
    require_created_run_root,
    require_disjoint_run_roots,
    require_fresh_empty_run_root,
    require_plain_file_identity,
    sha256_plain_file,
)


V4_PROVENANCE_CROSS_SIDE_MERGE_BUDGET_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_STREAM_DESCRIPTOR_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_CHANNEL_INTENT_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_CHANNEL_RESULT_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_ARTIFACT_DESCRIPTOR_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_CURSOR_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_FILE_DESCRIPTOR_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_RESOURCE_RECEIPT_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_MANIFEST_SCHEMA = 1
V4_PROVENANCE_CROSS_SIDE_MERGE_PUBLICATION_RESULT_SCHEMA = 1

V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CURSOR = 1
V4_PROVENANCE_CROSS_SIDE_ARTIFACT_RECEIPT = 2
V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CHANNEL_INTENT = 3
V4_PROVENANCE_CROSS_SIDE_RECEIPT_STATUS_ZERO_INTERSECTION_MECHANICAL = 1

V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF = (
    V4_PROVENANCE_PROJECTION_STREAM_SOURCE_REF)
V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT = (
    V4_PROVENANCE_PROJECTION_STREAM_CONTENT)
V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE = (
    V4_PROVENANCE_PROJECTION_STREAM_LINEAGE)

_CHANNELS = (
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT,
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE,
)
_CHANNEL_STAGE_NAMES = {
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF:
        "stage-20-cross-side-source-ref",
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT:
        "stage-21-cross-side-content",
    V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE:
        "stage-22-cross-side-lineage",
}
_INTENT_STREAM_FILE = Path("intent.pifrs")
_MERGED_STREAM_FILE = Path("merged.pifrs")
_CURSOR_STREAM_FILE = Path("cursor.pifrs")
_RECEIPT_STREAM_FILE = Path("resource-receipt.pifrs")
_MANIFEST_FILE = Path("manifest.pii")
_RECEIPT_STAGE_NAME = "stage-26-cross-side-resource-receipt"


# object-model: exception
class ConversationHeldOutV4ProvenanceCrossSideMergeError(RuntimeError):
    """P2-C 的输入身份、排序、闭合或封存边界不成立。"""


# object-model: exception
class ConversationHeldOutV4ProvenanceCrossSideMergeBudgetExceeded(
        ConversationHeldOutV4ProvenanceCrossSideMergeError):
    """P2-C 的 record、stream、physical 或打开文件预算耗尽。"""


# object-model: exception
class ConversationHeldOutV4ProvenanceCrossSideOverlapError(
        ConversationHeldOutV4ProvenanceCrossSideMergeError):
    """完整 primary key 同时出现于 training 与 held-out。"""


def _fail(message: str) -> None:
    """统一产生不携带输入正文的 P2-C fail-closed 错误。"""
    raise ConversationHeldOutV4ProvenanceCrossSideMergeError(message)


def _budget_fail(message: str) -> None:
    """区分资源上限与 provenance 结构/身份异常。"""
    raise ConversationHeldOutV4ProvenanceCrossSideMergeBudgetExceeded(message)


def _p0_read_fail(error: BaseException, *, label: str) -> None:
    """把底层 P0 损坏或预算异常统一收口为 P2-C 可审计错误。"""
    if isinstance(error, IntegerFramedStreamBudgetExceeded):
        _budget_fail(f"{label} P0 framing 超过预算")
    _fail(f"{label} P0 framing 不完整或损坏")


def _require_positive_int(value: int, *, label: str) -> int:
    """拒绝 bool、零和负数的固定容量字段。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def _require_nonnegative_int(value: int, *, label: str) -> int:
    """拒绝 bool 与负数的累计计数。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _require_stage_name(value: str) -> str:
    """限制 stage 为可审计的短 ASCII 标识，不能带路径语义。"""
    if not isinstance(value, str) or not value:
        raise TypeError("P2-C logical_stage_name 必须是非空 str")
    if len(value) > 56:
        raise ValueError("P2-C logical_stage_name 超过长度上限")
    if not value[0].isascii() or not value[0].isalnum():
        raise ValueError("P2-C logical_stage_name 必须以 ASCII 字母或数字开始")
    if any(not item.isascii() or not (item.isalnum() or item in {"-", "_", "."})
           for item in value):
        raise ValueError("P2-C logical_stage_name 只能包含 ASCII 字母、数字、-、_、.")
    return value


def _relative_path_key(value: Path, *, label: str) -> tuple[int, ...]:
    """把受限 output 相对路径写为不含本机 root 的 ASCII 整数键。"""
    if (not isinstance(value, Path) or value.is_absolute() or value.drive
            or value.root or not value.parts or ".." in value.parts
            or any(part in {"", ".", ".."} for part in value.parts)):
        raise ValueError(f"{label} 必须是非空且不含 .. 的相对 Path")
    text = value.as_posix()
    if any(not item.isascii() for item in text):
        raise ValueError(f"{label} 必须只含 ASCII")
    return tuple(ord(item) for item in text)


def _digest_key(value: KRunFileDigest) -> tuple[int, ...]:
    """展开完整物理字节数和 SHA-256，不把摘要误作 descriptor 本体。"""
    if not isinstance(value, KRunFileDigest):
        raise TypeError("P2-C physical 必须是 KRunFileDigest")
    return (value.byte_count, *value.sha256)


def _packed_sort_key(*parts: tuple[int, ...]) -> tuple[int, ...]:
    """长度分帧拼接排序字段，避免相邻完整整数键产生歧义。"""
    result: list[int] = []
    for part in parts:
        pack_key(result, part)
    return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeBudget:
    """P2-C 单通道 merge 与后续 cursor/receipt/manifest 的完整有限预算。"""

    max_record_payload_bytes: int
    max_stream_record_count: int
    max_stream_payload_bytes: int
    max_stream_physical_bytes: int
    max_total_materialized_physical_bytes: int
    max_open_files: int
    max_intent_physical_bytes: int
    max_cursor_physical_bytes: int
    max_receipt_physical_bytes: int
    max_manifest_payload_bytes: int
    max_pre_manifest_file_count: int

    def __post_init__(self) -> None:
        """拒绝无界 stream、publication 元数据和 I/O 并发配置。"""
        for label, value in (
                ("max_record_payload_bytes", self.max_record_payload_bytes),
                ("max_stream_record_count", self.max_stream_record_count),
                ("max_stream_payload_bytes", self.max_stream_payload_bytes),
                ("max_stream_physical_bytes", self.max_stream_physical_bytes),
                ("max_total_materialized_physical_bytes",
                 self.max_total_materialized_physical_bytes),
                ("max_open_files", self.max_open_files),
                ("max_intent_physical_bytes", self.max_intent_physical_bytes),
                ("max_cursor_physical_bytes", self.max_cursor_physical_bytes),
                ("max_receipt_physical_bytes", self.max_receipt_physical_bytes),
                ("max_manifest_payload_bytes", self.max_manifest_payload_bytes),
                ("max_pre_manifest_file_count",
                 self.max_pre_manifest_file_count)):
            _require_positive_int(value, label=f"P2-C {label}")
        if self.max_record_payload_bytes > self.max_stream_payload_bytes:
            raise ValueError("P2-C 单 record payload 不得超过 stream payload 预算")
        if self.max_intent_physical_bytes > self.max_stream_physical_bytes:
            raise ValueError("P2-C intent physical 不得超过 stream physical 预算")
        if self.max_cursor_physical_bytes > self.max_stream_physical_bytes:
            raise ValueError("P2-C cursor physical 不得超过 stream physical 预算")
        if self.max_receipt_physical_bytes > self.max_stream_physical_bytes:
            raise ValueError("P2-C receipt physical 不得超过 stream physical 预算")
        if self.max_open_files < 3:
            raise ValueError("P2-C 至少需要两个 reader 和一个 writer")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 K root 的全部预算字段。"""
        return (
            V4_PROVENANCE_CROSS_SIDE_MERGE_BUDGET_SCHEMA,
            self.max_record_payload_bytes,
            self.max_stream_record_count,
            self.max_stream_payload_bytes,
            self.max_stream_physical_bytes,
            self.max_total_materialized_physical_bytes,
            self.max_open_files,
            self.max_intent_physical_bytes,
            self.max_cursor_physical_bytes,
            self.max_receipt_physical_bytes,
            self.max_manifest_payload_bytes,
            self.max_pre_manifest_file_count,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回可供 cursor/receipt 绑定的 schema-backed 完整预算键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeStreamDescriptor:
    """P2-C 在 output run 中新建的 sealed merged P0 stream 身份。"""

    channel: int
    output_relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """固定通道、相对定位和完整 P0/physical identity。"""
        if type(self.channel) is not int or self.channel not in _CHANNELS:
            raise ValueError("P2-C merged stream channel 未注册")
        _relative_path_key(
            self.output_relative_path,
            label="P2-C merged output_relative_path",
        )
        if not isinstance(self.p0_footer, IntegerFramedStreamFooter):
            raise TypeError("P2-C merged p0_footer 类型错误")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("P2-C merged physical 类型错误")
        if self.physical.byte_count <= 0:
            raise ValueError("P2-C merged sealed P0 physical bytes 必须为正")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 output root 的完整流 identity。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_STREAM_DESCRIPTOR_SCHEMA,
                  self.channel]
        for value in (
                _relative_path_key(
                    self.output_relative_path,
                    label="P2-C merged output_relative_path"),
                self.p0_footer.integer_tuple(),
                _digest_key(self.physical)):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 cursor/receipt 可携带的完整 output descriptor key。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeChannelIntent:
    """在 channel output 写入前封存的冻结启动身份，不含任何运行结果。"""

    logical_stage_name: str
    channel: int
    training_b3_stable_key: tuple[int, ...]
    held_out_b3_stable_key: tuple[int, ...]
    training_input: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor
    held_out_input: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor
    code_identity: ProtocolKey
    budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget

    def __post_init__(self) -> None:
        """要求 intent 完整绑定本次确定性 merge 的全部冻结输入。"""
        _require_stage_name(self.logical_stage_name)
        if type(self.channel) is not int or self.channel not in _CHANNELS:
            raise ValueError("P2-C channel intent channel 未注册")
        for label, value in (
                ("training_b3_stable_key", self.training_b3_stable_key),
                ("held_out_b3_stable_key", self.held_out_b3_stable_key)):
            try:
                strict_integer_tuple(value, label=f"P2-C channel intent {label}")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"P2-C channel intent {label} 必须是严格整数 tuple") from exc
        for label, descriptor in (
                ("training_input", self.training_input),
                ("held_out_input", self.held_out_input)):
            if not isinstance(
                    descriptor,
                    ConversationHeldOutV4ProvenanceProjectionStreamDescriptor):
                raise TypeError(
                    f"P2-C channel intent {label} 必须是 B3 projection descriptor")
            if descriptor.stream_kind != self.channel:
                raise ValueError(
                    f"P2-C channel intent {label} channel 与 intent 不一致")
        if not isinstance(self.code_identity, ProtocolKey):
            raise TypeError("P2-C channel intent code_identity 必须是 ProtocolKey")
        if not isinstance(
                self.budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
            raise TypeError("P2-C channel intent budget 类型错误")

    def integer_stream(self) -> tuple[int, ...]:
        """返回可封存为单 record P0 的完整启动身份。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_CHANNEL_INTENT_SCHEMA]
        pack_key(result, tuple(ord(item) for item in self.logical_stage_name))
        result.append(self.channel)
        for value in (
                self.training_b3_stable_key,
                self.held_out_b3_stable_key,
                self.training_input.integer_stream(),
                self.held_out_input.integer_stream(),
                self.code_identity.components,
                self.budget.integer_stream()):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 output root 的完整 channel intent identity。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult:
    """一个零交集 channel 的输入、精确去重统计与 sealed output。"""

    logical_stage_name: str
    channel: int
    training_input: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor
    held_out_input: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor
    intent: ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor
    output: ConversationHeldOutV4ProvenanceCrossSideMergeStreamDescriptor
    training_input_record_count: int
    held_out_input_record_count: int
    training_exact_duplicate_count: int
    held_out_exact_duplicate_count: int
    emitted_record_count: int
    materialized_physical_bytes: int
    budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget

    def __post_init__(self) -> None:
        """绑定完整 input descriptor、输出和每侧 exact duplicate 统计。"""
        _require_stage_name(self.logical_stage_name)
        if type(self.channel) is not int or self.channel not in _CHANNELS:
            raise ValueError("P2-C channel result channel 未注册")
        for label, descriptor in (
                ("training_input", self.training_input),
                ("held_out_input", self.held_out_input)):
            if not isinstance(descriptor,
                              ConversationHeldOutV4ProvenanceProjectionStreamDescriptor):
                raise TypeError(f"P2-C {label} 必须是 B3 projection descriptor")
            if descriptor.stream_kind != self.channel:
                raise ValueError(f"P2-C {label} channel 与 result 不一致")
        if not isinstance(self.output,
                          ConversationHeldOutV4ProvenanceCrossSideMergeStreamDescriptor):
            raise TypeError("P2-C channel result output 类型错误")
        if self.output.channel != self.channel:
            raise ValueError("P2-C channel result output channel 不一致")
        if not isinstance(
                self.intent,
                ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor):
            raise TypeError("P2-C channel result intent 类型错误")
        expected_intent_path = (
            Path(_CHANNEL_STAGE_NAMES[self.channel]) / _INTENT_STREAM_FILE)
        if (self.intent.artifact_kind
                    != V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CHANNEL_INTENT
                or self.intent.output_relative_path != expected_intent_path
                or self.intent.p0_footer.record_count != 1):
            raise ValueError("P2-C channel result intent descriptor 不闭合")
        for label, value in (
                ("training_input_record_count", self.training_input_record_count),
                ("held_out_input_record_count", self.held_out_input_record_count),
                ("training_exact_duplicate_count",
                 self.training_exact_duplicate_count),
                ("held_out_exact_duplicate_count",
                 self.held_out_exact_duplicate_count),
                ("emitted_record_count", self.emitted_record_count),
                ("materialized_physical_bytes",
                 self.materialized_physical_bytes)):
            _require_nonnegative_int(value, label=f"P2-C {label}")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
            raise TypeError("P2-C channel result budget 类型错误")
        if (self.training_input_record_count
                != self.training_input.p0_footer.record_count):
            raise ValueError("P2-C training input count 与 footer 不一致")
        if (self.held_out_input_record_count
                != self.held_out_input.p0_footer.record_count):
            raise ValueError("P2-C held-out input count 与 footer 不一致")
        if self.training_exact_duplicate_count > self.training_input_record_count:
            raise ValueError("P2-C training duplicate count 超过输入")
        if self.held_out_exact_duplicate_count > self.held_out_input_record_count:
            raise ValueError("P2-C held-out duplicate count 超过输入")
        expected_emitted = (
            self.training_input_record_count - self.training_exact_duplicate_count
            + self.held_out_input_record_count
            - self.held_out_exact_duplicate_count)
        if self.emitted_record_count != expected_emitted:
            raise ValueError("P2-C emitted count 与 input/duplicate 统计不一致")
        if self.emitted_record_count != self.output.p0_footer.record_count:
            raise ValueError("P2-C emitted count 与 output footer 不一致")
        if self.materialized_physical_bytes != (
                self.intent.physical.byte_count + self.output.physical.byte_count):
            raise ValueError("P2-C materialized bytes 必须等于 intent/output physical 之和")
        if self.materialized_physical_bytes > self.budget.max_total_materialized_physical_bytes:
            raise ValueError("P2-C channel materialized physical 超过总预算")

    def integer_stream(self) -> tuple[int, ...]:
        """返回绑定 input/output/统计/预算的完整 channel identity。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_CHANNEL_RESULT_SCHEMA]
        pack_key(result, tuple(ord(item) for item in self.logical_stage_name))
        result.append(self.channel)
        for value in (
                self.training_input.integer_stream(),
                self.held_out_input.integer_stream(),
                self.intent.integer_stream(),
                self.output.integer_stream()):
            pack_key(result, value)
        result.extend((
            self.training_input_record_count,
            self.held_out_input_record_count,
            self.training_exact_duplicate_count,
            self.held_out_exact_duplicate_count,
            self.emitted_record_count,
            self.materialized_physical_bytes,
        ))
        pack_key(result, self.budget.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回供后续 cursor/receipt 精确绑定的 channel key。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor:
    """P2-C intent、cursor、receipt 等 sealed P0 artifact 的物理身份。"""

    artifact_kind: int
    output_relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """固定 artifact 类型、单层 stage 定位和完整 P0/physical identity。"""
        if self.artifact_kind not in {
                V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CURSOR,
                V4_PROVENANCE_CROSS_SIDE_ARTIFACT_RECEIPT,
                V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CHANNEL_INTENT}:
            raise ValueError("P2-C artifact_kind 未注册")
        path_key = _relative_path_key(
            self.output_relative_path,
            label="P2-C artifact output_relative_path",
        )
        if len(self.output_relative_path.parts) != 2 or not path_key:
            raise ValueError("P2-C artifact 必须定位到单层 stage 文件")
        if not isinstance(self.p0_footer, IntegerFramedStreamFooter):
            raise TypeError("P2-C artifact p0_footer 类型错误")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("P2-C artifact physical 类型错误")
        if self.physical.byte_count <= 0:
            raise ValueError("P2-C artifact sealed P0 physical bytes 必须为正")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 output root 的完整 artifact descriptor。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_ARTIFACT_DESCRIPTOR_SCHEMA,
                  self.artifact_kind]
        for value in (
                _relative_path_key(
                    self.output_relative_path,
                    label="P2-C artifact output_relative_path"),
                self.p0_footer.integer_tuple(),
                _digest_key(self.physical)):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 cursor artifact 的完整稳定 identity。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeCursor:
    """已封存 channel checkpoint 的纯整数状态，不含绝对 output root。"""

    logical_stage_name: str
    training_b3_stable_key: tuple[int, ...]
    held_out_b3_stable_key: tuple[int, ...]
    code_identity: ProtocolKey
    budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget
    completed_channel_results: tuple[
        ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult,
        ...,
    ]

    def __post_init__(self) -> None:
        """要求完成通道是 SourceRef/content/lineage 的无缺前缀。"""
        _require_stage_name(self.logical_stage_name)
        for label, value in (
                ("training_b3_stable_key", self.training_b3_stable_key),
                ("held_out_b3_stable_key", self.held_out_b3_stable_key)):
            try:
                strict_integer_tuple(value, label=f"P2-C cursor {label}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"P2-C cursor {label} 必须是严格整数 tuple") from exc
        if not isinstance(self.code_identity, ProtocolKey):
            raise TypeError("P2-C cursor code_identity 必须是 ProtocolKey")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
            raise TypeError("P2-C cursor budget 类型错误")
        if (not isinstance(self.completed_channel_results, tuple)
                or any(not isinstance(
                    item, ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult)
                    for item in self.completed_channel_results)):
            raise TypeError("P2-C cursor completed_channel_results 类型错误")
        if len(self.completed_channel_results) > len(_CHANNELS):
            raise ValueError("P2-C cursor completed channel 数超过固定通道数")
        for index, result in enumerate(self.completed_channel_results):
            if result.channel != _CHANNELS[index]:
                raise ValueError("P2-C cursor completed channel 顺序或前缀不一致")
            if result.logical_stage_name != self.logical_stage_name:
                raise ValueError("P2-C cursor channel logical_stage_name 不一致")
            if result.budget != self.budget:
                raise ValueError("P2-C cursor channel budget 不一致")

    @property
    def is_complete(self) -> bool:
        """仅在三个固定通道均有 sealed result 后返回真。"""
        return len(self.completed_channel_results) == len(_CHANNELS)

    def integer_stream(self) -> tuple[int, ...]:
        """返回可写为单 record P0 cursor 的完整状态，不依赖 JSON。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_CURSOR_SCHEMA]
        pack_key(result, tuple(ord(item) for item in self.logical_stage_name))
        pack_key(result, self.training_b3_stable_key)
        pack_key(result, self.held_out_b3_stable_key)
        pack_key(result, self.code_identity.components)
        pack_key(result, self.budget.integer_stream())
        result.append(len(self.completed_channel_results))
        for item in self.completed_channel_results:
            pack_key(result, item.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 resume 可精确比对的完整 cursor state key。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact:
    """在 output run 内封存的 cursor state 与其 P0 descriptor。"""

    cursor: ConversationHeldOutV4ProvenanceCrossSideMergeCursor
    descriptor: ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor

    def __post_init__(self) -> None:
        """要求 cursor artifact 精确是一条 P0 state record。"""
        if not isinstance(self.cursor, ConversationHeldOutV4ProvenanceCrossSideMergeCursor):
            raise TypeError("P2-C cursor artifact cursor 类型错误")
        if not isinstance(self.descriptor,
                          ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor):
            raise TypeError("P2-C cursor artifact descriptor 类型错误")
        if self.descriptor.artifact_kind != V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CURSOR:
            raise ValueError("P2-C cursor artifact kind 不一致")
        if self.descriptor.p0_footer.record_count != 1:
            raise ValueError("P2-C cursor artifact 必须恰含一条 state record")

    def stable_key(self) -> tuple[int, ...]:
        """返回 state 与 physical descriptor 均不遗漏的 cursor identity。"""
        result = list(self.cursor.integer_stream())
        pack_key(result, self.descriptor.integer_stream())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor:
    """manifest/receipt 闭包中的一个普通文件相对路径与完整 physical identity。"""

    relative_path: Path
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """拒绝绝对/逃逸路径、非普通 physical digest 和空 closure 项。"""
        _relative_path_key(self.relative_path, label="P2-C file descriptor relative_path")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("P2-C file descriptor physical 类型错误")
        if self.physical.byte_count <= 0:
            raise ValueError("P2-C file descriptor physical bytes 必须为正")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 K output root 的完整 file closure 项。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_FILE_DESCRIPTOR_SCHEMA]
        pack_key(result, _relative_path_key(
            self.relative_path, label="P2-C file descriptor relative_path"))
        pack_key(result, _digest_key(self.physical))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回资源回执/manifest 可精确绑定的文件 identity。"""
        return self.integer_stream()


def _validate_sorted_file_descriptors(
        value: tuple[ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor,
                     ...],
        *, label: str,
        ) -> int:
    """要求 closure 文件按相对路径严格排序并返回有限累计 physical bytes。"""
    if (not isinstance(value, tuple)
            or not value
            or any(not isinstance(
                item, ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor)
                   for item in value)):
        raise TypeError(f"{label} 必须是非空 FileDescriptor tuple")
    previous: tuple[int, ...] | None = None
    total = 0
    for item in value:
        key = _relative_path_key(item.relative_path, label=f"{label} path")
        if previous is not None and key <= previous:
            raise ValueError(f"{label} 必须按 path 严格递增")
        previous = key
        total += item.physical.byte_count
    return total


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeResourceReceipt:
    """三路零交集后的纯整数资源回执状态，尚不包含自身 P0 physical descriptor。"""

    logical_stage_name: str
    training_b3_stable_key: tuple[int, ...]
    held_out_b3_stable_key: tuple[int, ...]
    code_identity: ProtocolKey
    budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget
    final_cursor_stable_key: tuple[int, ...]
    channel_results: tuple[
        ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult,
        ...,
    ]
    pre_receipt_files: tuple[
        ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor,
        ...,
    ]
    pre_receipt_physical_bytes: int
    temporary_physical_bytes: int
    unadopted_residue_physical_bytes: int
    max_open_files_observed: int
    status: int

    def __post_init__(self) -> None:
        """绑定三路成功 checkpoint、精确闭包与零 residue 的机械状态。"""
        _require_stage_name(self.logical_stage_name)
        for label, value in (
                ("training_b3_stable_key", self.training_b3_stable_key),
                ("held_out_b3_stable_key", self.held_out_b3_stable_key),
                ("final_cursor_stable_key", self.final_cursor_stable_key)):
            try:
                strict_integer_tuple(value, label=f"P2-C receipt {label}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"P2-C receipt {label} 必须是严格整数 tuple") from exc
        if not isinstance(self.code_identity, ProtocolKey):
            raise TypeError("P2-C receipt code_identity 必须是 ProtocolKey")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
            raise TypeError("P2-C receipt budget 类型错误")
        if (not isinstance(self.channel_results, tuple)
                or len(self.channel_results) != len(_CHANNELS)
                or any(not isinstance(
                    item, ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult)
                    for item in self.channel_results)):
            raise TypeError("P2-C receipt 必须绑定全部三条 channel result")
        for index, item in enumerate(self.channel_results):
            if (item.channel != _CHANNELS[index]
                    or item.logical_stage_name != self.logical_stage_name
                    or item.budget != self.budget):
                raise ValueError("P2-C receipt channel result 与冻结状态不一致")
        if len(self.pre_receipt_files) != len(_CHANNELS) * 3:
            raise ValueError("P2-C receipt pre-receipt closure 必须恰含三 intent、三 output 与三 cursor")
        if len(self.pre_receipt_files) + 1 > self.budget.max_pre_manifest_file_count:
            raise ValueError("P2-C receipt 及 manifest closure 超过文件数预算")
        closure_bytes = _validate_sorted_file_descriptors(
            self.pre_receipt_files, label="P2-C receipt pre_receipt_files")
        expected_paths = frozenset(
            [Path(_CHANNEL_STAGE_NAMES[channel]) / _INTENT_STREAM_FILE
             for channel in _CHANNELS]
            + [Path(_CHANNEL_STAGE_NAMES[channel]) / _MERGED_STREAM_FILE
             for channel in _CHANNELS]
            + [Path(_cursor_stage_name(index + 1)) / _CURSOR_STREAM_FILE
               for index in range(len(_CHANNELS))]
        )
        actual_paths = frozenset(
            item.relative_path for item in self.pre_receipt_files)
        if actual_paths != expected_paths:
            raise ValueError("P2-C receipt pre-receipt closure 路径不是规范 intent/output/cursor 集合")
        for item in self.channel_results:
            expected_intent = (Path(_CHANNEL_STAGE_NAMES[item.channel])
                               / _INTENT_STREAM_FILE)
            expected_output = (Path(_CHANNEL_STAGE_NAMES[item.channel])
                               / _MERGED_STREAM_FILE)
            if item.intent.output_relative_path != expected_intent:
                raise ValueError("P2-C receipt channel intent 路径不是规范 stage")
            if item.output.output_relative_path != expected_output:
                raise ValueError("P2-C receipt channel output 路径不是规范 stage")
            matched_intent = next(
                candidate for candidate in self.pre_receipt_files
                if candidate.relative_path == expected_intent)
            matched = next(
                candidate for candidate in self.pre_receipt_files
                if candidate.relative_path == expected_output)
            if matched_intent.physical != item.intent.physical:
                raise ValueError("P2-C receipt intent physical 与 channel result 不一致")
            if matched.physical != item.output.physical:
                raise ValueError("P2-C receipt output physical 与 channel result 不一致")
        for label, value in (
                ("pre_receipt_physical_bytes", self.pre_receipt_physical_bytes),
                ("temporary_physical_bytes", self.temporary_physical_bytes),
                ("unadopted_residue_physical_bytes",
                 self.unadopted_residue_physical_bytes),
                ("max_open_files_observed", self.max_open_files_observed)):
            _require_nonnegative_int(value, label=f"P2-C receipt {label}")
        if self.pre_receipt_physical_bytes != closure_bytes:
            raise ValueError("P2-C receipt pre-receipt physical bytes 与 closure 不一致")
        if self.unadopted_residue_physical_bytes != 0:
            raise ValueError("P2-C receipt 不允许未采用残片")
        if (self.pre_receipt_physical_bytes
                > self.budget.max_total_materialized_physical_bytes):
            raise ValueError("P2-C receipt pre-receipt physical 超过总预算")
        if (self.max_open_files_observed <= 0
                or self.max_open_files_observed > self.budget.max_open_files):
            raise ValueError("P2-C receipt max_open_files_observed 超出预算")
        if self.status != V4_PROVENANCE_CROSS_SIDE_RECEIPT_STATUS_ZERO_INTERSECTION_MECHANICAL:
            raise ValueError("P2-C receipt status 只能是零交集机械状态")

    def integer_stream(self) -> tuple[int, ...]:
        """返回 resource receipt 的完整单 record P0 载荷。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_RESOURCE_RECEIPT_SCHEMA]
        pack_key(result, tuple(ord(item) for item in self.logical_stage_name))
        for value in (
                self.training_b3_stable_key,
                self.held_out_b3_stable_key,
                self.code_identity.components,
                self.budget.integer_stream(),
                self.final_cursor_stable_key):
            pack_key(result, value)
        result.append(len(self.channel_results))
        for item in self.channel_results:
            pack_key(result, item.integer_stream())
        result.append(len(self.pre_receipt_files))
        for item in self.pre_receipt_files:
            pack_key(result, item.integer_stream())
        result.extend((
            self.pre_receipt_physical_bytes,
            self.temporary_physical_bytes,
            self.unadopted_residue_physical_bytes,
            self.max_open_files_observed,
            self.status,
        ))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 manifest 可精确引用的 resource receipt state identity。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeReceiptArtifact:
    """封存 resource receipt state 与其单 record P0 descriptor。"""

    receipt: ConversationHeldOutV4ProvenanceCrossSideMergeResourceReceipt
    descriptor: ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor

    def __post_init__(self) -> None:
        """receipt 不允许以 cursor kind、空 P0 或多 record artifact 表示。"""
        if not isinstance(self.receipt,
                          ConversationHeldOutV4ProvenanceCrossSideMergeResourceReceipt):
            raise TypeError("P2-C receipt artifact receipt 类型错误")
        if not isinstance(self.descriptor,
                          ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor):
            raise TypeError("P2-C receipt artifact descriptor 类型错误")
        if self.descriptor.artifact_kind != V4_PROVENANCE_CROSS_SIDE_ARTIFACT_RECEIPT:
            raise ValueError("P2-C receipt artifact kind 不一致")
        if self.descriptor.p0_footer.record_count != 1:
            raise ValueError("P2-C receipt artifact 必须恰含一条 receipt record")

    def stable_key(self) -> tuple[int, ...]:
        """返回 receipt state/physical descriptor 均被绑定的完整 identity。"""
        result = list(self.receipt.integer_stream())
        pack_key(result, self.descriptor.integer_stream())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergeManifest:
    """manifest-last 的规范整数 payload，不使用 JSON 或绝对路径。"""

    logical_stage_name: str
    training_b3_stable_key: tuple[int, ...]
    held_out_b3_stable_key: tuple[int, ...]
    receipt_stable_key: tuple[int, ...]
    final_cursor_stable_key: tuple[int, ...]
    code_identity: ProtocolKey
    budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget
    pre_manifest_files: tuple[
        ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor,
        ...,
    ]

    def __post_init__(self) -> None:
        """绑定 receipt/cursor 与含 receipt 的完整 pre-manifest file closure。"""
        _require_stage_name(self.logical_stage_name)
        for label, value in (
                ("training_b3_stable_key", self.training_b3_stable_key),
                ("held_out_b3_stable_key", self.held_out_b3_stable_key),
                ("receipt_stable_key", self.receipt_stable_key),
                ("final_cursor_stable_key", self.final_cursor_stable_key)):
            try:
                strict_integer_tuple(value, label=f"P2-C manifest {label}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"P2-C manifest {label} 必须是严格整数 tuple") from exc
        if not isinstance(self.code_identity, ProtocolKey):
            raise TypeError("P2-C manifest code_identity 必须是 ProtocolKey")
        if not isinstance(self.budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
            raise TypeError("P2-C manifest budget 类型错误")
        if len(self.pre_manifest_files) != len(_CHANNELS) * 3 + 1:
            raise ValueError("P2-C manifest closure 必须含三 intent、三 output、三 cursor 和 receipt")
        if len(self.pre_manifest_files) > self.budget.max_pre_manifest_file_count:
            raise ValueError("P2-C manifest closure 超过文件数预算")
        closure_bytes = _validate_sorted_file_descriptors(
            self.pre_manifest_files, label="P2-C manifest pre_manifest_files")
        if closure_bytes > self.budget.max_total_materialized_physical_bytes:
            raise ValueError("P2-C manifest pre-manifest physical 超过总预算")
        expected_paths = frozenset(
            [Path(_CHANNEL_STAGE_NAMES[channel]) / _INTENT_STREAM_FILE
             for channel in _CHANNELS]
            + [Path(_CHANNEL_STAGE_NAMES[channel]) / _MERGED_STREAM_FILE
             for channel in _CHANNELS]
            + [Path(_cursor_stage_name(index + 1)) / _CURSOR_STREAM_FILE
               for index in range(len(_CHANNELS))]
            + [Path(_RECEIPT_STAGE_NAME) / _RECEIPT_STREAM_FILE]
        )
        if frozenset(item.relative_path for item in self.pre_manifest_files) != expected_paths:
            raise ValueError("P2-C manifest closure 路径不是规范 intent/output/cursor/receipt 集合")

    def integer_stream(self) -> tuple[int, ...]:
        """返回给 manifest-last 的有界规范整数 payload。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_MANIFEST_SCHEMA]
        pack_key(result, tuple(ord(item) for item in self.logical_stage_name))
        for value in (
                self.training_b3_stable_key,
                self.held_out_b3_stable_key,
                self.receipt_stable_key,
                self.final_cursor_stable_key,
                self.code_identity.components,
                self.budget.integer_stream()):
            pack_key(result, value)
        result.append(len(self.pre_manifest_files))
        for item in self.pre_manifest_files:
            pack_key(result, item.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 publication result 可引用的完整 manifest identity。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult:
    """P2-C 最终机械 publication；不表达训练资格或来源独立结论。"""

    receipt_artifact: ConversationHeldOutV4ProvenanceCrossSideMergeReceiptArtifact
    manifest: ConversationHeldOutV4ProvenanceCrossSideMergeManifest
    manifest_file: ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor
    total_published_physical_bytes: int

    def __post_init__(self) -> None:
        """固定 receipt、manifest 与最终物理累计，不允许缺任何 publication 项。"""
        if not isinstance(self.receipt_artifact,
                          ConversationHeldOutV4ProvenanceCrossSideMergeReceiptArtifact):
            raise TypeError("P2-C publication receipt_artifact 类型错误")
        if not isinstance(self.manifest,
                          ConversationHeldOutV4ProvenanceCrossSideMergeManifest):
            raise TypeError("P2-C publication manifest 类型错误")
        if not isinstance(self.manifest_file,
                          ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor):
            raise TypeError("P2-C publication manifest_file 类型错误")
        if self.manifest_file.relative_path != _MANIFEST_FILE:
            raise ValueError("P2-C publication manifest 文件名不一致")
        receipt = self.receipt_artifact.receipt
        if (self.manifest.logical_stage_name != receipt.logical_stage_name
                or self.manifest.training_b3_stable_key
                != receipt.training_b3_stable_key
                or self.manifest.held_out_b3_stable_key
                != receipt.held_out_b3_stable_key
                or self.manifest.code_identity != receipt.code_identity
                or self.manifest.budget != receipt.budget):
            raise ValueError("P2-C publication manifest 与 receipt 冻结输入不一致")
        if self.manifest.receipt_stable_key != self.receipt_artifact.stable_key():
            raise ValueError("P2-C publication manifest 未绑定 receipt artifact identity")
        if self.manifest.final_cursor_stable_key != receipt.final_cursor_stable_key:
            raise ValueError("P2-C publication manifest 未绑定最终 cursor identity")
        _require_nonnegative_int(
            self.total_published_physical_bytes,
            label="P2-C total_published_physical_bytes",
        )
        expected = sum(
            item.physical.byte_count for item in self.manifest.pre_manifest_files)
        if self.total_published_physical_bytes != expected + self.manifest_file.physical.byte_count:
            raise ValueError("P2-C publication total physical bytes 与 manifest closure 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 output root 的完整最终 publication identity。"""
        result = [V4_PROVENANCE_CROSS_SIDE_MERGE_PUBLICATION_RESULT_SCHEMA]
        pack_key(result, self.receipt_artifact.stable_key())
        pack_key(result, self.manifest.stable_key())
        pack_key(result, self.manifest_file.integer_stream())
        result.append(self.total_published_physical_bytes)
        return tuple(result)


# object-model: resource_owner; representation=protocol; interop=pending
class _DigestingBinaryHandle:
    """在单一 P0 handle 生命周期中计算有限 physical SHA-256。"""

    __slots__ = ("_byte_count", "_digest", "_max_bytes", "_stream")

    def __init__(self, stream: BinaryIO, *, max_bytes: int) -> None:
        """接管已经 capability 校验的 handle，不缓存整文件。"""
        self._byte_count = 0
        self._digest = hashlib.sha256()
        self._max_bytes = _require_positive_int(
            max_bytes, label="P2-C physical max_bytes")
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        """在交给 P0 前累计实际读取字节，超物理预算立即拒绝。"""
        if type(size) is not int:
            raise TypeError("P2-C physical read size 必须是严格整数")
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
        """只把底层确认写出的前缀纳入物理 identity。"""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("P2-C physical write data 必须是 bytes-like")
        if len(data) > self._max_bytes - self._byte_count:
            _budget_fail("P2-C P0 stream physical bytes 超过预算")
        written = self._stream.write(data)
        if type(written) is int and 0 < written <= len(data):
            self._accept(bytes(memoryview(data)[:written]))
        return written

    def flush(self) -> None:
        """透传 P0 seal 所需 flush。"""
        self._stream.flush()

    def close(self) -> None:
        """关闭唯一底层 handle。"""
        self._stream.close()

    def physical(self) -> KRunFileDigest:
        """返回同一实际 handle 完整消费的有限物理 identity。"""
        return KRunFileDigest(self._byte_count, tuple(self._digest.digest()))

    def _accept(self, value: bytes) -> None:
        """拒绝越过容量的成功 I/O，避免继续解析超预算内容。"""
        if len(value) > self._max_bytes - self._byte_count:
            _budget_fail("P2-C P0 stream physical bytes 超过预算")
        self._digest.update(value)
        self._byte_count += len(value)


# object-model: resource_owner; representation=protocol; interop=pending
class _VerifiedP0Reader:
    """完整消费一个 B3 descriptor，并在 finish 时重验 P0/physical/path identity。"""

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
            budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
            label: str,
            ) -> None:
        """固定 capture/open/post-read 三段 identity，不裸路径重开。"""
        if physical.byte_count > budget.max_stream_physical_bytes:
            _budget_fail(f"{label} physical bytes 超过 P2-C stream 预算")
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
        except (IntegerCodecError, OSError) as exc:
            try:
                self._handle.close()
            except OSError:
                pass
            _p0_read_fail(exc, label=label)
            raise AssertionError from exc
        except BaseException:
            try:
                self._handle.close()
            except OSError:
                pass
            raise

    def __enter__(self) -> "_VerifiedP0Reader":
        """返回唯一 P0 cursor；正常路径必须调用 finish。"""
        return self

    def __exit__(self, exc_type: object, exc_value: object,
                 traceback: object) -> None:
        """异常路径只释放尚未可信的输入 handle。"""
        self.close()

    def read_record(self) -> tuple[int, ...] | None:
        """读取下一条 P0 record，不预读或聚合后续记录。"""
        try:
            return self._reader.read_record()
        except (IntegerCodecError, OSError) as exc:
            _p0_read_fail(exc, label="P2-C verified P0 reader")
            raise AssertionError from exc

    def finish(self) -> IntegerFramedStreamFooter:
        """完整读尽后比较 footer、physical SHA 与读后路径 identity。"""
        if self._finished:
            _fail("P2-C verified P0 reader 不得重复 finish")
        try:
            footer = self._reader.finish()
        except (IntegerCodecError, OSError) as exc:
            _p0_read_fail(exc, label="P2-C verified P0 reader")
            raise AssertionError from exc
        physical = self._handle.physical()
        require_plain_file_identity(
            self._root,
            self._relative,
            self._file_identity,
            label="P2-C P0 post-read",
        )
        if footer != self._expected_footer:
            _fail("P2-C P0 footer 与冻结 descriptor 漂移")
        if physical != self._expected_physical:
            _fail("P2-C P0 physical bytes 或 SHA-256 与冻结 descriptor 漂移")
        self._finished = True
        return footer

    def close(self) -> None:
        """关闭 P0 reader；成功 finish 后保持幂等。"""
        self._reader.close()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _DecodedProjectionRecord:
    """当前 P0 record 的规范 tuple、primary key 与完整 B3 sort key。"""

    record: tuple[int, ...]
    primary_key: tuple[int, ...]
    full_sort_key: tuple[int, ...]


# object-model: resource_owner; representation=protocol; interop=pending
class _ProjectionInputCursor:
    """验证单侧 B3 projection 排序、side 与 exact duplicates 的流式 cursor。"""

    __slots__ = (
        "_budget", "_channel", "_expected_side", "_finished", "_previous_key",
        "_reader", "duplicate_count", "record_count",
    )

    def __init__(
            self,
            root: KRunRoot,
            descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
            *,
            channel: int,
            expected_side: int,
            budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
            label: str,
            ) -> None:
        """只打开 descriptor 所在单层 B3 stage，并固定一侧的允许 side。"""
        if descriptor.stream_kind != channel:
            _fail(f"{label} B3 descriptor channel 错位")
        stage_root, relative = _stage_local_input(root, descriptor, label=label)
        self._budget = budget
        self._channel = channel
        self._expected_side = expected_side
        self._finished = False
        self._previous_key: tuple[int, ...] | None = None
        self._reader = _VerifiedP0Reader(
            stage_root,
            relative,
            footer=descriptor.p0_footer,
            physical=descriptor.physical,
            budget=budget,
            label=label,
        )
        self.duplicate_count = 0
        self.record_count = 0

    def __enter__(self) -> "_ProjectionInputCursor":
        """返回当前侧唯一的有界 input cursor。"""
        return self

    def __exit__(self, exc_type: object, exc_value: object,
                 traceback: object) -> None:
        """异常路径关闭尚未完全验证的 reader。"""
        self.close()

    def next_unique(self) -> _DecodedProjectionRecord | None:
        """读到下一条完整唯一 record，连续 exact duplicate 只计数不返回。"""
        while True:
            record = self._reader.read_record()
            if record is None:
                self._reader.finish()
                self._finished = True
                return None
            decoded = _decode_projection_record(
                self._channel,
                record,
                expected_side=self._expected_side,
            )
            self.record_count += 1
            if self.record_count > self._budget.max_stream_record_count:
                _budget_fail("P2-C input record 数超过 stream 预算")
            previous = self._previous_key
            if previous is not None:
                if decoded.full_sort_key < previous:
                    _fail("P2-C B3 projection sort key 倒序")
                if decoded.full_sort_key == previous:
                    self.duplicate_count += 1
                    continue
            self._previous_key = decoded.full_sort_key
            return decoded

    def close(self) -> None:
        """关闭输入 reader；未 finish 的异常路径不产生可信 identity。"""
        self._reader.close()


def _stage_local_input(
        work_root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        *, label: str,
        ) -> tuple[KRunRoot, Path]:
    """从 B3 root-relative descriptor 重新取得单层 stage capability。"""
    relative = descriptor.work_relative_path
    _relative_path_key(relative, label=f"{label} descriptor path")
    if len(relative.parts) != 2:
        _fail(f"{label} B3 descriptor 必须定位到单层 stage sibling")
    stage, filename = relative.parts
    stage_root = open_existing_run_root(
        work_root.path / stage,
        require_k_drive=not work_root.test_transport,
        label=f"{label} B3 source stage",
    )
    return stage_root, Path(filename)


def _verify_projection_descriptor(
        root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> None:
    """完整读取一个未解释 descriptor，先固定 physical identity 再允许写 output。"""
    stage_root, relative = _stage_local_input(root, descriptor, label=label)
    with _VerifiedP0Reader(
            stage_root,
            relative,
            footer=descriptor.p0_footer,
            physical=descriptor.physical,
            budget=budget,
            label=label,
    ) as reader:
        reader.finish()


def _decode_projection_record(
        channel: int,
        record: tuple[int, ...],
        *, expected_side: int,
        ) -> _DecodedProjectionRecord:
    """严格 decode/re-encode B3 record，得到完整 primary 与 B3 sort key。"""
    try:
        strict_integer_tuple(record, label="P2-C B3 projection record")
        if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF:
            source_ref, side, leaf = (
                decode_v4_provenance_source_ref_projection_record(record))
            canonical = encode_v4_provenance_source_ref_projection_record(
                source_ref, side, leaf)
            primary_key = source_ref.stable_key()
            full_sort_key = _packed_sort_key(
                primary_key, (side,), leaf.integer_stream())
        elif channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT:
            content_sha256, side, leaf = (
                decode_v4_provenance_content_projection_record(record))
            canonical = encode_v4_provenance_content_projection_record(
                content_sha256, side, leaf)
            primary_key = content_sha256
            full_sort_key = _packed_sort_key(
                primary_key, (side,), leaf.integer_stream())
        elif channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE:
            namespace, ancestor, snapshot, side, leaf = (
                decode_v4_provenance_lineage_projection_record(record))
            canonical = encode_v4_provenance_lineage_projection_record(
                namespace, ancestor, snapshot, side, leaf)
            primary_key = _packed_sort_key(
                (namespace,),
                ancestor.components,
                snapshot.integer_stream(),
            )
            full_sort_key = _packed_sort_key(
                primary_key, (side,), leaf.integer_stream())
        else:
            raise ValueError("P2-C projection channel 未注册")
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
            TypeError, ValueError) as exc:
        _fail("P2-C B3 projection record 非法或不闭合")
        raise AssertionError from exc
    if side != expected_side:
        _fail("P2-C B3 projection side 与输入侧不一致")
    if record != canonical:
        _fail("P2-C B3 projection record 不是规范完整编码")
    return _DecodedProjectionRecord(record, primary_key, full_sort_key)


def _new_channel_stage(output_run_root: KRunRoot, channel: int) -> tuple[KRunRoot, str]:
    """排他创建此前不存在的 P2-C channel stage。"""
    stage = _CHANNEL_STAGE_NAMES[channel]
    result = create_new_run_root(
        output_run_root.path / stage,
        require_k_drive=not output_run_root.test_transport,
        label=f"P2-C {stage}",
    )
    require_fresh_empty_run_root(result, label=f"P2-C {stage}")
    return result, stage


def _verify_merged_output(
        output_stage_root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceCrossSideMergeStreamDescriptor,
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> None:
    """对新写的 sealed output 回读，固定 footer/physical/path identity。"""
    if len(descriptor.output_relative_path.parts) != 2:
        _fail("P2-C merged output descriptor 必须定位到单层 stage")
    _stage, filename = descriptor.output_relative_path.parts
    with _VerifiedP0Reader(
            output_stage_root,
            Path(filename),
            footer=descriptor.p0_footer,
            physical=descriptor.physical,
            budget=budget,
            label=label,
    ) as reader:
        reader.finish()


def _build_v4_provenance_cross_side_merge_channel(
        training_work_root: KRunRoot,
        training_descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        held_out_work_root: KRunRoot,
        held_out_descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        output_run_root: KRunRoot,
        *,
        channel: int,
        training_b3_stable_key: tuple[int, ...],
        held_out_b3_stable_key: tuple[int, ...],
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        allow_reopened_roots: bool = False,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult:
    """在已验证 caller 内对一个 B3 projection channel 作流式 cross-side exact merge。

    输入先完整 revalidate，随后只持有双方当前 record。相同 primary key 跨侧出现即
    失败；同侧连续的完全相同 B3 record 只计数一次。每个 output 前先封存完整冻结
    intent，因而孤儿 output 不会丢失 code、B3 identity、预算或 stage 身份。本函数不
    发布 cursor、receipt 或 manifest，调用方必须在三路全部成功后才进入这些阶段。
    """
    if not isinstance(training_work_root, KRunRoot):
        raise TypeError("P2-C training_work_root 必须是 KRunRoot")
    if not isinstance(held_out_work_root, KRunRoot):
        raise TypeError("P2-C held_out_work_root 必须是 KRunRoot")
    if not isinstance(output_run_root, KRunRoot):
        raise TypeError("P2-C output_run_root 必须是 KRunRoot")
    if not isinstance(training_descriptor,
                      ConversationHeldOutV4ProvenanceProjectionStreamDescriptor):
        raise TypeError("P2-C training_descriptor 类型错误")
    if not isinstance(held_out_descriptor,
                      ConversationHeldOutV4ProvenanceProjectionStreamDescriptor):
        raise TypeError("P2-C held_out_descriptor 类型错误")
    if type(channel) is not int or channel not in _CHANNELS:
        raise ValueError("P2-C channel 未注册")
    if not isinstance(code_identity, ProtocolKey):
        raise TypeError("P2-C code_identity 必须是 ProtocolKey")
    if not isinstance(budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
        raise TypeError("P2-C budget 类型错误")
    if type(allow_reopened_roots) is not bool:
        raise TypeError("P2-C allow_reopened_roots 必须是 bool")
    _require_stage_name(logical_stage_name)
    if training_descriptor.stream_kind != channel:
        _fail("P2-C training descriptor channel 错位")
    if held_out_descriptor.stream_kind != channel:
        _fail("P2-C held-out descriptor channel 错位")
    intent = ConversationHeldOutV4ProvenanceCrossSideMergeChannelIntent(
        logical_stage_name,
        channel,
        training_b3_stable_key,
        held_out_b3_stable_key,
        training_descriptor,
        held_out_descriptor,
        code_identity,
        budget,
    )
    if not allow_reopened_roots:
        require_created_run_root(training_work_root, label="P2-C training_work_root")
        require_created_run_root(held_out_work_root, label="P2-C held_out_work_root")
        require_created_run_root(output_run_root, label="P2-C output_run_root")
    require_disjoint_run_roots(
        training_work_root, held_out_work_root,
        label="P2-C training/held-out work roots")
    require_disjoint_run_roots(
        training_work_root, output_run_root,
        label="P2-C training/output work roots")
    require_disjoint_run_roots(
        held_out_work_root, output_run_root,
        label="P2-C held-out/output work roots")
    if budget.max_open_files < 3:
        _budget_fail("P2-C merge 需要两个 reader 和一个 writer")

    # 所有实际 output 创建前完整重验双方 descriptor，避免只把 SHA 当成可复用快照。
    _verify_projection_descriptor(
        training_work_root,
        training_descriptor,
        budget=budget,
        label="P2-C training projection revalidation",
    )
    _verify_projection_descriptor(
        held_out_work_root,
        held_out_descriptor,
        budget=budget,
        label="P2-C held-out projection revalidation",
    )

    output_stage_root, stage = _new_channel_stage(output_run_root, channel)
    intent_descriptor = _write_single_record_artifact(
        output_run_root,
        stage=stage,
        filename=_INTENT_STREAM_FILE,
        artifact_kind=V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CHANNEL_INTENT,
        record=intent.integer_stream(),
        max_physical_bytes=budget.max_intent_physical_bytes,
        budget=budget,
        label="P2-C channel intent",
        existing_stage_root=output_stage_root,
    )
    stream = open_exclusive_binary(
        output_stage_root,
        _MERGED_STREAM_FILE,
        label="P2-C merged channel output",
    )
    handle = _DigestingBinaryHandle(
        stream,
        max_bytes=budget.max_stream_physical_bytes,
    )
    writer = IntegerFramedStreamWriter.from_open_binary(
        handle,
        path=_MERGED_STREAM_FILE,
    )
    emitted_count = 0
    emitted_payload_bytes = 0
    training_cursor: _ProjectionInputCursor | None = None
    held_out_cursor: _ProjectionInputCursor | None = None
    try:
        training_cursor = _ProjectionInputCursor(
            training_work_root,
            training_descriptor,
            channel=channel,
            expected_side=V4_PROVENANCE_SIDE_TRAINING,
            budget=budget,
            label="P2-C training projection input",
        )
        held_out_cursor = _ProjectionInputCursor(
            held_out_work_root,
            held_out_descriptor,
            channel=channel,
            expected_side=V4_PROVENANCE_SIDE_HELD_OUT,
            budget=budget,
            label="P2-C held-out projection input",
        )
        training_record = training_cursor.next_unique()
        held_out_record = held_out_cursor.next_unique()

        def emit(value: _DecodedProjectionRecord) -> None:
            """受完整 record/payload/stream 预算约束地写一条 canonical B3 record。"""
            nonlocal emitted_count
            nonlocal emitted_payload_bytes
            payload_size = encoded_integer_tuple_size(value.record)
            if payload_size > budget.max_record_payload_bytes:
                _budget_fail("P2-C merged record payload 超过预算")
            if emitted_count >= budget.max_stream_record_count:
                _budget_fail("P2-C merged record 数超过预算")
            if payload_size > budget.max_stream_payload_bytes - emitted_payload_bytes:
                _budget_fail("P2-C merged payload 超过预算")
            writer.append(value.record)
            emitted_count += 1
            emitted_payload_bytes += payload_size

        while training_record is not None or held_out_record is not None:
            if training_record is None:
                emit(held_out_record)
                held_out_record = held_out_cursor.next_unique()
                continue
            if held_out_record is None:
                emit(training_record)
                training_record = training_cursor.next_unique()
                continue
            if training_record.primary_key < held_out_record.primary_key:
                emit(training_record)
                training_record = training_cursor.next_unique()
                continue
            if training_record.primary_key > held_out_record.primary_key:
                emit(held_out_record)
                held_out_record = held_out_cursor.next_unique()
                continue
            raise ConversationHeldOutV4ProvenanceCrossSideOverlapError(
                "P2-C 检测到完整 primary key 的 cross-side overlap")

        footer = writer.seal()
    finally:
        writer.close()
        if training_cursor is not None:
            training_cursor.close()
        if held_out_cursor is not None:
            held_out_cursor.close()

    physical = handle.physical()
    materialized_physical_bytes = (
        intent_descriptor.physical.byte_count + physical.byte_count)
    if materialized_physical_bytes > budget.max_total_materialized_physical_bytes:
        _budget_fail("P2-C channel intent/output 超过累计物化预算")
    identity = capture_plain_file_identity(
        output_stage_root,
        _MERGED_STREAM_FILE,
        label="P2-C merged output identity",
    )
    if (footer.record_count != emitted_count
            or footer.total_payload_bytes != emitted_payload_bytes
            or identity.byte_count != physical.byte_count):
        _fail("P2-C merged output footer 或 physical identity 漂移")
    descriptor = ConversationHeldOutV4ProvenanceCrossSideMergeStreamDescriptor(
        channel,
        Path(stage) / _MERGED_STREAM_FILE,
        footer,
        physical,
    )
    _verify_merged_output(
        output_stage_root,
        descriptor,
        budget=budget,
        label="P2-C merged output verification",
    )
    require_exact_file_closure(
        output_stage_root,
        frozenset({_INTENT_STREAM_FILE, _MERGED_STREAM_FILE}),
        label="P2-C channel stage closure",
    )
    if training_cursor is None or held_out_cursor is None:
        raise AssertionError("P2-C merge 缺 input cursor")
    return ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult(
        logical_stage_name,
        channel,
        training_descriptor,
        held_out_descriptor,
        intent_descriptor,
        descriptor,
        training_cursor.record_count,
        held_out_cursor.record_count,
        training_cursor.duplicate_count,
        held_out_cursor.duplicate_count,
        emitted_count,
        materialized_physical_bytes,
        budget,
    )


def build_v4_provenance_cross_side_merge_channel(
        training_work_root: KRunRoot,
        training_descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        held_out_work_root: KRunRoot,
        held_out_descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        output_run_root: KRunRoot,
        *,
        channel: int,
        training_b3_stable_key: tuple[int, ...],
        held_out_b3_stable_key: tuple[int, ...],
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult:
    """构建单个新 channel，公开入口只接受本进程创建的 capability。

    跨进程重开 capability 只能由已完成完整 prefix 重建的 ``advance`` 私有路径
    使用，避免一般调用方绕过 cursor、intent 与输出闭包的恢复验证。
    """
    return _build_v4_provenance_cross_side_merge_channel(
        training_work_root,
        training_descriptor,
        held_out_work_root,
        held_out_descriptor,
        output_run_root,
        channel=channel,
        training_b3_stable_key=training_b3_stable_key,
        held_out_b3_stable_key=held_out_b3_stable_key,
        code_identity=code_identity,
        budget=budget,
        logical_stage_name=logical_stage_name,
        allow_reopened_roots=False,
    )


def _projection_descriptor_for_channel(
        result: ConversationHeldOutV4ProvenanceProjectionResult,
        channel: int,
        ) -> ConversationHeldOutV4ProvenanceProjectionStreamDescriptor:
    """从 B3 result 按固定通道取 descriptor，不允许调用方重排三路输入。"""
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF:
        return result.source_ref_projection_stream
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT:
        return result.content_projection_stream
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE:
        return result.lineage_projection_stream
    raise ValueError("P2-C projection channel 未注册")


def _expected_channel_intent(
        training_result: ConversationHeldOutV4ProvenanceProjectionResult,
        held_out_result: ConversationHeldOutV4ProvenanceProjectionResult,
        *,
        channel: int,
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeChannelIntent:
    """从当前完整 B3 result 重建唯一允许的 channel 启动 intent。"""
    return ConversationHeldOutV4ProvenanceCrossSideMergeChannelIntent(
        logical_stage_name,
        channel,
        training_result.stable_key(),
        held_out_result.stable_key(),
        _projection_descriptor_for_channel(training_result, channel),
        _projection_descriptor_for_channel(held_out_result, channel),
        code_identity,
        budget,
    )


def _projection_count_for_channel(
        result: ConversationHeldOutV4ProvenanceProjectionResult,
        channel: int,
        ) -> int:
    """返回 B3 result 已冻结的 channel record count。"""
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF:
        return result.source_ref_projection_count
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT:
        return result.content_projection_count
    if channel == V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE:
        return result.lineage_projection_count
    raise ValueError("P2-C projection count channel 未注册")


def _verify_projection_semantics(
        root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        *,
        channel: int,
        expected_side: int,
        expected_count: int,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> None:
    """在任何 P2-C output 前完整校验 B3 descriptor 的物理与语义排序。"""
    with _ProjectionInputCursor(
            root,
            descriptor,
            channel=channel,
            expected_side=expected_side,
            budget=budget,
            label=label,
    ) as cursor:
        while cursor.next_unique() is not None:
            pass
        if cursor.record_count != expected_count:
            _fail(f"{label} record count 与 B3 result 不一致")


def _verify_b3_projection_result(
        root: KRunRoot,
        result: ConversationHeldOutV4ProvenanceProjectionResult,
        *,
        expected_side: int,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> None:
    """重验一个完整 B3 result 的全部三路 descriptor，而非只验下一 channel。"""
    if not isinstance(result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError(f"{label} B3 result 类型错误")
    for channel in _CHANNELS:
        descriptor = _projection_descriptor_for_channel(result, channel)
        _verify_projection_semantics(
            root,
            descriptor,
            channel=channel,
            expected_side=expected_side,
            expected_count=_projection_count_for_channel(result, channel),
            budget=budget,
            label=f"{label} channel {channel} pre-output revalidation",
        )


def _cursor_stage_name(completed_channel_count: int) -> str:
    """从固定 completed prefix 派生不可与 channel output 混淆的 cursor stage。"""
    if (type(completed_channel_count) is not int
            or completed_channel_count < 1
            or completed_channel_count > len(_CHANNELS)):
        raise ValueError("P2-C cursor completed channel count 非法")
    return f"stage-{22 + completed_channel_count:02d}-cross-side-cursor"


def _write_single_record_artifact(
        output_run_root: KRunRoot,
        *,
        stage: str,
        filename: Path,
        artifact_kind: int,
        record: tuple[int, ...],
        max_physical_bytes: int,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        existing_stage_root: KRunRoot | None = None,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor:
    """在排他新 stage 或已新建的 channel stage 写一条 sealed P0 metadata record。"""
    _require_stage_name(stage)
    if artifact_kind not in {
            V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CURSOR,
            V4_PROVENANCE_CROSS_SIDE_ARTIFACT_RECEIPT,
            V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CHANNEL_INTENT}:
        raise ValueError("P2-C metadata artifact_kind 未注册")
    if not isinstance(filename, Path) or len(filename.parts) != 1:
        raise ValueError("P2-C metadata filename 必须是单层 Path")
    _relative_path_key(filename, label=f"{label} filename")
    _require_positive_int(max_physical_bytes, label=f"{label} max_physical_bytes")
    strict_integer_tuple(record, label=f"{label} record")
    payload_bytes = encoded_integer_tuple_size(record)
    if payload_bytes > budget.max_record_payload_bytes:
        _budget_fail(f"{label} record payload 超过预算")
    if payload_bytes > budget.max_stream_payload_bytes:
        _budget_fail(f"{label} stream payload 超过预算")
    if existing_stage_root is None:
        stage_root = create_new_run_root(
            output_run_root.path / stage,
            require_k_drive=not output_run_root.test_transport,
            label=f"P2-C {stage}",
        )
    else:
        if not isinstance(existing_stage_root, KRunRoot):
            raise TypeError("P2-C existing_stage_root 必须是 KRunRoot 或 None")
        if existing_stage_root.path != output_run_root.path / stage:
            _fail("P2-C existing metadata stage 不属于指定 output capability")
        stage_root = existing_stage_root
    require_fresh_empty_run_root(stage_root, label=f"P2-C {stage}")
    stream = open_exclusive_binary(stage_root, filename, label=label)
    handle = _DigestingBinaryHandle(stream, max_bytes=max_physical_bytes)
    writer = IntegerFramedStreamWriter.from_open_binary(handle, path=filename)
    try:
        writer.append(record)
        footer = writer.seal()
    finally:
        writer.close()
    physical = handle.physical()
    if physical.byte_count > max_physical_bytes:
        _budget_fail(f"{label} physical bytes 超过预算")
    identity = capture_plain_file_identity(stage_root, filename, label=f"{label} identity")
    if (footer.record_count != 1
            or footer.total_payload_bytes != payload_bytes
            or identity.byte_count != physical.byte_count):
        _fail(f"{label} footer 或 physical identity 漂移")
    descriptor = ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor(
        artifact_kind,
        Path(stage) / filename,
        footer,
        physical,
    )
    _verify_single_record_artifact(
        output_run_root,
        descriptor,
        expected_record=record,
        max_physical_bytes=max_physical_bytes,
        budget=budget,
        label=f"{label} post-write verification",
    )
    return descriptor


def _verify_single_record_artifact(
        output_run_root: KRunRoot,
        descriptor: ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor,
        *,
        expected_record: tuple[int, ...],
        max_physical_bytes: int,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> None:
    """回读一个 intent/cursor/receipt P0，并精确匹配唯一规范 metadata record。"""
    if descriptor.physical.byte_count > max_physical_bytes:
        _budget_fail(f"{label} physical bytes 超过预算")
    if len(descriptor.output_relative_path.parts) != 2:
        _fail(f"{label} metadata artifact path 必须是单层 stage")
    stage, filename = descriptor.output_relative_path.parts
    stage_root = open_existing_run_root(
        output_run_root.path / stage,
        require_k_drive=not output_run_root.test_transport,
        label=f"{label} metadata stage",
    )
    with _VerifiedP0Reader(
            stage_root,
            Path(filename),
            footer=descriptor.p0_footer,
            physical=descriptor.physical,
            budget=budget,
            label=label,
    ) as reader:
        record = reader.read_record()
        if record != expected_record:
            _fail(f"{label} metadata record 与期望 identity 不一致")
        if reader.read_record() is not None:
            _fail(f"{label} metadata P0 不止一条 record")
        reader.finish()


def _write_cursor_artifact(
        output_run_root: KRunRoot,
        cursor: ConversationHeldOutV4ProvenanceCrossSideMergeCursor,
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact:
    """把一条完整 cursor state 排他封存为 P0，不覆盖任何旧 checkpoint。"""
    stage = _cursor_stage_name(len(cursor.completed_channel_results))
    descriptor = _write_single_record_artifact(
        output_run_root,
        stage=stage,
        filename=_CURSOR_STREAM_FILE,
        artifact_kind=V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CURSOR,
        record=cursor.integer_stream(),
        max_physical_bytes=budget.max_cursor_physical_bytes,
        budget=budget,
        label="P2-C cursor output",
    )
    artifact = ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact(
        cursor, descriptor)
    _verify_cursor_artifact_stream(
        output_run_root,
        artifact,
        budget=budget,
        label="P2-C cursor post-write verification",
    )
    return artifact


def _verify_cursor_artifact_stream(
        output_run_root: KRunRoot,
        artifact: ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact,
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> None:
    """回读 cursor P0，并要求唯一 record 精确等于调用方 state。"""
    descriptor = artifact.descriptor
    if descriptor.physical.byte_count > budget.max_cursor_physical_bytes:
        _budget_fail(f"{label} cursor physical bytes 超过预算")
    expected_stage = _cursor_stage_name(
        len(artifact.cursor.completed_channel_results))
    if descriptor.output_relative_path != Path(expected_stage) / _CURSOR_STREAM_FILE:
        _fail(f"{label} cursor relative path 与完成前缀不一致")
    stage_root = open_existing_run_root(
        output_run_root.path / expected_stage,
        require_k_drive=not output_run_root.test_transport,
        label=f"{label} cursor stage",
    )
    with _VerifiedP0Reader(
            stage_root,
            _CURSOR_STREAM_FILE,
            footer=descriptor.p0_footer,
            physical=descriptor.physical,
            budget=budget,
            label=label,
    ) as reader:
        record = reader.read_record()
        if record != artifact.cursor.integer_stream():
            _fail(f"{label} cursor record 与 state identity 不一致")
        if reader.read_record() is not None:
            _fail(f"{label} cursor P0 不止一条 record")
        reader.finish()


def _describe_sealed_p0_file(
        root: KRunRoot,
        relative: Path,
        *,
        max_physical_bytes: int,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> tuple[IntegerFramedStreamFooter, KRunFileDigest]:
    """从既有 sealed P0 重新取得 footer/physical，不信任未绑定的目录项元数据。"""
    identity = capture_plain_file_identity(root, relative, label=f"{label} capture")
    if identity.byte_count > max_physical_bytes:
        _budget_fail(f"{label} physical bytes 超过预算")
    stream = open_plain_binary(
        root,
        relative,
        label=label,
        expected_identity=identity,
    )
    handle = _DigestingBinaryHandle(stream, max_bytes=max_physical_bytes)
    try:
        reader = IntegerFramedStreamReader.from_open_binary(
            handle,
            path=relative,
            max_frame_bytes=budget.max_record_payload_bytes,
            max_record_count=budget.max_stream_record_count,
            max_total_payload_bytes=budget.max_stream_payload_bytes,
        )
    except (IntegerCodecError, OSError) as exc:
        try:
            handle.close()
        except OSError:
            pass
        _p0_read_fail(exc, label=label)
        raise AssertionError from exc
    except BaseException:
        try:
            handle.close()
        except OSError:
            pass
        raise
    try:
        while reader.read_record() is not None:
            pass
        footer = reader.finish()
    except (IntegerCodecError, OSError) as exc:
        _p0_read_fail(exc, label=label)
        raise AssertionError from exc
    finally:
        reader.close()
    physical = handle.physical()
    require_plain_file_identity(root, relative, identity, label=f"{label} post-read")
    if physical.byte_count != identity.byte_count:
        _fail(f"{label} physical bytes 与文件 identity 漂移")
    return footer, physical


def _load_channel_intent_descriptor(
        output_run_root: KRunRoot,
        intent: ConversationHeldOutV4ProvenanceCrossSideMergeChannelIntent,
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor:
    """只从固定 channel stage 回读 intent，并逐 record 比对当前冻结身份。"""
    if not isinstance(
            intent, ConversationHeldOutV4ProvenanceCrossSideMergeChannelIntent):
        raise TypeError("P2-C expected channel intent 类型错误")
    stage = _CHANNEL_STAGE_NAMES[intent.channel]
    stage_root = open_existing_run_root(
        output_run_root.path / stage,
        require_k_drive=not output_run_root.test_transport,
        label=f"{label} intent stage",
    )
    footer, physical = _describe_sealed_p0_file(
        stage_root,
        _INTENT_STREAM_FILE,
        max_physical_bytes=budget.max_intent_physical_bytes,
        budget=budget,
        label=f"{label} intent",
    )
    descriptor = ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor(
        V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CHANNEL_INTENT,
        Path(stage) / _INTENT_STREAM_FILE,
        footer,
        physical,
    )
    _verify_single_record_artifact(
        output_run_root,
        descriptor,
        expected_record=intent.integer_stream(),
        max_physical_bytes=budget.max_intent_physical_bytes,
        budget=budget,
        label=f"{label} intent verification",
    )
    return descriptor


def _replay_channel_output(
        training_work_root: KRunRoot,
        training_descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        held_out_work_root: KRunRoot,
        held_out_descriptor: ConversationHeldOutV4ProvenanceProjectionStreamDescriptor,
        output_stage_root: KRunRoot,
        output_descriptor: ConversationHeldOutV4ProvenanceCrossSideMergeStreamDescriptor,
        *,
        channel: int,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> tuple[int, int, int, int, int, int]:
    """将当前 inputs 与 sealed output 逐条重放比对，返回六项精确计数。"""
    training_cursor = _ProjectionInputCursor(
        training_work_root,
        training_descriptor,
        channel=channel,
        expected_side=V4_PROVENANCE_SIDE_TRAINING,
        budget=budget,
        label=f"{label} training input",
    )
    held_out_cursor = _ProjectionInputCursor(
        held_out_work_root,
        held_out_descriptor,
        channel=channel,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT,
        budget=budget,
        label=f"{label} held-out input",
    )
    try:
        with _VerifiedP0Reader(
                output_stage_root,
                _MERGED_STREAM_FILE,
                footer=output_descriptor.p0_footer,
                physical=output_descriptor.physical,
                budget=budget,
                label=f"{label} merged output",
        ) as output_reader:
            training_record = training_cursor.next_unique()
            held_out_record = held_out_cursor.next_unique()
            emitted = 0
            payload_bytes = 0
            while training_record is not None or held_out_record is not None:
                if training_record is None:
                    expected = held_out_record
                    held_out_record = held_out_cursor.next_unique()
                elif held_out_record is None:
                    expected = training_record
                    training_record = training_cursor.next_unique()
                elif training_record.primary_key < held_out_record.primary_key:
                    expected = training_record
                    training_record = training_cursor.next_unique()
                elif training_record.primary_key > held_out_record.primary_key:
                    expected = held_out_record
                    held_out_record = held_out_cursor.next_unique()
                else:
                    _fail(f"{label} 当前 B3 inputs 存在 cross-side overlap")
                    raise AssertionError
                actual = output_reader.read_record()
                if actual != expected.record:
                    _fail(f"{label} merged record 与重算结果不一致")
                emitted += 1
                payload_bytes += encoded_integer_tuple_size(expected.record)
            if output_reader.read_record() is not None:
                _fail(f"{label} merged output 含多余 record")
            output_reader.finish()
    finally:
        training_cursor.close()
        held_out_cursor.close()
    return (
        training_cursor.record_count,
        held_out_cursor.record_count,
        training_cursor.duplicate_count,
        held_out_cursor.duplicate_count,
        emitted,
        payload_bytes,
    )


def _recover_sealed_channel_without_cursor(
        training_work_root: KRunRoot,
        training_result: ConversationHeldOutV4ProvenanceProjectionResult,
        held_out_work_root: KRunRoot,
        held_out_result: ConversationHeldOutV4ProvenanceProjectionResult,
        output_run_root: KRunRoot,
        *,
        channel: int,
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        allow_existing_cursor: bool = False,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult | None:
    """接管 intent/output 均已 seal 但 cursor 前中断的 channel；无法重放即拒绝。"""
    stage_name = _CHANNEL_STAGE_NAMES[channel]
    stage_path = output_run_root.path / stage_name
    if not os.path.lexists(stage_path):
        return None
    cursor_stage_path = output_run_root.path / _cursor_stage_name(
        _CHANNELS.index(channel) + 1)
    if os.path.lexists(cursor_stage_path) and not allow_existing_cursor:
        _fail("P2-C 未声明的 cursor stage 与待恢复 channel 同时存在")
    stage_root = open_existing_run_root(
        stage_path,
        require_k_drive=not output_run_root.test_transport,
        label="P2-C orphaned channel output stage",
    )
    require_exact_file_closure(
        stage_root,
        frozenset({_INTENT_STREAM_FILE, _MERGED_STREAM_FILE}),
        label="P2-C orphaned channel output closure",
    )
    intent = _expected_channel_intent(
        training_result,
        held_out_result,
        channel=channel,
        code_identity=code_identity,
        budget=budget,
        logical_stage_name=logical_stage_name,
    )
    intent_descriptor = _load_channel_intent_descriptor(
        output_run_root,
        intent,
        budget=budget,
        label="P2-C orphaned channel output",
    )
    footer, physical = _describe_sealed_p0_file(
        stage_root,
        _MERGED_STREAM_FILE,
        max_physical_bytes=budget.max_stream_physical_bytes,
        budget=budget,
        label="P2-C orphaned channel output",
    )
    descriptor = ConversationHeldOutV4ProvenanceCrossSideMergeStreamDescriptor(
        channel,
        Path(stage_name) / _MERGED_STREAM_FILE,
        footer,
        physical,
    )
    training_descriptor = _projection_descriptor_for_channel(training_result, channel)
    held_out_descriptor = _projection_descriptor_for_channel(held_out_result, channel)
    counts = _replay_channel_output(
        training_work_root,
        training_descriptor,
        held_out_work_root,
        held_out_descriptor,
        stage_root,
        descriptor,
        channel=channel,
        budget=budget,
        label="P2-C orphaned channel replay",
    )
    return ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult(
        logical_stage_name,
        channel,
        training_descriptor,
        held_out_descriptor,
        intent_descriptor,
        descriptor,
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        intent_descriptor.physical.byte_count + physical.byte_count,
        budget,
    )


def _verify_checkpoint_channel(
        training_work_root: KRunRoot,
        training_result: ConversationHeldOutV4ProvenanceProjectionResult,
        held_out_work_root: KRunRoot,
        held_out_result: ConversationHeldOutV4ProvenanceProjectionResult,
        output_run_root: KRunRoot,
        result: ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult,
        *,
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> None:
    """先重验 channel intent，再流式重算 checkpoint，防止伪造或重解释 cursor。"""
    channel = result.channel
    training_descriptor = _projection_descriptor_for_channel(training_result, channel)
    held_out_descriptor = _projection_descriptor_for_channel(held_out_result, channel)
    if (result.training_input != training_descriptor
            or result.held_out_input != held_out_descriptor):
        _fail(f"{label} checkpoint input descriptor 与当前 B3 result 漂移")
    expected_relative = Path(_CHANNEL_STAGE_NAMES[channel]) / _MERGED_STREAM_FILE
    if result.output.output_relative_path != expected_relative:
        _fail(f"{label} checkpoint output relative path 非法")
    output_stage_root = open_existing_run_root(
        output_run_root.path / _CHANNEL_STAGE_NAMES[channel],
        require_k_drive=not output_run_root.test_transport,
        label=f"{label} checkpoint output stage",
    )
    require_exact_file_closure(
        output_stage_root,
        frozenset({_INTENT_STREAM_FILE, _MERGED_STREAM_FILE}),
        label=f"{label} checkpoint channel closure",
    )
    expected_intent = _expected_channel_intent(
        training_result,
        held_out_result,
        channel=channel,
        code_identity=code_identity,
        budget=budget,
        logical_stage_name=result.logical_stage_name,
    )
    actual_intent = _load_channel_intent_descriptor(
        output_run_root,
        expected_intent,
        budget=budget,
        label=f"{label} checkpoint",
    )
    if actual_intent != result.intent:
        _fail(f"{label} checkpoint intent identity 与封存 state 不一致")
    if result.output.channel != channel:
        _fail(f"{label} checkpoint output channel 错位")
    training_cursor = _ProjectionInputCursor(
        training_work_root,
        training_descriptor,
        channel=channel,
        expected_side=V4_PROVENANCE_SIDE_TRAINING,
        budget=budget,
        label=f"{label} training input",
    )
    held_out_cursor = _ProjectionInputCursor(
        held_out_work_root,
        held_out_descriptor,
        channel=channel,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT,
        budget=budget,
        label=f"{label} held-out input",
    )
    try:
        with _VerifiedP0Reader(
                output_stage_root,
                _MERGED_STREAM_FILE,
                footer=result.output.p0_footer,
                physical=result.output.physical,
                budget=budget,
                label=f"{label} merged output",
        ) as output_reader:
            training_record = training_cursor.next_unique()
            held_out_record = held_out_cursor.next_unique()
            emitted = 0
            payload_bytes = 0
            while training_record is not None or held_out_record is not None:
                if training_record is None:
                    expected = held_out_record
                    held_out_record = held_out_cursor.next_unique()
                elif held_out_record is None:
                    expected = training_record
                    training_record = training_cursor.next_unique()
                elif training_record.primary_key < held_out_record.primary_key:
                    expected = training_record
                    training_record = training_cursor.next_unique()
                elif training_record.primary_key > held_out_record.primary_key:
                    expected = held_out_record
                    held_out_record = held_out_cursor.next_unique()
                else:
                    _fail(f"{label} 当前 B3 inputs 存在 cross-side overlap")
                    raise AssertionError
                actual = output_reader.read_record()
                if actual != expected.record:
                    _fail(f"{label} checkpoint merged record 与重算结果不一致")
                emitted += 1
                payload_bytes += encoded_integer_tuple_size(expected.record)
            if output_reader.read_record() is not None:
                _fail(f"{label} checkpoint output 含多余 record")
            output_reader.finish()
    finally:
        training_cursor.close()
        held_out_cursor.close()
    if (training_cursor.record_count != result.training_input_record_count
            or held_out_cursor.record_count != result.held_out_input_record_count
            or training_cursor.duplicate_count
            != result.training_exact_duplicate_count
            or held_out_cursor.duplicate_count
            != result.held_out_exact_duplicate_count
            or emitted != result.emitted_record_count):
        _fail(f"{label} checkpoint record/duplicate count 与重算结果不一致")
    if result.output.p0_footer.total_payload_bytes != payload_bytes:
        _fail(f"{label} checkpoint output payload count 与重算结果不一致")
    if result.materialized_physical_bytes != (
            actual_intent.physical.byte_count + result.output.physical.byte_count):
        _fail(f"{label} checkpoint intent/output physical 统计不一致")


def _verify_resume_cursor(
        training_work_root: KRunRoot,
        training_result: ConversationHeldOutV4ProvenanceProjectionResult,
        held_out_work_root: KRunRoot,
        held_out_result: ConversationHeldOutV4ProvenanceProjectionResult,
        output_run_root: KRunRoot,
        artifact: ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact,
        *,
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeCursor:
    """验证 sealed cursor、全部已完成 output 和所有冻结 resume 输入。"""
    if not isinstance(artifact,
                      ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact):
        raise TypeError("P2-C resume cursor artifact 类型错误")
    cursor = artifact.cursor
    if cursor.logical_stage_name != logical_stage_name:
        _fail("P2-C resume logical_stage_name 漂移")
    if cursor.training_b3_stable_key != training_result.stable_key():
        _fail("P2-C resume training B3 stable key 漂移")
    if cursor.held_out_b3_stable_key != held_out_result.stable_key():
        _fail("P2-C resume held-out B3 stable key 漂移")
    if cursor.code_identity != code_identity:
        _fail("P2-C resume code identity 漂移")
    if cursor.budget != budget:
        _fail("P2-C resume budget 漂移")
    if not cursor.completed_channel_results:
        _fail("P2-C resume cursor 不得是零 channel 内存构造 state")
    _verify_cursor_artifact_stream(
        output_run_root,
        artifact,
        budget=budget,
        label="P2-C resume cursor verification",
    )
    for index, channel_result in enumerate(cursor.completed_channel_results):
        _verify_checkpoint_channel(
            training_work_root,
            training_result,
            held_out_work_root,
            held_out_result,
            output_run_root,
            channel_result,
            code_identity=code_identity,
            budget=budget,
            label=f"P2-C resume checkpoint {index + 1}",
        )
    return cursor


def _cursor_prefix_file_paths(
        cursor: ConversationHeldOutV4ProvenanceCrossSideMergeCursor,
        ) -> frozenset[Path]:
    """返回一个已构造 cursor 前缀唯一允许的 output/cursor 普通文件集合。"""
    if not isinstance(cursor, ConversationHeldOutV4ProvenanceCrossSideMergeCursor):
        raise TypeError("P2-C cursor prefix 类型错误")
    result: set[Path] = set()
    for index, channel_result in enumerate(cursor.completed_channel_results):
        channel = _CHANNELS[index]
        expected_intent = Path(_CHANNEL_STAGE_NAMES[channel]) / _INTENT_STREAM_FILE
        expected_output = Path(_CHANNEL_STAGE_NAMES[channel]) / _MERGED_STREAM_FILE
        if (channel_result.intent.output_relative_path != expected_intent
                or channel_result.output.output_relative_path != expected_output):
            _fail("P2-C cursor prefix intent/output path 不符合固定通道顺序")
        result.add(expected_intent)
        result.add(expected_output)
        result.add(Path(_cursor_stage_name(index + 1)) / _CURSOR_STREAM_FILE)
    return frozenset(result)


def load_v4_provenance_cross_side_merge_resume_cursor(
        training_result: ConversationHeldOutV4ProvenanceProjectionResult,
        training_work_root: KRunRoot,
        held_out_result: ConversationHeldOutV4ProvenanceProjectionResult,
        held_out_work_root: KRunRoot,
        output_run_root: KRunRoot,
        *,
        completed_channel_count: int,
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact:
    """重建调用方指定的连续 cursor prefix，不扫描、反序列化或写入磁盘 state。"""
    if not isinstance(training_result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P2-C loader training_result 类型错误")
    if not isinstance(held_out_result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P2-C loader held_out_result 类型错误")
    if not isinstance(training_work_root, KRunRoot):
        raise TypeError("P2-C loader training_work_root 必须是 KRunRoot")
    if not isinstance(held_out_work_root, KRunRoot):
        raise TypeError("P2-C loader held_out_work_root 必须是 KRunRoot")
    if not isinstance(output_run_root, KRunRoot):
        raise TypeError("P2-C loader output_run_root 必须是 KRunRoot")
    if not isinstance(code_identity, ProtocolKey):
        raise TypeError("P2-C loader code_identity 必须是 ProtocolKey")
    if not isinstance(budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
        raise TypeError("P2-C loader budget 类型错误")
    if (type(completed_channel_count) is not int
            or completed_channel_count < 1
            or completed_channel_count > len(_CHANNELS)):
        raise ValueError("P2-C loader completed_channel_count 必须是 1 到 3 的严格整数")
    _require_stage_name(logical_stage_name)
    require_disjoint_run_roots(
        training_work_root, held_out_work_root,
        label="P2-C loader training/held-out work roots")
    require_disjoint_run_roots(
        training_work_root, output_run_root,
        label="P2-C loader training/output work roots")
    require_disjoint_run_roots(
        held_out_work_root, output_run_root,
        label="P2-C loader held-out/output work roots")
    if (os.path.lexists(output_run_root.path / _RECEIPT_STAGE_NAME)
            or os.path.lexists(output_run_root.path / _MANIFEST_FILE)):
        _fail("P2-C loader 不接受 receipt 或 manifest 已存在的 output run")

    _verify_b3_projection_result(
        training_work_root,
        training_result,
        expected_side=V4_PROVENANCE_SIDE_TRAINING,
        budget=budget,
        label="P2-C loader training B3 result",
    )
    _verify_b3_projection_result(
        held_out_work_root,
        held_out_result,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT,
        budget=budget,
        label="P2-C loader held-out B3 result",
    )

    completed_results: list[
        ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult] = []
    for index, channel in enumerate(_CHANNELS[:completed_channel_count]):
        if not os.path.lexists(output_run_root.path / _CHANNEL_STAGE_NAMES[channel]):
            _fail("P2-C loader 指定 cursor prefix 缺少 channel output")
        if not os.path.lexists(output_run_root.path / _cursor_stage_name(index + 1)):
            _fail("P2-C loader 指定 cursor prefix 缺少 cursor stage")
        reconstructed = _recover_sealed_channel_without_cursor(
            training_work_root,
            training_result,
            held_out_work_root,
            held_out_result,
            output_run_root,
            channel=channel,
            code_identity=code_identity,
            budget=budget,
            logical_stage_name=logical_stage_name,
            allow_existing_cursor=True,
        )
        if reconstructed is None:
            _fail("P2-C loader 指定 channel output 无法重建")
        completed_results.append(reconstructed)
    cursor = ConversationHeldOutV4ProvenanceCrossSideMergeCursor(
        logical_stage_name,
        training_result.stable_key(),
        held_out_result.stable_key(),
        code_identity,
        budget,
        tuple(completed_results),
    )
    prefix_artifacts: list[
        ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact] = []
    for completed_count in range(1, completed_channel_count + 1):
        prefix = ConversationHeldOutV4ProvenanceCrossSideMergeCursor(
            logical_stage_name,
            training_result.stable_key(),
            held_out_result.stable_key(),
            code_identity,
            budget,
            tuple(completed_results[:completed_count]),
        )
        prefix_artifacts.append(_load_cursor_artifact_for_prefix(
            output_run_root,
            prefix,
            budget=budget,
            label=f"P2-C loader prefix {completed_count}",
        ))
    artifact = prefix_artifacts[-1]
    expected_files = set(_cursor_prefix_file_paths(cursor))
    if completed_channel_count < len(_CHANNELS):
        orphan_channel = _CHANNELS[completed_channel_count]
        orphan_stage = output_run_root.path / _CHANNEL_STAGE_NAMES[orphan_channel]
        orphan_cursor_stage = output_run_root.path / _cursor_stage_name(
            completed_channel_count + 1)
        if os.path.lexists(orphan_cursor_stage):
            _fail("P2-C loader 指定 prefix 后出现未声明 cursor stage")
        if os.path.lexists(orphan_stage):
            orphan = _recover_sealed_channel_without_cursor(
                training_work_root,
                training_result,
                held_out_work_root,
                held_out_result,
                output_run_root,
                channel=orphan_channel,
                code_identity=code_identity,
                budget=budget,
                logical_stage_name=logical_stage_name,
            )
            if orphan is None:
                _fail("P2-C loader 指定 prefix 后的 orphan output 无法重建")
            expected_files.add(orphan.intent.output_relative_path)
            expected_files.add(orphan.output.output_relative_path)
    for later_index, later_channel in enumerate(
            _CHANNELS[completed_channel_count + 1:], completed_channel_count + 1):
        if (os.path.lexists(output_run_root.path / _CHANNEL_STAGE_NAMES[later_channel])
                or os.path.lexists(output_run_root.path
                                  / _cursor_stage_name(later_index + 1))):
            _fail("P2-C loader 指定 prefix 后出现提前 stage")
    require_exact_file_closure(
        output_run_root,
        frozenset(expected_files),
        label="P2-C loader output closure",
    )
    return artifact


def _load_cursor_artifact_for_prefix(
        output_run_root: KRunRoot,
        cursor: ConversationHeldOutV4ProvenanceCrossSideMergeCursor,
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact:
    """以构造出的 prefix state 回读唯一 cursor P0，不从其 payload 反序列化对象。"""
    completed_count = len(cursor.completed_channel_results)
    if completed_count < 1:
        _fail(f"{label} cursor prefix 不能为空")
    stage = _cursor_stage_name(completed_count)
    stage_root = open_existing_run_root(
        output_run_root.path / stage,
        require_k_drive=not output_run_root.test_transport,
        label=f"{label} cursor stage",
    )
    footer, physical = _describe_sealed_p0_file(
        stage_root,
        _CURSOR_STREAM_FILE,
        max_physical_bytes=budget.max_cursor_physical_bytes,
        budget=budget,
        label=f"{label} cursor",
    )
    artifact = ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact(
        cursor,
        ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor(
            V4_PROVENANCE_CROSS_SIDE_ARTIFACT_CURSOR,
            Path(stage) / _CURSOR_STREAM_FILE,
            footer,
            physical,
        ),
    )
    _verify_cursor_artifact_stream(
        output_run_root,
        artifact,
        budget=budget,
        label=f"{label} cursor verification",
    )
    return artifact


def _reconstruct_completed_cursor_artifacts(
        output_run_root: KRunRoot,
        cursor: ConversationHeldOutV4ProvenanceCrossSideMergeCursor,
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        ) -> tuple[ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact,
                   ...]:
    """从全部完成前缀重建并回读三份 cursor，不能只相信最终 checkpoint。"""
    if not cursor.is_complete:
        _fail("P2-C resource receipt 需要完整三通道 cursor")
    artifacts: list[ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact] = []
    for completed_count in range(1, len(_CHANNELS) + 1):
        prefix = ConversationHeldOutV4ProvenanceCrossSideMergeCursor(
            cursor.logical_stage_name,
            cursor.training_b3_stable_key,
            cursor.held_out_b3_stable_key,
            cursor.code_identity,
            cursor.budget,
            cursor.completed_channel_results[:completed_count],
        )
        artifacts.append(_load_cursor_artifact_for_prefix(
            output_run_root,
            prefix,
            budget=budget,
            label="P2-C receipt",
        ))
    return tuple(artifacts)


def _pre_receipt_file_specs(
        cursor: ConversationHeldOutV4ProvenanceCrossSideMergeCursor,
        cursor_artifacts: tuple[
            ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact,
            ...,
        ],
        ) -> tuple[tuple[Path, KRunFileDigest, int], ...]:
    """给出九个必需 pre-receipt 文件及其已核验的 digest/单文件上限。"""
    if (not cursor.is_complete
            or len(cursor_artifacts) != len(_CHANNELS)):
        _fail("P2-C pre-receipt files 需要完整 cursor 前缀")
    result: list[tuple[Path, KRunFileDigest, int]] = []
    for channel_result in cursor.completed_channel_results:
        result.append((
            channel_result.intent.output_relative_path,
            channel_result.intent.physical,
            channel_result.budget.max_intent_physical_bytes,
        ))
        result.append((
            channel_result.output.output_relative_path,
            channel_result.output.physical,
            channel_result.budget.max_stream_physical_bytes,
        ))
    for artifact in cursor_artifacts:
        result.append((
            artifact.descriptor.output_relative_path,
            artifact.descriptor.physical,
            cursor.budget.max_cursor_physical_bytes,
        ))
    paths = frozenset(item[0] for item in result)
    if len(paths) != len(result):
        _fail("P2-C pre-receipt files 存在重复路径")
    return tuple(sorted(result, key=lambda item: _relative_path_key(
        item[0], label="P2-C pre-receipt file path")))


def _reverify_file_descriptors(
        output_run_root: KRunRoot,
        specs: tuple[tuple[Path, KRunFileDigest, int], ...],
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        label: str,
        ) -> tuple[ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor,
                   ...]:
    """重新哈希全部声明文件，拒绝 receipt/manifest 前同路径物理漂移。"""
    if not isinstance(specs, tuple) or not specs:
        raise TypeError("P2-C file descriptor specs 必须是非空 tuple")
    result: list[ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor] = []
    total_physical_bytes = 0
    previous: tuple[int, ...] | None = None
    for relative, expected_physical, max_physical_bytes in specs:
        if not isinstance(expected_physical, KRunFileDigest):
            raise TypeError("P2-C expected physical 类型错误")
        _require_positive_int(
            max_physical_bytes, label="P2-C expected file max_physical_bytes")
        path_key = _relative_path_key(relative, label=f"{label} path")
        if previous is not None and path_key <= previous:
            _fail(f"{label} 文件路径未严格排序")
        previous = path_key
        actual_physical = sha256_plain_file(
            output_run_root,
            relative,
            max_bytes=max_physical_bytes,
            label=f"{label} {relative.as_posix()}",
        )
        if actual_physical != expected_physical:
            _fail(f"{label} 文件 physical identity 漂移")
        total_physical_bytes += actual_physical.byte_count
        if total_physical_bytes > budget.max_total_materialized_physical_bytes:
            _budget_fail(f"{label} 文件累计 physical bytes 超过总预算")
        result.append(ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor(
            relative,
            actual_physical,
        ))
    return tuple(result)


def _build_resource_receipt(
        cursor_artifact: ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact,
        pre_receipt_files: tuple[
            ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor,
            ...,
        ],
        *,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeResourceReceipt:
    """由已重放的完整 cursor 与六文件 closure 构建零交集机械资源回执。"""
    cursor = cursor_artifact.cursor
    if not cursor.is_complete:
        _fail("P2-C resource receipt 不能绑定未完成 cursor")
    if len(pre_receipt_files) + 1 > budget.max_pre_manifest_file_count:
        _budget_fail("P2-C resource receipt closure 文件数超过预算")
    pre_receipt_physical_bytes = sum(
        item.physical.byte_count for item in pre_receipt_files)
    if pre_receipt_physical_bytes > budget.max_total_materialized_physical_bytes:
        _budget_fail("P2-C resource receipt pre-receipt physical 超过总预算")
    return ConversationHeldOutV4ProvenanceCrossSideMergeResourceReceipt(
        cursor.logical_stage_name,
        cursor.training_b3_stable_key,
        cursor.held_out_b3_stable_key,
        cursor.code_identity,
        budget,
        cursor_artifact.stable_key(),
        cursor.completed_channel_results,
        pre_receipt_files,
        pre_receipt_physical_bytes,
        0,
        0,
        3,
        V4_PROVENANCE_CROSS_SIDE_RECEIPT_STATUS_ZERO_INTERSECTION_MECHANICAL,
    )


def _write_or_recover_resource_receipt(
        output_run_root: KRunRoot,
        receipt: ConversationHeldOutV4ProvenanceCrossSideMergeResourceReceipt,
        *,
        pre_receipt_paths: frozenset[Path],
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeReceiptArtifact:
    """排他写入 receipt，或只在其逐字节重建时接管 seal 后 manifest 前中断。"""
    receipt_relative = Path(_RECEIPT_STAGE_NAME) / _RECEIPT_STREAM_FILE
    expected_with_receipt = frozenset((*pre_receipt_paths, receipt_relative))
    stage_path = output_run_root.path / _RECEIPT_STAGE_NAME
    if not os.path.lexists(stage_path):
        require_exact_file_closure(
            output_run_root,
            pre_receipt_paths,
            label="P2-C receipt pre-publication closure",
        )
        descriptor = _write_single_record_artifact(
            output_run_root,
            stage=_RECEIPT_STAGE_NAME,
            filename=_RECEIPT_STREAM_FILE,
            artifact_kind=V4_PROVENANCE_CROSS_SIDE_ARTIFACT_RECEIPT,
            record=receipt.integer_stream(),
            max_physical_bytes=budget.max_receipt_physical_bytes,
            budget=budget,
            label="P2-C resource receipt",
        )
    else:
        require_exact_file_closure(
            output_run_root,
            expected_with_receipt,
            label="P2-C receipt recovery closure",
        )
        stage_root = open_existing_run_root(
            stage_path,
            require_k_drive=not output_run_root.test_transport,
            label="P2-C receipt recovery stage",
        )
        footer, physical = _describe_sealed_p0_file(
            stage_root,
            _RECEIPT_STREAM_FILE,
            max_physical_bytes=budget.max_receipt_physical_bytes,
            budget=budget,
            label="P2-C receipt recovery",
        )
        descriptor = ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor(
            V4_PROVENANCE_CROSS_SIDE_ARTIFACT_RECEIPT,
            receipt_relative,
            footer,
            physical,
        )
    artifact = ConversationHeldOutV4ProvenanceCrossSideMergeReceiptArtifact(
        receipt,
        descriptor,
    )
    _verify_single_record_artifact(
        output_run_root,
        descriptor,
        expected_record=receipt.integer_stream(),
        max_physical_bytes=budget.max_receipt_physical_bytes,
        budget=budget,
        label="P2-C resource receipt verification",
    )
    return artifact


def load_v4_provenance_cross_side_merge_publication(
        training_result: ConversationHeldOutV4ProvenanceProjectionResult,
        training_work_root: KRunRoot,
        held_out_result: ConversationHeldOutV4ProvenanceProjectionResult,
        held_out_work_root: KRunRoot,
        output_run_root: KRunRoot,
        *,
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult:
    """从已发布的固定闭包重建 P2-C publication，不信任磁盘 payload 为 state。

    这个入口只读：它以当前 B3 inputs 重新构造三条 channel、三个 cursor、receipt 与
    manifest 的规范 payload，再逐个比对已封存文件。它不接受 partial run，也绝不补写
    receipt 或 manifest；后续 P3 只能消费返回的重建结果。
    """
    if not isinstance(training_result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P2-C publication loader training_result 类型错误")
    if not isinstance(held_out_result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P2-C publication loader held_out_result 类型错误")
    if not isinstance(training_work_root, KRunRoot):
        raise TypeError("P2-C publication loader training_work_root 必须是 KRunRoot")
    if not isinstance(held_out_work_root, KRunRoot):
        raise TypeError("P2-C publication loader held_out_work_root 必须是 KRunRoot")
    if not isinstance(output_run_root, KRunRoot):
        raise TypeError("P2-C publication loader output_run_root 必须是 KRunRoot")
    if not isinstance(code_identity, ProtocolKey):
        raise TypeError("P2-C publication loader code_identity 必须是 ProtocolKey")
    if not isinstance(budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
        raise TypeError("P2-C publication loader budget 类型错误")
    _require_stage_name(logical_stage_name)
    require_disjoint_run_roots(
        training_work_root, held_out_work_root,
        label="P2-C publication loader training/held-out work roots")
    require_disjoint_run_roots(
        training_work_root, output_run_root,
        label="P2-C publication loader training/output work roots")
    require_disjoint_run_roots(
        held_out_work_root, output_run_root,
        label="P2-C publication loader held-out/output work roots")
    if not os.path.lexists(output_run_root.path / _RECEIPT_STAGE_NAME):
        _fail("P2-C publication loader 缺少 receipt stage")
    if not os.path.lexists(output_run_root.path / _MANIFEST_FILE):
        _fail("P2-C publication loader 缺少 manifest")

    _verify_b3_projection_result(
        training_work_root,
        training_result,
        expected_side=V4_PROVENANCE_SIDE_TRAINING,
        budget=budget,
        label="P2-C publication loader training B3 result",
    )
    _verify_b3_projection_result(
        held_out_work_root,
        held_out_result,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT,
        budget=budget,
        label="P2-C publication loader held-out B3 result",
    )

    # Each cursor prefix is reconstructed independently.  A final cursor cannot
    # conceal a changed earlier checkpoint or a channel output from another run.
    completed_results: list[
        ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult] = []
    cursor_artifacts: list[
        ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact] = []
    for index, channel in enumerate(_CHANNELS):
        if not os.path.lexists(output_run_root.path / _CHANNEL_STAGE_NAMES[channel]):
            _fail("P2-C publication loader 缺少 channel stage")
        if not os.path.lexists(output_run_root.path / _cursor_stage_name(index + 1)):
            _fail("P2-C publication loader 缺少 cursor stage")
        channel_result = _recover_sealed_channel_without_cursor(
            training_work_root,
            training_result,
            held_out_work_root,
            held_out_result,
            output_run_root,
            channel=channel,
            code_identity=code_identity,
            budget=budget,
            logical_stage_name=logical_stage_name,
            allow_existing_cursor=True,
        )
        if channel_result is None:
            _fail("P2-C publication loader channel output 无法重建")
        completed_results.append(channel_result)
        prefix = ConversationHeldOutV4ProvenanceCrossSideMergeCursor(
            logical_stage_name,
            training_result.stable_key(),
            held_out_result.stable_key(),
            code_identity,
            budget,
            tuple(completed_results),
        )
        cursor_artifacts.append(_load_cursor_artifact_for_prefix(
            output_run_root,
            prefix,
            budget=budget,
            label=f"P2-C publication loader prefix {index + 1}",
        ))
    final_cursor = cursor_artifacts[-1]
    cursor = final_cursor.cursor
    if not cursor.is_complete:
        _fail("P2-C publication loader 最终 cursor 不完整")

    pre_receipt_specs = _pre_receipt_file_specs(cursor, tuple(cursor_artifacts))
    pre_receipt_paths = frozenset(item[0] for item in pre_receipt_specs)
    pre_receipt_files = _reverify_file_descriptors(
        output_run_root,
        pre_receipt_specs,
        budget=budget,
        label="P2-C publication loader pre-receipt",
    )
    receipt = _build_resource_receipt(
        final_cursor,
        pre_receipt_files,
        budget=budget,
    )
    receipt_relative = Path(_RECEIPT_STAGE_NAME) / _RECEIPT_STREAM_FILE
    receipt_stage_root = open_existing_run_root(
        output_run_root.path / _RECEIPT_STAGE_NAME,
        require_k_drive=not output_run_root.test_transport,
        label="P2-C publication loader receipt stage",
    )
    receipt_footer, receipt_physical = _describe_sealed_p0_file(
        receipt_stage_root,
        _RECEIPT_STREAM_FILE,
        max_physical_bytes=budget.max_receipt_physical_bytes,
        budget=budget,
        label="P2-C publication loader receipt",
    )
    receipt_artifact = ConversationHeldOutV4ProvenanceCrossSideMergeReceiptArtifact(
        receipt,
        ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor(
            V4_PROVENANCE_CROSS_SIDE_ARTIFACT_RECEIPT,
            receipt_relative,
            receipt_footer,
            receipt_physical,
        ),
    )
    _verify_single_record_artifact(
        output_run_root,
        receipt_artifact.descriptor,
        expected_record=receipt.integer_stream(),
        max_physical_bytes=budget.max_receipt_physical_bytes,
        budget=budget,
        label="P2-C publication loader receipt verification",
    )

    pre_manifest_specs = tuple(sorted(
        (*pre_receipt_specs,
         (receipt_relative, receipt_artifact.descriptor.physical,
          budget.max_receipt_physical_bytes)),
        key=lambda item: _relative_path_key(
            item[0], label="P2-C publication loader pre-manifest file path"),
    ))
    pre_manifest_files = _reverify_file_descriptors(
        output_run_root,
        pre_manifest_specs,
        budget=budget,
        label="P2-C publication loader pre-manifest",
    )
    manifest = ConversationHeldOutV4ProvenanceCrossSideMergeManifest(
        logical_stage_name,
        training_result.stable_key(),
        held_out_result.stable_key(),
        receipt_artifact.stable_key(),
        final_cursor.stable_key(),
        code_identity,
        budget,
        pre_manifest_files,
    )
    manifest_payload = encode_integer_tuple(manifest.integer_stream())
    if len(manifest_payload) > budget.max_manifest_payload_bytes:
        _budget_fail("P2-C publication loader manifest payload 超过预算")
    expected_manifest_physical = KRunFileDigest(
        len(manifest_payload),
        tuple(hashlib.sha256(manifest_payload).digest()),
    )
    manifest_physical = sha256_plain_file(
        output_run_root,
        _MANIFEST_FILE,
        max_bytes=budget.max_manifest_payload_bytes,
        label="P2-C publication loader manifest",
    )
    if manifest_physical != expected_manifest_physical:
        _fail("P2-C publication loader manifest 与重建 payload 不一致")
    manifest_file = ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor(
        _MANIFEST_FILE,
        manifest_physical,
    )
    expected_closure = frozenset((*pre_receipt_paths, receipt_relative, _MANIFEST_FILE))
    require_exact_file_closure(
        output_run_root,
        expected_closure,
        label="P2-C publication loader final closure",
    )
    total_published_physical_bytes = (
        sum(item.physical.byte_count for item in pre_manifest_files)
        + manifest_file.physical.byte_count)
    if total_published_physical_bytes > budget.max_total_materialized_physical_bytes:
        _budget_fail("P2-C publication loader total physical bytes 超过预算")
    return ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult(
        receipt_artifact,
        manifest,
        manifest_file,
        total_published_physical_bytes,
    )


def publish_v4_provenance_cross_side_merge(
        training_result: ConversationHeldOutV4ProvenanceProjectionResult,
        training_work_root: KRunRoot,
        held_out_result: ConversationHeldOutV4ProvenanceProjectionResult,
        held_out_work_root: KRunRoot,
        output_run_root: KRunRoot,
        *,
        completed_cursor: ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact,
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult:
    """在三路零交集 cursor 后发布唯一 receipt 与 manifest-last，不产生资格结论。"""
    if not isinstance(training_result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P2-C publication training_result 类型错误")
    if not isinstance(held_out_result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P2-C publication held_out_result 类型错误")
    if not isinstance(training_work_root, KRunRoot):
        raise TypeError("P2-C publication training_work_root 必须是 KRunRoot")
    if not isinstance(held_out_work_root, KRunRoot):
        raise TypeError("P2-C publication held_out_work_root 必须是 KRunRoot")
    if not isinstance(output_run_root, KRunRoot):
        raise TypeError("P2-C publication output_run_root 必须是 KRunRoot")
    if not isinstance(completed_cursor,
                      ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact):
        raise TypeError("P2-C publication completed_cursor 类型错误")
    if not isinstance(code_identity, ProtocolKey):
        raise TypeError("P2-C publication code_identity 必须是 ProtocolKey")
    if not isinstance(budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
        raise TypeError("P2-C publication budget 类型错误")
    _require_stage_name(logical_stage_name)
    require_created_run_root(training_work_root, label="P2-C publication training_work_root")
    require_created_run_root(held_out_work_root, label="P2-C publication held_out_work_root")
    require_created_run_root(output_run_root, label="P2-C publication output_run_root")
    require_disjoint_run_roots(
        training_work_root, held_out_work_root,
        label="P2-C publication training/held-out work roots")
    require_disjoint_run_roots(
        training_work_root, output_run_root,
        label="P2-C publication training/output work roots")
    require_disjoint_run_roots(
        held_out_work_root, output_run_root,
        label="P2-C publication held-out/output work roots")
    if os.path.lexists(output_run_root.path / _MANIFEST_FILE):
        _fail("P2-C manifest 已存在，禁止重复发布")

    _verify_b3_projection_result(
        training_work_root,
        training_result,
        expected_side=V4_PROVENANCE_SIDE_TRAINING,
        budget=budget,
        label="P2-C publication training B3 result",
    )
    _verify_b3_projection_result(
        held_out_work_root,
        held_out_result,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT,
        budget=budget,
        label="P2-C publication held-out B3 result",
    )
    cursor = _verify_resume_cursor(
        training_work_root,
        training_result,
        held_out_work_root,
        held_out_result,
        output_run_root,
        completed_cursor,
        code_identity=code_identity,
        budget=budget,
        logical_stage_name=logical_stage_name,
    )
    if not cursor.is_complete:
        _fail("P2-C publication 要求最终三通道 cursor")
    cursor_artifacts = _reconstruct_completed_cursor_artifacts(
        output_run_root,
        cursor,
        budget=budget,
    )
    final_cursor = cursor_artifacts[-1]
    if final_cursor.stable_key() != completed_cursor.stable_key():
        _fail("P2-C publication final cursor 与重建 checkpoint 不一致")

    pre_receipt_specs = _pre_receipt_file_specs(cursor, cursor_artifacts)
    pre_receipt_paths = frozenset(item[0] for item in pre_receipt_specs)
    receipt_relative = Path(_RECEIPT_STAGE_NAME) / _RECEIPT_STREAM_FILE
    expected_pre_receipt_closure = pre_receipt_paths
    if os.path.lexists(output_run_root.path / _RECEIPT_STAGE_NAME):
        expected_pre_receipt_closure = frozenset(
            (*pre_receipt_paths, receipt_relative))
    require_exact_file_closure(
        output_run_root,
        expected_pre_receipt_closure,
        label="P2-C publication pre-receipt closure",
    )
    pre_receipt_files = _reverify_file_descriptors(
        output_run_root,
        pre_receipt_specs,
        budget=budget,
        label="P2-C publication pre-receipt",
    )
    receipt = _build_resource_receipt(
        final_cursor,
        pre_receipt_files,
        budget=budget,
    )
    receipt_artifact = _write_or_recover_resource_receipt(
        output_run_root,
        receipt,
        pre_receipt_paths=pre_receipt_paths,
        budget=budget,
    )

    receipt_relative = receipt_artifact.descriptor.output_relative_path
    pre_manifest_specs = tuple(sorted(
        (*pre_receipt_specs,
         (receipt_relative, receipt_artifact.descriptor.physical,
          budget.max_receipt_physical_bytes)),
        key=lambda item: _relative_path_key(
            item[0], label="P2-C pre-manifest file path"),
    ))
    pre_manifest_paths = frozenset(item[0] for item in pre_manifest_specs)
    require_exact_file_closure(
        output_run_root,
        pre_manifest_paths,
        label="P2-C publication pre-manifest closure",
    )
    pre_manifest_files = _reverify_file_descriptors(
        output_run_root,
        pre_manifest_specs,
        budget=budget,
        label="P2-C publication pre-manifest",
    )
    manifest = ConversationHeldOutV4ProvenanceCrossSideMergeManifest(
        logical_stage_name,
        training_result.stable_key(),
        held_out_result.stable_key(),
        receipt_artifact.stable_key(),
        final_cursor.stable_key(),
        code_identity,
        budget,
        pre_manifest_files,
    )
    manifest_payload = encode_integer_tuple(manifest.integer_stream())
    if len(manifest_payload) > budget.max_manifest_payload_bytes:
        _budget_fail("P2-C manifest payload 超过预算")
    total_published_physical_bytes = (
        sum(item.physical.byte_count for item in pre_manifest_files)
        + len(manifest_payload))
    if total_published_physical_bytes > budget.max_total_materialized_physical_bytes:
        _budget_fail("P2-C manifest 总 physical bytes 超过预算")
    publish_manifest_last(
        output_run_root,
        _MANIFEST_FILE,
        manifest_payload,
        pre_manifest_paths,
        label="P2-C publication manifest-last",
    )
    expected_manifest_physical = KRunFileDigest(
        len(manifest_payload),
        tuple(hashlib.sha256(manifest_payload).digest()),
    )
    manifest_physical = sha256_plain_file(
        output_run_root,
        _MANIFEST_FILE,
        max_bytes=budget.max_manifest_payload_bytes,
        label="P2-C publication manifest verification",
    )
    if manifest_physical != expected_manifest_physical:
        _fail("P2-C manifest physical identity 漂移")
    manifest_file = ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor(
        _MANIFEST_FILE,
        manifest_physical,
    )
    require_exact_file_closure(
        output_run_root,
        frozenset((*pre_manifest_paths, _MANIFEST_FILE)),
        label="P2-C publication final closure",
    )
    return ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult(
        receipt_artifact,
        manifest,
        manifest_file,
        total_published_physical_bytes,
    )


def advance_v4_provenance_cross_side_merge(
        training_result: ConversationHeldOutV4ProvenanceProjectionResult,
        training_work_root: KRunRoot,
        held_out_result: ConversationHeldOutV4ProvenanceProjectionResult,
        held_out_work_root: KRunRoot,
        output_run_root: KRunRoot,
        *,
        code_identity: ProtocolKey,
        budget: ConversationHeldOutV4ProvenanceCrossSideMergeBudget,
        logical_stage_name: str,
        resume_cursor: ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact
        | None = None,
        ) -> ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact:
    """推进恰好一个固定 P2-C channel，并将其结果作为新的 sealed cursor 发布。

    首次空 output 仍要求本进程新建 capability；已完整重放的 resume 可使用重开 root。每次
    推进都先重验双方 B3 的三条 descriptor，随后只写一个此前不存在的 channel stage 和一个
    此前不存在的 cursor stage。三通道完成后本函数停止，receipt/manifest 属于后续切片。
    """
    if not isinstance(training_result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P2-C training_result 类型错误")
    if not isinstance(held_out_result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P2-C held_out_result 类型错误")
    if not isinstance(training_work_root, KRunRoot):
        raise TypeError("P2-C training_work_root 必须是 KRunRoot")
    if not isinstance(held_out_work_root, KRunRoot):
        raise TypeError("P2-C held_out_work_root 必须是 KRunRoot")
    if not isinstance(output_run_root, KRunRoot):
        raise TypeError("P2-C output_run_root 必须是 KRunRoot")
    if not isinstance(code_identity, ProtocolKey):
        raise TypeError("P2-C code_identity 必须是 ProtocolKey")
    if not isinstance(budget, ConversationHeldOutV4ProvenanceCrossSideMergeBudget):
        raise TypeError("P2-C budget 类型错误")
    _require_stage_name(logical_stage_name)
    output_was_created_here = is_created_run_root(output_run_root)
    require_disjoint_run_roots(
        training_work_root, held_out_work_root,
        label="P2-C training/held-out work roots")
    require_disjoint_run_roots(
        training_work_root, output_run_root,
        label="P2-C training/output work roots")
    require_disjoint_run_roots(
        held_out_work_root, output_run_root,
        label="P2-C held-out/output work roots")

    # 六条输入先全部重验，不能在只检查 SourceRef 后就开始写 P2-C output。
    _verify_b3_projection_result(
        training_work_root,
        training_result,
        expected_side=V4_PROVENANCE_SIDE_TRAINING,
        budget=budget,
        label="P2-C training B3 result",
    )
    _verify_b3_projection_result(
        held_out_work_root,
        held_out_result,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT,
        budget=budget,
        label="P2-C held-out B3 result",
    )

    if resume_cursor is None:
        try:
            with os.scandir(output_run_root.path) as children:
                has_existing_entry = next(children, None) is not None
        except OSError as exc:
            _fail("P2-C initial output_run_root 不可枚举")
            raise AssertionError from exc
        if not has_existing_entry:
            if not output_was_created_here:
                _fail("P2-C 空 output 首次推进必须使用本进程新建 capability")
            require_fresh_empty_run_root(
                output_run_root, label="P2-C initial output_run_root")
            cursor = ConversationHeldOutV4ProvenanceCrossSideMergeCursor(
                logical_stage_name,
                training_result.stable_key(),
                held_out_result.stable_key(),
                code_identity,
                budget,
                (),
            )
        else:
            require_exact_file_closure(
                output_run_root,
                frozenset({
                    Path(_CHANNEL_STAGE_NAMES[
                        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF])
                    / _INTENT_STREAM_FILE,
                    Path(_CHANNEL_STAGE_NAMES[
                        V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF])
                    / _MERGED_STREAM_FILE,
                }),
                label="P2-C initial orphan recovery closure",
            )
            recovered = _recover_sealed_channel_without_cursor(
                training_work_root,
                training_result,
                held_out_work_root,
                held_out_result,
                output_run_root,
                channel=V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF,
                code_identity=code_identity,
                budget=budget,
                logical_stage_name=logical_stage_name,
            )
            if recovered is None:
                _fail("P2-C initial output run 非空但没有可恢复的 source-ref output")
            cursor = ConversationHeldOutV4ProvenanceCrossSideMergeCursor(
                logical_stage_name,
                training_result.stable_key(),
                held_out_result.stable_key(),
                code_identity,
                budget,
                (recovered,),
            )
            return _write_cursor_artifact(
                output_run_root, cursor, budget=budget)
    else:
        cursor = _verify_resume_cursor(
            training_work_root,
            training_result,
            held_out_work_root,
            held_out_result,
            output_run_root,
            resume_cursor,
            code_identity=code_identity,
            budget=budget,
            logical_stage_name=logical_stage_name,
        )
        reconstructed = load_v4_provenance_cross_side_merge_resume_cursor(
            training_result,
            training_work_root,
            held_out_result,
            held_out_work_root,
            output_run_root,
            completed_channel_count=len(cursor.completed_channel_results),
            code_identity=code_identity,
            budget=budget,
            logical_stage_name=logical_stage_name,
        )
        if reconstructed.stable_key() != resume_cursor.stable_key():
            _fail("P2-C resume cursor 未通过完整跨进程 prefix 重建")
        cursor = reconstructed.cursor
    if cursor.is_complete:
        _fail("P2-C 三个 channel 已完成，不能继续调用 advance")
    channel = _CHANNELS[len(cursor.completed_channel_results)]
    expected_resume_files = set(_cursor_prefix_file_paths(cursor))
    current_intent_relative = Path(_CHANNEL_STAGE_NAMES[channel]) / _INTENT_STREAM_FILE
    current_output_relative = Path(_CHANNEL_STAGE_NAMES[channel]) / _MERGED_STREAM_FILE
    current_cursor_stage = output_run_root.path / _cursor_stage_name(
        len(cursor.completed_channel_results) + 1)
    if os.path.lexists(current_cursor_stage):
        _fail("P2-C 当前 cursor 前缀后已有未声明 cursor stage")
    if os.path.lexists(output_run_root.path / _CHANNEL_STAGE_NAMES[channel]):
        expected_resume_files.add(current_intent_relative)
        expected_resume_files.add(current_output_relative)
    if expected_resume_files:
        require_exact_file_closure(
            output_run_root,
            frozenset(expected_resume_files),
            label="P2-C advance resume closure",
        )
    else:
        require_fresh_empty_run_root(
            output_run_root,
            label="P2-C initial channel output_run_root",
        )
    recovered = _recover_sealed_channel_without_cursor(
        training_work_root,
        training_result,
        held_out_work_root,
        held_out_result,
        output_run_root,
        channel=channel,
        code_identity=code_identity,
        budget=budget,
        logical_stage_name=logical_stage_name,
    )
    if recovered is None:
        channel_result = _build_v4_provenance_cross_side_merge_channel(
            training_work_root,
            _projection_descriptor_for_channel(training_result, channel),
            held_out_work_root,
            _projection_descriptor_for_channel(held_out_result, channel),
            output_run_root,
            channel=channel,
            training_b3_stable_key=training_result.stable_key(),
            held_out_b3_stable_key=held_out_result.stable_key(),
            code_identity=code_identity,
            budget=budget,
            logical_stage_name=logical_stage_name,
            allow_reopened_roots=True,
        )
    else:
        channel_result = recovered
    next_cursor = ConversationHeldOutV4ProvenanceCrossSideMergeCursor(
        logical_stage_name,
        training_result.stable_key(),
        held_out_result.stable_key(),
        code_identity,
        budget,
        (*cursor.completed_channel_results, channel_result),
    )
    return _write_cursor_artifact(output_run_root, next_cursor, budget=budget)


__all__ = [
    "ConversationHeldOutV4ProvenanceCrossSideMergeBudget",
    "ConversationHeldOutV4ProvenanceCrossSideMergeArtifactDescriptor",
    "ConversationHeldOutV4ProvenanceCrossSideMergeChannelIntent",
    "ConversationHeldOutV4ProvenanceCrossSideMergeChannelResult",
    "ConversationHeldOutV4ProvenanceCrossSideMergeBudgetExceeded",
    "ConversationHeldOutV4ProvenanceCrossSideMergeCursor",
    "ConversationHeldOutV4ProvenanceCrossSideMergeCursorArtifact",
    "ConversationHeldOutV4ProvenanceCrossSideMergeError",
    "ConversationHeldOutV4ProvenanceCrossSideMergeFileDescriptor",
    "ConversationHeldOutV4ProvenanceCrossSideMergeManifest",
    "ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult",
    "ConversationHeldOutV4ProvenanceCrossSideMergeReceiptArtifact",
    "ConversationHeldOutV4ProvenanceCrossSideMergeResourceReceipt",
    "ConversationHeldOutV4ProvenanceCrossSideMergeStreamDescriptor",
    "ConversationHeldOutV4ProvenanceCrossSideOverlapError",
    "V4_PROVENANCE_CROSS_SIDE_CHANNEL_CONTENT",
    "V4_PROVENANCE_CROSS_SIDE_CHANNEL_LINEAGE",
    "V4_PROVENANCE_CROSS_SIDE_CHANNEL_SOURCE_REF",
    "advance_v4_provenance_cross_side_merge",
    "build_v4_provenance_cross_side_merge_channel",
    "load_v4_provenance_cross_side_merge_publication",
    "load_v4_provenance_cross_side_merge_resume_cursor",
    "publish_v4_provenance_cross_side_merge",
]
