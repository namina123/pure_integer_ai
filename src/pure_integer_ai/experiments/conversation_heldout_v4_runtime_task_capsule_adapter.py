"""DLG-05 v4 P3-C1c 同次 payload replay 到 typed external capsule 的受限适配器。

本模块只把已经由 P3-A/P3-B/P3-A2/P3-C1a/P3-C1b 闭合的 test transport
task，与同次 explicit payload replay 逐项核对后送入既有 external capsule v1。
它不执行 runtime、不调用 artifact writer、trainer 或 teacher，也不把该机械 transport
解释为来源资格、学习、理解、回答或断奶能力。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Iterable

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.experiments.conversation_heldout_v4_active_intake import (
    ConversationHeldOutV4ActiveIntakeError,
    ConversationHeldOutV4ActiveIntakeMapping,
    ConversationHeldOutV4ActiveIntakeResult,
    load_v4_active_intake_publication,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback import (
    ConversationHeldOutV4ActiveRosterReadbackError,
    ConversationHeldOutV4ActiveRosterReadbackResult,
    revalidate_v4_active_roster_bidirectional_readback,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4DependencyBinding,
    ConversationHeldOutV4SourceRecord,
)
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    ConversationHeldOutV4RuntimeEvidencePlan,
    ConversationHeldOutV4RuntimeInput,
    ConversationHeldOutV4RuntimeSourceCapsule,
    V4_RUNTIME_FAMILY_KEY,
    V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    ConversationHeldOutV4ExternalCapsuleBudget,
    ConversationHeldOutV4ExternalCapsuleError,
    ConversationHeldOutV4ExternalInputCapsule,
    ConversationHeldOutV4ExternalProducer,
    prepare_v4_external_input_capsule,
    read_budgeted_v4_external_input_capsule,
    write_prepared_v4_external_input_capsule,
)
from pure_integer_ai.experiments.conversation_heldout_v4_payload_witness_ledger import (
    ConversationHeldOutV4PayloadWitness,
    ConversationHeldOutV4PayloadWitnessLedgerError,
    ConversationHeldOutV4PayloadWitnessReceiptRecord,
    ConversationHeldOutV4PayloadWitnessResult,
    load_v4_payload_witness_publication,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation import (
    V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation_provenance import (
    ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
    load_v4_runtime_task_annotation_provenance_publication,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_blueprint import (
    ConversationHeldOutV4RuntimeTaskBlueprintError,
    ConversationHeldOutV4RuntimeTaskBlueprintPublication,
    ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord,
    ConversationHeldOutV4RuntimeTaskBlueprintResult,
    ConversationHeldOutV4RuntimeTaskBlueprintTurn,
    load_v4_runtime_task_blueprint_publication,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerFramedStreamError,
    IntegerFramedStreamFooter,
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    encoded_integer_tuple_size,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError,
    KRunFileDigest,
    KRunRoot,
    capture_plain_file_identity,
    ensure_normal_relative_directory,
    open_exclusive_binary,
    open_plain_binary,
    publish_manifest_last,
    require_created_run_root,
    require_disjoint_run_roots,
    require_exact_file_closure,
    require_fresh_empty_run_root,
    require_plain_file_identity,
    sha256_plain_file,
)


V4_RUNTIME_TASK_CAPSULE_ADAPTER_BUDGET_SCHEMA = 2
V4_RUNTIME_TASK_CAPSULE_ADAPTER_METADATA_SCHEMA = 1
V4_RUNTIME_TASK_CAPSULE_ADAPTER_RECORD_SCHEMA = 1
V4_RUNTIME_TASK_CAPSULE_ADAPTER_RECEIPT_SCHEMA = 1
V4_RUNTIME_TASK_CAPSULE_ADAPTER_RESULT_SCHEMA = 1
V4_RUNTIME_TASK_CAPSULE_ADAPTER_MANIFEST_SCHEMA = 1
V4_RUNTIME_TASK_CAPSULE_ADAPTER_STATUS_TEST_ONLY = "P3_C1C_TYPED_CAPSULE_TEST_ONLY"
V4_RUNTIME_TASK_CAPSULE_ADAPTER_METADATA_ORIGIN_TEST_TRANSPORT_AUTHORED = (
    V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED
)

_STAGING_RECEIPT_FILE = Path("candidate") / "receipt.pifrs"
_WORK_RECEIPT_FILE = Path("verified") / "receipt.pifrs"
_RECEIPT_FILE = Path("adapter") / "receipt.pifrs"
_MANIFEST_FILE = Path("manifest.pii")
_CAPSULE_RELATIVE_ROOT = Path("external-capsule")
_SHA256_SIZE = hashlib.sha256().digest_size

_DOMAIN_ACTIVE = b"dlg05.v4.p3c1c.active.v1\x00"
_DOMAIN_READBACK = b"dlg05.v4.p3c1c.readback.v1\x00"
_DOMAIN_PAYLOAD = b"dlg05.v4.p3c1c.payload.v1\x00"
_DOMAIN_ANNOTATION = b"dlg05.v4.p3c1c.annotation.v1\x00"
_DOMAIN_BLUEPRINT = b"dlg05.v4.p3c1c.blueprint.v1\x00"
_DOMAIN_ADAPTER_CODE = b"dlg05.v4.p3c1c.adapter-code.v1\x00"
_DOMAIN_PRODUCER = b"dlg05.v4.p3c1c.producer.v1\x00"
_DOMAIN_DEPENDENCIES = b"dlg05.v4.p3c1c.dependencies.v1\x00"
_DOMAIN_CASE = b"dlg05.v4.p3c1c.case.v1\x00"
_DOMAIN_TURN = b"dlg05.v4.p3c1c.turn.v1\x00"
_DOMAIN_BLUEPRINT_RECORD = b"dlg05.v4.p3c1c.blueprint-record.v1\x00"
_DOMAIN_PAYLOAD_RECORD = b"dlg05.v4.p3c1c.payload-record.v1\x00"
_DOMAIN_SOURCE_METADATA = b"dlg05.v4.p3c1c.source-metadata.v1\x00"
_DOMAIN_SOURCE_RECORD = b"dlg05.v4.p3c1c.source-record.v1\x00"
_DOMAIN_RUNTIME_INPUT = b"dlg05.v4.p3c1c.runtime-input.v1\x00"
_DOMAIN_RUNTIME_CAPSULE = b"dlg05.v4.p3c1c.runtime-capsule.v1\x00"


# object-model: exception
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(RuntimeError):
    """P3-C1c replay、typed capsule、receipt 或 K-root 闭合失败。"""


def _fail(message: str) -> None:
    """统一产生不携带 payload 或本机路径的 fail-closed 错误。"""
    raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(message)


def _require_digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """要求完整 SHA-256 byte tuple，拒绝截断或 bool。"""
    if (not isinstance(value, tuple) or len(value) != _SHA256_SIZE
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ValueError(f"{label} must be a SHA-256 byte tuple")
    return value


def _require_positive(value: int, *, label: str) -> int:
    """要求资源预算和 ordinal 为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive strict integer")
    return value


def _text_scalars(value: str, *, label: str) -> tuple[int, ...]:
    """将公开小元数据转为非空规范 Unicode scalar tuple。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty str")
    result = tuple(ord(item) for item in value)
    if any(item < 0 or item > 0x10FFFF or 0xD800 <= item <= 0xDFFF
           for item in result):
        raise ValueError(f"{label} contains invalid Unicode scalars")
    return result


def _text_from_scalars(value: tuple[int, ...], *, label: str) -> str:
    """从规范 scalar tuple 恢复公开小元数据。"""
    strict_integer_tuple(value, label=label)
    try:
        result = "".join(chr(item) for item in value)
    except ValueError as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            f"{label} is not Unicode") from exc
    if not result or any(0xD800 <= item <= 0xDFFF for item in value):
        _fail(f"{label} is invalid")
    return result


def _require_scalars(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """要求 license/attribution 为非空、无 surrogate 的 scalar tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} must be a nonempty scalar tuple")
    strict_integer_tuple(value, label=label)
    if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in value):
        raise ValueError(f"{label} contains invalid Unicode scalars")
    try:
        "".join(chr(item) for item in value)
    except ValueError as exc:
        raise ValueError(f"{label} is not Unicode") from exc
    return value


def _stage_name(value: str) -> str:
    """限制 logical stage 为短 ASCII token，避免隐式环境标识。"""
    if (not isinstance(value, str) or not value or len(value) > 48
            or not value[0].isascii() or not value[0].isalnum()
            or any(not item.isascii() or not (item.isalnum() or item in {"-", "_", "."})
                   for item in value)):
        raise ValueError("P3-C1c logical stage name is invalid")
    return value


def _identity_digest(
        value: tuple[int, ...], *, domain: bytes, label: str,
        ) -> tuple[int, ...]:
    """为公开 receipt 生成域分离 typed identity digest。"""
    strict_integer_tuple(value, label=label)
    if not isinstance(domain, bytes) or not domain:
        raise TypeError("P3-C1c identity digest domain is invalid")
    return tuple(hashlib.sha256(domain + encode_integer_tuple(value)).digest())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration:
    """一条 replay source metadata 的显式 test-only 来源声明。"""

    source_ref: SourceRef
    origin_status: str
    metadata_source_identity: ProtocolKey
    metadata_revision_identity: ProtocolKey
    frozen_source_manifest_identity: ProtocolKey
    constructing_code_identity: ProtocolKey
    transform_code_identity: ProtocolKey
    applicable_domain_identity: ProtocolKey

    def __post_init__(self) -> None:
        """限制当前 slice 只能使用已声明的 test transport metadata。"""
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("P3-C1c metadata source_ref type is invalid")
        if (self.origin_status
                != V4_RUNTIME_TASK_CAPSULE_ADAPTER_METADATA_ORIGIN_TEST_TRANSPORT_AUTHORED):
            raise ValueError("P3-C1c metadata origin is not allowed")
        for label, value in (
                ("metadata source", self.metadata_source_identity),
                ("metadata revision", self.metadata_revision_identity),
                ("source manifest", self.frozen_source_manifest_identity),
                ("constructing code", self.constructing_code_identity),
                ("transform code", self.transform_code_identity),
                ("applicable domain", self.applicable_domain_identity)):
            if not isinstance(value, ProtocolKey):
                raise TypeError(f"P3-C1c {label} identity type is invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回仅受限内存使用的完整 metadata declaration identity。"""
        result = [V4_RUNTIME_TASK_CAPSULE_ADAPTER_METADATA_SCHEMA]
        for value in (
                self.source_ref.stable_key(),
                _text_scalars(self.origin_status, label="P3-C1c metadata origin"),
                self.metadata_source_identity.components,
                self.metadata_revision_identity.components,
                self.frozen_source_manifest_identity.components,
                self.constructing_code_identity.components,
                self.transform_code_identity.components,
                self.applicable_domain_identity.components):
            pack_key(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskPayloadReplayItem:
    """同次 replay 的一个完整 witness 与无法由 P3-A2 推导的 source metadata。"""

    witness: ConversationHeldOutV4PayloadWitness
    license_scalars: tuple[int, ...]
    attribution_scalars: tuple[int, ...]
    metadata_declaration: ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration

    def __post_init__(self) -> None:
        """使 metadata 与同一 witness 的 SourceRef/manifest 先在内存闭合。"""
        if not isinstance(self.witness, ConversationHeldOutV4PayloadWitness):
            raise TypeError("P3-C1c replay witness type is invalid")
        _require_scalars(self.license_scalars, label="P3-C1c replay license")
        _require_scalars(self.attribution_scalars, label="P3-C1c replay attribution")
        if not isinstance(self.metadata_declaration,
                          ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration):
            raise TypeError("P3-C1c replay metadata declaration type is invalid")
        if (self.metadata_declaration.source_ref != self.witness.source_ref
                or self.metadata_declaration.frozen_source_manifest_identity
                != self.witness.frozen_source_manifest_identity):
            raise ValueError("P3-C1c replay metadata binding drifted")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskPayloadReplayBinding:
    """结果保留的无 payload replay binding，供后续回读比较 metadata。"""

    witness_record_identity: tuple[int, ...]
    metadata_declaration: ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration

    def __post_init__(self) -> None:
        """保持公开 P3-A2 record identity 与受限 metadata declaration 的一对一关系。"""
        strict_integer_tuple(self.witness_record_identity,
                             label="P3-C1c replay witness record identity")
        if not isinstance(self.metadata_declaration,
                          ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration):
            raise TypeError("P3-C1c replay binding metadata type is invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回 payload-free replay binding identity。"""
        result: list[int] = []
        for value in (self.witness_record_identity,
                      self.metadata_declaration.stable_key()):
            pack_key(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget:
    """限制 C1c 上游回读、一次 replay、receipt 与 v1 interop payload 的资源。"""

    max_upstream_witness_count: int
    max_upstream_receipt_record_count: int
    max_upstream_stream_payload_bytes: int
    max_upstream_stream_physical_bytes: int
    max_turn_count: int
    max_total_content_bytes: int
    max_total_content_scalars: int
    max_receipt_record_payload_bytes: int
    max_receipt_payload_bytes: int
    max_receipt_physical_bytes: int
    max_manifest_bytes: int
    capsule_budget: ConversationHeldOutV4ExternalCapsuleBudget

    def __post_init__(self) -> None:
        """拒绝未受上限的 replay、公开 receipt 或 capsule transport。"""
        values = (
            self.max_upstream_witness_count,
            self.max_upstream_receipt_record_count,
            self.max_upstream_stream_payload_bytes,
            self.max_upstream_stream_physical_bytes,
            self.max_turn_count,
            self.max_total_content_bytes,
            self.max_total_content_scalars,
            self.max_receipt_record_payload_bytes,
            self.max_receipt_payload_bytes,
            self.max_receipt_physical_bytes,
            self.max_manifest_bytes,
        )
        for label, value in zip((
                "max_upstream_witness_count",
                "max_upstream_receipt_record_count",
                "max_upstream_stream_payload_bytes",
                "max_upstream_stream_physical_bytes",
                "max_turn_count",
                "max_total_content_bytes", "max_total_content_scalars",
                "max_receipt_record_payload_bytes", "max_receipt_payload_bytes",
                "max_receipt_physical_bytes", "max_manifest_bytes"), values):
            _require_positive(value, label=f"P3-C1c {label}")
        if self.max_receipt_record_payload_bytes > self.max_receipt_payload_bytes:
            raise ValueError("P3-C1c receipt record budget exceeds receipt budget")
        if self.max_upstream_witness_count > self.max_upstream_receipt_record_count:
            raise ValueError("P3-C1c selected witness budget exceeds upstream receipt budget")
        if not isinstance(self.capsule_budget, ConversationHeldOutV4ExternalCapsuleBudget):
            raise TypeError("P3-C1c capsule budget type is invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """返回版本化的固定预算字段及下游 v1 capsule budget。"""
        result = [V4_RUNTIME_TASK_CAPSULE_ADAPTER_BUDGET_SCHEMA]
        result.extend((
            self.max_upstream_witness_count,
            self.max_upstream_receipt_record_count,
            self.max_upstream_stream_payload_bytes,
            self.max_upstream_stream_physical_bytes,
            self.max_turn_count,
            self.max_total_content_bytes,
            self.max_total_content_scalars,
            self.max_receipt_record_payload_bytes,
            self.max_receipt_payload_bytes,
            self.max_receipt_physical_bytes,
            self.max_manifest_bytes,
        ))
        pack_key(result, self.capsule_budget.integer_tuple())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterCounts:
    """记录 turn、去重 witness 与实际 replay item 消费次数。"""

    turn_count: int
    train_turn_count: int
    held_out_turn_count: int
    unique_witness_count: int
    actual_payload_read_count: int
    content_byte_count: int
    content_scalar_count: int

    def __post_init__(self) -> None:
        """闭合 split、去重 witness 与一次 item 消费的诚实计数。"""
        values = self.integer_tuple()
        if any(type(item) is not int or item < 0 for item in values):
            raise ValueError("P3-C1c counts must be nonnegative strict integers")
        if (self.turn_count == 0
                or self.turn_count != self.train_turn_count + self.held_out_turn_count
                or self.unique_witness_count == 0
                or self.unique_witness_count > self.turn_count
                or self.actual_payload_read_count != self.unique_witness_count
                or self.content_byte_count == 0
                or self.content_scalar_count == 0):
            raise ValueError("P3-C1c counts do not close")

    def integer_tuple(self) -> tuple[int, ...]:
        """返回固定顺序的公开聚合计数。"""
        return (
            self.turn_count,
            self.train_turn_count,
            self.held_out_turn_count,
            self.unique_witness_count,
            self.actual_payload_read_count,
            self.content_byte_count,
            self.content_scalar_count,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord:
    """一条无正文 C1c turn envelope，保留 C1b/P3-A2 到 RuntimeInput 的摘要绑定。"""

    case_key_identity_sha256: tuple[int, ...]
    turn_key_identity_sha256: tuple[int, ...]
    ordinal: int
    blueprint_receipt_record_identity_sha256: tuple[int, ...]
    blueprint_turn_identity_sha256: tuple[int, ...]
    annotation_witness_identity_sha256: tuple[int, ...]
    payload_witness_identity_sha256: tuple[int, ...]
    source_ref: SourceRef
    content_sha256: tuple[int, ...]
    content_byte_count: int
    content_scalar_count: int
    source_span_byte_start: int
    source_span_byte_end: int
    source_span_scalar_start: int
    source_span_scalar_end: int
    split: str
    license_partition: str
    annotation_origin_status: str
    source_metadata_declaration_identity_sha256: tuple[int, ...]
    source_record_identity_sha256: tuple[int, ...]
    runtime_input_identity_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝 receipt 漂移、正文偷渡和不完整的 turn/source binding。"""
        for label, value in (
                ("case key", self.case_key_identity_sha256),
                ("turn key", self.turn_key_identity_sha256),
                ("blueprint receipt record", self.blueprint_receipt_record_identity_sha256),
                ("blueprint turn", self.blueprint_turn_identity_sha256),
                ("annotation witness", self.annotation_witness_identity_sha256),
                ("payload witness", self.payload_witness_identity_sha256),
                ("source metadata declaration",
                 self.source_metadata_declaration_identity_sha256),
                ("source record", self.source_record_identity_sha256),
                ("runtime input", self.runtime_input_identity_sha256),
                ("content", self.content_sha256)):
            _require_digest(value, label=f"P3-C1c receipt {label} identity")
        _require_positive(self.ordinal, label="P3-C1c receipt ordinal")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("P3-C1c receipt source_ref type is invalid")
        for label, value in (("byte count", self.content_byte_count),
                             ("scalar count", self.content_scalar_count)):
            _require_positive(value, label=f"P3-C1c receipt {label}")
        spans = (
            self.source_span_byte_start,
            self.source_span_byte_end,
            self.source_span_scalar_start,
            self.source_span_scalar_end,
        )
        if (any(type(item) is not int or item < 0 for item in spans)
                or self.source_span_byte_start >= self.source_span_byte_end
                or self.source_span_byte_end > self.content_byte_count
                or self.source_span_scalar_start >= self.source_span_scalar_end
                or self.source_span_scalar_end > self.content_scalar_count):
            raise ValueError("P3-C1c receipt source span is invalid")
        if self.split not in {"train", "held_out"}:
            raise ValueError("P3-C1c receipt split is invalid")
        _text_scalars(self.license_partition, label="P3-C1c receipt license partition")
        if (self.annotation_origin_status
                != V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED):
            raise ValueError("P3-C1c receipt annotation origin is invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """编码固定字段顺序的无正文 P0 record。"""
        result = [V4_RUNTIME_TASK_CAPSULE_ADAPTER_RECORD_SCHEMA]
        for value in (
                self.case_key_identity_sha256,
                self.turn_key_identity_sha256,
                (self.ordinal,),
                self.blueprint_receipt_record_identity_sha256,
                self.blueprint_turn_identity_sha256,
                self.annotation_witness_identity_sha256,
                self.payload_witness_identity_sha256,
                self.source_ref.stable_key(),
                self.content_sha256,
                (self.content_byte_count, self.content_scalar_count),
                (self.source_span_byte_start, self.source_span_byte_end,
                 self.source_span_scalar_start, self.source_span_scalar_end),
                _text_scalars(self.split, label="P3-C1c receipt split"),
                _text_scalars(self.license_partition,
                              label="P3-C1c receipt license partition"),
                _text_scalars(self.annotation_origin_status,
                              label="P3-C1c receipt annotation origin"),
                self.source_metadata_declaration_identity_sha256,
                self.source_record_identity_sha256,
                self.runtime_input_identity_sha256):
            pack_key(result, value)
        return tuple(result)

    @classmethod
    def from_integer_stream(
            cls, value: tuple[int, ...],
            ) -> "ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord":
        """回读公开 P0 record 并拒绝截断、非规范或字段偷换。"""
        try:
            reader = IntegerStreamReader(value)
            if reader.read_positive(label="P3-C1c receipt schema") != V4_RUNTIME_TASK_CAPSULE_ADAPTER_RECORD_SCHEMA:
                _fail("P3-C1c receipt schema is incompatible")
            case_key = reader.read_key(label="P3-C1c receipt case key")
            turn_key = reader.read_key(label="P3-C1c receipt turn key")
            ordinal = reader.read_key(label="P3-C1c receipt ordinal")
            if len(ordinal) != 1:
                _fail("P3-C1c receipt ordinal is invalid")
            blueprint_record = reader.read_key(label="P3-C1c receipt blueprint record")
            blueprint_turn = reader.read_key(label="P3-C1c receipt blueprint turn")
            annotation_witness = reader.read_key(label="P3-C1c receipt annotation witness")
            payload_witness = reader.read_key(label="P3-C1c receipt payload witness")
            source_ref = SourceRef.from_stable_key(reader.read_key(
                label="P3-C1c receipt source_ref"))
            content_sha256 = reader.read_key(label="P3-C1c receipt content hash")
            counts = reader.read_key(label="P3-C1c receipt content counts")
            if len(counts) != 2:
                _fail("P3-C1c receipt content counts are invalid")
            spans = reader.read_key(label="P3-C1c receipt source spans")
            if len(spans) != 4:
                _fail("P3-C1c receipt source spans are invalid")
            split = _text_from_scalars(reader.read_key(label="P3-C1c receipt split"),
                                       label="P3-C1c receipt split")
            license_partition = _text_from_scalars(
                reader.read_key(label="P3-C1c receipt license partition"),
                label="P3-C1c receipt license partition")
            annotation_origin = _text_from_scalars(
                reader.read_key(label="P3-C1c receipt annotation origin"),
                label="P3-C1c receipt annotation origin")
            metadata = reader.read_key(label="P3-C1c receipt metadata declaration")
            source_record = reader.read_key(label="P3-C1c receipt source record")
            runtime_input = reader.read_key(label="P3-C1c receipt runtime input")
            result = cls(
                case_key, turn_key, ordinal[0], blueprint_record, blueprint_turn,
                annotation_witness, payload_witness, source_ref, content_sha256,
                counts[0], counts[1], *spans, split, license_partition,
                annotation_origin, metadata, source_record, runtime_input)
            reader.finish()
        except (IntegerCodecError, TypeError, ValueError, IndexError) as exc:
            raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
                "P3-C1c receipt record is corrupt") from exc
        if result.integer_stream() != value:
            _fail("P3-C1c receipt record is not canonical")
        return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptStream:
    """C1c P0 receipt 的相对路径、footer 与物理文件 identity。"""

    relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """限制 descriptor 指向单一相对普通文件。"""
        if (not isinstance(self.relative_path, Path) or self.relative_path.is_absolute()
                or self.relative_path.drive or self.relative_path.root
                or not self.relative_path.parts or ".." in self.relative_path.parts
                or not isinstance(self.p0_footer, IntegerFramedStreamFooter)
                or not isinstance(self.physical, KRunFileDigest)
                or self.physical.byte_count <= 0):
            raise ValueError("P3-C1c receipt stream identity is invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """返回可由 manifest 绑定的 receipt stream identity。"""
        result: list[int] = []
        for value in (
                _text_scalars(self.relative_path.as_posix(),
                              label="P3-C1c receipt path"),
                self.p0_footer.integer_tuple(),
                (self.physical.byte_count, *self.physical.sha256)):
            pack_key(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceipt:
    """绑定全部上游、adapter、capsule identity 与受预算聚合计数的公开 receipt。"""

    active_intake_identity_sha256: tuple[int, ...]
    readback_identity_sha256: tuple[int, ...]
    payload_witness_identity_sha256: tuple[int, ...]
    annotation_provenance_identity_sha256: tuple[int, ...]
    blueprint_identity_sha256: tuple[int, ...]
    adapter_code_identity_sha256: tuple[int, ...]
    producer_identity_sha256: tuple[int, ...]
    dependency_identity_sha256: tuple[int, ...]
    budget: ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget
    counts: ConversationHeldOutV4RuntimeTaskCapsuleAdapterCounts
    capsule_manifest_sha256: tuple[int, ...]
    runtime_capsule_identity_sha256: tuple[int, ...]
    status: str

    def __post_init__(self) -> None:
        """拒绝任何缺失上游、payload identity、预算或越级状态。"""
        for label, value in (
                ("active intake", self.active_intake_identity_sha256),
                ("readback", self.readback_identity_sha256),
                ("payload witness", self.payload_witness_identity_sha256),
                ("annotation provenance", self.annotation_provenance_identity_sha256),
                ("blueprint", self.blueprint_identity_sha256),
                ("adapter code", self.adapter_code_identity_sha256),
                ("producer", self.producer_identity_sha256),
                ("dependencies", self.dependency_identity_sha256),
                ("capsule manifest", self.capsule_manifest_sha256),
                ("runtime capsule", self.runtime_capsule_identity_sha256)):
            _require_digest(value, label=f"P3-C1c receipt {label} identity")
        if (not isinstance(self.budget, ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget)
                or not isinstance(self.counts, ConversationHeldOutV4RuntimeTaskCapsuleAdapterCounts)
                or self.status != V4_RUNTIME_TASK_CAPSULE_ADAPTER_STATUS_TEST_ONLY):
            raise ValueError("P3-C1c receipt fields are invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """编码不含 root、payload 或 typed key 原值的全局 receipt。"""
        result = [V4_RUNTIME_TASK_CAPSULE_ADAPTER_RECEIPT_SCHEMA]
        for value in (
                self.active_intake_identity_sha256,
                self.readback_identity_sha256,
                self.payload_witness_identity_sha256,
                self.annotation_provenance_identity_sha256,
                self.blueprint_identity_sha256,
                self.adapter_code_identity_sha256,
                self.producer_identity_sha256,
                self.dependency_identity_sha256,
                self.budget.integer_stream(),
                self.counts.integer_tuple(),
                self.capsule_manifest_sha256,
                self.runtime_capsule_identity_sha256,
                _text_scalars(self.status, label="P3-C1c receipt status")):
            pack_key(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleDescriptor:
    """C1c 交给后续 gate 的 payload-free capsule 定位与安全绑定摘要。"""

    parent_run_root: KRunRoot
    relative_root: Path
    manifest_sha256: tuple[int, ...]
    runtime_capsule_identity_sha256: tuple[int, ...]
    turn_count: int

    def __post_init__(self) -> None:
        """固定 child、test transport 与安全摘要，拒绝任何自由路径或裸 v1 描述。"""
        if (not isinstance(self.parent_run_root, KRunRoot)
                or self.relative_root != _CAPSULE_RELATIVE_ROOT
                or self.relative_root.is_absolute()
                or self.relative_root.drive
                or self.relative_root.root
                or ".." in self.relative_root.parts
                or not self.parent_run_root.test_transport):
            raise ValueError("P3-C1c capsule descriptor root is invalid")
        _require_digest(self.manifest_sha256,
                        label="P3-C1c capsule descriptor manifest")
        _require_digest(self.runtime_capsule_identity_sha256,
                        label="P3-C1c capsule descriptor runtime identity")
        _require_positive(self.turn_count, label="P3-C1c capsule descriptor turn count")

    def stable_key(self) -> tuple[int, ...]:
        """仅编码固定 relative child、摘要与计数，绝不把本机 path 纳入稳定身份。"""
        result: list[int] = []
        for item in (
                _text_scalars(self.relative_root.as_posix(),
                              label="P3-C1c capsule descriptor relative root"),
                self.manifest_sha256,
                self.runtime_capsule_identity_sha256,
                (self.turn_count,)):
            pack_key(result, item)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput:
    """P3-C1c 唯一入口；所有 payload 与 metadata 都必须经 explicit replay factory。"""

    active_intake: ConversationHeldOutV4ActiveIntakeResult
    readback: ConversationHeldOutV4ActiveRosterReadbackResult
    payload_witness: ConversationHeldOutV4PayloadWitnessResult
    blueprint: ConversationHeldOutV4RuntimeTaskBlueprintResult
    payload_replay_factory: Callable[[], Iterable[ConversationHeldOutV4RuntimeTaskPayloadReplayItem]]
    producer: ConversationHeldOutV4ExternalProducer
    dependencies: ConversationHeldOutV4DependencyBinding
    adapter_code_identity: ProtocolKey
    staging_run_root: KRunRoot
    work_run_root: KRunRoot
    publication_run_root: KRunRoot
    capsule_parent_run_root: KRunRoot
    budget: ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget
    logical_stage_name: str

    def __post_init__(self) -> None:
        """拒绝隐式 payload、默认 producer、未受限 root 或不受预算的 adapter 调用。"""
        if (not isinstance(self.active_intake, ConversationHeldOutV4ActiveIntakeResult)
                or not isinstance(self.readback, ConversationHeldOutV4ActiveRosterReadbackResult)
                or not isinstance(self.payload_witness, ConversationHeldOutV4PayloadWitnessResult)
                or not isinstance(self.blueprint, ConversationHeldOutV4RuntimeTaskBlueprintResult)
                or not callable(self.payload_replay_factory)
                or not isinstance(self.producer, ConversationHeldOutV4ExternalProducer)
                or not isinstance(self.dependencies, ConversationHeldOutV4DependencyBinding)
                or not isinstance(self.adapter_code_identity, ProtocolKey)
                or not isinstance(self.budget, ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget)):
            raise TypeError("P3-C1c input types are invalid")
        for root in (
                self.staging_run_root,
                self.work_run_root,
                self.publication_run_root,
                self.capsule_parent_run_root):
            if not isinstance(root, KRunRoot):
                raise TypeError("P3-C1c run root type is invalid")
        _stage_name(self.logical_stage_name)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult:
    """C1c publication、metadata replay binding 与 capsule descriptor；不保留 raw payload。"""

    active_intake: ConversationHeldOutV4ActiveIntakeResult
    readback: ConversationHeldOutV4ActiveRosterReadbackResult
    payload_witness: ConversationHeldOutV4PayloadWitnessResult
    blueprint: ConversationHeldOutV4RuntimeTaskBlueprintResult
    publication_run_root: KRunRoot
    receipt: ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceipt
    receipt_stream: ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptStream
    manifest_sha256: tuple[int, ...]
    capsule_descriptor: ConversationHeldOutV4RuntimeTaskCapsuleDescriptor
    replay_bindings: tuple[ConversationHeldOutV4RuntimeTaskPayloadReplayBinding, ...]

    def __post_init__(self) -> None:
        """只保存无正文 bound capability；C1c 不把 runtime capsule 交给 consumer。"""
        if (not isinstance(self.active_intake, ConversationHeldOutV4ActiveIntakeResult)
                or not isinstance(self.readback, ConversationHeldOutV4ActiveRosterReadbackResult)
                or not isinstance(self.payload_witness, ConversationHeldOutV4PayloadWitnessResult)
                or not isinstance(self.blueprint, ConversationHeldOutV4RuntimeTaskBlueprintResult)
                or not isinstance(self.publication_run_root, KRunRoot)
                or not isinstance(self.receipt, ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceipt)
                or not isinstance(self.receipt_stream,
                                  ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptStream)
                or not isinstance(self.capsule_descriptor,
                                  ConversationHeldOutV4RuntimeTaskCapsuleDescriptor)
                or not isinstance(self.replay_bindings, tuple)
                or any(not isinstance(item,
                                      ConversationHeldOutV4RuntimeTaskPayloadReplayBinding)
                       for item in self.replay_bindings)):
            raise TypeError("P3-C1c result types are invalid")
        if (self.receipt_stream.relative_path != _RECEIPT_FILE
                or self.receipt.counts.turn_count != self.capsule_descriptor.turn_count
                or self.receipt.capsule_manifest_sha256
                != self.capsule_descriptor.manifest_sha256
                or self.receipt.runtime_capsule_identity_sha256
                != self.capsule_descriptor.runtime_capsule_identity_sha256
                or len(self.replay_bindings)
                != self.receipt.counts.unique_witness_count
                or len({item.witness_record_identity for item in self.replay_bindings})
                != len(self.replay_bindings)):
            raise ValueError("P3-C1c result inventory or capsule origin drifted")
        _require_digest(self.manifest_sha256, label="P3-C1c manifest hash")
        if (not self.publication_run_root.test_transport
                or not self.capsule_descriptor.parent_run_root.test_transport
                or self.receipt.status != V4_RUNTIME_TASK_CAPSULE_ADAPTER_STATUS_TEST_ONLY):
            raise ValueError("P3-C1c result transport status drifted")

    def stable_key(self) -> tuple[int, ...]:
        """返回仅依赖公开 receipt、manifest 与 capsule identity 的跨阶段稳定键。"""
        result = [V4_RUNTIME_TASK_CAPSULE_ADAPTER_RESULT_SCHEMA]
        for value in (
                self.receipt.integer_stream(),
                self.receipt_stream.integer_stream(),
                self.manifest_sha256,
                self.receipt.capsule_manifest_sha256,
                self.receipt.runtime_capsule_identity_sha256):
            pack_key(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskCapsuleAdapterPublication:
    """C1c 受限 loader 结果；records 无正文，result 仅是 bound root capability。"""

    result: ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult
    records: tuple[ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord, ...]

    def __post_init__(self) -> None:
        """限制 loader 只返回已封存的连续 turn receipt inventory。"""
        if (not isinstance(self.result, ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult)
                or not isinstance(self.records, tuple)
                or any(not isinstance(item,
                                      ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord)
                       for item in self.records)
                or len(self.records) != self.result.receipt.counts.turn_count
                or tuple(item.ordinal for item in self.records)
                != tuple(range(1, len(self.records) + 1))):
            raise TypeError("P3-C1c publication types are invalid")


def _validate_roots(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput,
        *,
        active: ConversationHeldOutV4ActiveIntakeResult,
        readback: ConversationHeldOutV4ActiveRosterReadbackResult,
        payload_witness: ConversationHeldOutV4PayloadWitnessResult,
        blueprint: ConversationHeldOutV4RuntimeTaskBlueprintResult,
        ) -> None:
    """要求本切片仅走显式 test transport，并隔离所有新输出根。"""
    annotation = blueprint.annotation_provenance
    outputs = (
        value.staging_run_root,
        value.work_run_root,
        value.publication_run_root,
        value.capsule_parent_run_root,
    )
    inputs = (
        active.publication_run_root,
        readback.publication_run_root,
        payload_witness.publication_run_root,
        annotation.publication_run_root,
        blueprint.publication_run_root,
    )
    if not all(root.test_transport for root in (*outputs, *inputs)):
        _fail("P3-C1c production transport has no qualified source-metadata adapter")
    for index, left in enumerate(outputs):
        for right in outputs[index + 1:]:
            require_disjoint_run_roots(left, right, label="P3-C1c output roots")
    for output in outputs:
        for source in inputs:
            require_disjoint_run_roots(output, source, label="P3-C1c input/output roots")
        require_fresh_empty_run_root(output, label="P3-C1c fresh output root")


def _preflight_upstream_budgets(
        *,
        active: ConversationHeldOutV4ActiveIntakeResult,
        payload_witness: ConversationHeldOutV4PayloadWitnessResult,
        blueprint: ConversationHeldOutV4RuntimeTaskBlueprintResult,
        budget: ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget,
        ) -> None:
    """在打开任一上游 P0 前用已封存声明拒绝超出 C1c 读取预算的链。

    当前 P3-A2 是顺序 P0 ledger，没有 manifest-bound offset index；因此 C1c 必须先
    限制完整上游 inventory，不能把小 C1b 子集误称为避免了全量 ledger 读取。未来若要
    支持大 ledger 的选择性 replay，必须另建可审计的索引和选择性 reader。
    """
    upstream_streams = (
        ("P3-A mapping", active.counts.mapping_record_count,
         active.mapping_stream),
        ("P3-A2 witness", payload_witness.receipt.counts.witness_count,
         payload_witness.receipt_stream),
    )
    for label, declared_count, stream in upstream_streams:
        if (declared_count > budget.max_upstream_receipt_record_count
                or stream.p0_footer.record_count > budget.max_upstream_receipt_record_count):
            _fail(f"P3-C1c {label} record count exceeds upstream budget")
        if (stream.p0_footer.total_payload_bytes
                > budget.max_upstream_stream_payload_bytes
                or stream.physical.byte_count
                > budget.max_upstream_stream_physical_bytes):
            _fail(f"P3-C1c {label} stream bytes exceed upstream budget")
    annotation = blueprint.annotation_provenance
    turn_streams = (
        ("P3-C1a annotation", annotation.receipt.annotation_witness_count,
         annotation.receipt_stream),
        ("P3-C1b blueprint", blueprint.receipt.counts.turn_count,
         blueprint.receipt_stream),
    )
    for label, declared_count, stream in turn_streams:
        if (declared_count > budget.max_turn_count
                or stream.p0_footer.record_count > budget.max_turn_count):
            _fail(f"P3-C1c {label} record count exceeds turn budget")
        if (stream.p0_footer.total_payload_bytes
                > budget.max_upstream_stream_payload_bytes
                or stream.physical.byte_count
                > budget.max_upstream_stream_physical_bytes):
            _fail(f"P3-C1c {label} stream bytes exceed upstream budget")


def _load_upstreams(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput,
        ) -> tuple[
            ConversationHeldOutV4ActiveIntakeResult,
            ConversationHeldOutV4ActiveRosterReadbackResult,
            ConversationHeldOutV4PayloadWitnessResult,
            ConversationHeldOutV4RuntimeTaskBlueprintPublication,
        ]:
    """完整回读每级 public evidence，并拒绝显式输入与嵌套 binding 漂移。"""
    _preflight_upstream_budgets(
        active=value.active_intake,
        payload_witness=value.payload_witness,
        blueprint=value.blueprint,
        budget=value.budget,
    )
    try:
        active_publication = load_v4_active_intake_publication(value.active_intake)
        readback = revalidate_v4_active_roster_bidirectional_readback(value.readback)
        payload_publication = load_v4_payload_witness_publication(
            value.payload_witness,
            max_record_count=value.budget.max_upstream_receipt_record_count,
            max_total_payload_bytes=value.budget.max_upstream_stream_payload_bytes,
            max_physical_bytes=value.budget.max_upstream_stream_physical_bytes,
        )
        annotation_publication = load_v4_runtime_task_annotation_provenance_publication(
            value.blueprint.annotation_provenance)
        blueprint_publication = load_v4_runtime_task_blueprint_publication(value.blueprint)
    except (ConversationHeldOutV4ActiveIntakeError,
            ConversationHeldOutV4ActiveRosterReadbackError,
            ConversationHeldOutV4PayloadWitnessLedgerError,
            ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
            ConversationHeldOutV4RuntimeTaskBlueprintError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c upstream publication readback failed") from exc
    active = active_publication.result
    payload_witness = payload_publication.result
    blueprint = blueprint_publication.result
    annotation = annotation_publication.result
    if (active != value.active_intake
            or readback != value.readback
            or payload_witness != value.payload_witness
            or blueprint != value.blueprint
            or annotation != blueprint.annotation_provenance
            or annotation.payload_witness != payload_witness):
        _fail("P3-C1c explicit upstream result drifted")
    if (readback.receipt.active_intake_stable_key != active.stable_key()
            or payload_witness.receipt.active_intake_stable_key != active.stable_key()
            or payload_witness.receipt.readback_stable_key != readback.stable_key()
            or blueprint.annotation_provenance.payload_witness.stable_key()
            != payload_witness.stable_key()):
        _fail("P3-C1c P3-A/P3-B/P3-A2/C1a bindings do not close")
    return active, readback, payload_witness, blueprint_publication


def _mapping_for_payload_record(
        record: ConversationHeldOutV4PayloadWitnessReceiptRecord,
        *, active_mappings: dict[tuple[int, ...], ConversationHeldOutV4ActiveIntakeMapping],
        ) -> ConversationHeldOutV4ActiveIntakeMapping:
    """从 P3-A2 record 恢复完整 P3-A mapping，并核对全部共享来源字段。"""
    try:
        mapping = ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(
            record.mapping_identity)
    except (ConversationHeldOutV4ActiveIntakeError, IntegerCodecError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c P3-A2 mapping identity is corrupt") from exc
    current = active_mappings.get(mapping.stable_key())
    if current != mapping:
        _fail("P3-C1c P3-A2 mapping is not a current P3-A member")
    if (mapping.record_kind != record.record_kind
            or mapping.record_key != record.witness_record_key
            or mapping.source_ref != record.source_ref
            or mapping.leaf != record.leaf
            or mapping.content_sha256 != record.content_sha256):
        _fail("P3-C1c P3-A2 mapping receipt binding drifted")
    return mapping


def _validate_blueprint_pair(
        record: ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord,
        turn: ConversationHeldOutV4RuntimeTaskBlueprintTurn,
        *, payload_records: dict[tuple[int, ...], ConversationHeldOutV4PayloadWitnessReceiptRecord],
        ) -> ConversationHeldOutV4PayloadWitnessReceiptRecord:
    """重放 C1b 的公开/in-memory pairing，且只允许其引用 sealed P3-A2 record。"""
    if (record.ordinal != turn.ordinal
            or record.source_ref != turn.source_ref
            or record.content_sha256 != turn.content_sha256
            or record.content_byte_count != turn.content_byte_count
            or record.content_scalar_count != turn.content_scalar_count
            or record.source_span_byte_start != turn.source_span_byte_start
            or record.source_span_byte_end != turn.source_span_byte_end
            or record.source_span_scalar_start != turn.source_span_scalar_start
            or record.source_span_scalar_end != turn.source_span_scalar_end
            or record.split != turn.split
            or record.license_partition != turn.license_partition
            or record.annotation_origin_status != turn.annotation_origin_status):
        _fail("P3-C1c C1b receipt/in-memory turn binding drifted")
    payload_record = payload_records.get(turn.witness_record_identity)
    if payload_record is None:
        _fail("P3-C1c C1b turn references an undeclared P3-A2 witness")
    if (payload_record.source_ref != turn.source_ref
            or payload_record.content_sha256 != turn.content_sha256
            or payload_record.content_byte_count != turn.content_byte_count
            or payload_record.content_scalar_count != turn.content_scalar_count
            or payload_record.source_span_byte_start != turn.source_span_byte_start
            or payload_record.source_span_byte_end != turn.source_span_byte_end
            or payload_record.source_span_scalar_start != turn.source_span_scalar_start
            or payload_record.source_span_scalar_end != turn.source_span_scalar_end):
        _fail("P3-C1c C1b/P3-A2 content or span binding drifted")
    return payload_record


def _validate_replay_item(
        item: ConversationHeldOutV4RuntimeTaskPayloadReplayItem,
        *,
        record: ConversationHeldOutV4PayloadWitnessReceiptRecord,
        mapping: ConversationHeldOutV4ActiveIntakeMapping,
        ) -> tuple[ConversationHeldOutV4SourceRecord, bytes, tuple[int, ...]]:
    """逐字段闭合 factory payload、P3-A2 record 与当前 P3-A mapping。"""
    if not isinstance(item, ConversationHeldOutV4RuntimeTaskPayloadReplayItem):
        raise TypeError("P3-C1c replay factory yielded an invalid item")
    witness = item.witness
    if (witness.active_record_key != record.witness_record_key
            or witness.record_kind != record.record_kind
            or witness.source_ref != record.source_ref
            or witness.leaf != record.leaf
            or witness.snapshot != mapping.snapshot
            or witness.split != mapping.split
            or witness.license_partition != mapping.license_partition
            or witness.consumed_input_identity != mapping.leaf.consumed_input_identity
            or witness.source_uri_scalars != mapping.snapshot.official_uri_scalars
            or witness.source_span_byte_start != record.source_span_byte_start
            or witness.source_span_byte_end != record.source_span_byte_end
            or witness.source_span_scalar_start != record.source_span_scalar_start
            or witness.source_span_scalar_end != record.source_span_scalar_end
            or witness.encoding != record.encoding
            or witness.frozen_source_manifest_identity
            != record.frozen_source_manifest_identity
            or witness.source_payload_identity != record.source_payload_identity):
        _fail("P3-C1c replay witness binding drifted")
    try:
        payload, scalars = witness.canonical_bytes_and_scalars()
    except ConversationHeldOutV4PayloadWitnessLedgerError as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c replay witness UTF-8 validation failed") from exc
    if (tuple(hashlib.sha256(payload).digest()) != record.content_sha256
            or len(payload) != record.content_byte_count
            or len(scalars) != record.content_scalar_count
            or witness.source_span_byte_end > len(payload)
            or witness.source_span_scalar_end > len(scalars)):
        _fail("P3-C1c replay witness content identity drifted")
    if (_text_scalars(mapping.license_partition, label="P3-C1c mapping license")
            != item.license_scalars
            or item.metadata_declaration.source_ref != record.source_ref
            or item.metadata_declaration.frozen_source_manifest_identity
            != record.frozen_source_manifest_identity):
        _fail("P3-C1c replay metadata/license binding drifted")
    try:
        source_record = ConversationHeldOutV4SourceRecord(
            record.source_ref,
            scalars,
            record.content_sha256,
            item.license_scalars,
            item.attribution_scalars,
            witness.source_uri_scalars,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c replay SourceRecord construction failed") from exc
    return source_record, payload, scalars


def _consume_payload_replay(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput,
        *,
        payload_records: tuple[ConversationHeldOutV4PayloadWitnessReceiptRecord, ...],
        active_mappings: dict[tuple[int, ...], ConversationHeldOutV4ActiveIntakeMapping],
        turns: tuple[ConversationHeldOutV4RuntimeTaskBlueprintTurn, ...],
        ) -> tuple[
            dict[tuple[int, ...], ConversationHeldOutV4SourceRecord],
            dict[tuple[int, ...], ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration],
            ConversationHeldOutV4RuntimeTaskCapsuleAdapterCounts,
        ]:
    """按 P3-A2 sealed order 的 C1b 过滤子序列一次性消费同次 payload replay。"""
    required = {turn.witness_record_identity for turn in turns}
    if not required or len(required) > value.budget.max_upstream_witness_count:
        _fail("P3-C1c required P3-A2 witness count exceeds budget")
    expected_records = tuple(
        record for record in payload_records if record.integer_stream() in required)
    if (len(expected_records) != len(required)
            or {record.integer_stream() for record in expected_records} != required):
        _fail("P3-C1c required witness inventory is not a P3-A2 sealed subset")
    try:
        iterator = iter(value.payload_replay_factory())
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c payload replay factory is not iterable") from exc
    source_by_witness: dict[tuple[int, ...], ConversationHeldOutV4SourceRecord] = {}
    metadata_by_witness: dict[
        tuple[int, ...], ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration] = {}
    source_by_ref: dict[SourceRef, ConversationHeldOutV4SourceRecord] = {}
    metadata_by_ref: dict[
        SourceRef, ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration] = {}
    content_bytes = 0
    content_scalars = 0
    try:
        for record in expected_records:
            try:
                item = next(iterator)
            except StopIteration:
                _fail("P3-C1c payload replay is missing a required witness")
            mapping = _mapping_for_payload_record(
                record, active_mappings=active_mappings)
            source_record, payload, scalars = _validate_replay_item(
                item, record=record, mapping=mapping)
            identity = record.integer_stream()
            prior = source_by_ref.get(source_record.source)
            if prior is not None and prior != source_record:
                _fail("P3-C1c identical SourceRef replay metadata or payload drifted")
            prior_metadata = metadata_by_ref.get(source_record.source)
            if (prior_metadata is not None
                    and prior_metadata != item.metadata_declaration):
                _fail("P3-C1c identical SourceRef metadata declaration drifted")
            source_by_ref[source_record.source] = source_record
            metadata_by_ref[source_record.source] = item.metadata_declaration
            source_by_witness[identity] = source_record
            metadata_by_witness[identity] = item.metadata_declaration
            content_bytes += len(payload)
            content_scalars += len(scalars)
            if (content_bytes > value.budget.max_total_content_bytes
                    or content_scalars > value.budget.max_total_content_scalars):
                _fail("P3-C1c replay content exceeds budget")
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            _fail("P3-C1c payload replay has an extra or out-of-order witness")
    except ConversationHeldOutV4RuntimeTaskCapsuleAdapterError:
        raise
    except (ConversationHeldOutV4PayloadWitnessLedgerError, TypeError,
            ValueError, UnicodeError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c payload replay validation failed") from exc
    counts = ConversationHeldOutV4RuntimeTaskCapsuleAdapterCounts(
        len(turns),
        sum(turn.split == "train" for turn in turns),
        sum(turn.split == "held_out" for turn in turns),
        len(expected_records),
        len(expected_records),
        content_bytes,
        content_scalars,
    )
    return source_by_witness, metadata_by_witness, counts


def _runtime_input_for_turn(
        turn: ConversationHeldOutV4RuntimeTaskBlueprintTurn,
        *, source_record: ConversationHeldOutV4SourceRecord,
        ) -> ConversationHeldOutV4RuntimeInput:
    """仅逐字段展开 C1b typed task；不重建或推断问题、表达或 Evidence 语义。"""
    try:
        return ConversationHeldOutV4RuntimeInput(
            turn.case_key,
            turn.turn_key,
            turn.ordinal,
            turn.request,
            tuple(item.representation for item in turn.representations),
            (source_record,),
            tuple(ConversationHeldOutV4RuntimeEvidencePlan(
                item.target,
                item.competition_key,
                item.stances,
                item.source,
                item.source_span_start,
                item.source_span_end,
            ) for item in turn.evidence_plans),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c typed RuntimeInput conversion failed") from exc


def _receipt_record_for(
        *,
        blueprint_record: ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord,
        turn: ConversationHeldOutV4RuntimeTaskBlueprintTurn,
        payload_record: ConversationHeldOutV4PayloadWitnessReceiptRecord,
        metadata: ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration,
        source_record: ConversationHeldOutV4SourceRecord,
        runtime_input: ConversationHeldOutV4RuntimeInput,
        ) -> ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord:
    """将同一 turn 的闭合链投影为不携带正文或完整 key 的 C1c P0 record。"""
    return ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord(
        _identity_digest(turn.case_key.components, domain=_DOMAIN_CASE,
                         label="P3-C1c case key"),
        _identity_digest(turn.turn_key.components, domain=_DOMAIN_TURN,
                         label="P3-C1c turn key"),
        turn.ordinal,
        _identity_digest(blueprint_record.integer_stream(),
                         domain=_DOMAIN_BLUEPRINT_RECORD,
                         label="P3-C1c blueprint record"),
        _identity_digest(turn.stable_key(), domain=_DOMAIN_TURN,
                         label="P3-C1c blueprint turn"),
        blueprint_record.annotation_witness_identity_sha256,
        _identity_digest(payload_record.integer_stream(),
                         domain=_DOMAIN_PAYLOAD_RECORD,
                         label="P3-C1c payload record"),
        turn.source_ref,
        turn.content_sha256,
        turn.content_byte_count,
        turn.content_scalar_count,
        turn.source_span_byte_start,
        turn.source_span_byte_end,
        turn.source_span_scalar_start,
        turn.source_span_scalar_end,
        turn.split,
        turn.license_partition,
        turn.annotation_origin_status,
        _identity_digest(metadata.stable_key(), domain=_DOMAIN_SOURCE_METADATA,
                         label="P3-C1c source metadata"),
        _identity_digest(source_record.stable_key(), domain=_DOMAIN_SOURCE_RECORD,
                         label="P3-C1c source record"),
        _identity_digest(runtime_input.stable_key(), domain=_DOMAIN_RUNTIME_INPUT,
                         label="P3-C1c runtime input"),
    )


def _write_receipt(
        root: KRunRoot,
        relative: Path,
        records: Iterable[ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord],
        *, budget: ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget,
        label: str,
        ) -> ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptStream:
    """以独占 P0 流写无正文 C1c receipt，并绑定 footer/physical identity。"""
    ensure_normal_relative_directory(root, relative.parent, label=f"{label} parent")
    raw = open_exclusive_binary(root, relative, label=label)
    writer = IntegerFramedStreamWriter.from_open_binary(raw, path=relative)
    total = 0
    count = 0
    try:
        for record in records:
            payload = record.integer_stream()
            payload_size = encoded_integer_tuple_size(payload)
            if payload_size > budget.max_receipt_record_payload_bytes:
                _fail(f"{label} record exceeds budget")
            if total + payload_size > budget.max_receipt_payload_bytes:
                _fail(f"{label} payload exceeds budget")
            writer.append(payload)
            total += payload_size
            count += 1
        footer = writer.seal()
    except (IntegerCodecError, IntegerFramedStreamError, OSError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            f"{label} write failed") from exc
    finally:
        writer.close()
    if count == 0 or footer.record_count != count or footer.total_payload_bytes != total:
        _fail(f"{label} footer/count drifted")
    physical = sha256_plain_file(
        root, relative, max_bytes=budget.max_receipt_physical_bytes,
        label=f"{label} physical")
    return ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptStream(
        relative, footer, physical)


def _read_receipt(
        root: KRunRoot,
        identity: ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptStream,
        *, budget: ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget,
        max_count: int,
        label: str,
        ) -> tuple[ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord, ...]:
    """先完成 P0/footer/physical 有界回读，再向后续阶段暴露 records。"""
    try:
        file_identity = capture_plain_file_identity(root, identity.relative_path, label=label)
        stream = open_plain_binary(root, identity.relative_path, label=label,
                                   expected_identity=file_identity)
        reader = IntegerFramedStreamReader.from_open_binary(
            stream,
            path=identity.relative_path,
            max_frame_bytes=budget.max_receipt_record_payload_bytes,
            max_record_count=max_count,
            max_total_payload_bytes=budget.max_receipt_payload_bytes,
        )
        try:
            records = tuple(
                ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord
                .from_integer_stream(raw) for raw in reader)
            footer = reader.finish()
        finally:
            reader.close()
        require_plain_file_identity(root, identity.relative_path, file_identity,
                                    label=f"{label} post-read")
        physical = sha256_plain_file(
            root, identity.relative_path,
            max_bytes=budget.max_receipt_physical_bytes,
            label=f"{label} physical")
    except ConversationHeldOutV4RuntimeTaskCapsuleAdapterError:
        raise
    except (IntegerCodecError, IntegerFramedStreamError, KRunBoundaryError,
            OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            f"{label} readback failed") from exc
    if (footer != identity.p0_footer or physical != identity.physical
            or len(records) != max_count):
        _fail(f"{label} stream identity drifted")
    return records


def _manifest_integer_stream(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult,
        ) -> tuple[int, ...]:
    """构造只绑定 P0 receipt 和安全 capsule digest 的 C1c manifest。"""
    result = [V4_RUNTIME_TASK_CAPSULE_ADAPTER_MANIFEST_SCHEMA]
    for item in (value.receipt.integer_stream(), value.receipt_stream.integer_stream()):
        pack_key(result, item)
    return tuple(result)


def _read_manifest(
        root: KRunRoot,
        *, budget: ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget,
        label: str,
        ) -> bytes:
    """在 P0 publication root 内有界回读整数 manifest。"""
    try:
        identity = capture_plain_file_identity(root, _MANIFEST_FILE, label=label)
        with open_plain_binary(root, _MANIFEST_FILE, label=label,
                               expected_identity=identity) as stream:
            payload = stream.read(budget.max_manifest_bytes + 1)
        require_plain_file_identity(root, _MANIFEST_FILE, identity,
                                    label=f"{label} post-read")
    except (KRunBoundaryError, OSError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            f"{label} readback failed") from exc
    if len(payload) > budget.max_manifest_bytes:
        _fail(f"{label} exceeds budget")
    return payload


def _revalidate_result_upstreams(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult,
        ) -> tuple[
            ConversationHeldOutV4RuntimeTaskBlueprintPublication,
            tuple[ConversationHeldOutV4PayloadWitnessReceiptRecord, ...],
            dict[tuple[int, ...], ConversationHeldOutV4ActiveIntakeMapping],
        ]:
    """重验 result 所指的每一层上游，并将 C1c receipt 的全局摘要逐项回绑。"""
    _preflight_upstream_budgets(
        active=value.active_intake,
        payload_witness=value.payload_witness,
        blueprint=value.blueprint,
        budget=value.receipt.budget,
    )
    try:
        active_publication = load_v4_active_intake_publication(value.active_intake)
        readback = revalidate_v4_active_roster_bidirectional_readback(value.readback)
        payload_publication = load_v4_payload_witness_publication(
            value.payload_witness,
            max_record_count=value.receipt.budget.max_upstream_receipt_record_count,
            max_total_payload_bytes=(
                value.receipt.budget.max_upstream_stream_payload_bytes),
            max_physical_bytes=(
                value.receipt.budget.max_upstream_stream_physical_bytes),
        )
        annotation_publication = load_v4_runtime_task_annotation_provenance_publication(
            value.blueprint.annotation_provenance)
        blueprint_publication = load_v4_runtime_task_blueprint_publication(value.blueprint)
    except (ConversationHeldOutV4ActiveIntakeError,
            ConversationHeldOutV4ActiveRosterReadbackError,
            ConversationHeldOutV4PayloadWitnessLedgerError,
            ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
            ConversationHeldOutV4RuntimeTaskBlueprintError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c result upstream revalidation failed") from exc
    active = active_publication.result
    payload_witness = payload_publication.result
    blueprint = blueprint_publication.result
    annotation = annotation_publication.result
    if (active != value.active_intake
            or readback != value.readback
            or payload_witness != value.payload_witness
            or blueprint != value.blueprint
            or annotation != blueprint.annotation_provenance
            or annotation.payload_witness != payload_witness
            or readback.receipt.active_intake_stable_key != active.stable_key()
            or payload_witness.receipt.active_intake_stable_key != active.stable_key()
            or payload_witness.receipt.readback_stable_key != readback.stable_key()):
        _fail("P3-C1c result upstream identity drifted")
    expected = (
        _identity_digest(active.stable_key(), domain=_DOMAIN_ACTIVE,
                         label="P3-C1c revalidate active"),
        _identity_digest(readback.stable_key(), domain=_DOMAIN_READBACK,
                         label="P3-C1c revalidate readback"),
        _identity_digest(payload_witness.stable_key(), domain=_DOMAIN_PAYLOAD,
                         label="P3-C1c revalidate payload"),
        _identity_digest(annotation.stable_key(), domain=_DOMAIN_ANNOTATION,
                         label="P3-C1c revalidate annotation"),
        _identity_digest(blueprint.stable_key(), domain=_DOMAIN_BLUEPRINT,
                         label="P3-C1c revalidate blueprint"),
    )
    if expected != (
            value.receipt.active_intake_identity_sha256,
            value.receipt.readback_identity_sha256,
            value.receipt.payload_witness_identity_sha256,
            value.receipt.annotation_provenance_identity_sha256,
            value.receipt.blueprint_identity_sha256):
        _fail("P3-C1c result receipt upstream binding drifted")
    active_mappings = {item.stable_key(): item for item in active_publication.mappings}
    if len(active_mappings) != len(active_publication.mappings):
        _fail("P3-C1c result P3-A mapping identities are duplicated")
    return blueprint_publication, payload_publication.records, active_mappings


def _revalidate_replay_bindings(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult,
        *,
        payload_records: tuple[ConversationHeldOutV4PayloadWitnessReceiptRecord, ...],
        active_mappings: dict[tuple[int, ...], ConversationHeldOutV4ActiveIntakeMapping],
        ) -> dict[tuple[int, ...], ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration]:
    """保留并重验 test-only metadata declaration，避免 P0 摘要失去可回放绑定。"""
    records_by_identity = {item.integer_stream(): item for item in payload_records}
    if len(records_by_identity) != len(payload_records):
        _fail("P3-C1c result P3-A2 receipt identities are duplicated")
    required = {turn.witness_record_identity for turn in value.blueprint.turns}
    expected_order = tuple(
        record.integer_stream() for record in payload_records
        if record.integer_stream() in required)
    actual_order = tuple(item.witness_record_identity for item in value.replay_bindings)
    if actual_order != expected_order:
        _fail("P3-C1c replay binding order or inventory drifted")
    metadata_by_witness: dict[
        tuple[int, ...], ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration] = {}
    metadata_by_source: dict[
        SourceRef, ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration] = {}
    for binding in value.replay_bindings:
        record = records_by_identity.get(binding.witness_record_identity)
        if record is None:
            _fail("P3-C1c replay binding references an unknown P3-A2 record")
        mapping = _mapping_for_payload_record(record, active_mappings=active_mappings)
        metadata = binding.metadata_declaration
        if (metadata.source_ref != record.source_ref
                or metadata.frozen_source_manifest_identity
                != record.frozen_source_manifest_identity
                or metadata.source_ref != mapping.source_ref):
            _fail("P3-C1c replay metadata declaration binding drifted")
        prior = metadata_by_source.get(metadata.source_ref)
        if prior is not None and prior != metadata:
            _fail("P3-C1c replay metadata declaration SourceRef drifted")
        metadata_by_source[metadata.source_ref] = metadata
        metadata_by_witness[binding.witness_record_identity] = metadata
    return metadata_by_witness


def _validate_capsule_readback(
        *,
        root: Path,
        budget: ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget,
        dependencies: ConversationHeldOutV4DependencyBinding,
        producer: ConversationHeldOutV4ExternalProducer,
        inputs: tuple[ConversationHeldOutV4RuntimeInput, ...],
        manifest_sha256: tuple[int, ...],
        ) -> ConversationHeldOutV4RuntimeSourceCapsule:
    """仅在 C1c 内即时有界回读 v1 root，随后调用方只能保留安全摘要。"""
    try:
        actual = read_budgeted_v4_external_input_capsule(
            root, budget=budget.capsule_budget, require_k_drive=False)
    except (ConversationHeldOutV4ExternalCapsuleError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c external capsule budgeted readback failed") from exc
    expected = ConversationHeldOutV4RuntimeSourceCapsule(
        V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL,
        manifest_sha256,
        dependencies,
        inputs,
        producer.producer_key,
        producer.declaration,
    )
    if actual != expected:
        _fail("P3-C1c external capsule readback drifted")
    return actual


def materialize_v4_runtime_task_capsule_adapter(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput,
        ) -> ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult:
    """执行 C1c：只在 test transport 组合同次 replay，并发布 payload-free binding。"""
    if not isinstance(value, ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput):
        raise TypeError("P3-C1c materialize requires RuntimeTaskCapsuleAdapterInput")
    active, readback, payload_witness, blueprint_publication = _load_upstreams(value)
    blueprint = blueprint_publication.result
    try:
        _validate_roots(value, active=active, readback=readback,
                        payload_witness=payload_witness, blueprint=blueprint)
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c verification root boundary failed") from exc
    if (len(payload_witness.receipt_stream.p0_footer.integer_tuple()) == 0
            or len(blueprint_publication.records) > value.budget.max_turn_count
            or len(payload_witness.receipt.counts.integer_stream()) == 0):
        _fail("P3-C1c upstream receipt budget binding is invalid")
    active_publication = load_v4_active_intake_publication(active)
    payload_publication = load_v4_payload_witness_publication(
        payload_witness,
        max_record_count=value.budget.max_upstream_receipt_record_count,
        max_total_payload_bytes=value.budget.max_upstream_stream_payload_bytes,
        max_physical_bytes=value.budget.max_upstream_stream_physical_bytes,
    )
    active_mappings = {item.stable_key(): item for item in active_publication.mappings}
    if len(active_mappings) != len(active_publication.mappings):
        _fail("P3-C1c P3-A mapping identities are duplicated")
    payload_records = payload_publication.records
    payload_by_identity = {item.integer_stream(): item for item in payload_records}
    if len(payload_by_identity) != len(payload_records):
        _fail("P3-C1c P3-A2 receipt identities are duplicated")
    turns = blueprint.turns
    if len(turns) != len(blueprint_publication.records):
        _fail("P3-C1c C1b turn inventory drifted")
    for record, turn in zip(blueprint_publication.records, turns):
        _validate_blueprint_pair(record, turn, payload_records=payload_by_identity)
    source_by_witness, metadata_by_witness, counts = _consume_payload_replay(
        value,
        payload_records=payload_records,
        active_mappings=active_mappings,
        turns=turns,
    )
    required_witnesses = {turn.witness_record_identity for turn in turns}
    replay_bindings = tuple(
        ConversationHeldOutV4RuntimeTaskPayloadReplayBinding(
            record.integer_stream(), metadata_by_witness[record.integer_stream()])
        for record in payload_records
        if record.integer_stream() in required_witnesses
    )
    if len(replay_bindings) != counts.unique_witness_count:
        _fail("P3-C1c replay binding inventory drifted")
    inputs: list[ConversationHeldOutV4RuntimeInput] = []
    records: list[ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord] = []
    for blueprint_record, turn in zip(blueprint_publication.records, turns):
        witness_id = turn.witness_record_identity
        source_record = source_by_witness.get(witness_id)
        metadata = metadata_by_witness.get(witness_id)
        payload_record = payload_by_identity.get(witness_id)
        if source_record is None or metadata is None or payload_record is None:
            _fail("P3-C1c C1b turn has no consumed replay witness")
        runtime_input = _runtime_input_for_turn(turn, source_record=source_record)
        inputs.append(runtime_input)
        records.append(_receipt_record_for(
            blueprint_record=blueprint_record,
            turn=turn,
            payload_record=payload_record,
            metadata=metadata,
            source_record=source_record,
            runtime_input=runtime_input,
        ))
    typed_inputs = tuple(inputs)
    try:
        capsule = ConversationHeldOutV4ExternalInputCapsule(
            V4_RUNTIME_FAMILY_KEY, value.producer, value.dependencies, typed_inputs)
        prepared = prepare_v4_external_input_capsule(capsule, value.budget.capsule_budget)
        capsule_root = write_prepared_v4_external_input_capsule(
            value.capsule_parent_run_root.path / _CAPSULE_RELATIVE_ROOT,
            prepared,
            require_k_drive=False,
        )
    except (ConversationHeldOutV4ExternalCapsuleError,
            TypeError, ValueError, OSError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c external capsule publication failed") from exc
    runtime_capsule = _validate_capsule_readback(
        root=capsule_root,
        budget=value.budget,
        dependencies=value.dependencies,
        producer=value.producer,
        inputs=typed_inputs,
        manifest_sha256=tuple(hashlib.sha256(prepared.manifest_payload).digest()),
    )
    try:
        require_exact_file_closure(
            value.capsule_parent_run_root,
            frozenset({
                _CAPSULE_RELATIVE_ROOT / Path("input_capsule.json"),
                _CAPSULE_RELATIVE_ROOT / Path("input.canonical.ints"),
                _CAPSULE_RELATIVE_ROOT / Path("manifest.json"),
            }),
            label="P3-C1c capsule parent",
        )
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c capsule parent closure failed") from exc
    receipt = ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceipt(
        _identity_digest(active.stable_key(), domain=_DOMAIN_ACTIVE,
                         label="P3-C1c active identity"),
        _identity_digest(readback.stable_key(), domain=_DOMAIN_READBACK,
                         label="P3-C1c readback identity"),
        _identity_digest(payload_witness.stable_key(), domain=_DOMAIN_PAYLOAD,
                         label="P3-C1c payload witness identity"),
        _identity_digest(blueprint.annotation_provenance.stable_key(),
                         domain=_DOMAIN_ANNOTATION,
                         label="P3-C1c annotation provenance identity"),
        _identity_digest(blueprint.stable_key(), domain=_DOMAIN_BLUEPRINT,
                         label="P3-C1c blueprint identity"),
        _identity_digest(value.adapter_code_identity.components, domain=_DOMAIN_ADAPTER_CODE,
                         label="P3-C1c adapter code identity"),
        _identity_digest(value.producer.stable_key(), domain=_DOMAIN_PRODUCER,
                         label="P3-C1c producer identity"),
        _identity_digest(value.dependencies.stable_key(), domain=_DOMAIN_DEPENDENCIES,
                         label="P3-C1c dependency identity"),
        value.budget,
        counts,
        runtime_capsule.manifest_sha256,
        _identity_digest(runtime_capsule.stable_key(), domain=_DOMAIN_RUNTIME_CAPSULE,
                         label="P3-C1c runtime capsule identity"),
        V4_RUNTIME_TASK_CAPSULE_ADAPTER_STATUS_TEST_ONLY,
    )
    descriptor = ConversationHeldOutV4RuntimeTaskCapsuleDescriptor(
        value.capsule_parent_run_root,
        _CAPSULE_RELATIVE_ROOT,
        runtime_capsule.manifest_sha256,
        receipt.runtime_capsule_identity_sha256,
        counts.turn_count,
    )
    try:
        staging = _write_receipt(
            value.staging_run_root, _STAGING_RECEIPT_FILE, records,
            budget=value.budget, label="P3-C1c staging receipt")
        require_exact_file_closure(
            value.staging_run_root,
            frozenset({_STAGING_RECEIPT_FILE}),
            label="P3-C1c staging",
        )
        staged_records = _read_receipt(
            value.staging_run_root, staging, budget=value.budget,
            max_count=counts.turn_count,
            label="P3-C1c staging receipt",
        )
        require_exact_file_closure(
            value.staging_run_root,
            frozenset({_STAGING_RECEIPT_FILE}),
            label="P3-C1c staging post-read",
        )
        work = _write_receipt(
            value.work_run_root, _WORK_RECEIPT_FILE,
            staged_records,
            budget=value.budget, label="P3-C1c work receipt")
        require_exact_file_closure(
            value.work_run_root,
            frozenset({_WORK_RECEIPT_FILE}),
            label="P3-C1c work",
        )
        worked_records = _read_receipt(
            value.work_run_root, work, budget=value.budget,
            max_count=counts.turn_count,
            label="P3-C1c work receipt",
        )
        require_exact_file_closure(
            value.work_run_root,
            frozenset({_WORK_RECEIPT_FILE}),
            label="P3-C1c work post-read",
        )
        stream = _write_receipt(
            value.publication_run_root, _RECEIPT_FILE,
            worked_records,
            budget=value.budget, label="P3-C1c receipt")
        provisional = ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult(
            active, readback, payload_witness, blueprint,
            value.publication_run_root, receipt, stream,
            (0,) * _SHA256_SIZE, descriptor, replay_bindings)
        manifest_payload = encode_integer_tuple(_manifest_integer_stream(provisional))
        if len(manifest_payload) > value.budget.max_manifest_bytes:
            _fail("P3-C1c manifest exceeds budget")
        publish_manifest_last(value.publication_run_root, _MANIFEST_FILE, manifest_payload,
                              frozenset({_RECEIPT_FILE}), label="P3-C1c publication")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c receipt publication boundary failed") from exc
    result = ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult(
        active, readback, payload_witness, blueprint,
        value.publication_run_root, receipt, stream,
        tuple(hashlib.sha256(manifest_payload).digest()),
        descriptor, replay_bindings)
    return revalidate_v4_runtime_task_capsule_adapter(result)


def revalidate_v4_runtime_task_capsule_adapter(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult,
        ) -> ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult:
    """有界重验 C1c P0 publication 与固定 capsule root，但不把 raw content 返回。"""
    if not isinstance(value, ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult):
        raise TypeError("P3-C1c revalidate requires RuntimeTaskCapsuleAdapterResult")
    blueprint_publication, payload_records, active_mappings = _revalidate_result_upstreams(
        value)
    metadata_by_witness = _revalidate_replay_bindings(
        value,
        payload_records=payload_records,
        active_mappings=active_mappings,
    )
    payload_by_identity = {record.integer_stream(): record for record in payload_records}
    if len(payload_by_identity) != len(payload_records):
        _fail("P3-C1c result P3-A2 receipt identities are duplicated")
    if len(blueprint_publication.records) != len(value.blueprint.turns):
        _fail("P3-C1c result C1b inventory drifted")
    for blueprint_record, turn in zip(blueprint_publication.records, value.blueprint.turns):
        _validate_blueprint_pair(blueprint_record, turn,
                                 payload_records=payload_by_identity)
    try:
        require_exact_file_closure(value.publication_run_root,
                                   frozenset({_RECEIPT_FILE, _MANIFEST_FILE}),
                                   label="P3-C1c publication")
        records = _read_receipt(
            value.publication_run_root, value.receipt_stream,
            budget=value.receipt.budget,
            max_count=value.receipt.counts.turn_count,
            label="P3-C1c publication receipt")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c publication boundary readback failed") from exc
    if tuple(record.ordinal for record in records) != tuple(range(1, len(records) + 1)):
        _fail("P3-C1c publication receipt ordinal drifted")
    manifest_payload = _read_manifest(
        value.publication_run_root, budget=value.receipt.budget,
        label="P3-C1c manifest")
    try:
        decoded = decode_integer_tuple(manifest_payload)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c manifest is corrupt") from exc
    if (encode_integer_tuple(decoded) != manifest_payload
            or decoded != _manifest_integer_stream(value)
            or tuple(hashlib.sha256(manifest_payload).digest()) != value.manifest_sha256):
        _fail("P3-C1c manifest identity drifted")
    try:
        require_created_run_root(value.capsule_descriptor.parent_run_root,
                                 label="P3-C1c capsule parent")
        require_exact_file_closure(
            value.capsule_descriptor.parent_run_root,
            frozenset({
                _CAPSULE_RELATIVE_ROOT / Path("input_capsule.json"),
                _CAPSULE_RELATIVE_ROOT / Path("input.canonical.ints"),
                _CAPSULE_RELATIVE_ROOT / Path("manifest.json"),
            }),
            label="P3-C1c capsule parent",
        )
        runtime_capsule = read_budgeted_v4_external_input_capsule(
            value.capsule_descriptor.parent_run_root.path
            / value.capsule_descriptor.relative_root,
            budget=value.receipt.budget.capsule_budget,
            require_k_drive=False)
    except (ConversationHeldOutV4ExternalCapsuleError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c external capsule revalidation failed") from exc
    if (runtime_capsule.manifest_sha256 != value.receipt.capsule_manifest_sha256
            or _identity_digest(runtime_capsule.stable_key(),
                                domain=_DOMAIN_RUNTIME_CAPSULE,
                                label="P3-C1c runtime capsule identity")
            != value.receipt.runtime_capsule_identity_sha256
            or len(runtime_capsule.inputs) != value.receipt.counts.turn_count):
        _fail("P3-C1c external capsule receipt binding drifted")
    try:
        runtime_producer = ConversationHeldOutV4ExternalProducer(
            runtime_capsule.external_producer_key,
            runtime_capsule.external_producer_declaration,
        )
    except (ConversationHeldOutV4ExternalCapsuleError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskCapsuleAdapterError(
            "P3-C1c external capsule producer is invalid") from exc
    if (value.receipt.producer_identity_sha256 != _identity_digest(
            runtime_producer.stable_key(), domain=_DOMAIN_PRODUCER,
            label="P3-C1c revalidate producer")
            or value.receipt.dependency_identity_sha256 != _identity_digest(
                runtime_capsule.dependencies.stable_key(), domain=_DOMAIN_DEPENDENCIES,
                label="P3-C1c revalidate dependencies")):
        _fail("P3-C1c external capsule producer/dependency binding drifted")
    payload_record_by_digest: dict[
        tuple[int, ...], ConversationHeldOutV4PayloadWitnessReceiptRecord] = {}
    for record, runtime_input, blueprint_record, turn in zip(
            records, runtime_capsule.inputs, blueprint_publication.records,
            value.blueprint.turns):
        source_records = tuple(item for item in runtime_input.source_records
                               if item.source == record.source_ref)
        if len(source_records) != 1:
            _fail("P3-C1c receipt RuntimeInput source binding drifted")
        source_record = source_records[0]
        if (record.case_key_identity_sha256 != _identity_digest(
                runtime_input.case_key.components, domain=_DOMAIN_CASE,
                label="P3-C1c readback case key")
                or record.turn_key_identity_sha256 != _identity_digest(
                    runtime_input.turn_key.components, domain=_DOMAIN_TURN,
                    label="P3-C1c readback turn key")
                or record.ordinal != runtime_input.ordinal
                or record.content_sha256 != source_record.content_sha256
                or record.content_scalar_count != len(source_record.raw_text_scalars)
                or record.content_byte_count != len(
                    "".join(chr(item) for item in source_record.raw_text_scalars).encode("utf-8"))
                or _text_scalars(record.license_partition,
                                 label="P3-C1c readback license")
                != source_record.license_scalars
                or record.source_record_identity_sha256 != _identity_digest(
                    source_record.stable_key(), domain=_DOMAIN_SOURCE_RECORD,
                    label="P3-C1c readback source record")
                or record.runtime_input_identity_sha256 != _identity_digest(
                    runtime_input.stable_key(), domain=_DOMAIN_RUNTIME_INPUT,
                    label="P3-C1c readback runtime input")):
            _fail("P3-C1c receipt RuntimeInput content binding drifted")
        payload_record = payload_by_identity.get(turn.witness_record_identity)
        if payload_record is None:
            _fail("P3-C1c revalidate C1b witness no longer exists")
        metadata = metadata_by_witness.get(turn.witness_record_identity)
        if metadata is None:
            _fail("P3-C1c revalidate witness has no metadata declaration")
        if (record.blueprint_receipt_record_identity_sha256
                != _identity_digest(blueprint_record.integer_stream(),
                                    domain=_DOMAIN_BLUEPRINT_RECORD,
                                    label="P3-C1c revalidate blueprint record")
                or record.blueprint_turn_identity_sha256
                != _identity_digest(turn.stable_key(), domain=_DOMAIN_TURN,
                                    label="P3-C1c revalidate blueprint turn")
                or record.annotation_witness_identity_sha256
                != blueprint_record.annotation_witness_identity_sha256
                or record.payload_witness_identity_sha256
                != _identity_digest(payload_record.integer_stream(),
                                    domain=_DOMAIN_PAYLOAD_RECORD,
                                    label="P3-C1c revalidate payload record")
                or record.source_metadata_declaration_identity_sha256
                != _identity_digest(metadata.stable_key(),
                                    domain=_DOMAIN_SOURCE_METADATA,
                                    label="P3-C1c revalidate metadata declaration")
                or record.source_ref != turn.source_ref
                or record.content_sha256 != turn.content_sha256
                or record.content_byte_count != turn.content_byte_count
                or record.content_scalar_count != turn.content_scalar_count
                or (record.source_span_byte_start, record.source_span_byte_end,
                    record.source_span_scalar_start, record.source_span_scalar_end)
                != (turn.source_span_byte_start, turn.source_span_byte_end,
                    turn.source_span_scalar_start, turn.source_span_scalar_end)
                or record.split != turn.split
                or record.license_partition != turn.license_partition
                or record.annotation_origin_status != turn.annotation_origin_status):
            _fail("P3-C1c receipt C1b/P3-A2 binding drifted")
        prior = payload_record_by_digest.get(record.payload_witness_identity_sha256)
        if prior is not None and prior != payload_record:
            _fail("P3-C1c repeated payload witness receipt binding drifted")
        payload_record_by_digest[record.payload_witness_identity_sha256] = payload_record
    unique_records = tuple(payload_record_by_digest.values())
    if (len(unique_records) != value.receipt.counts.unique_witness_count
            or value.receipt.counts.actual_payload_read_count
            != value.receipt.counts.unique_witness_count
            or sum(item.split == "train" for item in records)
            != value.receipt.counts.train_turn_count
            or sum(item.split == "held_out" for item in records)
            != value.receipt.counts.held_out_turn_count
            or sum(item.content_byte_count for item in unique_records)
            != value.receipt.counts.content_byte_count
            or sum(item.content_scalar_count for item in unique_records)
            != value.receipt.counts.content_scalar_count):
        _fail("P3-C1c receipt aggregate counts drifted")
    return value


def load_v4_runtime_task_capsule_adapter_publication(
        value: ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult,
        ) -> ConversationHeldOutV4RuntimeTaskCapsuleAdapterPublication:
    """回读无正文 C1c receipt；runtime payload 保持在内部即时验证后丢弃。"""
    result = revalidate_v4_runtime_task_capsule_adapter(value)
    records = _read_receipt(
        result.publication_run_root, result.receipt_stream,
        budget=result.receipt.budget,
        max_count=result.receipt.counts.turn_count,
        label="P3-C1c public receipt")
    return ConversationHeldOutV4RuntimeTaskCapsuleAdapterPublication(result, records)


__all__ = [
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget",
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterCounts",
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterError",
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput",
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterPublication",
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceipt",
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptRecord",
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterReceiptStream",
    "ConversationHeldOutV4RuntimeTaskCapsuleAdapterResult",
    "ConversationHeldOutV4RuntimeTaskCapsuleDescriptor",
    "ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration",
    "ConversationHeldOutV4RuntimeTaskPayloadReplayBinding",
    "ConversationHeldOutV4RuntimeTaskPayloadReplayItem",
    "V4_RUNTIME_TASK_CAPSULE_ADAPTER_STATUS_TEST_ONLY",
    "load_v4_runtime_task_capsule_adapter_publication",
    "materialize_v4_runtime_task_capsule_adapter",
    "revalidate_v4_runtime_task_capsule_adapter",
]
