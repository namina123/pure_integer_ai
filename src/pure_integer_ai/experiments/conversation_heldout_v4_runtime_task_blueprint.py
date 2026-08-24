"""DLG-05 v4 P3-C1b 无载荷运行时任务蓝图。

P3-C1b 只把显式 authored 的 typed task 结构绑定到 P3-A/P3-B/P3-A2 的公开身份。
它不读取或写出原始载荷，不构造 external capsule，不调用 runtime/writer/trainer，也不把
测试 transport 解释为理解、训练资格或主动来源覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Iterator

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
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
from pure_integer_ai.experiments.conversation_heldout_v4_payload_witness_ledger import (
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
    ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult,
    ConversationHeldOutV4RuntimeTaskAnnotationWitness,
    annotation_witness_public_identity_sha256,
    load_v4_runtime_task_annotation_provenance_publication,
    require_v4_runtime_task_annotation_witness_binding,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_structure import (
    ConversationHeldOutV4BlueprintEvidencePlan,
    ConversationHeldOutV4BlueprintRepresentation,
    ConversationHeldOutV4RuntimeTaskStructureError,
    validate_v4_runtime_task_structure,
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
    require_disjoint_run_roots,
    require_exact_file_closure,
    require_fresh_empty_run_root,
    require_plain_file_identity,
    sha256_plain_file,
)


V4_RUNTIME_TASK_BLUEPRINT_BUDGET_SCHEMA = 1
V4_RUNTIME_TASK_BLUEPRINT_RECORD_SCHEMA = 2
V4_RUNTIME_TASK_BLUEPRINT_RECEIPT_SCHEMA = 2
V4_RUNTIME_TASK_BLUEPRINT_RESULT_SCHEMA = 1
V4_RUNTIME_TASK_BLUEPRINT_MANIFEST_SCHEMA = 1
V4_RUNTIME_TASK_BLUEPRINT_STATUS_TEST_ONLY = "P3_C1B_BLUEPRINT_TEST_ONLY"
V4_RUNTIME_TASK_BLUEPRINT_STATUS_COVERAGE_NE = "ACTIVE_PROVENANCE_COVERAGE_NE"
V4_RUNTIME_TASK_BLUEPRINT_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED = (
    V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED
)

_STAGING_RECEIPT_FILE = Path("candidate") / "receipt.pifrs"
_WORK_RECEIPT_FILE = Path("verified") / "receipt.pifrs"
_RECEIPT_FILE = Path("blueprint") / "receipt.pifrs"
_MANIFEST_FILE = Path("manifest.pii")
_SHA256_SIZE = hashlib.sha256().digest_size
_DOMAIN_ACTIVE = b"dlg05.v4.p3c1b.active.v1\x00"
_DOMAIN_READBACK = b"dlg05.v4.p3c1b.readback.v1\x00"
_DOMAIN_CASE_KEY = b"dlg05.v4.p3c1b.case-key.v1\x00"
_DOMAIN_TURN_KEY = b"dlg05.v4.p3c1b.turn-key.v1\x00"
_DOMAIN_WITNESS_RECORD = b"dlg05.v4.p3c1b.witness-record.v1\x00"
_DOMAIN_PAYLOAD_WITNESS = b"dlg05.v4.p3c1b.payload-witness.v1\x00"
_DOMAIN_ANNOTATION_MANIFEST = b"dlg05.v4.p3c1b.annotation-manifest.v1\x00"
_DOMAIN_ANNOTATION_PROVENANCE = b"dlg05.v4.p3c1b.annotation-provenance.v1\x00"
_DOMAIN_CODE = b"dlg05.v4.p3c1b.code.v1\x00"
_DOMAIN_ANNOTATION_SOURCE = b"dlg05.v4.p3c1b.annotation-source.v1\x00"
_DOMAIN_ANNOTATION_REVISION = b"dlg05.v4.p3c1b.annotation-revision.v1\x00"
_DOMAIN_ANNOTATION_TRANSFORM = b"dlg05.v4.p3c1b.annotation-transform.v1\x00"
_DOMAIN_APPLICABLE_DOMAIN = b"dlg05.v4.p3c1b.applicable-domain.v1\x00"
_DOMAIN_TURN = b"dlg05.v4.p3c1b.turn.v1\x00"
_DOMAIN_REQUEST = b"dlg05.v4.p3c1b.request.v1\x00"
_DOMAIN_REPRESENTATION = b"dlg05.v4.p3c1b.representation.v1\x00"
_DOMAIN_EVIDENCE = b"dlg05.v4.p3c1b.evidence.v1\x00"


class ConversationHeldOutV4RuntimeTaskBlueprintError(RuntimeError):
    """P3-C1b identity、typed task、receipt 或 run-root 闭合失败。"""


def _fail(message: str) -> None:
    """统一产生 P3-C1b fail-closed 错误。"""
    raise ConversationHeldOutV4RuntimeTaskBlueprintError(message)


def _text_scalars(value: str, *, label: str) -> tuple[int, ...]:
    """将公开元数据文本规范化为非空 Unicode scalar tuple。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty str")
    result = tuple(ord(item) for item in value)
    if any(item < 0 or item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in result):
        raise ValueError(f"{label} contains invalid Unicode scalars")
    return result


def _text_from_scalars(value: tuple[int, ...], *, label: str) -> str:
    """恢复 receipt 文本字段并拒绝 surrogate 或空值。"""
    strict_integer_tuple(value, label=label)
    try:
        result = "".join(chr(item) for item in value)
    except ValueError as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            f"{label} is not Unicode") from exc
    if not result or any(0xD800 <= item <= 0xDFFF for item in value):
        _fail(f"{label} is invalid")
    return result


def _require_digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证 SHA-256 的完整字节整数表示。"""
    if (not isinstance(value, tuple) or len(value) != _SHA256_SIZE
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ValueError(f"{label} must be a SHA-256 byte tuple")
    return value


def _require_positive(value: int, *, label: str) -> int:
    """拒绝 bool、零和负数进入资源或 ordinal 字段。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive strict int")
    return value


def _stage_name(value: str) -> str:
    """限制逻辑 stage 名字为可审计的 ASCII token。"""
    if (not isinstance(value, str) or not value or len(value) > 48
            or not value[0].isascii() or not value[0].isalnum()
            or any(not item.isascii() or not (item.isalnum() or item in {"-", "_", "."})
                   for item in value)):
        raise ValueError("P3-C1b logical_stage_name is invalid")
    return value


def _identity_digest(
        value: tuple[int, ...], *, domain: bytes, label: str,
        ) -> tuple[int, ...]:
    """为公开 receipt 生成完整 typed identity 的不可逆摘要，不输出表面 scalar。"""
    strict_integer_tuple(value, label=label)
    if not isinstance(domain, bytes) or not domain:
        raise TypeError("P3-C1b identity digest domain is invalid")
    return tuple(hashlib.sha256(domain + encode_integer_tuple(value)).digest())


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintBudget:
    """限制 P3-C1b 公开回读、内存 turn 和 receipt 发布资源。"""

    max_witness_record_count: int
    max_turn_count: int
    max_representations_per_turn: int
    max_evidence_plans_per_turn: int
    max_receipt_record_payload_bytes: int
    max_receipt_payload_bytes: int
    max_receipt_physical_bytes: int
    max_manifest_bytes: int

    def __post_init__(self) -> None:
        """使所有资源上限为正整数，且单 record 不超过总 receipt。"""
        for label, value in zip((
                "max_witness_record_count", "max_turn_count",
                "max_representations_per_turn", "max_evidence_plans_per_turn",
                "max_receipt_record_payload_bytes", "max_receipt_payload_bytes",
                "max_receipt_physical_bytes", "max_manifest_bytes"), self.integer_stream()[1:]):
            _require_positive(value, label=f"P3-C1b {label}")
        if self.max_receipt_record_payload_bytes > self.max_receipt_payload_bytes:
            raise ValueError("P3-C1b receipt record budget exceeds receipt budget")

    def integer_stream(self) -> tuple[int, ...]:
        """返回版本化、规范顺序的预算整数流。"""
        return (
            V4_RUNTIME_TASK_BLUEPRINT_BUDGET_SCHEMA,
            self.max_witness_record_count,
            self.max_turn_count,
            self.max_representations_per_turn,
            self.max_evidence_plans_per_turn,
            self.max_receipt_record_payload_bytes,
            self.max_receipt_payload_bytes,
            self.max_receipt_physical_bytes,
            self.max_manifest_bytes,
        )


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintTurn:
    """一条只在内存中存在的 authored task 与其完整 P3-A2 witness binding。"""

    case_key: ProtocolKey
    turn_key: ProtocolKey
    ordinal: int
    witness_record_identity: tuple[int, ...]
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
    frozen_annotation_manifest_identity: ProtocolKey
    constructing_code_identity: ProtocolKey
    annotation_source_identity: ProtocolKey
    annotation_revision_identity: ProtocolKey
    annotation_transform_code_identity: ProtocolKey
    applicable_domain_identity: ProtocolKey
    request: QuestionRequest
    representations: tuple[ConversationHeldOutV4BlueprintRepresentation, ...]
    evidence_plans: tuple[ConversationHeldOutV4BlueprintEvidencePlan, ...]

    def __post_init__(self) -> None:
        """只接受显式来源与构造身份的 typed blueprint turn。"""
        if not isinstance(self.case_key, ProtocolKey):
            raise TypeError("P3-C1b case_key type is invalid")
        if not isinstance(self.turn_key, ProtocolKey):
            raise TypeError("P3-C1b turn_key type is invalid")
        _require_positive(self.ordinal, label="P3-C1b ordinal")
        strict_integer_tuple(self.witness_record_identity,
                             label="P3-C1b witness record identity")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("P3-C1b source_ref type is invalid")
        _require_digest(self.content_sha256, label="P3-C1b content SHA-256")
        for label, value in (
                ("content byte count", self.content_byte_count),
                ("content scalar count", self.content_scalar_count)):
            _require_positive(value, label=f"P3-C1b {label}")
        spans = (
            self.source_span_byte_start, self.source_span_byte_end,
            self.source_span_scalar_start, self.source_span_scalar_end,
        )
        if (any(type(item) is not int or item < 0 for item in spans)
                or self.source_span_byte_start >= self.source_span_byte_end
                or self.source_span_byte_end > self.content_byte_count
                or self.source_span_scalar_start >= self.source_span_scalar_end
                or self.source_span_scalar_end > self.content_scalar_count):
            raise ValueError("P3-C1b source span is invalid")
        if self.split not in {"train", "held_out"}:
            raise ValueError("P3-C1b split is invalid")
        _text_scalars(self.license_partition, label="P3-C1b license partition")
        if (self.annotation_origin_status
                != V4_RUNTIME_TASK_BLUEPRINT_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED):
            raise ValueError("P3-C1b annotation origin is not allowed in this slice")
        for label, value in (
                ("annotation manifest", self.frozen_annotation_manifest_identity),
                ("constructing code", self.constructing_code_identity),
                ("annotation source", self.annotation_source_identity),
                ("annotation revision", self.annotation_revision_identity),
                ("annotation transform code", self.annotation_transform_code_identity),
                ("applicable domain", self.applicable_domain_identity)):
            if not isinstance(value, ProtocolKey):
                raise TypeError(f"P3-C1b {label} identity type is invalid")
        if not isinstance(self.request, QuestionRequest):
            raise TypeError("P3-C1b request type is invalid")
        if (not isinstance(self.representations, tuple) or not self.representations
                or any(not isinstance(item, ConversationHeldOutV4BlueprintRepresentation)
                       for item in self.representations)):
            raise TypeError("P3-C1b representations type is invalid")
        if (not isinstance(self.evidence_plans, tuple)
                or any(not isinstance(item, ConversationHeldOutV4BlueprintEvidencePlan)
                       for item in self.evidence_plans)):
            raise TypeError("P3-C1b evidence plans type is invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回仅在内存中使用的完整 turn identity，公开时必须仅发布摘要。"""
        result = [V4_RUNTIME_TASK_BLUEPRINT_RECORD_SCHEMA]
        for value in (
                self.case_key.components,
                self.turn_key.components,
                (self.ordinal,),
                self.witness_record_identity,
                self.source_ref.stable_key(),
                self.content_sha256,
                (self.content_byte_count, self.content_scalar_count),
                (self.source_span_byte_start, self.source_span_byte_end,
                 self.source_span_scalar_start, self.source_span_scalar_end),
                _text_scalars(self.split, label="P3-C1b split"),
                _text_scalars(self.license_partition, label="P3-C1b license partition"),
                _text_scalars(self.annotation_origin_status,
                              label="P3-C1b annotation origin"),
                self.frozen_annotation_manifest_identity.components,
                self.constructing_code_identity.components,
                self.annotation_source_identity.components,
                self.annotation_revision_identity.components,
                self.annotation_transform_code_identity.components,
                self.applicable_domain_identity.components,
                self.request.stable_key()):
            pack_key(result, value)
        result.append(len(self.representations))
        for item in self.representations:
            pack_key(result, item.stable_key())
        result.append(len(self.evidence_plans))
        for item in self.evidence_plans:
            pack_key(result, item.stable_key())
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord:
    """无正文公开 turn receipt；所有可能含表面内容的 typed identity 均为 digest。"""

    case_key_identity_sha256: tuple[int, ...]
    turn_key_identity_sha256: tuple[int, ...]
    ordinal: int
    witness_record_identity_sha256: tuple[int, ...]
    annotation_witness_identity_sha256: tuple[int, ...]
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
    frozen_annotation_manifest_identity_sha256: tuple[int, ...]
    constructing_code_identity_sha256: tuple[int, ...]
    annotation_source_identity_sha256: tuple[int, ...]
    annotation_revision_identity_sha256: tuple[int, ...]
    annotation_transform_code_identity_sha256: tuple[int, ...]
    applicable_domain_identity_sha256: tuple[int, ...]
    turn_identity_sha256: tuple[int, ...]
    request_identity_sha256: tuple[int, ...]
    representation_identity_sha256: tuple[tuple[int, ...], ...]
    evidence_plan_identity_sha256: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """验证 receipt 不带正文且仍完整绑定每个 task 结构身份。"""
        _require_digest(self.case_key_identity_sha256,
                        label="P3-C1b receipt case key identity")
        _require_digest(self.turn_key_identity_sha256,
                        label="P3-C1b receipt turn key identity")
        _require_positive(self.ordinal, label="P3-C1b receipt ordinal")
        _require_digest(self.witness_record_identity_sha256,
                        label="P3-C1b receipt witness identity")
        _require_digest(self.annotation_witness_identity_sha256,
                        label="P3-C1b receipt annotation witness identity")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("P3-C1b receipt source type is invalid")
        _require_digest(self.content_sha256, label="P3-C1b receipt content SHA-256")
        for label, value in (
                ("byte count", self.content_byte_count),
                ("scalar count", self.content_scalar_count)):
            _require_positive(value, label=f"P3-C1b receipt {label}")
        spans = (
            self.source_span_byte_start, self.source_span_byte_end,
            self.source_span_scalar_start, self.source_span_scalar_end,
        )
        if (any(type(item) is not int or item < 0 for item in spans)
                or self.source_span_byte_start >= self.source_span_byte_end
                or self.source_span_byte_end > self.content_byte_count
                or self.source_span_scalar_start >= self.source_span_scalar_end
                or self.source_span_scalar_end > self.content_scalar_count):
            raise ValueError("P3-C1b receipt span is invalid")
        if self.split not in {"train", "held_out"}:
            raise ValueError("P3-C1b receipt split is invalid")
        _text_scalars(self.license_partition, label="P3-C1b receipt license partition")
        if (self.annotation_origin_status
                != V4_RUNTIME_TASK_BLUEPRINT_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED):
            raise ValueError("P3-C1b receipt annotation origin is invalid")
        for label, value in (
                ("annotation manifest", self.frozen_annotation_manifest_identity_sha256),
                ("constructing code", self.constructing_code_identity_sha256),
                ("annotation source", self.annotation_source_identity_sha256),
                ("annotation revision", self.annotation_revision_identity_sha256),
                ("annotation transform code", self.annotation_transform_code_identity_sha256),
                ("applicable domain", self.applicable_domain_identity_sha256)):
            _require_digest(value, label=f"P3-C1b receipt {label} identity")
        _require_digest(self.turn_identity_sha256, label="P3-C1b receipt turn identity")
        _require_digest(self.request_identity_sha256, label="P3-C1b receipt request identity")
        for label, values in (
                ("representation identities", self.representation_identity_sha256),
                ("Evidence identities", self.evidence_plan_identity_sha256)):
            if (not isinstance(values, tuple)
                    or any(_require_digest(item, label=f"P3-C1b receipt {label}")
                           != item for item in values)):
                raise ValueError(f"P3-C1b receipt {label} are invalid")
        if not self.representation_identity_sha256:
            raise ValueError("P3-C1b receipt must bind a Representation")

    def integer_stream(self) -> tuple[int, ...]:
        """将无正文 receipt 记录编码为有边界的规范整数流。"""
        result = [V4_RUNTIME_TASK_BLUEPRINT_RECORD_SCHEMA]
        for value in (
                self.case_key_identity_sha256,
                self.turn_key_identity_sha256,
                (self.ordinal,),
                self.witness_record_identity_sha256,
                self.annotation_witness_identity_sha256,
                self.source_ref.stable_key(),
                self.content_sha256,
                (self.content_byte_count, self.content_scalar_count),
                (self.source_span_byte_start, self.source_span_byte_end,
                 self.source_span_scalar_start, self.source_span_scalar_end),
                _text_scalars(self.split, label="P3-C1b receipt split"),
                _text_scalars(self.license_partition, label="P3-C1b receipt license partition"),
                _text_scalars(self.annotation_origin_status,
                              label="P3-C1b receipt annotation origin"),
                self.frozen_annotation_manifest_identity_sha256,
                self.constructing_code_identity_sha256,
                self.annotation_source_identity_sha256,
                self.annotation_revision_identity_sha256,
                self.annotation_transform_code_identity_sha256,
                self.applicable_domain_identity_sha256,
                self.turn_identity_sha256,
                self.request_identity_sha256):
            pack_key(result, value)
        result.append(len(self.representation_identity_sha256))
        for item in self.representation_identity_sha256:
            pack_key(result, item)
        result.append(len(self.evidence_plan_identity_sha256))
        for item in self.evidence_plan_identity_sha256:
            pack_key(result, item)
        return tuple(result)

    @classmethod
    def from_integer_stream(
            cls, value: tuple[int, ...],
            ) -> "ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord":
        """从公开 P0 record 恢复结构，并拒绝截断、非规范或正文偷渡。"""
        try:
            reader = IntegerStreamReader(value)
            if reader.read_positive(label="P3-C1b receipt schema") != V4_RUNTIME_TASK_BLUEPRINT_RECORD_SCHEMA:
                _fail("P3-C1b receipt schema is incompatible")
            case_key = reader.read_key(label="P3-C1b receipt case key")
            turn_key = reader.read_key(label="P3-C1b receipt turn key")
            ordinal_values = reader.read_key(label="P3-C1b receipt ordinal")
            if len(ordinal_values) != 1:
                _fail("P3-C1b receipt ordinal is invalid")
            witness_identity = reader.read_key(label="P3-C1b receipt witness identity")
            annotation_witness_identity = reader.read_key(
                label="P3-C1b receipt annotation witness identity")
            source_ref = SourceRef.from_stable_key(reader.read_key(
                label="P3-C1b receipt source"))
            content_sha256 = reader.read_key(label="P3-C1b receipt content SHA-256")
            counts = reader.read_key(label="P3-C1b receipt counts")
            if len(counts) != 2:
                _fail("P3-C1b receipt counts are invalid")
            spans = reader.read_key(label="P3-C1b receipt spans")
            if len(spans) != 4:
                _fail("P3-C1b receipt spans are invalid")
            split = _text_from_scalars(reader.read_key(label="P3-C1b receipt split"),
                                       label="P3-C1b receipt split")
            license_partition = _text_from_scalars(
                reader.read_key(label="P3-C1b receipt license partition"),
                label="P3-C1b receipt license partition")
            annotation_origin = _text_from_scalars(
                reader.read_key(label="P3-C1b receipt annotation origin"),
                label="P3-C1b receipt annotation origin")
            annotation_manifest = reader.read_key(
                label="P3-C1b receipt annotation manifest")
            constructing_code = reader.read_key(
                label="P3-C1b receipt constructing code")
            annotation_source = reader.read_key(label="P3-C1b receipt annotation source")
            annotation_revision = reader.read_key(label="P3-C1b receipt annotation revision")
            annotation_transform = reader.read_key(label="P3-C1b receipt annotation transform")
            applicable_domain = reader.read_key(label="P3-C1b receipt applicable domain")
            turn_identity = reader.read_key(label="P3-C1b receipt turn identity")
            request_identity = reader.read_key(label="P3-C1b receipt request identity")
            representation_count = reader.read_positive(
                label="P3-C1b receipt representation count")
            representations = tuple(
                reader.read_key(label="P3-C1b receipt Representation identity")
                for _ in range(representation_count))
            evidence_count = reader.read_nonnegative(
                label="P3-C1b receipt Evidence count")
            evidence = tuple(
                reader.read_key(label="P3-C1b receipt Evidence identity")
                for _ in range(evidence_count))
            result = cls(
                case_key, turn_key, ordinal_values[0], witness_identity,
                annotation_witness_identity, source_ref,
                content_sha256, counts[0], counts[1], *spans, split, license_partition,
                annotation_origin, annotation_manifest, constructing_code,
                annotation_source, annotation_revision, annotation_transform,
                applicable_domain, turn_identity, request_identity, representations, evidence)
            reader.finish()
        except (IntegerCodecError, TypeError, ValueError, IndexError) as exc:
            raise ConversationHeldOutV4RuntimeTaskBlueprintError(
                "P3-C1b receipt record is corrupt") from exc
        if result.integer_stream() != value:
            _fail("P3-C1b receipt record is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream:
    """P3-C1b receipt P0 footer 与物理文件身份。"""

    relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """限制 descriptor 只能指向相对普通 receipt 文件。"""
        if (not isinstance(self.relative_path, Path) or self.relative_path.is_absolute()
                or self.relative_path.drive or self.relative_path.root
                or not self.relative_path.parts or ".." in self.relative_path.parts
                or not isinstance(self.p0_footer, IntegerFramedStreamFooter)
                or not isinstance(self.physical, KRunFileDigest)
                or self.physical.byte_count <= 0):
            raise ValueError("P3-C1b receipt stream identity is invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """返回 receipt 路径、P0 footer 和完整物理 digest。"""
        result: list[int] = []
        for value in (
                _text_scalars(self.relative_path.as_posix(), label="P3-C1b receipt path"),
                self.p0_footer.integer_tuple(),
                (self.physical.byte_count, *self.physical.sha256)):
            pack_key(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintCounts:
    """P3-C1b turn、split、Representation 和 Evidence 的公开计数。"""

    turn_count: int
    train_turn_count: int
    held_out_turn_count: int
    representation_count: int
    evidence_plan_count: int

    def __post_init__(self) -> None:
        """保持 split 计数闭合且至少有一个 blueprint turn。"""
        values = self.integer_stream()
        for label, value in zip((
                "turn_count", "train_turn_count", "held_out_turn_count",
                "representation_count", "evidence_plan_count"), values):
            if type(value) is not int or value < 0:
                raise ValueError(f"P3-C1b {label} must be a nonnegative strict int")
        if self.turn_count == 0 or self.turn_count != self.train_turn_count + self.held_out_turn_count:
            raise ValueError("P3-C1b turn split counts do not close")
        if self.representation_count < self.turn_count:
            raise ValueError("P3-C1b Representation count does not close")

    def integer_stream(self) -> tuple[int, ...]:
        """返回计数的固定字段顺序。"""
        return (
            self.turn_count, self.train_turn_count, self.held_out_turn_count,
            self.representation_count, self.evidence_plan_count,
        )


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintReceipt:
    """绑定 P3-A/P3-B/P3-A2、sealed annotation inventory 和公开计数的 receipt。"""

    active_intake_identity_sha256: tuple[int, ...]
    readback_identity_sha256: tuple[int, ...]
    payload_witness_identity_sha256: tuple[int, ...]
    annotation_provenance_identity_sha256: tuple[int, ...]
    budget: ConversationHeldOutV4RuntimeTaskBlueprintBudget
    counts: ConversationHeldOutV4RuntimeTaskBlueprintCounts
    status: str

    def __post_init__(self) -> None:
        """拒绝缺 upstream identity、预算、计数或越级状态。"""
        for label, value in (
                ("active intake", self.active_intake_identity_sha256),
                ("readback", self.readback_identity_sha256),
                ("payload witness", self.payload_witness_identity_sha256),
                ("annotation provenance", self.annotation_provenance_identity_sha256)):
            _require_digest(value, label=f"P3-C1b {label} identity")
        if (not isinstance(self.budget, ConversationHeldOutV4RuntimeTaskBlueprintBudget)
                or not isinstance(self.counts, ConversationHeldOutV4RuntimeTaskBlueprintCounts)
                or self.status not in {
                    V4_RUNTIME_TASK_BLUEPRINT_STATUS_TEST_ONLY,
                    V4_RUNTIME_TASK_BLUEPRINT_STATUS_COVERAGE_NE}):
            raise ValueError("P3-C1b receipt fields are invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """编码整个 receipt 的无路径、可回读身份。"""
        result = [V4_RUNTIME_TASK_BLUEPRINT_RECEIPT_SCHEMA]
        for value in (
                self.active_intake_identity_sha256,
                self.readback_identity_sha256,
                self.payload_witness_identity_sha256,
                self.annotation_provenance_identity_sha256,
                self.budget.integer_stream(),
                self.counts.integer_stream(),
                _text_scalars(self.status, label="P3-C1b status")):
            pack_key(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintInput:
    """P3-C1b 唯一入口；factory 只能生成显式 authored 的内存 typed items。"""

    active_intake: ConversationHeldOutV4ActiveIntakeResult
    readback: ConversationHeldOutV4ActiveRosterReadbackResult
    payload_witness: ConversationHeldOutV4PayloadWitnessResult
    annotation_provenance: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult
    blueprint_factory: Callable[[], Iterable[ConversationHeldOutV4RuntimeTaskBlueprintTurn]]
    staging_run_root: KRunRoot
    work_run_root: KRunRoot
    publication_run_root: KRunRoot
    budget: ConversationHeldOutV4RuntimeTaskBlueprintBudget
    logical_stage_name: str

    def __post_init__(self) -> None:
        """固定唯一输入类型，禁止隐式 payload、capsule 或环境回退。"""
        if (not isinstance(self.active_intake, ConversationHeldOutV4ActiveIntakeResult)
                or not isinstance(self.readback, ConversationHeldOutV4ActiveRosterReadbackResult)
                or not isinstance(self.payload_witness, ConversationHeldOutV4PayloadWitnessResult)
                or not isinstance(self.annotation_provenance,
                                  ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult)
                or not callable(self.blueprint_factory)
                or not isinstance(self.budget, ConversationHeldOutV4RuntimeTaskBlueprintBudget)):
            raise TypeError("P3-C1b input types are invalid")
        for root in (self.staging_run_root, self.work_run_root, self.publication_run_root):
            if not isinstance(root, KRunRoot):
                raise TypeError("P3-C1b run root type is invalid")
        _stage_name(self.logical_stage_name)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintResult:
    """已发布无正文 receipt 与仅内存 typed turns 的 P3-C1b 结果。"""

    publication_run_root: KRunRoot
    receipt: ConversationHeldOutV4RuntimeTaskBlueprintReceipt
    receipt_stream: ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream
    manifest_sha256: tuple[int, ...]
    annotation_provenance: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult
    turns: tuple[ConversationHeldOutV4RuntimeTaskBlueprintTurn, ...]

    def __post_init__(self) -> None:
        """确保 publication 不泄漏正文，turn 只作为当前进程的受限内存结果。"""
        if (not isinstance(self.publication_run_root, KRunRoot)
                or not isinstance(self.receipt, ConversationHeldOutV4RuntimeTaskBlueprintReceipt)
                or not isinstance(self.receipt_stream, ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream)
                or not isinstance(self.annotation_provenance,
                                  ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult)):
            raise TypeError("P3-C1b result types are invalid")
        if self.receipt_stream.relative_path != _RECEIPT_FILE:
            raise ValueError("P3-C1b receipt path drifted")
        _require_digest(self.manifest_sha256, label="P3-C1b manifest hash")
        if (not isinstance(self.turns, tuple)
                or any(not isinstance(item, ConversationHeldOutV4RuntimeTaskBlueprintTurn)
                       for item in self.turns)
                or len(self.turns) != self.receipt.counts.turn_count):
            raise ValueError("P3-C1b result turns are invalid")
        expected_status = (
            V4_RUNTIME_TASK_BLUEPRINT_STATUS_TEST_ONLY
            if self.publication_run_root.test_transport
            else V4_RUNTIME_TASK_BLUEPRINT_STATUS_COVERAGE_NE)
        if self.receipt.status != expected_status:
            raise ValueError("P3-C1b result transport status drifted")

    def stable_key(self) -> tuple[int, ...]:
        """返回只含公开 receipt 与 manifest 的跨阶段稳定身份，不含正文 turns。"""
        result = [V4_RUNTIME_TASK_BLUEPRINT_RESULT_SCHEMA]
        for value in (
                self.receipt.integer_stream(), self.receipt_stream.integer_stream(),
                self.manifest_sha256):
            pack_key(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskBlueprintPublication:
    """P3-C1b 的受限公开 loader 结果，供 C1c 逐条回读已封存 receipt。"""

    result: ConversationHeldOutV4RuntimeTaskBlueprintResult
    records: tuple[ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord, ...]

    def __post_init__(self) -> None:
        """限制公开 records 的类型、计数和连续 ordinal，不暴露新的构造入口。"""
        if (not isinstance(self.result, ConversationHeldOutV4RuntimeTaskBlueprintResult)
                or not isinstance(self.records, tuple)
                or any(not isinstance(item, ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord)
                       for item in self.records)
                or len(self.records) != self.result.receipt.counts.turn_count
                or tuple(item.ordinal for item in self.records)
                != tuple(range(1, len(self.records) + 1))):
            raise TypeError("P3-C1b publication types are invalid")


def _validate_roots(value: ConversationHeldOutV4RuntimeTaskBlueprintInput) -> str:
    """验证三新输出根与 P3-A/P3-B/P3-A2 输入根严格隔离、同一 transport。"""
    outputs = (value.staging_run_root, value.work_run_root, value.publication_run_root)
    inputs = (
        value.active_intake.publication_run_root,
        value.readback.publication_run_root,
        value.payload_witness.publication_run_root,
        value.annotation_provenance.publication_run_root,
    )
    if len({root.test_transport for root in (*outputs, *inputs)}) != 1:
        _fail("P3-C1b roots must use one transport mode")
    for left_index, left in enumerate(outputs):
        for right in outputs[left_index + 1:]:
            require_disjoint_run_roots(left, right, label="P3-C1b output roots")
    for output in outputs:
        for source in inputs:
            require_disjoint_run_roots(output, source, label="P3-C1b input/output roots")
        require_fresh_empty_run_root(output, label="P3-C1b fresh verification root")
    return (V4_RUNTIME_TASK_BLUEPRINT_STATUS_TEST_ONLY
            if value.publication_run_root.test_transport
            else V4_RUNTIME_TASK_BLUEPRINT_STATUS_COVERAGE_NE)


def _mapping_for_witness(
        record: ConversationHeldOutV4PayloadWitnessReceiptRecord,
        ) -> ConversationHeldOutV4ActiveIntakeMapping:
    """从 P3-A2 canonical mapping identity 恢复 P3-A mapping 并复核全部共享字段。"""
    try:
        mapping = ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(
            record.mapping_identity)
    except (ConversationHeldOutV4ActiveIntakeError, IntegerCodecError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b witness mapping identity is corrupt") from exc
    if (mapping.stable_key() != record.mapping_identity
            or mapping.record_kind != record.record_kind
            or mapping.record_key != record.witness_record_key
            or mapping.source_ref != record.source_ref
            or mapping.leaf != record.leaf
            or mapping.content_sha256 != record.content_sha256):
        _fail("P3-C1b witness mapping binding drifted")
    return mapping


def _validate_turn(
        turn: ConversationHeldOutV4RuntimeTaskBlueprintTurn,
        *,
        witness: ConversationHeldOutV4PayloadWitnessReceiptRecord,
        mapping: ConversationHeldOutV4ActiveIntakeMapping,
        active_mapping_identities: frozenset[tuple[int, ...]],
        frozen_source_manifest_identity: ProtocolKey,
        annotation_witness: ConversationHeldOutV4RuntimeTaskAnnotationWitness,
        budget: ConversationHeldOutV4RuntimeTaskBlueprintBudget,
        ) -> ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord:
    """机械闭合一条 authored turn，不从任何正文、样例或 capsule 推导语义。"""
    if not isinstance(turn, ConversationHeldOutV4RuntimeTaskBlueprintTurn):
        raise TypeError("P3-C1b factory yielded an invalid turn")
    if mapping.stable_key() not in active_mapping_identities:
        _fail("P3-C1b witness mapping is not a current P3-A publication member")
    if len(turn.representations) > budget.max_representations_per_turn:
        _fail("P3-C1b Representation count exceeds budget")
    if len(turn.evidence_plans) > budget.max_evidence_plans_per_turn:
        _fail("P3-C1b Evidence plan count exceeds budget")
    if turn.witness_record_identity != witness.integer_stream():
        _fail("P3-C1b turn witness canonical identity drifted")
    if (turn.source_ref != witness.source_ref
            or turn.content_sha256 != witness.content_sha256
            or turn.content_byte_count != witness.content_byte_count
            or turn.content_scalar_count != witness.content_scalar_count
            or turn.source_span_byte_start != witness.source_span_byte_start
            or turn.source_span_byte_end != witness.source_span_byte_end
            or turn.source_span_scalar_start != witness.source_span_scalar_start
            or turn.source_span_scalar_end != witness.source_span_scalar_end):
        _fail("P3-C1b turn witness content binding drifted")
    if (witness.frozen_source_manifest_identity != frozen_source_manifest_identity
            or mapping.split != turn.split
            or mapping.license_partition != turn.license_partition):
        _fail("P3-C1b turn split/license/source manifest binding drifted")
    try:
        require_v4_runtime_task_annotation_witness_binding(
            annotation_witness,
            case_key=turn.case_key,
            turn_key=turn.turn_key,
            ordinal=turn.ordinal,
            witness_record_identity=turn.witness_record_identity,
            typed_task_identity=turn.stable_key(),
            annotation_origin_status=turn.annotation_origin_status,
            annotation_source_identity=turn.annotation_source_identity,
            annotation_revision_identity=turn.annotation_revision_identity,
            frozen_annotation_manifest_identity=turn.frozen_annotation_manifest_identity,
            constructing_code_identity=turn.constructing_code_identity,
            annotation_transform_code_identity=turn.annotation_transform_code_identity,
            applicable_domain_identity=turn.applicable_domain_identity)
    except ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b annotation witness binding failed") from exc
    for representation in turn.representations:
        if (representation.annotation_source_identity != turn.annotation_source_identity
                or representation.annotation_revision_identity != turn.annotation_revision_identity
                or representation.transform_code_identity
                != turn.annotation_transform_code_identity):
            _fail("P3-C1b Representation annotation binding drifted")
    try:
        validate_v4_runtime_task_structure(
            turn.request, turn.representations, turn.evidence_plans,
            source=turn.source_ref,
            scalar_start=turn.source_span_scalar_start,
            scalar_end=turn.source_span_scalar_end,
            label="P3-C1b blueprint turn")
    except ConversationHeldOutV4RuntimeTaskStructureError as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b task structure replay failed") from exc
    return ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord(
        _identity_digest(turn.case_key.components, domain=_DOMAIN_CASE_KEY,
                         label="P3-C1b case key identity"),
        _identity_digest(turn.turn_key.components, domain=_DOMAIN_TURN_KEY,
                         label="P3-C1b turn key identity"),
        turn.ordinal,
        _identity_digest(turn.witness_record_identity, domain=_DOMAIN_WITNESS_RECORD,
                         label="P3-C1b witness identity"),
        annotation_witness_public_identity_sha256(annotation_witness),
        turn.source_ref, turn.content_sha256, turn.content_byte_count,
        turn.content_scalar_count, turn.source_span_byte_start, turn.source_span_byte_end,
        turn.source_span_scalar_start, turn.source_span_scalar_end, turn.split,
        turn.license_partition, turn.annotation_origin_status,
        _identity_digest(turn.frozen_annotation_manifest_identity.components,
                         domain=_DOMAIN_ANNOTATION_MANIFEST,
                         label="P3-C1b annotation manifest identity"),
        _identity_digest(turn.constructing_code_identity.components, domain=_DOMAIN_CODE,
                         label="P3-C1b constructing code identity"),
        _identity_digest(turn.annotation_source_identity.components,
                         domain=_DOMAIN_ANNOTATION_SOURCE,
                         label="P3-C1b annotation source identity"),
        _identity_digest(turn.annotation_revision_identity.components,
                         domain=_DOMAIN_ANNOTATION_REVISION,
                         label="P3-C1b annotation revision identity"),
        _identity_digest(turn.annotation_transform_code_identity.components,
                         domain=_DOMAIN_ANNOTATION_TRANSFORM,
                         label="P3-C1b annotation transform identity"),
        _identity_digest(turn.applicable_domain_identity.components,
                         domain=_DOMAIN_APPLICABLE_DOMAIN,
                         label="P3-C1b applicable domain identity"),
        _identity_digest(turn.stable_key(), domain=_DOMAIN_TURN,
                         label="P3-C1b turn identity"),
        _identity_digest(turn.request.stable_key(), domain=_DOMAIN_REQUEST,
                         label="P3-C1b request identity"),
        tuple(_identity_digest(item.stable_key(), domain=_DOMAIN_REPRESENTATION,
                               label="P3-C1b Representation identity")
              for item in turn.representations),
        tuple(_identity_digest(item.stable_key(), domain=_DOMAIN_EVIDENCE,
                               label="P3-C1b Evidence identity")
              for item in turn.evidence_plans),
    )


def _require_receipt_turn_binding(
        record: ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord,
        turn: ConversationHeldOutV4RuntimeTaskBlueprintTurn,
        annotation_witness: ConversationHeldOutV4RuntimeTaskAnnotationWitness,
        ) -> None:
    """确保仅内存 turn 仍精确对应已发布的无正文 receipt，防止 adapter 消费被替换对象。"""
    if (record.case_key_identity_sha256 != _identity_digest(
                turn.case_key.components, domain=_DOMAIN_CASE_KEY,
                label="P3-C1b case key identity")
            or record.turn_key_identity_sha256 != _identity_digest(
                turn.turn_key.components, domain=_DOMAIN_TURN_KEY,
                label="P3-C1b turn key identity")
            or record.ordinal != turn.ordinal
            or record.witness_record_identity_sha256 != _identity_digest(
                turn.witness_record_identity, domain=_DOMAIN_WITNESS_RECORD,
                label="P3-C1b witness identity")
            or record.annotation_witness_identity_sha256
            != annotation_witness_public_identity_sha256(annotation_witness)
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
            or record.annotation_origin_status != turn.annotation_origin_status
            or record.frozen_annotation_manifest_identity_sha256 != _identity_digest(
                turn.frozen_annotation_manifest_identity.components,
                domain=_DOMAIN_ANNOTATION_MANIFEST,
                label="P3-C1b annotation manifest identity")
            or record.constructing_code_identity_sha256 != _identity_digest(
                turn.constructing_code_identity.components, domain=_DOMAIN_CODE,
                label="P3-C1b constructing code identity")
            or record.annotation_source_identity_sha256 != _identity_digest(
                turn.annotation_source_identity.components,
                domain=_DOMAIN_ANNOTATION_SOURCE,
                label="P3-C1b annotation source identity")
            or record.annotation_revision_identity_sha256 != _identity_digest(
                turn.annotation_revision_identity.components,
                domain=_DOMAIN_ANNOTATION_REVISION,
                label="P3-C1b annotation revision identity")
            or record.annotation_transform_code_identity_sha256 != _identity_digest(
                turn.annotation_transform_code_identity.components,
                domain=_DOMAIN_ANNOTATION_TRANSFORM,
                label="P3-C1b annotation transform identity")
            or record.applicable_domain_identity_sha256 != _identity_digest(
                turn.applicable_domain_identity.components,
                domain=_DOMAIN_APPLICABLE_DOMAIN,
                label="P3-C1b applicable domain identity")
            or record.turn_identity_sha256 != _identity_digest(
                turn.stable_key(), domain=_DOMAIN_TURN,
                label="P3-C1b turn identity")
            or record.request_identity_sha256 != _identity_digest(
                turn.request.stable_key(), domain=_DOMAIN_REQUEST,
                label="P3-C1b request identity")
            or record.representation_identity_sha256 != tuple(
                _identity_digest(item.stable_key(), domain=_DOMAIN_REPRESENTATION,
                                 label="P3-C1b Representation identity")
                for item in turn.representations)
            or record.evidence_plan_identity_sha256 != tuple(
                _identity_digest(item.stable_key(), domain=_DOMAIN_EVIDENCE,
                                 label="P3-C1b Evidence identity")
                for item in turn.evidence_plans)):
        _fail("P3-C1b in-memory turn no longer matches published receipt")


def _write_receipt(
        root: KRunRoot,
        relative: Path,
        records: Iterable[ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord],
        *, budget: ConversationHeldOutV4RuntimeTaskBlueprintBudget,
        label: str,
        ) -> ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream:
    """排他地流式发布无正文 receipt，并返回 footer/physical identity。"""
    ensure_normal_relative_directory(root, relative.parent, label=f"{label} parent")
    raw = open_exclusive_binary(root, relative, label=label)
    writer = IntegerFramedStreamWriter.from_open_binary(raw, path=relative)
    total = 0
    count = 0
    try:
        for item in records:
            payload = item.integer_stream()
            payload_size = encoded_integer_tuple_size(payload)
            if payload_size > budget.max_receipt_record_payload_bytes:
                _fail(f"{label} record exceeds budget")
            if total + payload_size > budget.max_receipt_payload_bytes:
                _fail(f"{label} payload exceeds budget")
            writer.append(payload)
            total += payload_size
            count += 1
        footer = writer.seal()
    except (IntegerCodecError, IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            f"{label} write failed") from exc
    finally:
        writer.close()
    if count == 0 or footer.record_count != count or footer.total_payload_bytes != total:
        _fail(f"{label} footer/count drifted")
    physical = sha256_plain_file(root, relative,
                                 max_bytes=budget.max_receipt_physical_bytes,
                                 label=f"{label} physical")
    return ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream(relative, footer, physical)


def _iter_receipt_records(
        root: KRunRoot,
        identity: ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream,
        *, budget: ConversationHeldOutV4RuntimeTaskBlueprintBudget,
        max_count: int,
        label: str,
        ) -> Iterator[ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord]:
    """有界回读并先闭合 footer/physical identity，之后才向 relay 暴露 records。"""
    try:
        file_identity = capture_plain_file_identity(root, identity.relative_path, label=label)
        stream = open_plain_binary(root, identity.relative_path, label=label,
                                   expected_identity=file_identity)
        reader = IntegerFramedStreamReader.from_open_binary(
            stream, path=identity.relative_path,
            max_frame_bytes=budget.max_receipt_record_payload_bytes,
            max_record_count=max_count,
            max_total_payload_bytes=budget.max_receipt_payload_bytes)
        try:
            records = tuple(
                ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord.from_integer_stream(raw)
                for raw in reader)
            footer = reader.finish()
        finally:
            reader.close()
        require_plain_file_identity(root, identity.relative_path, file_identity,
                                    label=f"{label} post-read")
        physical = sha256_plain_file(root, identity.relative_path,
                                     max_bytes=budget.max_receipt_physical_bytes,
                                     label=f"{label} physical")
    except ConversationHeldOutV4RuntimeTaskBlueprintError:
        raise
    except (IntegerCodecError, IntegerFramedStreamError, KRunBoundaryError,
            OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            f"{label} readback failed") from exc
    if footer != identity.p0_footer or physical != identity.physical:
        _fail(f"{label} stream identity drifted")
    yield from records


def _annotation_witnesses_by_case_turn(
        value: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult,
        ) -> dict[tuple[ProtocolKey, ProtocolKey], ConversationHeldOutV4RuntimeTaskAnnotationWitness]:
    """恢复已封存 P3-C1a inventory 的唯一 case/turn 索引，拒绝任何内存重复。"""
    if not isinstance(value, ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult):
        raise TypeError("P3-C1b annotation provenance type is invalid")
    result: dict[tuple[ProtocolKey, ProtocolKey], ConversationHeldOutV4RuntimeTaskAnnotationWitness] = {}
    for witness in value.witnesses:
        key = (witness.case_key, witness.turn_key)
        if key in result:
            _fail("P3-C1b annotation witness case/turn identity is duplicated")
        result[key] = witness
    if not result:
        _fail("P3-C1b annotation witness inventory is empty")
    return result


def _stage_blueprint_receipt(
        value: ConversationHeldOutV4RuntimeTaskBlueprintInput,
        witnesses: dict[tuple[int, ...], ConversationHeldOutV4PayloadWitnessReceiptRecord],
        *, frozen_source_manifest_identity: ProtocolKey,
        active_mapping_identities: frozenset[tuple[int, ...]],
        annotation_witnesses: dict[
            tuple[ProtocolKey, ProtocolKey],
            ConversationHeldOutV4RuntimeTaskAnnotationWitness],
        ) -> tuple[
            ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream,
            ConversationHeldOutV4RuntimeTaskBlueprintCounts,
            tuple[ConversationHeldOutV4RuntimeTaskBlueprintTurn, ...],
        ]:
    """消费 explicit factory，先完整回放再只将无正文 receipt 流式写入 staging。"""
    try:
        factory_items = iter(value.blueprint_factory())
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b blueprint factory is not iterable") from exc
    root = value.staging_run_root
    ensure_normal_relative_directory(root, _STAGING_RECEIPT_FILE.parent,
                                     label="P3-C1b staging receipt parent")
    raw = open_exclusive_binary(root, _STAGING_RECEIPT_FILE,
                                label="P3-C1b staging receipt")
    writer = IntegerFramedStreamWriter.from_open_binary(raw, path=_STAGING_RECEIPT_FILE)
    turns: list[ConversationHeldOutV4RuntimeTaskBlueprintTurn] = []
    total_payload_bytes = 0
    representation_count = 0
    evidence_plan_count = 0
    train_turn_count = 0
    held_out_turn_count = 0
    identities: set[tuple[ProtocolKey, ProtocolKey]] = set()
    consumed_annotation_keys: set[tuple[ProtocolKey, ProtocolKey]] = set()
    expected_ordinal = 1
    try:
        for turn in factory_items:
            if len(turns) >= value.budget.max_turn_count:
                _fail("P3-C1b turn count exceeds budget")
            if not isinstance(turn, ConversationHeldOutV4RuntimeTaskBlueprintTurn):
                raise TypeError("P3-C1b blueprint factory yielded an invalid turn")
            if turn.ordinal != expected_ordinal:
                _fail("P3-C1b blueprint ordinal is not consecutive")
            identity = (turn.case_key, turn.turn_key)
            if identity in identities:
                _fail("P3-C1b blueprint case/turn identity is duplicated")
            annotation_witness = annotation_witnesses.get(identity)
            if annotation_witness is None:
                _fail("P3-C1b turn has no sealed annotation witness")
            witness = witnesses.get(turn.witness_record_identity)
            if witness is None:
                _fail("P3-C1b turn references an undeclared P3-A2 witness")
            mapping = _mapping_for_witness(witness)
            receipt_record = _validate_turn(
                turn, witness=witness, mapping=mapping,
                active_mapping_identities=active_mapping_identities,
                frozen_source_manifest_identity=frozen_source_manifest_identity,
                annotation_witness=annotation_witness,
                budget=value.budget)
            payload = receipt_record.integer_stream()
            payload_size = encoded_integer_tuple_size(payload)
            if payload_size > value.budget.max_receipt_record_payload_bytes:
                _fail("P3-C1b staging receipt record exceeds budget")
            if total_payload_bytes + payload_size > value.budget.max_receipt_payload_bytes:
                _fail("P3-C1b staging receipt payload exceeds budget")
            writer.append(payload)
            total_payload_bytes += payload_size
            turns.append(turn)
            identities.add(identity)
            consumed_annotation_keys.add(identity)
            representation_count += len(turn.representations)
            evidence_plan_count += len(turn.evidence_plans)
            if turn.split == "train":
                train_turn_count += 1
            else:
                held_out_turn_count += 1
            expected_ordinal += 1
        if consumed_annotation_keys != set(annotation_witnesses):
            _fail("P3-C1b annotation witness inventory does not close")
        footer = writer.seal()
    except ConversationHeldOutV4RuntimeTaskBlueprintError:
        raise
    except (IntegerCodecError, IntegerFramedStreamError, KRunBoundaryError,
            OSError, TypeError, ValueError, UnicodeError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b blueprint staging failed") from exc
    finally:
        writer.close()
    if not turns:
        _fail("P3-C1b blueprint factory yielded no turns")
    counts = ConversationHeldOutV4RuntimeTaskBlueprintCounts(
        len(turns), train_turn_count, held_out_turn_count,
        representation_count, evidence_plan_count)
    if footer.record_count != counts.turn_count or footer.total_payload_bytes != total_payload_bytes:
        _fail("P3-C1b staging receipt footer/count drifted")
    physical = sha256_plain_file(
        root, _STAGING_RECEIPT_FILE,
        max_bytes=value.budget.max_receipt_physical_bytes,
        label="P3-C1b staging receipt physical")
    return (
        ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream(
            _STAGING_RECEIPT_FILE, footer, physical),
        counts,
        tuple(turns),
    )


def _manifest_integer_stream(
        value: ConversationHeldOutV4RuntimeTaskBlueprintResult,
        ) -> tuple[int, ...]:
    """生成仅绑定公开 receipt 的 manifest，绝不序列化内存 typed turns。"""
    result = [V4_RUNTIME_TASK_BLUEPRINT_MANIFEST_SCHEMA]
    for item in (value.receipt.integer_stream(), value.receipt_stream.integer_stream()):
        pack_key(result, item)
    return tuple(result)


def materialize_v4_runtime_task_blueprint(
        value: ConversationHeldOutV4RuntimeTaskBlueprintInput,
        ) -> ConversationHeldOutV4RuntimeTaskBlueprintResult:
    """运行 P3-C1b：回读三个上游公开身份，发布无正文 blueprint receipt。"""
    if not isinstance(value, ConversationHeldOutV4RuntimeTaskBlueprintInput):
        raise TypeError("P3-C1b materialize requires RuntimeTaskBlueprintInput")
    try:
        active_publication = load_v4_active_intake_publication(value.active_intake)
        active = active_publication.result
        readback = revalidate_v4_active_roster_bidirectional_readback(value.readback)
        payload_publication = load_v4_payload_witness_publication(
            value.payload_witness,
            max_record_count=value.budget.max_witness_record_count,
        )
        annotation_publication = load_v4_runtime_task_annotation_provenance_publication(
            value.annotation_provenance,
            max_upstream_payload_witness_record_count=(
                value.budget.max_witness_record_count),
        )
    except (ConversationHeldOutV4ActiveIntakeError,
            ConversationHeldOutV4ActiveRosterReadbackError,
            ConversationHeldOutV4PayloadWitnessLedgerError,
            ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b upstream publication readback failed") from exc
    payload_witness = payload_publication.result
    annotation_provenance = annotation_publication.result
    if (readback.receipt.active_intake_stable_key != active.stable_key()
            or payload_witness.receipt.active_intake_stable_key != active.stable_key()
            or payload_witness.receipt.readback_stable_key != readback.stable_key()
            or annotation_provenance.payload_witness.stable_key()
            != payload_witness.stable_key()):
        _fail("P3-C1b P3-A/P3-B/P3-A2 stable keys are not mutually bound")
    if len(payload_publication.records) > value.budget.max_witness_record_count:
        _fail("P3-C1b P3-A2 witness record count exceeds budget")
    try:
        status = _validate_roots(value)
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b verification root boundary failed") from exc
    witnesses = {
        item.integer_stream(): item for item in payload_publication.records
    }
    if len(witnesses) != len(payload_publication.records):
        _fail("P3-C1b P3-A2 witness identities are duplicated")
    active_mapping_identities = frozenset(
        item.stable_key() for item in active_publication.mappings)
    if len(active_mapping_identities) != len(active_publication.mappings):
        _fail("P3-C1b P3-A mapping identities are duplicated")
    witness_mapping_identities = []
    for witness in payload_publication.records:
        mapping = _mapping_for_witness(witness)
        mapping_identity = mapping.stable_key()
        if mapping_identity not in active_mapping_identities:
            _fail("P3-C1b P3-A2 witness mapping is not a current P3-A member")
        witness_mapping_identities.append(mapping_identity)
    if (len(set(witness_mapping_identities)) != len(witness_mapping_identities)
            or frozenset(witness_mapping_identities) != active_mapping_identities):
        _fail("P3-C1b P3-A/P3-A2 mapping inventory set does not close")
    annotation_witnesses = _annotation_witnesses_by_case_turn(annotation_provenance)
    staged_receipt, counts, turns = _stage_blueprint_receipt(
        value, witnesses,
        frozen_source_manifest_identity=payload_witness.receipt.frozen_source_manifest_identity,
        active_mapping_identities=active_mapping_identities,
        annotation_witnesses=annotation_witnesses)
    receipt = ConversationHeldOutV4RuntimeTaskBlueprintReceipt(
        _identity_digest(active.stable_key(), domain=_DOMAIN_ACTIVE,
                         label="P3-C1b active identity"),
        _identity_digest(readback.stable_key(), domain=_DOMAIN_READBACK,
                         label="P3-C1b readback identity"),
        _identity_digest(payload_witness.stable_key(), domain=_DOMAIN_PAYLOAD_WITNESS,
                         label="P3-C1b payload witness identity"),
        _identity_digest(annotation_provenance.stable_key(),
                         domain=_DOMAIN_ANNOTATION_PROVENANCE,
                         label="P3-C1b annotation provenance identity"),
        value.budget, counts, status)
    try:
        verified_staging_records = tuple(_iter_receipt_records(
            value.staging_run_root, staged_receipt, budget=value.budget,
            max_count=counts.turn_count, label="P3-C1b staging receipt"))
        work_receipt = _write_receipt(
            value.work_run_root, _WORK_RECEIPT_FILE, verified_staging_records,
            budget=value.budget, label="P3-C1b work receipt")
        verified_work_records = tuple(_iter_receipt_records(
            value.work_run_root, work_receipt, budget=value.budget,
            max_count=counts.turn_count, label="P3-C1b work receipt"))
        receipt_stream = _write_receipt(
            value.publication_run_root, _RECEIPT_FILE, verified_work_records,
            budget=value.budget, label="P3-C1b receipt")
        provisional = ConversationHeldOutV4RuntimeTaskBlueprintResult(
            value.publication_run_root, receipt, receipt_stream,
            (0,) * _SHA256_SIZE, annotation_provenance, turns)
        manifest_payload = encode_integer_tuple(_manifest_integer_stream(provisional))
        if len(manifest_payload) > value.budget.max_manifest_bytes:
            _fail("P3-C1b manifest exceeds budget")
        publish_manifest_last(value.publication_run_root, _MANIFEST_FILE, manifest_payload,
                              frozenset({_RECEIPT_FILE}), label="P3-C1b publication")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b publication boundary failed") from exc
    result = ConversationHeldOutV4RuntimeTaskBlueprintResult(
        value.publication_run_root, receipt, receipt_stream,
        tuple(hashlib.sha256(manifest_payload).digest()),
        annotation_provenance, turns)
    return revalidate_v4_runtime_task_blueprint(result)


def revalidate_v4_runtime_task_blueprint(
        value: ConversationHeldOutV4RuntimeTaskBlueprintResult,
        ) -> ConversationHeldOutV4RuntimeTaskBlueprintResult:
    """只读回验 P3-C1b public receipt/manifest，不访问 raw payload 或 typed turn 正文。"""
    if not isinstance(value, ConversationHeldOutV4RuntimeTaskBlueprintResult):
        raise TypeError("P3-C1b revalidate requires RuntimeTaskBlueprintResult")
    try:
        annotation_publication = load_v4_runtime_task_annotation_provenance_publication(
            value.annotation_provenance,
            max_upstream_payload_witness_record_count=(
                value.receipt.budget.max_witness_record_count),
        )
    except (ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b annotation provenance readback failed") from exc
    annotation_provenance = annotation_publication.result
    try:
        require_exact_file_closure(value.publication_run_root,
                                   frozenset({_RECEIPT_FILE, _MANIFEST_FILE}),
                                   label="P3-C1b publication")
        records = tuple(_iter_receipt_records(
            value.publication_run_root, value.receipt_stream,
            budget=value.receipt.budget, max_count=value.receipt.counts.turn_count,
            label="P3-C1b receipt"))
    except (KRunBoundaryError, IntegerCodecError, IntegerFramedStreamError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b receipt publication readback failed") from exc
    if len(records) != value.receipt.counts.turn_count:
        _fail("P3-C1b receipt count drifted")
    if tuple(item.ordinal for item in records) != tuple(range(1, len(records) + 1)):
        _fail("P3-C1b receipt ordinal drifted")
    if len({(item.case_key_identity_sha256, item.turn_key_identity_sha256)
            for item in records}) != len(records):
        _fail("P3-C1b receipt case/turn identity drifted")
    if (value.receipt.annotation_provenance_identity_sha256
            != _identity_digest(annotation_provenance.stable_key(),
                                domain=_DOMAIN_ANNOTATION_PROVENANCE,
                                label="P3-C1b annotation provenance identity")
            or value.receipt.payload_witness_identity_sha256
            != _identity_digest(annotation_provenance.payload_witness.stable_key(),
                                domain=_DOMAIN_PAYLOAD_WITNESS,
                                label="P3-C1b payload witness identity")):
        _fail("P3-C1b annotation provenance receipt binding drifted")
    annotation_witnesses = _annotation_witnesses_by_case_turn(annotation_provenance)
    consumed_annotation_keys: set[tuple[ProtocolKey, ProtocolKey]] = set()
    for record, turn in zip(records, value.turns):
        key = (turn.case_key, turn.turn_key)
        annotation_witness = annotation_witnesses.get(key)
        if annotation_witness is None:
            _fail("P3-C1b in-memory turn has no sealed annotation witness")
        try:
            require_v4_runtime_task_annotation_witness_binding(
                annotation_witness,
                case_key=turn.case_key,
                turn_key=turn.turn_key,
                ordinal=turn.ordinal,
                witness_record_identity=turn.witness_record_identity,
                typed_task_identity=turn.stable_key(),
                annotation_origin_status=turn.annotation_origin_status,
                annotation_source_identity=turn.annotation_source_identity,
                annotation_revision_identity=turn.annotation_revision_identity,
                frozen_annotation_manifest_identity=turn.frozen_annotation_manifest_identity,
                constructing_code_identity=turn.constructing_code_identity,
                annotation_transform_code_identity=turn.annotation_transform_code_identity,
                applicable_domain_identity=turn.applicable_domain_identity)
        except ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError as exc:
            raise ConversationHeldOutV4RuntimeTaskBlueprintError(
                "P3-C1b in-memory turn annotation witness drifted") from exc
        _require_receipt_turn_binding(record, turn, annotation_witness)
        consumed_annotation_keys.add(key)
    if consumed_annotation_keys != set(annotation_witnesses):
        _fail("P3-C1b annotation witness inventory does not close")
    if (sum(item.split == "train" for item in records) != value.receipt.counts.train_turn_count
            or sum(item.split == "held_out" for item in records)
            != value.receipt.counts.held_out_turn_count
            or sum(len(item.representation_identity_sha256) for item in records)
            != value.receipt.counts.representation_count
            or sum(len(item.evidence_plan_identity_sha256) for item in records)
            != value.receipt.counts.evidence_plan_count):
        _fail("P3-C1b receipt aggregate counts drifted")
    try:
        manifest_identity = capture_plain_file_identity(value.publication_run_root, _MANIFEST_FILE,
                                                        label="P3-C1b manifest")
        with open_plain_binary(value.publication_run_root, _MANIFEST_FILE,
                               label="P3-C1b manifest", expected_identity=manifest_identity) as stream:
            manifest_payload = stream.read(value.receipt.budget.max_manifest_bytes + 1)
        require_plain_file_identity(value.publication_run_root, _MANIFEST_FILE, manifest_identity,
                                    label="P3-C1b manifest post-read")
    except (KRunBoundaryError, OSError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b manifest readback failed") from exc
    if len(manifest_payload) > value.receipt.budget.max_manifest_bytes:
        _fail("P3-C1b manifest exceeds budget")
    try:
        decoded = decode_integer_tuple(manifest_payload)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b manifest is corrupt") from exc
    if (encode_integer_tuple(decoded) != manifest_payload
            or decoded != _manifest_integer_stream(value)
            or tuple(hashlib.sha256(manifest_payload).digest()) != value.manifest_sha256):
        _fail("P3-C1b manifest identity drifted")
    return value


def load_v4_runtime_task_blueprint_publication(
        value: ConversationHeldOutV4RuntimeTaskBlueprintResult,
        ) -> ConversationHeldOutV4RuntimeTaskBlueprintPublication:
    """完整回读 C1b publication，并返回仍绑定内存 turn 的无正文 receipt records。"""
    result = revalidate_v4_runtime_task_blueprint(value)
    try:
        records = tuple(_iter_receipt_records(
            result.publication_run_root, result.receipt_stream,
            budget=result.receipt.budget,
            max_count=result.receipt.counts.turn_count,
            label="P3-C1b public receipt"))
    except (KRunBoundaryError, IntegerCodecError, IntegerFramedStreamError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskBlueprintError(
            "P3-C1b public receipt loader failed") from exc
    if len(records) != len(result.turns):
        _fail("P3-C1b public receipt/turn count drifted")
    return ConversationHeldOutV4RuntimeTaskBlueprintPublication(result, records)


__all__ = [
    "ConversationHeldOutV4RuntimeTaskBlueprintBudget",
    "ConversationHeldOutV4RuntimeTaskBlueprintCounts",
    "ConversationHeldOutV4RuntimeTaskBlueprintError",
    "ConversationHeldOutV4RuntimeTaskBlueprintInput",
    "ConversationHeldOutV4RuntimeTaskBlueprintPublication",
    "ConversationHeldOutV4RuntimeTaskBlueprintReceipt",
    "ConversationHeldOutV4RuntimeTaskBlueprintReceiptRecord",
    "ConversationHeldOutV4RuntimeTaskBlueprintReceiptStream",
    "ConversationHeldOutV4RuntimeTaskBlueprintResult",
    "ConversationHeldOutV4RuntimeTaskBlueprintTurn",
    "V4_RUNTIME_TASK_BLUEPRINT_STATUS_COVERAGE_NE",
    "V4_RUNTIME_TASK_BLUEPRINT_STATUS_TEST_ONLY",
    "load_v4_runtime_task_blueprint_publication",
    "materialize_v4_runtime_task_blueprint",
    "revalidate_v4_runtime_task_blueprint",
]
