"""DLG-05 v4 P3-A2 载荷见证 ledger。

P3-A2 只证明调用方给出的载荷 bytes 是已封存 P3-A mapping 的实际内容。它只发布
identity、hash、计数和 span，绝不发布正文；也不构造 runtime capsule 或导入
runtime、teacher、owner、training 设施。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Iterator

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.unicode_representation import (
    validate_unicode_scalars,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_intake import (
    ConversationHeldOutV4ActiveIntakeError,
    ConversationHeldOutV4ActiveIntakeMapping,
    ConversationHeldOutV4ActiveIntakeResult,
    V4_ACTIVE_INTAKE_RECORD_OBSERVATION,
    V4_ACTIVE_INTAKE_RECORD_SOURCE,
    V4_ACTIVE_INTAKE_RECORD_TEACHER,
    revalidate_v4_active_intake,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback import (
    ConversationHeldOutV4ActiveRosterReadbackError,
    ConversationHeldOutV4ActiveRosterReadbackResult,
    revalidate_v4_active_roster_bidirectional_readback,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4ProvenanceLeaf,
    ConversationHeldOutV4SnapshotIdentity,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import StableRecordKey
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


V4_PAYLOAD_WITNESS_ENCODING_UTF8_BYTES = "UTF8_BYTES_V1"
V4_PAYLOAD_WITNESS_ENCODING_UTF8_SCALARS = "UTF8_SCALARS_V1"
V4_PAYLOAD_WITNESS_BUDGET_SCHEMA = 1
V4_PAYLOAD_WITNESS_RECORD_SCHEMA = 1
V4_PAYLOAD_WITNESS_RECEIPT_SCHEMA = 1
V4_PAYLOAD_WITNESS_RESULT_SCHEMA = 1
V4_PAYLOAD_WITNESS_MANIFEST_SCHEMA = 1
V4_PAYLOAD_WITNESS_STATUS_TEST_ONLY = "P3_A2_PAYLOAD_WITNESS_TEST_ONLY"
V4_PAYLOAD_WITNESS_STATUS_COVERAGE_NE = "ACTIVE_PROVENANCE_COVERAGE_NE"

_MAPPING_FILE = Path("mapping") / "records.pifrs"
_STAGING_RECEIPT_FILE = Path("candidate") / "receipt.pifrs"
_WORK_RECEIPT_FILE = Path("verified") / "receipt.pifrs"
_RECEIPT_FILE = Path("witness") / "receipt.pifrs"
_MANIFEST_FILE = Path("manifest.pii")
_SHA256_SIZE = hashlib.sha256().digest_size
_ENCODINGS = frozenset({
    V4_PAYLOAD_WITNESS_ENCODING_UTF8_BYTES,
    V4_PAYLOAD_WITNESS_ENCODING_UTF8_SCALARS,
})


class ConversationHeldOutV4PayloadWitnessLedgerError(RuntimeError):
    """P3-A2 binding、byte witness、root 边界或 receipt 闭合失败。"""


def _fail(message: str) -> None:
    raise ConversationHeldOutV4PayloadWitnessLedgerError(message)


def _text_scalars(value: str, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty str")
    return validate_unicode_scalars(tuple(ord(item) for item in value))


def _require_digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or len(value) != _SHA256_SIZE
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ValueError(f"{label} must be a SHA-256 byte tuple")
    return value


def _require_nonnegative(value: int, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative strict int")
    return value


def _require_positive(value: int, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive strict int")
    return value


def _stage_name(value: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 48
            or not value[0].isascii() or not value[0].isalnum()
            or any(not item.isascii() or not (item.isalnum() or item in {"-", "_", "."})
                   for item in value)):
        raise ValueError("P3-A2 logical_stage_name is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitnessBudget:
    """约束 P3-A2 mapping 读取、payload 检查和公开 evidence 写入。"""

    max_mapping_record_count: int
    max_mapping_record_payload_bytes: int
    max_mapping_stream_payload_bytes: int
    max_mapping_stream_physical_bytes: int
    max_witness_content_bytes: int
    max_total_content_bytes: int
    max_receipt_record_payload_bytes: int
    max_receipt_payload_bytes: int
    max_receipt_physical_bytes: int
    max_manifest_bytes: int

    def __post_init__(self) -> None:
        for label, value in zip((
                "max_mapping_record_count", "max_mapping_record_payload_bytes",
                "max_mapping_stream_payload_bytes", "max_mapping_stream_physical_bytes",
                "max_witness_content_bytes", "max_total_content_bytes", "max_receipt_record_payload_bytes",
                "max_receipt_payload_bytes", "max_receipt_physical_bytes",
                "max_manifest_bytes"), self.integer_stream()):
            _require_positive(value, label=f"P3-A2 {label}")
        if self.max_receipt_record_payload_bytes > self.max_receipt_payload_bytes:
            raise ValueError("P3-A2 receipt record budget exceeds receipt budget")

    def integer_stream(self) -> tuple[int, ...]:
        return (
            V4_PAYLOAD_WITNESS_BUDGET_SCHEMA,
            self.max_mapping_record_count,
            self.max_mapping_record_payload_bytes,
            self.max_mapping_stream_payload_bytes,
            self.max_mapping_stream_physical_bytes,
            self.max_witness_content_bytes,
            self.max_total_content_bytes,
            self.max_receipt_record_payload_bytes,
            self.max_receipt_payload_bytes,
            self.max_receipt_physical_bytes,
            self.max_manifest_bytes,
        )


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitness:
    """一条仅在内存中的 payload observation；``content_*`` 绝不进入 output。"""

    active_record_key: StableRecordKey
    record_kind: int
    source_ref: SourceRef
    leaf: ConversationHeldOutV4ProvenanceLeaf
    snapshot: ConversationHeldOutV4SnapshotIdentity
    split: str
    license_partition: str
    consumed_input_identity: ProtocolKey
    source_uri_scalars: tuple[int, ...]
    source_span_byte_start: int
    source_span_byte_end: int
    source_span_scalar_start: int
    source_span_scalar_end: int
    encoding: str
    frozen_source_manifest_identity: ProtocolKey
    source_payload_identity: ProtocolKey
    content_bytes: bytes | None = None
    content_scalars: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.active_record_key, StableRecordKey):
            raise TypeError("P3-A2 witness active_record_key type is invalid")
        if self.record_kind not in {
                V4_ACTIVE_INTAKE_RECORD_SOURCE,
                V4_ACTIVE_INTAKE_RECORD_OBSERVATION}:
            raise ValueError("P3-A2 witness record_kind is invalid")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("P3-A2 witness source_ref type is invalid")
        if not isinstance(self.leaf, ConversationHeldOutV4ProvenanceLeaf):
            raise TypeError("P3-A2 witness leaf type is invalid")
        if not isinstance(self.snapshot, ConversationHeldOutV4SnapshotIdentity):
            raise TypeError("P3-A2 witness snapshot type is invalid")
        if not isinstance(self.consumed_input_identity, ProtocolKey):
            raise TypeError("P3-A2 witness consumed identity type is invalid")
        if not isinstance(self.source_payload_identity, ProtocolKey):
            raise TypeError("P3-A2 witness source payload identity type is invalid")
        if not isinstance(self.frozen_source_manifest_identity, ProtocolKey):
            raise TypeError("P3-A2 witness source manifest identity type is invalid")
        if self.encoding not in _ENCODINGS:
            raise ValueError("P3-A2 witness encoding is unregistered")
        if not isinstance(self.split, str) or self.split not in {"train", "held_out"}:
            raise ValueError("P3-A2 witness split is invalid")
        if not isinstance(self.license_partition, str) or not self.license_partition:
            raise ValueError("P3-A2 witness license partition is invalid")
        validate_unicode_scalars(self.source_uri_scalars)
        if self.source_uri_scalars != self.snapshot.official_uri_scalars:
            raise ValueError("P3-A2 witness URI does not match snapshot")
        if (self.content_bytes is None) == (self.content_scalars is None):
            raise ValueError("P3-A2 witness must carry exactly one content representation")
        if self.encoding == V4_PAYLOAD_WITNESS_ENCODING_UTF8_BYTES:
            if not isinstance(self.content_bytes, bytes) or not self.content_bytes:
                raise ValueError("P3-A2 UTF-8 bytes witness content is invalid")
            if self.content_scalars is not None:
                raise ValueError("P3-A2 UTF-8 bytes witness carries scalars")
        else:
            if self.content_bytes is not None:
                raise ValueError("P3-A2 UTF-8 scalar witness carries bytes")
            validate_unicode_scalars(self.content_scalars or ())
        self._validate_spans()

    def canonical_bytes_and_scalars(self) -> tuple[bytes, tuple[int, ...]]:
        """把显式编码正规化为 canonical UTF-8 bytes 和 scalars。"""
        if self.encoding == V4_PAYLOAD_WITNESS_ENCODING_UTF8_BYTES:
            assert self.content_bytes is not None
            try:
                text = self.content_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ConversationHeldOutV4PayloadWitnessLedgerError(
                    "P3-A2 UTF-8 bytes witness is not decodable") from exc
            if text.encode("utf-8") != self.content_bytes:
                _fail("P3-A2 UTF-8 bytes witness is not canonical")
            scalars = validate_unicode_scalars(tuple(ord(item) for item in text))
            return self.content_bytes, scalars
        scalars = validate_unicode_scalars(self.content_scalars or ())
        payload = "".join(chr(item) for item in scalars).encode("utf-8")
        return payload, scalars

    def _validate_spans(self) -> None:
        payload, scalars = self.canonical_bytes_and_scalars()
        for label, value in (
                ("byte start", self.source_span_byte_start),
                ("byte end", self.source_span_byte_end),
                ("scalar start", self.source_span_scalar_start),
                ("scalar end", self.source_span_scalar_end)):
            _require_nonnegative(value, label=f"P3-A2 witness {label}")
        if (self.source_span_byte_start >= self.source_span_byte_end
                or self.source_span_byte_end > len(payload)
                or self.source_span_scalar_start >= self.source_span_scalar_end
                or self.source_span_scalar_end > len(scalars)):
            raise ValueError("P3-A2 witness source span is outside content")
        prefix = "".join(chr(item) for item in scalars[:self.source_span_scalar_start])
        selected = "".join(chr(item) for item in scalars[
            self.source_span_scalar_start:self.source_span_scalar_end])
        byte_start = len(prefix.encode("utf-8"))
        byte_end = byte_start + len(selected.encode("utf-8"))
        if (byte_start != self.source_span_byte_start
                or byte_end != self.source_span_byte_end):
            raise ValueError("P3-A2 witness byte/scalar spans disagree")

    def canonical_sort_key(self) -> tuple[int, ...]:
        """factory 顺序就是 sealed P3-A mapping 顺序，不接受隐式排序。"""
        return (self.record_kind, *self.active_record_key.components)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitnessReceiptRecord:
    """一条 mapping/witness equality 的公开无正文 evidence。"""

    mapping_identity: tuple[int, ...]
    record_kind: int
    witness_record_key: StableRecordKey
    source_ref: SourceRef
    leaf: ConversationHeldOutV4ProvenanceLeaf
    content_sha256: tuple[int, ...]
    content_byte_count: int
    content_scalar_count: int
    source_span_byte_start: int
    source_span_byte_end: int
    source_span_scalar_start: int
    source_span_scalar_end: int
    encoding: str
    frozen_source_manifest_identity: ProtocolKey
    source_payload_identity: ProtocolKey

    def __post_init__(self) -> None:
        strict_integer_tuple(self.mapping_identity, label="P3-A2 receipt mapping identity")
        if self.record_kind not in {
                V4_ACTIVE_INTAKE_RECORD_SOURCE,
                V4_ACTIVE_INTAKE_RECORD_OBSERVATION}:
            raise ValueError("P3-A2 receipt record_kind is invalid")
        if not isinstance(self.witness_record_key, StableRecordKey):
            raise TypeError("P3-A2 receipt record key type is invalid")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("P3-A2 receipt source_ref type is invalid")
        if not isinstance(self.leaf, ConversationHeldOutV4ProvenanceLeaf):
            raise TypeError("P3-A2 receipt leaf type is invalid")
        _require_digest(self.content_sha256, label="P3-A2 receipt content hash")
        _require_positive(self.content_byte_count, label="P3-A2 receipt byte count")
        _require_positive(self.content_scalar_count, label="P3-A2 receipt scalar count")
        for label, item in (("byte start", self.source_span_byte_start),
                            ("byte end", self.source_span_byte_end),
                            ("scalar start", self.source_span_scalar_start),
                            ("scalar end", self.source_span_scalar_end)):
            _require_nonnegative(item, label=f"P3-A2 receipt {label}")
        if (self.source_span_byte_start >= self.source_span_byte_end
                or self.source_span_byte_end > self.content_byte_count
                or self.source_span_scalar_start >= self.source_span_scalar_end
                or self.source_span_scalar_end > self.content_scalar_count):
            raise ValueError("P3-A2 receipt span is invalid")
        if self.encoding not in _ENCODINGS:
            raise ValueError("P3-A2 receipt encoding is unregistered")
        if not isinstance(self.source_payload_identity, ProtocolKey):
            raise TypeError("P3-A2 receipt source payload identity type is invalid")
        if not isinstance(self.frozen_source_manifest_identity, ProtocolKey):
            raise TypeError("P3-A2 receipt source manifest identity type is invalid")

    def integer_stream(self) -> tuple[int, ...]:
        result = [V4_PAYLOAD_WITNESS_RECORD_SCHEMA]
        for value in (
                self.mapping_identity,
                (self.record_kind,),
                self.witness_record_key.components,
                self.source_ref.stable_key(),
                self.leaf.integer_stream(),
                self.content_sha256,
                (self.content_byte_count, self.content_scalar_count),
                (self.source_span_byte_start, self.source_span_byte_end,
                 self.source_span_scalar_start, self.source_span_scalar_end),
                _text_scalars(self.encoding, label="P3-A2 receipt encoding"),
                self.frozen_source_manifest_identity.components,
                self.source_payload_identity.components):
            pack_key(result, value)
        return tuple(result)

    @classmethod
    def from_integer_stream(cls, value: tuple[int, ...]) -> "ConversationHeldOutV4PayloadWitnessReceiptRecord":
        try:
            reader = IntegerStreamReader(value)
            if reader.read_positive(label="P3-A2 receipt schema") != V4_PAYLOAD_WITNESS_RECORD_SCHEMA:
                _fail("P3-A2 receipt schema is incompatible")
            mapping_identity = reader.read_key(label="P3-A2 receipt mapping identity")
            record_kind = reader.read_key(label="P3-A2 receipt record kind")
            if len(record_kind) != 1:
                _fail("P3-A2 receipt record kind is invalid")
            record_key = StableRecordKey(reader.read_key(label="P3-A2 receipt record key"))
            source_ref = SourceRef.from_stable_key(reader.read_key(label="P3-A2 receipt source"))
            leaf = ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(
                reader.read_key(label="P3-A2 receipt leaf"))
            content_sha256 = reader.read_key(label="P3-A2 receipt content hash")
            content_counts = reader.read_key(label="P3-A2 receipt content counts")
            if len(content_counts) != 2:
                _fail("P3-A2 receipt content counts are invalid")
            spans = reader.read_key(label="P3-A2 receipt spans")
            encoding = "".join(chr(item) for item in reader.read_key(
                label="P3-A2 receipt encoding"))
            source_manifest_identity = ProtocolKey(reader.read_key(
                label="P3-A2 receipt source manifest identity"))
            source_payload_identity = ProtocolKey(reader.read_key(
                label="P3-A2 receipt source payload identity"))
            result = cls(
                mapping_identity, record_kind[0], record_key, source_ref, leaf, content_sha256,
                content_counts[0],
                content_counts[1],
                *spans, encoding, source_manifest_identity, source_payload_identity,
            )
            reader.finish()
        except (IntegerCodecError, TypeError, ValueError, IndexError) as exc:
            raise ConversationHeldOutV4PayloadWitnessLedgerError(
                "P3-A2 receipt record is corrupt") from exc
        if result.integer_stream() != value:
            _fail("P3-A2 receipt record is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitnessReceiptStream:
    relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        if (not isinstance(self.relative_path, Path) or self.relative_path.is_absolute()
                or self.relative_path.drive or self.relative_path.root
                or not self.relative_path.parts or ".." in self.relative_path.parts
                or not isinstance(self.p0_footer, IntegerFramedStreamFooter)
                or not isinstance(self.physical, KRunFileDigest)
                or self.physical.byte_count <= 0):
            raise ValueError("P3-A2 receipt stream identity is invalid")

    def integer_stream(self) -> tuple[int, ...]:
        result = []
        for value in (
                _text_scalars(self.relative_path.as_posix(), label="P3-A2 receipt path"),
                self.p0_footer.integer_tuple(),
                (self.physical.byte_count, *self.physical.sha256)):
            pack_key(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitnessCounts:
    mapping_count: int
    witness_count: int
    content_byte_count: int
    content_scalar_count: int

    def __post_init__(self) -> None:
        for label, value in zip(("mapping_count", "witness_count", "content_byte_count", "content_scalar_count"),
                                self.integer_stream()):
            _require_nonnegative(value, label=f"P3-A2 {label}")
        if self.mapping_count == 0 or self.mapping_count != self.witness_count:
            raise ValueError("P3-A2 mapping/witness counts do not close")

    def integer_stream(self) -> tuple[int, ...]:
        return (self.mapping_count, self.witness_count,
                self.content_byte_count, self.content_scalar_count)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitnessReceipt:
    active_intake_stable_key: tuple[int, ...]
    readback_stable_key: tuple[int, ...]
    frozen_source_manifest_identity: ProtocolKey
    code_identity: ProtocolKey
    budget: ConversationHeldOutV4PayloadWitnessBudget
    counts: ConversationHeldOutV4PayloadWitnessCounts
    status: str

    def __post_init__(self) -> None:
        strict_integer_tuple(self.active_intake_stable_key, label="P3-A2 active intake key")
        strict_integer_tuple(self.readback_stable_key, label="P3-A2 readback key")
        if (not isinstance(self.frozen_source_manifest_identity, ProtocolKey)
                or not isinstance(self.code_identity, ProtocolKey)
                or not isinstance(self.budget, ConversationHeldOutV4PayloadWitnessBudget)
                or not isinstance(self.counts, ConversationHeldOutV4PayloadWitnessCounts)
                or self.status not in {
                    V4_PAYLOAD_WITNESS_STATUS_TEST_ONLY,
                    V4_PAYLOAD_WITNESS_STATUS_COVERAGE_NE}):
            raise ValueError("P3-A2 receipt fields are invalid")

    def integer_stream(self) -> tuple[int, ...]:
        result = [V4_PAYLOAD_WITNESS_RECEIPT_SCHEMA]
        for value in (
                self.active_intake_stable_key, self.readback_stable_key,
                self.frozen_source_manifest_identity.components,
                self.code_identity.components, self.budget.integer_stream(),
                self.counts.integer_stream(), _text_scalars(self.status, label="P3-A2 status")):
            pack_key(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitnessInput:
    active_intake: ConversationHeldOutV4ActiveIntakeResult
    readback: ConversationHeldOutV4ActiveRosterReadbackResult
    frozen_source_manifest_identity: ProtocolKey
    code_identity: ProtocolKey
    witness_factory: Callable[[], Iterable[ConversationHeldOutV4PayloadWitness]]
    staging_run_root: KRunRoot
    work_run_root: KRunRoot
    publication_run_root: KRunRoot
    budget: ConversationHeldOutV4PayloadWitnessBudget
    logical_stage_name: str

    def __post_init__(self) -> None:
        if (not isinstance(self.active_intake, ConversationHeldOutV4ActiveIntakeResult)
                or not isinstance(self.readback, ConversationHeldOutV4ActiveRosterReadbackResult)
                or not isinstance(self.frozen_source_manifest_identity, ProtocolKey)
                or not isinstance(self.code_identity, ProtocolKey)
                or not callable(self.witness_factory)
                or not isinstance(self.budget, ConversationHeldOutV4PayloadWitnessBudget)):
            raise TypeError("P3-A2 input types are invalid")
        for root in (self.staging_run_root, self.work_run_root, self.publication_run_root):
            if not isinstance(root, KRunRoot):
                raise TypeError("P3-A2 run root type is invalid")
        _stage_name(self.logical_stage_name)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitnessResult:
    publication_run_root: KRunRoot
    receipt: ConversationHeldOutV4PayloadWitnessReceipt
    receipt_stream: ConversationHeldOutV4PayloadWitnessReceiptStream
    manifest_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.publication_run_root, KRunRoot)
                or not isinstance(self.receipt, ConversationHeldOutV4PayloadWitnessReceipt)
                or not isinstance(self.receipt_stream, ConversationHeldOutV4PayloadWitnessReceiptStream)):
            raise TypeError("P3-A2 result types are invalid")
        if self.receipt_stream.relative_path != _RECEIPT_FILE:
            raise ValueError("P3-A2 result receipt path drifted")
        if self.receipt_stream.p0_footer.record_count != self.receipt.counts.witness_count:
            raise ValueError("P3-A2 result receipt/footer count drifted")
        _require_digest(self.manifest_sha256, label="P3-A2 manifest hash")
        expected_status = (V4_PAYLOAD_WITNESS_STATUS_TEST_ONLY
                           if self.publication_run_root.test_transport
                           else V4_PAYLOAD_WITNESS_STATUS_COVERAGE_NE)
        if self.receipt.status != expected_status:
            raise ValueError("P3-A2 result transport status drifted")

    def stable_key(self) -> tuple[int, ...]:
        result = [V4_PAYLOAD_WITNESS_RESULT_SCHEMA]
        for value in (self.receipt.integer_stream(), self.receipt_stream.integer_stream(),
                      self.manifest_sha256):
            pack_key(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4PayloadWitnessPublication:
    """已回读的 P3-A2 公开见证回执；只包含无正文 P0 records。"""

    result: ConversationHeldOutV4PayloadWitnessResult
    records: tuple[ConversationHeldOutV4PayloadWitnessReceiptRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.result, ConversationHeldOutV4PayloadWitnessResult):
            raise TypeError("P3-A2 publication result type is invalid")
        if (not isinstance(self.records, tuple)
                or any(not isinstance(item, ConversationHeldOutV4PayloadWitnessReceiptRecord)
                       for item in self.records)):
            raise TypeError("P3-A2 publication records are invalid")
        if len(self.records) != self.result.receipt.counts.witness_count:
            raise ValueError("P3-A2 publication record count drifted")


def _validate_roots(value: ConversationHeldOutV4PayloadWitnessInput) -> str:
    outputs = (value.staging_run_root, value.work_run_root, value.publication_run_root)
    inputs = (value.active_intake.publication_run_root, value.readback.publication_run_root)
    if len({root.test_transport for root in (*outputs, *inputs)}) != 1:
        _fail("P3-A2 roots must use one transport mode")
    for left_index, left in enumerate(outputs):
        for right in outputs[left_index + 1:]:
            require_disjoint_run_roots(left, right, label="P3-A2 output roots")
    for output in outputs:
        for source in inputs:
            require_disjoint_run_roots(output, source, label="P3-A2 input/output roots")
        require_fresh_empty_run_root(output, label="P3-A2 fresh verification root")
    return (V4_PAYLOAD_WITNESS_STATUS_TEST_ONLY if value.publication_run_root.test_transport
            else V4_PAYLOAD_WITNESS_STATUS_COVERAGE_NE)


def _iter_mappings(
        active: ConversationHeldOutV4ActiveIntakeResult,
        *, budget: ConversationHeldOutV4PayloadWitnessBudget,
        ) -> Iterator[ConversationHeldOutV4ActiveIntakeMapping]:
    """在公开 revalidation 后读取 sealed P3-A mapping stream。"""
    try:
        identity = capture_plain_file_identity(
            active.publication_run_root, _MAPPING_FILE, label="P3-A2 active mappings")
        stream = open_plain_binary(active.publication_run_root, _MAPPING_FILE,
                                   label="P3-A2 active mappings", expected_identity=identity)
        reader = IntegerFramedStreamReader.from_open_binary(
            stream, path=_MAPPING_FILE,
            max_frame_bytes=budget.max_mapping_record_payload_bytes,
            max_record_count=budget.max_mapping_record_count,
            max_total_payload_bytes=budget.max_mapping_stream_payload_bytes)
        try:
            for raw in reader:
                yield ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(raw)
            footer = reader.finish()
        finally:
            reader.close()
        require_plain_file_identity(active.publication_run_root, _MAPPING_FILE, identity,
                                    label="P3-A2 active mappings post-read")
        if footer != active.mapping_stream.p0_footer:
            _fail("P3-A2 active mapping footer drifted")
        physical = sha256_plain_file(
            active.publication_run_root, _MAPPING_FILE,
            max_bytes=budget.max_mapping_stream_physical_bytes,
            label="P3-A2 active mapping physical")
        if physical != active.mapping_stream.physical:
            _fail("P3-A2 active mapping physical identity drifted")
    except ConversationHeldOutV4PayloadWitnessLedgerError:
        raise
    except (ConversationHeldOutV4ActiveIntakeError, IntegerCodecError,
            IntegerFramedStreamError, KRunBoundaryError, OSError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 active mapping readback failed") from exc


def _verify_mapping_inventory(
        active: ConversationHeldOutV4ActiveIntakeResult,
        *, budget: ConversationHeldOutV4PayloadWitnessBudget,
        ) -> None:
    """先完整回读 P3-A mapping，确保 teacher 检查发生在 factory 调用之前。"""
    count = 0
    for mapping in _iter_mappings(active, budget=budget):
        if mapping.record_kind == V4_ACTIVE_INTAKE_RECORD_TEACHER:
            _fail("P3-A2 active mapping contains teacher records")
        count += 1
    if count != active.counts.mapping_record_count:
        _fail("P3-A2 active mapping count drifted")


def _verify_witness(
        mapping: ConversationHeldOutV4ActiveIntakeMapping,
        witness: ConversationHeldOutV4PayloadWitness,
        *, budget: ConversationHeldOutV4PayloadWitnessBudget,
        frozen_source_manifest_identity: ProtocolKey,
        ) -> tuple[ConversationHeldOutV4PayloadWitnessReceiptRecord, int, int]:
    if not isinstance(witness, ConversationHeldOutV4PayloadWitness):
        raise TypeError("P3-A2 witness factory yielded an invalid item")
    if (witness.record_kind != mapping.record_kind
            or witness.active_record_key != mapping.record_key
            or witness.source_ref != mapping.source_ref
            or witness.leaf != mapping.leaf
            or witness.snapshot != mapping.snapshot
            or witness.split != mapping.split
            or witness.license_partition != mapping.license_partition
            or witness.consumed_input_identity != mapping.leaf.consumed_input_identity
            or witness.frozen_source_manifest_identity != frozen_source_manifest_identity
            or witness.source_uri_scalars != mapping.snapshot.official_uri_scalars):
        _fail("P3-A2 witness/mapping binding drifted")
    payload, scalars = witness.canonical_bytes_and_scalars()
    if len(payload) > budget.max_witness_content_bytes:
        _fail("P3-A2 witness content exceeds budget")
    digest = tuple(hashlib.sha256(payload).digest())
    if digest != mapping.content_sha256:
        _fail("P3-A2 witness content SHA-256 differs from mapping")
    record = ConversationHeldOutV4PayloadWitnessReceiptRecord(
        mapping.stable_key(), witness.record_kind, witness.active_record_key, witness.source_ref, witness.leaf,
        digest, len(payload), len(scalars), witness.source_span_byte_start, witness.source_span_byte_end,
        witness.source_span_scalar_start, witness.source_span_scalar_end, witness.encoding,
        witness.frozen_source_manifest_identity, witness.source_payload_identity)
    return record, len(payload), len(scalars)


def _write_receipt(
        root: KRunRoot,
        relative: Path,
        records: Iterable[ConversationHeldOutV4PayloadWitnessReceiptRecord],
        *, budget: ConversationHeldOutV4PayloadWitnessBudget,
        label: str,
        ) -> ConversationHeldOutV4PayloadWitnessReceiptStream:
    """排他写出无正文 receipt stream，并返回可回读的物理 identity。"""
    ensure_normal_relative_directory(root, relative.parent, label=f"{label} parent")
    raw = open_exclusive_binary(root, relative, label=label)
    writer = IntegerFramedStreamWriter.from_open_binary(raw, path=relative)
    total = 0
    count = 0
    try:
        for item in records:
            payload = item.integer_stream()
            size = encoded_integer_tuple_size(payload)
            if size > budget.max_receipt_record_payload_bytes:
                _fail(f"{label} record exceeds budget")
            if total + size > budget.max_receipt_payload_bytes:
                _fail(f"{label} payload exceeds budget")
            writer.append(payload)
            total += size
            count += 1
        footer = writer.seal()
    except (IntegerCodecError, IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            f"{label} write failed") from exc
    finally:
        writer.close()
    if count == 0 or footer.record_count != count or footer.total_payload_bytes != total:
        _fail(f"{label} footer/count drifted")
    physical = sha256_plain_file(root, relative,
                                 max_bytes=budget.max_receipt_physical_bytes,
                                 label=f"{label} physical")
    return ConversationHeldOutV4PayloadWitnessReceiptStream(relative, footer, physical)


def _iter_receipt_records(
        root: KRunRoot, identity: ConversationHeldOutV4PayloadWitnessReceiptStream,
        *, budget: ConversationHeldOutV4PayloadWitnessBudget, label: str,
        ) -> Iterator[ConversationHeldOutV4PayloadWitnessReceiptRecord]:
    """流式回读一个已封存的无正文 receipt，拒绝 footer 或物理身份漂移。"""
    try:
        file_identity = capture_plain_file_identity(
            root, identity.relative_path, label=label)
        stream = open_plain_binary(
            root, identity.relative_path, label=label,
            expected_identity=file_identity)
        reader = IntegerFramedStreamReader.from_open_binary(
            stream, path=identity.relative_path,
            max_frame_bytes=budget.max_receipt_record_payload_bytes,
            max_record_count=budget.max_mapping_record_count,
            max_total_payload_bytes=budget.max_receipt_payload_bytes)
        try:
            for raw in reader:
                yield ConversationHeldOutV4PayloadWitnessReceiptRecord.from_integer_stream(raw)
            footer = reader.finish()
        finally:
            reader.close()
        require_plain_file_identity(root, identity.relative_path, file_identity,
                                    label=f"{label} post-read")
        physical = sha256_plain_file(
            root, identity.relative_path,
            max_bytes=budget.max_receipt_physical_bytes,
            label=f"{label} physical")
    except ConversationHeldOutV4PayloadWitnessLedgerError:
        raise
    except (IntegerCodecError, IntegerFramedStreamError, KRunBoundaryError,
            OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            f"{label} readback failed") from exc
    if footer != identity.p0_footer or physical != identity.physical:
        _fail(f"{label} stream identity drifted")


def _stage_witness_receipt(
        value: ConversationHeldOutV4PayloadWitnessInput,
        active: ConversationHeldOutV4ActiveIntakeResult,
        ) -> tuple[ConversationHeldOutV4PayloadWitnessReceiptStream,
                   ConversationHeldOutV4PayloadWitnessCounts]:
    """把 factory 的逐项闭合写入 staging 元数据流，不在内存累积 mapping 或 receipt。"""
    try:
        factory_items = iter(value.witness_factory())
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 witness factory is not iterable") from exc
    root = value.staging_run_root
    relative = _STAGING_RECEIPT_FILE
    ensure_normal_relative_directory(root, relative.parent,
                                     label="P3-A2 staging receipt parent")
    raw = open_exclusive_binary(root, relative, label="P3-A2 staging receipt")
    writer = IntegerFramedStreamWriter.from_open_binary(raw, path=relative)
    content_bytes = 0
    content_scalars = 0
    mapping_count = 0
    receipt_payload_bytes = 0
    prior_key: tuple[int, ...] | None = None
    try:
        for mapping in _iter_mappings(active, budget=value.budget):
            if mapping.record_kind == V4_ACTIVE_INTAKE_RECORD_TEACHER:
                _fail("P3-A2 active mapping contains teacher records")
            try:
                witness = next(factory_items)
            except StopIteration:
                _fail("P3-A2 witness factory is missing an active mapping")
            if not isinstance(witness, ConversationHeldOutV4PayloadWitness):
                raise TypeError("P3-A2 witness factory yielded an invalid item")
            key = witness.canonical_sort_key()
            if prior_key is not None and key <= prior_key:
                _fail("P3-A2 witness factory order is not canonical")
            if key != (mapping.record_kind, *mapping.record_key.components):
                _fail("P3-A2 witness factory order does not match sealed mapping")
            prior_key = key
            record, byte_count, scalar_count = _verify_witness(
                mapping, witness, budget=value.budget,
                frozen_source_manifest_identity=value.frozen_source_manifest_identity)
            if content_bytes + byte_count > value.budget.max_total_content_bytes:
                _fail("P3-A2 total witness content exceeds budget")
            payload = record.integer_stream()
            payload_size = encoded_integer_tuple_size(payload)
            if payload_size > value.budget.max_receipt_record_payload_bytes:
                _fail("P3-A2 staging receipt record exceeds budget")
            if (receipt_payload_bytes + payload_size
                    > value.budget.max_receipt_payload_bytes):
                _fail("P3-A2 staging receipt payload exceeds budget")
            writer.append(payload)
            receipt_payload_bytes += payload_size
            content_bytes += byte_count
            content_scalars += scalar_count
            mapping_count += 1
        try:
            next(factory_items)
        except StopIteration:
            pass
        else:
            _fail("P3-A2 witness factory has an extra witness")
        if mapping_count != active.counts.mapping_record_count:
            _fail("P3-A2 staged mapping count drifted")
        footer = writer.seal()
    except ConversationHeldOutV4PayloadWitnessLedgerError:
        raise
    except (IntegerCodecError, IntegerFramedStreamError, KRunBoundaryError,
            OSError, TypeError, ValueError, UnicodeError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 witness staging failed") from exc
    finally:
        writer.close()
    counts = ConversationHeldOutV4PayloadWitnessCounts(
        mapping_count, mapping_count, content_bytes, content_scalars)
    if (footer.record_count != counts.witness_count
            or footer.total_payload_bytes != receipt_payload_bytes):
        _fail("P3-A2 staging receipt footer/count drifted")
    physical = sha256_plain_file(
        root, relative, max_bytes=value.budget.max_receipt_physical_bytes,
        label="P3-A2 staging receipt physical")
    return (ConversationHeldOutV4PayloadWitnessReceiptStream(relative, footer, physical),
            counts)


def _manifest_integer_stream(value: ConversationHeldOutV4PayloadWitnessResult) -> tuple[int, ...]:
    result = [V4_PAYLOAD_WITNESS_MANIFEST_SCHEMA]
    for item in (value.receipt.integer_stream(), value.receipt_stream.integer_stream()):
        pack_key(result, item)
    return tuple(result)


def materialize_v4_payload_witness_ledger(
        value: ConversationHeldOutV4PayloadWitnessInput,
        ) -> ConversationHeldOutV4PayloadWitnessResult:
    """运行 P3-A2，不持久化 raw payload，也不进入 P3-C1。"""
    if not isinstance(value, ConversationHeldOutV4PayloadWitnessInput):
        raise TypeError("P3-A2 materialize requires ConversationHeldOutV4PayloadWitnessInput")
    try:
        active = revalidate_v4_active_intake(value.active_intake)
        readback = revalidate_v4_active_roster_bidirectional_readback(value.readback)
    except (ConversationHeldOutV4ActiveIntakeError,
            ConversationHeldOutV4ActiveRosterReadbackError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 upstream P3-A/P3-B readback failed") from exc
    if readback.receipt.active_intake_stable_key != active.stable_key():
        _fail("P3-A2 P3-B receipt is not bound to active intake")
    # teacher payload 被禁止；此检查必须先于 factory 调用。
    if active.counts.teacher_record_count != 0:
        _fail("P3-A2 active intake contains teacher records")
    try:
        status = _validate_roots(value)
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 verification root boundary failed") from exc

    # 首先完成一遍无 payload inventory 回读。第二遍才调用 factory，并把每条已验证
    # receipt 直接写到 staging，避免全量 mapping 或 receipt 常驻 Python 内存。
    _verify_mapping_inventory(active, budget=value.budget)
    staged_receipt, counts = _stage_witness_receipt(value, active)
    receipt = ConversationHeldOutV4PayloadWitnessReceipt(
        active.stable_key(), readback.stable_key(), value.frozen_source_manifest_identity,
        value.code_identity, value.budget, counts, status)
    try:
        work_receipt = _write_receipt(
            value.work_run_root, _WORK_RECEIPT_FILE,
            _iter_receipt_records(
                value.staging_run_root, staged_receipt,
                budget=value.budget, label="P3-A2 staging receipt"),
            budget=value.budget, label="P3-A2 work receipt")
        receipt_stream = _write_receipt(
            value.publication_run_root, _RECEIPT_FILE,
            _iter_receipt_records(
                value.work_run_root, work_receipt,
                budget=value.budget, label="P3-A2 work receipt"),
            budget=value.budget, label="P3-A2 receipt")
        provisional = ConversationHeldOutV4PayloadWitnessResult(
            value.publication_run_root, receipt, receipt_stream, (0,) * _SHA256_SIZE)
        manifest_payload = encode_integer_tuple(_manifest_integer_stream(provisional))
        if len(manifest_payload) > value.budget.max_manifest_bytes:
            _fail("P3-A2 manifest exceeds budget")
        publish_manifest_last(value.publication_run_root, _MANIFEST_FILE, manifest_payload,
                              frozenset({_RECEIPT_FILE}), label="P3-A2 publication")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 publication boundary failed") from exc
    result = ConversationHeldOutV4PayloadWitnessResult(
        value.publication_run_root, receipt, receipt_stream,
        tuple(hashlib.sha256(manifest_payload).digest()))
    return revalidate_v4_payload_witness_ledger(result)


def load_v4_payload_witness_publication(
        value: ConversationHeldOutV4PayloadWitnessResult,
        *,
        max_record_count: int | None = None,
        max_total_payload_bytes: int | None = None,
        max_physical_bytes: int | None = None,
        ) -> ConversationHeldOutV4PayloadWitnessPublication:
    """回读公开 P3-A2 receipt，并允许调用方把实际扫描限制在更小预算。"""
    if not isinstance(value, ConversationHeldOutV4PayloadWitnessResult):
        raise TypeError("P3-A2 publication loader requires ConversationHeldOutV4PayloadWitnessResult")
    if max_record_count is None:
        reader_max_record_count = value.receipt.counts.witness_count
    else:
        if type(max_record_count) is not int or max_record_count <= 0:
            raise ValueError("P3-A2 caller max_record_count 必须为正整数")
        reader_max_record_count = max_record_count
        if (value.receipt.counts.witness_count > reader_max_record_count
                or value.receipt_stream.p0_footer.record_count
                > reader_max_record_count):
            _fail("P3-A2 receipt exceeds caller record budget")
    if max_total_payload_bytes is None:
        reader_max_total_payload_bytes = value.receipt.budget.max_receipt_payload_bytes
    else:
        if type(max_total_payload_bytes) is not int or max_total_payload_bytes <= 0:
            raise ValueError("P3-A2 caller max_total_payload_bytes 必须为正整数")
        reader_max_total_payload_bytes = min(
            value.receipt.budget.max_receipt_payload_bytes,
            max_total_payload_bytes,
        )
        if (value.receipt_stream.p0_footer.total_payload_bytes
                > reader_max_total_payload_bytes):
            _fail("P3-A2 receipt exceeds caller payload budget")
    if max_physical_bytes is None:
        reader_max_physical_bytes = value.receipt.budget.max_receipt_physical_bytes
    else:
        if type(max_physical_bytes) is not int or max_physical_bytes <= 0:
            raise ValueError("P3-A2 caller max_physical_bytes 必须为正整数")
        reader_max_physical_bytes = min(
            value.receipt.budget.max_receipt_physical_bytes,
            max_physical_bytes,
        )
        if value.receipt_stream.physical.byte_count > reader_max_physical_bytes:
            _fail("P3-A2 receipt exceeds caller physical budget")
    try:
        require_exact_file_closure(value.publication_run_root,
                                   frozenset({_RECEIPT_FILE, _MANIFEST_FILE}),
                                   label="P3-A2 publication")
        identity = capture_plain_file_identity(value.publication_run_root, _RECEIPT_FILE,
                                               label="P3-A2 receipt")
        if identity.byte_count > reader_max_physical_bytes:
            _fail("P3-A2 receipt actual file exceeds caller physical budget")
        stream = open_plain_binary(value.publication_run_root, _RECEIPT_FILE,
                                   label="P3-A2 receipt", expected_identity=identity)
        reader = IntegerFramedStreamReader.from_open_binary(
            stream, path=_RECEIPT_FILE,
            max_frame_bytes=value.receipt.budget.max_receipt_record_payload_bytes,
            max_record_count=reader_max_record_count,
            max_total_payload_bytes=reader_max_total_payload_bytes)
        try:
            records = tuple(ConversationHeldOutV4PayloadWitnessReceiptRecord.from_integer_stream(raw)
                            for raw in reader)
            footer = reader.finish()
        finally:
            reader.close()
        require_plain_file_identity(value.publication_run_root, _RECEIPT_FILE, identity,
                                    label="P3-A2 receipt post-read")
        physical = sha256_plain_file(value.publication_run_root, _RECEIPT_FILE,
                                     max_bytes=reader_max_physical_bytes,
                                     label="P3-A2 receipt physical")
    except (KRunBoundaryError, IntegerCodecError, IntegerFramedStreamError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 receipt publication readback failed") from exc
    if (len(records) != value.receipt.counts.witness_count
            or footer != value.receipt_stream.p0_footer
            or physical != value.receipt_stream.physical):
        _fail("P3-A2 receipt stream identity drifted")
    try:
        manifest_identity = capture_plain_file_identity(value.publication_run_root, _MANIFEST_FILE,
                                                        label="P3-A2 manifest")
        with open_plain_binary(value.publication_run_root, _MANIFEST_FILE,
                               label="P3-A2 manifest", expected_identity=manifest_identity) as stream:
            manifest_payload = stream.read(value.receipt.budget.max_manifest_bytes + 1)
        require_plain_file_identity(value.publication_run_root, _MANIFEST_FILE, manifest_identity,
                                    label="P3-A2 manifest post-read")
    except (KRunBoundaryError, OSError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 manifest readback failed") from exc
    if len(manifest_payload) > value.receipt.budget.max_manifest_bytes:
        _fail("P3-A2 manifest exceeds budget")
    try:
        decoded = decode_integer_tuple(manifest_payload)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4PayloadWitnessLedgerError(
            "P3-A2 manifest is corrupt") from exc
    if (encode_integer_tuple(decoded) != manifest_payload
            or decoded != _manifest_integer_stream(value)
            or tuple(hashlib.sha256(manifest_payload).digest()) != value.manifest_sha256):
        _fail("P3-A2 manifest identity drifted")
    return ConversationHeldOutV4PayloadWitnessPublication(value, records)


def revalidate_v4_payload_witness_ledger(
        value: ConversationHeldOutV4PayloadWitnessResult,
        ) -> ConversationHeldOutV4PayloadWitnessResult:
    """只读回验公开 P3-A2 receipt/manifest 闭合，不访问 payload。"""
    return load_v4_payload_witness_publication(value).result


__all__ = [
    "ConversationHeldOutV4PayloadWitness",
    "ConversationHeldOutV4PayloadWitnessBudget",
    "ConversationHeldOutV4PayloadWitnessInput",
    "ConversationHeldOutV4PayloadWitnessLedgerError",
    "ConversationHeldOutV4PayloadWitnessPublication",
    "ConversationHeldOutV4PayloadWitnessReceipt",
    "ConversationHeldOutV4PayloadWitnessReceiptRecord",
    "ConversationHeldOutV4PayloadWitnessResult",
    "V4_PAYLOAD_WITNESS_ENCODING_UTF8_BYTES",
    "V4_PAYLOAD_WITNESS_ENCODING_UTF8_SCALARS",
    "V4_PAYLOAD_WITNESS_STATUS_COVERAGE_NE",
    "V4_PAYLOAD_WITNESS_STATUS_TEST_ONLY",
    "load_v4_payload_witness_publication",
    "materialize_v4_payload_witness_ledger",
    "revalidate_v4_payload_witness_ledger",
]
