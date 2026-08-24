"""DLG-05 v4 P3-C1a 的 sealed annotation witness manifest。

P3-C1a 在 P3-C1b typed task factory 前冻结每条 test-authored annotation witness。
公开 P0 只包含域分离摘要，不读取 payload、不构造 capsule、不调用 runtime/writer/trainer。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Iterable

from pure_integer_ai.experiments.conversation_heldout_v4_payload_witness_ledger import (
    ConversationHeldOutV4PayloadWitnessLedgerError,
    ConversationHeldOutV4PayloadWitnessResult,
    load_v4_payload_witness_publication,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation import (
    ConversationHeldOutV4RuntimeTaskAnnotationDeclaration,
    V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError, IntegerFramedStreamError, IntegerFramedStreamFooter,
    IntegerFramedStreamReader, IntegerFramedStreamWriter, IntegerStreamReader,
    decode_integer_tuple, encode_integer_tuple, encoded_integer_tuple_size, pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunBoundaryError, KRunFileDigest, KRunRoot, capture_plain_file_identity,
    ensure_normal_relative_directory, open_exclusive_binary, open_plain_binary,
    publish_manifest_last, require_disjoint_run_roots, require_exact_file_closure,
    require_fresh_empty_run_root, require_plain_file_identity, sha256_plain_file,
)


V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_BUDGET_SCHEMA = 2
V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_WITNESS_SCHEMA = 1
V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_RECORD_SCHEMA = 1
V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_RECEIPT_SCHEMA = 1
V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_RESULT_SCHEMA = 1
V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_MANIFEST_SCHEMA = 1
V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_TEST_ONLY = "P3_C1A_ANNOTATION_PROVENANCE_TEST_ONLY"
V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_COVERAGE_NE = "ACTIVE_PROVENANCE_COVERAGE_NE"

_STAGING_RECEIPT_FILE = Path("candidate") / "receipt.pifrs"
_WORK_RECEIPT_FILE = Path("verified") / "receipt.pifrs"
_RECEIPT_FILE = Path("annotation") / "receipt.pifrs"
_MANIFEST_FILE = Path("manifest.pii")
_SHA256_SIZE = hashlib.sha256().digest_size
_DOMAIN_PAYLOAD_WITNESS = b"dlg05.v4.p3c1a.payload-witness.v1\x00"
_DOMAIN_CASE = b"dlg05.v4.p3c1a.case-key.v1\x00"
_DOMAIN_TURN = b"dlg05.v4.p3c1a.turn-key.v1\x00"
_DOMAIN_WITNESS_RECORD = b"dlg05.v4.p3c1a.witness-record.v1\x00"
_DOMAIN_TYPED_TASK = b"dlg05.v4.p3c1a.typed-task.v1\x00"
_DOMAIN_ANNOTATION_WITNESS = b"dlg05.v4.p3c1a.annotation-witness.v1\x00"
_DOMAIN_DECLARATION = b"dlg05.v4.p3c1a.annotation-declaration.v1\x00"
_DOMAIN_SOURCE = b"dlg05.v4.p3c1a.annotation-source.v1\x00"
_DOMAIN_REVISION = b"dlg05.v4.p3c1a.annotation-revision.v1\x00"
_DOMAIN_MANIFEST = b"dlg05.v4.p3c1a.annotation-manifest.v1\x00"
_DOMAIN_CONSTRUCTING_CODE = b"dlg05.v4.p3c1a.constructing-code.v1\x00"
_DOMAIN_TRANSFORM_CODE = b"dlg05.v4.p3c1a.annotation-transform-code.v1\x00"
_DOMAIN_DOMAIN = b"dlg05.v4.p3c1a.applicable-domain.v1\x00"


class ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(RuntimeError):
    """P3-C1a annotation witness、receipt 或 K-run 边界没有闭合。"""


def _fail(message: str) -> None:
    """统一产生 P3-C1a fail-closed 错误。"""
    raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(message)


def _positive(value: int, *, label: str) -> int:
    """要求严格正整数。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive strict int")
    return value


def _digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """要求完整 SHA-256 byte tuple。"""
    if (not isinstance(value, tuple) or len(value) != _SHA256_SIZE
            or any(type(item) is not int or item < 0 or item > 255 for item in value)):
        raise ValueError(f"{label} must be a SHA-256 byte tuple")
    return value


def _text(value: str, *, label: str) -> tuple[int, ...]:
    """编码固定公开文本。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty str")
    result = tuple(ord(item) for item in value)
    if any(0xD800 <= item <= 0xDFFF for item in result):
        raise ValueError(f"{label} contains invalid Unicode scalars")
    return result


def _read_text(value: tuple[int, ...], *, label: str) -> str:
    """解码固定公开文本。"""
    strict_integer_tuple(value, label=label)
    try:
        result = "".join(chr(item) for item in value)
    except ValueError as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            f"{label} is not Unicode") from exc
    if not result or any(0xD800 <= item <= 0xDFFF for item in value):
        _fail(f"{label} is invalid")
    return result


def _stage(value: str) -> str:
    """限制逻辑 stage 为无路径 ASCII token。"""
    if (not isinstance(value, str) or not value or len(value) > 48
            or not value[0].isascii() or not value[0].isalnum()
            or any(not item.isascii() or not (item.isalnum() or item in {"-", "_", "."})
                   for item in value)):
        raise ValueError("P3-C1a logical_stage_name is invalid")
    return value


def _hash(value: tuple[int, ...], *, domain: bytes, label: str) -> tuple[int, ...]:
    """计算 domain-separated 完整身份摘要。"""
    strict_integer_tuple(value, label=label)
    if not isinstance(domain, bytes) or not domain:
        raise TypeError("P3-C1a identity digest domain is invalid")
    return tuple(hashlib.sha256(domain + encode_integer_tuple(value)).digest())


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget:
    """限制完整 P3-A2 输入账本、annotation inventory 与 P0/manifest 资源。"""

    max_upstream_payload_witness_record_count: int
    max_annotation_witness_count: int
    max_receipt_record_payload_bytes: int
    max_receipt_payload_bytes: int
    max_receipt_physical_bytes: int
    max_manifest_bytes: int

    def __post_init__(self) -> None:
        """验证所有资源上限并闭合单 record/总量关系。"""
        for label, item in zip((
                "max_upstream_payload_witness_record_count",
                "max_annotation_witness_count", "max_receipt_record_payload_bytes",
                "max_receipt_payload_bytes", "max_receipt_physical_bytes",
                "max_manifest_bytes"), self.integer_stream()[1:]):
            _positive(item, label=f"P3-C1a {label}")
        if self.max_receipt_record_payload_bytes > self.max_receipt_payload_bytes:
            raise ValueError("P3-C1a receipt record budget exceeds receipt budget")

    def integer_stream(self) -> tuple[int, ...]:
        """返回版本化、规范序预算。"""
        return (
            V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_BUDGET_SCHEMA,
            self.max_upstream_payload_witness_record_count,
            self.max_annotation_witness_count, self.max_receipt_record_payload_bytes,
            self.max_receipt_payload_bytes, self.max_receipt_physical_bytes,
            self.max_manifest_bytes,
        )


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationWitness:
    """独立 manifest 中一条仅内存 annotation witness。"""

    case_key: ProtocolKey
    turn_key: ProtocolKey
    ordinal: int
    witness_record_identity: tuple[int, ...]
    typed_task_identity: tuple[int, ...]
    declaration: ConversationHeldOutV4RuntimeTaskAnnotationDeclaration

    def __post_init__(self) -> None:
        """要求每条 annotation 指向当前 P3-A2 witness 和完整 typed task。"""
        if not isinstance(self.case_key, ProtocolKey) or not isinstance(self.turn_key, ProtocolKey):
            raise TypeError("P3-C1a annotation case/turn key type is invalid")
        _positive(self.ordinal, label="P3-C1a annotation ordinal")
        strict_integer_tuple(self.witness_record_identity,
                             label="P3-C1a annotation witness record identity")
        strict_integer_tuple(self.typed_task_identity,
                             label="P3-C1a annotation typed task identity")
        if not isinstance(self.declaration, ConversationHeldOutV4RuntimeTaskAnnotationDeclaration):
            raise TypeError("P3-C1a annotation declaration type is invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 witness identity，公开层必须只发布摘要。"""
        result = [V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_WITNESS_SCHEMA]
        for item in (
                self.case_key.components, self.turn_key.components, (self.ordinal,),
                self.witness_record_identity, self.typed_task_identity,
                self.declaration.stable_key()):
            pack_key(result, item)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord:
    """无正文、可回读的 annotation witness receipt record。"""

    case_key_identity_sha256: tuple[int, ...]
    turn_key_identity_sha256: tuple[int, ...]
    ordinal: int
    witness_record_identity_sha256: tuple[int, ...]
    typed_task_identity_sha256: tuple[int, ...]
    annotation_witness_identity_sha256: tuple[int, ...]
    declaration_identity_sha256: tuple[int, ...]
    annotation_origin_status: str
    annotation_source_identity_sha256: tuple[int, ...]
    annotation_revision_identity_sha256: tuple[int, ...]
    frozen_annotation_manifest_identity_sha256: tuple[int, ...]
    constructing_code_identity_sha256: tuple[int, ...]
    annotation_transform_code_identity_sha256: tuple[int, ...]
    applicable_domain_identity_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """禁止完整 key/task/witness 或非 test origin 进入 receipt。"""
        for label, item in (
                ("case key", self.case_key_identity_sha256),
                ("turn key", self.turn_key_identity_sha256),
                ("witness record", self.witness_record_identity_sha256),
                ("typed task", self.typed_task_identity_sha256),
                ("annotation witness", self.annotation_witness_identity_sha256),
                ("declaration", self.declaration_identity_sha256),
                ("annotation source", self.annotation_source_identity_sha256),
                ("annotation revision", self.annotation_revision_identity_sha256),
                ("annotation manifest", self.frozen_annotation_manifest_identity_sha256),
                ("constructing code", self.constructing_code_identity_sha256),
                ("annotation transform code", self.annotation_transform_code_identity_sha256),
                ("applicable domain", self.applicable_domain_identity_sha256)):
            _digest(item, label=f"P3-C1a receipt {label} identity")
        _positive(self.ordinal, label="P3-C1a receipt ordinal")
        if self.annotation_origin_status != V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED:
            raise ValueError("P3-C1a receipt annotation origin is invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """编码 public P0 record，绝不写完整 ProtocolKey。"""
        result = [V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_RECORD_SCHEMA]
        for item in (
                self.case_key_identity_sha256, self.turn_key_identity_sha256,
                (self.ordinal,), self.witness_record_identity_sha256,
                self.typed_task_identity_sha256, self.annotation_witness_identity_sha256,
                self.declaration_identity_sha256,
                _text(self.annotation_origin_status, label="P3-C1a receipt annotation origin"),
                self.annotation_source_identity_sha256,
                self.annotation_revision_identity_sha256,
                self.frozen_annotation_manifest_identity_sha256,
                self.constructing_code_identity_sha256,
                self.annotation_transform_code_identity_sha256,
                self.applicable_domain_identity_sha256):
            pack_key(result, item)
        return tuple(result)

    @classmethod
    def from_integer_stream(
            cls, value: tuple[int, ...],
            ) -> "ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord":
        """回读 record 并拒绝非 canonical 或附加字段。"""
        try:
            reader = IntegerStreamReader(value)
            if reader.read_positive(label="P3-C1a receipt schema") != V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_RECORD_SCHEMA:
                _fail("P3-C1a receipt schema is incompatible")
            case = reader.read_key(label="P3-C1a receipt case")
            turn = reader.read_key(label="P3-C1a receipt turn")
            ordinal = reader.read_key(label="P3-C1a receipt ordinal")
            if len(ordinal) != 1:
                _fail("P3-C1a receipt ordinal is invalid")
            values = tuple(reader.read_key(label=f"P3-C1a receipt field {index}")
                           for index in range(11))
            result = cls(
                case, turn, ordinal[0], values[0], values[1], values[2], values[3],
                _read_text(values[4], label="P3-C1a receipt annotation origin"),
                *values[5:])
            reader.finish()
        except (IntegerCodecError, TypeError, ValueError, IndexError) as exc:
            raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
                "P3-C1a receipt record is corrupt") from exc
        if result.integer_stream() != value:
            _fail("P3-C1a receipt record is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptStream:
    """P3-C1a P0 footer 与物理文件身份。"""

    relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """限制 stream 指向 root 内的单一普通相对文件。"""
        if (not isinstance(self.relative_path, Path) or self.relative_path.is_absolute()
                or self.relative_path.drive or self.relative_path.root
                or not self.relative_path.parts or ".." in self.relative_path.parts
                or not isinstance(self.p0_footer, IntegerFramedStreamFooter)
                or not isinstance(self.physical, KRunFileDigest)
                or self.physical.byte_count <= 0):
            raise ValueError("P3-C1a receipt stream identity is invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """返回无绝对路径的 footer/physical identity。"""
        result: list[int] = []
        for item in (
                _text(self.relative_path.as_posix(), label="P3-C1a receipt path"),
                self.p0_footer.integer_tuple(),
                (self.physical.byte_count, *self.physical.sha256)):
            pack_key(result, item)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceipt:
    """绑定 P3-A2 public identity、预算、inventory 和状态。"""

    payload_witness_identity_sha256: tuple[int, ...]
    budget: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget
    annotation_witness_count: int
    status: str

    def __post_init__(self) -> None:
        """拒绝空 inventory、超预算或来源覆盖 PASS。"""
        _digest(self.payload_witness_identity_sha256, label="P3-C1a payload witness identity")
        _positive(self.annotation_witness_count, label="P3-C1a annotation witness count")
        if (not isinstance(self.budget, ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget)
                or self.annotation_witness_count > self.budget.max_annotation_witness_count
                or self.status not in {
                    V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_TEST_ONLY,
                    V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_COVERAGE_NE}):
            raise ValueError("P3-C1a receipt fields are invalid")

    def integer_stream(self) -> tuple[int, ...]:
        """编码无正文 receipt 元数据。"""
        result = [V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_RECEIPT_SCHEMA]
        for item in (
                self.payload_witness_identity_sha256, self.budget.integer_stream(),
                (self.annotation_witness_count,), _text(self.status, label="P3-C1a receipt status")):
            pack_key(result, item)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput:
    """P3-C1a 输入：P3-A2、独立 annotation factory 与三个 fresh K roots。"""

    payload_witness: ConversationHeldOutV4PayloadWitnessResult
    annotation_factory: Callable[[], Iterable[ConversationHeldOutV4RuntimeTaskAnnotationWitness]]
    staging_run_root: KRunRoot
    work_run_root: KRunRoot
    publication_run_root: KRunRoot
    budget: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget
    logical_stage_name: str

    def __post_init__(self) -> None:
        """拒绝裸 key、无 P3-A2 或不受预算的 annotation factory。"""
        if (not isinstance(self.payload_witness, ConversationHeldOutV4PayloadWitnessResult)
                or not callable(self.annotation_factory)
                or not isinstance(self.budget, ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget)):
            raise TypeError("P3-C1a input types are invalid")
        for root in (self.staging_run_root, self.work_run_root, self.publication_run_root):
            if not isinstance(root, KRunRoot):
                raise TypeError("P3-C1a run root type is invalid")
        _stage(self.logical_stage_name)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult:
    """封存的 annotation witness manifest；完整 witnesses 仅进程内。"""

    payload_witness: ConversationHeldOutV4PayloadWitnessResult
    publication_run_root: KRunRoot
    receipt: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceipt
    receipt_stream: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptStream
    manifest_sha256: tuple[int, ...]
    witnesses: tuple[ConversationHeldOutV4RuntimeTaskAnnotationWitness, ...]

    def __post_init__(self) -> None:
        """保持 result 类型、inventory、路径和 transport 状态闭合。"""
        if (not isinstance(self.payload_witness, ConversationHeldOutV4PayloadWitnessResult)
                or not isinstance(self.publication_run_root, KRunRoot)
                or not isinstance(self.receipt, ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceipt)
                or not isinstance(self.receipt_stream,
                                  ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptStream)
                or not isinstance(self.witnesses, tuple)
                or any(not isinstance(item, ConversationHeldOutV4RuntimeTaskAnnotationWitness)
                       for item in self.witnesses)):
            raise TypeError("P3-C1a result types are invalid")
        if (self.receipt_stream.relative_path != _RECEIPT_FILE
                or len(self.witnesses) != self.receipt.annotation_witness_count):
            raise ValueError("P3-C1a result witness inventory drifted")
        _digest(self.manifest_sha256, label="P3-C1a manifest hash")
        expected = (V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_TEST_ONLY
                    if self.publication_run_root.test_transport
                    else V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_COVERAGE_NE)
        if self.receipt.status != expected:
            raise ValueError("P3-C1a result transport status drifted")

    def stable_key(self) -> tuple[int, ...]:
        """返回只由公开 receipt/manifest 构成的稳定身份。"""
        result = [V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_RESULT_SCHEMA]
        for item in (self.receipt.integer_stream(), self.receipt_stream.integer_stream(),
                     self.manifest_sha256):
            pack_key(result, item)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationProvenancePublication:
    """P3-C1a 受限 loader 结果；P0 records 无正文，result 是进程内 capability。"""

    result: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult
    records: tuple[ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord, ...]

    def __post_init__(self) -> None:
        """限定 loader inventory 类型和数量。"""
        if (not isinstance(self.result, ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult)
                or not isinstance(self.records, tuple)
                or any(not isinstance(item,
                                      ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord)
                       for item in self.records)
                or len(self.records) != self.result.receipt.annotation_witness_count):
            raise TypeError("P3-C1a publication types are invalid")


def _record_for(witness: ConversationHeldOutV4RuntimeTaskAnnotationWitness
                ) -> ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord:
    """将完整 witness 变为角色分离的 public digest record。"""
    if not isinstance(witness, ConversationHeldOutV4RuntimeTaskAnnotationWitness):
        raise TypeError("P3-C1a annotation witness type is invalid")
    declaration = witness.declaration
    return ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord(
        _hash(witness.case_key.components, domain=_DOMAIN_CASE, label="P3-C1a case key"),
        _hash(witness.turn_key.components, domain=_DOMAIN_TURN, label="P3-C1a turn key"),
        witness.ordinal,
        _hash(witness.witness_record_identity, domain=_DOMAIN_WITNESS_RECORD,
              label="P3-C1a witness record"),
        _hash(witness.typed_task_identity, domain=_DOMAIN_TYPED_TASK,
              label="P3-C1a typed task"),
        _hash(witness.stable_key(), domain=_DOMAIN_ANNOTATION_WITNESS,
              label="P3-C1a annotation witness"),
        _hash(declaration.stable_key(), domain=_DOMAIN_DECLARATION,
              label="P3-C1a declaration"),
        declaration.annotation_origin_status,
        _hash(declaration.annotation_source_identity.components, domain=_DOMAIN_SOURCE,
              label="P3-C1a annotation source"),
        _hash(declaration.annotation_revision_identity.components, domain=_DOMAIN_REVISION,
              label="P3-C1a annotation revision"),
        _hash(declaration.frozen_annotation_manifest_identity.components, domain=_DOMAIN_MANIFEST,
              label="P3-C1a annotation manifest"),
        _hash(declaration.constructing_code_identity.components, domain=_DOMAIN_CONSTRUCTING_CODE,
              label="P3-C1a constructing code"),
        _hash(declaration.annotation_transform_code_identity.components,
              domain=_DOMAIN_TRANSFORM_CODE, label="P3-C1a transform code"),
        _hash(declaration.applicable_domain_identity.components, domain=_DOMAIN_DOMAIN,
              label="P3-C1a applicable domain"),
    )


def _require_binding(record: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord | object,
                     witness: ConversationHeldOutV4RuntimeTaskAnnotationWitness) -> None:
    """要求公开 record 精确对应仍在内存的完整 annotation witness。"""
    if not isinstance(record, ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord):
        raise TypeError("P3-C1a receipt record type is invalid")
    if record != _record_for(witness):
        _fail("P3-C1a annotation witness no longer matches published receipt")


def annotation_witness_public_identity_sha256(
        witness: ConversationHeldOutV4RuntimeTaskAnnotationWitness,
        ) -> tuple[int, ...]:
    """返回已封存 annotation witness 的公开身份摘要，不暴露其完整 stable key。"""
    return _record_for(witness).annotation_witness_identity_sha256


def require_v4_runtime_task_annotation_witness_binding(
        witness: ConversationHeldOutV4RuntimeTaskAnnotationWitness,
        *,
        case_key: ProtocolKey,
        turn_key: ProtocolKey,
        ordinal: int,
        witness_record_identity: tuple[int, ...],
        typed_task_identity: tuple[int, ...],
        annotation_origin_status: str,
        annotation_source_identity: ProtocolKey,
        annotation_revision_identity: ProtocolKey,
        frozen_annotation_manifest_identity: ProtocolKey,
        constructing_code_identity: ProtocolKey,
        annotation_transform_code_identity: ProtocolKey,
        applicable_domain_identity: ProtocolKey,
        ) -> None:
    """要求 C1b turn 与预封存 C1a witness 的来源和任务身份逐字段一致。"""
    if not isinstance(witness, ConversationHeldOutV4RuntimeTaskAnnotationWitness):
        raise TypeError("P3-C1a annotation witness type is invalid")
    declaration = witness.declaration
    if (witness.case_key != case_key
            or witness.turn_key != turn_key
            or witness.ordinal != ordinal
            or witness.witness_record_identity != witness_record_identity
            or witness.typed_task_identity != typed_task_identity
            or declaration.annotation_origin_status != annotation_origin_status
            or declaration.annotation_source_identity != annotation_source_identity
            or declaration.annotation_revision_identity != annotation_revision_identity
            or declaration.frozen_annotation_manifest_identity
            != frozen_annotation_manifest_identity
            or declaration.constructing_code_identity != constructing_code_identity
            or declaration.annotation_transform_code_identity
            != annotation_transform_code_identity
            or declaration.applicable_domain_identity != applicable_domain_identity):
        _fail("P3-C1a annotation witness/task binding drifted")


def _roots(value: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput) -> str:
    """验证 P3-A2 输入和三个 output root 的 transport/隔离/fresh 条件。"""
    outputs = (value.staging_run_root, value.work_run_root, value.publication_run_root)
    if len({root.test_transport for root in (*outputs, value.payload_witness.publication_run_root)}) != 1:
        _fail("P3-C1a roots must use one transport mode")
    for index, left in enumerate(outputs):
        for right in outputs[index + 1:]:
            require_disjoint_run_roots(left, right, label="P3-C1a output roots")
    for output in outputs:
        require_disjoint_run_roots(output, value.payload_witness.publication_run_root,
                                   label="P3-C1a input/output roots")
        require_fresh_empty_run_root(output, label="P3-C1a fresh verification root")
    return (V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_TEST_ONLY
            if value.publication_run_root.test_transport
            else V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_COVERAGE_NE)


def _write(root: KRunRoot, relative: Path,
           records: Iterable[ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord], *,
           budget: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget,
           label: str) -> ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptStream:
    """排他流式写 public P0 records。"""
    ensure_normal_relative_directory(root, relative.parent, label=f"{label} parent")
    raw = open_exclusive_binary(root, relative, label=label)
    writer = IntegerFramedStreamWriter.from_open_binary(raw, path=relative)
    total = count = 0
    try:
        for record in records:
            payload = record.integer_stream()
            size = encoded_integer_tuple_size(payload)
            if size > budget.max_receipt_record_payload_bytes:
                _fail(f"{label} record exceeds budget")
            if total + size > budget.max_receipt_payload_bytes:
                _fail(f"{label} payload exceeds budget")
            if count >= budget.max_annotation_witness_count:
                _fail(f"{label} annotation witness count exceeds budget")
            writer.append(payload)
            total += size
            count += 1
        footer = writer.seal()
    except (IntegerCodecError, IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            f"{label} write failed") from exc
    finally:
        writer.close()
    if count == 0 or footer.record_count != count or footer.total_payload_bytes != total:
        _fail(f"{label} footer/count drifted")
    return ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptStream(
        relative, footer, sha256_plain_file(root, relative,
                                            max_bytes=budget.max_receipt_physical_bytes,
                                            label=f"{label} physical"))


def _read(root: KRunRoot, identity: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptStream,
          *, budget: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget,
          max_count: int, label: str
          ) -> tuple[ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord, ...]:
    """完整验证 footer/physical identity 后才返回所有 P0 records。"""
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
                ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord
                .from_integer_stream(raw) for raw in reader)
            footer = reader.finish()
        finally:
            reader.close()
        require_plain_file_identity(root, identity.relative_path, file_identity,
                                    label=f"{label} post-read")
        physical = sha256_plain_file(root, identity.relative_path,
                                     max_bytes=budget.max_receipt_physical_bytes,
                                     label=f"{label} physical")
    except ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError:
        raise
    except (IntegerCodecError, IntegerFramedStreamError, KRunBoundaryError,
            OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            f"{label} readback failed") from exc
    if footer != identity.p0_footer or physical != identity.physical:
        _fail(f"{label} stream identity drifted")
    if not records:
        _fail(f"{label} record count drifted")
    return records


def _manifest(value: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult) -> tuple[int, ...]:
    """生成只绑定公开 receipt 的 manifest。"""
    result = [V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_MANIFEST_SCHEMA]
    for item in (value.receipt.integer_stream(), value.receipt_stream.integer_stream()):
        pack_key(result, item)
    return tuple(result)


def _collect(value: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput,
             *, payload_ids: frozenset[tuple[int, ...]]
             ) -> tuple[ConversationHeldOutV4RuntimeTaskAnnotationWitness, ...]:
    """在任何输出写入前收集并验证独立 annotation inventory。"""
    try:
        items = iter(value.annotation_factory())
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            "P3-C1a annotation factory is not iterable") from exc
    result: list[ConversationHeldOutV4RuntimeTaskAnnotationWitness] = []
    keys: set[tuple[ProtocolKey, ProtocolKey]] = set()
    ordinal = 1
    try:
        for witness in items:
            if len(result) >= value.budget.max_annotation_witness_count:
                _fail("P3-C1a annotation witness count exceeds budget")
            if not isinstance(witness, ConversationHeldOutV4RuntimeTaskAnnotationWitness):
                raise TypeError("P3-C1a annotation factory yielded an invalid witness")
            if witness.ordinal != ordinal:
                _fail("P3-C1a annotation witness ordinal is not consecutive")
            key = (witness.case_key, witness.turn_key)
            if key in keys:
                _fail("P3-C1a annotation case/turn identity is duplicated")
            if witness.witness_record_identity not in payload_ids:
                _fail("P3-C1a annotation witness is not a current P3-A2 record")
            result.append(witness)
            keys.add(key)
            ordinal += 1
    except ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            "P3-C1a annotation factory yielded invalid data") from exc
    if not result:
        _fail("P3-C1a annotation factory yielded no witnesses")
    return tuple(result)


def materialize_v4_runtime_task_annotation_provenance(
        value: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput,
        ) -> ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult:
    """封存 annotation witnesses，三级完整回读后发布无正文 receipt。"""
    if not isinstance(value, ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput):
        raise TypeError("P3-C1a materialize requires AnnotationProvenanceInput")
    try:
        payload_publication = load_v4_payload_witness_publication(
            value.payload_witness,
            max_record_count=value.budget.max_upstream_payload_witness_record_count,
        )
        status = _roots(value)
    except (ConversationHeldOutV4PayloadWitnessLedgerError, KRunBoundaryError,
            TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            "P3-C1a input readback or root boundary failed") from exc
    payload = payload_publication.result
    payload_ids = frozenset(item.integer_stream() for item in payload_publication.records)
    if len(payload_ids) != len(payload_publication.records):
        _fail("P3-C1a P3-A2 witness identities are duplicated")
    witnesses = _collect(value, payload_ids=payload_ids)
    records = tuple(_record_for(item) for item in witnesses)
    staging = _write(value.staging_run_root, _STAGING_RECEIPT_FILE, records,
                     budget=value.budget, label="P3-C1a staging receipt")
    work = _write(value.work_run_root, _WORK_RECEIPT_FILE,
                  _read(value.staging_run_root, staging, budget=value.budget,
                        max_count=len(records), label="P3-C1a staging receipt"),
                  budget=value.budget, label="P3-C1a work receipt")
    stream = _write(value.publication_run_root, _RECEIPT_FILE,
                    _read(value.work_run_root, work, budget=value.budget,
                          max_count=len(records), label="P3-C1a work receipt"),
                    budget=value.budget, label="P3-C1a receipt")
    receipt = ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceipt(
        _hash(payload.stable_key(), domain=_DOMAIN_PAYLOAD_WITNESS,
              label="P3-C1a payload witness"), value.budget, len(witnesses), status)
    provisional = ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult(
        payload, value.publication_run_root, receipt, stream, (0,) * _SHA256_SIZE, witnesses)
    manifest_payload = encode_integer_tuple(_manifest(provisional))
    if len(manifest_payload) > value.budget.max_manifest_bytes:
        _fail("P3-C1a manifest exceeds budget")
    try:
        publish_manifest_last(value.publication_run_root, _MANIFEST_FILE, manifest_payload,
                              frozenset({_RECEIPT_FILE}), label="P3-C1a publication")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            "P3-C1a publication boundary failed") from exc
    result = ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult(
        payload, value.publication_run_root, receipt, stream,
        tuple(hashlib.sha256(manifest_payload).digest()), witnesses)
    return revalidate_v4_runtime_task_annotation_provenance(result)


def revalidate_v4_runtime_task_annotation_provenance(
        value: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult,
        *, max_upstream_payload_witness_record_count: int | None = None,
        ) -> ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult:
    """重验 P3-A2/C1a，并允许下游收紧完整 P3-A2 receipt 读取上限。"""
    if not isinstance(value, ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult):
        raise TypeError("P3-C1a revalidate requires AnnotationProvenanceResult")
    own_limit = value.receipt.budget.max_upstream_payload_witness_record_count
    if max_upstream_payload_witness_record_count is None:
        upstream_limit = own_limit
    else:
        if (type(max_upstream_payload_witness_record_count) is not int
                or max_upstream_payload_witness_record_count <= 0):
            raise ValueError("P3-C1a caller upstream witness cap must be a positive strict int")
        upstream_limit = min(own_limit, max_upstream_payload_witness_record_count)
    try:
        payload_publication = load_v4_payload_witness_publication(
            value.payload_witness,
            max_record_count=upstream_limit,
        )
        require_exact_file_closure(value.publication_run_root,
                                   frozenset({_RECEIPT_FILE, _MANIFEST_FILE}),
                                   label="P3-C1a publication")
        records = _read(value.publication_run_root, value.receipt_stream,
                        budget=value.receipt.budget,
                        max_count=value.receipt.annotation_witness_count,
                        label="P3-C1a receipt")
    except (ConversationHeldOutV4PayloadWitnessLedgerError, KRunBoundaryError,
            IntegerCodecError, IntegerFramedStreamError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            "P3-C1a receipt publication readback failed") from exc
    payload = payload_publication.result
    if value.receipt.payload_witness_identity_sha256 != _hash(
            payload.stable_key(), domain=_DOMAIN_PAYLOAD_WITNESS,
            label="P3-C1a payload witness"):
        _fail("P3-C1a P3-A2 receipt binding drifted")
    if (len(records) != value.receipt.annotation_witness_count
            or tuple(item.ordinal for item in records) != tuple(range(1, len(records) + 1))
            or len({(item.case_key_identity_sha256, item.turn_key_identity_sha256)
                    for item in records}) != len(records)):
        _fail("P3-C1a receipt annotation inventory drifted")
    payload_ids = frozenset(item.integer_stream() for item in payload_publication.records)
    if len(payload_ids) != len(payload_publication.records):
        _fail("P3-C1a P3-A2 witness identities are duplicated")
    for record, witness in zip(records, value.witnesses):
        if witness.witness_record_identity not in payload_ids:
            _fail("P3-C1a in-memory witness P3-A2 binding drifted")
        _require_binding(record, witness)
    try:
        identity = capture_plain_file_identity(value.publication_run_root, _MANIFEST_FILE,
                                               label="P3-C1a manifest")
        with open_plain_binary(value.publication_run_root, _MANIFEST_FILE,
                               label="P3-C1a manifest", expected_identity=identity) as stream:
            manifest_payload = stream.read(value.receipt.budget.max_manifest_bytes + 1)
        require_plain_file_identity(value.publication_run_root, _MANIFEST_FILE, identity,
                                    label="P3-C1a manifest post-read")
    except (KRunBoundaryError, OSError) as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            "P3-C1a manifest readback failed") from exc
    if len(manifest_payload) > value.receipt.budget.max_manifest_bytes:
        _fail("P3-C1a manifest exceeds budget")
    try:
        decoded = decode_integer_tuple(manifest_payload)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError(
            "P3-C1a manifest is corrupt") from exc
    if (encode_integer_tuple(decoded) != manifest_payload
            or decoded != _manifest(value)
            or tuple(hashlib.sha256(manifest_payload).digest()) != value.manifest_sha256):
        _fail("P3-C1a manifest identity drifted")
    return value


def load_v4_runtime_task_annotation_provenance_publication(
        value: ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult,
        *, max_upstream_payload_witness_record_count: int | None = None,
        ) -> ConversationHeldOutV4RuntimeTaskAnnotationProvenancePublication:
    """受限读取 public receipt，并把 caller 的 P3-A2 receipt 上限传入重验。"""
    result = revalidate_v4_runtime_task_annotation_provenance(
        value,
        max_upstream_payload_witness_record_count=(
            max_upstream_payload_witness_record_count),
    )
    records = _read(result.publication_run_root, result.receipt_stream,
                    budget=result.receipt.budget,
                    max_count=result.receipt.annotation_witness_count,
                    label="P3-C1a public receipt")
    for record, witness in zip(records, result.witnesses):
        _require_binding(record, witness)
    return ConversationHeldOutV4RuntimeTaskAnnotationProvenancePublication(result, records)


__all__ = [
    "ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget",
    "ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError",
    "ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput",
    "ConversationHeldOutV4RuntimeTaskAnnotationProvenancePublication",
    "ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceipt",
    "ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptRecord",
    "ConversationHeldOutV4RuntimeTaskAnnotationProvenanceReceiptStream",
    "ConversationHeldOutV4RuntimeTaskAnnotationProvenanceResult",
    "ConversationHeldOutV4RuntimeTaskAnnotationWitness",
    "V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_COVERAGE_NE",
    "V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_TEST_ONLY",
    "annotation_witness_public_identity_sha256",
    "load_v4_runtime_task_annotation_provenance_publication",
    "materialize_v4_runtime_task_annotation_provenance",
    "revalidate_v4_runtime_task_annotation_provenance",
    "require_v4_runtime_task_annotation_witness_binding",
]
