"""DLG-05 v4 R04b P3-B 的活动 roster 与 P2 双向只读回读。

本模块只消费已封存的 P3-A、P2-A/B1/B3 和 P2-C artifact。它把每一侧的完整
``ProvenanceLeaf`` 写为受限的临时整数流、经外排排序后逐条 merge join；不读取正文、
label、runtime 或训练状态，也不写入任何既有 input root。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator

from pure_integer_ai.experiments.conversation_heldout_v4_active_intake import (
    ConversationHeldOutV4ActiveIntakeError,
    ConversationHeldOutV4ActiveIntakeMapping,
    ConversationHeldOutV4ActiveIntakeResult,
    revalidate_v4_active_intake,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4ProvenanceError,
    ConversationHeldOutV4ProvenanceLeaf,
    V4_PROVENANCE_SIDE_HELD_OUT,
    V4_PROVENANCE_SIDE_TRAINING,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable import (
    ConversationHeldOutV4ProvenanceScalableError,
    ConversationHeldOutV4ProvenanceStreamCatalog,
    revalidate_v4_provenance_stream_catalog,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_dag import (
    ConversationHeldOutV4ProvenanceDirectDagResult,
    ConversationHeldOutV4ProvenanceScalableDagError,
    revalidate_v4_provenance_direct_dag_result,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_merge import (
    ConversationHeldOutV4ProvenanceCrossSideMergeError,
    ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult,
    load_v4_provenance_cross_side_merge_publication,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable_projection import (
    ConversationHeldOutV4ProvenanceProjectionResult,
    ConversationHeldOutV4ProvenanceScalableProjectionError,
    decode_v4_provenance_content_projection_record,
    decode_v4_provenance_lineage_projection_record,
    decode_v4_provenance_source_ref_projection_record,
    revalidate_v4_provenance_leaf_projections,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerFramedStreamError,
    IntegerFramedStreamFooter,
    IntegerFramedStreamReader,
    IntegerFramedStreamWriter,
    decode_integer_tuple,
    encode_integer_tuple,
    encoded_integer_tuple_size,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.integer_external_sort import (
    IntegerExternalSortBudget,
    IntegerExternalSortError,
    IntegerExternalSortResult,
    external_sort_sealed_integer_records,
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


V4_ACTIVE_ROSTER_READBACK_BUDGET_SCHEMA = 1
V4_ACTIVE_ROSTER_READBACK_STREAM_IDENTITY_SCHEMA = 1
V4_ACTIVE_ROSTER_READBACK_RECEIPT_SCHEMA = 2
V4_ACTIVE_ROSTER_READBACK_RESULT_SCHEMA = 1
V4_ACTIVE_ROSTER_READBACK_MANIFEST_SCHEMA = 1

V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY = "P3_ROSTER_READBACK_TEST_ONLY"
V4_ACTIVE_ROSTER_READBACK_STATUS_COVERAGE_NE = "ACTIVE_PROVENANCE_COVERAGE_NE"

_RAW_ACTIVE_TRAIN = Path("raw/active-train-leaves.pifrs")
_RAW_ACTIVE_HELD_OUT = Path("raw/active-held-out-leaves.pifrs")
_RAW_P2_TRAIN_SOURCE = Path("raw/p2-training-source-leaves.pifrs")
_RAW_P2_HELD_OUT_SOURCE = Path("raw/p2-held-out-source-leaves.pifrs")
_RAW_P2_TRAIN_CONTENT = Path("raw/p2-training-content-leaves.pifrs")
_RAW_P2_HELD_OUT_CONTENT = Path("raw/p2-held-out-content-leaves.pifrs")
_RAW_P2_TRAIN_LINEAGE = Path("raw/p2-training-lineage-leaves.pifrs")
_RAW_P2_HELD_OUT_LINEAGE = Path("raw/p2-held-out-lineage-leaves.pifrs")

_SORTED_ACTIVE_TRAIN = Path("sorted/active-train-leaves.pifrs")
_SORTED_ACTIVE_HELD_OUT = Path("sorted/active-held-out-leaves.pifrs")
_SORTED_P2_TRAIN_SOURCE = Path("sorted/p2-training-source-leaves.pifrs")
_SORTED_P2_HELD_OUT_SOURCE = Path("sorted/p2-held-out-source-leaves.pifrs")
_SORTED_P2_TRAIN_CONTENT = Path("sorted/p2-training-content-leaves.pifrs")
_SORTED_P2_HELD_OUT_CONTENT = Path("sorted/p2-held-out-content-leaves.pifrs")
_SORTED_P2_TRAIN_LINEAGE = Path("sorted/p2-training-lineage-leaves.pifrs")
_SORTED_P2_HELD_OUT_LINEAGE = Path("sorted/p2-held-out-lineage-leaves.pifrs")

_RECEIPT_FILE = Path("readback/receipt.pifrs")
_MANIFEST_FILE = Path("manifest.pii")


# object-model: exception
class ConversationHeldOutV4ActiveRosterReadbackError(RuntimeError):
    """P3-B 的 root、输入身份、leaf equality 或 publication 回读不成立。"""


def _fail(message: str) -> None:
    """统一生成不携带正文、label 或本机路径的 P3-B fail-closed 错误。"""
    raise ConversationHeldOutV4ActiveRosterReadbackError(message)


def _require_positive_int(value: int, *, label: str) -> int:
    """校验预算中的正严格整数，拒绝 bool、零和负数。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def _require_nonnegative_int(value: int, *, label: str) -> int:
    """校验统计字段中的非负严格整数。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _stage_name(value: str) -> str:
    """限制 P3-B stage 为无路径语义的短 ASCII 标识。"""
    if not isinstance(value, str) or not value or len(value) > 56:
        raise ValueError("P3-B logical_stage_name 长度非法")
    if not value[0].isascii() or not value[0].isalnum():
        raise ValueError("P3-B logical_stage_name 必须以 ASCII 字母或数字开始")
    if any(not item.isascii() or not (item.isalnum() or item in {"-", "_", "."})
           for item in value):
        raise ValueError("P3-B logical_stage_name 含非法字符")
    return value


def _relative_path_key(value: Path, *, label: str) -> tuple[int, ...]:
    """把受控的 ASCII 相对路径写成不含 root 的稳定整数键。"""
    if (not isinstance(value, Path) or value.is_absolute() or value.drive
            or value.root or not value.parts or ".." in value.parts
            or any(part in {"", ".", ".."} for part in value.parts)):
        raise ValueError(f"{label} 必须是非空相对 Path")
    text = value.as_posix()
    if not text.isascii():
        raise ValueError(f"{label} 必须只含 ASCII")
    return tuple(ord(item) for item in text)


def _text_scalars(value: str, *, label: str) -> tuple[int, ...]:
    """把有限状态文字作为规范 Unicode scalar tuple 封存。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} 必须是非空 str")
    scalars = tuple(ord(item) for item in value)
    if any(item < 1 or 0xD800 <= item <= 0xDFFF or item > 0x10FFFF
           for item in scalars):
        raise ValueError(f"{label} 含非法 Unicode scalar")
    return scalars


def _digest_key(value: KRunFileDigest, *, label: str) -> tuple[int, ...]:
    """展开物理 byte count 和完整 SHA-256，避免仅绑定裸摘要。"""
    if not isinstance(value, KRunFileDigest):
        raise TypeError(f"{label} 必须是 KRunFileDigest")
    return (value.byte_count, *value.sha256)


def _leaf_from_record(value: tuple[int, ...], *, label: str) -> ConversationHeldOutV4ProvenanceLeaf:
    """严格恢复完整 leaf，拒绝摘要或局部 endpoint 伪装为 roster 项。"""
    try:
        leaf = ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(value)
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            f"{label} full leaf 编码非法") from exc
    if leaf.integer_stream() != value:
        _fail(f"{label} full leaf 不是规范编码")
    return leaf


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveRosterReadbackBudget:
    """P3-B 读取、leaf transport、外排与最终 evidence 的有限预算。"""

    max_active_mapping_record_count: int
    max_projection_record_count: int
    max_input_record_payload_bytes: int
    max_leaf_record_payload_bytes: int
    max_leaf_stream_payload_bytes: int
    max_leaf_stream_physical_bytes: int
    max_receipt_payload_bytes: int
    max_receipt_physical_bytes: int
    max_manifest_bytes: int
    external_sort_budget: IntegerExternalSortBudget

    def __post_init__(self) -> None:
        """拒绝无界读写或无法容纳单条 leaf 的预算组合。"""
        for label, value in (
                ("max_active_mapping_record_count", self.max_active_mapping_record_count),
                ("max_projection_record_count", self.max_projection_record_count),
                ("max_input_record_payload_bytes", self.max_input_record_payload_bytes),
                ("max_leaf_record_payload_bytes", self.max_leaf_record_payload_bytes),
                ("max_leaf_stream_payload_bytes", self.max_leaf_stream_payload_bytes),
                ("max_leaf_stream_physical_bytes", self.max_leaf_stream_physical_bytes),
                ("max_receipt_payload_bytes", self.max_receipt_payload_bytes),
                ("max_receipt_physical_bytes", self.max_receipt_physical_bytes),
                ("max_manifest_bytes", self.max_manifest_bytes)):
            _require_positive_int(value, label=f"P3-B {label}")
        if self.max_leaf_record_payload_bytes > self.max_leaf_stream_payload_bytes:
            raise ValueError("P3-B 单 leaf payload 不得超过 leaf stream 预算")
        if self.max_receipt_payload_bytes > self.max_receipt_physical_bytes:
            raise ValueError("P3-B receipt payload 不得超过 receipt physical 预算")
        if not isinstance(self.external_sort_budget, IntegerExternalSortBudget):
            raise TypeError("P3-B external_sort_budget 类型错误")

    def integer_stream(self) -> tuple[int, ...]:
        """返回可被 receipt/manifest 绑定的完整预算身份。"""
        result = [V4_ACTIVE_ROSTER_READBACK_BUDGET_SCHEMA]
        result.extend((
            self.max_active_mapping_record_count,
            self.max_projection_record_count,
            self.max_input_record_payload_bytes,
            self.max_leaf_record_payload_bytes,
            self.max_leaf_stream_payload_bytes,
            self.max_leaf_stream_physical_bytes,
            self.max_receipt_payload_bytes,
            self.max_receipt_physical_bytes,
            self.max_manifest_bytes,
        ))
        pack_key(result, self.external_sort_budget.integer_stream())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveRosterReadbackStreamIdentity:
    """P3-B 临时 leaf 或 final receipt P0 stream 的可重验身份。"""

    relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """固定相对路径、sealed footer 和完整物理身份。"""
        _relative_path_key(self.relative_path, label="P3-B stream relative_path")
        if not isinstance(self.p0_footer, IntegerFramedStreamFooter):
            raise TypeError("P3-B stream p0_footer 类型错误")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("P3-B stream physical 类型错误")
        if self.physical.byte_count <= 0:
            raise ValueError("P3-B sealed stream physical bytes 必须为正")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 root 的 stream descriptor 身份。"""
        result = [V4_ACTIVE_ROSTER_READBACK_STREAM_IDENTITY_SCHEMA]
        for value in (
                _relative_path_key(self.relative_path, label="P3-B stream relative_path"),
                self.p0_footer.integer_tuple(),
                _digest_key(self.physical, label="P3-B stream physical")):
            pack_key(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveRosterReadbackCounts:
    """P3-B 实际读取、投影与去重后的各侧 leaf 数量。"""

    active_train_leaf_count: int
    active_held_out_leaf_count: int
    training_source_leaf_count: int
    held_out_source_leaf_count: int
    training_content_leaf_count: int
    held_out_content_leaf_count: int
    training_lineage_record_count: int
    held_out_lineage_record_count: int
    training_lineage_unique_leaf_count: int
    held_out_lineage_unique_leaf_count: int

    def __post_init__(self) -> None:
        """要求所有计数为非负，并让 SourceRef/content 与 roster 闭合。"""
        for label, value in zip(
                (
                    "active_train_leaf_count", "active_held_out_leaf_count",
                    "training_source_leaf_count", "held_out_source_leaf_count",
                    "training_content_leaf_count", "held_out_content_leaf_count",
                    "training_lineage_record_count", "held_out_lineage_record_count",
                    "training_lineage_unique_leaf_count",
                    "held_out_lineage_unique_leaf_count"),
                self.integer_stream()):
            _require_nonnegative_int(value, label=f"P3-B {label}")
        if (self.active_train_leaf_count != self.training_source_leaf_count
                or self.active_train_leaf_count != self.training_content_leaf_count
                or self.active_held_out_leaf_count != self.held_out_source_leaf_count
                or self.active_held_out_leaf_count != self.held_out_content_leaf_count):
            raise ValueError("P3-B SourceRef/content leaf count 未与 active roster 闭合")
        if (self.training_lineage_unique_leaf_count != self.active_train_leaf_count
                or self.held_out_lineage_unique_leaf_count
                != self.active_held_out_leaf_count):
            raise ValueError("P3-B lineage unique leaf count 未与 active roster 闭合")
        if (self.training_lineage_record_count < self.training_lineage_unique_leaf_count
                or self.held_out_lineage_record_count
                < self.held_out_lineage_unique_leaf_count):
            raise ValueError("P3-B lineage record count 不得少于 unique leaf count")

    def integer_stream(self) -> tuple[int, ...]:
        """返回 receipt 可绑定的完整实际读取统计。"""
        return (
            self.active_train_leaf_count,
            self.active_held_out_leaf_count,
            self.training_source_leaf_count,
            self.held_out_source_leaf_count,
            self.training_content_leaf_count,
            self.held_out_content_leaf_count,
            self.training_lineage_record_count,
            self.held_out_lineage_record_count,
            self.training_lineage_unique_leaf_count,
            self.held_out_lineage_unique_leaf_count,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveRosterReadbackReceipt:
    """P3-B 的无正文、无路径双向 leaf equality 证据本体。"""

    active_intake_stable_key: tuple[int, ...]
    training_catalog_stable_key: tuple[int, ...]
    held_out_catalog_stable_key: tuple[int, ...]
    training_b1_stable_key: tuple[int, ...]
    held_out_b1_stable_key: tuple[int, ...]
    training_b3_stable_key: tuple[int, ...]
    held_out_b3_stable_key: tuple[int, ...]
    p2_publication_stable_key: tuple[int, ...]
    p2_code_identity: ProtocolKey
    p3_code_identity: ProtocolKey
    budget: ConversationHeldOutV4ActiveRosterReadbackBudget
    counts: ConversationHeldOutV4ActiveRosterReadbackCounts
    status: str

    def __post_init__(self) -> None:
        """绑定全部上游 identity、实际计数与受限状态。"""
        for label, value in (
                ("active_intake_stable_key", self.active_intake_stable_key),
                ("training_catalog_stable_key", self.training_catalog_stable_key),
                ("held_out_catalog_stable_key", self.held_out_catalog_stable_key),
                ("training_b1_stable_key", self.training_b1_stable_key),
                ("held_out_b1_stable_key", self.held_out_b1_stable_key),
                ("training_b3_stable_key", self.training_b3_stable_key),
                ("held_out_b3_stable_key", self.held_out_b3_stable_key),
                ("p2_publication_stable_key", self.p2_publication_stable_key)):
            try:
                strict_integer_tuple(value, label=f"P3-B receipt {label}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"P3-B receipt {label} 非法") from exc
        if not isinstance(self.p2_code_identity, ProtocolKey):
            raise TypeError("P3-B receipt p2_code_identity 类型错误")
        if not isinstance(self.p3_code_identity, ProtocolKey):
            raise TypeError("P3-B receipt p3_code_identity 类型错误")
        if not isinstance(self.budget, ConversationHeldOutV4ActiveRosterReadbackBudget):
            raise TypeError("P3-B receipt budget 类型错误")
        if not isinstance(self.counts, ConversationHeldOutV4ActiveRosterReadbackCounts):
            raise TypeError("P3-B receipt counts 类型错误")
        if self.status not in {
                V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY,
                V4_ACTIVE_ROSTER_READBACK_STATUS_COVERAGE_NE}:
            raise ValueError("P3-B receipt status 未注册")

    def integer_stream(self) -> tuple[int, ...]:
        """返回一条可封存 P0 receipt 的完整规范整数表达。"""
        result = [V4_ACTIVE_ROSTER_READBACK_RECEIPT_SCHEMA]
        for value in (
                self.active_intake_stable_key,
                self.training_catalog_stable_key,
                self.held_out_catalog_stable_key,
                self.training_b1_stable_key,
                self.held_out_b1_stable_key,
                self.training_b3_stable_key,
                self.held_out_b3_stable_key,
                self.p2_publication_stable_key,
                self.p2_code_identity.components,
                self.p3_code_identity.components,
                self.budget.integer_stream(),
                self.counts.integer_stream(),
                _text_scalars(self.status, label="P3-B receipt status")):
            pack_key(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveRosterReadbackInput:
    """P3-B 唯一入口的显式已封存输入与新 verification roots。"""

    active_intake: ConversationHeldOutV4ActiveIntakeResult
    training_catalog: ConversationHeldOutV4ProvenanceStreamCatalog
    training_b1_result: ConversationHeldOutV4ProvenanceDirectDagResult
    training_b3_result: ConversationHeldOutV4ProvenanceProjectionResult
    training_b3_work_root: KRunRoot
    held_out_catalog: ConversationHeldOutV4ProvenanceStreamCatalog
    held_out_b1_result: ConversationHeldOutV4ProvenanceDirectDagResult
    held_out_b3_result: ConversationHeldOutV4ProvenanceProjectionResult
    held_out_b3_work_root: KRunRoot
    p2_publication: ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult
    p2_publication_root: KRunRoot
    staging_run_root: KRunRoot
    work_run_root: KRunRoot
    publication_run_root: KRunRoot
    code_identity: ProtocolKey
    budget: ConversationHeldOutV4ActiveRosterReadbackBudget
    logical_stage_name: str

    def __post_init__(self) -> None:
        """只接受完整 typed result、显式 roots 与冻结 code/budget。"""
        if not isinstance(self.active_intake, ConversationHeldOutV4ActiveIntakeResult):
            raise TypeError("P3-B active_intake 类型错误")
        if not isinstance(self.training_catalog,
                          ConversationHeldOutV4ProvenanceStreamCatalog):
            raise TypeError("P3-B training_catalog 类型错误")
        if not isinstance(self.training_b1_result,
                          ConversationHeldOutV4ProvenanceDirectDagResult):
            raise TypeError("P3-B training_b1_result 类型错误")
        if not isinstance(self.training_b3_result, ConversationHeldOutV4ProvenanceProjectionResult):
            raise TypeError("P3-B training_b3_result 类型错误")
        if not isinstance(self.held_out_catalog,
                          ConversationHeldOutV4ProvenanceStreamCatalog):
            raise TypeError("P3-B held_out_catalog 类型错误")
        if not isinstance(self.held_out_b1_result,
                          ConversationHeldOutV4ProvenanceDirectDagResult):
            raise TypeError("P3-B held_out_b1_result 类型错误")
        if not isinstance(self.held_out_b3_result, ConversationHeldOutV4ProvenanceProjectionResult):
            raise TypeError("P3-B held_out_b3_result 类型错误")
        if not isinstance(self.p2_publication,
                          ConversationHeldOutV4ProvenanceCrossSideMergePublicationResult):
            raise TypeError("P3-B p2_publication 类型错误")
        for label, root in (
                ("training_b3_work_root", self.training_b3_work_root),
                ("held_out_b3_work_root", self.held_out_b3_work_root),
                ("p2_publication_root", self.p2_publication_root),
                ("staging_run_root", self.staging_run_root),
                ("work_run_root", self.work_run_root),
                ("publication_run_root", self.publication_run_root)):
            if not isinstance(root, KRunRoot):
                raise TypeError(f"P3-B {label} 必须是 KRunRoot")
        if not isinstance(self.code_identity, ProtocolKey):
            raise TypeError("P3-B code_identity 必须是 ProtocolKey")
        if not isinstance(self.budget, ConversationHeldOutV4ActiveRosterReadbackBudget):
            raise TypeError("P3-B budget 类型错误")
        _stage_name(self.logical_stage_name)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveRosterReadbackResult:
    """P3-B 已发布 evidence，不表示训练或来源资格已通过。"""

    publication_run_root: KRunRoot
    receipt: ConversationHeldOutV4ActiveRosterReadbackReceipt
    receipt_stream: ConversationHeldOutV4ActiveRosterReadbackStreamIdentity
    manifest_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """确保 result 只指向自身 exact 两文件 publication root。"""
        if not isinstance(self.publication_run_root, KRunRoot):
            raise TypeError("P3-B result publication_run_root 类型错误")
        if not isinstance(self.receipt, ConversationHeldOutV4ActiveRosterReadbackReceipt):
            raise TypeError("P3-B result receipt 类型错误")
        if not isinstance(self.receipt_stream,
                          ConversationHeldOutV4ActiveRosterReadbackStreamIdentity):
            raise TypeError("P3-B result receipt_stream 类型错误")
        if self.receipt_stream.relative_path != _RECEIPT_FILE:
            raise ValueError("P3-B result receipt 路径漂移")
        if (not isinstance(self.manifest_sha256, tuple)
                or len(self.manifest_sha256) != 32
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.manifest_sha256)):
            raise ValueError("P3-B result manifest_sha256 非法")
        expected_status = (
            V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY
            if self.publication_run_root.test_transport
            else V4_ACTIVE_ROSTER_READBACK_STATUS_COVERAGE_NE)
        if self.receipt.status != expected_status:
            raise ValueError("P3-B result status 与 publication transport 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含绝对 root 的 P3-C 可绑定 readback identity。"""
        result = [V4_ACTIVE_ROSTER_READBACK_RESULT_SCHEMA]
        for value in (
                self.receipt.integer_stream(),
                self.receipt_stream.integer_stream(),
                self.manifest_sha256):
            pack_key(result, value)
        return tuple(result)


# object-model: resource_owner; representation=protocol; interop=pending
class _LeafStreamWriter:
    """P3-B 单条 raw leaf stream 的唯一 writer owner。"""

    __slots__ = ("_budget", "_count", "_label", "_payload_bytes", "_relative",
                 "_root", "_writer")

    def __init__(self, root: KRunRoot, relative: Path, *,
                 budget: ConversationHeldOutV4ActiveRosterReadbackBudget,
                 label: str) -> None:
        """以 K-boundary 排他 handle 创建一个受预算约束的 raw P0 leaf stream。"""
        parent = relative.parent
        if parent.parts:
            ensure_normal_relative_directory(root, parent, label=f"{label} parent")
        stream = open_exclusive_binary(root, relative, label=label)
        try:
            writer = IntegerFramedStreamWriter.from_open_binary(stream, path=relative)
        except BaseException:
            stream.close()
            raise
        self._budget = budget
        self._count = 0
        self._label = label
        self._payload_bytes = 0
        self._relative = relative
        self._root = root
        self._writer = writer

    def append(self, leaf: ConversationHeldOutV4ProvenanceLeaf, *, max_count: int) -> None:
        """按完整 leaf 本体追加一条 P0 record，绝不写局部键或摘要。"""
        if not isinstance(leaf, ConversationHeldOutV4ProvenanceLeaf):
            raise TypeError(f"{self._label} leaf 类型错误")
        record = leaf.integer_stream()
        try:
            strict_integer_tuple(record, label=f"{self._label} leaf record")
            payload_bytes = encoded_integer_tuple_size(record)
        except (IntegerCodecError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ActiveRosterReadbackError(
                f"{self._label} leaf record 非法") from exc
        if self._count >= max_count:
            _fail(f"{self._label} leaf count 超过预算")
        if payload_bytes > self._budget.max_leaf_record_payload_bytes:
            _fail(f"{self._label} 单 leaf payload 超过预算")
        if payload_bytes > self._budget.max_leaf_stream_payload_bytes - self._payload_bytes:
            _fail(f"{self._label} leaf stream payload 超过预算")
        try:
            self._writer.append(record)
        except (IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ActiveRosterReadbackError(
                f"{self._label} P0 写入失败") from exc
        self._count += 1
        self._payload_bytes += payload_bytes

    def seal(self) -> ConversationHeldOutV4ActiveRosterReadbackStreamIdentity:
        """封存 raw stream 并立即绑定完整物理 SHA-256。"""
        try:
            footer = self._writer.seal()
            physical = sha256_plain_file(
                self._root,
                self._relative,
                max_bytes=self._budget.max_leaf_stream_physical_bytes,
                label=f"{self._label} physical")
        except (IntegerFramedStreamError, KRunBoundaryError, OSError, TypeError,
                ValueError) as exc:
            raise ConversationHeldOutV4ActiveRosterReadbackError(
                f"{self._label} P0 封存或 physical 核验失败") from exc
        if (footer.record_count != self._count
                or footer.total_payload_bytes != self._payload_bytes):
            _fail(f"{self._label} P0 footer 与实际写入计数不一致")
        return ConversationHeldOutV4ActiveRosterReadbackStreamIdentity(
            self._relative, footer, physical)

    def close(self) -> None:
        """释放失败路径的 writer，不删除任何已写残片。"""
        self._writer.close()


def _iter_verified_p0_records(
        root: KRunRoot, relative: Path, *, expected_footer: IntegerFramedStreamFooter,
        expected_physical: KRunFileDigest, max_record_count: int,
        max_record_payload_bytes: int, max_total_payload_bytes: int,
        max_physical_bytes: int, label: str,
        ) -> Iterator[tuple[int, ...]]:
    """通过 capability 流式重读 sealed P0，并在前后重验 footer/physical/path。"""
    try:
        before_physical = sha256_plain_file(
            root, relative, max_bytes=max_physical_bytes, label=f"{label} pre-read")
        if before_physical != expected_physical:
            _fail(f"{label} physical identity 漂移")
        identity = capture_plain_file_identity(root, relative, label=f"{label} identity")
        count = 0
        with open_plain_binary(
                root, relative, label=label, expected_identity=identity) as stream:
            with IntegerFramedStreamReader.from_open_binary(
                    stream,
                    path=relative,
                    max_frame_bytes=max_record_payload_bytes,
                    max_record_count=max_record_count,
                    max_total_payload_bytes=max_total_payload_bytes,
            ) as reader:
                while True:
                    record = reader.read_record()
                    if record is None:
                        break
                    count += 1
                    yield record
                footer = reader.finish()
        require_plain_file_identity(root, relative, identity, label=f"{label} post-read")
        after_physical = sha256_plain_file(
            root, relative, max_bytes=max_physical_bytes, label=f"{label} post-read")
    except ConversationHeldOutV4ActiveRosterReadbackError:
        raise
    except (IntegerFramedStreamError, KRunBoundaryError, OSError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            f"{label} sealed P0 回读失败") from exc
    if footer != expected_footer or count != expected_footer.record_count:
        _fail(f"{label} P0 footer 或 record count 漂移")
    if after_physical != expected_physical:
        _fail(f"{label} post-read physical identity 漂移")


def _sort_leaf_stream(
        *, staging_root: KRunRoot, work_root: KRunRoot,
        raw: ConversationHeldOutV4ActiveRosterReadbackStreamIdentity,
        output_relative: Path, logical_stage_name: str,
        budget: ConversationHeldOutV4ActiveRosterReadbackBudget, label: str,
        ) -> ConversationHeldOutV4ActiveRosterReadbackStreamIdentity:
    """用现役 bounded external sort 以完整 leaf key 排序一条 raw leaf stream。"""
    def leaf_sort_key(record: tuple[int, ...]) -> tuple[int, ...]:
        """拒绝无完整本体的排序键，保证跨输入可确定比较。"""
        return _leaf_from_record(record, label=f"{label} sort").stable_key()

    try:
        result: IntegerExternalSortResult = external_sort_sealed_integer_records(
            staging_root,
            (raw.relative_path,),
            work_root,
            output_relative_path=output_relative,
            logical_stage_name=logical_stage_name,
            sort_key=leaf_sort_key,
            budget=budget.external_sort_budget,
        )
    except (IntegerExternalSortError, KRunBoundaryError, IntegerFramedStreamError,
            OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            f"{label} external sort 未通过") from exc
    identities = result.identity.input_identities
    if (len(identities) != 1
            or identities[0].relative_path != raw.relative_path
            or identities[0].footer != raw.p0_footer
            or identities[0].physical != raw.physical):
        _fail(f"{label} external sort 输入 identity 漂移")
    return ConversationHeldOutV4ActiveRosterReadbackStreamIdentity(
        output_relative,
        result.identity.output_footer,
        result.identity.output_physical,
    )


def _iter_sorted_leaves(
        root: KRunRoot, stream: ConversationHeldOutV4ActiveRosterReadbackStreamIdentity,
        *, budget: ConversationHeldOutV4ActiveRosterReadbackBudget, label: str,
        ) -> Iterator[ConversationHeldOutV4ProvenanceLeaf]:
    """流式重建一个已排序 leaf stream，保证每条都保留完整本体。"""
    for raw in _iter_verified_p0_records(
            root,
            stream.relative_path,
            expected_footer=stream.p0_footer,
            expected_physical=stream.physical,
            max_record_count=budget.max_projection_record_count,
            max_record_payload_bytes=budget.max_leaf_record_payload_bytes,
            max_total_payload_bytes=budget.max_leaf_stream_payload_bytes,
            max_physical_bytes=budget.max_leaf_stream_physical_bytes,
            label=label):
        yield _leaf_from_record(raw, label=label)


def _compare_leaf_set(
        active: ConversationHeldOutV4ActiveRosterReadbackStreamIdentity,
        candidate: ConversationHeldOutV4ActiveRosterReadbackStreamIdentity,
        *, work_root: KRunRoot, budget: ConversationHeldOutV4ActiveRosterReadbackBudget,
        label: str, allow_candidate_duplicates: bool,
        ) -> tuple[int, int]:
    """以两个流式排序 leaf 序列精确比较集合，返回 active/candidate unique count。"""
    active_iter = iter(_iter_sorted_leaves(
        work_root, active, budget=budget, label=f"{label} active"))
    candidate_iter = iter(_iter_sorted_leaves(
        work_root, candidate, budget=budget, label=f"{label} candidate"))
    active_previous: tuple[int, ...] | None = None
    candidate_previous: tuple[int, ...] | None = None
    active_count = 0
    candidate_raw_count = 0
    candidate_unique_count = 0
    active_item = next(active_iter, None)
    candidate_item = next(candidate_iter, None)
    while active_item is not None or candidate_item is not None:
        if active_item is not None:
            active_key = active_item.stable_key()
            if active_previous is not None and active_key <= active_previous:
                _fail(f"{label} active leaf 重复或排序漂移")
        else:
            active_key = None
        if candidate_item is not None:
            candidate_key = candidate_item.stable_key()
            candidate_raw_count += 1
            if candidate_previous is not None and candidate_key < candidate_previous:
                _fail(f"{label} candidate leaf 排序漂移")
            if candidate_previous is not None and candidate_key == candidate_previous:
                if not allow_candidate_duplicates:
                    _fail(f"{label} candidate full leaf 重复")
                candidate_item = next(candidate_iter, None)
                continue
        else:
            candidate_key = None
        if active_key is None or candidate_key is None or active_key != candidate_key:
            _fail(f"{label} active roster 与 P2 leaf set 不精确相等")
        active_previous = active_key
        candidate_previous = candidate_key
        active_count += 1
        candidate_unique_count += 1
        active_item = next(active_iter, None)
        candidate_item = next(candidate_iter, None)
    if candidate_raw_count < candidate_unique_count:
        raise AssertionError("P3-B candidate raw/unique 计数不变量损坏")
    return active_count, candidate_raw_count


def _p2_projection_leaf(
        raw: tuple[int, ...], *, kind: str, expected_side: int,
        label: str,
        ) -> ConversationHeldOutV4ProvenanceLeaf:
    """从一条 B3 projection 取回 full leaf，并交叉核验其 projection 字段。"""
    try:
        if kind == "source":
            source_ref, side, leaf = decode_v4_provenance_source_ref_projection_record(raw)
            if source_ref != leaf.source_ref:
                _fail(f"{label} SourceRef projection 与 leaf 不闭合")
        elif kind == "content":
            content_sha256, side, leaf = decode_v4_provenance_content_projection_record(raw)
            if content_sha256 != leaf.content_sha256:
                _fail(f"{label} content projection 与 leaf 不闭合")
        elif kind == "lineage":
            _namespace, _node, _snapshot, side, leaf = (
                decode_v4_provenance_lineage_projection_record(raw))
        else:
            raise AssertionError("P3-B projection kind 未注册")
    except ConversationHeldOutV4ActiveRosterReadbackError:
        raise
    except (ConversationHeldOutV4ProvenanceScalableProjectionError,
            ConversationHeldOutV4ProvenanceError, IntegerCodecError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            f"{label} B3 projection record 非法") from exc
    if side != expected_side or leaf.side != expected_side:
        _fail(f"{label} B3 projection side 漂移")
    return leaf


def _materialize_active_raw_leaves(
        value: ConversationHeldOutV4ActiveRosterReadbackInput,
        ) -> tuple[
            ConversationHeldOutV4ActiveRosterReadbackStreamIdentity,
            ConversationHeldOutV4ActiveRosterReadbackStreamIdentity,
        ]:
    """从 P3-A 无正文 mapping 投影 train/held-out full leaf，单次流式读取。"""
    train_writer = _LeafStreamWriter(
        value.staging_run_root, _RAW_ACTIVE_TRAIN, budget=value.budget,
        label="P3-B active train raw")
    held_writer = _LeafStreamWriter(
        value.staging_run_root, _RAW_ACTIVE_HELD_OUT, budget=value.budget,
        label="P3-B active held-out raw")
    count = 0
    try:
        mapping_stream = value.active_intake.mapping_stream
        for raw in _iter_verified_p0_records(
                value.active_intake.publication_run_root,
                mapping_stream.relative_path,
                expected_footer=mapping_stream.p0_footer,
                expected_physical=mapping_stream.physical,
                max_record_count=value.budget.max_active_mapping_record_count,
                max_record_payload_bytes=value.budget.max_input_record_payload_bytes,
                max_total_payload_bytes=value.active_intake.budget.max_stream_payload_bytes,
                max_physical_bytes=value.active_intake.budget.max_stream_physical_bytes,
                label="P3-B active mapping"):
            try:
                mapping = ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(raw)
            except ConversationHeldOutV4ActiveIntakeError as exc:
                raise ConversationHeldOutV4ActiveRosterReadbackError(
                    "P3-B active mapping record 非法") from exc
            if mapping.split == "train":
                train_writer.append(
                    mapping.leaf,
                    max_count=value.budget.max_active_mapping_record_count)
            elif mapping.split == "held_out":
                held_writer.append(
                    mapping.leaf,
                    max_count=value.budget.max_active_mapping_record_count)
            else:
                _fail("P3-B active mapping split 未注册")
            count += 1
        if count != value.active_intake.counts.mapping_record_count:
            _fail("P3-B active mapping 实际读取计数漂移")
        return train_writer.seal(), held_writer.seal()
    finally:
        train_writer.close()
        held_writer.close()


def _projection_descriptor(
        result: ConversationHeldOutV4ProvenanceProjectionResult, *, kind: str):
    """只按固定 B3 槽位取得 descriptor，拒绝由路径或枚举猜测 stream。"""
    if kind == "source":
        return result.source_ref_projection_stream
    if kind == "content":
        return result.content_projection_stream
    if kind == "lineage":
        return result.lineage_projection_stream
    raise AssertionError("P3-B projection kind 未注册")


def _materialize_p2_raw_leaves(
        root: KRunRoot, result: ConversationHeldOutV4ProvenanceProjectionResult,
        *, expected_side: int, kind: str, output_root: KRunRoot,
        output_relative: Path, value: ConversationHeldOutV4ActiveRosterReadbackInput,
        label: str,
        ) -> ConversationHeldOutV4ActiveRosterReadbackStreamIdentity:
    """从已公开复验的 B3 descriptor 逐条投影 full leaf 到 P3-B staging。"""
    descriptor = _projection_descriptor(result, kind=kind)
    writer = _LeafStreamWriter(output_root, output_relative, budget=value.budget,
                               label=label)
    expected_count = {
        "source": result.source_ref_projection_count,
        "content": result.content_projection_count,
        "lineage": result.lineage_projection_count,
    }[kind]
    count = 0
    try:
        for raw in _iter_verified_p0_records(
                root,
                descriptor.work_relative_path,
                expected_footer=descriptor.p0_footer,
                expected_physical=descriptor.physical,
                max_record_count=value.budget.max_projection_record_count,
                max_record_payload_bytes=value.budget.max_input_record_payload_bytes,
                max_total_payload_bytes=_budget_projection_payload_limit(
                    value.budget, result),
                max_physical_bytes=value.budget.max_leaf_stream_physical_bytes,
                label=label):
            writer.append(
                _p2_projection_leaf(raw, kind=kind, expected_side=expected_side,
                                    label=label),
                max_count=value.budget.max_projection_record_count)
            count += 1
        if count != expected_count:
            _fail(f"{label} B3 实际读取计数漂移")
        return writer.seal()
    finally:
        writer.close()


def _budget_projection_payload_limit(
        self: ConversationHeldOutV4ActiveRosterReadbackBudget,
        result: ConversationHeldOutV4ProvenanceProjectionResult,
        ) -> int:
    """读取 B3 时采用其冻结 stream 上限与 P3-B input 上限的较小值。"""
    if not isinstance(result, ConversationHeldOutV4ProvenanceProjectionResult):
        raise TypeError("P3-B B3 result 类型错误")
    return min(self.max_leaf_stream_payload_bytes,
               result.budget.max_stream_payload_bytes)


def _validate_roots(value: ConversationHeldOutV4ActiveRosterReadbackInput) -> str:
    """验证所有 root 的 transport mode，并只允许相同冻结输入的别名复用。"""
    input_roots = (
        value.active_intake.publication_run_root,
        value.training_catalog.root,
        value.training_b3_work_root,
        value.held_out_catalog.root,
        value.held_out_b3_work_root,
        value.p2_publication_root,
    )
    output_roots = (
        value.staging_run_root,
        value.work_run_root,
        value.publication_run_root,
    )
    roots = input_roots + output_roots
    if len({root.test_transport for root in roots}) != 1:
        _fail("P3-B test transport 与 production K root 不得混用")
    unique_input_roots: list[KRunRoot] = []
    for root in input_roots:
        if all(root.path != existing.path for existing in unique_input_roots):
            unique_input_roots.append(root)
    for index, left in enumerate(unique_input_roots):
        for right in unique_input_roots[index + 1:]:
            try:
                require_disjoint_run_roots(left, right, label="P3-B run roots")
            except KRunBoundaryError as exc:
                raise ConversationHeldOutV4ActiveRosterReadbackError(
                    "P3-B run roots 重叠或嵌套") from exc
    for index, left in enumerate(output_roots):
        for right in output_roots[index + 1:]:
            try:
                require_disjoint_run_roots(left, right, label="P3-B output roots")
            except KRunBoundaryError as exc:
                raise ConversationHeldOutV4ActiveRosterReadbackError(
                    "P3-B output roots 重叠或嵌套") from exc
    for output_root in output_roots:
        for input_root in unique_input_roots:
            try:
                require_disjoint_run_roots(
                    output_root, input_root, label="P3-B input/output roots")
            except KRunBoundaryError as exc:
                raise ConversationHeldOutV4ActiveRosterReadbackError(
                    "P3-B input/output roots 重叠或嵌套") from exc
    for label, root in (
            ("staging_run_root", value.staging_run_root),
            ("work_run_root", value.work_run_root),
            ("publication_run_root", value.publication_run_root)):
        try:
            require_created_run_root(root, label=f"P3-B {label}")
            require_fresh_empty_run_root(root, label=f"P3-B {label}")
        except KRunBoundaryError as exc:
            raise ConversationHeldOutV4ActiveRosterReadbackError(
                f"P3-B {label} 必须是本次 fresh K capability") from exc
    return (V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY if roots[0].test_transport
            else V4_ACTIVE_ROSTER_READBACK_STATUS_COVERAGE_NE)


def _manifest_integer_stream(
        value: ConversationHeldOutV4ActiveRosterReadbackResult,
        ) -> tuple[int, ...]:
    """构造 manifest-last 的无路径规范整数 payload。"""
    result = [V4_ACTIVE_ROSTER_READBACK_MANIFEST_SCHEMA]
    for item in (
            value.receipt.integer_stream(),
            value.receipt_stream.integer_stream()):
        pack_key(result, item)
    return tuple(result)


def _write_receipt(
        root: KRunRoot, receipt: ConversationHeldOutV4ActiveRosterReadbackReceipt,
        *, budget: ConversationHeldOutV4ActiveRosterReadbackBudget,
        ) -> ConversationHeldOutV4ActiveRosterReadbackStreamIdentity:
    """排他写入一条 P0 receipt，随后立即回读 physical identity。"""
    ensure_normal_relative_directory(root, _RECEIPT_FILE.parent,
                                     label="P3-B receipt parent")
    payload_size = encoded_integer_tuple_size(receipt.integer_stream())
    if payload_size > budget.max_receipt_payload_bytes:
        _fail("P3-B receipt payload 超过预算")
    stream = open_exclusive_binary(root, _RECEIPT_FILE, label="P3-B receipt")
    try:
        writer = IntegerFramedStreamWriter.from_open_binary(stream, path=_RECEIPT_FILE)
    except BaseException:
        stream.close()
        raise
    try:
        writer.append(receipt.integer_stream())
        footer = writer.seal()
    except (IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            "P3-B receipt 写入失败") from exc
    finally:
        writer.close()
    if footer.record_count != 1 or footer.total_payload_bytes != payload_size:
        _fail("P3-B receipt footer 与 payload 不一致")
    try:
        physical = sha256_plain_file(
            root, _RECEIPT_FILE, max_bytes=budget.max_receipt_physical_bytes,
            label="P3-B receipt physical")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            "P3-B receipt physical 核验失败") from exc
    return ConversationHeldOutV4ActiveRosterReadbackStreamIdentity(
        _RECEIPT_FILE, footer, physical)


def materialize_v4_active_roster_bidirectional_readback(
        value: ConversationHeldOutV4ActiveRosterReadbackInput,
        ) -> ConversationHeldOutV4ActiveRosterReadbackResult:
    """执行 P3-B 的只读 P2 回读和双向 full-leaf equality，不启动训练。

    输入先完整回读 P3-A、两侧 P2-A/B1/B3 和 P2-C publication；随后只在本次 fresh
    staging/work capability 下做有界 leaf transport 与排序。任一 mismatch 在 P3-B receipt
    前停止，而且不会修改训练、P3-A、P2-A/B1/B3 或 P2-C root。
    """
    if not isinstance(value, ConversationHeldOutV4ActiveRosterReadbackInput):
        raise TypeError("P3-B materialize 必须接收 ConversationHeldOutV4ActiveRosterReadbackInput")
    status = _validate_roots(value)
    try:
        active = revalidate_v4_active_intake(value.active_intake)
        training_catalog = revalidate_v4_provenance_stream_catalog(
            value.training_catalog)
        held_out_catalog = revalidate_v4_provenance_stream_catalog(
            value.held_out_catalog)
        training_b1 = revalidate_v4_provenance_direct_dag_result(
            value.training_b1_result, value.training_b3_work_root,
            catalog=training_catalog)
        held_out_b1 = revalidate_v4_provenance_direct_dag_result(
            value.held_out_b1_result, value.held_out_b3_work_root,
            catalog=held_out_catalog)
        training_b3 = revalidate_v4_provenance_leaf_projections(
            value.training_b3_result, value.training_b3_work_root,
            expected_side=V4_PROVENANCE_SIDE_TRAINING)
        held_out_b3 = revalidate_v4_provenance_leaf_projections(
            value.held_out_b3_result, value.held_out_b3_work_root,
            expected_side=V4_PROVENANCE_SIDE_HELD_OUT)
    except (ConversationHeldOutV4ActiveIntakeError,
            ConversationHeldOutV4ProvenanceScalableError,
            ConversationHeldOutV4ProvenanceScalableDagError,
            ConversationHeldOutV4ProvenanceScalableProjectionError,
            KRunBoundaryError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            "P3-B 上游 P3-A/P2-A/B1/B3 readback 未通过") from exc
    if (training_b3.direct_dag_stable_key != training_b1.stable_key()
            or held_out_b3.direct_dag_stable_key != held_out_b1.stable_key()):
        _fail("P3-B B3 direct DAG stable key 未绑定本次 B1 pair")
    p2_receipt = value.p2_publication.receipt_artifact.receipt
    try:
        reloaded_p2 = load_v4_provenance_cross_side_merge_publication(
            training_b3,
            value.training_b3_work_root,
            held_out_b3,
            value.held_out_b3_work_root,
            value.p2_publication_root,
            code_identity=p2_receipt.code_identity,
            budget=p2_receipt.budget,
            logical_stage_name=p2_receipt.logical_stage_name,
        )
    except (ConversationHeldOutV4ProvenanceCrossSideMergeError,
            KRunBoundaryError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            "P3-B 上游 P2-C publication readback 未通过") from exc
    if reloaded_p2.stable_key() != value.p2_publication.stable_key():
        _fail("P3-B P2-C publication stable key 漂移")
    if (p2_receipt.training_b3_stable_key != training_b3.stable_key()
            or p2_receipt.held_out_b3_stable_key != held_out_b3.stable_key()):
        _fail("P3-B P2-C receipt 未绑定本次 B3 pair")

    active_train, active_held_out = _materialize_active_raw_leaves(value)
    train_source = _materialize_p2_raw_leaves(
        value.training_b3_work_root, training_b3,
        expected_side=V4_PROVENANCE_SIDE_TRAINING, kind="source",
        output_root=value.staging_run_root, output_relative=_RAW_P2_TRAIN_SOURCE,
        value=value, label="P3-B training SourceRef")
    held_source = _materialize_p2_raw_leaves(
        value.held_out_b3_work_root, held_out_b3,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT, kind="source",
        output_root=value.staging_run_root, output_relative=_RAW_P2_HELD_OUT_SOURCE,
        value=value, label="P3-B held-out SourceRef")
    train_content = _materialize_p2_raw_leaves(
        value.training_b3_work_root, training_b3,
        expected_side=V4_PROVENANCE_SIDE_TRAINING, kind="content",
        output_root=value.staging_run_root, output_relative=_RAW_P2_TRAIN_CONTENT,
        value=value, label="P3-B training content")
    held_content = _materialize_p2_raw_leaves(
        value.held_out_b3_work_root, held_out_b3,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT, kind="content",
        output_root=value.staging_run_root, output_relative=_RAW_P2_HELD_OUT_CONTENT,
        value=value, label="P3-B held-out content")
    train_lineage = _materialize_p2_raw_leaves(
        value.training_b3_work_root, training_b3,
        expected_side=V4_PROVENANCE_SIDE_TRAINING, kind="lineage",
        output_root=value.staging_run_root, output_relative=_RAW_P2_TRAIN_LINEAGE,
        value=value, label="P3-B training lineage")
    held_lineage = _materialize_p2_raw_leaves(
        value.held_out_b3_work_root, held_out_b3,
        expected_side=V4_PROVENANCE_SIDE_HELD_OUT, kind="lineage",
        output_root=value.staging_run_root, output_relative=_RAW_P2_HELD_OUT_LINEAGE,
        value=value, label="P3-B held-out lineage")

    sorted_streams = {}
    for name, raw, destination in (
            ("active-train", active_train, _SORTED_ACTIVE_TRAIN),
            ("active-held-out", active_held_out, _SORTED_ACTIVE_HELD_OUT),
            ("p2-training-source", train_source, _SORTED_P2_TRAIN_SOURCE),
            ("p2-held-out-source", held_source, _SORTED_P2_HELD_OUT_SOURCE),
            ("p2-training-content", train_content, _SORTED_P2_TRAIN_CONTENT),
            ("p2-held-out-content", held_content, _SORTED_P2_HELD_OUT_CONTENT),
            ("p2-training-lineage", train_lineage, _SORTED_P2_TRAIN_LINEAGE),
            ("p2-held-out-lineage", held_lineage, _SORTED_P2_HELD_OUT_LINEAGE)):
        sorted_streams[name] = _sort_leaf_stream(
            staging_root=value.staging_run_root,
            work_root=value.work_run_root,
            raw=raw,
            output_relative=destination,
            logical_stage_name=f"{value.logical_stage_name}.{name}",
            budget=value.budget,
            label=f"P3-B {name}")

    train_active_count, train_source_count = _compare_leaf_set(
        sorted_streams["active-train"], sorted_streams["p2-training-source"],
        work_root=value.work_run_root, budget=value.budget,
        label="P3-B training SourceRef", allow_candidate_duplicates=False)
    held_active_count, held_source_count = _compare_leaf_set(
        sorted_streams["active-held-out"], sorted_streams["p2-held-out-source"],
        work_root=value.work_run_root, budget=value.budget,
        label="P3-B held-out SourceRef", allow_candidate_duplicates=False)
    _, train_content_count = _compare_leaf_set(
        sorted_streams["active-train"], sorted_streams["p2-training-content"],
        work_root=value.work_run_root, budget=value.budget,
        label="P3-B training content", allow_candidate_duplicates=False)
    _, held_content_count = _compare_leaf_set(
        sorted_streams["active-held-out"], sorted_streams["p2-held-out-content"],
        work_root=value.work_run_root, budget=value.budget,
        label="P3-B held-out content", allow_candidate_duplicates=False)
    _, train_lineage_raw_count = _compare_leaf_set(
        sorted_streams["active-train"], sorted_streams["p2-training-lineage"],
        work_root=value.work_run_root, budget=value.budget,
        label="P3-B training lineage", allow_candidate_duplicates=True)
    _, held_lineage_raw_count = _compare_leaf_set(
        sorted_streams["active-held-out"], sorted_streams["p2-held-out-lineage"],
        work_root=value.work_run_root, budget=value.budget,
        label="P3-B held-out lineage", allow_candidate_duplicates=True)
    counts = ConversationHeldOutV4ActiveRosterReadbackCounts(
        train_active_count,
        held_active_count,
        train_source_count,
        held_source_count,
        train_content_count,
        held_content_count,
        train_lineage_raw_count,
        held_lineage_raw_count,
        train_active_count,
        held_active_count,
    )
    receipt = ConversationHeldOutV4ActiveRosterReadbackReceipt(
        active.stable_key(),
        training_catalog.stable_key(),
        held_out_catalog.stable_key(),
        training_b1.stable_key(),
        held_out_b1.stable_key(),
        training_b3.stable_key(),
        held_out_b3.stable_key(),
        reloaded_p2.stable_key(),
        p2_receipt.code_identity,
        value.code_identity,
        value.budget,
        counts,
        status,
    )
    receipt_stream = _write_receipt(value.publication_run_root, receipt,
                                    budget=value.budget)
    provisional = ConversationHeldOutV4ActiveRosterReadbackResult(
        value.publication_run_root, receipt, receipt_stream, (0,) * 32)
    manifest_payload = encode_integer_tuple(_manifest_integer_stream(provisional))
    if len(manifest_payload) > value.budget.max_manifest_bytes:
        _fail("P3-B manifest payload 超过预算")
    try:
        publish_manifest_last(
            value.publication_run_root,
            _MANIFEST_FILE,
            manifest_payload,
            frozenset({_RECEIPT_FILE}),
            label="P3-B publication")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            "P3-B manifest-last 未通过") from exc
    result = ConversationHeldOutV4ActiveRosterReadbackResult(
        value.publication_run_root,
        receipt,
        receipt_stream,
        tuple(hashlib.sha256(manifest_payload).digest()),
    )
    return revalidate_v4_active_roster_bidirectional_readback(result)


def revalidate_v4_active_roster_bidirectional_readback(
        value: ConversationHeldOutV4ActiveRosterReadbackResult,
        ) -> ConversationHeldOutV4ActiveRosterReadbackResult:
    """只读重验 P3-B exact two-file evidence publication。"""
    if not isinstance(value, ConversationHeldOutV4ActiveRosterReadbackResult):
        raise TypeError("P3-B revalidate 必须接收 ConversationHeldOutV4ActiveRosterReadbackResult")
    try:
        require_exact_file_closure(
            value.publication_run_root,
            frozenset({_RECEIPT_FILE, _MANIFEST_FILE}),
            label="P3-B publication")
        records = tuple(_iter_verified_p0_records(
            value.publication_run_root,
            _RECEIPT_FILE,
            expected_footer=value.receipt_stream.p0_footer,
            expected_physical=value.receipt_stream.physical,
            max_record_count=1,
            max_record_payload_bytes=value.receipt.budget.max_receipt_payload_bytes,
            max_total_payload_bytes=value.receipt.budget.max_receipt_payload_bytes,
            max_physical_bytes=value.receipt.budget.max_receipt_physical_bytes,
            label="P3-B receipt"))
    except (KRunBoundaryError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            "P3-B receipt publication 回读失败") from exc
    if records != (value.receipt.integer_stream(),):
        _fail("P3-B receipt record 与 result identity 不一致")
    try:
        manifest_identity = capture_plain_file_identity(
            value.publication_run_root, _MANIFEST_FILE, label="P3-B manifest")
        with open_plain_binary(
                value.publication_run_root,
                _MANIFEST_FILE,
                label="P3-B manifest", expected_identity=manifest_identity) as stream:
            manifest_payload = stream.read(value.receipt.budget.max_manifest_bytes + 1)
        require_plain_file_identity(
            value.publication_run_root, _MANIFEST_FILE, manifest_identity,
            label="P3-B manifest post-read")
    except (KRunBoundaryError, OSError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            "P3-B manifest 回读失败") from exc
    if len(manifest_payload) > value.receipt.budget.max_manifest_bytes:
        _fail("P3-B manifest payload 超过预算")
    try:
        decoded = decode_integer_tuple(manifest_payload)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveRosterReadbackError(
            "P3-B manifest integer codec 损坏") from exc
    if (encode_integer_tuple(decoded) != manifest_payload
            or decoded != _manifest_integer_stream(value)):
        _fail("P3-B manifest 与 result identity 不一致")
    if tuple(hashlib.sha256(manifest_payload).digest()) != value.manifest_sha256:
        _fail("P3-B manifest SHA-256 漂移")
    return value


__all__ = [
    "ConversationHeldOutV4ActiveRosterReadbackBudget",
    "ConversationHeldOutV4ActiveRosterReadbackCounts",
    "ConversationHeldOutV4ActiveRosterReadbackError",
    "ConversationHeldOutV4ActiveRosterReadbackInput",
    "ConversationHeldOutV4ActiveRosterReadbackReceipt",
    "ConversationHeldOutV4ActiveRosterReadbackResult",
    "ConversationHeldOutV4ActiveRosterReadbackStreamIdentity",
    "V4_ACTIVE_ROSTER_READBACK_STATUS_COVERAGE_NE",
    "V4_ACTIVE_ROSTER_READBACK_STATUS_TEST_ONLY",
    "materialize_v4_active_roster_bidirectional_readback",
    "revalidate_v4_active_roster_bidirectional_readback",
]
