"""DLG-05 v4 P3-A 的活动摄取边界与全量 roster 绑定。

本模块只把显式、可重放的 PH2 source/Observation/teacher roster 绑定到完整 P1
SourceRef、snapshot、lineage node 和 leaf，并在 K-run capability 内物化 P0 catalog。
它不读取正文、label、private/formal，也不启动训练或写入任何 runtime state。D 盘只能
经 ``KRunRoot(test_transport=True)`` 的测试 transport 进入；该模式永远不会提升 coverage。
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
from pure_integer_ai.experiments.conversation_heldout_v4_provenance import (
    ConversationHeldOutV4LineageNode,
    ConversationHeldOutV4ProvenanceError,
    ConversationHeldOutV4ProvenanceLeaf,
    ConversationHeldOutV4SnapshotIdentity,
    V4_PROVENANCE_SIDE_HELD_OUT,
    V4_PROVENANCE_SIDE_TRAINING,
    V4_PROVENANCE_UPSTREAM_SHA1,
    V4_PROVENANCE_UPSTREAM_SHA256,
)
from pure_integer_ai.experiments.conversation_heldout_v4_provenance_scalable import (
    ConversationHeldOutV4ProvenanceCatalogBudget,
    ConversationHeldOutV4ProvenanceCatalogInput,
    ConversationHeldOutV4ProvenanceScalableError,
    ConversationHeldOutV4ProvenanceStreamCatalog,
    build_v4_provenance_stream_catalog,
    revalidate_v4_provenance_stream_catalog,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import (
    ALLOWED_LICENSE_IDS,
    W_STAGES,
    DatasetContractError,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_dataset_owner_records import (
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_records import (
    ObservationRecord,
    SourceRefRecord,
)
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
from pure_integer_ai.storage.integer_external_sort import (
    IntegerExternalSortBudget,
    IntegerExternalSortBudgetExceeded,
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
    require_disjoint_run_roots,
    require_exact_file_closure,
    require_fresh_empty_run_root,
    require_plain_file_identity,
    sha256_plain_file,
)


V4_ACTIVE_INTAKE_MAPPING_SCHEMA = 1
V4_ACTIVE_INTAKE_BUDGET_SCHEMA = 1
V4_ACTIVE_INTAKE_TEACHER_POLICY_SCHEMA = 1
V4_ACTIVE_INTAKE_STREAM_SCHEMA = 1
V4_ACTIVE_INTAKE_RESULT_SCHEMA = 1
V4_ACTIVE_INTAKE_MANIFEST_SCHEMA = 1
V4_ACTIVE_INTAKE_CONSUMED_IDENTITY_SCHEMA = 1

V4_ACTIVE_INTAKE_RECORD_SOURCE = 1
V4_ACTIVE_INTAKE_RECORD_OBSERVATION = 2
V4_ACTIVE_INTAKE_RECORD_TEACHER = 3

V4_ACTIVE_INTAKE_STATUS_TEST_ONLY = "P3_INTAKE_BOUNDARY_TEST_ONLY"
V4_ACTIVE_INTAKE_STATUS_COVERAGE_NE = "ACTIVE_PROVENANCE_COVERAGE_NE"

_RECORD_KINDS = frozenset({
    V4_ACTIVE_INTAKE_RECORD_SOURCE,
    V4_ACTIVE_INTAKE_RECORD_OBSERVATION,
    V4_ACTIVE_INTAKE_RECORD_TEACHER,
})
_P2_SPLITS = frozenset({"train", "held_out"})
_SHA256_SIZE = hashlib.sha256().digest_size
_MANIFEST_FILE = Path("manifest.pii")
_PUBLIC_NODE_FILE = Path("catalog") / "nodes.pifrs"
_PUBLIC_LEAF_FILE = Path("catalog") / "leaves.pifrs"
_PUBLIC_MAPPING_FILE = Path("mapping") / "records.pifrs"
_RAW_NODE_FILE = Path("raw") / "nodes.pifrs"
_RAW_LEAF_FILE = Path("raw") / "leaves.pifrs"
_RAW_MAPPING_FILE = Path("raw") / "mappings.pifrs"
_SORTED_NODE_FILE = Path("sorted") / "nodes.pifrs"
_SORTED_LEAF_FILE = Path("sorted") / "leaves.pifrs"
_SORTED_MAPPING_RECORD_FILE = Path("sorted") / "mapping-by-record.pifrs"
_SORTED_MAPPING_SOURCE_FILE = Path("sorted") / "mapping-by-source.pifrs"
_SORTED_MAPPING_OBSERVATION_FILE = Path("sorted") / "mapping-by-observation.pifrs"
_SORTED_MAPPING_NODE_FILE = Path("sorted") / "mapping-by-node.pifrs"
_SORTED_MAPPING_CLUSTER_FILE = Path("sorted") / "mapping-by-cluster.pifrs"
_SORTED_MAPPING_LEAF_FILE = Path("sorted") / "mapping-by-leaf.pifrs"


# object-model: exception
class ConversationHeldOutV4ActiveIntakeError(RuntimeError):
    """P3-A roster、来源绑定、K-run 边界或 manifest 机械闭合不成立。"""


def _fail(message: str) -> None:
    """统一产生 fail-closed 的 P3-A 错误。"""
    raise ConversationHeldOutV4ActiveIntakeError(message)


def _require_positive_int(value: int, *, label: str) -> int:
    """拒绝 bool、零和隐式数值，固定 P3-A 的显式资源上限。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正严格整数")
    return value


def _require_nonnegative_int(value: int, *, label: str) -> int:
    """校验计数或容量为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _require_digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """要求完整 SHA-256 使用严格字节整数 tuple，禁止摘要截断。"""
    try:
        strict_integer_tuple(value, label=label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是严格整数 tuple") from exc
    if len(value) != _SHA256_SIZE or any(item < 0 or item > 255 for item in value):
        raise ValueError(f"{label} 必须是完整 SHA-256 字节 tuple")
    return value


def _text_scalars(value: str, *, label: str, allow_empty: bool = False) -> tuple[int, ...]:
    """将元数据文本转为已验证 Unicode scalar，不接触任何正文 payload。"""
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} 必须是{'可空' if allow_empty else '非空'} str")
    result = tuple(ord(character) for character in value)
    try:
        return validate_unicode_scalars(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 含非法 Unicode scalar") from exc


def _text_from_scalars(
        value: tuple[int, ...], *, label: str, allow_empty: bool = False,
        ) -> str:
    """从完整 scalar 恢复元数据文本并拒绝非规范空值。"""
    if not isinstance(value, tuple) or (not allow_empty and not value):
        _fail(f"{label} scalar 不得为空")
    try:
        canonical = validate_unicode_scalars(value)
        text = "".join(chr(item) for item in canonical)
    except (TypeError, ValueError) as exc:
        _fail(f"{label} scalar 非法")
        raise AssertionError from exc
    if not allow_empty and not text:
        _fail(f"{label} scalar 不得为空")
    return text


def _record_key(value: object, *, label: str) -> StableRecordKey:
    """要求 PH2 record 关系保留完整 StableRecordKey，而非字符串索引。"""
    if not isinstance(value, StableRecordKey):
        raise TypeError(f"{label} 必须是 StableRecordKey")
    return value


def _protocol_key(value: object, *, label: str) -> ProtocolKey:
    """要求调用方显式保留完整 ProtocolKey，不从摘要或路径推导。"""
    if not isinstance(value, ProtocolKey):
        raise TypeError(f"{label} 必须是 ProtocolKey")
    return value


def _protocol_from_record_key(value: StableRecordKey) -> ProtocolKey:
    """把完整 PH2 stable key 显式提升为 nonnegative protocol identity。"""
    return ProtocolKey(value.components)


def _side_for_split(value: str) -> int:
    """将 P3-A 明确允许的两种 PH2 split 映射到 P2 provenance side。"""
    if value == "train":
        return V4_PROVENANCE_SIDE_TRAINING
    if value == "held_out":
        return V4_PROVENANCE_SIDE_HELD_OUT
    raise ValueError("P3-A split 只能是 train 或 held_out")


def _upstream_digest_from_record(
        value: SourceRefRecord,
        ) -> tuple[int, tuple[int, ...]]:
    """将 PH2 已验证的带算法 upstream checksum 转为 snapshot 的完整字节字段。"""
    algorithm_text, digest_text = value.upstream_checksum.split(":", 1)
    algorithm = {
        "sha1": V4_PROVENANCE_UPSTREAM_SHA1,
        "sha256": V4_PROVENANCE_UPSTREAM_SHA256,
    }.get(algorithm_text)
    if algorithm is None:
        raise AssertionError("SourceRefRecord 已拒绝未知 upstream checksum 算法")
    return algorithm, tuple(bytes.fromhex(digest_text))


def _local_digest_from_record(value: SourceRefRecord) -> tuple[int, ...]:
    """恢复 source record 的完整 local SHA-256；不计算 record JSON 的替代 hash。"""
    return tuple(bytes.fromhex(value.local_sha256))


def _consumed_input_identity(
        kind: int, record_key: StableRecordKey,
        ) -> ProtocolKey:
    """按 kind 加完整 stable key 构造唯一 leaf 消费身份，拒绝任意复用。"""
    if kind not in _RECORD_KINDS:
        raise ValueError("consumed input identity record kind 未注册")
    return ProtocolKey((
        V4_ACTIVE_INTAKE_CONSUMED_IDENTITY_SCHEMA,
        kind,
        *record_key.components,
    ))


def _validate_common_binding(
        *, source_ref: SourceRef, content_sha256: tuple[int, ...],
        snapshot: ConversationHeldOutV4SnapshotIdentity,
        lineage_node_key: ProtocolKey, split: str,
        source_cluster_key: StableRecordKey, license_partition: str,
        owner_key: ProtocolKey, label: str,
        ) -> None:
    """复核每条输入均携带完整来源本体，绝不从另一个 record 隐式补全。"""
    if not isinstance(source_ref, SourceRef):
        raise TypeError(f"{label} source_ref 必须是 SourceRef")
    _require_digest(content_sha256, label=f"{label} content_sha256")
    if not isinstance(snapshot, ConversationHeldOutV4SnapshotIdentity):
        raise TypeError(f"{label} snapshot 类型错误")
    _protocol_key(lineage_node_key, label=f"{label} lineage_node_key")
    _record_key(source_cluster_key, label=f"{label} source_cluster_key")
    _protocol_key(owner_key, label=f"{label} owner_key")
    if split not in _P2_SPLITS:
        raise ValueError(f"{label} split 只能是 train 或 held_out")
    if license_partition not in ALLOWED_LICENSE_IDS:
        raise ValueError(f"{label} license_partition 未登记")
    if snapshot.source_ref != source_ref:
        raise ValueError(f"{label} snapshot SourceRef 与 binding 不一致")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeTeacherPolicy:
    """活动 intake 唯一允许读取的 teacher owner、阶段和 withdrawal 边界。"""

    permitted_owner_keys: tuple[ProtocolKey, ...]
    allowed_visible_from_stages: tuple[str, ...]
    max_withdrawal_level: int

    def __post_init__(self) -> None:
        """冻结 permit 全集，拒绝隐式 owner、阶段或退场级别默认值。"""
        if (not isinstance(self.permitted_owner_keys, tuple)
                or not self.permitted_owner_keys
                or any(not isinstance(item, ProtocolKey)
                       for item in self.permitted_owner_keys)):
            raise ValueError("teacher policy permitted_owner_keys 必须是非空 ProtocolKey tuple")
        if self.permitted_owner_keys != tuple(sorted(
                self.permitted_owner_keys, key=lambda item: item.components)):
            raise ValueError("teacher policy owner keys 必须按完整 key 排序")
        if len(set(self.permitted_owner_keys)) != len(self.permitted_owner_keys):
            raise ValueError("teacher policy owner keys 不得重复")
        if (not isinstance(self.allowed_visible_from_stages, tuple)
                or not self.allowed_visible_from_stages
                or any(item not in W_STAGES
                       for item in self.allowed_visible_from_stages)):
            raise ValueError("teacher policy allowed stages 必须是非空 W stage tuple")
        canonical_stages = tuple(item for item in W_STAGES
                                 if item in self.allowed_visible_from_stages)
        if self.allowed_visible_from_stages != canonical_stages:
            raise ValueError("teacher policy stages 必须按 W stage 规范顺序且无重复")
        if type(self.max_withdrawal_level) is not int or not 0 <= self.max_withdrawal_level <= 3:
            raise ValueError("teacher policy max_withdrawal_level 必须在 0..3")

    def allows(self, value: TeacherEvidenceRecord) -> bool:
        """只根据冻结 permit 机械判定一条现有 teacher record 能否进入 roster。"""
        if not isinstance(value, TeacherEvidenceRecord):
            return False
        return (
            _protocol_from_record_key(value.owner_key) in self.permitted_owner_keys
            and value.visible_from_stage in self.allowed_visible_from_stages
            and value.withdrawal_level <= self.max_withdrawal_level
        )

    def integer_stream(self) -> tuple[int, ...]:
        """返回 manifest 可绑定的无路径、无 teacher 正文 permit identity。"""
        result = [V4_ACTIVE_INTAKE_TEACHER_POLICY_SCHEMA, len(self.permitted_owner_keys)]
        for item in self.permitted_owner_keys:
            pack_key(result, item.components)
        result.append(len(self.allowed_visible_from_stages))
        for item in self.allowed_visible_from_stages:
            pack_key(result, _text_scalars(item, label="teacher policy stage"))
        result.append(self.max_withdrawal_level)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 permit 规则的规范整数身份。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeBudget:
    """P3-A roster、P0 流、manifest 和外排的全部显式资源预算。"""

    max_lineage_node_count: int
    max_source_record_count: int
    max_observation_record_count: int
    max_teacher_record_count: int
    max_mapping_record_payload_bytes: int
    max_stream_payload_bytes: int
    max_stream_physical_bytes: int
    max_manifest_bytes: int
    catalog_budget: ConversationHeldOutV4ProvenanceCatalogBudget
    external_sort_budget: IntegerExternalSortBudget

    def __post_init__(self) -> None:
        """确保上层 roster 上限与 P2 catalog/external-sort 的容量一致。"""
        for label, value in (
                ("max_lineage_node_count", self.max_lineage_node_count),
                ("max_source_record_count", self.max_source_record_count),
                ("max_observation_record_count", self.max_observation_record_count),
                ("max_teacher_record_count", self.max_teacher_record_count),
                ("max_mapping_record_payload_bytes",
                 self.max_mapping_record_payload_bytes),
                ("max_stream_payload_bytes", self.max_stream_payload_bytes),
                ("max_stream_physical_bytes", self.max_stream_physical_bytes),
                ("max_manifest_bytes", self.max_manifest_bytes)):
            _require_positive_int(value, label=label)
        if not isinstance(self.catalog_budget, ConversationHeldOutV4ProvenanceCatalogBudget):
            raise TypeError("active intake catalog_budget 类型错误")
        if not isinstance(self.external_sort_budget, IntegerExternalSortBudget):
            raise TypeError("active intake external_sort_budget 类型错误")
        total_mapping_count = self.max_mapping_record_count
        if self.catalog_budget.max_total_shards < 2:
            raise ValueError("P3-A catalog 至少需要 node/leaf 两个 shard 预算")
        if self.catalog_budget.max_records_per_stream < max(
                self.max_lineage_node_count, total_mapping_count):
            raise ValueError("P3-A catalog record 预算不足")
        if self.catalog_budget.max_total_payload_bytes_per_stream < self.max_stream_payload_bytes:
            raise ValueError("P3-A catalog payload 预算不得小于 P3-A stream 预算")
        if self.catalog_budget.max_physical_bytes_per_shard < self.max_stream_physical_bytes:
            raise ValueError("P3-A catalog physical 预算不得小于 P3-A stream 预算")
        if self.catalog_budget.max_total_physical_bytes < 2 * self.max_stream_physical_bytes:
            raise ValueError("P3-A catalog total physical 预算不足两个最终 stream")
        if self.external_sort_budget.max_input_file_count < 1:
            raise ValueError("P3-A external sort 必须允许至少一个 input stream")
        if self.external_sort_budget.max_input_record_count < max(
                self.max_lineage_node_count, total_mapping_count):
            raise ValueError("P3-A external sort input record 预算不足")
        if self.external_sort_budget.max_record_payload_bytes < self.max_mapping_record_payload_bytes:
            raise ValueError("P3-A external sort 单 record 预算不足")
        if self.external_sort_budget.max_input_payload_bytes < self.max_stream_payload_bytes:
            raise ValueError("P3-A external sort input payload 预算不足")
        if self.external_sort_budget.max_input_physical_bytes < self.max_stream_physical_bytes:
            raise ValueError("P3-A external sort input physical 预算不足")
        if self.external_sort_budget.max_output_physical_bytes < self.max_stream_physical_bytes:
            raise ValueError("P3-A external sort output physical 预算不足")

    @property
    def max_mapping_record_count(self) -> int:
        """返回 source、Observation 和 teacher 三类 mapping 的合计上限。"""
        return (
            self.max_source_record_count
            + self.max_observation_record_count
            + self.max_teacher_record_count
        )

    def integer_stream(self) -> tuple[int, ...]:
        """返回完整资源预算，供 manifest 与后续 P3-B 回读绑定。"""
        result = [V4_ACTIVE_INTAKE_BUDGET_SCHEMA]
        result.extend((
            self.max_lineage_node_count,
            self.max_source_record_count,
            self.max_observation_record_count,
            self.max_teacher_record_count,
            self.max_mapping_record_payload_bytes,
            self.max_stream_payload_bytes,
            self.max_stream_physical_bytes,
            self.max_manifest_bytes,
        ))
        pack_key(result, self.catalog_budget.integer_stream())
        pack_key(result, self.external_sort_budget.integer_stream())
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回预算的 schema-backed 完整稳定键。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeMapping:
    """一条不含正文的 active record 到完整 provenance leaf 的显式绑定。"""

    record_kind: int
    record_key: StableRecordKey
    source_record_key: StableRecordKey
    observation_key: StableRecordKey | None
    source_ref: SourceRef
    content_sha256: tuple[int, ...]
    snapshot: ConversationHeldOutV4SnapshotIdentity
    lineage_node_key: ProtocolKey
    leaf: ConversationHeldOutV4ProvenanceLeaf
    split: str
    source_cluster_key: StableRecordKey
    license_partition: str
    owner_key: ProtocolKey

    def __post_init__(self) -> None:
        """拒绝由摘要、缺侧 leaf 或不完整来源字段构成的伪 mapping。"""
        if type(self.record_kind) is not int or self.record_kind not in _RECORD_KINDS:
            raise ValueError("active mapping record_kind 未注册")
        _record_key(self.record_key, label="active mapping record_key")
        _record_key(self.source_record_key, label="active mapping source_record_key")
        if self.observation_key is not None:
            _record_key(self.observation_key, label="active mapping observation_key")
        _validate_common_binding(
            source_ref=self.source_ref,
            content_sha256=self.content_sha256,
            snapshot=self.snapshot,
            lineage_node_key=self.lineage_node_key,
            split=self.split,
            source_cluster_key=self.source_cluster_key,
            license_partition=self.license_partition,
            owner_key=self.owner_key,
            label="active mapping",
        )
        if not isinstance(self.leaf, ConversationHeldOutV4ProvenanceLeaf):
            raise TypeError("active mapping leaf 类型错误")
        expected_identity = _consumed_input_identity(
            self.record_kind, self.record_key)
        if (self.leaf.side != _side_for_split(self.split)
                or self.leaf.source_ref != self.source_ref
                or self.leaf.content_sha256 != self.content_sha256
                or self.leaf.lineage_node_key != self.lineage_node_key
                or self.leaf.consumed_input_identity != expected_identity):
            raise ValueError("active mapping leaf 未完整绑定 record/source/snapshot/side")
        if self.record_kind == V4_ACTIVE_INTAKE_RECORD_SOURCE:
            if (self.record_key != self.source_record_key
                    or self.observation_key is not None):
                raise ValueError("source mapping 的 record/source/observation 关系非法")
        elif self.record_kind == V4_ACTIVE_INTAKE_RECORD_OBSERVATION:
            if self.observation_key != self.record_key:
                raise ValueError("Observation mapping 必须以自身 stable key 作为 observation key")
        elif self.observation_key is None:
            raise ValueError("teacher mapping 必须显式引用 Observation stable key")

    def integer_stream(self) -> tuple[int, ...]:
        """返回无正文 mapping 的完整、可逆、版本化 P0 record。"""
        result = [V4_ACTIVE_INTAKE_MAPPING_SCHEMA, self.record_kind]
        for value in (
                self.record_key.components,
                self.source_record_key.components):
            pack_key(result, value)
        if self.observation_key is None:
            result.append(0)
        else:
            result.append(1)
            pack_key(result, self.observation_key.components)
        for value in (
                self.source_ref.stable_key(),
                self.content_sha256,
                self.snapshot.integer_stream(),
                self.lineage_node_key.components,
                self.leaf.integer_stream(),
                _text_scalars(self.split, label="active mapping split"),
                self.source_cluster_key.components,
                _text_scalars(
                    self.license_partition,
                    label="active mapping license_partition"),
                self.owner_key.components):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 mapping 本体身份，不用哈希或 record key 单独替代。"""
        return self.integer_stream()

    @classmethod
    def from_integer_stream(
            cls, values: tuple[int, ...],
            ) -> "ConversationHeldOutV4ActiveIntakeMapping":
        """从 P0 record 重建 mapping，并重新执行所有本体闭合检查。"""
        try:
            reader = IntegerStreamReader(values)
            if reader.read_positive(label="active mapping schema") != V4_ACTIVE_INTAKE_MAPPING_SCHEMA:
                _fail("active mapping schema 不兼容")
            record_kind = reader.read_positive(label="active mapping record_kind")
            record_key = StableRecordKey(reader.read_key(label="active mapping record_key"))
            source_record_key = StableRecordKey(reader.read_key(
                label="active mapping source_record_key"))
            observation_present = reader.read_nonnegative(
                label="active mapping observation_present")
            if observation_present not in {0, 1}:
                _fail("active mapping observation_present 非法")
            observation_key = (
                StableRecordKey(reader.read_key(label="active mapping observation_key"))
                if observation_present else None
            )
            source_ref = SourceRef.from_stable_key(reader.read_key(
                label="active mapping source_ref"))
            content_sha256 = reader.read_key(label="active mapping content_sha256")
            snapshot = ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
                reader.read_key(label="active mapping snapshot"))
            lineage_node_key = ProtocolKey(reader.read_key(
                label="active mapping lineage_node_key"))
            leaf = ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(
                reader.read_key(label="active mapping leaf"))
            split = _text_from_scalars(
                reader.read_key(label="active mapping split"),
                label="active mapping split")
            source_cluster_key = StableRecordKey(reader.read_key(
                label="active mapping source_cluster_key"))
            license_partition = _text_from_scalars(
                reader.read_key(label="active mapping license_partition"),
                label="active mapping license_partition")
            owner_key = ProtocolKey(reader.read_key(label="active mapping owner_key"))
            reader.finish()
            result = cls(
                record_kind,
                record_key,
                source_record_key,
                observation_key,
                source_ref,
                content_sha256,
                snapshot,
                lineage_node_key,
                leaf,
                split,
                source_cluster_key,
                license_partition,
                owner_key,
            )
        except ConversationHeldOutV4ActiveIntakeError:
            raise
        except (ConversationHeldOutV4ProvenanceError, IntegerCodecError,
                DatasetContractError, AssertionError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ActiveIntakeError(
                "active mapping integer stream 损坏") from exc
        if result.integer_stream() != values:
            _fail("active mapping integer stream 不是规范表达")
        return result


def _leaf_for_binding(
        *, kind: int, record_key: StableRecordKey, source_ref: SourceRef,
        content_sha256: tuple[int, ...], lineage_node_key: ProtocolKey,
        split: str,
        ) -> ConversationHeldOutV4ProvenanceLeaf:
    """由已验证的完整 binding 构造唯一 leaf，不允许调用方伪造 consumed identity。"""
    return ConversationHeldOutV4ProvenanceLeaf(
        _side_for_split(split),
        source_ref,
        content_sha256,
        lineage_node_key,
        _consumed_input_identity(kind, record_key),
    )


def _validate_source_snapshot_record(
        value: SourceRefRecord, *, source_ref: SourceRef,
        snapshot: ConversationHeldOutV4SnapshotIdentity, label: str,
        ) -> tuple[int, ...]:
    """将 PH2 SourceRefRecord 与显式 P1 snapshot 逐字段比对，禁止临时推导。"""
    if not isinstance(value, SourceRefRecord):
        raise TypeError(f"{label} record 必须是 SourceRefRecord")
    content_sha256 = _local_digest_from_record(value)
    _validate_common_binding(
        source_ref=source_ref,
        content_sha256=content_sha256,
        snapshot=snapshot,
        lineage_node_key=ProtocolKey((1,)),
        split="train",
        source_cluster_key=value.source_cluster_key,
        license_partition=value.license_id,
        owner_key=ProtocolKey((1,)),
        label=label,
    )
    if snapshot.official_uri_scalars != _text_scalars(
            value.official_url, label=f"{label} official_url"):
        raise ValueError(f"{label} snapshot official URI 与 source record 不一致")
    if snapshot.revision_scalars != _text_scalars(
            value.revision_id, label=f"{label} revision_id", allow_empty=True):
        raise ValueError(f"{label} snapshot revision 与 source record 不一致")
    if snapshot.snapshot_scalars != _text_scalars(
            value.snapshot_id, label=f"{label} snapshot_id", allow_empty=True):
        raise ValueError(f"{label} snapshot id 与 source record 不一致")
    expected_upstream_algorithm, expected_upstream_digest = _upstream_digest_from_record(value)
    if (snapshot.upstream_digest_algorithm != expected_upstream_algorithm
            or snapshot.upstream_digest != expected_upstream_digest):
        raise ValueError(f"{label} snapshot upstream checksum 与 source record 不一致")
    if snapshot.local_sha256 != content_sha256:
        raise ValueError(f"{label} snapshot local SHA-256 与 source record 不一致")
    return content_sha256


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeSourceBinding:
    """一条 PH2 source record 的显式 P1 SourceRef/snapshot/node/owner 绑定。"""

    record: SourceRefRecord
    source_ref: SourceRef
    snapshot: ConversationHeldOutV4SnapshotIdentity
    lineage_node_key: ProtocolKey
    split: str
    owner_key: ProtocolKey

    def __post_init__(self) -> None:
        """要求 source record 的版本、hash、URI 与 supplied snapshot 完整一致。"""
        content_sha256 = _validate_source_snapshot_record(
            self.record,
            source_ref=self.source_ref,
            snapshot=self.snapshot,
            label="active source binding",
        )
        _validate_common_binding(
            source_ref=self.source_ref,
            content_sha256=content_sha256,
            snapshot=self.snapshot,
            lineage_node_key=self.lineage_node_key,
            split=self.split,
            source_cluster_key=self.record.source_cluster_key,
            license_partition=self.record.license_id,
            owner_key=self.owner_key,
            label="active source binding",
        )

    def to_mapping(self) -> ConversationHeldOutV4ActiveIntakeMapping:
        """生成 source 自身的无正文 mapping 和确定性 consumed leaf。"""
        content_sha256 = _local_digest_from_record(self.record)
        record_key = self.record.stable_key
        return ConversationHeldOutV4ActiveIntakeMapping(
            V4_ACTIVE_INTAKE_RECORD_SOURCE,
            record_key,
            record_key,
            None,
            self.source_ref,
            content_sha256,
            self.snapshot,
            self.lineage_node_key,
            _leaf_for_binding(
                kind=V4_ACTIVE_INTAKE_RECORD_SOURCE,
                record_key=record_key,
                source_ref=self.source_ref,
                content_sha256=content_sha256,
                lineage_node_key=self.lineage_node_key,
                split=self.split,
            ),
            self.split,
            self.record.source_cluster_key,
            self.record.license_id,
            self.owner_key,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeObservationBinding:
    """一条 train/held-out Observation 的实际消费内容与完整 provenance 绑定。"""

    record: ObservationRecord
    source_ref: SourceRef
    consumed_content_sha256: tuple[int, ...]
    snapshot: ConversationHeldOutV4SnapshotIdentity
    lineage_node_key: ProtocolKey
    source_cluster_key: StableRecordKey
    owner_key: ProtocolKey

    def __post_init__(self) -> None:
        """要求 Observation 自带的 split/许可与显式实际内容 binding 不发生漂移。"""
        if not isinstance(self.record, ObservationRecord):
            raise TypeError("active Observation binding record 必须是 ObservationRecord")
        _validate_common_binding(
            source_ref=self.source_ref,
            content_sha256=self.consumed_content_sha256,
            snapshot=self.snapshot,
            lineage_node_key=self.lineage_node_key,
            split=self.record.split,
            source_cluster_key=self.source_cluster_key,
            license_partition=self.record.license_partition,
            owner_key=self.owner_key,
            label="active Observation binding",
        )

    def to_mapping(self) -> ConversationHeldOutV4ActiveIntakeMapping:
        """生成 Observation 的无正文 mapping；payload 本体绝不进入 P3 artifact。"""
        record_key = self.record.stable_key
        return ConversationHeldOutV4ActiveIntakeMapping(
            V4_ACTIVE_INTAKE_RECORD_OBSERVATION,
            record_key,
            self.record.source_ref_key,
            record_key,
            self.source_ref,
            self.consumed_content_sha256,
            self.snapshot,
            self.lineage_node_key,
            _leaf_for_binding(
                kind=V4_ACTIVE_INTAKE_RECORD_OBSERVATION,
                record_key=record_key,
                source_ref=self.source_ref,
                content_sha256=self.consumed_content_sha256,
                lineage_node_key=self.lineage_node_key,
                split=self.record.split,
            ),
            self.record.split,
            self.source_cluster_key,
            self.record.license_partition,
            self.owner_key,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeTeacherBinding:
    """一条允许 teacher Evidence 的实际消费内容和完整 provenance 绑定。"""

    record: TeacherEvidenceRecord
    source_ref: SourceRef
    consumed_content_sha256: tuple[int, ...]
    snapshot: ConversationHeldOutV4SnapshotIdentity
    lineage_node_key: ProtocolKey
    split: str
    source_cluster_key: StableRecordKey
    license_partition: str

    def __post_init__(self) -> None:
        """保留 teacher 自有 owner，且只允许 P2 pair 内的显式 split。"""
        if not isinstance(self.record, TeacherEvidenceRecord):
            raise TypeError("active teacher binding record 必须是 TeacherEvidenceRecord")
        _validate_common_binding(
            source_ref=self.source_ref,
            content_sha256=self.consumed_content_sha256,
            snapshot=self.snapshot,
            lineage_node_key=self.lineage_node_key,
            split=self.split,
            source_cluster_key=self.source_cluster_key,
            license_partition=self.license_partition,
            owner_key=_protocol_from_record_key(self.record.owner_key),
            label="active teacher binding",
        )

    def to_mapping(self) -> ConversationHeldOutV4ActiveIntakeMapping:
        """生成 teacher mapping，不序列化 evidence 正文或任何答案内容。"""
        record_key = self.record.stable_key
        return ConversationHeldOutV4ActiveIntakeMapping(
            V4_ACTIVE_INTAKE_RECORD_TEACHER,
            record_key,
            self.record.source_ref_key,
            self.record.observation_key,
            self.source_ref,
            self.consumed_content_sha256,
            self.snapshot,
            self.lineage_node_key,
            _leaf_for_binding(
                kind=V4_ACTIVE_INTAKE_RECORD_TEACHER,
                record_key=record_key,
                source_ref=self.source_ref,
                content_sha256=self.consumed_content_sha256,
                lineage_node_key=self.lineage_node_key,
                split=self.split,
            ),
            self.split,
            self.source_cluster_key,
            self.license_partition,
            _protocol_from_record_key(self.record.owner_key),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeRosterFactories:
    """P3-A 的四个无默认 factory；每次运行由调用方显式提供冻结 roster。"""

    lineage_node_factory: Callable[[], Iterable[ConversationHeldOutV4LineageNode]]
    source_binding_factory: Callable[[], Iterable[ConversationHeldOutV4ActiveIntakeSourceBinding]]
    observation_binding_factory: Callable[[], Iterable[ConversationHeldOutV4ActiveIntakeObservationBinding]]
    teacher_binding_factory: Callable[[], Iterable[ConversationHeldOutV4ActiveIntakeTeacherBinding]]

    def __post_init__(self) -> None:
        """拒绝容器、裸 iterable 或自动发现；factory 必须能由调用方显式重放。"""
        for label, value in (
                ("lineage_node_factory", self.lineage_node_factory),
                ("source_binding_factory", self.source_binding_factory),
                ("observation_binding_factory", self.observation_binding_factory),
                ("teacher_binding_factory", self.teacher_binding_factory)):
            if not callable(value):
                raise TypeError(f"active intake {label} 必须是零参 callable")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeInput:
    """P3-A 单次物化请求，所有 root、身份、factory 与预算均必须显式给出。"""

    staging_run_root: KRunRoot
    work_run_root: KRunRoot
    publication_run_root: KRunRoot
    frozen_roster_manifest_identity: ProtocolKey
    code_identity: ProtocolKey
    teacher_policy: ConversationHeldOutV4ActiveIntakeTeacherPolicy
    roster_factories: ConversationHeldOutV4ActiveIntakeRosterFactories
    budget: ConversationHeldOutV4ActiveIntakeBudget
    logical_stage_name: str

    def __post_init__(self) -> None:
        """冻结 capability 三分离和调用方声明 identity；不接受路径或默认策略。"""
        for label, value in (
                ("staging_run_root", self.staging_run_root),
                ("work_run_root", self.work_run_root),
                ("publication_run_root", self.publication_run_root)):
            if not isinstance(value, KRunRoot):
                raise TypeError(f"active intake {label} 必须是 KRunRoot")
        _protocol_key(
            self.frozen_roster_manifest_identity,
            label="active intake frozen_roster_manifest_identity")
        _protocol_key(self.code_identity, label="active intake code_identity")
        if not isinstance(self.teacher_policy, ConversationHeldOutV4ActiveIntakeTeacherPolicy):
            raise TypeError("active intake teacher_policy 类型错误")
        if not isinstance(self.roster_factories, ConversationHeldOutV4ActiveIntakeRosterFactories):
            raise TypeError("active intake roster_factories 类型错误")
        if not isinstance(self.budget, ConversationHeldOutV4ActiveIntakeBudget):
            raise TypeError("active intake budget 类型错误")
        _require_stage_name(self.logical_stage_name)
        transports = {
            self.staging_run_root.test_transport,
            self.work_run_root.test_transport,
            self.publication_run_root.test_transport,
        }
        if len(transports) != 1:
            raise ValueError("active intake 三个 root 不得混用 production 与 test transport")


def _require_stage_name(value: str) -> str:
    """限制外排 namespace 为稳定 ASCII 标识，不让路径语义进入协议。"""
    if not isinstance(value, str) or not value or len(value) > 48:
        raise ValueError("active intake logical_stage_name 必须是长度受限非空 str")
    if (not value[0].isascii() or not value[0].isalnum()
            or any(not item.isascii() or not (
                item.isalnum() or item in {"-", "_", "."}) for item in value)):
        raise ValueError("active intake logical_stage_name 只能含 ASCII 字母、数字、-、_、.")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeStreamIdentity:
    """publication 内一个 sealed P0 stream 的相对定位、footer 与物理身份。"""

    relative_path: Path
    p0_footer: IntegerFramedStreamFooter
    physical: KRunFileDigest

    def __post_init__(self) -> None:
        """拒绝绝对路径、未封存 footer 或不完整 physical SHA。"""
        if (not isinstance(self.relative_path, Path)
                or self.relative_path.is_absolute() or self.relative_path.drive
                or self.relative_path.root or not self.relative_path.parts
                or ".." in self.relative_path.parts):
            raise ValueError("active intake stream relative_path 必须是非空相对 Path")
        if not isinstance(self.p0_footer, IntegerFramedStreamFooter):
            raise TypeError("active intake stream p0_footer 类型错误")
        if not isinstance(self.physical, KRunFileDigest):
            raise TypeError("active intake stream physical 类型错误")
        if self.physical.byte_count <= 0:
            raise ValueError("active intake stream 物理字节必须为正")

    def integer_stream(self) -> tuple[int, ...]:
        """返回不含 root 的 P0 stream identity，供 manifest 绑定。"""
        path_scalars = _text_scalars(
            self.relative_path.as_posix(), label="active intake stream relative path")
        result = [V4_ACTIVE_INTAKE_STREAM_SCHEMA]
        for value in (
                path_scalars,
                self.p0_footer.integer_tuple(),
                (self.physical.byte_count, *self.physical.sha256)):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 stream descriptor identity。"""
        return self.integer_stream()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeCounts:
    """本次 factory 实际产生并被物化的各类 roster 条目计数。"""

    lineage_node_count: int
    source_record_count: int
    observation_record_count: int
    teacher_record_count: int

    def __post_init__(self) -> None:
        """要求 source/node/Observation 均非空，teacher 可以为零但绝不能为负。"""
        for label, value in (
                ("lineage_node_count", self.lineage_node_count),
                ("source_record_count", self.source_record_count),
                ("observation_record_count", self.observation_record_count),
                ("teacher_record_count", self.teacher_record_count)):
            _require_nonnegative_int(value, label=f"active intake {label}")
        if (self.lineage_node_count == 0 or self.source_record_count == 0
                or self.observation_record_count == 0):
            raise ValueError("active intake 必须至少有一个 node/source/Observation")

    @property
    def mapping_record_count(self) -> int:
        """返回每条 active source/Observation/teacher 一一对应的 leaf/mapping 数。"""
        return (
            self.source_record_count
            + self.observation_record_count
            + self.teacher_record_count
        )

    def integer_stream(self) -> tuple[int, ...]:
        """返回 manifest 可绑定的实际读取计数。"""
        return (
            self.lineage_node_count,
            self.source_record_count,
            self.observation_record_count,
            self.teacher_record_count,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakeResult:
    """P3-A 已发布的 test-only/NE roster binding，不代表训练或来源资格通过。"""

    publication_run_root: KRunRoot
    frozen_roster_manifest_identity: ProtocolKey
    code_identity: ProtocolKey
    teacher_policy: ConversationHeldOutV4ActiveIntakeTeacherPolicy
    budget: ConversationHeldOutV4ActiveIntakeBudget
    catalog: ConversationHeldOutV4ProvenanceStreamCatalog
    mapping_stream: ConversationHeldOutV4ActiveIntakeStreamIdentity
    counts: ConversationHeldOutV4ActiveIntakeCounts
    status: str
    manifest_sha256: tuple[int, ...]

    def __post_init__(self) -> None:
        """确保 result 仅引用 publication root 内 P2 catalog 和无正文 mapping。"""
        if not isinstance(self.publication_run_root, KRunRoot):
            raise TypeError("active intake result publication_run_root 类型错误")
        _protocol_key(
            self.frozen_roster_manifest_identity,
            label="active intake result frozen_roster_manifest_identity")
        _protocol_key(self.code_identity, label="active intake result code_identity")
        if not isinstance(self.teacher_policy, ConversationHeldOutV4ActiveIntakeTeacherPolicy):
            raise TypeError("active intake result teacher_policy 类型错误")
        if not isinstance(self.budget, ConversationHeldOutV4ActiveIntakeBudget):
            raise TypeError("active intake result budget 类型错误")
        if not isinstance(self.catalog, ConversationHeldOutV4ProvenanceStreamCatalog):
            raise TypeError("active intake result catalog 类型错误")
        if self.catalog.root != self.publication_run_root:
            raise ValueError("active intake result catalog 必须绑定 publication root")
        if not isinstance(self.mapping_stream, ConversationHeldOutV4ActiveIntakeStreamIdentity):
            raise TypeError("active intake result mapping_stream 类型错误")
        if self.mapping_stream.relative_path != _PUBLIC_MAPPING_FILE:
            raise ValueError("active intake result mapping stream 路径漂移")
        if not isinstance(self.counts, ConversationHeldOutV4ActiveIntakeCounts):
            raise TypeError("active intake result counts 类型错误")
        if self.status not in {
                V4_ACTIVE_INTAKE_STATUS_TEST_ONLY,
                V4_ACTIVE_INTAKE_STATUS_COVERAGE_NE}:
            raise ValueError("active intake result status 未注册")
        expected_status = (
            V4_ACTIVE_INTAKE_STATUS_TEST_ONLY
            if self.publication_run_root.test_transport
            else V4_ACTIVE_INTAKE_STATUS_COVERAGE_NE
        )
        if self.status != expected_status:
            raise ValueError("active intake result status 与 transport 模式不一致")
        _require_digest(self.manifest_sha256, label="active intake result manifest_sha256")

    def stable_key(self) -> tuple[int, ...]:
        """返回 P3-B 可回读绑定的完整无路径 identity。"""
        result = [V4_ACTIVE_INTAKE_RESULT_SCHEMA]
        for value in (
                self.frozen_roster_manifest_identity.components,
                self.code_identity.components,
                self.teacher_policy.integer_stream(),
                self.budget.integer_stream(),
                self.catalog.stable_key(),
                self.mapping_stream.integer_stream(),
                self.counts.integer_stream(),
                _text_scalars(self.status, label="active intake result status"),
                self.manifest_sha256):
            pack_key(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ActiveIntakePublication:
    """已完整回读的 P3-A publication 与当前无正文 mapping 成员清单。"""

    result: ConversationHeldOutV4ActiveIntakeResult
    mappings: tuple[ConversationHeldOutV4ActiveIntakeMapping, ...]

    def __post_init__(self) -> None:
        """拒绝数量、类型或顺序漂移；物理成员资格由公开 loader 的受限回读保证。"""
        if not isinstance(self.result, ConversationHeldOutV4ActiveIntakeResult):
            raise TypeError("P3-A publication result 类型错误")
        if (not isinstance(self.mappings, tuple)
                or any(not isinstance(item, ConversationHeldOutV4ActiveIntakeMapping)
                       for item in self.mappings)):
            raise TypeError("P3-A publication mappings 类型错误")
        if len(self.mappings) != self.result.counts.mapping_record_count:
            raise ValueError("P3-A publication mapping 数量漂移")
        previous: tuple[int, ...] | None = None
        for mapping in self.mappings:
            current = (mapping.record_kind, *mapping.record_key.components)
            if previous is not None and current <= previous:
                raise ValueError("P3-A publication mapping 顺序或成员重复")
            previous = current


# object-model: resource_owner; representation=protocol; interop=pending
class _BoundedP0Writer:
    """P3-A 单个 raw stream 的唯一 writer owner，流式计数且绝不聚合 roster。"""

    __slots__ = (
        "_budget", "_count", "_label", "_max_count", "_payload_bytes",
        "_relative", "_root", "_writer",
    )

    def __init__(
            self, root: KRunRoot, relative: Path, *, max_count: int,
            budget: ConversationHeldOutV4ActiveIntakeBudget, label: str,
            ) -> None:
        """以 K-boundary 排他句柄创建一个未封存 raw P0 stream。"""
        if not isinstance(root, KRunRoot):
            raise TypeError("P3-A raw writer root 必须是 KRunRoot")
        _require_positive_int(max_count, label=f"{label} max_count")
        if not isinstance(relative, Path):
            raise TypeError(f"{label} relative 必须是 Path")
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
        self._max_count = max_count
        self._payload_bytes = 0
        self._relative = relative
        self._root = root
        self._writer = writer

    def append(self, record: tuple[int, ...]) -> None:
        """按 P0 record 预算逐条追加，越界时在写入前 fail closed。"""
        try:
            strict_integer_tuple(record, label=f"{self._label} record", empty=True)
            payload_bytes = encoded_integer_tuple_size(record)
        except (IntegerCodecError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ActiveIntakeError(
                f"{self._label} record 不是规范整数 tuple") from exc
        if self._count >= self._max_count:
            _fail(f"{self._label} record 数超过预算")
        if payload_bytes > self._budget.max_mapping_record_payload_bytes:
            _fail(f"{self._label} 单 record payload 超过预算")
        if self._payload_bytes + payload_bytes > self._budget.max_stream_payload_bytes:
            _fail(f"{self._label} stream payload 超过预算")
        try:
            self._writer.append(record)
        except (IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ActiveIntakeError(
                f"{self._label} P0 写入失败") from exc
        self._count += 1
        self._payload_bytes += payload_bytes

    def seal(self) -> ConversationHeldOutV4ActiveIntakeStreamIdentity:
        """唯一封存 raw stream，并以 K-boundary 重读完整 physical SHA。"""
        try:
            footer = self._writer.seal()
        except (IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ActiveIntakeError(
                f"{self._label} P0 封存失败") from exc
        if (footer.record_count != self._count
                or footer.total_payload_bytes != self._payload_bytes):
            _fail(f"{self._label} P0 footer 与流式计数不一致")
        try:
            physical = sha256_plain_file(
                self._root,
                self._relative,
                max_bytes=self._budget.max_stream_physical_bytes,
                label=f"{self._label} physical",
            )
        except KRunBoundaryError as exc:
            raise ConversationHeldOutV4ActiveIntakeError(
                f"{self._label} P0 physical 身份不可核验") from exc
        return ConversationHeldOutV4ActiveIntakeStreamIdentity(
            self._relative, footer, physical)

    def close(self) -> None:
        """释放未成功封存时的唯一 writer handle，不清理任何残片。"""
        self._writer.close()


def _iter_p0_records(
        root: KRunRoot, relative: Path, *,
        max_record_count: int, max_record_payload_bytes: int,
        max_stream_payload_bytes: int, label: str,
        ) -> Iterator[tuple[int, ...]]:
    """经 K-boundary handle 流式读取一个 sealed P0 stream，绝不以路径裸打开。"""
    try:
        identity = capture_plain_file_identity(root, relative, label=label)
        with open_plain_binary(
                root, relative, label=label, expected_identity=identity) as stream:
            with IntegerFramedStreamReader.from_open_binary(
                    stream,
                    path=relative,
                    max_frame_bytes=max_record_payload_bytes,
                    max_record_count=max_record_count,
                    max_total_payload_bytes=max_stream_payload_bytes,
            ) as reader:
                for record in reader:
                    yield record
                reader.finish()
        require_plain_file_identity(root, relative, identity, label=label)
    except ConversationHeldOutV4ActiveIntakeError:
        raise
    except (KRunBoundaryError, IntegerFramedStreamError, OSError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"{label} 未通过 sealed P0 流复核") from exc


def _read_p0_stream_identity(
        root: KRunRoot, relative: Path, *, max_record_count: int,
        budget: ConversationHeldOutV4ActiveIntakeBudget, label: str,
        ) -> ConversationHeldOutV4ActiveIntakeStreamIdentity:
    """完整消费 P0 stream 并同时回收 footer/physical identity，供 manifest 反向核验。"""
    try:
        file_identity = capture_plain_file_identity(root, relative, label=label)
        with open_plain_binary(
                root, relative, label=label, expected_identity=file_identity) as stream:
            with IntegerFramedStreamReader.from_open_binary(
                    stream,
                    path=relative,
                    max_frame_bytes=budget.max_mapping_record_payload_bytes,
                    max_record_count=max_record_count,
                    max_total_payload_bytes=budget.max_stream_payload_bytes,
            ) as reader:
                record_count = 0
                for _record in reader:
                    record_count += 1
                footer = reader.finish()
        require_plain_file_identity(root, relative, file_identity, label=label)
        if footer.record_count != record_count:
            _fail(f"{label} P0 footer record count 漂移")
        physical = sha256_plain_file(
            root,
            relative,
            max_bytes=budget.max_stream_physical_bytes,
            label=f"{label} physical")
    except ConversationHeldOutV4ActiveIntakeError:
        raise
    except (KRunBoundaryError, IntegerFramedStreamError, OSError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"{label} P0 stream identity 回读失败") from exc
    return ConversationHeldOutV4ActiveIntakeStreamIdentity(relative, footer, physical)


def _mapping_from_record(
        value: tuple[int, ...], *, label: str,
        ) -> ConversationHeldOutV4ActiveIntakeMapping:
    """统一把 P0 mapping decode 错误映射为 P3-A fail-closed 错误。"""
    try:
        return ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(value)
    except ConversationHeldOutV4ActiveIntakeError as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"{label} active mapping 非法") from exc


def _node_from_record(
        value: tuple[int, ...], *, label: str,
        ) -> ConversationHeldOutV4LineageNode:
    """统一解码 P0 node；不允许 mapping 以 node 摘要代替本体。"""
    try:
        return ConversationHeldOutV4LineageNode.from_integer_stream(value)
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"{label} lineage node 非法") from exc


def _leaf_from_record(
        value: tuple[int, ...], *, label: str,
        ) -> ConversationHeldOutV4ProvenanceLeaf:
    """统一解码 P0 leaf；不允许未完整来源的 leaf 继续进入 catalog。"""
    try:
        return ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(value)
    except (ConversationHeldOutV4ProvenanceError, IntegerCodecError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"{label} provenance leaf 非法") from exc


def _packed_sort_key(*values: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长度整数键构造无歧义 external-sort key。"""
    result: list[int] = []
    for value in values:
        try:
            strict_integer_tuple(value, label="active intake sort key")
        except (TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ActiveIntakeError(
                "active intake sort key 非法") from exc
        pack_key(result, value)
    return tuple(result)


def _mapping_record_sort_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """按 kind/record key 排序，供唯一记录 roster 回读。"""
    mapping = _mapping_from_record(value, label="mapping record sort")
    return _packed_sort_key((mapping.record_kind,), mapping.record_key.components)


def _mapping_source_sort_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """按 source record/key kind 排序，供 source-to-input 流式 join。"""
    mapping = _mapping_from_record(value, label="mapping source sort")
    return _packed_sort_key(
        mapping.source_record_key.components,
        (mapping.record_kind,),
        mapping.record_key.components,
    )


def _mapping_observation_sort_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """把 source 放在独立空组，其余按 Observation/key kind 排序。"""
    mapping = _mapping_from_record(value, label="mapping observation sort")
    if mapping.observation_key is None:
        return (0,)
    return _packed_sort_key(
        (1,),
        mapping.observation_key.components,
        (mapping.record_kind,),
        mapping.record_key.components,
    )


def _mapping_node_sort_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """以实际 ProtocolKey 顺序排序，便于与 node P0 stream merge join。"""
    mapping = _mapping_from_record(value, label="mapping node sort")
    # component 加一后用 0 终止，既保留可变长 ProtocolKey 的词典序，也避免
    # ``(1,) + kind`` 与 ``(1, kind, ...)`` 互相穿插。
    return (
        *(item + 1 for item in mapping.lineage_node_key.components),
        0,
        mapping.record_kind,
        *mapping.record_key.components,
    )


def _mapping_cluster_sort_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """按完整 source cluster 排序，检测不同 P2 side 的隐性复用。"""
    mapping = _mapping_from_record(value, label="mapping cluster sort")
    return _packed_sort_key(
        mapping.source_cluster_key.components,
        (mapping.record_kind,),
        mapping.record_key.components,
    )


def _mapping_leaf_sort_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """以 leaf 完整 stable key 排序，供 mapping/P0 leaf 一一 merge join。"""
    mapping = _mapping_from_record(value, label="mapping leaf sort")
    return mapping.leaf.stable_key()


def _node_sort_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """以实际 node key 排序，满足 P2-A catalog 的严格递增合同。"""
    return _node_from_record(value, label="node sort").node_key.components


def _leaf_sort_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """以完整 leaf key 排序，满足 P2-A catalog 的严格递增合同。"""
    return _leaf_from_record(value, label="leaf sort").stable_key()


def _sort_raw_stream(
        *, staging_root: KRunRoot, work_root: KRunRoot,
        raw_relative: Path, output_relative: Path, logical_stage_name: str,
        sort_key: Callable[[tuple[int, ...]], tuple[int, ...]],
        budget: ConversationHeldOutV4ActiveIntakeBudget, label: str,
        ) -> ConversationHeldOutV4ActiveIntakeStreamIdentity:
    """以现役 K external sort 排序一条 raw P0 stream，并保留完整输出 identity。"""
    try:
        result: IntegerExternalSortResult = external_sort_sealed_integer_records(
            staging_root,
            (raw_relative,),
            work_root,
            output_relative_path=output_relative,
            logical_stage_name=logical_stage_name,
            sort_key=sort_key,
            budget=budget.external_sort_budget,
        )
    except ConversationHeldOutV4ActiveIntakeError:
        raise
    except (IntegerExternalSortBudgetExceeded, IntegerExternalSortError,
            KRunBoundaryError, IntegerFramedStreamError, OSError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"{label} external sort 未通过") from exc
    return ConversationHeldOutV4ActiveIntakeStreamIdentity(
        output_relative,
        result.identity.output_footer,
        result.identity.output_physical,
    )


def _same_source_binding(
        value: ConversationHeldOutV4ActiveIntakeMapping,
        source: ConversationHeldOutV4ActiveIntakeMapping,
        *, label: str,
        ) -> None:
    """要求 source record 的所有派生输入都逐字段回指同一来源本体与 split。"""
    if value.source_record_key != source.record_key:
        _fail(f"{label} source record key 不一致")
    for field in (
            "source_ref", "snapshot", "lineage_node_key",
            "split", "source_cluster_key", "license_partition"):
        if getattr(value, field) != getattr(source, field):
            _fail(f"{label} 与 source record 的 {field} 漂移")


def _same_observation_binding(
        value: ConversationHeldOutV4ActiveIntakeMapping,
        observation: ConversationHeldOutV4ActiveIntakeMapping,
        *, label: str,
        ) -> None:
    """要求 teacher 完整回指唯一 Observation 的来源、内容、side 与 cluster。"""
    if value.observation_key != observation.record_key:
        _fail(f"{label} Observation key 不一致")
    for field in (
            "source_record_key", "source_ref", "snapshot",
            "lineage_node_key", "split", "source_cluster_key",
            "license_partition"):
        if getattr(value, field) != getattr(observation, field):
            _fail(f"{label} 与 Observation 的 {field} 漂移")


def _validate_mapping_record_order(
        root: KRunRoot, relative: Path, *,
        counts: ConversationHeldOutV4ActiveIntakeCounts,
        budget: ConversationHeldOutV4ActiveIntakeBudget,
        ) -> None:
    """重放 kind/record-order stream，拒绝漏项、重复项和 sort 输出漂移。"""
    expected = {
        V4_ACTIVE_INTAKE_RECORD_SOURCE: counts.source_record_count,
        V4_ACTIVE_INTAKE_RECORD_OBSERVATION: counts.observation_record_count,
        V4_ACTIVE_INTAKE_RECORD_TEACHER: counts.teacher_record_count,
    }
    actual = {kind: 0 for kind in _RECORD_KINDS}
    previous: tuple[int, ...] | None = None
    for index, record in enumerate(_iter_p0_records(
            root, relative,
            max_record_count=counts.mapping_record_count,
            max_record_payload_bytes=budget.max_mapping_record_payload_bytes,
            max_stream_payload_bytes=budget.max_stream_payload_bytes,
            label="P3-A mapping-by-record")):
        mapping = _mapping_from_record(record, label=f"mapping-by-record[{index}]")
        key = _mapping_record_sort_key(record)
        if previous is not None and key <= previous:
            _fail("P3-A mapping record key 不严格递增或重复")
        previous = key
        actual[mapping.record_kind] += 1
    if actual != expected:
        _fail("P3-A mapping record 类型计数与 factory 读取计数不一致")


def _validate_source_join(
        root: KRunRoot, relative: Path, *,
        counts: ConversationHeldOutV4ActiveIntakeCounts,
        budget: ConversationHeldOutV4ActiveIntakeBudget,
        ) -> None:
    """以 source record key 流式 merge group 验证所有 active record 的来源绑定。"""
    current_key: StableRecordKey | None = None
    source: ConversationHeldOutV4ActiveIntakeMapping | None = None
    group_count = 0
    group_index = 0
    previous_key: tuple[int, ...] | None = None

    def finish_group() -> None:
        """拒绝没有唯一 source mapping 的 observation/teacher group。"""
        nonlocal source, group_count, group_index
        if current_key is not None and source is None:
            _fail("P3-A source join group 缺 source mapping")
        if source is not None and source.record_kind != V4_ACTIVE_INTAKE_RECORD_SOURCE:
            _fail("P3-A source join group 首项不是 source mapping")
        source = None
        group_count = 0
        group_index += 1

    for index, record in enumerate(_iter_p0_records(
            root, relative,
            max_record_count=counts.mapping_record_count,
            max_record_payload_bytes=budget.max_mapping_record_payload_bytes,
            max_stream_payload_bytes=budget.max_stream_payload_bytes,
            label="P3-A mapping-by-source")):
        mapping = _mapping_from_record(record, label=f"mapping-by-source[{index}]")
        key = _mapping_source_sort_key(record)
        if previous_key is not None and key < previous_key:
            _fail("P3-A source join sort key 倒退")
        previous_key = key
        if current_key != mapping.source_record_key:
            if current_key is not None:
                finish_group()
            current_key = mapping.source_record_key
        if group_count == 0:
            if mapping.record_kind != V4_ACTIVE_INTAKE_RECORD_SOURCE:
                _fail("P3-A source join group 未以 source mapping 开始")
            source = mapping
        elif source is None:
            _fail("P3-A source join 丢失 source mapping")
        elif mapping.record_kind == V4_ACTIVE_INTAKE_RECORD_SOURCE:
            _fail("P3-A source join 同一 source record 重复 mapping")
        else:
            _same_source_binding(mapping, source, label="P3-A source join")
        group_count += 1
    if current_key is not None:
        finish_group()
    if group_index != counts.source_record_count:
        _fail("P3-A source join group 数与 source factory 读取计数不一致")


def _validate_observation_join(
        root: KRunRoot, relative: Path, *,
        counts: ConversationHeldOutV4ActiveIntakeCounts,
        budget: ConversationHeldOutV4ActiveIntakeBudget,
        ) -> None:
    """以 Observation key 流式验证每条 teacher 都有唯一且一致的 Observation。"""
    current_key: StableRecordKey | None = None
    observation: ConversationHeldOutV4ActiveIntakeMapping | None = None
    group_count = 0
    group_index = 0
    previous_key: tuple[int, ...] | None = None

    def finish_group() -> None:
        """拒绝 teacher 指向不存在或重复的 Observation mapping。"""
        nonlocal observation, group_count, group_index
        if current_key is not None and observation is None:
            _fail("P3-A Observation join group 缺 Observation mapping")
        observation = None
        group_count = 0
        group_index += 1

    for index, record in enumerate(_iter_p0_records(
            root, relative,
            max_record_count=counts.mapping_record_count,
            max_record_payload_bytes=budget.max_mapping_record_payload_bytes,
            max_stream_payload_bytes=budget.max_stream_payload_bytes,
            label="P3-A mapping-by-observation")):
        mapping = _mapping_from_record(record, label=f"mapping-by-observation[{index}]")
        if mapping.observation_key is None:
            if mapping.record_kind != V4_ACTIVE_INTAKE_RECORD_SOURCE:
                _fail("P3-A 非 source mapping 缺 Observation key")
            continue
        key = _mapping_observation_sort_key(record)
        if previous_key is not None and key < previous_key:
            _fail("P3-A Observation join sort key 倒退")
        previous_key = key
        if current_key != mapping.observation_key:
            if current_key is not None:
                finish_group()
            current_key = mapping.observation_key
        if group_count == 0:
            if mapping.record_kind != V4_ACTIVE_INTAKE_RECORD_OBSERVATION:
                _fail("P3-A Observation join group 未以 Observation mapping 开始")
            observation = mapping
        elif observation is None:
            _fail("P3-A Observation join 丢失 Observation mapping")
        elif mapping.record_kind == V4_ACTIVE_INTAKE_RECORD_OBSERVATION:
            _fail("P3-A Observation join 同一 key 重复 Observation mapping")
        else:
            _same_observation_binding(mapping, observation, label="P3-A Observation join")
        group_count += 1
    if current_key is not None:
        finish_group()
    if group_index != counts.observation_record_count:
        _fail("P3-A Observation join group 数与 Observation factory 读取计数不一致")


def _validate_cluster_split_join(
        root: KRunRoot, relative: Path, *,
        counts: ConversationHeldOutV4ActiveIntakeCounts,
        budget: ConversationHeldOutV4ActiveIntakeBudget,
        ) -> None:
    """按完整来源 cluster 流式检查 split，阻断 train/held-out 之间的 cluster 泄漏。"""
    current_key: StableRecordKey | None = None
    current_split: str | None = None
    previous_key: tuple[int, ...] | None = None
    for index, record in enumerate(_iter_p0_records(
            root, relative,
            max_record_count=counts.mapping_record_count,
            max_record_payload_bytes=budget.max_mapping_record_payload_bytes,
            max_stream_payload_bytes=budget.max_stream_payload_bytes,
            label="P3-A mapping-by-cluster")):
        mapping = _mapping_from_record(record, label=f"mapping-by-cluster[{index}]")
        key = _mapping_cluster_sort_key(record)
        if previous_key is not None and key < previous_key:
            _fail("P3-A cluster join sort key 倒退")
        previous_key = key
        if current_key != mapping.source_cluster_key:
            current_key = mapping.source_cluster_key
            current_split = mapping.split
        elif current_split != mapping.split:
            _fail("P3-A source cluster 跨 train/held_out split")


def _validate_node_join(
        *, work_root: KRunRoot, sorted_node_relative: Path,
        sorted_mapping_relative: Path,
        counts: ConversationHeldOutV4ActiveIntakeCounts,
        budget: ConversationHeldOutV4ActiveIntakeBudget,
        ) -> None:
    """将 mapping-by-node 与按 node key 排序的 P0 node 流 merge join，拒绝孤儿 node 引用。"""
    nodes = iter(_iter_p0_records(
        work_root,
        sorted_node_relative,
        max_record_count=counts.lineage_node_count,
        max_record_payload_bytes=budget.max_mapping_record_payload_bytes,
        max_stream_payload_bytes=budget.max_stream_payload_bytes,
        label="P3-A sorted nodes"))
    current_node: ConversationHeldOutV4LineageNode | None = None
    previous_node_key: tuple[int, ...] | None = None
    node_count = 0

    def advance_node() -> ConversationHeldOutV4LineageNode | None:
        """读取下一 node 并同时拒绝 catalog 前的重复或排序漂移。"""
        nonlocal previous_node_key, node_count
        try:
            raw = next(nodes)
        except StopIteration:
            return None
        node = _node_from_record(raw, label="P3-A sorted node")
        if previous_node_key is not None and node.node_key.components <= previous_node_key:
            _fail("P3-A sorted node key 不严格递增或重复")
        previous_node_key = node.node_key.components
        node_count += 1
        return node

    current_node = advance_node()
    previous_mapping_key: tuple[int, ...] | None = None
    mapping_count = 0
    for index, raw in enumerate(_iter_p0_records(
            work_root,
            sorted_mapping_relative,
            max_record_count=counts.mapping_record_count,
            max_record_payload_bytes=budget.max_mapping_record_payload_bytes,
            max_stream_payload_bytes=budget.max_stream_payload_bytes,
            label="P3-A mapping-by-node")):
        mapping = _mapping_from_record(raw, label=f"mapping-by-node[{index}]")
        mapping_key = mapping.lineage_node_key.components
        if previous_mapping_key is not None and mapping_key < previous_mapping_key:
            _fail("P3-A mapping node key 倒退")
        previous_mapping_key = mapping_key
        while (current_node is not None
               and current_node.node_key.components < mapping_key):
            current_node = advance_node()
        if current_node is None or current_node.node_key.components != mapping_key:
            _fail("P3-A mapping 指向未登记 lineage node")
        if (current_node.snapshot != mapping.snapshot
                or current_node.snapshot.source_ref != mapping.source_ref):
            _fail("P3-A mapping 与 lineage node snapshot/SourceRef 不一致")
        mapping_count += 1
    while current_node is not None:
        current_node = advance_node()
    if node_count != counts.lineage_node_count:
        _fail("P3-A sorted node 计数漂移")
    if mapping_count != counts.mapping_record_count:
        _fail("P3-A mapping-by-node 计数漂移")


def _validate_leaf_join(
        *, work_root: KRunRoot, sorted_leaf_relative: Path,
        sorted_mapping_relative: Path,
        counts: ConversationHeldOutV4ActiveIntakeCounts,
        budget: ConversationHeldOutV4ActiveIntakeBudget,
        ) -> None:
    """将 mapping-by-leaf 和最终 leaf P0 stream 一一 merge join，拒绝漏 leaf 或额外 leaf。"""
    leaves = iter(_iter_p0_records(
        work_root,
        sorted_leaf_relative,
        max_record_count=counts.mapping_record_count,
        max_record_payload_bytes=budget.max_mapping_record_payload_bytes,
        max_stream_payload_bytes=budget.max_stream_payload_bytes,
        label="P3-A sorted leaves"))
    previous_leaf_key: tuple[int, ...] | None = None
    index = 0
    for mapping_raw in _iter_p0_records(
            work_root,
            sorted_mapping_relative,
            max_record_count=counts.mapping_record_count,
            max_record_payload_bytes=budget.max_mapping_record_payload_bytes,
            max_stream_payload_bytes=budget.max_stream_payload_bytes,
            label="P3-A mapping-by-leaf"):
        mapping = _mapping_from_record(mapping_raw, label=f"mapping-by-leaf[{index}]")
        try:
            leaf_raw = next(leaves)
        except StopIteration:
            _fail("P3-A leaf stream 缺少 mapping 对应 leaf")
        leaf = _leaf_from_record(leaf_raw, label=f"sorted leaf[{index}]")
        leaf_key = leaf.stable_key()
        if previous_leaf_key is not None and leaf_key <= previous_leaf_key:
            _fail("P3-A sorted leaf key 不严格递增或重复")
        previous_leaf_key = leaf_key
        if leaf != mapping.leaf:
            _fail("P3-A mapping leaf 与 P0 leaf stream 不一致")
        index += 1
    try:
        next(leaves)
    except StopIteration:
        pass
    else:
        _fail("P3-A leaf stream 含未登记 mapping 的额外 leaf")
    if index != counts.mapping_record_count:
        _fail("P3-A leaf/mapping join 计数漂移")


def _copy_p0_stream(
        *, source_root: KRunRoot, source_relative: Path,
        publication_root: KRunRoot, publication_relative: Path,
        max_record_count: int,
        budget: ConversationHeldOutV4ActiveIntakeBudget,
        label: str,
        ) -> ConversationHeldOutV4ActiveIntakeStreamIdentity:
    """经已验证 P0 reader/writer 把 work 输出重放到无临时文件的 publication root。"""
    parent = publication_relative.parent
    if parent.parts:
        ensure_normal_relative_directory(
            publication_root, parent, label=f"{label} publication parent")
    source_identity = capture_plain_file_identity(
        source_root, source_relative, label=f"{label} source")
    input_footer: IntegerFramedStreamFooter | None = None
    writer: IntegerFramedStreamWriter | None = None
    try:
        with open_plain_binary(
                source_root,
                source_relative,
                label=f"{label} source",
                expected_identity=source_identity) as source_stream:
            with IntegerFramedStreamReader.from_open_binary(
                    source_stream,
                    path=source_relative,
                    max_frame_bytes=budget.max_mapping_record_payload_bytes,
                    max_record_count=max_record_count,
                    max_total_payload_bytes=budget.max_stream_payload_bytes,
            ) as reader:
                target_stream = open_exclusive_binary(
                    publication_root,
                    publication_relative,
                    label=f"{label} publication")
                try:
                    writer = IntegerFramedStreamWriter.from_open_binary(
                        target_stream, path=publication_relative)
                except BaseException:
                    target_stream.close()
                    raise
                for record in reader:
                    writer.append(record)
                input_footer = reader.finish()
                output_footer = writer.seal()
                writer = None
    except (KRunBoundaryError, IntegerFramedStreamError, OSError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"{label} P0 publication copy 失败") from exc
    finally:
        if writer is not None:
            writer.close()
    if input_footer is None:
        raise AssertionError("P3-A P0 copy 缺 input footer")
    if output_footer != input_footer:
        _fail(f"{label} P0 publication copy footer 漂移")
    require_plain_file_identity(
        source_root, source_relative, source_identity, label=f"{label} source")
    try:
        physical = sha256_plain_file(
            publication_root,
            publication_relative,
            max_bytes=budget.max_stream_physical_bytes,
            label=f"{label} publication physical")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"{label} publication physical 身份不可核验") from exc
    return ConversationHeldOutV4ActiveIntakeStreamIdentity(
        publication_relative, output_footer, physical)


def _read_manifest_payload(
        root: KRunRoot, *, budget: ConversationHeldOutV4ActiveIntakeBudget,
        ) -> bytes:
    """有界回读小型纯整数 manifest，拒绝裸 ``Path.read_bytes`` 或路径替换。"""
    try:
        identity = capture_plain_file_identity(root, _MANIFEST_FILE, label="P3-A manifest")
        if identity.byte_count > budget.max_manifest_bytes:
            _fail("P3-A manifest 超过预算")
        with open_plain_binary(
                root,
                _MANIFEST_FILE,
                label="P3-A manifest",
                expected_identity=identity) as stream:
            payload = stream.read(budget.max_manifest_bytes + 1)
        require_plain_file_identity(root, _MANIFEST_FILE, identity, label="P3-A manifest")
    except ConversationHeldOutV4ActiveIntakeError:
        raise
    except (KRunBoundaryError, OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            "P3-A manifest 无法经 K boundary 回读") from exc
    if not isinstance(payload, bytes) or not payload or len(payload) > budget.max_manifest_bytes:
        _fail("P3-A manifest payload 非法或超过预算")
    return payload


def _manifest_integer_stream(
        result: ConversationHeldOutV4ActiveIntakeResult,
        ) -> tuple[int, ...]:
    """构造 manifest 的唯一 canonical 整数闭包，manifest 本身不含自摘要。"""
    result.catalog.root  # 让类型与 root 绑定在调用时先被访问，避免脱离 capability 误用。
    node_stream = result.catalog.node_streams
    leaf_stream = result.catalog.leaf_streams
    if len(node_stream) != 1 or len(leaf_stream) != 1:
        _fail("P3-A publication catalog 必须恰有一个 node 和一个 leaf stream")
    result_node = ConversationHeldOutV4ActiveIntakeStreamIdentity(
        node_stream[0].relative_path,
        node_stream[0].p0_footer,
        KRunFileDigest(node_stream[0].physical_byte_count, node_stream[0].physical_sha256),
    )
    result_leaf = ConversationHeldOutV4ActiveIntakeStreamIdentity(
        leaf_stream[0].relative_path,
        leaf_stream[0].p0_footer,
        KRunFileDigest(leaf_stream[0].physical_byte_count, leaf_stream[0].physical_sha256),
    )
    status_code = (
        1 if result.status == V4_ACTIVE_INTAKE_STATUS_TEST_ONLY else 2)
    values = [V4_ACTIVE_INTAKE_MANIFEST_SCHEMA, status_code]
    for item in (
            result.frozen_roster_manifest_identity.components,
            result.code_identity.components,
            result.teacher_policy.integer_stream(),
            result.budget.integer_stream(),
            result.counts.integer_stream(),
            result.catalog.stable_key(),
            result_node.integer_stream(),
            result_leaf.integer_stream(),
            result.mapping_stream.integer_stream()):
        pack_key(values, item)
    return tuple(values)


def _iter_factory(
        factory: Callable[[], Iterable[object]], *, label: str,
        ) -> Iterable[object]:
    """调用一个显式 roster factory，拒绝失败、None 或不可迭代返回值。"""
    try:
        result = factory()
        iterator = iter(result)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            f"P3-A {label} factory 不可重放") from exc
    return iterator


def _append_mapping(
        mapping: ConversationHeldOutV4ActiveIntakeMapping, *,
        leaf_writer: _BoundedP0Writer, mapping_writer: _BoundedP0Writer,
        ) -> None:
    """把同一 canonical mapping 同步物化为 P2 leaf 与无正文 mapping 两条 P0 stream。"""
    leaf_writer.append(mapping.leaf.integer_stream())
    mapping_writer.append(mapping.integer_stream())


def materialize_v4_active_intake(
        value: ConversationHeldOutV4ActiveIntakeInput,
        ) -> ConversationHeldOutV4ActiveIntakeResult:
    """执行 P3-A 的唯一零训练写入活动 intake materialization。

    入口先要求三个当次新建且不重叠的 K-run capability；随后只消费调用方给出的四个
    roster factory，向 staging 写 raw P0、在 work 做有界 external sort/merge join，并把
    最终 node/leaf/mapping 重放到 exact publication closure。它从不调用 trainer、runtime、
    candidate、memory、companion、private/formal 或外部模型。
    """
    if not isinstance(value, ConversationHeldOutV4ActiveIntakeInput):
        raise TypeError("P3-A materialize 必须接收 ConversationHeldOutV4ActiveIntakeInput")
    try:
        require_disjoint_run_roots(
            value.staging_run_root,
            value.work_run_root,
            label="P3-A staging/work roots")
        require_disjoint_run_roots(
            value.staging_run_root,
            value.publication_run_root,
            label="P3-A staging/publication roots")
        require_disjoint_run_roots(
            value.work_run_root,
            value.publication_run_root,
            label="P3-A work/publication roots")
        for label, root in (
                ("P3-A staging root", value.staging_run_root),
                ("P3-A work root", value.work_run_root),
                ("P3-A publication root", value.publication_run_root)):
            require_fresh_empty_run_root(root, label=label)
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            "P3-A K-run root capability 未通过新建/隔离边界") from exc

    node_writer = _BoundedP0Writer(
        value.staging_run_root,
        _RAW_NODE_FILE,
        max_count=value.budget.max_lineage_node_count,
        budget=value.budget,
        label="P3-A raw nodes")
    leaf_writer = _BoundedP0Writer(
        value.staging_run_root,
        _RAW_LEAF_FILE,
        max_count=value.budget.max_mapping_record_count,
        budget=value.budget,
        label="P3-A raw leaves")
    mapping_writer = _BoundedP0Writer(
        value.staging_run_root,
        _RAW_MAPPING_FILE,
        max_count=value.budget.max_mapping_record_count,
        budget=value.budget,
        label="P3-A raw mappings")
    node_count = 0
    source_count = 0
    observation_count = 0
    teacher_count = 0
    try:
        for index, item in enumerate(_iter_factory(
                value.roster_factories.lineage_node_factory,
                label="lineage node")):
            if not isinstance(item, ConversationHeldOutV4LineageNode):
                _fail(f"P3-A lineage node factory[{index}] 类型错误")
            node_writer.append(item.integer_stream())
            node_count += 1
        for index, item in enumerate(_iter_factory(
                value.roster_factories.source_binding_factory,
                label="source binding")):
            if not isinstance(item, ConversationHeldOutV4ActiveIntakeSourceBinding):
                _fail(f"P3-A source binding factory[{index}] 类型错误")
            _append_mapping(
                item.to_mapping(),
                leaf_writer=leaf_writer,
                mapping_writer=mapping_writer)
            source_count += 1
        for index, item in enumerate(_iter_factory(
                value.roster_factories.observation_binding_factory,
                label="Observation binding")):
            if not isinstance(item, ConversationHeldOutV4ActiveIntakeObservationBinding):
                _fail(f"P3-A Observation binding factory[{index}] 类型错误")
            _append_mapping(
                item.to_mapping(),
                leaf_writer=leaf_writer,
                mapping_writer=mapping_writer)
            observation_count += 1
        for index, item in enumerate(_iter_factory(
                value.roster_factories.teacher_binding_factory,
                label="teacher binding")):
            if not isinstance(item, ConversationHeldOutV4ActiveIntakeTeacherBinding):
                _fail(f"P3-A teacher binding factory[{index}] 类型错误")
            if not value.teacher_policy.allows(item.record):
                _fail("P3-A teacher record 不在冻结 permit policy 内")
            if item.split != "train":
                _fail("P3-A teacher record 不得进入 held_out side")
            _append_mapping(
                item.to_mapping(),
                leaf_writer=leaf_writer,
                mapping_writer=mapping_writer)
            teacher_count += 1
        raw_nodes = node_writer.seal()
        raw_leaves = leaf_writer.seal()
        raw_mappings = mapping_writer.seal()
    finally:
        node_writer.close()
        leaf_writer.close()
        mapping_writer.close()

    counts = ConversationHeldOutV4ActiveIntakeCounts(
        node_count,
        source_count,
        observation_count,
        teacher_count,
    )
    if raw_nodes.p0_footer.record_count != counts.lineage_node_count:
        _fail("P3-A raw node footer count 漂移")
    if (raw_leaves.p0_footer.record_count != counts.mapping_record_count
            or raw_mappings.p0_footer.record_count != counts.mapping_record_count):
        _fail("P3-A raw leaf/mapping footer count 漂移")

    stage = value.logical_stage_name
    sorted_nodes = _sort_raw_stream(
        staging_root=value.staging_run_root,
        work_root=value.work_run_root,
        raw_relative=_RAW_NODE_FILE,
        output_relative=_SORTED_NODE_FILE,
        logical_stage_name=f"{stage}-nodes",
        sort_key=_node_sort_key,
        budget=value.budget,
        label="P3-A node")
    sorted_leaves = _sort_raw_stream(
        staging_root=value.staging_run_root,
        work_root=value.work_run_root,
        raw_relative=_RAW_LEAF_FILE,
        output_relative=_SORTED_LEAF_FILE,
        logical_stage_name=f"{stage}-leaves",
        sort_key=_leaf_sort_key,
        budget=value.budget,
        label="P3-A leaf")
    sorted_mapping_record = _sort_raw_stream(
        staging_root=value.staging_run_root,
        work_root=value.work_run_root,
        raw_relative=_RAW_MAPPING_FILE,
        output_relative=_SORTED_MAPPING_RECORD_FILE,
        logical_stage_name=f"{stage}-record",
        sort_key=_mapping_record_sort_key,
        budget=value.budget,
        label="P3-A mapping-by-record")
    sorted_mapping_source = _sort_raw_stream(
        staging_root=value.staging_run_root,
        work_root=value.work_run_root,
        raw_relative=_RAW_MAPPING_FILE,
        output_relative=_SORTED_MAPPING_SOURCE_FILE,
        logical_stage_name=f"{stage}-source",
        sort_key=_mapping_source_sort_key,
        budget=value.budget,
        label="P3-A mapping-by-source")
    sorted_mapping_observation = _sort_raw_stream(
        staging_root=value.staging_run_root,
        work_root=value.work_run_root,
        raw_relative=_RAW_MAPPING_FILE,
        output_relative=_SORTED_MAPPING_OBSERVATION_FILE,
        logical_stage_name=f"{stage}-observation",
        sort_key=_mapping_observation_sort_key,
        budget=value.budget,
        label="P3-A mapping-by-observation")
    sorted_mapping_node = _sort_raw_stream(
        staging_root=value.staging_run_root,
        work_root=value.work_run_root,
        raw_relative=_RAW_MAPPING_FILE,
        output_relative=_SORTED_MAPPING_NODE_FILE,
        logical_stage_name=f"{stage}-node",
        sort_key=_mapping_node_sort_key,
        budget=value.budget,
        label="P3-A mapping-by-node")
    sorted_mapping_cluster = _sort_raw_stream(
        staging_root=value.staging_run_root,
        work_root=value.work_run_root,
        raw_relative=_RAW_MAPPING_FILE,
        output_relative=_SORTED_MAPPING_CLUSTER_FILE,
        logical_stage_name=f"{stage}-cluster",
        sort_key=_mapping_cluster_sort_key,
        budget=value.budget,
        label="P3-A mapping-by-cluster")
    sorted_mapping_leaf = _sort_raw_stream(
        staging_root=value.staging_run_root,
        work_root=value.work_run_root,
        raw_relative=_RAW_MAPPING_FILE,
        output_relative=_SORTED_MAPPING_LEAF_FILE,
        logical_stage_name=f"{stage}-leaf",
        sort_key=_mapping_leaf_sort_key,
        budget=value.budget,
        label="P3-A mapping-by-leaf")

    _validate_mapping_record_order(
        value.work_run_root,
        sorted_mapping_record.relative_path,
        counts=counts,
        budget=value.budget)
    _validate_source_join(
        value.work_run_root,
        sorted_mapping_source.relative_path,
        counts=counts,
        budget=value.budget)
    _validate_observation_join(
        value.work_run_root,
        sorted_mapping_observation.relative_path,
        counts=counts,
        budget=value.budget)
    _validate_cluster_split_join(
        value.work_run_root,
        sorted_mapping_cluster.relative_path,
        counts=counts,
        budget=value.budget)
    _validate_node_join(
        work_root=value.work_run_root,
        sorted_node_relative=sorted_nodes.relative_path,
        sorted_mapping_relative=sorted_mapping_node.relative_path,
        counts=counts,
        budget=value.budget)
    _validate_leaf_join(
        work_root=value.work_run_root,
        sorted_leaf_relative=sorted_leaves.relative_path,
        sorted_mapping_relative=sorted_mapping_leaf.relative_path,
        counts=counts,
        budget=value.budget)

    public_nodes = _copy_p0_stream(
        source_root=value.work_run_root,
        source_relative=sorted_nodes.relative_path,
        publication_root=value.publication_run_root,
        publication_relative=_PUBLIC_NODE_FILE,
        max_record_count=counts.lineage_node_count,
        budget=value.budget,
        label="P3-A nodes")
    public_leaves = _copy_p0_stream(
        source_root=value.work_run_root,
        source_relative=sorted_leaves.relative_path,
        publication_root=value.publication_run_root,
        publication_relative=_PUBLIC_LEAF_FILE,
        max_record_count=counts.mapping_record_count,
        budget=value.budget,
        label="P3-A leaves")
    public_mapping = _copy_p0_stream(
        source_root=value.work_run_root,
        source_relative=sorted_mapping_record.relative_path,
        publication_root=value.publication_run_root,
        publication_relative=_PUBLIC_MAPPING_FILE,
        max_record_count=counts.mapping_record_count,
        budget=value.budget,
        label="P3-A mappings")

    try:
        catalog = build_v4_provenance_stream_catalog(
            ConversationHeldOutV4ProvenanceCatalogInput(
                value.publication_run_root,
                (_PUBLIC_NODE_FILE,),
                (_PUBLIC_LEAF_FILE,),
                value.budget.catalog_budget,
            ))
    except (ConversationHeldOutV4ProvenanceScalableError, KRunBoundaryError,
            IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            "P3-A publication node/leaf 未通过 P2-A catalog 回读") from exc
    if (catalog.node_streams[0].p0_footer != public_nodes.p0_footer
            or catalog.node_streams[0].physical_byte_count != public_nodes.physical.byte_count
            or catalog.node_streams[0].physical_sha256 != public_nodes.physical.sha256
            or catalog.leaf_streams[0].p0_footer != public_leaves.p0_footer
            or catalog.leaf_streams[0].physical_byte_count != public_leaves.physical.byte_count
            or catalog.leaf_streams[0].physical_sha256 != public_leaves.physical.sha256):
        _fail("P3-A publication catalog 与 P0 copy identity 漂移")
    _validate_mapping_record_order(
        value.publication_run_root,
        _PUBLIC_MAPPING_FILE,
        counts=counts,
        budget=value.budget)

    status = (
        V4_ACTIVE_INTAKE_STATUS_TEST_ONLY
        if value.publication_run_root.test_transport
        else V4_ACTIVE_INTAKE_STATUS_COVERAGE_NE
    )
    provisional = ConversationHeldOutV4ActiveIntakeResult(
        value.publication_run_root,
        value.frozen_roster_manifest_identity,
        value.code_identity,
        value.teacher_policy,
        value.budget,
        catalog,
        public_mapping,
        counts,
        status,
        (0,) * _SHA256_SIZE,
    )
    manifest_payload = encode_integer_tuple(_manifest_integer_stream(provisional))
    if len(manifest_payload) > value.budget.max_manifest_bytes:
        _fail("P3-A manifest 超过预算")
    try:
        publish_manifest_last(
            value.publication_run_root,
            _MANIFEST_FILE,
            manifest_payload,
            frozenset({_PUBLIC_NODE_FILE, _PUBLIC_LEAF_FILE, _PUBLIC_MAPPING_FILE}),
            label="P3-A publication")
    except KRunBoundaryError as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            "P3-A publication manifest-last 未通过") from exc
    result = ConversationHeldOutV4ActiveIntakeResult(
        value.publication_run_root,
        value.frozen_roster_manifest_identity,
        value.code_identity,
        value.teacher_policy,
        value.budget,
        catalog,
        public_mapping,
        counts,
        status,
        tuple(hashlib.sha256(manifest_payload).digest()),
    )
    return revalidate_v4_active_intake(result)


def revalidate_v4_active_intake(
        value: ConversationHeldOutV4ActiveIntakeResult,
        ) -> ConversationHeldOutV4ActiveIntakeResult:
    """完整回读 P3-A publication，拒绝 manifest、P0 footer、physical 或 catalog 漂移。"""
    if not isinstance(value, ConversationHeldOutV4ActiveIntakeResult):
        raise TypeError("P3-A revalidate 必须接收 ConversationHeldOutV4ActiveIntakeResult")
    expected_files = frozenset({
        _PUBLIC_NODE_FILE,
        _PUBLIC_LEAF_FILE,
        _PUBLIC_MAPPING_FILE,
        _MANIFEST_FILE,
    })
    try:
        require_exact_file_closure(
            value.publication_run_root,
            expected_files,
            label="P3-A publication")
        catalog = revalidate_v4_provenance_stream_catalog(value.catalog)
    except (ConversationHeldOutV4ProvenanceScalableError, KRunBoundaryError,
            IntegerFramedStreamError, OSError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            "P3-A publication catalog 回读失败") from exc
    if catalog.stable_key() != value.catalog.stable_key():
        _fail("P3-A publication catalog stable key 漂移")
    _validate_mapping_record_order(
        value.publication_run_root,
        _PUBLIC_MAPPING_FILE,
        counts=value.counts,
        budget=value.budget)
    actual_mapping_stream = _read_p0_stream_identity(
        value.publication_run_root,
        _PUBLIC_MAPPING_FILE,
        max_record_count=value.counts.mapping_record_count,
        budget=value.budget,
        label="P3-A publication mapping")
    if actual_mapping_stream != value.mapping_stream:
        _fail("P3-A publication mapping footer/physical identity 漂移")
    manifest_payload = _read_manifest_payload(
        value.publication_run_root, budget=value.budget)
    try:
        decoded_manifest = decode_integer_tuple(manifest_payload)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            "P3-A manifest integer codec 损坏") from exc
    if encode_integer_tuple(decoded_manifest) != manifest_payload:
        _fail("P3-A manifest 不是规范整数编码")
    if decoded_manifest != _manifest_integer_stream(value):
        _fail("P3-A manifest 与 publication identity 不一致")
    manifest_sha256 = tuple(hashlib.sha256(manifest_payload).digest())
    if manifest_sha256 != value.manifest_sha256:
        _fail("P3-A manifest SHA-256 漂移")
    return value


def load_v4_active_intake_publication(
        value: ConversationHeldOutV4ActiveIntakeResult,
        ) -> ConversationHeldOutV4ActiveIntakePublication:
    """回读当前 P3-A publication 的无正文 mapping，供下游检查完整 identity 成员资格。

    先完成 P3-A 的完整 publication revalidation，再用该 P3-A 自身的 mapping 预算和
    descriptor 受限读取。返回值不访问、恢复或携带任何 source payload；调用方必须以
    ``mapping.stable_key()`` 做完整 identity 比较，不能以 record key 或摘要替代。返回的
    ``result`` 已经完整回读，``mappings`` 恰为该 result 当前公开 stream 的成员。
    """
    active = revalidate_v4_active_intake(value)
    expected_files = frozenset({
        _PUBLIC_NODE_FILE,
        _PUBLIC_LEAF_FILE,
        _PUBLIC_MAPPING_FILE,
        _MANIFEST_FILE,
    })
    try:
        file_identity = capture_plain_file_identity(
            active.publication_run_root,
            _PUBLIC_MAPPING_FILE,
            label="P3-A public mapping loader",
        )
        with open_plain_binary(
                active.publication_run_root,
                _PUBLIC_MAPPING_FILE,
                label="P3-A public mapping loader",
                expected_identity=file_identity,
        ) as stream:
            with IntegerFramedStreamReader.from_open_binary(
                    stream,
                    path=_PUBLIC_MAPPING_FILE,
                    max_frame_bytes=active.budget.max_mapping_record_payload_bytes,
                    max_record_count=active.counts.mapping_record_count,
                    max_total_payload_bytes=active.budget.max_stream_payload_bytes,
            ) as reader:
                mappings = tuple(
                    _mapping_from_record(
                        record,
                        label=f"P3-A public mapping loader[{index}]",
                    )
                    for index, record in enumerate(reader)
                )
                footer = reader.finish()
        require_plain_file_identity(
            active.publication_run_root,
            _PUBLIC_MAPPING_FILE,
            file_identity,
            label="P3-A public mapping loader post-read",
        )
        physical = sha256_plain_file(
            active.publication_run_root,
            _PUBLIC_MAPPING_FILE,
            max_bytes=active.budget.max_stream_physical_bytes,
            label="P3-A public mapping loader physical",
        )
        require_exact_file_closure(
            active.publication_run_root,
            expected_files,
            label="P3-A public mapping loader",
        )
    except ConversationHeldOutV4ActiveIntakeError:
        raise
    except (KRunBoundaryError, IntegerFramedStreamError, OSError, TypeError,
            ValueError) as exc:
        raise ConversationHeldOutV4ActiveIntakeError(
            "P3-A public mapping publication readback failed") from exc
    if (len(mappings) != active.counts.mapping_record_count
            or footer != active.mapping_stream.p0_footer
            or physical != active.mapping_stream.physical):
        _fail("P3-A public mapping stream identity drifted")
    return ConversationHeldOutV4ActiveIntakePublication(active, mappings)


__all__ = [
    "ConversationHeldOutV4ActiveIntakeBudget",
    "ConversationHeldOutV4ActiveIntakeCounts",
    "ConversationHeldOutV4ActiveIntakeError",
    "ConversationHeldOutV4ActiveIntakeInput",
    "ConversationHeldOutV4ActiveIntakeMapping",
    "ConversationHeldOutV4ActiveIntakeObservationBinding",
    "ConversationHeldOutV4ActiveIntakePublication",
    "ConversationHeldOutV4ActiveIntakeResult",
    "ConversationHeldOutV4ActiveIntakeRosterFactories",
    "ConversationHeldOutV4ActiveIntakeSourceBinding",
    "ConversationHeldOutV4ActiveIntakeStreamIdentity",
    "ConversationHeldOutV4ActiveIntakeTeacherBinding",
    "ConversationHeldOutV4ActiveIntakeTeacherPolicy",
    "V4_ACTIVE_INTAKE_RECORD_OBSERVATION",
    "V4_ACTIVE_INTAKE_RECORD_SOURCE",
    "V4_ACTIVE_INTAKE_RECORD_TEACHER",
    "V4_ACTIVE_INTAKE_STATUS_COVERAGE_NE",
    "V4_ACTIVE_INTAKE_STATUS_TEST_ONLY",
    "load_v4_active_intake_publication",
    "materialize_v4_active_intake",
    "revalidate_v4_active_intake",
]
