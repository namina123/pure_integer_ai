"""DLG-05 v4 谱系本体的纯整数、无 I/O 机械合同。

本模块只定义 snapshot、parent DAG 和 provenance leaf 的完整可逆表示。它不读取
R04 transport、candidate runtime、owner、label、private/formal 或真实训练输入；因此
任何由此模块得到的结构都只能说明本体机械闭合，不能说明外部资格或训练覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.unicode_representation import (
    validate_unicode_scalars,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    pack_key,
    strict_integer_tuple,
)


V4_PROVENANCE_SNAPSHOT_SCHEMA = 1
V4_PROVENANCE_LINEAGE_NODE_SCHEMA = 1
V4_PROVENANCE_LEAF_SCHEMA = 1
V4_PROVENANCE_FREEZE_SCHEMA = 1

V4_PROVENANCE_UPSTREAM_SHA1 = 1
V4_PROVENANCE_UPSTREAM_SHA256 = 2

V4_PROVENANCE_SIDE_TRAINING = 1
V4_PROVENANCE_SIDE_HELD_OUT = 2

_UPSTREAM_DIGEST_SIZES = {
    V4_PROVENANCE_UPSTREAM_SHA1: 20,
    V4_PROVENANCE_UPSTREAM_SHA256: 32,
}
_PROVENANCE_SIDES = frozenset({
    V4_PROVENANCE_SIDE_TRAINING,
    V4_PROVENANCE_SIDE_HELD_OUT,
})
_SHA256_SIZE = hashlib.sha256().digest_size
_URI_NON_ASCII_WHITESPACE_SCALARS = frozenset({
    0x00A0,
    0x1680,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
})


# object-model: exception
class ConversationHeldOutV4ProvenanceError(ValueError):
    """v4 snapshot、谱系节点、leaf 或其规范整数 transport 不闭合。"""


def _require_scalars(
        value: tuple[int, ...], *, label: str, allow_empty: bool,
        ) -> tuple[int, ...]:
    """校验未经归一化的 Unicode scalar tuple，并按字段合同处理空值。"""
    if not isinstance(value, tuple):
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} 必须是 Unicode scalar tuple")
    if not value:
        if allow_empty:
            return value
        raise ConversationHeldOutV4ProvenanceError(f"{label} 不得为空")
    try:
        return validate_unicode_scalars(value)
    except (TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} 含非法 Unicode scalar") from exc


def _require_absolute_uri_scalars(
        value: tuple[int, ...], *, label: str,
        ) -> tuple[int, ...]:
    """按固定标量规则校验绝对 URI，拒绝空白、控制字符和相对定位。"""
    scalars = _require_scalars(value, label=label, allow_empty=False)
    text = "".join(chr(item) for item in scalars)
    if any(
            scalar <= 0x20
            or 0x7F <= scalar <= 0x9F
            or scalar in _URI_NON_ASCII_WHITESPACE_SCALARS
            for scalar in scalars):
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} 不得含空白或控制字符")
    separator = text.find(":")
    if separator <= 0 or separator == len(text) - 1:
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} 必须是非空 absolute URI")
    scheme = text[:separator]
    first = scheme[0]
    if not (("A" <= first <= "Z") or ("a" <= first <= "z")):
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} URI scheme 必须以 ASCII 字母开始")
    if any(not (("A" <= character <= "Z")
               or ("a" <= character <= "z")
               or ("0" <= character <= "9")
               or character in "+-.")
           for character in scheme[1:]):
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} URI scheme 含非法字符")
    return scalars


def _require_digest(
        value: tuple[int, ...], *, label: str, size: int,
        ) -> tuple[int, ...]:
    """校验完整固定长度的 digest，始终以字节整数 tuple 保存。"""
    if type(size) is not int or size <= 0:
        raise RuntimeError("内部 digest 长度合同非法")
    try:
        strict_integer_tuple(value, label=label)
    except (TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} 必须是完整严格整数 tuple") from exc
    if len(value) != size or any(item < 0 or item > 255 for item in value):
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} digest 长度或字节范围非法")
    return value


def _require_protocol_key(value: ProtocolKey, *, label: str) -> ProtocolKey:
    """要求调用方保留完整非摘要的 ProtocolKey 身份。"""
    if not isinstance(value, ProtocolKey):
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} 必须是 ProtocolKey")
    return value


def _protocol_key_from_stream(
        reader: IntegerStreamReader, *, label: str,
        ) -> ProtocolKey:
    """从长度分帧的完整整数键恢复 ProtocolKey。"""
    try:
        return ProtocolKey(reader.read_key(label=label))
    except (IntegerCodecError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} ProtocolKey 非法") from exc


def _source_ref_from_stream(
        reader: IntegerStreamReader, *, label: str,
        ) -> SourceRef:
    """从固定完整键恢复 SourceRef，不接受任何摘要替代。"""
    try:
        return SourceRef.from_stable_key(reader.read_key(label=label))
    except (AssertionError, IntegerCodecError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} SourceRef 非法") from exc


def _canonical_source_ref(value: SourceRef, *, label: str) -> SourceRef:
    """把 SourceRef 经完整十一整数键重建为严格基类实例，拒绝 bool 等伪键。"""
    if not isinstance(value, SourceRef):
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} 必须是 SourceRef")
    try:
        stable_key = value.stable_key()
        canonical = SourceRef.from_stable_key(stable_key)
    except (AssertionError, TypeError, ValueError) as exc:
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} SourceRef 必须是完整严格十一整数键") from exc
    if canonical.stable_key() != stable_key:
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} SourceRef 规范重建漂移")
    return canonical


def _read_schema(
        reader: IntegerStreamReader, *, expected: int, label: str,
        ) -> None:
    """读取并固定一个本体整数 transport 的 schema，拒绝隐式兼容。"""
    try:
        actual = reader.read_positive(label=f"{label} schema")
    except IntegerCodecError as exc:
        raise ConversationHeldOutV4ProvenanceError(
            f"{label} schema 缺失") from exc
    if actual != expected:
        raise ConversationHeldOutV4ProvenanceError(f"{label} schema 不兼容")


def _canonical_parent_keys(
        value: tuple[ProtocolKey, ...], *, node_key: ProtocolKey,
        ) -> tuple[ProtocolKey, ...]:
    """要求 parent 是严格排序、无重复且不指向自身的完整 node key 序列。"""
    if not isinstance(value, tuple):
        raise ConversationHeldOutV4ProvenanceError(
            "lineage parent_node_keys 必须是 ProtocolKey tuple")
    if any(not isinstance(item, ProtocolKey) for item in value):
        raise ConversationHeldOutV4ProvenanceError(
            "lineage parent_node_keys 含非 ProtocolKey")
    canonical = tuple(sorted(value, key=lambda item: item.components))
    if value != canonical:
        raise ConversationHeldOutV4ProvenanceError(
            "lineage parent_node_keys 必须按完整 node key 排序")
    if len(set(value)) != len(value):
        raise ConversationHeldOutV4ProvenanceError(
            "lineage parent_node_keys 不得重复")
    if node_key in value:
        raise ConversationHeldOutV4ProvenanceError(
            "lineage node 不得把自身列为 parent")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4SnapshotIdentity:
    """一个来源 snapshot 的完整官方、版本、digest、许可和代码身份。"""

    source_ref: SourceRef
    official_uri_scalars: tuple[int, ...]
    revision_scalars: tuple[int, ...]
    snapshot_scalars: tuple[int, ...]
    upstream_digest_algorithm: int
    upstream_digest: tuple[int, ...]
    local_sha256: tuple[int, ...]
    license_review_artifact_identity: ProtocolKey
    ingest_code_identity: ProtocolKey
    transform_code_identity: ProtocolKey

    def __post_init__(self) -> None:
        """要求可重建的官方定位、版本边界和每一段完整身份。"""
        object.__setattr__(
            self,
            "source_ref",
            _canonical_source_ref(
                self.source_ref, label="snapshot source_ref"),
        )
        _require_absolute_uri_scalars(
            self.official_uri_scalars,
            label="snapshot official_uri_scalars",
        )
        revision = _require_scalars(
            self.revision_scalars,
            label="snapshot revision_scalars",
            allow_empty=True,
        )
        snapshot = _require_scalars(
            self.snapshot_scalars,
            label="snapshot snapshot_scalars",
            allow_empty=True,
        )
        if not revision and not snapshot:
            raise ConversationHeldOutV4ProvenanceError(
                "snapshot revision/snapshot 至少一个必须非空")
        if (type(self.upstream_digest_algorithm) is not int
                or self.upstream_digest_algorithm not in _UPSTREAM_DIGEST_SIZES):
            raise ConversationHeldOutV4ProvenanceError(
                "snapshot upstream digest algorithm 未注册")
        _require_digest(
            self.upstream_digest,
            label="snapshot upstream_digest",
            size=_UPSTREAM_DIGEST_SIZES[self.upstream_digest_algorithm],
        )
        _require_digest(
            self.local_sha256,
            label="snapshot local_sha256",
            size=_SHA256_SIZE,
        )
        for label, value in (
                ("snapshot license_review_artifact_identity",
                 self.license_review_artifact_identity),
                ("snapshot ingest_code_identity", self.ingest_code_identity),
                ("snapshot transform_code_identity",
                 self.transform_code_identity)):
            _require_protocol_key(value, label=label)

    def integer_stream(self) -> tuple[int, ...]:
        """返回含全部 snapshot 本体字段的版本化、可逆整数序列。"""
        result = [V4_PROVENANCE_SNAPSHOT_SCHEMA]
        pack_key(result, self.source_ref.stable_key())
        for value in (
                self.official_uri_scalars,
                self.revision_scalars,
                self.snapshot_scalars):
            pack_key(result, value)
        result.append(self.upstream_digest_algorithm)
        for value in (
                self.upstream_digest,
                self.local_sha256,
                self.license_review_artifact_identity.components,
                self.ingest_code_identity.components,
                self.transform_code_identity.components):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回不省略 URI、版本、digest、许可或代码身份的完整 snapshot 键。"""
        return self.integer_stream()

    @classmethod
    def from_integer_stream(
            cls, values: tuple[int, ...],
            ) -> "ConversationHeldOutV4SnapshotIdentity":
        """从完整整数序列恢复 snapshot，并拒绝尾字段或非规范结构。"""
        try:
            reader = IntegerStreamReader(values)
            _read_schema(
                reader,
                expected=V4_PROVENANCE_SNAPSHOT_SCHEMA,
                label="snapshot",
            )
            source_ref = _source_ref_from_stream(
                reader, label="snapshot source_ref")
            official_uri_scalars = reader.read_key(
                label="snapshot official_uri_scalars")
            revision_scalars = reader.read_key(
                label="snapshot revision_scalars", empty=True)
            snapshot_scalars = reader.read_key(
                label="snapshot snapshot_scalars", empty=True)
            upstream_digest_algorithm = reader.read_positive(
                label="snapshot upstream_digest_algorithm")
            upstream_digest = reader.read_key(label="snapshot upstream_digest")
            local_sha256 = reader.read_key(label="snapshot local_sha256")
            license_review_artifact_identity = _protocol_key_from_stream(
                reader, label="snapshot license_review_artifact_identity")
            ingest_code_identity = _protocol_key_from_stream(
                reader, label="snapshot ingest_code_identity")
            transform_code_identity = _protocol_key_from_stream(
                reader, label="snapshot transform_code_identity")
            reader.finish()
            result = cls(
                source_ref,
                official_uri_scalars,
                revision_scalars,
                snapshot_scalars,
                upstream_digest_algorithm,
                upstream_digest,
                local_sha256,
                license_review_artifact_identity,
                ingest_code_identity,
                transform_code_identity,
            )
        except ConversationHeldOutV4ProvenanceError:
            raise
        except (IntegerCodecError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ProvenanceError(
                "snapshot integer stream 损坏") from exc
        if result.integer_stream() != values:
            raise ConversationHeldOutV4ProvenanceError(
                "snapshot integer stream 不是规范表达")
        return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4LineageNode:
    """一个带完整 snapshot 和直接 parent 关系的谱系 DAG 节点。"""

    node_key: ProtocolKey
    snapshot: ConversationHeldOutV4SnapshotIdentity
    parent_node_keys: tuple[ProtocolKey, ...] = ()

    def __post_init__(self) -> None:
        """固定 node、snapshot 与直接 parent 的规范一等关系。"""
        _require_protocol_key(self.node_key, label="lineage node_key")
        if not isinstance(self.snapshot, ConversationHeldOutV4SnapshotIdentity):
            raise ConversationHeldOutV4ProvenanceError(
                "lineage snapshot 类型错误")
        _canonical_parent_keys(self.parent_node_keys, node_key=self.node_key)

    def integer_stream(self) -> tuple[int, ...]:
        """返回 node、完整 snapshot 和所有直接 parent 的可逆整数序列。"""
        result = [V4_PROVENANCE_LINEAGE_NODE_SCHEMA]
        pack_key(result, self.node_key.components)
        pack_key(result, self.snapshot.integer_stream())
        result.append(len(self.parent_node_keys))
        for parent in self.parent_node_keys:
            pack_key(result, parent.components)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回包含 node、snapshot 与所有直接 parent 的完整结构键。"""
        return self.integer_stream()

    @classmethod
    def from_integer_stream(
            cls, values: tuple[int, ...],
            ) -> "ConversationHeldOutV4LineageNode":
        """从完整整数序列重建 node，并拒绝未闭合的直接关系编码。"""
        try:
            reader = IntegerStreamReader(values)
            _read_schema(
                reader,
                expected=V4_PROVENANCE_LINEAGE_NODE_SCHEMA,
                label="lineage node",
            )
            node_key = _protocol_key_from_stream(reader, label="lineage node_key")
            snapshot = ConversationHeldOutV4SnapshotIdentity.from_integer_stream(
                reader.read_key(label="lineage snapshot"))
            parent_count = reader.read_nonnegative(
                label="lineage parent_count")
            parent_node_keys = tuple(
                _protocol_key_from_stream(
                    reader, label=f"lineage parent_node_keys[{index}]")
                for index in range(parent_count)
            )
            reader.finish()
            result = cls(node_key, snapshot, parent_node_keys)
        except ConversationHeldOutV4ProvenanceError:
            raise
        except (IntegerCodecError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ProvenanceError(
                "lineage node integer stream 损坏") from exc
        if result.integer_stream() != values:
            raise ConversationHeldOutV4ProvenanceError(
                "lineage node integer stream 不是规范表达")
        return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4ProvenanceLeaf:
    """训练或 held-out 一条输入的完整来源、内容和谱系端点本体。"""

    side: int
    source_ref: SourceRef
    content_sha256: tuple[int, ...]
    lineage_node_key: ProtocolKey
    consumed_input_identity: ProtocolKey

    def __post_init__(self) -> None:
        """拒绝缺侧、摘要来源、残缺内容 hash 或缺失实际输入身份。"""
        if type(self.side) is not int or self.side not in _PROVENANCE_SIDES:
            raise ConversationHeldOutV4ProvenanceError("provenance leaf side 非法")
        object.__setattr__(
            self,
            "source_ref",
            _canonical_source_ref(
                self.source_ref, label="provenance leaf source_ref"),
        )
        _require_digest(
            self.content_sha256,
            label="provenance leaf content_sha256",
            size=_SHA256_SIZE,
        )
        _require_protocol_key(
            self.lineage_node_key, label="provenance leaf lineage_node_key")
        _require_protocol_key(
            self.consumed_input_identity,
            label="provenance leaf consumed_input_identity")

    def integer_stream(self) -> tuple[int, ...]:
        """返回 P2 可逐帧写入的完整 leaf record，而非任意 lineage 摘要。"""
        result = [V4_PROVENANCE_LEAF_SCHEMA, self.side]
        for value in (
                self.source_ref.stable_key(),
                self.content_sha256,
                self.lineage_node_key.components,
                self.consumed_input_identity.components):
            pack_key(result, value)
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回包含 leaf 全部本体字段的稳定整数键，供后续投影引用。"""
        return self.integer_stream()

    @classmethod
    def from_integer_stream(
            cls, values: tuple[int, ...],
            ) -> "ConversationHeldOutV4ProvenanceLeaf":
        """从完整整数序列恢复 leaf，并拒绝尾字段或摘要化来源。"""
        try:
            reader = IntegerStreamReader(values)
            _read_schema(
                reader,
                expected=V4_PROVENANCE_LEAF_SCHEMA,
                label="provenance leaf",
            )
            side = reader.read_positive(label="provenance leaf side")
            source_ref = _source_ref_from_stream(
                reader, label="provenance leaf source_ref")
            content_sha256 = reader.read_key(
                label="provenance leaf content_sha256")
            lineage_node_key = _protocol_key_from_stream(
                reader, label="provenance leaf lineage_node_key")
            consumed_input_identity = _protocol_key_from_stream(
                reader, label="provenance leaf consumed_input_identity")
            reader.finish()
            result = cls(
                side,
                source_ref,
                content_sha256,
                lineage_node_key,
                consumed_input_identity,
            )
        except ConversationHeldOutV4ProvenanceError:
            raise
        except (IntegerCodecError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ProvenanceError(
                "provenance leaf integer stream 损坏") from exc
        if result.integer_stream() != values:
            raise ConversationHeldOutV4ProvenanceError(
                "provenance leaf integer stream 不是规范表达")
        return result


def _validate_acyclic(
        nodes_by_key: dict[ProtocolKey, ConversationHeldOutV4LineageNode],
        ) -> None:
    """用显式栈验证 parent DAG，避免深谱系因 Python 递归深度而失真。"""
    states: dict[ProtocolKey, int] = {}
    for start in nodes_by_key:
        if states.get(start, 0) != 0:
            continue
        stack: list[tuple[ProtocolKey, int]] = [(start, 0)]
        while stack:
            key, parent_index = stack[-1]
            state = states.get(key, 0)
            if state == 0:
                states[key] = 1
            parents = nodes_by_key[key].parent_node_keys
            if parent_index >= len(parents):
                states[key] = 2
                stack.pop()
                continue
            parent = parents[parent_index]
            stack[-1] = (key, parent_index + 1)
            parent_state = states.get(parent, 0)
            if parent_state == 1:
                raise ConversationHeldOutV4ProvenanceError(
                    "lineage freeze 检测到 parent DAG 环")
            if parent_state == 0:
                stack.append((parent, 0))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4LineageFreeze:
    """一个已验证 parent DAG 与其 test-only provenance leaf 的不可变快照。"""

    nodes: tuple[ConversationHeldOutV4LineageNode, ...]
    leaves: tuple[ConversationHeldOutV4ProvenanceLeaf, ...] = ()

    def __post_init__(self) -> None:
        """验证 node/leaf 的规范排序、父闭合、无环和端点闭合。"""
        if (not isinstance(self.nodes, tuple) or not self.nodes
                or any(not isinstance(item, ConversationHeldOutV4LineageNode)
                       for item in self.nodes)):
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze nodes 必须是非空 LineageNode tuple")
        node_order = tuple(sorted(
            self.nodes, key=lambda item: item.node_key.components))
        if self.nodes != node_order:
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze nodes 必须按完整 node key 排序")
        node_keys = tuple(item.node_key for item in self.nodes)
        if len(set(node_keys)) != len(node_keys):
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze node_key 不得重复")
        if (not isinstance(self.leaves, tuple)
                or any(not isinstance(item, ConversationHeldOutV4ProvenanceLeaf)
                       for item in self.leaves)):
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze leaves 必须是 ProvenanceLeaf tuple")
        leaf_order = tuple(sorted(self.leaves, key=lambda item: item.stable_key()))
        if self.leaves != leaf_order:
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze leaves 必须按完整 leaf key 排序")
        if len(set(self.leaves)) != len(self.leaves):
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze 不得重复完整 provenance leaf")
        nodes_by_key = {item.node_key: item for item in self.nodes}
        for node in self.nodes:
            for parent in node.parent_node_keys:
                if parent not in nodes_by_key:
                    raise ConversationHeldOutV4ProvenanceError(
                        "lineage freeze parent node 未闭合")
        _validate_acyclic(nodes_by_key)
        for leaf in self.leaves:
            node = nodes_by_key.get(leaf.lineage_node_key)
            if node is None:
                raise ConversationHeldOutV4ProvenanceError(
                    "lineage freeze provenance leaf 指向未知 node")
            if leaf.source_ref != node.snapshot.source_ref:
                raise ConversationHeldOutV4ProvenanceError(
                    "lineage freeze provenance leaf SourceRef 与 snapshot 不一致")

    def integer_stream(self) -> tuple[int, ...]:
        """返回完整 node/leaf 本体快照的版本化、可逆整数序列。"""
        result = [V4_PROVENANCE_FREEZE_SCHEMA, len(self.nodes)]
        for node in self.nodes:
            pack_key(result, node.integer_stream())
        result.append(len(self.leaves))
        for leaf in self.leaves:
            pack_key(result, leaf.integer_stream())
        return tuple(result)

    def to_bytes(self) -> bytes:
        """把 test-only freeze 编为现役规范整数 codec 字节，不写任何路径。"""
        return encode_integer_tuple(self.integer_stream())

    def canonical_sha256(self) -> tuple[int, ...]:
        """返回完整规范 freeze 字节的派生 SHA-256，不以其替代 freeze 本体。"""
        return tuple(hashlib.sha256(self.to_bytes()).digest())

    @classmethod
    def from_integer_stream(
            cls, values: tuple[int, ...],
            ) -> "ConversationHeldOutV4LineageFreeze":
        """从完整整数序列恢复 DAG freeze，并重新执行所有闭合不变量。"""
        try:
            reader = IntegerStreamReader(values)
            _read_schema(
                reader,
                expected=V4_PROVENANCE_FREEZE_SCHEMA,
                label="lineage freeze",
            )
            node_count = reader.read_nonnegative(label="lineage freeze node_count")
            nodes = tuple(
                ConversationHeldOutV4LineageNode.from_integer_stream(
                    reader.read_key(label=f"lineage freeze nodes[{index}]"))
                for index in range(node_count)
            )
            leaf_count = reader.read_nonnegative(label="lineage freeze leaf_count")
            leaves = tuple(
                ConversationHeldOutV4ProvenanceLeaf.from_integer_stream(
                    reader.read_key(label=f"lineage freeze leaves[{index}]"))
                for index in range(leaf_count)
            )
            reader.finish()
            result = cls(nodes, leaves)
        except ConversationHeldOutV4ProvenanceError:
            raise
        except (IntegerCodecError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze integer stream 损坏") from exc
        if result.integer_stream() != values:
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze integer stream 不是规范表达")
        return result

    @classmethod
    def from_bytes(
            cls, data: bytes,
            ) -> "ConversationHeldOutV4LineageFreeze":
        """从规范整数 codec 字节恢复 freeze，拒绝截断、尾随和非规范 varint。"""
        try:
            result = cls.from_integer_stream(decode_integer_tuple(data))
        except ConversationHeldOutV4ProvenanceError:
            raise
        except (IntegerCodecError, TypeError, ValueError) as exc:
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze bytes 损坏") from exc
        if result.to_bytes() != data:
            raise ConversationHeldOutV4ProvenanceError(
                "lineage freeze bytes 不是规范表达")
        return result


__all__ = [
    "ConversationHeldOutV4LineageFreeze",
    "ConversationHeldOutV4LineageNode",
    "ConversationHeldOutV4ProvenanceError",
    "ConversationHeldOutV4ProvenanceLeaf",
    "ConversationHeldOutV4SnapshotIdentity",
    "V4_PROVENANCE_FREEZE_SCHEMA",
    "V4_PROVENANCE_LEAF_SCHEMA",
    "V4_PROVENANCE_LINEAGE_NODE_SCHEMA",
    "V4_PROVENANCE_SIDE_HELD_OUT",
    "V4_PROVENANCE_SIDE_TRAINING",
    "V4_PROVENANCE_SNAPSHOT_SCHEMA",
    "V4_PROVENANCE_UPSTREAM_SHA1",
    "V4_PROVENANCE_UPSTREAM_SHA256",
]
