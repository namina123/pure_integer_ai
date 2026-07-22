"""GraphObject 完整身份的版本化纯整数物理 codec。

本模块只压缩 ``ObjectIdentity.stable_key()`` 的物理重复，不定义语言、分词或结构
语义。普通对象使用固定槽和开放 overflow；来源化对象通过已登记 SourceRef hash
恢复重复来源段；Hypothesis 以通用整数竞争组和最长公共前缀压缩候选键。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_SCOPE,
    IDENTITY_SOURCE_RECORD,
)
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT


GRAPH_OBJECT_COMPONENT_TABLE = "graph_object_component"
GRAPH_HYPOTHESIS_GROUP_TABLE = "graph_hypothesis_group"
GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE = (
    "graph_hypothesis_group_component")

GRAPH_OBJECT_CODEC_GENERIC_V1 = 1
GRAPH_OBJECT_CODEC_OCCURRENCE_SOURCE_V1 = 2
GRAPH_OBJECT_CODEC_SPAN_SOURCE_V1 = 3
GRAPH_OBJECT_CODEC_HYPOTHESIS_V1 = 4

GRAPH_OBJECT_CODEC_VALUE_COUNT = 8
_HYPOTHESIS_SUFFIX_INLINE_COUNT = GRAPH_OBJECT_CODEC_VALUE_COUNT - 3
_SPAN_INLINE_MEMBER_COUNT = (
    GRAPH_OBJECT_CODEC_VALUE_COUNT - 2) // 2

GROUP_COMPONENT_HYPOTHESIS_KIND = 1
GROUP_COMPONENT_COMPETITION_KEY = 2

GRAPH_OBJECT_IDENTITY_COLUMNS = [
    ("identity_codec", TYPE_INT),
    ("owner_tenant_id", TYPE_INT),
    ("owner_user_id", TYPE_INT),
    ("owner_session_id", TYPE_INT),
    ("owner_visibility", TYPE_INT),
    ("corpus_version", TYPE_INT),
    ("parser_version", TYPE_INT),
    ("primitive_version", TYPE_INT),
    ("curriculum_version", TYPE_INT),
    ("component_size", TYPE_INT),
    ("codec_ref_hash", TYPE_INT),
    *[(f"codec_value_{index}", TYPE_INT)
      for index in range(GRAPH_OBJECT_CODEC_VALUE_COUNT)],
]

GRAPH_OBJECT_COMPONENT_COLUMNS = [
    ("identity_hash", TYPE_INT),
    ("component_ordinal", TYPE_INT),
    ("component_value", TYPE_INT),
]
GRAPH_HYPOTHESIS_GROUP_COLUMNS = [
    ("group_hash", TYPE_INT),
    ("hypothesis_version", TYPE_INT),
    ("hypothesis_kind_size", TYPE_INT),
    ("competition_size", TYPE_INT),
    ("scope_hash", TYPE_INT),
    ("observation_hash", TYPE_INT),
]
GRAPH_HYPOTHESIS_GROUP_COMPONENT_COLUMNS = [
    ("group_hash", TYPE_INT),
    ("component_role", TYPE_INT),
    ("component_ordinal", TYPE_INT),
    ("component_value", TYPE_INT),
]

_GROUP_HASHER = Hasher("graph_hypothesis_group.v1")


class GraphObjectIdentityError(RuntimeError):
    """GraphObject 身份 codec 错误基类。"""


class GraphObjectIdentityCollisionError(GraphObjectIdentityError):
    """物理 group hash 命中不同完整竞争组。"""


class GraphObjectIdentityIncompleteError(GraphObjectIdentityError):
    """主记录、overflow 或竞争组出现缺失、重复和非规范编码。"""


def register_graph_object_identity_tables(backend: StorageBackend) -> None:
    """注册 GraphObject overflow 和 Hypothesis 竞争组核心表。"""
    backend.register_table(
        GRAPH_OBJECT_COMPONENT_TABLE,
        GRAPH_OBJECT_COMPONENT_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("identity_hash",),
            ("identity_hash", "component_ordinal"),
        ],
        core=True,
    )
    backend.register_table(
        GRAPH_HYPOTHESIS_GROUP_TABLE,
        GRAPH_HYPOTHESIS_GROUP_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("group_hash",),
            ("scope_hash",),
            ("observation_hash",),
        ],
        core=True,
    )
    backend.register_table(
        GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE,
        GRAPH_HYPOTHESIS_GROUP_COMPONENT_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("group_hash",),
            ("group_hash", "component_role", "component_ordinal"),
        ],
        core=True,
    )


def _strict_int(value: int, *, where: str,
                nonnegative: bool = False,
                positive: bool = False) -> int:
    """校验 codec 字段使用严格整数，并按职责限制取值范围。"""
    if type(value) is not int:
        assert_int(value, _where=where)
        raise ValueError(f"{where} 必须为严格整数")
    if positive and value <= 0:
        raise ValueError(f"{where} 必须为正整数")
    if nonnegative and value < 0:
        raise ValueError(f"{where} 必须为非负整数")
    return value


def _integer_tuple(value: tuple[int, ...], *, where: str,
                   allow_empty: bool = False) -> tuple[int, ...]:
    """校验开放整数 tuple，是否允许空键由具体物理字段声明。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{where} 必须为整数 tuple")
    for index, item in enumerate(value):
        _strict_int(item, where=f"{where}[{index}]")
    return value


@dataclass(frozen=True)
class _ObjectEnvelope:
    """从 ObjectIdentity 完整键拆出的固定 envelope 和 components。"""

    object_kind: int
    owner_key: tuple[int, int, int, int]
    version_key: tuple[int, int, int, int]
    components: tuple[int, ...]


def _split_object_key(key: tuple[int, ...]) -> _ObjectEnvelope:
    """按固定 ObjectIdentity 协议拆键，拒绝截断、尾随和非法长度。"""
    key = _integer_tuple(key, where="graph object stable key")
    if len(key) < 11:
        raise ValueError("graph object stable key 长度不足")
    object_kind = _strict_int(
        key[0], where="graph object object_kind", positive=True)
    owner_key = key[1:5]
    version_key = key[5:9]
    for index, value in enumerate(owner_key):
        _strict_int(
            value,
            where=f"graph object owner[{index}]",
            positive=index == 3,
            nonnegative=index != 3,
        )
    for index, value in enumerate(version_key):
        _strict_int(
            value, where=f"graph object version[{index}]",
            nonnegative=True)
    component_size = _strict_int(
        key[9], where="graph object component_size", positive=True)
    if len(key) != 10 + component_size:
        raise ValueError("graph object components 长度不一致")
    return _ObjectEnvelope(
        object_kind,
        owner_key,
        version_key,
        key[10:],
    )


def _source_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """校验当前 SourceRef 固定稳定键，并保留完整 owner/version。"""
    value = _integer_tuple(value, where="source stable key")
    if len(value) != 11:
        raise ValueError("source stable key 长度必须为 11")
    _strict_int(value[0], where="source_kind", positive=True)
    _strict_int(value[1], where="source_id", positive=True)
    _strict_int(value[2], where="document_id", nonnegative=True)
    return value


def _assert_source_envelope(
        envelope: _ObjectEnvelope, source_key: tuple[int, ...]) -> None:
    """核验来源化对象 envelope 与 SourceRef owner/version 完全一致。"""
    if envelope.owner_key != source_key[3:7]:
        raise ValueError("来源化对象 owner 与 SourceRef 不一致")
    if envelope.version_key != source_key[7:11]:
        raise ValueError("来源化对象 version 与 SourceRef 不一致")


def _padded(values: tuple[int, ...], size: int) -> tuple[int, ...]:
    """把不超过固定容量的整数段补规范零值。"""
    if len(values) > size:
        raise ValueError("固定 codec 槽容量不足")
    return values + (0,) * (size - len(values))


def _common_prefix_size(
        first: tuple[int, ...], second: tuple[int, ...]) -> int:
    """返回两个开放整数 tuple 的最长公共前缀长度。"""
    size = 0
    for left, right in zip(first, second):
        if left != right:
            break
        size += 1
    return size


@dataclass(frozen=True)
class HypothesisGroupSpec:
    """Hypothesis 候选共享的版本、kind、竞争键、scope 和来源引用。"""

    hypothesis_version: int
    hypothesis_kind: tuple[int, ...]
    competition_key: tuple[int, ...]
    scope_hash: int
    scope_key: tuple[int, ...]
    observation_hash: int
    observation_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_int(
            self.hypothesis_version,
            where="hypothesis_version", positive=True)
        _integer_tuple(
            self.hypothesis_kind, where="hypothesis_kind")
        _integer_tuple(
            self.competition_key, where="competition_key")
        _strict_int(self.scope_hash, where="scope_hash", positive=True)
        _integer_tuple(self.scope_key, where="scope_key")
        _strict_int(
            self.observation_hash,
            where="observation_hash", positive=True)
        _source_key(self.observation_key)

    def stable_key(self) -> tuple[int, ...]:
        """返回 group hash 的完整可核验物理键。"""
        return (
            self.hypothesis_version,
            len(self.hypothesis_kind),
            *self.hypothesis_kind,
            len(self.competition_key),
            *self.competition_key,
            self.scope_hash,
            self.observation_hash,
        )

    def group_hash(self) -> int:
        """生成非零竞争组索引，完整组内容仍由主表和子表核验。"""
        value = _GROUP_HASHER.h63(self.stable_key())
        return value if value > 0 else 1


@dataclass(frozen=True)
class GraphObjectIdentitySpec:
    """一个 GraphObject 完整键及其可逆物理编码计划。"""

    stable_key: tuple[int, ...]
    identity_codec: int
    codec_ref_hash: int
    codec_values: tuple[int, ...]
    overflow_components: tuple[tuple[int, int], ...] = ()
    hypothesis_group: HypothesisGroupSpec | None = None

    def __post_init__(self) -> None:
        _split_object_key(self.stable_key)
        if self.identity_codec not in {
                GRAPH_OBJECT_CODEC_GENERIC_V1,
                GRAPH_OBJECT_CODEC_OCCURRENCE_SOURCE_V1,
                GRAPH_OBJECT_CODEC_SPAN_SOURCE_V1,
                GRAPH_OBJECT_CODEC_HYPOTHESIS_V1}:
            raise ValueError("graph object identity codec 未注册")
        _strict_int(
            self.codec_ref_hash,
            where="codec_ref_hash", nonnegative=True)
        if len(self.codec_values) != GRAPH_OBJECT_CODEC_VALUE_COUNT:
            raise ValueError("codec_values 长度不一致")
        _integer_tuple(
            self.codec_values, where="codec_values", allow_empty=False)
        previous = -1
        for ordinal, value in self.overflow_components:
            _strict_int(
                ordinal, where="overflow ordinal", nonnegative=True)
            _strict_int(value, where="overflow value")
            if ordinal <= previous:
                raise ValueError("overflow ordinal 必须严格递增")
            previous = ordinal
        if ((self.identity_codec == GRAPH_OBJECT_CODEC_HYPOTHESIS_V1)
                != (self.hypothesis_group is not None)):
            raise ValueError("Hypothesis codec 与竞争组必须同时存在")

    @classmethod
    def generic(cls, stable_key: tuple[int, ...]) -> "GraphObjectIdentitySpec":
        """为任意 ObjectIdentity 建固定前缀加开放 overflow codec。"""
        envelope = _split_object_key(stable_key)
        inline = envelope.components[:GRAPH_OBJECT_CODEC_VALUE_COUNT]
        overflow = tuple(enumerate(
            envelope.components[GRAPH_OBJECT_CODEC_VALUE_COUNT:],
            start=GRAPH_OBJECT_CODEC_VALUE_COUNT,
        ))
        return cls(
            stable_key,
            GRAPH_OBJECT_CODEC_GENERIC_V1,
            0,
            _padded(inline, GRAPH_OBJECT_CODEC_VALUE_COUNT),
            overflow,
        )

    @classmethod
    def occurrence_source(
            cls, stable_key: tuple[int, ...], *,
            source_hash: int, source_key: tuple[int, ...]
            ) -> "GraphObjectIdentitySpec":
        """用 SourceRef hash 和三个位置字段编码来源化 Occurrence。"""
        envelope = _split_object_key(stable_key)
        source_key = _source_key(source_key)
        _assert_source_envelope(envelope, source_key)
        if envelope.components[:11] != source_key:
            raise ValueError("Occurrence components 未引用给定 SourceRef")
        position = envelope.components[11:]
        if len(position) != 3:
            raise ValueError("Occurrence components 长度非法")
        _strict_int(source_hash, where="source_hash", positive=True)
        return cls(
            stable_key,
            GRAPH_OBJECT_CODEC_OCCURRENCE_SOURCE_V1,
            source_hash,
            _padded(position, GRAPH_OBJECT_CODEC_VALUE_COUNT),
        )

    @classmethod
    def span_source(
            cls, stable_key: tuple[int, ...], *,
            source_hash: int, source_key: tuple[int, ...]
            ) -> "GraphObjectIdentitySpec":
        """用 SourceRef hash、成员数和区间序编码来源化 Span。"""
        envelope = _split_object_key(stable_key)
        source_key = _source_key(source_key)
        _assert_source_envelope(envelope, source_key)
        if envelope.components[:11] != source_key:
            raise ValueError("Span components 未引用给定 SourceRef")
        payload = envelope.components[11:]
        if len(payload) < 4:
            raise ValueError("Span components 长度不足")
        member_count = _strict_int(
            payload[1], where="span member_count", positive=True)
        if len(payload) != 2 + member_count * 2:
            raise ValueError("Span member_count 与 components 不一致")
        _strict_int(source_hash, where="source_hash", positive=True)
        inline = payload[:GRAPH_OBJECT_CODEC_VALUE_COUNT]
        overflow_start = 11 + GRAPH_OBJECT_CODEC_VALUE_COUNT
        overflow = tuple(enumerate(
            payload[GRAPH_OBJECT_CODEC_VALUE_COUNT:],
            start=overflow_start,
        ))
        return cls(
            stable_key,
            GRAPH_OBJECT_CODEC_SPAN_SOURCE_V1,
            source_hash,
            _padded(inline, GRAPH_OBJECT_CODEC_VALUE_COUNT),
            overflow,
        )

    @classmethod
    def hypothesis(
            cls, stable_key: tuple[int, ...], *,
            hypothesis_version: int,
            hypothesis_kind: tuple[int, ...],
            candidate_key: tuple[int, ...],
            competition_key: tuple[int, ...],
            scope_hash: int,
            scope_key: tuple[int, ...],
            observation_hash: int,
            observation_key: tuple[int, ...],
            ) -> "GraphObjectIdentitySpec":
        """按通用 Hypothesis 协议共享竞争组并保存完整候选 suffix。"""
        envelope = _split_object_key(stable_key)
        hypothesis_kind = _integer_tuple(
            hypothesis_kind, where="hypothesis_kind")
        candidate_key = _integer_tuple(
            candidate_key, where="candidate_key")
        competition_key = _integer_tuple(
            competition_key, where="competition_key")
        scope_key = _integer_tuple(scope_key, where="scope_key")
        observation_key = _source_key(observation_key)
        _assert_source_envelope(envelope, observation_key)
        expected_components = (
            hypothesis_version,
            len(hypothesis_kind),
            *hypothesis_kind,
            len(candidate_key),
            *candidate_key,
            len(competition_key),
            *competition_key,
            len(scope_key),
            *scope_key,
            len(observation_key),
            *observation_key,
        )
        if envelope.components != expected_components:
            raise ValueError("Hypothesis 字段不能重建 ObjectIdentity components")
        group = HypothesisGroupSpec(
            hypothesis_version,
            hypothesis_kind,
            competition_key,
            scope_hash,
            scope_key,
            observation_hash,
            observation_key,
        )
        prefix_size = _common_prefix_size(candidate_key, competition_key)
        suffix = candidate_key[prefix_size:]
        inline_suffix = suffix[:_HYPOTHESIS_SUFFIX_INLINE_COUNT]
        values = (
            hypothesis_version,
            len(candidate_key),
            prefix_size,
            *inline_suffix,
        )
        overflow = tuple(enumerate(
            suffix[_HYPOTHESIS_SUFFIX_INLINE_COUNT:],
            start=_HYPOTHESIS_SUFFIX_INLINE_COUNT,
        ))
        return cls(
            stable_key,
            GRAPH_OBJECT_CODEC_HYPOTHESIS_V1,
            group.group_hash(),
            _padded(values, GRAPH_OBJECT_CODEC_VALUE_COUNT),
            overflow,
            group,
        )

    def main_row_fields(self) -> dict[str, int]:
        """生成并入 graph_object 主记录的 self-headed 固定字段。"""
        envelope = _split_object_key(self.stable_key)
        row = {
            "identity_codec": self.identity_codec,
            "owner_tenant_id": envelope.owner_key[0],
            "owner_user_id": envelope.owner_key[1],
            "owner_session_id": envelope.owner_key[2],
            "owner_visibility": envelope.owner_key[3],
            "corpus_version": envelope.version_key[0],
            "parser_version": envelope.version_key[1],
            "primitive_version": envelope.version_key[2],
            "curriculum_version": envelope.version_key[3],
            "component_size": len(envelope.components),
            "codec_ref_hash": self.codec_ref_hash,
        }
        for index, value in enumerate(self.codec_values):
            row[f"codec_value_{index}"] = value
        return row


IdentityKeyReader = Callable[[int, int], tuple[int, ...]]


class GraphObjectIdentityStore:
    """写入和恢复 GraphObject codec 子表，并严格核验竞争组共享。"""

    def __init__(self, backend: StorageBackend,
                 key_reader: IdentityKeyReader) -> None:
        self._backend = backend
        self._key_reader = key_reader
        self._groups_by_hash: dict[int, HypothesisGroupSpec] = {}

    def auxiliary_namespace_is_empty(self) -> bool:
        """检查所有 GraphObject codec 子表为空，供首次批量快路声明。"""
        return (
            self._backend.count(GRAPH_OBJECT_COMPONENT_TABLE) == 0
            and self._backend.count(GRAPH_HYPOTHESIS_GROUP_TABLE) == 0
            and self._backend.count(
                GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE) == 0
        )

    def append_new(self, identity_hash: int,
                   spec: GraphObjectIdentitySpec) -> None:
        """在已核验空命名空间中追加新 codec 子载荷并做内存碰撞核验。"""
        _strict_int(identity_hash, where="identity_hash", positive=True)
        self._append_group_new(spec.hypothesis_group)
        self._insert_overflow(identity_hash, spec.overflow_components)

    def register(self, identity_hash: int,
                 spec: GraphObjectIdentitySpec) -> None:
        """在既有命名空间严格追加 codec 子载荷，拒绝孤儿和组碰撞。"""
        _strict_int(identity_hash, where="identity_hash", positive=True)
        rows = self._component_rows(identity_hash)
        if rows:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} 存在孤儿 component")
        self._ensure_group(spec.hypothesis_group)
        self._insert_overflow(identity_hash, spec.overflow_components)

    def read_optional(
            self, identity_hash: int, row: dict[str, Any]
            ) -> tuple[int, ...] | None:
        """从新 codec 主记录和子表恢复完整键；旧行缺全部新列时返回空。"""
        _strict_int(identity_hash, where="identity_hash", positive=True)
        row_hash = _strict_int(
            row["identity_hash"], where="graph_object.identity_hash",
            positive=True)
        if row_hash != identity_hash:
            raise GraphObjectIdentityIncompleteError(
                "graph_object identity_hash 列不一致")
        raw_fields = tuple(
            row.get(name) for name, _ in GRAPH_OBJECT_IDENTITY_COLUMNS)
        if all(value is None for value in raw_fields):
            return None
        if any(value is None for value in raw_fields):
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} identity 主字段不完整")
        codec = _strict_int(
            row["identity_codec"], where="identity_codec", positive=True)
        values = tuple(_strict_int(
            row[f"codec_value_{index}"],
            where=f"codec_value_{index}")
            for index in range(GRAPH_OBJECT_CODEC_VALUE_COUNT))
        component_size = _strict_int(
            row["component_size"], where="component_size", positive=True)
        ref_hash = _strict_int(
            row["codec_ref_hash"],
            where="codec_ref_hash", nonnegative=True)
        overflow_rows = self._component_rows(identity_hash)

        if codec == GRAPH_OBJECT_CODEC_GENERIC_V1:
            components = self._decode_generic(
                identity_hash, component_size, ref_hash,
                values, overflow_rows)
        elif codec == GRAPH_OBJECT_CODEC_OCCURRENCE_SOURCE_V1:
            components = self._decode_occurrence(
                identity_hash, component_size, ref_hash,
                values, overflow_rows)
        elif codec == GRAPH_OBJECT_CODEC_SPAN_SOURCE_V1:
            components = self._decode_span(
                identity_hash, component_size, ref_hash,
                values, overflow_rows)
        elif codec == GRAPH_OBJECT_CODEC_HYPOTHESIS_V1:
            components = self._decode_hypothesis(
                identity_hash, component_size, ref_hash,
                values, overflow_rows)
        else:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} codec 未注册")
        key = self._outer_key(row, components)
        envelope = _split_object_key(key)
        if codec in {
                GRAPH_OBJECT_CODEC_OCCURRENCE_SOURCE_V1,
                GRAPH_OBJECT_CODEC_SPAN_SOURCE_V1}:
            source_key = self._read_reference_key(
                IDENTITY_SOURCE_RECORD, ref_hash, label="source")
            _assert_source_envelope(envelope, _source_key(source_key))
        elif codec == GRAPH_OBJECT_CODEC_HYPOTHESIS_V1:
            group = self._load_group(ref_hash)
            _assert_source_envelope(envelope, group.observation_key)
        return key

    def clear_runtime_caches(self) -> None:
        """外部 load 或故障注入后清空已核验竞争组缓存。"""
        self._groups_by_hash.clear()

    def _append_group_new(self, group: HypothesisGroupSpec | None) -> None:
        """在空命名空间追加首次竞争组，同 hash 异组立即拒绝。"""
        if group is None:
            return
        group_hash = group.group_hash()
        cached = self._groups_by_hash.get(group_hash)
        if cached is not None:
            if cached != group:
                raise GraphObjectIdentityCollisionError(
                    f"hypothesis group hash={group_hash} 发生碰撞")
            return
        self._insert_group(group_hash, group)
        self._groups_by_hash[group_hash] = group

    def _ensure_group(self, group: HypothesisGroupSpec | None) -> None:
        """严格幂等登记竞争组，并核验既有主记录和全部组件。"""
        if group is None:
            return
        group_hash = group.group_hash()
        cached = self._groups_by_hash.get(group_hash)
        if cached is not None:
            if cached != group:
                raise GraphObjectIdentityCollisionError(
                    f"hypothesis group hash={group_hash} 发生碰撞")
            return
        rows = self._backend.select(
            GRAPH_HYPOTHESIS_GROUP_TABLE,
            where={"group_hash": group_hash},
        )
        components = self._group_component_rows(group_hash)
        if rows:
            existing = self._read_group(group_hash, rows, components)
            if existing != group:
                raise GraphObjectIdentityCollisionError(
                    f"hypothesis group hash={group_hash} 命中不同完整组")
            self._groups_by_hash[group_hash] = existing
            return
        if components:
            raise GraphObjectIdentityIncompleteError(
                f"hypothesis group hash={group_hash} 存在孤儿 component")
        self._insert_group(group_hash, group)
        self._groups_by_hash[group_hash] = group

    def _insert_group(self, group_hash: int,
                      group: HypothesisGroupSpec) -> None:
        """先追加完整 kind/competition 组件，再追加唯一竞争组主记录。"""
        for role, values in (
                (GROUP_COMPONENT_HYPOTHESIS_KIND,
                 group.hypothesis_kind),
                (GROUP_COMPONENT_COMPETITION_KEY,
                 group.competition_key)):
            for ordinal, value in enumerate(values):
                self._backend.insert(
                    GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE,
                    {
                        "group_hash": group_hash,
                        "component_role": role,
                        "component_ordinal": ordinal,
                        "component_value": value,
                    },
                )
        self._backend.insert(GRAPH_HYPOTHESIS_GROUP_TABLE, {
            "group_hash": group_hash,
            "hypothesis_version": group.hypothesis_version,
            "hypothesis_kind_size": len(group.hypothesis_kind),
            "competition_size": len(group.competition_key),
            "scope_hash": group.scope_hash,
            "observation_hash": group.observation_hash,
        })

    def _insert_overflow(
            self, identity_hash: int,
            overflow: tuple[tuple[int, int], ...]) -> None:
        """按原逻辑 ordinal 追加对象 overflow 组件。"""
        for ordinal, value in overflow:
            self._backend.insert(GRAPH_OBJECT_COMPONENT_TABLE, {
                "identity_hash": identity_hash,
                "component_ordinal": ordinal,
                "component_value": value,
            })

    def _decode_generic(
            self, identity_hash: int, component_size: int,
            ref_hash: int, values: tuple[int, ...],
            rows: list[dict[str, Any]]) -> tuple[int, ...]:
        """恢复普通对象完整 components，并校验未用固定槽和 overflow。"""
        if ref_hash != 0:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} generic ref 必须为零")
        inline_size = min(component_size, GRAPH_OBJECT_CODEC_VALUE_COUNT)
        self._require_unused_zero(identity_hash, values, inline_size)
        overflow = self._read_overflow(
            identity_hash,
            rows,
            start_ordinal=GRAPH_OBJECT_CODEC_VALUE_COUNT,
            expected_size=max(
                component_size - GRAPH_OBJECT_CODEC_VALUE_COUNT, 0),
        )
        return values[:inline_size] + overflow

    def _decode_occurrence(
            self, identity_hash: int, component_size: int,
            source_hash: int, values: tuple[int, ...],
            rows: list[dict[str, Any]]) -> tuple[int, ...]:
        """从 SourceRef hash 和位置三元组恢复 Occurrence components。"""
        if component_size != 14:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} occurrence 长度非法")
        self._require_unused_zero(identity_hash, values, 3)
        if rows:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} occurrence 不应有 overflow")
        source_key = self._read_reference_key(
            IDENTITY_SOURCE_RECORD, source_hash, label="source")
        return (*_source_key(source_key), *values[:3])

    def _decode_span(
            self, identity_hash: int, component_size: int,
            source_hash: int, values: tuple[int, ...],
            rows: list[dict[str, Any]]) -> tuple[int, ...]:
        """从 SourceRef hash、成员数、固定区间和 overflow 恢复 Span。"""
        member_count = _strict_int(
            values[1], where="span member_count", positive=True)
        expected_size = 13 + member_count * 2
        if component_size != expected_size:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} span 长度非法")
        inline_member_count = min(
            member_count, _SPAN_INLINE_MEMBER_COUNT)
        used_size = 2 + inline_member_count * 2
        self._require_unused_zero(identity_hash, values, used_size)
        overflow = self._read_overflow(
            identity_hash,
            rows,
            start_ordinal=11 + GRAPH_OBJECT_CODEC_VALUE_COUNT,
            expected_size=max(
                member_count - _SPAN_INLINE_MEMBER_COUNT, 0) * 2,
        )
        source_key = self._read_reference_key(
            IDENTITY_SOURCE_RECORD, source_hash, label="source")
        return (
            *_source_key(source_key),
            *values[:used_size],
            *overflow,
        )

    def _decode_hypothesis(
            self, identity_hash: int, component_size: int,
            group_hash: int, values: tuple[int, ...],
            rows: list[dict[str, Any]]) -> tuple[int, ...]:
        """恢复共享竞争组和最长公共前缀压缩后的 Hypothesis components。"""
        group = self._load_group(group_hash)
        hypothesis_version = _strict_int(
            values[0], where="hypothesis_version", positive=True)
        if hypothesis_version != group.hypothesis_version:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} hypothesis version 冲突")
        candidate_size = _strict_int(
            values[1], where="candidate_size", positive=True)
        prefix_size = _strict_int(
            values[2], where="candidate_prefix_size", nonnegative=True)
        if prefix_size > min(candidate_size, len(group.competition_key)):
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} candidate prefix 非法")
        suffix_size = candidate_size - prefix_size
        inline_suffix_size = min(
            suffix_size, _HYPOTHESIS_SUFFIX_INLINE_COUNT)
        used_size = 3 + inline_suffix_size
        self._require_unused_zero(identity_hash, values, used_size)
        suffix = values[3:used_size] + self._read_overflow(
            identity_hash,
            rows,
            start_ordinal=_HYPOTHESIS_SUFFIX_INLINE_COUNT,
            expected_size=max(
                suffix_size - _HYPOTHESIS_SUFFIX_INLINE_COUNT, 0),
        )
        candidate = group.competition_key[:prefix_size] + suffix
        if (len(candidate) != candidate_size
                or _common_prefix_size(
                    candidate, group.competition_key) != prefix_size):
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} candidate prefix 非规范")
        scope_key = self._read_reference_key(
            IDENTITY_SCOPE, group.scope_hash, label="scope")
        observation_key = self._read_reference_key(
            IDENTITY_SOURCE_RECORD,
            group.observation_hash,
            label="observation",
        )
        if scope_key != group.scope_key:
            raise GraphObjectIdentityIncompleteError(
                f"hypothesis group hash={group_hash} scope 引用冲突")
        if observation_key != group.observation_key:
            raise GraphObjectIdentityIncompleteError(
                f"hypothesis group hash={group_hash} observation 引用冲突")
        components = (
            hypothesis_version,
            len(group.hypothesis_kind),
            *group.hypothesis_kind,
            len(candidate),
            *candidate,
            len(group.competition_key),
            *group.competition_key,
            len(scope_key),
            *scope_key,
            len(observation_key),
            *observation_key,
        )
        if len(components) != component_size:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} hypothesis 长度非法")
        return components

    def _load_group(self, group_hash: int) -> HypothesisGroupSpec:
        """读取唯一竞争组；缓存只保存已经完整物理核验的结果。"""
        _strict_int(group_hash, where="group_hash", positive=True)
        cached = self._groups_by_hash.get(group_hash)
        if cached is not None:
            return cached
        rows = self._backend.select(
            GRAPH_HYPOTHESIS_GROUP_TABLE,
            where={"group_hash": group_hash},
        )
        components = self._group_component_rows(group_hash)
        group = self._read_group(group_hash, rows, components)
        self._groups_by_hash[group_hash] = group
        return group

    def _read_group(
            self, group_hash: int,
            rows: list[dict[str, Any]],
            components: list[dict[str, Any]]) -> HypothesisGroupSpec:
        """核验竞争组主记录唯一、两类组件连续并重算 group hash。"""
        if len(rows) != 1:
            raise GraphObjectIdentityIncompleteError(
                f"hypothesis group hash={group_hash} 主记录数量={len(rows)}")
        row = rows[0]
        if _strict_int(
                row["group_hash"], where="group_hash", positive=True
                ) != group_hash:
            raise GraphObjectIdentityIncompleteError("group hash 列不一致")
        kind_size = _strict_int(
            row["hypothesis_kind_size"],
            where="hypothesis_kind_size", positive=True)
        competition_size = _strict_int(
            row["competition_size"],
            where="competition_size", positive=True)
        by_role: dict[int, list[dict[str, Any]]] = {
            GROUP_COMPONENT_HYPOTHESIS_KIND: [],
            GROUP_COMPONENT_COMPETITION_KEY: [],
        }
        for component in components:
            role = _strict_int(
                component["component_role"],
                where="group component_role", positive=True)
            if role not in by_role:
                raise GraphObjectIdentityIncompleteError(
                    f"hypothesis group hash={group_hash} component role 非法")
            by_role[role].append(component)
        hypothesis_kind = self._ordered_group_values(
            group_hash,
            by_role[GROUP_COMPONENT_HYPOTHESIS_KIND],
            kind_size,
        )
        competition_key = self._ordered_group_values(
            group_hash,
            by_role[GROUP_COMPONENT_COMPETITION_KEY],
            competition_size,
        )
        scope_hash = _strict_int(
            row["scope_hash"], where="scope_hash", positive=True)
        observation_hash = _strict_int(
            row["observation_hash"],
            where="observation_hash", positive=True)
        scope_key = self._read_reference_key(
            IDENTITY_SCOPE, scope_hash, label="scope")
        observation_key = self._read_reference_key(
            IDENTITY_SOURCE_RECORD,
            observation_hash,
            label="observation",
        )
        group = HypothesisGroupSpec(
            _strict_int(
                row["hypothesis_version"],
                where="hypothesis_version", positive=True),
            hypothesis_kind,
            competition_key,
            scope_hash,
            scope_key,
            observation_hash,
            observation_key,
        )
        if group.group_hash() != group_hash:
            raise GraphObjectIdentityCollisionError(
                f"hypothesis group hash={group_hash} 重算不一致")
        return group

    @staticmethod
    def _ordered_group_values(
            group_hash: int, rows: list[dict[str, Any]],
            expected_size: int) -> tuple[int, ...]:
        """按 ordinal 恢复一种竞争组组件并拒绝缺失、重复和断序。"""
        ordered = sorted(rows, key=lambda row: row["component_ordinal"])
        if len(ordered) != expected_size:
            raise GraphObjectIdentityIncompleteError(
                f"hypothesis group hash={group_hash} component 数量不一致")
        values: list[int] = []
        for expected_ordinal, row in enumerate(ordered):
            ordinal = _strict_int(
                row["component_ordinal"],
                where="group component_ordinal", nonnegative=True)
            if ordinal != expected_ordinal:
                raise GraphObjectIdentityIncompleteError(
                    f"hypothesis group hash={group_hash} component ordinal 断序")
            values.append(_strict_int(
                row["component_value"], where="group component_value"))
        return tuple(values)

    @staticmethod
    def _read_overflow(
            identity_hash: int,
            rows: list[dict[str, Any]], *,
            start_ordinal: int,
            expected_size: int) -> tuple[int, ...]:
        """核验对象 overflow 数量和逻辑 ordinal 后恢复完整值序。"""
        if len(rows) != expected_size:
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} component 数量不一致")
        values: list[int] = []
        for offset, row in enumerate(rows):
            expected_ordinal = start_ordinal + offset
            ordinal = _strict_int(
                row["component_ordinal"],
                where="component_ordinal", nonnegative=True)
            if ordinal != expected_ordinal:
                raise GraphObjectIdentityIncompleteError(
                    f"graph object hash={identity_hash} component ordinal 断序")
            values.append(_strict_int(
                row["component_value"], where="component_value"))
        return tuple(values)

    @staticmethod
    def _require_unused_zero(
            identity_hash: int, values: tuple[int, ...], used_size: int) -> None:
        """要求固定槽未使用部分为规范零值，防止同一键多种物理表示。"""
        if any(value != 0 for value in values[used_size:]):
            raise GraphObjectIdentityIncompleteError(
                f"graph object hash={identity_hash} 未使用 codec 槽非零")

    def _read_reference_key(
            self, identity_kind: int, identity_hash: int, *,
            label: str) -> tuple[int, ...]:
        """读取 codec 引用的完整键，并把缺失统一转换为物理完整性错误。"""
        _strict_int(identity_hash, where=f"{label}_hash", positive=True)
        try:
            return self._key_reader(identity_kind, identity_hash)
        except KeyError as exc:
            raise GraphObjectIdentityIncompleteError(
                f"graph object codec 引用的 {label} 不存在") from exc

    @staticmethod
    def _outer_key(
            row: dict[str, Any], components: tuple[int, ...]
            ) -> tuple[int, ...]:
        """用主记录 envelope 和已恢复 components 重建 ObjectIdentity 完整键。"""
        object_kind = _strict_int(
            row["object_kind"], where="object_kind", positive=True)
        owner = (
            _strict_int(
                row["owner_tenant_id"],
                where="owner_tenant_id", nonnegative=True),
            _strict_int(
                row["owner_user_id"],
                where="owner_user_id", nonnegative=True),
            _strict_int(
                row["owner_session_id"],
                where="owner_session_id", nonnegative=True),
            _strict_int(
                row["owner_visibility"],
                where="owner_visibility", positive=True),
        )
        versions = tuple(_strict_int(
            row[name], where=name, nonnegative=True)
            for name in (
                "corpus_version",
                "parser_version",
                "primitive_version",
                "curriculum_version",
            ))
        component_size = _strict_int(
            row["component_size"], where="component_size", positive=True)
        if len(components) != component_size:
            raise GraphObjectIdentityIncompleteError(
                "graph object components 与主记录长度不一致")
        key = (object_kind, *owner, *versions, component_size, *components)
        _split_object_key(key)
        return key

    def _component_rows(
            self, identity_hash: int) -> list[dict[str, Any]]:
        """按 ordinal 读取同一 GraphObject 的全部 overflow 行。"""
        return self._backend.select(
            GRAPH_OBJECT_COMPONENT_TABLE,
            where={"identity_hash": identity_hash},
            order_by="component_ordinal",
        )

    def _group_component_rows(
            self, group_hash: int) -> list[dict[str, Any]]:
        """读取同一竞争组全部 kind/competition 组件，不用 limit 掩盖重复。"""
        return self._backend.select(
            GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE,
            where={"group_hash": group_hash},
        )


__all__ = [
    "GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE",
    "GRAPH_HYPOTHESIS_GROUP_TABLE",
    "GRAPH_OBJECT_CODEC_GENERIC_V1",
    "GRAPH_OBJECT_CODEC_HYPOTHESIS_V1",
    "GRAPH_OBJECT_CODEC_OCCURRENCE_SOURCE_V1",
    "GRAPH_OBJECT_CODEC_SPAN_SOURCE_V1",
    "GRAPH_OBJECT_COMPONENT_TABLE",
    "GRAPH_OBJECT_IDENTITY_COLUMNS",
    "GraphObjectIdentityCollisionError",
    "GraphObjectIdentityError",
    "GraphObjectIdentityIncompleteError",
    "GraphObjectIdentitySpec",
    "GraphObjectIdentityStore",
    "HypothesisGroupSpec",
    "register_graph_object_identity_tables",
]
