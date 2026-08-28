"""普通对话后继命题的纯整数检索投影。

权威关系仍位于一等 SemanticGraph、Occurrence、SourceRecord 与来源顺序中；这里
只保存可删除重建的稀疏 feature posting 和端点摘要，使生成侧无需扫描完整长期图。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    StorageBackend,
    TYPE_INT,
    register_extension_table,
)


DIALOGUE_SUCCESSOR_TABLE = "dialogue_successor_projection"
DIALOGUE_SUCCESSOR_FEATURE_TABLE = "dialogue_successor_feature"

FEATURE_CURRENT_TURN = 1
FEATURE_HISTORY_TURN = 2
_FEATURE_KINDS = frozenset({FEATURE_CURRENT_TURN, FEATURE_HISTORY_TURN})

_PROJECTION_COLUMNS = [
    ("protocol_version", TYPE_INT),
    ("proposition_space_id", TYPE_INT),
    ("proposition_local_id", TYPE_INT),
    ("source_hash", TYPE_INT),
    ("current_start_space_id", TYPE_INT),
    ("current_start_local_id", TYPE_INT),
    ("current_end_space_id", TYPE_INT),
    ("current_end_local_id", TYPE_INT),
    ("response_start_space_id", TYPE_INT),
    ("response_start_local_id", TYPE_INT),
    ("response_end_space_id", TYPE_INT),
    ("response_end_local_id", TYPE_INT),
    ("current_feature_count", TYPE_INT),
    ("history_feature_count", TYPE_INT),
    ("context_turn_count", TYPE_INT),
    ("response_turn_ordinal", TYPE_INT),
    ("graph_assertion_count", TYPE_INT),
    ("graph_assertion_digest", TYPE_INT),
    ("evidence_id", TYPE_INT),
]

_FEATURE_COLUMNS = [
    ("proposition_space_id", TYPE_INT),
    ("proposition_local_id", TYPE_INT),
    ("feature_kind", TYPE_INT),
    ("feature_ordinal", TYPE_INT),
    ("feature_hash", TYPE_INT),
    ("occurrence_space_id", TYPE_INT),
    ("occurrence_local_id", TYPE_INT),
    ("occurrence_codepoint_ordinal", TYPE_INT),
    ("turn_distance", TYPE_INT),
]


def register_dialogue_successor_tables(backend: StorageBackend) -> None:
    """注册可由权威图和 occurrence 重建的 append-only 稀疏投影。"""
    register_extension_table(
        backend,
        DIALOGUE_SUCCESSOR_TABLE,
        _PROJECTION_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("proposition_space_id", "proposition_local_id"),
            ("source_hash",),
        ],
        recovery_key=("proposition_space_id", "proposition_local_id"),
    )
    register_extension_table(
        backend,
        DIALOGUE_SUCCESSOR_FEATURE_TABLE,
        _FEATURE_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("feature_hash",),
            ("proposition_space_id", "proposition_local_id"),
        ],
        recovery_key=(
            "proposition_space_id", "proposition_local_id",
            "feature_kind", "feature_ordinal",
        ),
    )


@dataclass(frozen=True, slots=True)
class DialogueSuccessorProjection:
    """一个已被图命题和支持 Evidence 承重的对话后继索引摘要。"""

    protocol_version: int
    proposition_space_id: int
    proposition_local_id: int
    source_hash: int
    current_start_space_id: int
    current_start_local_id: int
    current_end_space_id: int
    current_end_local_id: int
    response_start_space_id: int
    response_start_local_id: int
    response_end_space_id: int
    response_end_local_id: int
    current_feature_count: int
    history_feature_count: int
    context_turn_count: int
    response_turn_ordinal: int
    graph_assertion_count: int
    graph_assertion_digest: int
    evidence_id: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name)
                       for name, _type in _PROJECTION_COLUMNS)
        assert_int(*values, _where="DialogueSuccessorProjection")
        if (any(type(value) is not int for value in values)
                or min(values[:12]) <= 0
                or self.current_feature_count <= 0
                or self.history_feature_count < 0
                or self.context_turn_count <= 0
                or self.response_turn_ordinal <= 1
                or self.graph_assertion_count <= 0
                or self.graph_assertion_digest <= 0
                or self.evidence_id <= 0):
            raise ValueError("dialogue successor projection 字段非法")

    def row(self) -> dict[str, int]:
        """返回与冻结列序无关的整数行。"""
        return {name: getattr(self, name)
                for name, _type in _PROJECTION_COLUMNS}


@dataclass(frozen=True, slots=True)
class DialogueSuccessorFeature:
    """一个输入 feature 到来源 occurrence 的可核验 posting。"""

    proposition_space_id: int
    proposition_local_id: int
    feature_kind: int
    feature_ordinal: int
    feature_hash: int
    occurrence_space_id: int
    occurrence_local_id: int
    occurrence_codepoint_ordinal: int
    turn_distance: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name)
                       for name, _type in _FEATURE_COLUMNS)
        assert_int(*values, _where="DialogueSuccessorFeature")
        if (any(type(value) is not int for value in values)
                or min(self.proposition_space_id,
                       self.proposition_local_id,
                       self.feature_hash,
                       self.occurrence_space_id,
                       self.occurrence_local_id) <= 0
                or self.feature_kind not in _FEATURE_KINDS
                or self.feature_ordinal < 0
                or self.occurrence_codepoint_ordinal < 0
                or self.turn_distance <= 0):
            raise ValueError("dialogue successor feature 字段非法")

    def row(self) -> dict[str, int]:
        """返回与冻结列序无关的整数行。"""
        return {name: getattr(self, name) for name, _type in _FEATURE_COLUMNS}


class DialogueSuccessorProjectionStore:
    """幂等追加后继投影；同一命题任何字段漂移都 fail closed。"""

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend
        register_dialogue_successor_tables(backend)

    @staticmethod
    def _projection_from_row(row: dict) -> DialogueSuccessorProjection:
        return DialogueSuccessorProjection(*(
            row[name] for name, _type in _PROJECTION_COLUMNS))

    @staticmethod
    def _feature_from_row(row: dict) -> DialogueSuccessorFeature:
        return DialogueSuccessorFeature(*(
            row[name] for name, _type in _FEATURE_COLUMNS))

    def preflight(
            self,
            projection: DialogueSuccessorProjection,
            features: tuple[DialogueSuccessorFeature, ...],
            ) -> bool:
        """在图写入前核验投影可新建或精确重放，返回是否已存在。"""
        if not isinstance(projection, DialogueSuccessorProjection):
            raise TypeError("projection 类型错误")
        if (not isinstance(features, tuple) or not features
                or any(not isinstance(item, DialogueSuccessorFeature)
                       for item in features)):
            raise TypeError("features 必须是非空 DialogueSuccessorFeature tuple")
        key = {
            "proposition_space_id": projection.proposition_space_id,
            "proposition_local_id": projection.proposition_local_id,
        }
        expected_ordinals = {
            kind: tuple(range(sum(item.feature_kind == kind
                                  for item in features)))
            for kind in _FEATURE_KINDS
        }
        actual_ordinals = {
            kind: tuple(sorted(item.feature_ordinal for item in features
                               if item.feature_kind == kind))
            for kind in _FEATURE_KINDS
        }
        if actual_ordinals != expected_ordinals:
            raise ValueError("dialogue successor feature ordinal 不连续")
        if (sum(item.feature_kind == FEATURE_CURRENT_TURN for item in features)
                != projection.current_feature_count
                or sum(item.feature_kind == FEATURE_HISTORY_TURN
                       for item in features)
                != projection.history_feature_count
                or any((item.proposition_space_id,
                        item.proposition_local_id)
                       != (projection.proposition_space_id,
                           projection.proposition_local_id)
                       for item in features)):
            raise ValueError("dialogue successor feature 与 projection 不一致")
        rows = self.backend.select(DIALOGUE_SUCCESSOR_TABLE, where=key)
        if not rows:
            return False
        if (len(rows) != 1
                or self._projection_from_row(rows[0]) != projection):
            raise RuntimeError("dialogue successor 命题绑定了冲突 projection")
        stored = tuple(sorted(
            (self._feature_from_row(row)
             for row in self.backend.select(
                 DIALOGUE_SUCCESSOR_FEATURE_TABLE, where=key)),
            key=lambda item: (item.feature_kind, item.feature_ordinal),
        ))
        expected = tuple(sorted(
            features,
            key=lambda item: (item.feature_kind, item.feature_ordinal),
        ))
        if stored != expected:
            raise RuntimeError("dialogue successor 命题绑定了冲突 feature")
        return True

    def record(
            self,
            projection: DialogueSuccessorProjection,
            features: tuple[DialogueSuccessorFeature, ...],
            ) -> DialogueSuccessorProjection:
        """先完整核验再追加 projection 和 posting；精确重放不重复写。"""
        if self.preflight(projection, features):
            return projection
        self.backend.insert(DIALOGUE_SUCCESSOR_TABLE, projection.row())
        for item in sorted(
                features,
                key=lambda value: (value.feature_kind,
                                   value.feature_ordinal)):
            self.backend.insert(DIALOGUE_SUCCESSOR_FEATURE_TABLE, item.row())
        return projection

    def counts(self) -> tuple[int, int]:
        """返回后继命题和 feature posting 的当前数量。"""
        return (
            self.backend.count(DIALOGUE_SUCCESSOR_TABLE),
            self.backend.count(DIALOGUE_SUCCESSOR_FEATURE_TABLE),
        )


__all__ = [
    "DIALOGUE_SUCCESSOR_FEATURE_TABLE",
    "DIALOGUE_SUCCESSOR_TABLE",
    "DialogueSuccessorFeature",
    "DialogueSuccessorProjection",
    "DialogueSuccessorProjectionStore",
    "FEATURE_CURRENT_TURN",
    "FEATURE_HISTORY_TURN",
    "register_dialogue_successor_tables",
]
