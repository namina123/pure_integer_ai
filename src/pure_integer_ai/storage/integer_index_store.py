"""整数词元/聚合索引的可迁移存储协议。

课程 sidecar 负责交换；本模块把 sidecar 的唯一内容投影到训练后端。核心
表只保存非负整数：Unicode 码点、token sequence/aggregate 成员和 occurrence 引用。
来源、scope、位置仍由 occurrence 表保存，因而内容去重不会吞掉语境身份。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.integer_token_index import (
    IntegerAggregateIndex,
    IntegerTokenIndex,
)
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    StorageBackend,
    TYPE_INT,
    register_extension_table,
)


INTEGER_INDEX_HEADER_TABLE = "integer_index_header"
INTEGER_INDEX_MEMBER_TABLE = "integer_index_member"
INTEGER_INDEX_OCCURRENCE_TABLE = "integer_index_occurrence"
_HASH_WORDS = 8
INDEX_KIND_TOKEN = 1
INDEX_KIND_AGGREGATE = 2
LAYER_KIND_VOCABULARY = 1
LAYER_KIND_TOKEN_SEQUENCE = 2
LAYER_KIND_AGGREGATE = 3
MEMBER_KIND_ATOM = 1
MEMBER_KIND_TOKEN_SEQUENCE = 2
MEMBER_KIND_AGGREGATE = 3
MEMBER_KIND_CODEPOINT = 4

_HASH_COLUMNS = [(f"hash_{index}", TYPE_INT) for index in range(_HASH_WORDS)]
_HEADER_COLUMNS = [
    ("index_kind", TYPE_INT), *_HASH_COLUMNS,
    *[(f"token_hash_{index}", TYPE_INT) for index in range(_HASH_WORDS)],
    ("token_sequence_count", TYPE_INT), ("aggregate_count", TYPE_INT),
]
_MEMBER_COLUMNS = [
    *_HASH_COLUMNS, ("layer_kind", TYPE_INT), ("parent_ordinal", TYPE_INT),
    ("member_ordinal", TYPE_INT), ("member_kind", TYPE_INT),
    ("member_ref", TYPE_INT),
]
_OCCURRENCE_COLUMNS = [
    *_HASH_COLUMNS, ("occurrence_ordinal", TYPE_INT),
    ("target_kind", TYPE_INT), ("target_ordinal", TYPE_INT),
]


def register_integer_index_tables(backend: StorageBackend) -> None:
    """注册唯一索引、成员和 occurrence 引用表。"""
    register_extension_table(
        backend, INTEGER_INDEX_HEADER_TABLE, _HEADER_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [tuple(name for name, _ in _HASH_COLUMNS), ("index_kind",)],
        recovery_key=tuple(name for name, _ in _HASH_COLUMNS),
    )
    register_extension_table(
        backend, INTEGER_INDEX_MEMBER_TABLE, _MEMBER_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [tuple(name for name, _ in _HASH_COLUMNS),
         tuple(name for name, _ in _HASH_COLUMNS) + ("layer_kind", "parent_ordinal")],
        recovery_key=tuple(name for name, _ in _HASH_COLUMNS)
        + ("layer_kind", "parent_ordinal", "member_ordinal"),
    )
    register_extension_table(
        backend, INTEGER_INDEX_OCCURRENCE_TABLE, _OCCURRENCE_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [tuple(name for name, _ in _HASH_COLUMNS), ("occurrence_ordinal",)],
        recovery_key=tuple(name for name, _ in _HASH_COLUMNS)
        + ("occurrence_ordinal",),
    )


def _hash_words(sha256: str) -> tuple[int, ...]:
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("索引 SHA-256 必须为 64 位十六进制文本")
    try:
        raw = bytes.fromhex(sha256)
    except ValueError as error:
        raise ValueError("索引 SHA-256 非法") from error
    if len(raw) != 32:
        raise ValueError("索引 SHA-256 非法")
    return tuple(int.from_bytes(raw[offset:offset + 4], "big")
                 for offset in range(0, 32, 4))


def _hash_where(words: tuple[int, ...]) -> dict[str, int]:
    return {f"hash_{index}": value for index, value in enumerate(words)}


@dataclass(frozen=True, slots=True)
class IntegerIndexBinding:
    """已登记索引的整数身份和规模摘要。"""

    index_kind: int
    hash_words: tuple[int, ...]
    token_hash_words: tuple[int, ...]
    token_sequence_count: int
    aggregate_count: int
    member_count: int


class IntegerIndexStore:
    """把 token/aggregate sidecar 幂等投影到任意 StorageBackend。"""

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend
        register_integer_index_tables(backend)

    def bind(self, token_index: IntegerTokenIndex,
             aggregate_index: IntegerAggregateIndex | None = None
             ) -> IntegerIndexBinding:
        """登记唯一 sequence/aggregate 成员；重复绑定逐行幂等。"""
        if not isinstance(token_index, IntegerTokenIndex):
            raise TypeError("bind 需要 IntegerTokenIndex")
        if aggregate_index is not None:
            if not isinstance(aggregate_index, IntegerAggregateIndex):
                raise TypeError("aggregate_index 类型非法")
            # Validate every aggregate occurrence before projecting rows.  A
            # single first-occurrence probe can miss a damaged later member or
            # a cycle reachable only from another occurrence.
            for ordinal in range(len(aggregate_index.occurrence_ordinals)):
                aggregate_index.render(token_index, ordinal)
        token_words = _hash_words(token_index.sha256)
        aggregate_words = (None if aggregate_index is None
                           else _hash_words(aggregate_index.sha256))
        if aggregate_index is None:
            index_kind, index_words = INDEX_KIND_TOKEN, token_words
            token_sequence_count = len(token_index.sequences)
            aggregate_count = 0
            if any(len(atom) != 1 for atom in token_index.vocabulary):
                raise ValueError("token vocabulary atom 必须是单个 Unicode 码点")
        else:
            index_kind, index_words = INDEX_KIND_AGGREGATE, aggregate_words
            token_sequence_count = aggregate_index.token_sequence_count
            aggregate_count = len(aggregate_index.aggregate_sequences)
        assert index_words is not None
        header = {
            "index_kind": index_kind,
            **_hash_where(index_words),
            **{f"token_hash_{i}": value
               for i, value in enumerate(token_words)},
            "token_sequence_count": token_sequence_count,
            "aggregate_count": aggregate_count,
        }
        self._ensure_header(header)
        members: list[dict[str, int]] = []
        occurrences: list[dict[str, int]] = []
        if aggregate_index is None:
            for parent, atom in enumerate(token_index.vocabulary):
                members.append({**_hash_where(index_words),
                                "layer_kind": LAYER_KIND_VOCABULARY,
                                "parent_ordinal": parent,
                                "member_ordinal": 0,
                                "member_kind": MEMBER_KIND_CODEPOINT,
                                "member_ref": ord(atom)})
            for parent, sequence in enumerate(token_index.sequences):
                for member, atom in enumerate(sequence):
                    members.append({**_hash_where(index_words),
                                    "layer_kind": LAYER_KIND_TOKEN_SEQUENCE,
                                    "parent_ordinal": parent,
                                    "member_ordinal": member,
                                    "member_kind": MEMBER_KIND_ATOM,
                                    "member_ref": atom})
            for ordinal, target in enumerate(token_index.occurrence_ordinals):
                occurrences.append({**_hash_where(index_words),
                                    "occurrence_ordinal": ordinal,
                                    "target_kind": MEMBER_KIND_TOKEN_SEQUENCE,
                                    "target_ordinal": target})
        else:
            for parent, sequence in enumerate(aggregate_index.aggregate_sequences):
                for member, ref in enumerate(sequence):
                    members.append({**_hash_where(index_words),
                                    "layer_kind": LAYER_KIND_AGGREGATE,
                                    "parent_ordinal": parent,
                                    "member_ordinal": member,
                                    "member_kind": (
                                        MEMBER_KIND_TOKEN_SEQUENCE
                                        if ref < aggregate_index.token_sequence_count
                                        else MEMBER_KIND_AGGREGATE),
                                    "member_ref": (
                                        ref if ref < aggregate_index.token_sequence_count
                                        else ref - aggregate_index.token_sequence_count),
                    })
            for ordinal, target in enumerate(aggregate_index.occurrence_ordinals):
                occurrences.append({**_hash_where(index_words),
                                    "occurrence_ordinal": ordinal,
                                    "target_kind": MEMBER_KIND_AGGREGATE,
                                    "target_ordinal": target})
        self._ensure_members(members)
        for occurrence in occurrences:
            self._ensure_occurrence(occurrence)
        return IntegerIndexBinding(
            index_kind, index_words, token_words, token_sequence_count,
            aggregate_count, len(members),
        )

    def _ensure_header(self, row: dict[str, int]) -> None:
        rows = self.backend.select(INTEGER_INDEX_HEADER_TABLE,
                                   where=_hash_where(tuple(row[f"hash_{i}"]
                                                           for i in range(_HASH_WORDS))))
        if rows:
            if len(rows) != 1 or rows[0] != row:
                raise ValueError("整数索引身份已绑定冲突 header")
            return
        self.backend.insert(INTEGER_INDEX_HEADER_TABLE, row)

    def _ensure_members(self, rows: list[dict[str, int]]) -> None:
        if not rows:
            return
        words = tuple(rows[0][f"hash_{i}"] for i in range(_HASH_WORDS))
        def key(row: dict[str, int]) -> tuple[int, int, int]:
            return (row["layer_kind"], row["parent_ordinal"],
                    row["member_ordinal"])

        existing: dict[tuple[int, int, int], dict[str, int]] = {}
        for row in self.backend.select(
                INTEGER_INDEX_MEMBER_TABLE, where=_hash_where(words)):
            identity = key(row)
            if identity in existing:
                raise ValueError("整数索引已存在重复 member identity")
            existing[identity] = row
        incoming: dict[tuple[int, int, int], dict[str, int]] = {}
        for row in rows:
            identity = key(row)
            previous = incoming.get(identity)
            if previous is not None and previous != row:
                raise ValueError("整数索引输入含冲突 member identity")
            incoming[identity] = row
        missing = []
        for identity, row in incoming.items():
            previous = existing.get(identity)
            if previous is None:
                missing.append(row)
            elif previous != row:
                raise ValueError("整数索引身份已绑定冲突 member")
        if missing:
            self.backend.insert_many(INTEGER_INDEX_MEMBER_TABLE, missing)

    def replay_occurrence(
            self,
            token_index_sha256: str,
            occurrence_ordinal: int,
            *,
            aggregate_index_sha256: str | None = None,
            ) -> tuple[int, ...]:
        """从纯整数表按需重现一次 occurrence 的 Unicode 码点序列。"""
        if type(occurrence_ordinal) is not int or occurrence_ordinal < 0:
            raise ValueError("occurrence ordinal 必须是非负整数")
        token_words = _hash_words(token_index_sha256)
        token_header = self._read_header(token_words, INDEX_KIND_TOKEN)
        if aggregate_index_sha256 is None:
            target = self._read_occurrence(
                token_words, occurrence_ordinal, MEMBER_KIND_TOKEN_SEQUENCE)
            return self._replay_token_sequence(
                token_words, token_header, target)
        aggregate_words = _hash_words(aggregate_index_sha256)
        aggregate_header = self._read_header(
            aggregate_words, INDEX_KIND_AGGREGATE)
        bound_token_words = tuple(
            aggregate_header[f"token_hash_{index}"]
            for index in range(_HASH_WORDS))
        if bound_token_words != token_words:
            raise ValueError("aggregate header 绑定的 token index 漂移")
        if (aggregate_header["token_sequence_count"]
                != token_header["token_sequence_count"]):
            raise ValueError("aggregate header 绑定的 token sequence count 漂移")
        target = self._read_occurrence(
            aggregate_words, occurrence_ordinal, MEMBER_KIND_AGGREGATE)
        return self._replay_aggregate(
            aggregate_words, aggregate_header, token_words, token_header,
            target, set())

    def render_occurrence(
            self,
            token_index_sha256: str,
            occurrence_ordinal: int,
            *,
            aggregate_index_sha256: str | None = None,
            ) -> str:
        """把整数回放结果转换为表层文本；转换只发生在消费边界。"""
        return "".join(chr(value) for value in self.replay_occurrence(
            token_index_sha256, occurrence_ordinal,
            aggregate_index_sha256=aggregate_index_sha256))

    def _read_header(self, words: tuple[int, ...],
                     expected_kind: int) -> dict[str, int]:
        rows = self.backend.select(
            INTEGER_INDEX_HEADER_TABLE, where=_hash_where(words))
        if len(rows) != 1 or rows[0]["index_kind"] != expected_kind:
            raise ValueError("整数索引 header 缺失、重复或类型不匹配")
        return rows[0]

    def _read_occurrence(self, words: tuple[int, ...], ordinal: int,
                         expected_kind: int) -> int:
        rows = self.backend.select(
            INTEGER_INDEX_OCCURRENCE_TABLE,
            where={**_hash_where(words), "occurrence_ordinal": ordinal})
        if len(rows) != 1 or rows[0]["target_kind"] != expected_kind:
            raise ValueError("整数索引 occurrence 缺失、重复或类型不匹配")
        target = rows[0]["target_ordinal"]
        if type(target) is not int or target < 0:
            raise ValueError("整数索引 occurrence target 非法")
        return target

    def _ordered_members(
            self, words: tuple[int, ...], layer_kind: int,
            parent_ordinal: int,
            ) -> list[dict[str, int]]:
        where = {**_hash_where(words), "layer_kind": layer_kind,
                 "parent_ordinal": parent_ordinal}
        rows = self.backend.select(
            INTEGER_INDEX_MEMBER_TABLE, where=where,
            order_by="member_ordinal")
        if not rows or any(
                row["member_ordinal"] != ordinal
                for ordinal, row in enumerate(rows)):
            raise ValueError("整数索引 member 序列缺失、重复或不连续")
        return rows

    def _replay_token_sequence(
            self, words: tuple[int, ...], header: dict[str, int],
            sequence_ordinal: int,
            ) -> tuple[int, ...]:
        if sequence_ordinal >= header["token_sequence_count"]:
            raise ValueError("token sequence ordinal 越界")
        result = []
        for row in self._ordered_members(
                words, LAYER_KIND_TOKEN_SEQUENCE, sequence_ordinal):
            if row["member_kind"] != MEMBER_KIND_ATOM:
                raise ValueError("token sequence member kind 非法")
            atom_ordinal = row["member_ref"]
            atom_rows = self._ordered_members(
                words, LAYER_KIND_VOCABULARY, atom_ordinal)
            if len(atom_rows) != 1:
                raise ValueError("vocabulary atom 必须恰有一个 Unicode 码点")
            if atom_rows[0]["member_kind"] != MEMBER_KIND_CODEPOINT:
                raise ValueError("vocabulary member kind 非法")
            codepoint = atom_rows[0]["member_ref"]
            if not 0 <= codepoint <= 0x10FFFF:
                raise ValueError("Unicode 码点越界")
            result.append(codepoint)
        return tuple(result)

    def _replay_aggregate(
            self, aggregate_words: tuple[int, ...],
            aggregate_header: dict[str, int],
            token_words: tuple[int, ...], token_header: dict[str, int],
            aggregate_ordinal: int, active: set[int],
            ) -> tuple[int, ...]:
        if (aggregate_ordinal >= aggregate_header["aggregate_count"]
                or aggregate_ordinal in active):
            raise ValueError("aggregate ordinal 越界或引用成环")
        active.add(aggregate_ordinal)
        result: list[int] = []
        try:
            for row in self._ordered_members(
                    aggregate_words, LAYER_KIND_AGGREGATE,
                    aggregate_ordinal):
                if row["member_kind"] == MEMBER_KIND_TOKEN_SEQUENCE:
                    result.extend(self._replay_token_sequence(
                        token_words, token_header, row["member_ref"]))
                elif row["member_kind"] == MEMBER_KIND_AGGREGATE:
                    if row["member_ref"] >= aggregate_ordinal:
                        raise ValueError("aggregate 出现前向引用")
                    result.extend(self._replay_aggregate(
                        aggregate_words, aggregate_header,
                        token_words, token_header, row["member_ref"], active))
                else:
                    raise ValueError("aggregate member kind 非法")
        finally:
            active.remove(aggregate_ordinal)
        return tuple(result)

    def _ensure_occurrence(self, row: dict[str, int]) -> None:
        words = tuple(row[f"hash_{i}"] for i in range(_HASH_WORDS))
        where = {**_hash_where(words),
                 "occurrence_ordinal": row["occurrence_ordinal"]}
        rows = self.backend.select(INTEGER_INDEX_OCCURRENCE_TABLE, where=where)
        if rows:
            if len(rows) != 1 or rows[0] != row:
                raise ValueError("整数 aggregate occurrence 引用冲突")
            return
        self.backend.insert(INTEGER_INDEX_OCCURRENCE_TABLE, row)


__all__ = [
    "INDEX_KIND_AGGREGATE", "INDEX_KIND_TOKEN", "IntegerIndexBinding",
    "IntegerIndexStore", "INTEGER_INDEX_HEADER_TABLE",
    "INTEGER_INDEX_MEMBER_TABLE", "INTEGER_INDEX_OCCURRENCE_TABLE",
    "MEMBER_KIND_AGGREGATE", "MEMBER_KIND_ATOM", "MEMBER_KIND_TOKEN_SEQUENCE",
    "MEMBER_KIND_CODEPOINT", "LAYER_KIND_AGGREGATE",
    "LAYER_KIND_TOKEN_SEQUENCE", "LAYER_KIND_VOCABULARY",
    "register_integer_index_tables",
]
