"""可跨语言重放的整数词元字典、唯一序列与出现序列。

课程/资料层只保存一次重复 ``surface``，每条内容保存非负整数 ``token_ids``。
出现边通过 occurrence ordinal 重建；这层不改变现有 ``CollectedItem.tokens`` API，属于
可渐进接入的存储与训练传输层。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


def _u8(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ValueError("整数词元字段必须是非负整数")
    width = max(1, (value.bit_length() + 7) // 8)
    return width.to_bytes(2, "big") + value.to_bytes(width, "big")


@dataclass(frozen=True, slots=True)
class IntegerTokenIndex:
    """词元字典、唯一序列、出现 ordinal 和可审计摘要。"""

    vocabulary: tuple[str, ...]
    sequences: tuple[tuple[int, ...], ...]
    sequence_keys: tuple[str, ...]
    occurrence_ordinals: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        if not self.vocabulary or self.vocabulary != tuple(sorted(set(self.vocabulary))):
            raise ValueError("vocabulary 必须去重并按 codepoint 排序")
        if len(self.sequence_keys) != len(self.occurrence_ordinals):
            raise ValueError("occurrence identity 数量不一致")
        limit = len(self.vocabulary)
        if any(type(item) is not int or item < 0 or item >= limit
               for seq in self.sequences for item in seq):
            raise ValueError("token id 越界")
        if any(type(item) is not int or item < 0 or item >= len(self.sequences)
               for item in self.occurrence_ordinals):
            raise ValueError("occurrence sequence ordinal 越界")

    def render(self, ordinal: int) -> str:
        if (type(ordinal) is not int
                or not 0 <= ordinal < len(self.occurrence_ordinals)):
            raise IndexError("sequence ordinal 越界")
        sequence_ordinal = self.occurrence_ordinals[ordinal]
        return self.render_sequence(sequence_ordinal)

    def render_sequence(self, sequence_ordinal: int) -> str:
        """按唯一 token sequence ordinal 渲染，不经过 occurrence 表。"""
        if (type(sequence_ordinal) is not int
                or not 0 <= sequence_ordinal < len(self.sequences)):
            raise IndexError("token sequence ordinal 越界")
        return "".join(self.vocabulary[item]
                       for item in self.sequences[sequence_ordinal])

    def token_count(self) -> int:
        return sum(len(self.sequences[item]) for item in self.occurrence_ordinals)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "PURE_INTEGER_AI_INTEGER_TOKEN_INDEX_V2",
            "vocabulary": list(self.vocabulary),
            "sequences": [list(item) for item in self.sequences],
            "sequence_keys": list(self.sequence_keys),
            "occurrence_ordinals": list(self.occurrence_ordinals),
            "sha256": self.sha256,
        }


def build_integer_token_index(
        texts: Iterable[str], *, sequence_keys: Iterable[str] | None = None,
        ) -> IntegerTokenIndex:
    values = tuple(texts)
    if not values or any(type(item) is not str or not item for item in values):
        raise ValueError("texts 必须是非空文本序列")
    keys = tuple(sequence_keys or (str(index) for index in range(len(values))))
    if len(keys) != len(values) or len(set(keys)) != len(keys):
        raise ValueError("sequence_keys 必须与 texts 一一对应且唯一")
    vocabulary = tuple(sorted(set(char for text in values for char in text)))
    lookup = {item: index for index, item in enumerate(vocabulary)}
    unique_sequences: list[tuple[int, ...]] = []
    sequence_lookup: dict[str, int] = {}
    occurrence_ordinals: list[int] = []
    for text in values:
        sequence = tuple(lookup[char] for char in text)
        ordinal = sequence_lookup.get(text)
        if ordinal is None:
            ordinal = len(unique_sequences)
            sequence_lookup[text] = ordinal
            unique_sequences.append(sequence)
        occurrence_ordinals.append(ordinal)
    sequences = tuple(unique_sequences)
    payload = bytearray(b"PURE-INTEGER-AI/INTEGER-TOKEN-INDEX/V1")
    for token in vocabulary:
        encoded = token.encode("utf-8")
        payload.extend(_u8(len(encoded)))
        payload.extend(encoded)
    for ordinal, sequence in enumerate(sequences):
        payload.extend(_u8(len(sequence)))
        for value in sequence:
            payload.extend(_u8(value))
    for key, ordinal in zip(keys, occurrence_ordinals):
        encoded = key.encode("utf-8")
        payload.extend(_u8(len(encoded)))
        payload.extend(encoded)
        payload.extend(_u8(ordinal))
    payload.extend(_u8(len(occurrence_ordinals)))
    for ordinal in occurrence_ordinals:
        payload.extend(_u8(ordinal))
    return IntegerTokenIndex(
        vocabulary, sequences, keys, tuple(occurrence_ordinals),
        hashlib.sha256(payload).hexdigest())


def write_integer_token_index(path: str, index: IntegerTokenIndex) -> None:
    """写规范 JSON 交换文件；SQLite/自研后端可直接消费同一整数图。"""
    payload = json.dumps(index.to_dict(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")) + "\n"
    # Encode explicitly so the on-disk bytes are identical on every host.
    with open(path, "xb") as stream:
        stream.write(payload.encode("utf-8"))


def load_integer_token_index(path: str | Path) -> IntegerTokenIndex:
    """回读并验证规范 sidecar。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != \
            "PURE_INTEGER_AI_INTEGER_TOKEN_INDEX_V2":
        raise ValueError("integer token index format 非法")
    vocabulary = tuple(value.get("vocabulary", ()))
    sequences = tuple(tuple(item) for item in value.get("sequences", ()))
    keys = tuple(value.get("sequence_keys", ()))
    raw_occurrences = value.get("occurrence_ordinals")
    occurrences = (tuple(range(len(sequences))) if raw_occurrences is None
                   else tuple(raw_occurrences))
    index = build_integer_token_index(
        ("".join(vocabulary[item] for item in sequences[ordinal])
         for ordinal in occurrences), sequence_keys=keys)
    if index.sha256 != value.get("sha256"):
        raise ValueError("integer token index hash 漂移")
    return index


@dataclass(frozen=True, slots=True)
class IntegerAggregateIndex:
    """由 token-sequence 或既有 aggregate 引用组成的唯一聚合索引。

    ``aggregate_sequences`` 中的引用使用一个纯整数命名空间：
    ``0 <= ref < token_sequence_count`` 表示 token sequence，较大的值表示
    ``ref - token_sequence_count`` 号 aggregate。构造时只允许引用已经出现
    的 aggregate，因此图天然是有向无环的；Occurrence 只保存 ordinal。
    """

    token_index_sha256: str
    token_sequence_count: int
    aggregate_sequences: tuple[tuple[int, ...], ...]
    aggregate_keys: tuple[str, ...]
    occurrence_keys: tuple[str, ...]
    occurrence_ordinals: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.token_index_sha256, str)
                or len(self.token_index_sha256) != 64):
            raise ValueError("token index SHA-256 非法")
        if (type(self.token_sequence_count) is not int
                or self.token_sequence_count <= 0):
            raise ValueError("token sequence count 必须为正整数")
        if (len(self.aggregate_sequences) != len(self.aggregate_keys)
                or not self.aggregate_keys
                or len(set(self.aggregate_keys)) != len(self.aggregate_keys)):
            raise ValueError("aggregate key 必须非空且唯一")
        if (not self.occurrence_keys
                or len(set(self.occurrence_keys)) != len(self.occurrence_keys)
                or len(self.occurrence_keys) != len(self.occurrence_ordinals)):
            raise ValueError("occurrence key/ordinal 不稳定")
        limit = self.token_sequence_count + len(self.aggregate_sequences)
        for sequence in self.aggregate_sequences:
            if (not sequence
                    or any(type(ref) is not int or ref < 0 or ref >= limit
                           for ref in sequence)):
                raise ValueError("aggregate 引用越界")
        if any(type(ref) is not int or ref < 0
               or ref >= len(self.aggregate_sequences)
               for ref in self.occurrence_ordinals):
            raise ValueError("aggregate occurrence ordinal 越界")

    def _render_ref(self, ref: int, token_index: IntegerTokenIndex,
                    active: set[int]) -> str:
        """递归展开单个引用；active 同时防御损坏 sidecar 环。"""
        if ref < self.token_sequence_count:
            if ref >= len(token_index.sequences):
                raise ValueError("aggregate 引用的 token sequence 越界")
            return token_index.render_sequence(ref)
        ordinal = ref - self.token_sequence_count
        if ordinal in active:
            raise ValueError("aggregate sidecar 出现环")
        if ordinal >= len(self.aggregate_sequences):
            raise ValueError("aggregate 引用越界")
        active.add(ordinal)
        try:
            return "".join(self._render_ref(child, token_index, active)
                           for child in self.aggregate_sequences[ordinal])
        finally:
            active.remove(ordinal)

    def render(self, token_index: IntegerTokenIndex, ordinal: int) -> str:
        """按 occurrence ordinal 重现表层，并校验所绑定 token index。"""
        if not isinstance(token_index, IntegerTokenIndex):
            raise TypeError("render 需要 IntegerTokenIndex")
        if token_index.sha256 != self.token_index_sha256:
            raise ValueError("aggregate 绑定的 token index hash 漂移")
        if len(token_index.sequences) != self.token_sequence_count:
            raise ValueError("aggregate 绑定的 token sequence count 漂移")
        if (type(ordinal) is not int
                or not 0 <= ordinal < len(self.occurrence_ordinals)):
            raise IndexError("aggregate occurrence ordinal 越界")
        return self._render_ref(
            self.token_sequence_count + self.occurrence_ordinals[ordinal],
            token_index, set())

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "PURE_INTEGER_AI_INTEGER_AGGREGATE_INDEX_V1",
            "token_index_sha256": self.token_index_sha256,
            "token_sequence_count": self.token_sequence_count,
            "aggregate_sequences": [list(item) for item in self.aggregate_sequences],
            "aggregate_keys": list(self.aggregate_keys),
            "occurrence_keys": list(self.occurrence_keys),
            "occurrence_ordinals": list(self.occurrence_ordinals),
            "sha256": self.sha256,
        }


def _aggregate_payload(*, token_index_sha256: str, token_sequence_count: int,
                       aggregate_sequences: tuple[tuple[int, ...], ...],
                       aggregate_keys: tuple[str, ...],
                       occurrence_keys: tuple[str, ...],
                       occurrence_ordinals: tuple[int, ...]) -> bytes:
    """建立不依赖 JSON/宿主类型的 aggregate 摘要输入。"""
    payload = bytearray(b"PURE-INTEGER-AI/INTEGER-AGGREGATE-INDEX/V1")
    payload.extend(bytes.fromhex(token_index_sha256))
    for value in (token_sequence_count, len(aggregate_sequences)):
        payload.extend(_u8(value))
    for sequence, key in zip(aggregate_sequences, aggregate_keys):
        payload.extend(_u8(len(sequence)))
        for ref in sequence:
            payload.extend(_u8(ref))
        encoded = key.encode("utf-8")
        payload.extend(_u8(len(encoded)))
        payload.extend(encoded)
    for key, ordinal in zip(occurrence_keys, occurrence_ordinals):
        encoded = key.encode("utf-8")
        payload.extend(_u8(len(encoded)))
        payload.extend(encoded)
        payload.extend(_u8(ordinal))
    return bytes(payload)


def build_integer_aggregate_index(
        token_index: IntegerTokenIndex,
        records: Iterable[tuple[str, Iterable[int]]],
        ) -> IntegerAggregateIndex:
    """把重复的引用序列聚合成新索引，且不复制 token sequence 内容。

    ``records`` 的顺序是调用方提供的稳定构造序；引用小于 token sequence
    数量时指向 token sequence，更大的引用指向此前已登记的 aggregate。这个
    前向引用限制是跨语言可复现的无环合同，而不是依赖 Python 调用栈的运行时约定。
    """
    if not isinstance(token_index, IntegerTokenIndex):
        raise TypeError("token_index 类型非法")
    raw = tuple(records)
    if not raw:
        raise ValueError("aggregate records 不能为空")
    keys = tuple(item[0] for item in raw)
    if (any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("aggregate record key 必须非空且唯一")
    unique: list[tuple[int, ...]] = []
    key_for_unique: list[str] = []
    lookup: dict[tuple[int, ...], int] = {}
    occurrence_ordinals: list[int] = []
    leaf_count = len(token_index.sequences)
    for key, values in raw:
        sequence = tuple(values)
        if not sequence or any(type(ref) is not int or ref < 0 for ref in sequence):
            raise ValueError("aggregate 引用必须是非空非负整数序列")
        current_limit = leaf_count + len(unique)
        if any(ref >= current_limit for ref in sequence):
            raise ValueError("aggregate 只能引用 token sequence 或此前 aggregate")
        ordinal = lookup.get(sequence)
        if ordinal is None:
            ordinal = len(unique)
            lookup[sequence] = ordinal
            unique.append(sequence)
            key_for_unique.append(key)
        occurrence_ordinals.append(ordinal)
    aggregate_sequences = tuple(unique)
    aggregate_keys = tuple(key_for_unique)
    occurrence_keys = keys
    occurrences = tuple(occurrence_ordinals)
    digest = hashlib.sha256(_aggregate_payload(
        token_index_sha256=token_index.sha256,
        token_sequence_count=leaf_count,
        aggregate_sequences=aggregate_sequences,
        aggregate_keys=aggregate_keys,
        occurrence_keys=occurrence_keys,
        occurrence_ordinals=occurrences,
    )).hexdigest()
    return IntegerAggregateIndex(
        token_index.sha256, leaf_count, aggregate_sequences, aggregate_keys,
        occurrence_keys, occurrences, digest)


def write_integer_aggregate_index(path: str | Path,
                                  index: IntegerAggregateIndex) -> None:
    """写规范 aggregate JSON sidecar；不覆盖既有文件。"""
    if not isinstance(index, IntegerAggregateIndex):
        raise TypeError("index 类型非法")
    payload = json.dumps(index.to_dict(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")) + "\n"
    with open(path, "xb") as stream:
        stream.write(payload.encode("utf-8"))


def load_integer_aggregate_index(path: str | Path) -> IntegerAggregateIndex:
    """回读并验证 aggregate sidecar 的规范整数和摘要。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != \
            "PURE_INTEGER_AI_INTEGER_AGGREGATE_INDEX_V1":
        raise ValueError("integer aggregate index format 非法")
    token_sha = value.get("token_index_sha256")
    token_count = value.get("token_sequence_count")
    sequences = tuple(tuple(item) for item in value.get("aggregate_sequences", ()))
    aggregate_keys = tuple(value.get("aggregate_keys", ()))
    occurrence_keys = tuple(value.get("occurrence_keys", ()))
    occurrences = tuple(value.get("occurrence_ordinals", ()))
    if not isinstance(token_sha, str) or len(token_sha) != 64:
        raise ValueError("aggregate token index hash 非法")
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("aggregate token sequence count 非法")
    if any(type(ref) is not int or ref < 0
           for seq in sequences for ref in seq):
        raise ValueError("aggregate sidecar 引用必须为非负整数")
    digest = hashlib.sha256(_aggregate_payload(
        token_index_sha256=token_sha,
        token_sequence_count=token_count,
        aggregate_sequences=sequences,
        aggregate_keys=aggregate_keys,
        occurrence_keys=occurrence_keys,
        occurrence_ordinals=occurrences,
    )).hexdigest()
    if digest != value.get("sha256"):
        raise ValueError("integer aggregate index hash 漂移")
    return IntegerAggregateIndex(
        token_sha, token_count, sequences, aggregate_keys,
        occurrence_keys, occurrences, digest)


__all__ = [
    "IntegerAggregateIndex", "IntegerTokenIndex",
    "build_integer_aggregate_index", "build_integer_token_index",
    "load_integer_aggregate_index", "load_integer_token_index",
    "write_integer_aggregate_index", "write_integer_token_index",
]
