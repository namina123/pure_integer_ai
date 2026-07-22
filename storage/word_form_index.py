"""语言地基的持久化词形目录。

核心图只保存整数引用。本扩展表保存分词所需的反向映射：
语言 + 词形引用 -> 有序码点。它不承载任何语义关系，并保持只增。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

WORD_FORM_INDEX_TABLE = "word_form_index"
WORD_FORM_LEGACY_BRIDGE_TABLE = "word_form_legacy_bridge"

_COLUMNS = [
    ("space_id", TYPE_INT),
    ("language", TYPE_INT),
    ("word_space_id", TYPE_INT),
    ("word_local_id", TYPE_INT),
    ("order_index", TYPE_INT),
    ("codepoint", TYPE_INT),
]
_INDEXES = [
    ("space_id", "language"),
    ("space_id", "word_space_id", "word_local_id", "language"),
]

_BRIDGE_COLUMNS = [
    ("space_id", TYPE_INT),
    ("legacy_local_id", TYPE_INT),
    ("legacy_node_type", TYPE_INT),
    ("object_kind", TYPE_INT),
    ("object_space_id", TYPE_INT),
    ("object_local_id", TYPE_INT),
]
_BRIDGE_INDEXES = [
    ("space_id", "legacy_local_id"),
    ("object_kind", "object_space_id", "object_local_id"),
]


class WordFormLegacyBridgeConflict(ValueError):
    """同一 legacy 节点被迁移到不同权威对象时拒绝继续。"""


def register_word_form_index(backend: StorageBackend) -> None:
    """注册旧词形目录和指向权威语言对象的只增迁移桥。"""
    register_extension_table(backend, WORD_FORM_INDEX_TABLE, _COLUMNS,
                             disc.DISC_APPEND_ONLY, _INDEXES)
    register_extension_table(
        backend,
        WORD_FORM_LEGACY_BRIDGE_TABLE,
        _BRIDGE_COLUMNS,
        disc.DISC_APPEND_ONLY,
        _BRIDGE_INDEXES,
    )


def record_word_form(backend: StorageBackend, *, space_id: int, language: int,
                     word_ref: tuple[int, int], codepoints) -> None:
    """以整数码点幂等记录一个词形。"""
    word_space_id, word_local_id = word_ref
    cps = tuple(codepoints)
    assert_int(space_id, language, word_space_id, word_local_id,
               *_ints(cps), _where="record_word_form")
    if not cps:
        raise ValueError("词形不能为空")
    try:
        existing = backend.select(WORD_FORM_INDEX_TABLE, where={
            "space_id": space_id,
            "word_space_id": word_space_id,
            "word_local_id": word_local_id,
            "language": language,
        })
    except KeyError:
        return
    old = tuple(r["codepoint"] for r in sorted(existing,
                                                key=lambda r: r["order_index"]))
    if existing:
        if old != cps:
            raise ValueError("同一词形身份已绑定不同码点序列")
        return
    for order_index, codepoint in enumerate(cps):
        backend.insert(WORD_FORM_INDEX_TABLE, {
            "space_id": space_id,
            "language": language,
            "word_space_id": word_space_id,
            "word_local_id": word_local_id,
            "order_index": order_index,
            "codepoint": codepoint,
        })


def load_word_forms(backend: StorageBackend, *, space_id: int,
                    language: int) -> list[tuple[tuple[int, ...], tuple[int, int]]]:
    """按确定顺序读取 ``(码点序列, 词形引用)``。"""
    assert_int(space_id, language, _where="load_word_forms")
    try:
        rows = backend.select(WORD_FORM_INDEX_TABLE, where={
            "space_id": space_id, "language": language,
        })
    except KeyError:
        return []
    grouped: dict[tuple[int, int], list[dict[str, int]]] = {}
    for row in rows:
        ref = (row["word_space_id"], row["word_local_id"])
        grouped.setdefault(ref, []).append(row)
    entries = []
    for ref, ref_rows in grouped.items():
        ordered = sorted(ref_rows, key=lambda r: r["order_index"])
        entries.append((tuple(r["codepoint"] for r in ordered), ref))
    return sorted(entries, key=lambda item: (item[0], item[1]))


def record_legacy_word_form_bridge(
        backend: StorageBackend, *, legacy_ref: tuple[int, int],
        legacy_node_type: int, object_ref: tuple[int, int, int]) -> None:
    """幂等记录 legacy 节点到权威分型对象的显式兼容桥。"""
    legacy_space_id, legacy_local_id = legacy_ref
    object_kind, object_space_id, object_local_id = object_ref
    assert_int(
        legacy_space_id,
        legacy_local_id,
        legacy_node_type,
        object_kind,
        object_space_id,
        object_local_id,
        _where="record_legacy_word_form_bridge",
    )
    try:
        existing = backend.select(WORD_FORM_LEGACY_BRIDGE_TABLE, where={
            "space_id": legacy_space_id,
            "legacy_local_id": legacy_local_id,
        })
    except KeyError:
        return
    expected = {
        "space_id": legacy_space_id,
        "legacy_local_id": legacy_local_id,
        "legacy_node_type": legacy_node_type,
        "object_kind": object_kind,
        "object_space_id": object_space_id,
        "object_local_id": object_local_id,
    }
    if existing:
        if len(existing) != 1 or existing[0] != expected:
            raise WordFormLegacyBridgeConflict(
                "同一 legacy 节点存在不同或重复的权威对象桥")
        return
    backend.insert(WORD_FORM_LEGACY_BRIDGE_TABLE, expected)


def load_legacy_word_form_bridges(
        backend: StorageBackend, *, legacy_ref: tuple[int, int]
        ) -> tuple[tuple[int, int, int, int], ...]:
    """读取 legacy 节点的 ``(node_type, object_kind, sid, lid)`` 桥。"""
    legacy_space_id, legacy_local_id = legacy_ref
    assert_int(
        legacy_space_id, legacy_local_id,
        _where="load_legacy_word_form_bridges")
    try:
        rows = backend.select(WORD_FORM_LEGACY_BRIDGE_TABLE, where={
            "space_id": legacy_space_id,
            "legacy_local_id": legacy_local_id,
        })
    except KeyError:
        return ()
    values = {
        (
            row["legacy_node_type"],
            row["object_kind"],
            row["object_space_id"],
            row["object_local_id"],
        )
        for row in rows
    }
    if len(values) != len(rows):
        raise WordFormLegacyBridgeConflict("legacy 迁移桥存在重复行")
    return tuple(sorted(values))


def _ints(values):
    """把任意码点序列整理成整数守卫的可变参数。"""
    return tuple(values)


__all__ = [
    "WORD_FORM_INDEX_TABLE",
    "WORD_FORM_LEGACY_BRIDGE_TABLE",
    "WordFormLegacyBridgeConflict",
    "load_legacy_word_form_bridges",
    "load_word_forms",
    "record_legacy_word_form_bridge",
    "record_word_form",
    "register_word_form_index",
]
