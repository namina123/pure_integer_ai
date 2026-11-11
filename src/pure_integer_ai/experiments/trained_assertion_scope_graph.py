"""来源化陈述作用域模型的有类型整数存储；不持久化课程句子。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.understanding.query_assertion_scope import TrainedAssertionScopeFrame
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


FRAME = "assertion_scope_frame"
PART = "assertion_scope_part"
META = "assertion_scope_graph"
GRAPH_VERSION = 92383


def _hash(keys):
    """规范编码为四字节长度与八字节无符号整数，再取非零 63 位摘要。"""
    digest = hashlib.sha256()
    for key in keys:
        digest.update(len(key).to_bytes(4, "big"))
        for value in key:
            digest.update(value.to_bytes(8, "big"))
    return (int.from_bytes(digest.digest()[:8], "big") & (2**63 - 1)) or 1


def _register(backend):
    """注册可删除索引后恢复的 append-only 整数主记录与有序成员。"""
    for name, columns, indexes, recovery in (
        (FRAME, ("frame_id", "frame_hash"), [("frame_id",)], ("frame_id",)),
        (PART, ("frame_id", "ordinal", "value"), [("frame_id", "ordinal")],
         ("frame_id", "ordinal")),
        (META, ("version", "frame_count", "graph_hash"), [("version",)], ("version",)),
    ):
        register_extension_table(backend, name, [(column, TYPE_INT) for column in columns],
            discipline=disc.DISC_APPEND_ONLY, indexes=indexes, recovery_key=recovery)


def materialize_assertion_scope_graph(frames, database):
    """只在新 K 盘路径写入完整学习记录；无外部数据或旧答案依赖。"""
    target = Path(database).resolve()
    if (target.drive.upper() != "K:" or target.exists() or not target.parent.is_dir()
            or type(frames) is not tuple or not frames
            or any(not isinstance(frame, TrainedAssertionScopeFrame) for frame in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError("scope graph 需要新 K 盘路径和唯一完整训练记录")
    frames = tuple(sorted(frames, key=lambda frame: frame.stable_key()))
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    try:
        _register(backend)
        for ordinal, frame in enumerate(frames, 1):
            key = frame.stable_key()
            backend.insert(FRAME, {"frame_id": ordinal, "frame_hash": _hash((key,))})
            backend.insert_many(PART, [{"frame_id": ordinal, "ordinal": index, "value": value}
                                      for index, value in enumerate(key)])
        backend.insert(META, {"version": GRAPH_VERSION, "frame_count": len(frames),
                              "graph_hash": _hash(tuple(frame.stable_key() for frame in frames))})
        backend.commit()
    finally:
        backend.close()
    return {"version": GRAPH_VERSION, "frame_count": len(frames),
            "graph_hash": _hash(tuple(frame.stable_key() for frame in frames))}


def read_assertion_scope_graph(database):
    """只读冷恢复完整语法、算子、来源和许可；缺失或孤儿记录一律拒绝。"""
    backend = SQLiteBackend(str(Path(database).resolve(strict=True)), read_only=True)
    try:
        _register(backend)
        metas = backend.select(META, order_by="version")
        rows = backend.select(FRAME, order_by="frame_id")
        parts = backend.select(PART, order_by="frame_id, ordinal")
        if (len(metas) != 1 or metas[0]["version"] != GRAPH_VERSION or not rows
                or len(rows) != metas[0]["frame_count"]
                or tuple(row["frame_id"] for row in rows) != tuple(range(1, len(rows) + 1))):
            raise ValueError("scope graph 缺少完整元数据或框架")
        grouped = {row["frame_id"]: [] for row in rows}
        for part in parts:
            if part["frame_id"] not in grouped:
                raise ValueError("scope graph 存在孤儿成员")
            grouped[part["frame_id"]].append(part)
        frames = []
        for row in rows:
            members = grouped[row["frame_id"]]
            if tuple(part["ordinal"] for part in members) != tuple(range(len(members))):
                raise ValueError("scope graph 成员不连续")
            key = tuple(part["value"] for part in members)
            frame = TrainedAssertionScopeFrame.from_stable_key(key)
            if frame.stable_key() != key or _hash((key,)) != row["frame_hash"]:
                raise ValueError("scope graph 记录身份漂移")
            frames.append(frame)
        canonical = tuple(sorted(set(frames), key=lambda frame: frame.stable_key()))
        if (tuple(frames) != canonical
                or _hash(tuple(frame.stable_key() for frame in canonical)) != metas[0]["graph_hash"]):
            raise ValueError("scope graph 整体身份漂移")
        return canonical
    finally:
        backend.close()
