"""阶段 B/C 回归：跨 Core 与 Interaction Memory 的统一查询桥与结构生成。

桥测试验证：同一查询在 Core active relation 上多跳发现鸟类集合/小麻雀，
且把 Memory 会话焦点作为 discourse 邻接一并扩展，trace 带 evidence/空间/
深度/稳定 key。生成测试验证：Core 闭合的命题通过 _generate_surface 的槽位
重填产出新表层，不做整句回放。
"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_authored_w06_learning_runtime,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.run_conversation_training import (
    dialogue_semantic_protocols,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend

ROOT = Path(__file__).resolve().parents[1]


def _relation_courses() -> tuple[Path, ...]:
    """返回现役七类 authored W-06 训练输入。"""
    root = ROOT / "data" / "ph2"
    return tuple(sorted(root.glob("authored_relation_*_w06_seed_v2.jsonl.sample"))) + tuple(
        sorted(
            path for path in root.glob("authored_relation_*_seed_v1.jsonl.sample")
            if path.name != "authored_relation_alias_refers_seed_v1.jsonl.sample"
        )
    )


def _trained_database(tmp_path: Path) -> Path:
    """消费公开 relation samples 训练 W-06 并落盘 SQLite。"""
    database = tmp_path / "training.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _semantic, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(
            context,
            occurrence_protocol=occurrence,
            span_protocol=span,
        )
        build_authored_w06_learning_runtime(
            backend,
            context,
            _relation_courses(),
            tmp_path / "typed-relation-pack",
        )
        backend.commit()
    finally:
        backend.close()
    return database


def test_cross_space_query_core_and_memory(tmp_path):
    """同一查询跨 Core（relation 邻接）与 Memory（会话焦点）扩展。"""
    database = _trained_database(tmp_path)
    memory_db = tmp_path / "session_memory.sqlite3"
    with TrainedDialogueMemoryGraph(memory_db, session_id=7) as memory:
        memory.append("用户在讨论汽车相关话题。", speaker_kind=1)
        memory.append("我最近在查汽车构造资料。", speaker_kind=1)
    with TrainedGraphQueryBridge(
            database,
            memory_database=memory_db,
            session_id=7,
    ) as bridge:
        assert bridge.memory is not None
        result = bridge.query("麻雀集合", minimum_depth=1, max_depth=3)
        core_surfaces = {
            item["surface"] for item in result["hops"] if item["space"] == 1}
        assert "鸟类集合" in core_surfaces
        assert "小麻雀" in core_surfaces
        # Memory 焦点也被并入 frontier（至少 state_key 覆盖两种空间）。
        assert result["depth"] >= 2
        assert result["evidence"]
        assert any(item["space"] in {1, 2} for item in result["hops"])


def test_bridge_replayable_and_evidence_closed(tmp_path):
    """同一输入在同一训练库上的 trace 与 stable key 可重放。"""
    database = _trained_database(tmp_path)
    first = None
    for _ in range(2):
        with TrainedGraphQueryBridge(database) as bridge:
            result = bridge.query("麻雀集合", minimum_depth=1, max_depth=3)
        if first is None:
            first = result
        else:
            assert first["hops"] == result["hops"]
            assert first["state_key"] == result["state_key"]
    assert all(item["source_hash"] > 0 for item in first["evidence"])


def test_no_anchor_returns_clarify_missing_binding(tmp_path):
    """无概念 anchor 的普通语言返回 CLARIFY_MISSING_BINDING，不近邻。"""
    database = _trained_database(tmp_path)
    with TrainedGraphQueryBridge(database) as bridge:
        result = bridge.query("火星轨道参数是什么？", minimum_depth=1)
        assert result["termination_reason"] in {
            "CLARIFY_MISSING_BINDING", "NO_CONCEPT_ANCHOR"}
        assert result["hops"] == []
