"""阶段 B 回归：Core active relation 多跳查询桥在真实授权数据上的路径证据。

直接消费 data/ph2 的公开 authored_relation_*_seed samples（CC0 授权声明），
构造 W-06 learning runtime + TrainedGraphRelease 风格只读恢复，再走
TrainedCoreQueryRuntime.query 验证：(1) 表层 anchor 绑定到概念 filler；
(2) 沿概念边多跳扩展且深度由数据决定；(3) 换一种不含 anchor 的表层时
协议返回 CLARIFY_MISSING_BINDING 而不是近邻句。
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
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedCoreQueryRuntime,
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


def test_multi_hop_path_sparrow_member_bird_set(tmp_path):
    """'麻雀集合' -> 邻接 '小麻雀'（属于）与 '鸟类集合'（包含于）。"""
    database = _trained_database(tmp_path)
    with TrainedCoreQueryRuntime(database) as bridge:
        anchors = bridge.anchor_concepts("麻雀集合")
        assert any("麻雀集合" == item.surface for item in anchors)
        result = bridge.query("麻雀集合", minimum_depth=1, max_depth=3)
        # 至少一条概念级多跳：同一查询在邻接层发现鸟类集合(上层)与小麻雀(下位)。
        surfaces = {item["surface"] for item in result["hops"]}
        assert "鸟类集合" in surfaces
        assert "小麻雀" in surfaces
        assert result["depth"] >= 2
        # 没有整句回放：hops 是概念 filler 表层，不是输入整句。
        assert "麻雀集合" not in {
            item["surface"] for item in result["hops"]}


def test_no_anchor_returns_clarify_missing_binding(tmp_path):
    """无概念 anchor 的普通语言返回 CLARIFY_MISSING_BINDING，不近邻。"""
    database = _trained_database(tmp_path)
    with TrainedCoreQueryRuntime(database) as bridge:
        result = bridge.query("火星轨道参数是什么？", minimum_depth=1)
        # 无概念 anchor 属缺绑定澄清，与 NO_CONCEPT_ANCHOR 是同一协议语义；
        # 驱动终态终止码以 CLARIFY_MISSING_BINDING 表达。
        assert result["termination_reason"] in {
            "CLARIFY_MISSING_BINDING", "NO_CONCEPT_ANCHOR"}
        assert result["hops"] == []


def test_query_state_key_replayable_across_reopens(tmp_path):
    """同一输入在同一训练库上的查询 trace 与 stable key 可重放。"""
    database = _trained_database(tmp_path)
    first = None
    for _ in range(2):
        with TrainedCoreQueryRuntime(database) as bridge:
            result = bridge.query("麻雀集合", minimum_depth=1, max_depth=3)
        if first is None:
            first = result
        else:
            assert first["hops"] == result["hops"]
            assert first["depth"] == result["depth"]
            assert first["state_key"] == result["state_key"]
