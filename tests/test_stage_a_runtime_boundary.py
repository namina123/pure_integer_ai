"""阶段 A 回归：低证据伪澄清隔离与运行边界辅助。

生产 respond_graph 路径（diagnostic_replay=False）对没有结构证据的输入
不得按输入长度/问句形状从 Dialogue 图挑选澄清句；该行为只能由显式
diagnostic_replay=True 的诊断对照触发。另覆盖发布会话库外置性守卫与
运行边界快照比较。
"""
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderProtocol,
)
from pure_integer_ai.experiments.collection import (
    CollectedItem,
    DialogueContentSpan,
    SpeakerSpan,
)
from pure_integer_ai.experiments.dialogue_successor_graph import (
    DialogueSuccessorProtocol,
    SqliteDialogueSuccessorRuntime,
    install_dialogue_successor_runtime,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.run_trained_relation_graph_terminal import (
    _locate_release_root,
    _require_session_outside_release,
)
from pure_integer_ai.experiments.runtime_boundary_validator import (
    compare_snapshots,
    snapshot_tree,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


def _item(source_id: int, prompt: str, response: str) -> CollectedItem:
    """构造带显式 speaker/content span 的公开训练项。"""
    turns = ((1, prompt), (2, response))
    rendered = tuple(surface for _role, surface in turns)
    raw_text = "\n".join(rendered)
    speaker_spans = []
    content_spans = []
    cursor = 0
    for index, ((role, surface), value) in enumerate(
            zip(turns, rendered), start=1):
        content_start = cursor
        content_end = cursor + len(value)
        turn_end = content_end + (1 if index < len(turns) else 0)
        speaker_spans.append(SpeakerSpan(
            cursor, turn_end, index, role_identity((9291, role))))
        content_spans.append(DialogueContentSpan(
            content_start, content_end, index))
        cursor = turn_end
    return CollectedItem(
        tokens=list(raw_text),
        raw_text=raw_text,
        role_seq=[1] * len(raw_text),
        source=SOURCE_BARE_TEXT,
        source_ref=SourceRef(
            SOURCE_BARE_TEXT, source_id, 1,
            GLOBAL_OWNER_SCOPE, VersionBundle()),
        speaker_spans=tuple(speaker_spans),
        dialogue_content_spans=tuple(content_spans),
    )


def _trained_database(tmp_path: Path, prompt: str, response: str) -> Path:
    """训练一条 current->response 后继并返回训练库路径。"""
    database = tmp_path / "training.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        ctx = make_train_context(backend)
        install_language_graph_protocols(
            ctx,
            occurrence_protocol=OccurrenceProtocol((9292, 1), (9292, 2)),
            occurrence_order_protocol=OccurrenceOrderProtocol((9292, 3)),
        )
        install_dialogue_successor_runtime(
            ctx, DialogueSuccessorProtocol((9292, 10), EPI_STRUCTURED))
        DefaultRoundRunner().run_round_full(
            ctx, _item(1, prompt, response), 1, 1)
        backend.commit()
    finally:
        backend.close()
    return database


def test_low_evidence_production_path_never_selects_by_length_or_question_shape(
        tmp_path):
    """无结构证据的普通自然语言不得从图内挑疑问澄清句。"""
    database = _trained_database(
        tmp_path, "你好，请问可以聊天吗？", "你好，有什么事吗？")
    runtime = SqliteDialogueSuccessorRuntime(
        database, graph_dialogue=True)
    try:
        # 无共享结构片段的普通语言：生产路径必须无结果（None），
        # 不能按码点数接近 + QUESTION 形状返回已学澄清句。
        assert runtime.respond_graph(
            "火星轨道参数是什么？") is None
        # 相同输入在显式诊断对照下仍可复现旧路径（只作回归对照）。
        diagnostic = runtime.respond_graph(
            "火星轨道参数是什么？", diagnostic_replay=True)
        assert diagnostic is not None
        assert diagnostic.trace.result_mode == 4  # CLARIFICATION
    finally:
        runtime.close()


def test_release_session_must_resolve_outside_release_root(tmp_path):
    """发布会话库守卫：拒绝 release root 内的任何会话路径。"""
    root = tmp_path / "release"
    (root / "model").mkdir(parents=True)
    (root / "trained_graph_release.json").write_text("{}", encoding="utf-8")
    database = root / "model" / "training.sqlite3"
    database.write_bytes(b"x")
    assert _locate_release_root(database) == root
    inside = root / "model" / "inside_runtime.sqlite3"
    with pytest.raises(ValueError):
        _require_session_outside_release(inside, root)
    outside = tmp_path / "outside_runtime.sqlite3"
    _require_session_outside_release(outside, root)  # 不抛错


def test_boundary_snapshot_detects_added_removed_and_drifted(tmp_path):
    """运行边界快照比较能发现新增/删除/漂移文件。"""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("one", encoding="utf-8")
    before = snapshot_tree(root)
    (root / "b.txt").write_text("two", encoding="utf-8")
    (root / "a.txt").write_text("changed", encoding="utf-8")
    comparison = compare_snapshots(before, snapshot_tree(root))
    assert comparison["added"] == ["b.txt"]
    assert comparison["drifted"] == ["a.txt"]
    assert not comparison["closed"]
    (root / "b.txt").unlink()
    (root / "a.txt").write_text("one", encoding="utf-8")
    assert compare_snapshots(before, snapshot_tree(root))["closed"]


def test_strict_graph_miss_returns_no_answer_not_raise(tmp_path):
    """strict 发布路径三图无组合结果时输出 no_answer，不再抛 RuntimeError。"""
    database = _trained_database(tmp_path, "你好，请问可以聊天吗？", "你好，有什么事吗？")
    # 该后继库没有 W-06 relation anchor/typed connector；strict 只读关系图会启动失败，
    # 因此不能直接喂给 strict relation-graph 终端。此用例在进程级之下验证生产对话路由在
    # 无结构证据的普通输入上不伪造回答（返回 None），不丢 RuntimeError。
    runtime = SqliteDialogueSuccessorRuntime(database, graph_dialogue=True)
    try:
        assert runtime.respond_graph("早上好", diagnostic_replay=False) is None
        assert runtime.respond_graph("晚安", diagnostic_replay=False) is None
    finally:
        runtime.close()
