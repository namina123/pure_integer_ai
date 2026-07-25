"""COOCCURS A' 跨段去重测试（总收口 0.1·解 LIVE 病灶①·阻塞 #734）。

EdgeStore.add append-only 不去重 → 同 (a,b,COOCCURS) 跨段重复 pair 堆叠（vocab=50 爆炸 9684·真语料跑不动）。
A' = add_cooccurs_dedup（SELECT→UPDATE strength+=1 / INSERT strength=1·返 True 新建 / False UPDATE）+
reader 4 处改读 strength 累加（hub_degree / compute_hub_set / _cooccurs_count / collide_score·
gate OFF strength 恒 1 累加=数行 bit-identical·gate ON 累加=频次）。

覆盖：
  - add_cooccurs_dedup 单元（同 pair 合并 / 返 True/False / strength=频次 / 边数不膨胀）
  - build_cooccurs gate ON 合并（同 pair 跨段 1 行 strength=N）/ gate OFF 现状（N 行 strength=1）
  - reader 4 处读 strength（gate ON 频次 / gate OFF 等价数行）
  - built_edges gate ON=真实边数（dedup 后大降·非配对数虚高）
  - 反 theater 核心：dedup 边数 << append-only（解阻塞效果可观测）
  - bit-identical gate OFF 两跑一致
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_COOCCURS
from pure_integer_ai.cognition.shared.hub_detect import (
    HubDegreeState,
    hub_degree,
    compute_hub_set,
    THETA_HUB_DEGREE,
)
from pure_integer_ai.cognition.understanding.emergent_relation_signal import _cooccurs_count
from pure_integer_ai.cognition.understanding.cooccurs import build_cooccurs
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def core():
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    yield b, sp.space_id, es
    b.close()


# ---- add_cooccurs_dedup 单元 ----

def test_add_cooccurs_dedup_merges_same_pair(core):
    """同 (a,b,COOCCURS) 多次 dedup → 1 行 strength=N（非 N 行）·返 True 首次 / False 后续。"""
    b, sid, es = core
    A, B = 1, 2
    r1 = es.add_cooccurs_dedup(space_id_from=sid, local_id_from=A,
                               space_id_to=sid, local_id_to=B,
                               edge_type=EDGE_COOCCURS, source=4)
    assert r1 is True                                      # 首次新建
    rows = es.query_type(EDGE_COOCCURS)
    assert len(rows) == 1
    assert rows[0]["strength"] == 1

    r2 = es.add_cooccurs_dedup(space_id_from=sid, local_id_from=A,
                               space_id_to=sid, local_id_to=B,
                               edge_type=EDGE_COOCCURS, source=4)
    assert r2 is False                                     # 已存在 UPDATE
    rows = es.query_type(EDGE_COOCCURS)
    assert len(rows) == 1                                  # 仍 1 行（非 2·不膨胀）
    assert rows[0]["strength"] == 2                        # strength 累加

    for _ in range(3):
        es.add_cooccurs_dedup(space_id_from=sid, local_id_from=A,
                              space_id_to=sid, local_id_to=B,
                              edge_type=EDGE_COOCCURS, source=4)
    rows = es.query_type(EDGE_COOCCURS)
    assert len(rows) == 1
    assert rows[0]["strength"] == 5                        # 5 次 = strength 5


def test_add_cooccurs_dedup_distinct_pairs_separate(core):
    """不同 pair 各自独立行（a→b 与 a→c 不合并·按 from,to 去重）。"""
    b, sid, es = core
    for to in (2, 3, 4):
        es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                              space_id_to=sid, local_id_to=to,
                              edge_type=EDGE_COOCCURS, source=4)
    rows = es.query_type(EDGE_COOCCURS)
    assert len(rows) == 3                                  # 3 不同 pair = 3 行


# ---- build_cooccurs gate 分支 ----

def test_build_cooccurs_gate_on_merges_cross_segment(core):
    """gate ON·两段同 refs → 同 pair UPDATE·n2=0·边数不翻倍·strength=2。"""
    b, sid, es = core
    refs = [(sid, 1), (sid, 2), (sid, 3)]                  # C(3,2)=3 pair
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    try:
        n1 = build_cooccurs(es, refs, lang=1, domain=1, source=4, space_id=sid)
        assert n1 == 3                                     # 3 新边
        n2 = build_cooccurs(es, refs, lang=1, domain=1, source=4, space_id=sid)
        assert n2 == 0                                     # 全已存在·无新边（built_edges=真实边数）
        rows = es.query_type(EDGE_COOCCURS)
        assert len(rows) == 3                              # 仍 3 边（非 6）
        assert all(r["strength"] == 2 for r in rows)       # 2 段各建 1 次
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


def test_build_cooccurs_updates_context_hub_state_on_dedup(core):
    """同一 pair 的 INSERT 和 strength UPDATE 都向上下文 hub 状态提交增量。"""
    b, sid, es = core
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    try:
        state = HubDegreeState(es, theta=2)
        assert state.hub_set() == set()
        refs = [(sid, 1), (sid, 2)]
        build_cooccurs(
            es, refs, lang=1, domain=1, source=4, space_id=sid,
            hub_degree_state=state,
        )
        assert (sid, 1) not in state.hub_set()
        build_cooccurs(
            es, refs, lang=1, domain=1, source=4, space_id=sid,
            hub_degree_state=state,
        )
        assert (sid, 1) in state.hub_set()
        assert es.query_type(EDGE_COOCCURS)[0]["strength"] == 2
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


def test_build_cooccurs_gate_off_append_only(core):
    """gate OFF·旧 add·两段同 refs → 6 行 strength=1（append-only 现状·bit-identical）。"""
    b, sid, es = core
    refs = [(sid, 1), (sid, 2), (sid, 3)]
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = False
    try:
        build_cooccurs(es, refs, lang=1, domain=1, source=4, space_id=sid)
        build_cooccurs(es, refs, lang=1, domain=1, source=4, space_id=sid)
        rows = es.query_type(EDGE_COOCCURS)
        assert len(rows) == 6                              # 2 段 × 3 pair = 6（append-only 堆叠）
        assert all(r["strength"] == 1 for r in rows)       # 旧 add strength 恒 1
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


# ---- reader 4 处读 strength ----

def test_hub_degree_reads_strength(core):
    """gate ON·hub_degree 读 strength 累加（dedup 边 strength=N 计入·非数行=1）。"""
    b, sid, es = core
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    try:
        for _ in range(3):                                 # (1→2) 共现 3 次 → strength=3
            es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                                  space_id_to=sid, local_id_to=2,
                                  edge_type=EDGE_COOCCURS, source=4)
        es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                              space_id_to=sid, local_id_to=3,
                              edge_type=EDGE_COOCCURS, source=4)
        es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                              space_id_to=sid, local_id_to=4,
                              edge_type=EDGE_COOCCURS, source=4)
        # hub_degree(1) = strength(1→2)=3 + strength(1→3)=1 + strength(1→4)=1 = 5
        assert hub_degree((sid, 1), es) == 5
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


def test_compute_hub_set_reads_strength(core):
    """gate ON·compute_hub_set degree 累加 strength（单 dedup 边 strength=θ 即达 hub·非数行需 θ 条边）。"""
    b, sid, es = core
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    try:
        for _ in range(THETA_HUB_DEGREE):                  # (1→2) strength=θ
            es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                                  space_id_to=sid, local_id_to=2,
                                  edge_type=EDGE_COOCCURS, source=4)
        hs = compute_hub_set(es)
        assert (sid, 1) in hs                              # degree=θ（单边 strength 累加）
        assert (sid, 2) in hs                              # to 端同累加
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


def test_cooccurs_count_reads_strength(core):
    """gate ON·_cooccurs_count 读 strength（dedup (a,b) N 次 → N·非 1）。"""
    b, sid, es = core
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    try:
        for _ in range(5):
            es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                                  space_id_to=sid, local_id_to=2,
                                  edge_type=EDGE_COOCCURS, source=4)
        assert _cooccurs_count(es, (sid, 1), (sid, 2)) == 5
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


def test_collide_score_reads_strength(core):
    """gate ON·collide_score 读 strength（c-ctx 频次计分·保持消歧判别力·非集合基数骤降）。"""
    b, sid, es = core
    g = ConceptGraph(b)
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    try:
        for _ in range(4):
            es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                                  space_id_to=sid, local_id_to=2,
                                  edge_type=EDGE_COOCCURS, source=4)
        assert g.collide_score((sid, 1), [(sid, 2)]) == 4  # strength=4（频次）
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


# ---- gate OFF reader 等价（bit-identical 核心） ----

def test_reader_gate_off_equivalent_row_count(core):
    """gate OFF·reader 读 strength（恒 1）累加 = 数行 = 旧 row-count 语义·bit-identical。"""
    b, sid, es = core
    g = ConceptGraph(b)
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = False
    try:
        for _ in range(3):                                 # 旧 add·3 条同 (1,2) 边
            es.add(space_id_from=sid, local_id_from=1, space_id_to=sid, local_id_to=2,
                   edge_type=EDGE_COOCCURS, strength=1, source=4)
        # 读 strength（每行 1）累加 = 3 = 数行
        assert hub_degree((sid, 1), es) == 3
        assert _cooccurs_count(es, (sid, 1), (sid, 2)) == 3
        assert g.collide_score((sid, 1), [(sid, 2)]) == 3
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


# ---- 反 theater 核心：dedup 边数 << append-only（解阻塞效果） ----

def _build_n_edges(dedup: bool) -> int:
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    refs = [(sp.space_id, i) for i in range(1, 6)]         # C(5,2)=10 pair/段
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = dedup
    try:
        for _ in range(10):                                # 10 段重复同 refs
            build_cooccurs(es, refs, lang=1, domain=1, source=4, space_id=sp.space_id)
    finally:
        gates.COOCCURS_DEDUP_MODE = saved
    n = len(es.query_type(EDGE_COOCCURS))
    b.close()
    return n


def test_dedup_reduces_edge_count_vs_append_only():
    """反 theater 核心：dedup 边数 << append-only（10× 降·解 #734 真语料阻塞）。"""
    assert _build_n_edges(True) == 10                      # dedup：10 唯一 pair
    assert _build_n_edges(False) == 100                    # append-only：10 段 × 10 pair


# ---- bit-identical gate OFF 两跑 ----

def test_gate_off_bit_identical_two_runs():
    """gate OFF 两跑 edge 表一致（CI=生产 bit-identical·确定性序）。"""
    def _run():
        b = DictBackend()
        bootstrap(b)
        reg = SpaceRegistry(b)
        sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(b)
        refs = [(sp.space_id, 1), (sp.space_id, 2), (sp.space_id, 3)]
        saved = gates.COOCCURS_DEDUP_MODE
        gates.COOCCURS_DEDUP_MODE = False
        try:
            build_cooccurs(es, refs, lang=1, domain=1, source=4, space_id=sp.space_id)
        finally:
            gates.COOCCURS_DEDUP_MODE = saved
        rows = [{
            "f": (r["space_id_from"], r["local_id_from"]),
            "t": (r["space_id_to"], r["local_id_to"]),
            "s": r["strength"],
        } for r in es.query_type(EDGE_COOCCURS)]
        b.close()
        return rows
    assert _run() == _run()


# ---- 对抗审 P1/P2 补测 ----

def test_build_cooccurs_gate_on_reverse_direction(core):
    """P1-4 核心论点：跨段反向 pair（段1 [a,b]→a→b·段2 [b,a]→b→a）→ dedup 产 2 行（不合两向）·reader 双向累加正确。"""
    b, sid, es = core
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    try:
        build_cooccurs(es, [(sid, 1), (sid, 2)], lang=1, domain=1, source=4, space_id=sid)  # 段1 → (1→2)
        build_cooccurs(es, [(sid, 2), (sid, 1)], lang=1, domain=1, source=4, space_id=sid)  # 段2 → (2→1)
        rows = es.query_type(EDGE_COOCCURS)
        assert len(rows) == 2                                     # 两方向各 1 行（dedup 按 from,to 不合两向）
        # _cooccurs_count 双向 query 累加 = 2（query_from(1)→2 命中 + query_from(2)→1 命中）
        assert _cooccurs_count(es, (sid, 1), (sid, 2)) == 2
        # hub_degree(1) = strength(1→2 from 端)=1 + strength(2→1 to 端)=1 = 2
        assert hub_degree((sid, 1), es) == 2
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


def test_collide_score_gate_on_off_same_value():
    """P1-4 核心论点：同数据 gate ON（dedup strength=N）vs OFF（N 行 str=1）collide_score 相等·消歧不破。"""
    def _score(dedup: bool) -> int:
        bk = DictBackend()
        bootstrap(bk)
        reg = SpaceRegistry(bk)
        sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(bk)
        g = ConceptGraph(bk)
        saved = gates.COOCCURS_DEDUP_MODE
        gates.COOCCURS_DEDUP_MODE = dedup
        try:
            for _ in range(4):                                    # c=1 与 ctx=2 跨段共现 4 次
                build_cooccurs(es, [(sp.space_id, 1), (sp.space_id, 2)],
                               lang=1, domain=1, source=4, space_id=sp.space_id)
        finally:
            gates.COOCCURS_DEDUP_MODE = saved
        s = g.collide_score((sp.space_id, 1), [(sp.space_id, 2)])
        bk.close()
        return s
    assert _score(True) == 4                                      # gate ON：1 行 strength=4
    assert _score(False) == 4                                     # gate OFF：4 行 strength=1 each


def test_build_cooccurs_gate_on_partial_overlap(core):
    """P2-9：部分重叠 段1 [a,b,c]·段2 [a,b,d] → n2=2 新边（a-d·b-d）+ (a,b) 旧 UPDATE。"""
    b, sid, es = core
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = True
    try:
        n1 = build_cooccurs(es, [(sid, 1), (sid, 2), (sid, 3)],
                            lang=1, domain=1, source=4, space_id=sid)
        assert n1 == 3                                            # (1,2)(1,3)(2,3)
        n2 = build_cooccurs(es, [(sid, 1), (sid, 2), (sid, 4)],
                            lang=1, domain=1, source=4, space_id=sid)
        assert n2 == 2                                            # 仅 (1,4)(2,4) 新·(1,2) 旧 UPDATE
        assert len(es.query_type(EDGE_COOCCURS)) == 5            # 3+2=5 唯一 pair
    finally:
        gates.COOCCURS_DEDUP_MODE = saved


def test_add_cooccurs_dedup_raises_on_legacy_dup_rows(core):
    """P1-3：旧 append-only 重复行（跨 gate 迁移）遇 dedup → raise（防 add_strength 全量+1 静默过计·fail-fast）。"""
    b, sid, es = core
    saved = gates.COOCCURS_DEDUP_MODE
    gates.COOCCURS_DEDUP_MODE = False
    try:
        # 旧 add 建 2 行同 (1,2) COOCCURS（append-only 重复·模拟 OFF dump）
        for _ in range(2):
            es.add(space_id_from=sid, local_id_from=1, space_id_to=sid, local_id_to=2,
                   edge_type=EDGE_COOCCURS, strength=1, source=4)
    finally:
        gates.COOCCURS_DEDUP_MODE = saved
    # 翻 gate ON·add_cooccurs_dedup 遇 2 行 → raise（不静默过计）
    with pytest.raises(RuntimeError, match="重复 COOCCURS 边"):
        es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                              space_id_to=sid, local_id_to=2,
                              edge_type=EDGE_COOCCURS, source=4)


def test_add_cooccurs_dedup_rejects_non_cooccurs(core):
    """P2-5：add_cooccurs_dedup assert 拒绝非 EDGE_COOCCURS（防误用合并 PRECEDES 腐蚀 order_index）。"""
    from pure_integer_ai.storage.edge_types import EDGE_PRECEDES
    b, sid, es = core
    with pytest.raises(AssertionError, match="仅 EDGE_COOCCURS"):
        es.add_cooccurs_dedup(space_id_from=sid, local_id_from=1,
                              space_id_to=sid, local_id_to=2,
                              edge_type=EDGE_PRECEDES, source=4)


def test_formal_train_cooccurs_gates_restored(tmp_path, monkeypatch):
    """P2-2：formal_train finally 守 COOCCURS_DEDUP_MODE/WINDOW_MODE 回归 OFF（CI/生产 default·防误删 reset）。"""
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES
    from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
    corpus = [CollectedItem(tokens=["雨", "引发", "洪水"], role_seq=[1, 1, 1],
                            collect_type=COLLECT_PRECEDES, source=SOURCE_BARE_TEXT)
              for _ in range(3)]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="dedup_gate_restore",
                            rounds_per_stage=1)
    assert gates.COOCCURS_DEDUP_MODE is False, "默认 OFF"
    assert gates.COOCCURS_WINDOW_MODE is False, "默认 OFF"
    formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())
    assert gates.COOCCURS_DEDUP_MODE is False, "finally 回归 OFF（防误删 formal_train reset）"
    assert gates.COOCCURS_WINDOW_MODE is False, "finally 回归 OFF"
