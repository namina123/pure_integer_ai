"""#478 STRUCT_BIND 跨模态槽位级绑定建边测试（doc/重来_任务0478_STRUCT_BIND_设计.md·§8.7-P2）。

覆盖（实施 4·决断 4 + 决断 7）：
  - collect_skeleton_slot_refs：骨架 PARAM 槽 ref DFS 前序提取（镜像 _collect_slot_lcas 范式）。
  - build_struct_bind_edge：单条边 + 自环跳（反同模态自绑）+ order_index 槽序。
  - bootstrap_struct_bind_edges：批量建边 + **幂等 query_from 按源 skip** + **空 pairs 短路**（bit-identical 硬守）。
  - 通用模态对：bind 模态A 骨架槽 ↔ 模态B 骨架槽（决断 1 通用原语·模态无关）。
  - 反 theater 形态 2：建边 caller live 但**零消费 reader**（#730 未跟进·诚实标·决断 7）。
  - bit-identical：gate STRUCT_BIND_BOOT_MODE default OFF → formal_train 全流程零 STRUCT_BIND 边。
  - loader：load_struct_bind_pairs_file / resolve_struct_bind_pairs（E10 + E5 graceful·决断 2 来源 a）。

铁律：纯整数 / 确定性 bit-identical / 不写死（外部文件·core 不 import）/ 单向依赖（cognition/process→storage 向下）。
诚实边界：stable≠correct（结构对齐非语义绑定·#479 墙）/ 来源 b 跨模态结构对齐 defer / 边 theater 形态 2（#730 reader 未跟进）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, SOURCE_CONCEPTNET
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_STRUCT_BIND, EDGE_COMPOSES
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, record_composes_attr, ATTR_OPERAND,
)
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.process.struct_bind import (
    bootstrap_struct_bind_edges, build_struct_bind_edge,
    collect_skeleton_slot_refs, STRUCT_BIND_STRENGTH,
)
from pure_integer_ai.experiments.collection import (
    load_struct_bind_pairs_file, resolve_struct_bind_pairs,
    CollectedItem, COLLECT_PRECEDES,
)


# ---- fixtures ----

@pytest.fixture
def env():
    """struct_bind 单测环境（dict backend·core space·composes_attr 注册·edge_store + concept_index + graph）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    g = ConceptGraph(b)
    sid = sp.space_id
    yield b, sid, es, ci, g
    b.close()


def _build_skeleton(ci: ConceptIndex, es: EdgeStore, g: ConceptGraph, sid: int,
                    n_slots: int, *, surface_prefix: str) -> ConceptRef:
    """建 n_slots 槽骨架（NOP root + n PARAM 叶·EDGE_COMPOSES order_index=slot·叶 ATTR_OPERAND=make_variable(slot)）。

    镜像 structure_discover._SkeletonBuilder build:310/328/376（record_composes_attr ATTR_OPERAND）+
    EDGE_COMPOSES 建边。返 skeleton root ref（collect_skeleton_slot_refs 输入）。
    """
    root = ci.ensure(f"{surface_prefix}_root", space_id=sid, tier=TIER_PRIMARY)
    for slot in range(n_slots):
        leaf = ci.ensure(f"{surface_prefix}_slot{slot}", space_id=sid, tier=TIER_PRIMARY)
        record_composes_attr(ci._b, ref=leaf, kind=ATTR_OPERAND, int_a=make_variable(slot))
        es.add(
            space_id_from=root[0], local_id_from=root[1],
            space_id_to=leaf[0], local_id_to=leaf[1],
            edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_CONCEPTNET,
            order_index=slot,   # DFS 前序（read_composes_tree 按 order_index 排）
            tier=TIER_PRIMARY,
        )
    return root


# ============ unit：collect_skeleton_slot_refs（PARAM 槽 ref DFS 前序） ============

def test_collect_skeleton_slot_refs_returns_param_slots_in_order(env):
    """2 槽骨架 → collect_skeleton_slot_refs 返 2 ref·DFS 前序（slot0 < slot1 by order_index）。"""
    b, sid, es, ci, g = env
    root = _build_skeleton(ci, es, g, sid, n_slots=2, surface_prefix="sk_a")
    refs = collect_skeleton_slot_refs(b, g, root)
    assert len(refs) == 2
    # DFS 前序：slot0 ref != slot1 ref·各为 PARAM 叶 ConceptRef
    assert refs[0] != refs[1]
    assert all(isinstance(r, tuple) and len(r) == 2 for r in refs)


def test_collect_skeleton_slot_refs_empty_skeleton(env):
    """0 槽骨架（root 无 PARAM 叶）→ 返空 list（无 slot 可绑定）。"""
    b, sid, es, ci, g = env
    root = ci.ensure("bare_root", space_id=sid, tier=TIER_PRIMARY)   # 无子·无 ATTR_OPERAND
    refs = collect_skeleton_slot_refs(b, g, root)
    assert refs == []


# ============ unit：build_struct_bind_edge（单条 + 自环跳 + 槽序） ============

def test_build_struct_bind_edge_creates_edge_with_order_index(env):
    """单条边：slot_a → slot_b·edge_type=STRUCT_BIND·order_index=传入值·strength=STRUCT_BIND_STRENGTH。"""
    b, sid, es, ci, g = env
    slot_a = ci.ensure("a_slot0", space_id=sid, tier=TIER_PRIMARY)
    slot_b = ci.ensure("b_slot0", space_id=sid, tier=TIER_PRIMARY)
    n = build_struct_bind_edge(es, slot_a, slot_b, source=SOURCE_TEACHER,
                               space_id=sid, order_index=0)
    assert n == 1
    rows = es.query_from(slot_a[0], slot_a[1], edge_type=EDGE_STRUCT_BIND)
    assert len(rows) == 1
    row = rows[0]
    assert row["edge_type"] == EDGE_STRUCT_BIND
    assert (row["space_id_to"], row["local_id_to"]) == slot_b
    assert row["order_index"] == 0
    assert row["strength"] == STRUCT_BIND_STRENGTH
    assert row["source"] == SOURCE_TEACHER


def test_build_struct_bind_edge_self_loop_skipped(env):
    """自环跳（slot_a==slot_b·反同模态自绑·决断 1 STRUCT_BIND 不合并身份）。"""
    b, sid, es, ci, g = env
    slot = ci.ensure("self_slot", space_id=sid, tier=TIER_PRIMARY)
    n = build_struct_bind_edge(es, slot, slot, source=SOURCE_TEACHER,
                               space_id=sid, order_index=0)
    assert n == 0
    assert es.query_from(slot[0], slot[1], edge_type=EDGE_STRUCT_BIND) == []


def test_build_struct_bind_edge_cross_space_asserts(env):
    """跨 space 错配防御（审 1 P2-1）：slot_ref space ≠ space_id 入参 → assert 抛（防教师标注/跨 space corrupt）。

    用合成 cross-space ref（sid+99）·因 AbstractSpace.create 二次 registry 实例会复用 space_id（in-memory
    per-instance·非本测关切）·assert 在 edge_store.add 前 fire·合成 ref 足验。"""
    b, sid, es, ci, g = env
    slot_a = ci.ensure("a", space_id=sid, tier=TIER_PRIMARY)
    slot_b = (sid + 99, 1)   # 合成异 space ref（assert 先于 DB 写 fire·无须真建 space）
    with pytest.raises(AssertionError):
        build_struct_bind_edge(es, slot_a, slot_b, source=SOURCE_TEACHER,
                               space_id=sid, order_index=0)


def test_collect_skeleton_slot_refs_param_only_skips_immediate(env):
    """PARAM-only 诚实边界（审 2 P1-2）：immediate 叶（ATTR_IMMEDIATE·无 ATTR_OPERAND/ATTR_OPERATOR）
    静默不入 refs → refs 序号 = PARAM-only 序（slot_map 须按 PARAM-only 标·非 build 全槽序）。
    """
    from pure_integer_ai.storage.composes_attr import ATTR_IMMEDIATE
    b, sid, es, ci, g = env
    root = ci.ensure("mixed_root", space_id=sid, tier=TIER_PRIMARY)
    # PARAM 叶（入 refs）+ immediate 叶（不入 refs）
    param_leaf = ci.ensure("param0", space_id=sid, tier=TIER_PRIMARY)
    imm_leaf = ci.ensure("imm0", space_id=sid, tier=TIER_PRIMARY)
    record_composes_attr(b, ref=param_leaf, kind=ATTR_OPERAND, int_a=make_variable(0))
    record_composes_attr(b, ref=imm_leaf, kind=ATTR_IMMEDIATE, int_a=42)   # immediate·无 ATTR_OPERAND
    es.add(space_id_from=root[0], local_id_from=root[1],
           space_id_to=param_leaf[0], local_id_to=param_leaf[1],
           edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_CONCEPTNET,
           order_index=0, tier=TIER_PRIMARY)
    es.add(space_id_from=root[0], local_id_from=root[1],
           space_id_to=imm_leaf[0], local_id_to=imm_leaf[1],
           edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_CONCEPTNET,
           order_index=1, tier=TIER_PRIMARY)
    refs = collect_skeleton_slot_refs(b, g, root)
    assert refs == [param_leaf]   # 仅 PARAM operand 叶入 refs·immediate 叶静默不入


# ============ unit：bootstrap_struct_bind_edges（批量 + 幂等 + 空短路 + 通用模态对） ============

def test_bootstrap_empty_pairs_short_circuits_no_side_effects(env):
    """空 bind_pairs → return 0·**绝不调 query_from/build**（bit-identical P0·镜像 bootstrap_is_a_edges:119）。

    用 monkeypatch 守 query_from 未被调（空 pairs 路径硬守）。
    """
    b, sid, es, ci, g = env
    calls = {"query_from": 0}
    orig = es.query_from

    def _count(*a, **k):
        calls["query_from"] += 1
        return orig(*a, **k)
    es.query_from = _count   # type: ignore[assignment]
    try:
        n = bootstrap_struct_bind_edges(es, [], space_id=sid)
    finally:
        es.query_from = orig   # type: ignore[assignment]
    assert n == 0
    assert calls["query_from"] == 0   # 空 pairs 短路·query_from 未调


def test_bootstrap_builds_edges_for_generic_modality_pair(env):
    """通用模态对：模态A 2 槽骨架 ↔ 模态B 2 槽骨架·slot_map 0:0 1:1 → 2 STRUCT_BIND 边（决断 1 通用原语）。"""
    b, sid, es, ci, g = env
    root_a = _build_skeleton(ci, es, g, sid, n_slots=2, surface_prefix="arith_add")
    root_b = _build_skeleton(ci, es, g, sid, n_slots=2, surface_prefix="lang_sum")
    slots_a = collect_skeleton_slot_refs(b, g, root_a)
    slots_b = collect_skeleton_slot_refs(b, g, root_b)
    bind_pairs = [(slots_a[0], slots_b[0]), (slots_a[1], slots_b[1])]
    n = bootstrap_struct_bind_edges(es, bind_pairs, space_id=sid)
    assert n == 2
    # 验证两槽均建边·order_index=0/1
    r0 = es.query_from(slots_a[0][0], slots_a[0][1], edge_type=EDGE_STRUCT_BIND)
    r1 = es.query_from(slots_a[1][0], slots_a[1][1], edge_type=EDGE_STRUCT_BIND)
    assert len(r0) == 1 and (r0[0]["space_id_to"], r0[0]["local_id_to"]) == slots_b[0]
    assert len(r1) == 1 and (r1[0]["space_id_to"], r1[0]["local_id_to"]) == slots_b[1]
    assert r0[0]["order_index"] == 0 and r1[0]["order_index"] == 1


def test_bootstrap_idempotent_skip_same_source_same_pair(env):
    """幂等 skip（镜像 bootstrap_is_a_edges:128-137）：重复建同 (slot_a,slot_b,source) → skip·返 0。"""
    b, sid, es, ci, g = env
    slot_a = ci.ensure("a0", space_id=sid, tier=TIER_PRIMARY)
    slot_b = ci.ensure("b0", space_id=sid, tier=TIER_PRIMARY)
    n1 = bootstrap_struct_bind_edges(es, [(slot_a, slot_b)], space_id=sid)
    n2 = bootstrap_struct_bind_edges(es, [(slot_a, slot_b)], space_id=sid)   # 同源同三元组
    assert n1 == 1 and n2 == 0
    assert len(es.query_from(slot_a[0], slot_a[1], edge_type=EDGE_STRUCT_BIND)) == 1


def test_bootstrap_different_source_not_skipped(env):
    """异源不 skip（按源细化·决断 4 审 2 P2-1）：同 (slot_a,slot_b) 异 source → 各建一条（防错 merge）。"""
    b, sid, es, ci, g = env
    slot_a = ci.ensure("a0", space_id=sid, tier=TIER_PRIMARY)
    slot_b = ci.ensure("b0", space_id=sid, tier=TIER_PRIMARY)
    n1 = bootstrap_struct_bind_edges(es, [(slot_a, slot_b)], space_id=sid, source=SOURCE_TEACHER)
    n2 = bootstrap_struct_bind_edges(es, [(slot_a, slot_b)], space_id=sid, source=SOURCE_CONCEPTNET)
    assert n1 == 1 and n2 == 1   # 异源各建一条
    rows = es.query_from(slot_a[0], slot_a[1], edge_type=EDGE_STRUCT_BIND)
    assert len(rows) == 2
    assert sorted(r["source"] for r in rows) == [SOURCE_CONCEPTNET, SOURCE_TEACHER]


def test_bootstrap_idempotent_repeat_call_order_index_stable(env):
    """重复调幂等 + order_index 稳定（决断 4）：同 bind_pairs 两次 bootstrap → 第二次全 skip·order_index 不重排。

    注：本测在同 EdgeStore 实例上重复调（验 query_from skip + order_index 稳定）·**非 load_run resume 场景**。
    真跨实例 resume（dump/load 还原 edge 行 + 新 EdgeStore boot 再种）由 e2e
    `test_formal_train_struct_bind_gate_on_boot_caller_builds_edges` 两阶段 formal_train 部分覆盖
    （phase1 gate OFF / phase2 gate ON 各建独立 DictBackend·同 corpus 同名确定性）。
    """
    b, sid, es, ci, g = env
    root_a = _build_skeleton(ci, es, g, sid, n_slots=2, surface_prefix="x_a")
    root_b = _build_skeleton(ci, es, g, sid, n_slots=2, surface_prefix="x_b")
    slots_a = collect_skeleton_slot_refs(b, g, root_a)
    slots_b = collect_skeleton_slot_refs(b, g, root_b)
    pairs = [(slots_a[0], slots_b[0]), (slots_a[1], slots_b[1])]
    n1 = bootstrap_struct_bind_edges(es, pairs, space_id=sid)
    n2 = bootstrap_struct_bind_edges(es, pairs, space_id=sid)
    assert n1 == 2 and n2 == 0   # 第二次全 skip（幂等）
    # order_index 稳定（slot0→0·slot1→1·未因 skip 重排）
    r0 = es.query_from(slots_a[0][0], slots_a[0][1], edge_type=EDGE_STRUCT_BIND)
    r1 = es.query_from(slots_a[1][0], slots_a[1][1], edge_type=EDGE_STRUCT_BIND)
    assert r0[0]["order_index"] == 0 and r1[0]["order_index"] == 1


# ============ unit：loader（load_struct_bind_pairs_file / resolve·E10 + E5 graceful） ============

def test_load_struct_bind_pairs_file_parses_sample():
    """sample 文件解析：name_a name_b a:b c:d → (name_a, name_b, [(a,b),(c,d)])·决断 2 按位序。"""
    pairs = load_struct_bind_pairs_file("data/struct_bind_pairs.txt.sample")
    assert len(pairs) == 1
    na, nb, slot_map = pairs[0]
    assert na == "__op_disc_6575600255134327604"
    assert nb == "__op_disc_7725092591093092863"
    assert slot_map == [(0, 0), (1, 1)]


def test_load_struct_bind_pairs_file_graceful_malformed(tmp_path):
    """E5 graceful：注释/空行/<3 段/坏 slot 对 skip·不抛崩·决断 2 容错范式。"""
    p = tmp_path / "sb.txt"
    p.write_text(
        "# comment\n\n"
        "badline\n"              # <3 段 skip
        "name_a name_b 0:0\n"   # 1 对
        "x y 1:2 3:4\n"          # 2 对
        "z w notapair 5:6\n"    # notapair skip·5:6 留
        "p q -1:2\n",            # 负 idx skip·全对失败 → 行 skip
        encoding="utf-8",
    )
    pairs = load_struct_bind_pairs_file(str(p))
    assert pairs == [
        ("name_a", "name_b", [(0, 0)]),
        ("x", "y", [(1, 2), (3, 4)]),
        ("z", "w", [(5, 6)]),
    ]


def test_resolve_struct_bind_pairs_no_env_returns_empty(monkeypatch):
    """resolve 无 PURE_INTEGER_AI_LOCAL_DIR → 返 []（bit-identical·CI/生产 default·镜像 resolve_is_a_facts）。"""
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    assert resolve_struct_bind_pairs() == []


def test_resolve_struct_bind_pairs_missing_file_returns_empty(monkeypatch, tmp_path):
    """resolve 目录存在但文件不存在 → 返 []（E5 graceful·bit-identical）。"""
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(tmp_path))
    assert resolve_struct_bind_pairs() == []


# ============ e2e：formal_train boot 接线 + gate bit-identical（反 theater 形态 2） ============

def test_formal_train_struct_bind_gate_off_no_struct_bind_edges(tmp_path):
    """gate STRUCT_BIND_BOOT_MODE default OFF → formal_train 全流程零 STRUCT_BIND 边（bit-identical·决断 8）。

    反 theater 形态 2 诚实标：建边 caller code live（formal_train:1162+ 接线）但 gate OFF 不激活·
    builder 真由 unit 测验（上方）·#730 reader 未跟进=形态 2 theater·gate 控制 activate 时机（同 MODE_B 范式）。
    """
    from pure_integer_ai.config import gates
    from pure_integer_ai.experiments.formal_train import make_train_context, formal_train, FormalTrainConfig

    saved = gates.STRUCT_BIND_BOOT_MODE
    gates.STRUCT_BIND_BOOT_MODE = False
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        # 空 corpus（无 lang/arith/code item）→ discover 无算子→STRUCT_BIND 无可绑·gate OFF 亦不调
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "gate_off"), run_id="struct_bind_gate_off")
        formal_train(cfg, [], backend=b)
        es = ctx.edge_store
        # 守：全 edge 表零 STRUCT_BIND 边（gate OFF bit-identical）
        rows = es._b.select("edge", where={"edge_type": EDGE_STRUCT_BIND})
        assert rows == []
    finally:
        gates.STRUCT_BIND_BOOT_MODE = saved


def test_formal_train_struct_bind_gate_on_boot_caller_builds_edges(tmp_path, monkeypatch):
    """**反 theater e2e**（核心·决断 7）：gate ON + struct_bind_pairs.txt → formal_train boot caller
    经 resolve → discovered_operators name→ref → collect_skeleton_slot_refs → bootstrap_struct_bind_edges
    建**真**EDGE_STRUCT_BIND 边（builder 真活·非 theater·#730 reader 跟进后闭环合法）。

    两阶段（operator name 是 hash·须先 discover 拿名再写 file）：
      1. gate OFF run（2 异形 arith corpus·square arity1 + add arity2）→ discover 2 算子 → 捕获 name。
      2. 写 struct_bind_pairs.txt（bind square slot0 ↔ add slot0·异 skeleton 非自环）→
         gate ON + PURE_INTEGER_AI_LOCAL_DIR → formal_train → 验证 STRUCT_BIND 边真建。
    名确定性（_shape_name hash·同 corpus 同名）→ gate OFF run 捕获的名可用于 gate ON run。
    """
    from pure_integer_ai.config import gates
    from pure_integer_ai.experiments.formal_train import make_train_context, formal_train, FormalTrainConfig
    from pure_integer_ai.cognition.shared.types import MODALITY_ARITH, DOMAIN_MATH, LANG_NONE
    from pure_integer_ai.storage.edge_store import SOURCE_MATH

    def _two_op_corpus():
        items = []
        for p in ["b", "c", "d", "e"]:   # 'a'/'i' 是 lambda 保留名禁用
            items.append(CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                                       source=SOURCE_MATH, collect_type=COLLECT_PRECEDES,
                                       arith_source=f"lambda {p}: {p} * {p}"))   # square arity1
        for p in ["f", "g", "h", "j"]:
            items.append(CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                                       source=SOURCE_MATH, collect_type=COLLECT_PRECEDES,
                                       arith_source=f"lambda {p},q: {p} + q"))   # add arity2
        return items

    # --- 阶段 1：gate OFF → discover → 捕获名 ---
    saved = gates.STRUCT_BIND_BOOT_MODE
    gates.STRUCT_BIND_BOOT_MODE = False
    try:
        b1 = DictBackend()
        cfg1 = FormalTrainConfig(run_dir=str(tmp_path / "phase1"), run_id="phase1")
        result_off = formal_train(cfg1, _two_op_corpus(), backend=b1)
    finally:
        gates.STRUCT_BIND_BOOT_MODE = saved
    ops_by_arity = {o.arity: o for o in result_off.discovered_operators}
    assert 1 in ops_by_arity and 2 in ops_by_arity, \
        f"须 discover arity1+arity2 两算子·实得 {list(ops_by_arity)}"
    name_sq = ops_by_arity[1].name
    name_add = ops_by_arity[2].name

    # --- 阶段 2：写 file + gate ON → boot caller 建边 ---
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "struct_bind_pairs.txt").write_text(
        f"# bind square slot0 ↔ add slot0（异 skeleton 非自环）\n"
        f"{name_sq} {name_add} 0:0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(local_dir))
    gates.STRUCT_BIND_BOOT_MODE = True
    try:
        b2 = DictBackend()
        ctx2 = make_train_context(b2)
        cfg2 = FormalTrainConfig(run_dir=str(tmp_path / "phase2"), run_id="phase2")
        formal_train(cfg2, _two_op_corpus(), backend=b2)
        # 验证 STRUCT_BIND 边真建（boot caller 链路完整：resolve→name lookup→slot collect→build）
        rows = b2.select("edge", where={"edge_type": EDGE_STRUCT_BIND})
        assert len(rows) == 1, f"须建 1 STRUCT_BIND 边·实得 {len(rows)}"
        row = rows[0]
        assert row["source"] == SOURCE_TEACHER   # 默认 来源 a 教师标注
        assert row["strength"] == STRUCT_BIND_STRENGTH
        assert row["order_index"] == 0
    finally:
        gates.STRUCT_BIND_BOOT_MODE = saved


def test_formal_train_struct_bind_gate_on_missing_name_skips_pair(tmp_path, monkeypatch):
    """E5 graceful：gate ON + file 含未命中 name（跨 run mismatch/教师 typo）→ 该对 skip·不抛崩·零边。

    决断 4 name→ref 索引查不到（name 不在 discovered_operators）→ continue·建 0 边·formal_train 正常完成。
    """
    from pure_integer_ai.config import gates
    from pure_integer_ai.experiments.formal_train import make_train_context, formal_train, FormalTrainConfig
    from pure_integer_ai.cognition.shared.types import MODALITY_ARITH, DOMAIN_MATH, LANG_NONE
    from pure_integer_ai.storage.edge_store import SOURCE_MATH

    local_dir = tmp_path / "local2"
    local_dir.mkdir()
    (local_dir / "struct_bind_pairs.txt").write_text(
        "__op_disc_nonexistent_a __op_disc_nonexistent_b 0:0\n", encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(local_dir))
    saved = gates.STRUCT_BIND_BOOT_MODE
    gates.STRUCT_BIND_BOOT_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "typo"), run_id="typo")
        # 单 arith corpus（discover 1 算子·file 引用不存在的名→全 skip）
        corpus = [CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                                source=SOURCE_MATH, collect_type=COLLECT_PRECEDES,
                                arith_source="lambda b: b * b")]
        formal_train(cfg, corpus, backend=b)
        assert b.select("edge", where={"edge_type": EDGE_STRUCT_BIND}) == []   # 全 skip·零边·不抛崩
    finally:
        gates.STRUCT_BIND_BOOT_MODE = saved
