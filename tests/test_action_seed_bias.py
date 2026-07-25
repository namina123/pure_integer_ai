"""B-PR4 attractor 多节点种子偏向测试（doc §19·镜像 test_728_memory_replay / test_action_intent_override 反 theater 范式）。

B-PR4 = 命令态动作词概念（D:11 源端 word concept）作多节点种子注入 attractor e₀（mirror #728 replay 扩张）→
PR 偏向动作拓扑邻域（§13.3·复用 attractor 不改数学）。experience_count 率消费者在此接通：
**洗净 filter（sn==0 tested-never-verified 滤除·ACTIVE 率消费者·非 theater）** + rate-sort survivors（dormant ordering）。

覆盖：
  - _collect_action_seed_candidates 原语：洗净 sn==0 滤除 / 冷启动 None→给机会 / sn>0 注入 / rate-sort 降序 /
    SHADOW 隔离 / 非动作词 skip / 纯读
  - dag_path 注入（mirror #728）：gate ON+子图内动作词→path 含动作支（反 theater 主锚）/ 子图外过滤 / gate OFF bit-identical
  - **元节点 no-op theater 牙**（矛盾 A·核心）：ACTION_* 元概念不在 PR matrix→add_seed no-op→强制选 (b) 动作词源端
  - gate 注册 + STOP/D6 合规（零新增表/ATTR）

**反 theater 牙**：元节点 no-op（矛盾 A）+ 注入 path 变（mirror #728 真活）+ 洗净 filter（率真消费）+ bit-identical（gate OFF 零变）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES, EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.experience_count import (
    register_experience_count, record_experience_outcome, DEFAULT_CTX_CODE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import IntentType, INTENT_QUESTION, INTENT_COMMAND, LANG_ZH
from pure_integer_ai.cognition.shared.action_primitives import (
    ACTION_GENERATE, ACTION_COMPUTE, ensure_action_primitives,
)
from pure_integer_ai.cognition.understanding.word_concept_signal import bootstrap_action_signals
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.dag_path import dag_path_step
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper
from pure_integer_ai.experiments.formal_train import _collect_action_seed_candidates
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def action_seed_env():
    """B-PR4 单测环境（dict backend·core space·composes_attr + experience_count 注册·boot 种 action D:11 PRIMARY 边）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)         # ATTR_OPERATION_INTENT=23 标记表（lookup_word_action readback）
    register_experience_count(b)      # _collect_action_seed_candidates 读率 + dag_path read_effective_freq
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    action_refs = ensure_action_primitives(ci, b, space_id=sid)   # 5 ACTION_INTENT_* concept + ATTR 旗标
    bootstrap_action_signals(ci, es, b, space_id=sid, langs={LANG_ZH})   # 种子词 D:11 边（帮我/请/生成/计算 PRIMARY）
    yield b, sid, es, ci, action_refs
    b.close()


def _segs(*tokens: str):
    """构造 segments stand-in（helper 只读 seg.tokens·SimpleNamespace 够）。"""
    return [SimpleNamespace(tokens=list(tokens))]


# ============ _collect_action_seed_candidates（洗净 filter + rate-sort·ACTIVE 率消费者） ============

def test_collect_cold_start_none_injected(action_seed_env):
    """冷启动：动作词 action_ref 无 experience_count 行→read None→给机会注入（rate=0·mirror structure_discover:1148）。"""
    b, sid, es, ci, action_refs = action_seed_env
    word_ref = ci.lookup("计算", sid)   # 计算→ACTION_COMPUTE D:11 PRIMARY
    assert word_ref is not None
    out = _collect_action_seed_candidates(segments=_segs("计算"), backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci,
                                          intent_type=INTENT_COMMAND, ctx_code=DEFAULT_CTX_CODE)
    assert out == [word_ref], "冷启动 None→给机会注入（计算 word_ref）"


def test_collect_sn_positive_injected(action_seed_env):
    """sn>0（reward>0 验证过）：动作词注入（率>0·非洗净滤除）。"""
    b, sid, es, ci, action_refs = action_seed_env
    word_ref = ci.lookup("计算", sid)
    compute_ref = action_refs[ACTION_COMPUTE]
    record_experience_outcome(b, ref=compute_ref, reward=1, ctx_code=DEFAULT_CTX_CODE)   # sn=1,tn=1→率 500
    out = _collect_action_seed_candidates(segments=_segs("计算"), backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci,
                                          intent_type=INTENT_COMMAND, ctx_code=DEFAULT_CTX_CODE)
    assert out == [word_ref], "sn>0→注入"


def test_collect_wash_filter_sn0_excluded(action_seed_env):
    """**洗净 filter 牙**（ACTIVE 率消费者·非 theater）：action_ref sn==0（tested-never-verified·reward==0 写）→滤除不注入。

    与 gate③ D:11 存在性正交（gate③=边存在·B-PR4=经验质量）：计算有 D:11 边（gate③ 过）但 sn==0（验过皆败）→B-PR4 滤除。
    """
    b, sid, es, ci, action_refs = action_seed_env
    compute_ref = action_refs[ACTION_COMPUTE]
    record_experience_outcome(b, ref=compute_ref, reward=0, ctx_code=DEFAULT_CTX_CODE)   # sn=0,tn=1→tested-never-verified
    out = _collect_action_seed_candidates(segments=_segs("计算"), backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci,
                                          intent_type=INTENT_COMMAND, ctx_code=DEFAULT_CTX_CODE)
    assert out == [], "sn==0 tested-never-verified→洗净滤除（ACTIVE 率消费者·非 theater）"


def test_collect_rate_sort_descending(action_seed_env):
    """rate-sort 降序（mirror structure_discover:1154 stable sort）：计算率 333 < 生成率 1000→返 [生成, 计算]。"""
    b, sid, es, ci, action_refs = action_seed_env
    calc_ref = ci.lookup("计算", sid)
    gen_ref = ci.lookup("生成", sid)
    assert calc_ref is not None and gen_ref is not None
    compute_meta = action_refs[ACTION_COMPUTE]
    generate_meta = action_refs[ACTION_GENERATE]
    # 计算：1 成功 2 失败 → sn=1,tn=3 → 率 1*1000//4=250
    record_experience_outcome(b, ref=compute_meta, reward=1, ctx_code=DEFAULT_CTX_CODE)
    record_experience_outcome(b, ref=compute_meta, reward=0, ctx_code=DEFAULT_CTX_CODE)
    record_experience_outcome(b, ref=compute_meta, reward=0, ctx_code=DEFAULT_CTX_CODE)
    # 生成：2 成功 0 失败 → sn=2,tn=2 → 率 2*1000//4=500
    record_experience_outcome(b, ref=generate_meta, reward=1, ctx_code=DEFAULT_CTX_CODE)
    record_experience_outcome(b, ref=generate_meta, reward=1, ctx_code=DEFAULT_CTX_CODE)
    out = _collect_action_seed_candidates(segments=_segs("计算", "生成"), backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci,
                                          intent_type=INTENT_COMMAND, ctx_code=DEFAULT_CTX_CODE)
    assert out == [gen_ref, calc_ref], "率降序：生成(500) 先于 计算(250)"


def test_collect_shadow_isolated(action_seed_env):
    """SHADOW 隔离牙：仅 SHADOW D:11 边的词→tier_filter=PRIMARY 过滤→不收（反 theater：未验证假设不注入）。"""
    b, sid, es, ci, action_refs = action_seed_env
    shadow_word = ci.ensure("编写shadow", space_id=sid, tier=TIER_PRIMARY)
    gen_ref = action_refs[ACTION_GENERATE]
    es.add(space_id_from=shadow_word[0], local_id_from=shadow_word[1],
           space_id_to=gen_ref[0], local_id_to=gen_ref[1],
           edge_type=EDGE_RELATION_SIGNAL, strength=1, source=SOURCE_TEACHER,
           tier=TIER_SHADOW, epistemic_origin=EPI_STRUCTURED)
    out = _collect_action_seed_candidates(segments=_segs("编写shadow"), backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci,
                                          intent_type=INTENT_COMMAND, ctx_code=DEFAULT_CTX_CODE)
    assert out == [], "仅 SHADOW D:11→tier_filter PRIMARY 过滤→不收"


def test_collect_non_action_word_skipped(action_seed_env):
    """非动作词（无 D:11 PRIMARY ACTION_* 边）→ skip（ci.ensure 新建中性词）。"""
    b, sid, es, ci, action_refs = action_seed_env
    ci.ensure("中性词", space_id=sid, tier=TIER_PRIMARY)
    out = _collect_action_seed_candidates(segments=_segs("中性词"), backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci,
                                          intent_type=INTENT_COMMAND, ctx_code=DEFAULT_CTX_CODE)
    assert out == [], "非动作词→无 D:11 边→skip"


def test_collect_is_pure_read(action_seed_env):
    """_collect_action_seed_candidates 纯读（lookup + read_experience_count 只查不写）·调前后 edge/experience_count 零新增。"""
    b, sid, es, ci, action_refs = action_seed_env
    edges_before = len(b.select("edge", where=None))
    ec_before = len(b.select("experience_count", where=None))
    _collect_action_seed_candidates(segments=_segs("计算"), backend=b, edge_store=es,
                                    space_id=sid, concept_index=ci,
                                    intent_type=INTENT_COMMAND, ctx_code=DEFAULT_CTX_CODE)
    assert len(b.select("edge", where=None)) == edges_before, "helper 不写边"
    assert len(b.select("experience_count", where=None)) == ec_before, "helper 不写 experience_count"


def test_collect_question_no_budget(action_seed_env):
    """**intent 守牙**（mirror B-PR2 test_feed_question_no_write·doc §19.1 决断7 intent 闸）：QUESTION intent → 返 [] 不预算。

    helper 内部守 intent_type != INTENT_COMMAND → 早返 []（QUESTION 不路由动作执行·不预算动作种子·§13.3）。
    caller 守 gate·helper 守 intent（mirror B-PR2 _feed_action_experience:2654 范式·单测可直验）。
    """
    b, sid, es, ci, action_refs = action_seed_env
    out = _collect_action_seed_candidates(segments=_segs("计算"), backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci,
                                          intent_type=INTENT_QUESTION, ctx_code=DEFAULT_CTX_CODE)
    assert out == [], "QUESTION intent → helper intent 守返 []（不预算动作种子）"


def test_collect_beta_arith_tied_stable(action_seed_env):
    """**β_arith tied 牙**（doc §19.4·诚实降级）：两动作词率相同（无 veto·sn==tn→率恒 500）→ 全注入·stable sort 保插入序。

    β_arith 病（无 veto 时率恒 1·排序无判别力）：诚实降级——全注入·序由 segments 遍历序（bit-identical·
    Python stable sort 同 key 保原序）·非伪造判别力。判别力 defer veto 触发（§17.4 同病）。
    """
    b, sid, es, ci, action_refs = action_seed_env
    calc_ref = ci.lookup("计算", sid)
    gen_ref = ci.lookup("生成", sid)
    assert calc_ref is not None and gen_ref is not None
    compute_meta = action_refs[ACTION_COMPUTE]
    generate_meta = action_refs[ACTION_GENERATE]
    # 两词同率：各 1 成功 1 总 → sn=1,tn=1 → 率 500（tied·β_arith 无 veto 场景）
    record_experience_outcome(b, ref=compute_meta, reward=1, ctx_code=DEFAULT_CTX_CODE)
    record_experience_outcome(b, ref=generate_meta, reward=1, ctx_code=DEFAULT_CTX_CODE)
    out = _collect_action_seed_candidates(segments=_segs("计算", "生成"), backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci,
                                          intent_type=INTENT_COMMAND, ctx_code=DEFAULT_CTX_CODE)
    # 全注入（率 tied 不滤除·非洗净）+ stable sort 保插入序 [计算, 生成]（bit-identical）
    assert out == [calc_ref, gen_ref], "β_arith tied → 全注入·stable sort 保插入序（诚实降级非伪造判别力）"


# ============ dag_path 注入（mirror #728 replay 扩张·反 theater 主锚） ============

def _edge(es, sid, frm, to, et, *, strength=1):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)


def _path_nodes(pr) -> set:
    nodes = set()
    for e in pr.path.edges:
        nodes.add((e[0], e[1]))
        nodes.add((e[2], e[3]))
    return nodes


def test_dag_path_action_seed_injection_live(action_seed_env):
    """**反 theater 主锚**（mirror test_728:87）：gate ON + action_seed_candidates=[子图内动作词]→注入→path 含动作支。

    subgraph：seed→A→sink1（主支）+ 动作词→B（独立支·动作词非种子非 sink）。
    action_seed_candidates=[] → path 不含动作词支；=[动作词]（在 subgraph）→ path 含动作词/B。
    **反 theater 牙**：注入 path 变 = local_seeds 扩张真活在决策路径·非伪造标志。
    """
    b, sid, es, ci, action_refs = action_seed_env
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    act_word = ci.ensure("动作词", space_id=sid, tier=TIER_PRIMARY)   # 子图内动作词（占位·dag_path 读预算列表）
    B = ci.ensure("B", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    _edge(es, sid, act_word[1], B[1], EDGE_PRECEDES)
    dag_edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=sink1)

    saved = gates.ACTION_SEED_BIAS_MODE
    try:
        # action_seed_candidates=[] → path 不含动作词支
        gates.ACTION_SEED_BIAS_MODE = True
        wm1 = WorkMemory()
        pr1 = dag_path_step(dag_edges, [seed], wm1, intent, backend=b)
        nodes1 = _path_nodes(pr1)
        # action_seed_candidates=[act_word]（act_word in subgraph）→ path 含动作词支
        wm2 = WorkMemory()
        wm2.action_seed_candidates = [act_word]
        pr2 = dag_path_step(dag_edges, [seed], wm2, intent, backend=b)
        nodes2 = _path_nodes(pr2)
    finally:
        gates.ACTION_SEED_BIAS_MODE = saved
    assert act_word not in nodes1, "action_seed_candidates=[] path 不该含动作词支"
    assert act_word in nodes2, "action_seed_candidates=[act_word] path 该含动作词（注入真活）"
    assert B in nodes2, "action_seed_candidates=[act_word] path 该含 B（动作词→B 边加入）"
    assert nodes1 != nodes2, "注入 path 变 = B-PR4 真活非 theater"


def test_dag_path_action_seed_filter_outside_subgraph(action_seed_env):
    """subgraph 过滤牙（mirror #728:126）：action_seed_candidates=[子图外 ref]→过滤→bit-identical。"""
    b, sid, es, ci, action_refs = action_seed_env
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    dag_edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=sink1)

    saved = gates.ACTION_SEED_BIAS_MODE
    try:
        gates.ACTION_SEED_BIAS_MODE = True
        wm1 = WorkMemory()
        pr1 = dag_path_step(dag_edges, [seed], wm1, intent, backend=b)
        wm2 = WorkMemory()
        wm2.action_seed_candidates = [(sid, 9999)]   # 不在 subgraph
        pr2 = dag_path_step(dag_edges, [seed], wm2, intent, backend=b)
    finally:
        gates.ACTION_SEED_BIAS_MODE = saved
    assert _path_nodes(pr1) == _path_nodes(pr2), \
        "子图外动作词过滤 → local_seeds == seeds → bit-identical"


def test_dag_path_action_seed_gate_off_bit_identical(action_seed_env):
    """gate OFF bit-identical 牙（dag_path 侧 gate 守·三保险之一）：gate OFF + action_seed_candidates 非空→不注入→bit-identical。

    即使 workmem.action_seed_candidates 被设（防御），dag_path `if ACTION_SEED_BIAS_MODE:` 守跳过→local_seeds==seeds。
    """
    b, sid, es, ci, action_refs = action_seed_env
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    act_word = ci.ensure("动作词", space_id=sid, tier=TIER_PRIMARY)
    B = ci.ensure("B", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    _edge(es, sid, act_word[1], B[1], EDGE_PRECEDES)
    dag_edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=sink1)

    saved = gates.ACTION_SEED_BIAS_MODE
    try:
        gates.ACTION_SEED_BIAS_MODE = False   # 生产默认
        wm1 = WorkMemory()
        pr1 = dag_path_step(dag_edges, [seed], wm1, intent, backend=b)
        wm2 = WorkMemory()
        wm2.action_seed_candidates = [act_word]   # 非空但 gate OFF
        pr2 = dag_path_step(dag_edges, [seed], wm2, intent, backend=b)
    finally:
        gates.ACTION_SEED_BIAS_MODE = saved
    assert _path_nodes(pr1) == _path_nodes(pr2), \
        "gate OFF → dag_path 侧 gate 守跳过 → bit-identical（三保险之一）"


# ============ 元节点 no-op theater 牙（矛盾 A·核心·doc §19.0） ============

def test_meta_node_not_in_pr_matrix_noop(action_seed_env):
    """**矛盾 A 核心 theater 牙**：ACTION_* 元概念（D:11 目标端 target）不在 PR matrix.index → add_seed no-op → PR 不变。

    证注入 ACTION_* 元概念 = 静默 no-op theater（add_seed:236 `if c not in self.matrix.index: return`·
    solve_exact `if node in matrix.index` 静默丢弃）。强制 B-PR4 选 (b) 动作词源端（在 matrix）·非 (a) 元概念。
    对照：子图内节点 add_seed 真改 _x（in-matrix seed 真活）。
    """
    b, sid, es, ci, action_refs = action_seed_env
    n1 = ci.ensure("n1", space_id=sid, tier=TIER_PRIMARY)
    n2 = ci.ensure("n2", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, n1[1], n2[1], EDGE_PRECEDES)   # PRECEDES 边进 PR matrix（D:11 不进）
    dag_edges = b.select("edge")
    compute_meta = action_refs[ACTION_COMPUTE]   # __ACTION_COMPUTE__ 元概念（只有 D:11 边）
    wrapper = A3PRWrapper.build(dag_edges, backend=b)

    # 矛盾 A 核心事实：元概念不在 PR matrix.index（只有 D:11 边·D:11 不进 PR 邻接）
    assert compute_meta not in wrapper.matrix.index, \
        "ACTION_* 元概念不在 PR matrix（只有 D:11 边·D:11 不进 PR 邻接·矛盾 A）"
    assert n1 in wrapper.matrix.index and n2 in wrapper.matrix.index, "子图节点在 matrix（PRECEDES 边进 PR）"

    wrapper.solve([n1])   # 单种子 n1（in matrix）设 _x
    snap1 = wrapper.snapshot()
    wrapper.add_seed(compute_meta)   # 元概念 add_seed → no-op（c not in matrix.index）
    snap2 = wrapper.snapshot()
    assert snap2 == snap1, "元概念 add_seed no-op → PR 不变（注入元概念 = theater）"

    wrapper.add_seed(n2)   # 对照：子图内节点 add_seed → _x 真变（in-matrix seed 真活）
    snap3 = wrapper.snapshot()
    assert snap3 != snap1, "对照：子图内节点 add_seed 真改 PR（in-matrix seed 真活·非 no-op）"


# ============ gate 注册 + STOP/D6 合规 ============

def test_action_seed_bias_gate_exists_and_bool():
    """gate ACTION_SEED_BIAS_MODE 已注册 + bool 类型（default OFF·_flag(False) 守 CI·生产 try/finally 暂不翻）。

    实际 OFF 行为由 test_dag_path_action_seed_gate_off_bit_identical + 全量回归零翻证。"""
    assert hasattr(gates, "ACTION_SEED_BIAS_MODE"), "gate 已注册"
    assert isinstance(gates.ACTION_SEED_BIAS_MODE, bool), "gate 是 bool 类型"
    assert gates.ACTION_SEED_BIAS_MODE is False, "default OFF（守 CI bit-identical）"


def test_bpr4_no_new_attr_or_table():
    """STOP+D6 合规：B-PR4 复用 ATTR_OPERATION_INTENT=23（B-PR1）+ 既有 D:11/lookup_word_action/read_experience_count/
    workmem 字段·零新增 ATTR/表。action_seed_candidates 是 workmem 字段（非 ATTR/表/结构 kind）·守 minimal extension。"""
    from pure_integer_ai.storage import composes_attr as ca
    assert ca.ATTR_OPERATION_INTENT == 23, "复用 ATTR_OPERATION_INTENT=23（B-PR1·非 B-PR4 新增）"
    assert not hasattr(ca, "ATTR_ACTION_SEED"), "无 ATTR_ACTION_SEED（种子偏向走 workmem 字段非 ATTR）"
