"""B-PR2 动作意图经验回写测试（doc §17·镜像 test_action_d11_readback / test_negation_d11_readback 反 theater 范式）。

B-PR2 = ACTION_* concept 动作验证率经验回写（experience_count·§16.4 三层职责之"经验"层·对偶 op_confidence）。
W7+B-PR1 落了词法（D:11）+ 概念（ATTR_OPERATION_INTENT=23 旗标）两层·B-PR2 落第三层经验：命令态成功 episode →
ACTION_* concept 在 COMMAND ctx_code 桶累积 e_sn/e_tn（reward 驱动 R1）。

覆盖：
  - collect_action_intent_concepts 单元（D:11 PRIMARY readback→distinct ACTION_* refs·去重·未知词 skip）
  - _feed_action_experience helper（D3 激活 + R1 写符号 + ctx_code COMMAND 桶隔离 + gate OFF bit-identical）

**D3 reward>0 = R1 成功臂非排除闸**（设计审 B CONFIRMED·§17.1 决断2）：
  reward>0 → e_sn++&e_tn++ / reward==0 veto → e_tn++ only → 率<1 有判别力（硬排除→率恒1 β_arith 病→B-PR2 无意义）。

**反 theater 干预测试**：gate OFF → 零写 / type≠COMMAND → 零写 / terminal≠REACHED_SINK → 零写（D3 三闸）。
**ctx_code 桶隔离**（§17.2）：COMMAND 桶写·QUESTION 桶读→None（自动隔离·pack_ctx_code 第4维 intent_type）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr, ATTR_OPERATION_INTENT
from pure_integer_ai.storage.experience_count import (
    register_experience_count, read_experience_count, pack_ctx_code,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    Segment, INTENT_COMMAND, INTENT_QUESTION,
    MODALITY_LANGUAGE, LANG_ZH, DOMAIN_TEXT,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END,
)
from pure_integer_ai.cognition.shared.action_primitives import (
    INTENT_COMMAND_MOOD, ACTION_GENERATE, ensure_action_primitives,
)
from pure_integer_ai.cognition.understanding.cue_words import collect_action_intent_concepts
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_action_signals, record_word_concept,
)
from pure_integer_ai.experiments.formal_train import _feed_action_experience
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def action_exp_env():
    """B-PR2 单测环境（dict backend·core space·composes_attr + experience_count 注册·boot 种 action D:11 边）。

    比 action_d11 fixture 多 register_experience_count（B-PR2 写此表·须注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)        # ATTR_OPERATION_INTENT=23 标记表
    register_experience_count(b)     # B-PR2 经验回写表（record/read experience_count）
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    action_refs = ensure_action_primitives(ci, b, space_id=sid)   # 5 ACTION_INTENT_* concept + ATTR 旗标
    bootstrap_action_signals(ci, es, b, space_id=sid, langs={LANG_ZH})   # 种子词 D:11 边（帮我/请/生成/计算 PRIMARY）
    yield b, sid, es, ns, ci, action_refs
    b.close()


def _seg(tokens):
    """语言域 Segment（tokens + LANG_ZH·collector 扫 seg.tokens）。"""
    return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                   domain=DOMAIN_TEXT, tokens=tokens)


def _ctx_code(intent_type):
    """pack ctx_code（DOMAIN_TEXT + MODALITY_LANGUAGE + task=0 + intent_type·同 _ctx_tag 写桶==读桶）。"""
    return pack_ctx_code(DOMAIN_TEXT, MODALITY_LANGUAGE, 0, intent_type)


# ============ collect_action_intent_concepts 单元 ============

def test_collect_seeded_command_and_action(action_exp_env):
    """segments '帮我 生成' → collector 返 COMMAND_MOOD ref + GENERATE ref（两种子词 D:11 PRIMARY 命中）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    refs = collect_action_intent_concepts([_seg(["帮我", "生成"])],
                                          backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci)
    ref_set = {r for r, _k in refs}
    assert action_refs[INTENT_COMMAND_MOOD] in ref_set, "帮我→COMMAND_MOOD D:11 命中"
    assert action_refs[ACTION_GENERATE] in ref_set, "生成→ACTION_GENERATE D:11 命中"
    assert len(refs) == 2, "两 distinct ACTION_* concept"


def test_collect_distinct_dedup(action_exp_env):
    """segments 多次 '生成' → collector 返 GENERATE ref 一次（distinct by action_ref·镜像 concept_targets set 去重）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    refs = collect_action_intent_concepts([_seg(["生成", "生成", "生成"])],
                                          backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci)
    assert len(refs) == 1, "同 ACTION_* concept 去重·只返一次"
    assert refs[0][0] == action_refs[ACTION_GENERATE]


def test_collect_unknown_token_skipped(action_exp_env):
    """segments 含未概念化词（'xyz'·concept_index.lookup 返 None）→ skip·不报错·只返已知 ACTION_*。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    refs = collect_action_intent_concepts([_seg(["xyz", "生成"])],
                                          backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci)
    assert len(refs) == 1, "xyz 未概念化 skip·只 生成 命中"
    assert refs[0][0] == action_refs[ACTION_GENERATE]


def test_collect_no_action_token_empty(action_exp_env):
    """segments 全非动作意图词（'猫 追 老鼠'·无 D:11 ACTION_* 边）→ collector 返空。"""
    b, sid, es, ns, ci = action_exp_env[:5]
    refs = collect_action_intent_concepts([_seg(["猫", "追", "老鼠"])],
                                          backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci)
    assert refs == [], "无动作意图词→空"


def test_collect_injected_alias(action_exp_env):
    """fixture 注入开放变体 '编写'→ACTION_GENERATE D:11 边（SOURCE_TEACHER·PRIMARY）→ collector 命中（教师晋升 alias 可学）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    # 注入 编写→ACTION_GENERATE D:11 边（模拟教师晋升·非 boot 种子）
    record_word_concept(ci, es, "编写", action_refs[ACTION_GENERATE],
                        space_id=sid, source=SOURCE_TEACHER)
    refs = collect_action_intent_concepts([_seg(["编写"])],
                                          backend=b, edge_store=es,
                                          space_id=sid, concept_index=ci)
    assert len(refs) == 1, "编写→ACTION_GENERATE D:11 PRIMARY 命中（教师晋升 alias）"
    assert refs[0][0] == action_refs[ACTION_GENERATE]


def test_collect_is_pure_read_no_mutation(action_exp_env):
    """collector 纯读（设计审 D）·调前后 experience_count / composes_attr 零新增。"""
    b, sid, es, ns, ci = action_exp_env[:5]
    ec_before = len(b.select("experience_count", where=None))
    attr_before = len(b.select("composes_attr", where=None))
    collect_action_intent_concepts([_seg(["帮我", "生成"])],
                                   backend=b, edge_store=es,
                                   space_id=sid, concept_index=ci)
    assert len(b.select("experience_count", where=None)) == ec_before, "collector 不写 experience_count"
    assert len(b.select("composes_attr", where=None)) == attr_before, "collector 不写 composes_attr"


# ============ _feed_action_experience helper（D3 + R1 + 桶隔离） ============

def test_feed_gate_off_no_write(action_exp_env):
    """gate OFF → helper 早返→experience_count 零新增（bit-identical·反 theater 干预）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = False
    try:
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["帮我", "生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND, reward=1, terminal=TERMINAL_REACHED_SINK)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    rows = b.select("experience_count", where=None)
    assert rows == [], "gate OFF → 零写（bit-identical）"


def test_feed_command_sink_reward_pos(action_exp_env):
    """gate ON + COMMAND + REACHED_SINK + reward=1 → ACTION_* e_sn=1 & e_tn=1（R1 成功臂·COMMAND 桶）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["帮我", "生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND, reward=1, terminal=TERMINAL_REACHED_SINK)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    cmd_code = _ctx_code(INTENT_COMMAND)
    # COMMAND_MOOD + GENERATE 都写（帮我+生成）
    for kind in (INTENT_COMMAND_MOOD, ACTION_GENERATE):
        r = read_experience_count(b, action_refs[kind], ctx_code=cmd_code)
        assert r is not None, f"{kind} concept 写了 experience_count"
        assert r == (0, 1, 1), f"{kind} reward>0→e_sn=1 & e_tn=1（R1 成功臂·base_freq=0）"


def test_feed_command_sink_reward_zero(action_exp_env):
    """gate ON + COMMAND + REACHED_SINK + reward=0（veto）→ e_tn=1 only·e_sn=0（R1 失败臂·率<1 有判别力·非排除闸）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND, reward=0, terminal=TERMINAL_REACHED_SINK)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    cmd_code = _ctx_code(INTENT_COMMAND)
    r = read_experience_count(b, action_refs[ACTION_GENERATE], ctx_code=cmd_code)
    assert r == (0, 0, 1), "reward==0 veto→e_sn=0 & e_tn=1（R1 失败臂·非排除·率<1）"


def test_feed_question_no_write(action_exp_env):
    """gate ON + QUESTION + REACHED_SINK → D3 type 闸失败→零写（命令态才回写·非 QUESTION）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["帮我", "生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_QUESTION, reward=1, terminal=TERMINAL_REACHED_SINK)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    rows = b.select("experience_count", where=None)
    assert rows == [], "QUESTION→D3 type 闸失败→零写"


def test_feed_dead_end_no_write(action_exp_env):
    """gate ON + COMMAND + DEAD_END → D3 sink 闸失败→零写（命令未达目标·非动作失败·不计）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["帮我", "生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND, reward=1, terminal=TERMINAL_DEAD_END)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    rows = b.select("experience_count", where=None)
    assert rows == [], "DEAD_END→D3 sink 闸失败→零写"


def test_feed_command_bucket_isolation(action_exp_env):
    """ctx_code 桶隔离（§17.2）：COMMAND 桶写·QUESTION 桶读→None（pack_ctx_code 第4维 intent_type 自动隔离）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND, reward=1, terminal=TERMINAL_REACHED_SINK)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    gen_ref = action_refs[ACTION_GENERATE]
    # COMMAND 桶有写
    assert read_experience_count(b, gen_ref, ctx_code=_ctx_code(INTENT_COMMAND)) == (0, 1, 1), \
        "COMMAND 桶写了"
    # QUESTION 桶空（隔离）
    assert read_experience_count(b, gen_ref, ctx_code=_ctx_code(INTENT_QUESTION)) is None, \
        "QUESTION 桶隔离→None（ctx_code 第4维 intent_type 分桶）"


def test_feed_accumulates_across_episodes(action_exp_env):
    """两次命令成功 episode→e_sn=2 & e_tn=2（累积·MUTABLE_MONOTONE·镜像 op_confidence 跨 episode 累积）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        for _ in range(2):
            _feed_action_experience(
                backend=b, edge_store=es, space_id=sid, concept_index=ci,
                segments=[_seg(["生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
                intent_type=INTENT_COMMAND, reward=1, terminal=TERMINAL_REACHED_SINK)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    r = read_experience_count(b, action_refs[ACTION_GENERATE], ctx_code=_ctx_code(INTENT_COMMAND))
    assert r == (0, 2, 2), "两次成功→e_sn=2 & e_tn=2（跨 episode 累积·R1 sn 单调）"


def test_feed_mixed_reward_rate_below_one(action_exp_env):
    """1 成功 + 1 veto → e_sn=1 & e_tn=2 → 率=1/2（判别力·反 β_arith·设计审 B 核心）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        # 1 成功
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND, reward=1, terminal=TERMINAL_REACHED_SINK)
        # 1 veto（reward==0）
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND, reward=0, terminal=TERMINAL_REACHED_SINK)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    r = read_experience_count(b, action_refs[ACTION_GENERATE], ctx_code=_ctx_code(INTENT_COMMAND))
    assert r == (0, 1, 2), "1 成功 + 1 veto → e_sn=1 & e_tn=2 → 率 1/2（判别力）"


# ============ gate 默认 OFF（守 CI 回归） ============

def test_action_experience_feed_gate_exists_and_bool():
    """gate ACTION_EXPERIENCE_FEED_MODE 已注册 + bool 类型（default OFF·_flag(False) 守 CI·生产 try/finally 暂不翻）。

    实际 OFF 行为由 test_feed_gate_off_no_write 验（零写）+ 全量回归零翻证（reload gates 模块会污染 suite·改静态检查）。"""
    assert hasattr(gates, "ACTION_EXPERIENCE_FEED_MODE"), "gate 已注册"
    assert isinstance(gates.ACTION_EXPERIENCE_FEED_MODE, bool), "gate 是 bool 类型"


# ============ D6 + STOP 合规（B-PR2 零新增编号·复用 experience_count + ATTR_OPERATION_INTENT=23） ============

def test_bpr2_no_new_attr_or_table():
    """STOP+D6 合规：B-PR2 复用 experience_count 表 + record_experience_outcome + ATTR_OPERATION_INTENT=23·零新增表/ATTR/MARK。
    B-PR2 是接线（hook + collector）非新机制·守 minimal extension（§14.6）。"""
    from pure_integer_ai.storage import composes_attr as ca
    from pure_integer_ai.storage import experience_count as ec
    assert ca.ATTR_OPERATION_INTENT == 23, "复用 ATTR_OPERATION_INTENT=23（B-PR1 建·非 B-PR2 新增）"
    # B-PR2 无新增表（experience_count 既有·B-PR2 只复用 record/read）
    assert not hasattr(ca, "ATTR_ACTION_EXPERIENCE"), "无 ATTR_ACTION_EXPERIENCE（B-PR2 走 experience_count 非 ATTR）"
    assert not hasattr(ec, "record_action_outcome"), "无 record_action_outcome（B-PR2 复用 record_experience_outcome）"


# ============ 单向依赖 + 纯整数 ============

def test_feed_action_experience_int_only(action_exp_env):
    """纯整数铁律：reward/ctx_code/terminal/intent_type 全 int（assert_int 守·record_experience_outcome :227 + pack_ctx_code :62）。"""
    b, sid, es, ns, ci, action_refs = action_exp_env
    saved = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        # 全 int 传参不报错（assert_int 过）
        _feed_action_experience(
            backend=b, edge_store=es, space_id=sid, concept_index=ci,
            segments=[_seg(["生成"])], domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND, reward=1, terminal=TERMINAL_REACHED_SINK)
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved
    # 验证写的行 ctx_code 是纯整数位打包
    rows = b.select("experience_count", where=None)
    assert all(isinstance(r["ctx_code"], int) for r in rows), "ctx_code 纯整数"
    assert all(isinstance(r["e_sn"], int) and isinstance(r["e_tn"], int) for r in rows), "e_sn/e_tn 纯整数"
