"""B-PR3 gate③ _intent_override 接通测试（doc §18·镜像 test_negation_d11_readback / test_action_d11_readback 反 theater 范式）。

B-PR3 = gate③ `_intent_override` 接通（word_terminated 概念阻断三 gate 之③·首版返 0 未活）+ 断桥粗粒度 meta。
命令态（intent.type==INTENT_COMMAND）动作词（D:11 PRIMARY 边到 ACTION_*/COMMAND_MOOD）→ 返 1（不终止·留 path→
dag_path 导向动作拓扑·§13.3）。否则返 0（终止如常）。**B-PR1 ATTR_OPERATION_INTENT=23 推翻"defer S10"**（§18.1 决断1）。

覆盖（直调 _intent_override 单测·gate③ 只在 gate① freq 通过后触及·fixture eff_freq≪1000→生产 dormant·须单测证真机制）：
  - gate OFF bit-identical（零翻）/ COMMAND+D:11→override / 无 D:11→0 / SHADOW 隔离→0
  - COMMAND_MOOD 命令词→override / **QUESTION→0（intent 闸·§18.1 决断1 推翻设计审"不查 intent"）**
  - OP_*/裸概念交叉污染→0（lookup_word_action ATTR 过滤）/ None 退化→0
  - CHANNEL meta 映射正确（断桥粗粒度·inert）+ gate 注册 + 纯读

**反 theater 牙**：SHADOW 隔离（未验证假设不 override）+ 交叉污染（非 ACTION_* target 不激活）+ bit-identical（gate OFF 零变）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, TIER_SHADOW, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.experience_count import (
    register_experience_count, record_base_freq, DEFAULT_CTX_CODE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    IntentType, INTENT_COMMAND, INTENT_QUESTION, LANG_ZH,
)
from pure_integer_ai.cognition.shared.action_primitives import (
    INTENT_COMMAND_MOOD, ACTION_GENERATE, ACTION_COMPUTE, ACTION_ANALYZE, ACTION_SOLVE,
    ensure_action_primitives, action_channel,
    CHANNEL_NONE, CHANNEL_VM, CHANNEL_SERIALIZER, CHANNEL_JUDGE, _ACTION_CHANNEL_MAP,
)
from pure_integer_ai.cognition.understanding.word_concept_signal import bootstrap_action_signals
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.dag_path import _intent_override, word_terminated
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def action_override_env():
    """B-PR3 gate③ 单测环境（dict backend·core space·composes_attr 注册·boot 种 action D:11 PRIMARY 边）。

    镜像 test_action_experience_feed.action_exp_env·去 register_experience_count（gate③ 不读率·§18.1 决断2）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)        # ATTR_OPERATION_INTENT=23 标记表（lookup_word_action readback 用）
    register_experience_count(b)     # word_terminated read_effective_freq 用（integration 测·单元测不读）
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


def _intent(type_: int) -> IntentType:
    """构造 IntentType（给定 type·sink/标志默认·_intent_override 只读 .type）。"""
    return IntentType(type=type_)


# ============ _intent_override（D:11 per-word + intent 闸） ============

def test_override_gate_off_returns_zero(action_override_env):
    """gate OFF → _intent_override 首行早返 0（bit-identical·反 theater 干预·1881 零回归守）。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    word_ref = ci.lookup("计算", sid)   # 计算→ACTION_COMPUTE D:11 PRIMARY
    assert word_ref is not None, "计算 已 boot 种 D:11 边"
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = False
    try:
        r = _intent_override(word_ref, _intent(INTENT_COMMAND), None,
                             backend=b, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r == 0, "gate OFF → 返 0（bit-identical）"


def test_override_command_compute_returns_one(action_override_env):
    """gate ON + COMMAND + 词 D:11 PRIMARY 到 ACTION_COMPUTE → 返 1（不终止·导向动作拓扑·§13.3）。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    word_ref = ci.lookup("计算", sid)
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        r = _intent_override(word_ref, _intent(INTENT_COMMAND), None,
                             backend=b, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r == 1, "COMMAND + 计算 D:11 PRIMARY → 返 1（动作词留 path）"


def test_override_no_d11_edge_returns_zero(action_override_env):
    """gate ON + COMMAND + 无 D:11 动作边的词（ci.ensure 新建中性词）→ 返 0（终止如常·非动作词）。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    neutral_ref = ci.ensure("中性词", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        r = _intent_override(neutral_ref, _intent(INTENT_COMMAND), None,
                             backend=b, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r == 0, "无 D:11 动作边→lookup_word_action 返空→返 0"


def test_override_shadow_d11_isolated_returns_zero(action_override_env):
    """gate ON + COMMAND + 仅 SHADOW D:11 边（无 PRIMARY）→ tier_filter=PRIMARY 过滤→返 0（反 theater：未验证 SHADOW 不 override）。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    # 注入一个仅 SHADOW 的 D:11 边（record_word_concept 硬编码 PRIMARY·故 edge_store.add 直注 SHADOW）
    shadow_word = ci.ensure("编写shadow", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    gen_ref = action_refs[ACTION_GENERATE]
    es.add(space_id_from=shadow_word[0], local_id_from=shadow_word[1],
           space_id_to=gen_ref[0], local_id_to=gen_ref[1],
           edge_type=EDGE_RELATION_SIGNAL, strength=1, source=SOURCE_TEACHER,
           tier=TIER_SHADOW, epistemic_origin=EPI_STRUCTURED)
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        r = _intent_override(shadow_word, _intent(INTENT_COMMAND), None,
                             backend=b, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r == 0, "仅 SHADOW D:11→tier_filter PRIMARY 过滤→返 0（SHADOW=未验证假设不 override）"


def test_override_command_mood_returns_one(action_override_env):
    """gate ON + COMMAND + 词 D:11 到 INTENT_COMMAND_MOOD（帮我）→ 返 1（命令词保留·命令 mood 亦 override）。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    word_ref = ci.lookup("帮我", sid)   # 帮我→INTENT_COMMAND_MOOD D:11 PRIMARY
    assert word_ref is not None
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        r = _intent_override(word_ref, _intent(INTENT_COMMAND), None,
                             backend=b, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r == 1, "COMMAND + 帮我→COMMAND_MOOD D:11 PRIMARY → 返 1（命令词留 path）"


def test_override_question_returns_zero(action_override_env):
    """**intent 闸牙**（§18.1 决断1·推翻设计审"不查 intent"）：gate ON + QUESTION + 动作词 → intent≠COMMAND→返 0。

    QUESTION 不路由动作执行·gate① freq 终止合理·gate③ 不 override。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    word_ref = ci.lookup("计算", sid)   # 计算→ACTION_COMPUTE D:11 PRIMARY（动作词）
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        r = _intent_override(word_ref, _intent(INTENT_QUESTION), None,
                             backend=b, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r == 0, "QUESTION + 动作词→intent 闸返 0（仅 COMMAND override·§18.1 决断1）"


def test_override_cross_contamination_returns_zero(action_override_env):
    """gate ON + COMMAND + 词 D:11 到裸概念（无 ATTR_OPERATION_INTENT）→ lookup_word_action ATTR 过滤→返 0（不误激活）。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    # 建一个词→裸概念（无 ATTR_OPERATION_INTENT）的 D:11 PRIMARY 边（模拟 OP_*/REL_* target 或脏边）
    xword = ci.ensure("叉词", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    xtarget = ci.ensure("裸目标", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    es.add(space_id_from=xword[0], local_id_from=xword[1],
           space_id_to=xtarget[0], local_id_to=xtarget[1],
           edge_type=EDGE_RELATION_SIGNAL, strength=1, source=SOURCE_TEACHER,
           tier=TIER_PRIMARY, epistemic_origin=EPI_STRUCTURED)
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        r = _intent_override(xword, _intent(INTENT_COMMAND), None,
                             backend=b, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r == 0, "D:11 target 无 ATTR_OPERATION_INTENT→lookup_word_action 过滤返空→返 0（无交叉污染）"


def test_override_none_backend_edge_store_returns_zero(action_override_env):
    """gate ON + COMMAND + D:11 动作词 但 backend=None / edge_store=None → 退化返 0（bare fixture 安全·不 crash）。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    word_ref = ci.lookup("计算", sid)
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        r1 = _intent_override(word_ref, _intent(INTENT_COMMAND), None,
                              backend=None, edge_store=es)
        r2 = _intent_override(word_ref, _intent(INTENT_COMMAND), None,
                              backend=b, edge_store=None)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r1 == 0 and r2 == 0, "backend/edge_store None → 退化返 0（caller 未穿·不 crash）"


def test_override_is_pure_read(action_override_env):
    """_intent_override 纯读（lookup_word_action 只查不写）·调前后边数/ATTR 零新增。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    word_ref = ci.lookup("计算", sid)
    edges_before = len(b.select("edge", where=None))
    attrs_before = len(b.select("composes_attr", where=None))
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        _intent_override(word_ref, _intent(INTENT_COMMAND), None,
                         backend=b, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert len(b.select("edge", where=None)) == edges_before, "_intent_override 不写边"
    assert len(b.select("composes_attr", where=None)) == attrs_before, "_intent_override 不写 ATTR"


def test_word_terminated_integration_gate3_overrides(action_override_env):
    """**反 theater 主锚**：word_terminated 全 pipeline（gate① freq 过→gate③ 真 _intent_override→D:11 命中→不终止）。

    证 gate③ 在真 word_terminated pipeline 真活（非死码·非只单测）：动作词 计算 base_freq=10≥theta=5 →
    gate① freq 过 → gate② ctx 0 → gate③ 真 lookup_word_action D:11 PRIMARY 命中 → 返 1 → word_terminated 返 False（不终止·留 path）。
    对照：gate OFF → word_terminated 返 True（gate① freq 离群终止·无 override·bit-identical）。"""
    b, sid, es, ns, ci, action_refs = action_override_env
    word_ref = ci.lookup("计算", sid)
    record_base_freq(b, ref=word_ref, base_freq=10)   # eff_freq=10 ≥ theta_freq=5·gate① 过
    wm = WorkMemory()
    cmd_intent = _intent(INTENT_COMMAND)
    saved = gates.ACTION_INTENT_OVERRIDE_MODE
    # gate ON → gate③ 真 override → 不终止（False）
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        r_on = word_terminated(word_ref, wm, b, intent=cmd_intent,
                               theta_freq=5, ctx_code=DEFAULT_CTX_CODE, edge_store=es)
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved
    assert r_on is False, "gate ON + COMMAND + 计算(D:11) base_freq≥theta → gate③ override → 不终止（False·留 path）"
    # 对照：gate OFF → 终止（True·base_freq≥theta·gate① freq 离群·无 override·bit-identical）
    r_off = word_terminated(word_ref, wm, b, intent=cmd_intent,
                            theta_freq=5, ctx_code=DEFAULT_CTX_CODE, edge_store=es)
    assert r_off is True, "gate OFF → gate① freq 离群终止（True）·bit-identical 守"


# ============ CHANNEL meta（断桥粗粒度·inert·doc §18.1 决断3） ============

def test_channel_meta_mapping():
    """断桥粗粒度 meta 映射正确（纯 dict 断言·inert·无运行时消费者·Phase2 dispatch 接入）。"""
    assert _ACTION_CHANNEL_MAP[ACTION_COMPUTE] == CHANNEL_VM, "COMPUTE→VM"
    assert _ACTION_CHANNEL_MAP[ACTION_GENERATE] == CHANNEL_SERIALIZER, "GENERATE→序化器"
    assert _ACTION_CHANNEL_MAP[ACTION_ANALYZE] == CHANNEL_JUDGE, "ANALYZE→judge"
    assert _ACTION_CHANNEL_MAP[ACTION_SOLVE] == CHANNEL_VM, "SOLVE→VM（暂定·Phase2 修订·§18.1 决断3）"
    assert _ACTION_CHANNEL_MAP[INTENT_COMMAND_MOOD] == CHANNEL_NONE, "COMMAND_MOOD→无通道"
    # action_channel 查询函数（未知 kind 安全默认 CHANNEL_NONE）
    assert action_channel(ACTION_COMPUTE) == CHANNEL_VM
    assert action_channel(99) == CHANNEL_NONE, "未知 kind→CHANNEL_NONE（安全默认）"


# ============ gate 注册 + D6/STOP 合规 ============

def test_action_intent_override_gate_exists_and_bool():
    """gate ACTION_INTENT_OVERRIDE_MODE 已注册 + bool 类型（default OFF·_flag(False) 守 CI·生产 try/finally 暂不翻）。

    实际 OFF 行为由 test_override_gate_off_returns_zero 验（返 0）+ 全量回归零翻证。"""
    assert hasattr(gates, "ACTION_INTENT_OVERRIDE_MODE"), "gate 已注册"
    assert isinstance(gates.ACTION_INTENT_OVERRIDE_MODE, bool), "gate 是 bool 类型"


def test_bpr3_no_new_attr_or_table():
    """STOP+D6 合规：B-PR3 复用 ATTR_OPERATION_INTENT=23（B-PR1 建）+ 既有 D:11/lookup_word_action·零新增表/ATTR。
    CHANNEL_* enum + _ACTION_CHANNEL_MAP 是 meta 常量（非 ATTR/表/结构 kind）·守 minimal extension（§18.1 决断3）。"""
    from pure_integer_ai.storage import composes_attr as ca
    assert ca.ATTR_OPERATION_INTENT == 23, "复用 ATTR_OPERATION_INTENT=23（B-PR1·非 B-PR3 新增）"
    # B-PR3 无新增 ATTR（gate③ 读既有 ATTR_OPERATION_INTENT）
    assert not hasattr(ca, "ATTR_ACTION_CHANNEL"), "无 ATTR_ACTION_CHANNEL（CHANNEL 是 enum meta 非 ATTR）"
