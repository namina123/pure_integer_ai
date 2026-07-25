"""tests/test_modification_direction.py — G2 修饰方向A（ 的-cue head/modifier 统计·source+read-time）。

验 modification_direction（cognition/understanding）+ ConceptGraph.head_pref_score + dispatch_slot combine。
设计：doc/重来_G2_修饰方向A_设计_2026-07-15.md。镜像 test_emergent_role（position_hist）+ test_factor_e_intraseg（observe）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.cognition.shared.types import (
    SpaceContext, InputPayload, Segment,
    STAGE_TRAINING, WEANING_PRE, MODALITY_LANGUAGE, LANG_ZH, DOMAIN_TEXT,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.observe import ObservePipeline
from pure_integer_ai.cognition.understanding.modification_direction import (
    register_modification_hist, observe_modification, head_preference,
    ROLE_HEAD, ROLE_MODIFIER, HEAD_PREF_CAP,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.config import gates


# ---- light fixture（MD1-MD4 单元：backend + register） ----

@pytest.fixture
def b():
    backend = DictBackend()
    bootstrap(backend)
    register_modification_hist(backend)
    yield backend
    backend.close()


# ============ MD1 observe_modification 累加（head/modifier count） ============

def test_observe_modification_accumulates(b):
    """一次 "X 的 Y" → head(Y)=ROLE_HEAD 1·modifier(X)=ROLE_MODIFIER 1。重复→累加。"""
    head = (1, 10)
    mod = (1, 11)
    observe_modification(b, head_ref=head, modifier_ref=mod)
    assert head_preference(b, head) == (1, 0)   # head 作 head 1 次
    assert head_preference(b, mod) == (0, 1)    # mod 作 modifier 1 次
    observe_modification(b, head_ref=head, modifier_ref=mod)   # 再一次
    assert head_preference(b, head) == (2, 0)
    assert head_preference(b, mod) == (0, 2)


def test_head_preference_unknown_zero(b):
    """未知 ref（无行）→ (0, 0)。"""
    assert head_preference(b, (1, 999)) == (0, 0)


def test_head_preference_mixed_roles(b):
    """一概念既作 head 又作 modifier → (head_count, mod_count) 分开记。"""
    ref = (1, 10)
    observe_modification(b, head_ref=ref, modifier_ref=(1, 11))   # ref 作 head
    observe_modification(b, head_ref=(1, 12), modifier_ref=ref)   # ref 作 modifier
    assert head_preference(b, ref) == (1, 1)


# ============ MD3 ConceptGraph.head_pref_score（gate + cap + 方向） ============

def test_head_pref_score_gate_off_zero(b):
    """gate OFF（默认）→ head_pref_score 返 0（不读表·bit-identical）。"""
    assert gates.MODIFIER_DIRECTION_MODE is False   # 默认 OFF
    g = ConceptGraph(b)
    head = (1, 10)
    observe_modification(b, head_ref=head, modifier_ref=(1, 11))
    assert g.head_pref_score(head) == 0   # gate OFF → 0（即使表有数据）


def test_head_pref_score_gate_on_head_bonus(b):
    """gate ON + head-dominant（head_count > mod_count）→ bonus = head_count - mod_count。"""
    saved = gates.MODIFIER_DIRECTION_MODE
    gates.MODIFIER_DIRECTION_MODE = True
    try:
        g = ConceptGraph(b)
        head = (1, 10)
        observe_modification(b, head_ref=head, modifier_ref=(1, 11))
        observe_modification(b, head_ref=head, modifier_ref=(1, 12))   # head 2 次·0 modifier
        assert g.head_pref_score(head) == 2   # 2 - 0 = 2
    finally:
        gates.MODIFIER_DIRECTION_MODE = saved


def test_head_pref_score_modifier_dominant_zero(b):
    """gate ON + modifier-dominant（mod_count ≥ head_count）→ 0（只奖 head·不罚 modifier）。"""
    saved = gates.MODIFIER_DIRECTION_MODE
    gates.MODIFIER_DIRECTION_MODE = True
    try:
        g = ConceptGraph(b)
        ref = (1, 10)
        observe_modification(b, head_ref=(1, 20), modifier_ref=ref)   # ref 作 modifier 1
        observe_modification(b, head_ref=(1, 21), modifier_ref=ref)   # ref 作 modifier 2
        observe_modification(b, head_ref=ref, modifier_ref=(1, 22))   # ref 作 head 1
        # head_count=1, mod_count=2 → diff=-1 ≤ 0 → 0
        assert g.head_pref_score(ref) == 0
    finally:
        gates.MODIFIER_DIRECTION_MODE = saved


def test_head_pref_score_cap(b):
    """gate ON + head_count - mod_count > HEAD_PREF_CAP → cap 在 HEAD_PREF_CAP（守 collide 主轴）。"""
    saved = gates.MODIFIER_DIRECTION_MODE
    gates.MODIFIER_DIRECTION_MODE = True
    try:
        g = ConceptGraph(b)
        head = (1, 10)
        for _ in range(HEAD_PREF_CAP + 50):   # 远超 cap
            observe_modification(b, head_ref=head, modifier_ref=(1, 99))
        assert g.head_pref_score(head) == HEAD_PREF_CAP   # capped at 9
    finally:
        gates.MODIFIER_DIRECTION_MODE = saved


# ============ MD5 2-token lookback 集成（observe "X 的 Y"） ============

@pytest.fixture
def ctx():
    """三空间 SpaceContext（镜像 test_factor_e_intraseg·observe 集成用）。"""
    backend = DictBackend()
    bootstrap(backend)
    reg = SpaceRegistry(backend)
    core = AbstractSpace.create(reg, "core")
    mem_read = MemorySpace.create(reg, "mem_read")
    mem_interact = MemorySpace.create(reg, "mem_interact")
    comp = CompanionSpace.create(reg, "comp1")
    c = SpaceContext(
        core=core, memory_read=mem_read, memory_interact=mem_interact,
        companion=comp, stage=STAGE_TRAINING, memory_active=False, weaning_phase=WEANING_PRE,
    )
    yield c
    backend.close()


def _seg(tokens):
    return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                   domain=DOMAIN_TEXT, tokens=tokens)


def test_lookback_records_head_modifier(ctx):
    """observe "红色 的 苹果"（3 token）→ modification_hist：苹果=HEAD·红色=MODIFIER（ 的-cue 2-token lookback）。

    trace：ti=0(红)·ti=1(的)·ti=2(苹果)。ti=2 时 prev1_tok=的·prev2_ref=红 → head=苹果·modifier=红。
    source write gate-independent（表 populated 即使 gate OFF）。
    """
    es = EdgeStore(ctx.core.backend)
    ci = ConceptIndex(ctx.core.backend, ctx.companion)
    pipe = ObservePipeline(ctx, concept_index=ci)
    raw = InputPayload(
        segments=[_seg(["红色", "的", "苹果"])],
        source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING,
        modality=MODALITY_LANGUAGE, lang=LANG_ZH, domain=DOMAIN_TEXT,
    )
    pipe.observe(raw)
    b = ctx.core.backend
    # 查概念 ref（ensure 写入·surface=红/苹果）
    head_ref = ci.ensure("苹果", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    mod_ref = ci.ensure("红色", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    assert head_preference(b, head_ref) == (1, 0)   # 苹果 作 head
    assert head_preference(b, mod_ref) == (0, 1)    # 红色 作 modifier


def test_lookback_no_de_for_plain_segment(ctx):
    """observe 无 的 的段 → modification_hist 全空（source write 仅 的-cue 触发·bit-identical inert）。"""
    es = EdgeStore(ctx.core.backend)
    ci = ConceptIndex(ctx.core.backend, ctx.companion)
    pipe = ObservePipeline(ctx, concept_index=ci)
    raw = InputPayload(
        segments=[_seg(["动物", "是", "生物"])],   # 无 的
        source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING,
        modality=MODALITY_LANGUAGE, lang=LANG_ZH, domain=DOMAIN_TEXT,
    )
    pipe.observe(raw)
    b = ctx.core.backend
    for ref in [ci.ensure("动物", space_id=ctx.core.space_id, tier=TIER_PRIMARY),
                ci.ensure("生物", space_id=ctx.core.space_id, tier=TIER_PRIMARY)]:
        assert head_preference(b, ref) == (0, 0)   # 无 的 → 无记录


# ============ MD6 bit-identical gate OFF（dispatch_slot combine 不变） ============

def test_modifier_direction_gate_default_off():
    """MODIFIER_DIRECTION_MODE 默认 OFF → dispatch_slot combine 逐字现状（bit-identical 守）。"""
    assert gates.MODIFIER_DIRECTION_MODE is False


def test_head_pref_cap_value():
    """HEAD_PREF_CAP=9（modest 亚轴 tiebreak·同 PR_SLOT_BONUS_CAP 量级·守 _cap_sp 999 联合 cap）。"""
    assert HEAD_PREF_CAP == 9
    assert ROLE_HEAD == 2
    assert ROLE_MODIFIER == 1
