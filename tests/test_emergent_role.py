"""Stage 3 扩展：emergent_role 主导度闸测试（缺口#1·去 SVO 写死唯一合法替代）。

doc/重来·主线重审与重画.md:580 防塌C6 落盘：
  ① position_hist 专用表 ② argmax 主导位置 ③ 主导度闸 dominance≥MIN_DOMINANCE(500)
  ④ 主导不足→混合桶 ⑤ 冷启动 SUBJECT 兜底 ⑥ 防碎片垮塌（无闸 245 碎片 vs SVO 4 桶）

覆盖契约（legacy 行为契约参考·非搬代码）：
  - 冷启动兜底（position_hist 空 → ROLE_SUBJECT）
  - 主导度≥闸 → 位置桶（dominant_pos offset）
  - 主导度<闸 → 混合桶（不污染主导桶）
  - argmax tiebreak 最小位置（确定性）
  - 碎片防垮塌（次要位置噪声不抬主导度·主导位置收拢）
  - role 整数高段避撞（不与 edge_type/KIND 撞值）
  - observe 集成：parsed.role_seq 空时 emergent_role 兜底填（解致命6·generate 产空）
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.cognition.shared.types import SpaceContext, STAGE_TRAINING, WEANING_PRE
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.emergent_role import (
    PositionHistogramState, register_position_hist, observe_position, position_histogram,
    dominant_position, dominance, emergent_role, role_seq_for_tokens,
)
from pure_integer_ai.cognition.understanding.role_scheme import (
    ROLE_SUBJECT, ROLE_BUCKET_BASE, MIXED_BUCKET_OFFSET, MIN_DOMINANCE,
    position_bucket, mixed_bucket, is_role_value,
)


@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_position_hist(b)
    return b


# ============ role scheme 高段避撞 ============

def test_role_scheme_high_namespace_no_collision():
    """role 整数高段·不与 edge_type(1-127)/KIND(1-104)/opcode(≥2^60 低段) 撞值。"""
    assert ROLE_SUBJECT >= (1 << 50)
    assert ROLE_BUCKET_BASE > ROLE_SUBJECT
    assert MIXED_BUCKET_OFFSET > ROLE_BUCKET_BASE
    # SUBJECT / 位置桶 / 混合桶 互不重叠
    assert ROLE_SUBJECT < ROLE_BUCKET_BASE
    assert not (ROLE_BUCKET_BASE <= ROLE_SUBJECT < ROLE_BUCKET_BASE + (1 << 16))
    assert MIXED_BUCKET_OFFSET >= ROLE_BUCKET_BASE + (1 << 16)


def test_is_role_value_guards():
    assert is_role_value(ROLE_SUBJECT)
    assert is_role_value(position_bucket(0))
    assert is_role_value(position_bucket(3))
    assert is_role_value(mixed_bucket())
    assert not is_role_value(0)
    assert not is_role_value(101)          # legacy KIND_PLACEMENT 撞值·须拒
    assert not is_role_value(300)          # legacy ROLE_BUCKET_BASE 撞值·须拒


def test_position_bucket_and_mixed_disjoint():
    """位置桶 offset 区间与混合桶 offset 不撞（主导桶不污染混合桶·doc ④）。"""
    pb = position_bucket(0)
    mb = mixed_bucket()
    assert pb != mb
    assert is_role_value(pb) and is_role_value(mb)


# ============ 主导度闸核心 ============

def test_cold_start_subject_fallback(backend):
    """⑤ 冷启动兜底：position_hist 空 → ROLE_SUBJECT（解致命6·generate 产空）。"""
    ref = (1, 100)
    assert emergent_role(backend, ref) == ROLE_SUBJECT


def test_dominance_above_gate_position_bucket(backend):
    """③ 主导度≥闸 → 位置桶（dominant_pos offset·doc ③）。
    概念在位 0 出现 7 次·位 1 出现 3 次 → dominance=700≥500 → 位 0 桶。"""
    ref = (1, 1)
    for _ in range(7):
        observe_position(backend, ref, 0)
    for _ in range(3):
        observe_position(backend, ref, 1)
    assert emergent_role(backend, ref) == position_bucket(0)


def test_dominance_below_gate_mixed_bucket(backend):
    """④ 主导度<闸 → 混合桶（不污染主导桶·doc ④）。
    概念在位 0 出现 1 次·位 1 出现 2 次 → max=2,total=3,dominance=666... 重算：
    位 0×1 + 位 1×2 → mx=2,total=3,dominance=666≥500 → 位 1 桶。改均匀分布触发混合桶。"""
    ref = (1, 2)
    # 均匀分布：位 0×2 + 位 1×2 → mx=2,total=4,dominance=500 ≥500 仍位 0 桶（tiebreak 最小位）
    # 需严格 <500：位 0×1 + 位 1×1 + 位 2×1 → mx=1,total=3,dominance=333<500 → 混合桶
    for p in (0, 1, 2):
        observe_position(backend, ref, p)
    assert emergent_role(backend, ref) == mixed_bucket()


def test_argmax_tiebreak_smallest_position(backend):
    """② argmax tiebreak 最小位置码（确定性·平局取最小位）。"""
    ref = (1, 3)
    # 位 0×2 + 位 1×2 → 平局·tiebreak 位 0·dominance=500≥500 → 位 0 桶
    for _ in range(2):
        observe_position(backend, ref, 0)
    for _ in range(2):
        observe_position(backend, ref, 1)
    assert emergent_role(backend, ref) == position_bucket(0)


def test_dominant_position_deterministic():
    """dominant_position 纯函数确定性（平局最小位·升序首 max）。"""
    hist = [(0, 2), (1, 5), (2, 5)]   # 位 1/2 平局·取位 1
    assert dominant_position(hist) == (1, 5)
    assert dominant_position([]) is None
    hist2 = [(3, 1), (0, 1), (1, 1)]  # 已升序·平局取位 0
    # position_histogram 保升序·此处直接验 max 逻辑
    pos, cnt = max(hist2, key=lambda pc: pc[1])
    assert pos == 3   # max 取首见（hist2 未升序·首见=位 3）·position_histogram 才保升序


def test_dominance_pure_int():
    assert dominance(7, 10) == 700
    assert dominance(1, 3) == 333
    assert dominance(0, 0) == 0
    assert dominance(5, 0) == 0


# ============ 碎片防垮塌（doc ⑥） ============

def test_fragmentation_collapse_prevention(backend):
    """⑥ 防碎片垮塌：次要位置噪声不抬主导度·主导位置收拢（无闸 245 碎片 vs SVO 4 桶）。
    概念在位 0 主导(8次)+次要噪声散布(位1-4各1次) → dominance=800≥500 → 仍位 0 桶（收拢非碎片）。"""
    ref = (1, 4)
    for _ in range(8):
        observe_position(backend, ref, 0)
    for p in (1, 2, 3, 4):
        observe_position(backend, ref, p)
    assert emergent_role(backend, ref) == position_bucket(0)


def test_mixed_bucket_does_not_pollute_dominant(backend):
    """④ 混合桶概念不污染主导桶统计：弱主导概念落混合桶·其位置计数仍累加但不影响
    其他概念的 dominant_pos（position_hist 按概念隔离·per-concept 直方图）。"""
    strong = (1, 10)   # 强主导位 0
    weak = (1, 11)     # 弱主导（混合桶）
    for _ in range(9):
        observe_position(backend, strong, 0)
    for p in (0, 1, 2):
        observe_position(backend, weak, p)
    assert emergent_role(backend, strong) == position_bucket(0)   # 不被 weak 污染
    assert emergent_role(backend, weak) == mixed_bucket()


# ============ observe_position 累加 ============

def test_observe_position_accumulates(backend):
    """① position_hist 累加：同位多次 observe → count 单调增（MUTABLE_MONOTONE）。"""
    ref = (1, 20)
    observe_position(backend, ref, 0)
    observe_position(backend, ref, 0)
    observe_position(backend, ref, 0)
    hist = position_histogram(backend, ref)
    assert hist == [(0, 3)]


def test_position_histogram_ordered(backend):
    """position_histogram 按 position 升序（确定性·bit-identical）。"""
    ref = (1, 21)
    observe_position(backend, ref, 2)
    observe_position(backend, ref, 0)
    observe_position(backend, ref, 1)
    hist = position_histogram(backend, ref)
    assert [p for p, _ in hist] == [0, 1, 2]


def test_role_seq_for_tokens_cold_start_all_subject(backend):
    """role_seq_for_tokens 冷启动全 SUBJECT（position_hist 空·退化态·解致命6）。"""
    refs = [(1, 30), (1, 31), (1, 32)]
    rseq = role_seq_for_tokens(backend, refs)
    assert rseq == [ROLE_SUBJECT, ROLE_SUBJECT, ROLE_SUBJECT]


def test_role_seq_for_tokens_mixed(backend):
    """role_seq 逐 token 独立算：强主导→位置桶·冷启动→SUBJECT·混合。"""
    strong = (1, 40)
    cold = (1, 41)
    for _ in range(8):
        observe_position(backend, strong, 1)
    refs = [strong, cold]
    rseq = role_seq_for_tokens(backend, refs)
    assert rseq == [position_bucket(1), ROLE_SUBJECT]


def test_position_histogram_state_reuses_verified_bucket(backend, monkeypatch):
    """实例状态首次整桶读取一次，后续角色计算和同 writer 累加不重复 select。"""
    ref = (1, 50)
    observe_position(backend, ref, 0)
    original_select = backend.select
    position_selects = 0

    def counted_select(table, *args, **kwargs):
        """只统计位置桶读取，同时保持 backend 原始选择行为。"""
        nonlocal position_selects
        if table == "position_hist":
            position_selects += 1
        return original_select(table, *args, **kwargs)

    monkeypatch.setattr(backend, "select", counted_select)
    state = PositionHistogramState(backend)
    assert state.roles_for_tokens([ref, ref]) == [position_bucket(0), position_bucket(0)]
    state.observe(ref, 0)
    state.observe(ref, 1)
    assert state.histogram(ref) == [(0, 2), (1, 1)]
    assert position_selects == 1


def test_position_histogram_state_preserves_segment_read_before_write():
    """缓存路径与无状态路径逐段角色及最终 backend 快照完全一致。"""
    direct = DictBackend()
    cached = DictBackend()
    for current in (direct, cached):
        bootstrap(current)
        register_position_hist(current)
    state = PositionHistogramState(cached)
    segments = [
        [(1, 60), (1, 60), (1, 61)],
        [(1, 60), (1, 61), (1, 60)],
        [(1, 61), (1, 60)],
    ]
    direct_roles = []
    cached_roles = []
    for refs in segments:
        direct_roles.append(role_seq_for_tokens(direct, refs))
        cached_roles.append(state.roles_for_tokens(refs))
        for position, ref in enumerate(refs):
            observe_position(direct, ref, position)
            state.observe(ref, position)
    assert cached_roles == direct_roles
    assert cached.snapshot() == direct.snapshot()


def test_position_histogram_state_requires_invalidation_after_external_write(backend):
    """绕过实例的外部写不会被猜测；显式失效后重新读取权威桶。"""
    ref = (1, 70)
    state = PositionHistogramState(backend)
    assert state.role(ref) == ROLE_SUBJECT
    observe_position(backend, ref, 2)
    assert state.role(ref) == ROLE_SUBJECT
    state.invalidate(ref)
    assert state.role(ref) == position_bucket(2)
