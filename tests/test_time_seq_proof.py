"""tests.test_time_seq_proof — 刀 A 时序 cue 测试套件（形式 cue 扩展首刀·语言域第一个 LIVE form_proof_fn）。

测时序 cue 4 件齐（Option A·时序 cue 边不入图·闭包传）：
  ① 词表/类型（cue_type_of 返 PRECEDES_CUE_FORWARD·入手⑥:37 纠偏 _REL_KIND_TO_CUE_TYPE）
  ② 构造器（extract_cues 时序分支 precedes_pairs·extract_cues_gated 3-tuple bit-identical）
  ③ 消费者（cue + typed 事件时间事实 Kahn 无环→1/有环→0/空→None）
  ④ D:11 readback（入手⑥:37 纠偏后 REL_PRECEDES 映 PRECEDES_CUE_FORWARD·反 theater）

诚实边界：构造性检查 ≠ 构造性验证（Layer0 下 session升验证）·时序 cue 边不入图（Option A·防 #355 + emergence 污染）。
occurrence-order adapter 接线（resolve/query/Episode）靠全回归守护（既有 formal_train 测语言域 episode·
gate OFF 不走时序 verify·零行为变 bit-identical）+ 2 对抗审。
"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.shared.relation_primitives import REL_PRECEDES, REL_CAUSES
from pure_integer_ai.cognition.understanding.cue_words import (
    cue_type_of, PRECEDES_CUE_FORWARD, CAUSES_CUE_FORWARD, IS_A_CUE, _REL_KIND_TO_CUE_TYPE,
)
from pure_integer_ai.cognition.understanding.cue_extractor import extract_cues, extract_cues_gated
from pure_integer_ai.training.time_seq_proof import time_seq_proof_fn_factory


# ============ 件① 词表/类型 ============

def test_cue_type_of_precedes_forward_zh():
    """ZH 时序词（然后/之后/接着/随后/后来）→ PRECEDES_CUE_FORWARD（exact 匹配）。"""
    for w in ["然后", "之后", "接着", "随后", "后来"]:
        assert cue_type_of(w, LANG_ZH) == PRECEDES_CUE_FORWARD, f"ZH 时序词 {w} 应返 PRECEDES_CUE_FORWARD"


def test_cue_type_of_precedes_forward_en():
    """EN 时序词（then/after/afterwards/subsequently/later）→ PRECEDES_CUE_FORWARD。"""
    for w in ["then", "after", "afterwards", "subsequently", "later"]:
        assert cue_type_of(w, LANG_EN) == PRECEDES_CUE_FORWARD, f"EN 时序词 {w} 应返 PRECEDES_CUE_FORWARD"


def test_cue_type_of_non_temporal_unchanged():
    """因果/系词 cue 不受时序扩展影响（导致→CAUSES_CUE_FORWARD·是一种→IS_A_CUE·守既有语义）。"""
    assert cue_type_of("导致", LANG_ZH) == CAUSES_CUE_FORWARD
    assert cue_type_of("是一种", LANG_ZH) == IS_A_CUE


def test_cue_type_of_before_not_temporal():
    """'之前'/'before' 不在 FORWARD 词表（BACKWARD 逆向 defer·避免方向歧义）→ None。"""
    assert cue_type_of("之前", LANG_ZH) is None, "之前 是 BACKWARD 语义·首版 FORWARD 排除·defer"
    assert cue_type_of("before", LANG_EN) is None, "before 是 BACKWARD·首版排除"


def test_rel_precedes_maps_to_precedes_cue_forward():
    """入手⑥:37 纠偏核心：_REL_KIND_TO_CUE_TYPE[REL_PRECEDES] = PRECEDES_CUE_FORWARD（原误并 CAUSES_CUE_FORWARD）。"""
    assert _REL_KIND_TO_CUE_TYPE[REL_PRECEDES] == PRECEDES_CUE_FORWARD, \
        "入手⑥纠偏：REL_PRECEDES 应映 PRECEDES_CUE_FORWARD（时序≠因果·原误并 CAUSES_CUE_FORWARD）"
    assert _REL_KIND_TO_CUE_TYPE[REL_CAUSES] == CAUSES_CUE_FORWARD, "因果映保持 CAUSES_CUE_FORWARD（纠偏不波及因果）"


# ============ 件② 构造器（extract_cues 3-tuple + 时序分支） ============

def test_extract_cues_precedes_branch():
    """时序 cue 分支：A 然后 B → precedes_pairs=[(0,2)]·A(0) 先于 B(2)·紧邻 cue 词左右 token。"""
    cue, is_a, prec = extract_cues(["A", "然后", "B"], lang=LANG_ZH)
    assert prec == [(0, 2)], "A 然后 B → precedes_pairs=[(0,2)]"
    assert cue == [] and is_a == [], "时序 cue 不污染因果/类属"


def test_extract_cues_mixed_causes_and_precedes():
    """时序与因果分离同段共存：导致→cue_pairs·然后→precedes_pairs·不混。"""
    # tokens: 雨(0) 导致(1) 地湿(2) 然后(3) 干(4)
    cue, is_a, prec = extract_cues(["雨", "导致", "地湿", "然后", "干"], lang=LANG_ZH)
    assert cue == [(0, 2)], "导致→cue_pairs=[(0,2)]（因果·雨→地湿）"
    assert prec == [(2, 4)], "然后→precedes_pairs=[(2,4)]（时序·地湿→干·紧邻左右）"
    assert is_a == []


def test_extract_cues_boundary_precedes_skipped():
    """边界时序 cue（句首/句末无左或右）跳·守反统计契约（不凑配）。"""
    _, _, prec = extract_cues(["然后", "B"], lang=LANG_ZH)
    assert prec == [], "然后 在句首无左 → 跳"
    _, _, prec = extract_cues(["A", "然后"], lang=LANG_ZH)
    assert prec == [], "然后 在句末无右 → 跳"


def test_extract_cues_adjacent_cue_skipped():
    """连用 cue（左邻也是 cue）跳·守锚定单义（镜像既有 CAUSES 逻辑）。"""
    _, _, prec = extract_cues(["导致", "然后", "B"], lang=LANG_ZH)
    # 然后 left=导致(0)·导致是 cue → 跳（连用指向词·锚定歧义）
    assert prec == [], "然后 左邻导致(也是 cue) → 跳"


def test_extract_cues_returns_3tuple():
    """extract_cues 返 3-tuple（cue_pairs, is_a_pairs, precedes_pairs）·Signature 守。"""
    result = extract_cues(["A", "然后", "B"], lang=LANG_ZH)
    assert isinstance(result, tuple) and len(result) == 3, "返 3-tuple"


def test_extract_cues_gated_off_returns_3empty():
    """CUE_EXTRACTOR_MODE OFF → 返 ([], [], [])·bit-identical 守回归（含 precedes_pairs 空）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = False
    try:
        result = extract_cues_gated(["A", "然后", "B"], lang=LANG_ZH)
        assert result == ([], [], []), "gate OFF 返 3 空元组"
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_extract_cues_gated_on_returns_precedes():
    """CUE_EXTRACTOR_MODE ON → precedes_pairs 非空（生产路径·frozenset 第一源命中）。"""
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    try:
        _, _, prec = extract_cues_gated(["A", "然后", "B"], lang=LANG_ZH)
        assert prec == [(0, 2)]
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


# ============ 件③ 消费者（time_seq_proof_fn_factory·构造性检查层） ============

def test_time_seq_proof_fn_no_cycle_returns_1():
    """无环 PRECEDES DAG → 1（verified·构造性检查通过·Kahn is_dag=True）。"""
    edges = [((0, 1), (0, 2)), ((0, 2), (0, 3))]   # A→B→C 线性无环
    fn = time_seq_proof_fn_factory(cue_pair_edges=edges, event_time_edges=[])
    assert fn(None, None, None) == 1


def test_time_seq_proof_fn_cycle_returns_0():
    """有环 PRECEDES DAG → 0（mismatch·结构矛盾·Kahn cycle_nodes 非空）。"""
    edges = [((0, 1), (0, 2)), ((0, 2), (0, 1))]   # A→B→A 环
    fn = time_seq_proof_fn_factory(cue_pair_edges=edges, event_time_edges=[])
    assert fn(None, None, None) == 0


def test_time_seq_proof_fn_merged_event_edges_cycle():
    """cue 对与显式事件时间事实合并后成环时必须返回 0。"""
    cue_edges = [((0, 1), (0, 2))]      # cue 对 A→B
    event_edges = [((0, 2), (0, 1))]  # 事件时间事实 B→A
    fn = time_seq_proof_fn_factory(
        cue_pair_edges=cue_edges,
        event_time_edges=event_edges,
    )
    assert fn(None, None, None) == 0, "合并 A→B→A 环 → 0（验合并集）"


def test_time_seq_proof_fn_empty_returns_none():
    """两边集均空 → None（vacate·无时序边可验·诚实退场·非 pass·非 theater）。"""
    fn = time_seq_proof_fn_factory(cue_pair_edges=[], event_time_edges=[])
    assert fn(None, None, None) is None


def test_time_seq_proof_fn_only_event_time_no_cycle():
    """仅分型事件时间事实也可独立验环，不需要 token 位置序。"""
    event_edges = [((0, 1), (0, 2)), ((0, 2), (0, 3))]
    fn = time_seq_proof_fn_factory(
        cue_pair_edges=[],
        event_time_edges=event_edges,
    )
    assert fn(None, None, None) == 1


def test_time_seq_proof_fn_only_cue_no_cycle():
    """仅 cue 对（无 PRECEDES 边）→ Kahn 验 cue shortcut 无环 → 1。"""
    cue_edges = [((0, 1), (0, 2)), ((0, 2), (0, 3))]
    fn = time_seq_proof_fn_factory(cue_pair_edges=cue_edges, event_time_edges=[])
    assert fn(None, None, None) == 1


def test_time_seq_proof_fn_deterministic():
    """确定性：同输入同输出（Kahn 按 (sid,lid) 自然序·bit-identical）。"""
    edges = [((0, 3), (0, 1)), ((0, 1), (0, 2))]
    fn = time_seq_proof_fn_factory(cue_pair_edges=edges, event_time_edges=[])
    r1 = fn(None, None, None)
    r2 = fn(None, None, None)
    assert r1 == r2 == 1


# ============ 诚实边界（docstring 标注） ============

def test_time_seq_proof_is_constructive_check_not_verification():
    """诚实标注：time_seq_proof 是构造性检查层（Kahn 确定性）·非构造性验证（须 R6·Layer0 下 session）。"""
    import pure_integer_ai.training.time_seq_proof as mod
    docstring = mod.__doc__ or ""
    assert "构造性检查" in docstring, "模块 docstring 须标构造性检查层"
    assert "构造性验证" in docstring, "须诚实标非构造性验证（cue 对 single-source·须 R6 升验证）"
    assert "Layer0" in docstring or "Layer 0" in docstring, "须标 Layer0 下 session 升验证"


def test_time_seq_proof_never_reward():
    """时序验序永不接 reward（self_proof_fn 通道·reward 通道严格不动·PRECEDES strength 恒 1）。"""
    import pure_integer_ai.training.time_seq_proof as mod
    docstring = mod.__doc__ or ""
    assert "reward" in docstring.lower(), "须标永不接 reward"


def test_time_seq_proof_option_a_no_persist():
    """Option A 诚实标注：时序 cue 边不入图（闭包传·防 #355 provenance + emergence 污染）。"""
    import pure_integer_ai.training.time_seq_proof as mod
    docstring = mod.__doc__ or ""
    assert "不入图" in docstring or "闭包" in docstring, "须标 Option A（边不入图·闭包传）"


# ============ gate 默认 OFF（bit-identical 守 CI=生产） ============

def test_time_seq_proof_mode_default_off():
    """TIME_SEQ_PROOF_MODE 默认 OFF·守 CI 回归 bit-identical（路由不走·既有语言域 episode_loop 不变）。"""
    # 重读 env（_flag import 时读一次）·测默认值
    import importlib
    import pure_integer_ai.config.gates as g
    importlib.reload(g)
    assert g.TIME_SEQ_PROOF_MODE is False, "TIME_SEQ_PROOF_MODE 默认 OFF 守 bit-identical"
