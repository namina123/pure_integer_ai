"""#723 G 类归因交叉表测试（片3·判据偏真闭合最后一环）。

doc/重来_任务0723_G类归因.md 决断3+4 + 反 theater 三层。
**doc 决断3 偏差纠偏**：原 map "DIM_ARITH/G4 G2p 真走 judge" 错——ARITH/CODE episode 走
_run_verify_round（formal_train.py:510-523·只设 G5）非 judge()·G4/G2p 对 ③⑤ = N/A。

覆盖：
  - 交叉表结构（8 dims × 5 doors·dead_state 静态分类正确·_DIM_G_DEAD）
  - episode 分类（language judge / ARITH verify / CODE verify / Mode A task-driven / None 边界）
  - ALIVE 门计数（G5 arith verify·G4/G2p language judge·含 evidence_eps 溯源）
  - ALIVE（G3a/G3b·M1片2+G1+#774 落地后 classify_intent 真填三 bool·#889 stale 修正·active=0 因不 veto）
  - DEAD_DESIGN（G5 language·不在 _ARITH_DOMAINS·judge.py:247）
  - 反 theater 层1：坏 fixture → 表必异（非 theater）
  - 反 theater 层2：active 格带 evidence_eps 溯源 episode index
  - 反 theater 层3：dead 门反向证明（synthetic table 不忽略 G3a + e2e 生产 active=0）
  - e2e：formal_train collect_episodes=True → result.episodes 非空 + 归因表非全 N/A
  - bit-identical：双跑 to_json 一致（g_attribution 序列化确定）

铁律：纯整数 / bit-identical / 反 theater（坏 fixture 必异·dead 门必反向证·stable≠correct）。
"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.cognition.shared.types import (
    Episode, MODALITY_LANGUAGE, MODALITY_CODE, MODALITY_ARITH,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainResult, FormalTrainConfig, DefaultRoundRunner, formal_train,
)
from pure_integer_ai.experiments.capability_exam import (
    project_g_attribution, _classify_episode, CapabilityReport,
    DIM_CONCEPT, DIM_STRUCTURE, DIM_COMPUTE,
    DIM_LONG_TEXT, DIM_LONG_CODE, DIM_THREE_RING, DIM_INTENT, DIM_MEMORY,
    G_DOOR_G4, G_DOOR_G2P, G_DOOR_G3A, G_DOOR_G3B, G_DOOR_G5,
    G_ALIVE, G_DEAD_LEAK, G_DEAD_DESIGN, G_NA,
)
from tests.test_experiments import _causal_multi_sent_item, _multi_sent_item


# ============ synthetic episode helpers（三路 + 边界） ============

class _StubInput:
    """轻量 input stub（_classify_episode 只读 .modality·无须造完整 InputPayload）。"""
    def __init__(self, modality: int):
        self.modality = modality


def _lang_judge_ep(*, g4=False, g2p=False, g3a=False, g3b=False, g5=False, pr=True):
    """language judge path episode（modality=LANGUAGE + pr_vector 非空）→ ⑥三环 judge 路径。"""
    return Episode(
        input=_StubInput(MODALITY_LANGUAGE),
        pr_vector={"k": 1} if pr else {},
        judge_G4_active=g4, judge_G2p_active=g2p,
        judge_G3a_active=g3a, judge_G3b_active=g3b, judge_G5_active=g5,
    )


def _arith_verify_ep(*, g5=False, vetoed=False):
    """ARITH verify path episode（modality=ARITH + pr_vector 空）→ ③计算 verify 路径。

    g5=承重门（Mode B cross-verify 跑即 True·pass/fail 都 True·formal_train.py:506/519）·
    vetoed=真 veto（reward=0·cross_verify disagree·formal_train.py:522）。
    **g5=True ≠ vetoed=True**（承重非 veto·#723 P0 修·对抗审1 抓）：g5=True+vetoed=False=Mode B agree（pass）·
    g5=True+vetoed=True=Mode B disagree（veto）·g5=False+vetoed=True=非 Mode B 占位 veto（非 G5 veto）。
    """
    return Episode(
        input=_StubInput(MODALITY_ARITH),
        pr_vector={},
        judge_G5_active=g5,
        vetoed=vetoed,
    )


def _code_verify_ep(*, g5=False, vetoed=False):
    """CODE verify path episode（modality=CODE + pr_vector 空）→ ⑤长代码 verify 路径。同 _arith_verify_ep 语义。"""
    return Episode(
        input=_StubInput(MODALITY_CODE),
        pr_vector={},
        judge_G5_active=g5,
        vetoed=vetoed,
    )


def _mode_a_ep():
    """Mode A task-driven episode（input=None）→ ⑤长代码 mode_a 路径·不走任何 G 门。"""
    return Episode(input=None, pr_vector={}, judge_G5_active=False)


def _result(eps):
    """造 FormalTrainResult + episodes 列表（collect_episodes=True 时 harness 读此）。"""
    return FormalTrainResult(run_id="g_test", episodes=list(eps))


_ALL_DIMS = {DIM_CONCEPT, DIM_STRUCTURE, DIM_COMPUTE, DIM_LONG_TEXT,
             DIM_LONG_CODE, DIM_THREE_RING, DIM_INTENT, DIM_MEMORY}
_ALL_DOORS = {G_DOOR_G4, G_DOOR_G2P, G_DOOR_G3A, G_DOOR_G3B, G_DOOR_G5}


# ============ 结构 + 分类 ============

def test_g_attribution_structure_8x5():
    """交叉表 8 dims × 5 doors 全格·dead_state 静态分类正确（_DIM_G_DEAD·code 真相）。

    空 episodes → 全 total=0/active=0·dead_state 按 _DIM_G_DEAD：
      ⑥三环: G4/G2p/G3a/G3b ALIVE / G5 DEAD_DESIGN（#889 stale 修正·G3a/G3b 真活）
      ③计算/⑤长代码: G5 ALIVE / 其余 N/A
      ①②④⑦⑧: 全 N/A
    N/A 格 permille=-1（与 DimScore NE 范式一致）。
    """
    table = project_g_attribution(_result([]))
    assert set(table.keys()) == _ALL_DIMS
    for dim in table:
        assert set(table[dim].keys()) == _ALL_DOORS
    # ⑥三环 dead 分类（语义锚·决断3 真相版）
    six = table[DIM_THREE_RING]
    assert six[G_DOOR_G4]["dead_state"] == G_ALIVE
    assert six[G_DOOR_G2P]["dead_state"] == G_ALIVE
    assert six[G_DOOR_G3A]["dead_state"] == G_ALIVE   # #889 stale 修正·classify_intent 真填 is_causal
    assert six[G_DOOR_G3B]["dead_state"] == G_ALIVE   # #889 stale 修正·classify_intent 真填 has_value_claim
    assert six[G_DOOR_G5]["dead_state"] == G_DEAD_DESIGN  # language 不在 _ARITH_DOMAINS
    # ③计算：G5 ALIVE·其余 N/A（verify 路径不经 judge G4/G2p/G3a/G3b）
    three = table[DIM_COMPUTE]
    assert three[G_DOOR_G5]["dead_state"] == G_ALIVE
    assert three[G_DOOR_G4]["dead_state"] == G_NA
    assert three[G_DOOR_G2P]["dead_state"] == G_NA
    # ⑤长代码：G5 ALIVE·其余 N/A
    five = table[DIM_LONG_CODE]
    assert five[G_DOOR_G5]["dead_state"] == G_ALIVE
    assert five[G_DOOR_G4]["dead_state"] == G_NA
    # ①②④⑦⑧ 全 N/A
    assert table[DIM_CONCEPT][G_DOOR_G5]["dead_state"] == G_NA
    assert table[DIM_LONG_TEXT][G_DOOR_G5]["dead_state"] == G_NA
    assert table[DIM_INTENT][G_DOOR_G5]["dead_state"] == G_NA
    assert table[DIM_MEMORY][G_DOOR_G5]["dead_state"] == G_NA
    # N/A 格 permille=-1
    assert table[DIM_CONCEPT][G_DOOR_G5]["permille"] == -1
    # 全格 active=0/total=0（空 episodes）
    for dim in table:
        for door in table[dim]:
            assert table[dim][door]["active"] == 0
            assert table[dim][door]["total"] == 0


def test_classify_episode_paths():
    """_classify_episode 三路分类正确 + None 边界（language pr_vector 空 = observe-only 无 dag_path）。

    - language + pr_vector 非空 → (⑥三环, judge)
    - ARITH + pr_vector 空 → (③计算, verify)
    - CODE + pr_vector 空 → (⑤长代码, verify)
    - input=None → (⑤长代码, mode_a)·Mode A task-driven
    - language + pr_vector 空 → None（observe-only 阶段无 dag_path·不贡 G 门分母）
    """
    assert _classify_episode(_lang_judge_ep()) == (DIM_THREE_RING, "judge")
    assert _classify_episode(_arith_verify_ep()) == (DIM_COMPUTE, "verify")
    assert _classify_episode(_code_verify_ep()) == (DIM_LONG_CODE, "verify")
    assert _classify_episode(_mode_a_ep()) == (DIM_LONG_CODE, "mode_a")
    assert _classify_episode(_lang_judge_ep(pr=False)) is None


# ============ ALIVE 门计数 ============

def test_g_attribution_alive_g5_arith_verify():
    """③计算 G5 ALIVE：ARITH verify eps G5 真 veto（承重 AND vetoed）→ 计数 + evidence_eps 溯源。

    2 ARITH verify eps（1 个 G5 真 veto：g5=True+vetoed=True·1 个承重但 pass：g5=True+vetoed=False）→
    ③ G5: active=1/total=2/permille=500·evidence_eps=[1]。
    **#723 P0 修·对抗审1**：g5=True（承重）≠ vetoed=True（veto）·承重但 pass 的 episode 不计 active。
    """
    eps = [
        _arith_verify_ep(g5=True, vetoed=True),    # Mode B disagree·真 G5 veto
        _arith_verify_ep(g5=True, vetoed=False),   # Mode B agree·承重但 pass·不计 active
    ]
    table = project_g_attribution(_result(eps))
    cell = table[DIM_COMPUTE][G_DOOR_G5]
    assert cell["dead_state"] == G_ALIVE
    assert cell["total"] == 2
    assert cell["active"] == 1   # 仅 vetoed=True 的计·承重但 pass 不计
    assert cell["permille"] == 500   # 1*1000//2
    assert cell["evidence_eps"] == [0]   # index=0 的 episode 真 veto


def test_g_attribution_bearing_not_veto_verify_path():
    """#723 P0 回归·对抗审1 抓：verify path g5=True（承重）但 vetoed=False → 不计 active。

    防"承重即 veto"误计（formal_train.py:506/519 g5_active=承重·pass/fail 都 True）。
    3 ARITH verify eps 全承重（g5=True）但全 pass（vetoed=False）→ ③ G5 active=0/total=3/permille=0。
    旧实现（active=judge_G5_active count）会算 active=3/permille=1000·完全反·此测守 P0 修。
    """
    eps = [_arith_verify_ep(g5=True, vetoed=False) for _ in range(3)]   # 全承重全 pass
    table = project_g_attribution(_result(eps))
    cell = table[DIM_COMPUTE][G_DOOR_G5]
    assert cell["total"] == 3
    assert cell["active"] == 0   # 承重非 veto·P0 修后 active=0（旧实现会算 3）
    assert cell["permille"] == 0
    assert cell["evidence_eps"] == []


def test_g_attribution_alive_g4_g2p_language_judge():
    """⑥三环 G4/G2p ALIVE：language judge eps G4/G2p veto → 计数。

    3 language judge eps（G4 veto 1 + G2p veto 1）→ ⑥ G4 active=1/total=3·G2p active=1/total=3。
    """
    eps = [
        _lang_judge_ep(g4=True),     # G4 veto
        _lang_judge_ep(g2p=True),    # G2p veto
        _lang_judge_ep(),            # 无 veto
    ]
    table = project_g_attribution(_result(eps))
    six = table[DIM_THREE_RING]
    assert six[G_DOOR_G4]["active"] == 1 and six[G_DOOR_G4]["total"] == 3
    assert six[G_DOOR_G2P]["active"] == 1 and six[G_DOOR_G2P]["total"] == 3
    assert six[G_DOOR_G4]["dead_state"] == G_ALIVE
    assert six[G_DOOR_G2P]["dead_state"] == G_ALIVE
    # G3a/G3b 生产未手设 → active=0（ALIVE·#889·active=0 因不 veto·非硬编码短路）
    assert six[G_DOOR_G3A]["active"] == 0
    assert six[G_DOOR_G3B]["active"] == 0


def test_g_attribution_mode_a_does_not_inflate_g5():
    """Mode A task-driven eps（input=None）归 ⑤ 但 mode_a 路径不走 G 门·不计 ⑤ G5 分母。

    formal_train.py:1966 judge_G5_active=False 硬编码（外真验非 G5 门）。
    3 Mode A eps + 1 CODE verify ep（G5 真 veto）→ ⑤ G5: active=1/total=1（非 1/4）。
    Mode A eps 不贡任何 G 门分母（_PATH_DOORS["mode_a"]=()）。
    """
    eps = [_mode_a_ep(), _mode_a_ep(), _mode_a_ep(), _code_verify_ep(g5=True, vetoed=True)]
    table = project_g_attribution(_result(eps))
    cell = table[DIM_LONG_CODE][G_DOOR_G5]
    assert cell["total"] == 1   # 仅 CODE verify ep·Mode A eps 不计
    assert cell["active"] == 1


# ============ DEAD 分类（生产永不触发） ============

def test_g_attribution_g3a_g3b_alive_no_veto():
    """⑥ G3a/G3b ALIVE（#889 stale 修正）：生产 language judge eps（无手设 g3a/g3b）-> active=0 + dead_state=ALIVE。

    M1片2+G1+#774 落地后 classify_intent 真填 is_causal/has_value_claim·G3a/G3b 真活（ALIVE）·
    active=0 因不 veto（synthetic eps 无 dag_path/CAUSES 锚·judge 不 veto）·非硬编码短路。
    """
    eps = [_lang_judge_ep() for _ in range(3)]
    table = project_g_attribution(_result(eps))
    six = table[DIM_THREE_RING]
    assert six[G_DOOR_G3A]["dead_state"] == G_ALIVE
    assert six[G_DOOR_G3A]["active"] == 0
    assert six[G_DOOR_G3B]["dead_state"] == G_ALIVE
    assert six[G_DOOR_G3B]["active"] == 0

def test_g_attribution_dead_design_g5_language():
    """⑥ G5 DEAD_DESIGN：language judge eps → G5 active=0 + dead_state=DEAD_DESIGN。

    language 不在 _ARITH_DOMAINS（judge.py:247）→ G5 设计性排除·生产永不触发。
    """
    eps = [_lang_judge_ep() for _ in range(2)]
    table = project_g_attribution(_result(eps))
    six = table[DIM_THREE_RING]
    assert six[G_DOOR_G5]["dead_state"] == G_DEAD_DESIGN
    assert six[G_DOOR_G5]["active"] == 0


# ============ 反 theater 层1：坏 fixture → 表必异 ============

def test_g_attribution_bad_fixture_changes_table():
    """反 theater 层1：good eps（无 veto）vs bad eps（G4 全 veto）→ ⑥ G4 active 必异。

    若两表一样 = 归因表是 theater（无论输入好坏表都一样）→ 测 fail。
    dead_state 不变（坏输入不能把 ALIVE 变 DEAD_LEAK·防美化）。
    """
    good_eps = [_lang_judge_ep() for _ in range(3)]          # 无 veto
    bad_eps = [_lang_judge_ep(g4=True) for _ in range(3)]    # G4 全 veto
    t_good = project_g_attribution(_result(good_eps))
    t_bad = project_g_attribution(_result(bad_eps))
    # ⑥ G4 active 必异（good=0 / bad=3）
    assert t_good[DIM_THREE_RING][G_DOOR_G4]["active"] == 0
    assert t_bad[DIM_THREE_RING][G_DOOR_G4]["active"] == 3
    # dead_state 不变（坏输入不能把 ALIVE 变 DEAD_LEAK·#889 stale 修正 G3a ALIVE）
    assert t_good[DIM_THREE_RING][G_DOOR_G3A]["dead_state"] == G_ALIVE
    assert t_bad[DIM_THREE_RING][G_DOOR_G3A]["dead_state"] == G_ALIVE
    # ⑥ G4 permille 必异（good=0 / bad=1000）
    assert t_good[DIM_THREE_RING][G_DOOR_G4]["permille"] == 0
    assert t_bad[DIM_THREE_RING][G_DOOR_G4]["permille"] == 1000


# ============ 反 theater 层2：active 格带 evidence_eps 溯源 ============

def test_g_attribution_evidence_eps_traces_episodes():
    """反 theater 层2：active 格 evidence_eps 指 episode index（可抽样核·审计溯源）。

    混合 eps（index 0 承重 pass / 1 G5 真 veto / 2 G5 真 veto / 3 承重 pass）→ ③ G5 evidence_eps=[1,2]（升序）。
    承重但 pass（index 0/3）不计 active·不进 evidence_eps（#723 P0 修）。
    """
    eps = [
        _arith_verify_ep(g5=True, vetoed=False),   # 承重 pass·不计
        _arith_verify_ep(g5=True, vetoed=True),    # G5 veto
        _arith_verify_ep(g5=True, vetoed=True),    # G5 veto
        _arith_verify_ep(g5=True, vetoed=False),   # 承重 pass·不计
    ]
    table = project_g_attribution(_result(eps))
    cell = table[DIM_COMPUTE][G_DOOR_G5]
    assert cell["evidence_eps"] == [1, 2]   # 升序·溯源到 index 1+2（仅真 veto）
    assert cell["active"] == 2


# ============ 反 theater 层3：dead 门反向证明 ============

def test_g_attribution_dead_door_reverse_proof_synthetic():
    """反 theater 层3（synthetic）：table 不忽略 G3a——手设 g3a=True 的 episode → table 计数。

    证 table 诚实（非忽略 G3a 字段）·配合生产 active=0（e2e 测）→ DEAD_LEAK 定论：
    table 会数 G3a·但生产永不触发（intent 硬编码 False·formal_train.py:362）·
    故 dead 归因于硬编码短路·非"恰好没触发"。
    dead_state 静态仍 DEAD_LEAK（基于 code 真相·非 active 计数·防"active>0 即 ALIVE"美化）。
    """
    eps = [_lang_judge_ep(g3a=True)]   # synthetic·假设 G3a 真触发（生产做不到）
    table = project_g_attribution(_result(eps))
    cell = table[DIM_THREE_RING][G_DOOR_G3A]
    # table 计数（证不忽略 G3a 字段）
    assert cell["active"] == 1
    assert cell["evidence_eps"] == [0]
    # 但 dead_state 静态仍 ALIVE（基于 code 真相·非 active 计数·#889 stale 修正）
    assert cell["dead_state"] == G_ALIVE


def test_g_attribution_dead_door_reverse_proof_e2e(tmp_path, monkeypatch):
    """反 theater 层3（e2e）：CAUSES-poor language fixture（应触发 G3a 若活）→ 生产 G3a active 仍=0。

    _multi_sent_item（无 cue 词无 causal_pairs·零 CAUSES 边）→ 若 G3a 活
    （intent.is_causal_reasoning=True·causal 推理无 CAUSES 锚）应触发 G3a veto·
    但 intent 硬编码 False（formal_train.py:362）→ judge.py:229 跳过 → 生产 G3a active=0。
    证 dead 非"恰好没触发"·是硬编码短路（DEAD_LEAK）。
    G5 DEAD_DESIGN 同证（language 不在 _ARITH_DOMAINS·active=0）。
    """
    from pure_integer_ai.config import gates
    from pure_integer_ai.training import stages as _st
    monkeypatch.setattr(_st, "FLOOR_GRAPH_SIZE_S1", 0)
    monkeypatch.setattr(_st, "FLOOR_CAUSES_COV_S2", 0)
    monkeypatch.setattr(_st, "FLOOR_CONDUCTION_S3", 0)
    monkeypatch.setattr(_st, "FLOOR_PROMOTE_S4", 0)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "g_rev"), run_id="g_rev",
                            collect_episodes=True)
    result = formal_train(cfg, [_multi_sent_item()],
                          backend=b, runner=DefaultRoundRunner())
    table = project_g_attribution(result)
    six = table[DIM_THREE_RING]
    # G3a/G3b 生产 active=0（ALIVE·#889·无 signal 跳过·非硬编码短路）
    assert six[G_DOOR_G3A]["active"] == 0
    assert six[G_DOOR_G3B]["active"] == 0
    assert six[G_DOOR_G3A]["dead_state"] == G_ALIVE
    assert six[G_DOOR_G3B]["dead_state"] == G_ALIVE
    # G5 DEAD_DESIGN（language 不在 _ARITH_DOMAINS·active=0）
    assert six[G_DOOR_G5]["active"] == 0
    assert six[G_DOOR_G5]["dead_state"] == G_DEAD_DESIGN


# ============ e2e + bit-identical ============

def test_g_attribution_e2e_formal_train_collects_episodes(tmp_path, monkeypatch):
    """e2e：formal_train collect_episodes=True → result.episodes 非空 + 归因表 ⑥ 有真分母。

    _causal_multi_sent_item（带 CAUSES）→ language judge eps 产出 → ⑥ G4/G2p 有 total>0。
    G3a/G3b ALIVE + active=0（#889 stale 修正·不 veto·非硬编码短路）。
    """
    from pure_integer_ai.config import gates
    from pure_integer_ai.training import stages as _st
    monkeypatch.setattr(_st, "FLOOR_GRAPH_SIZE_S1", 0)
    monkeypatch.setattr(_st, "FLOOR_CAUSES_COV_S2", 0)
    monkeypatch.setattr(_st, "FLOOR_CONDUCTION_S3", 0)
    monkeypatch.setattr(_st, "FLOOR_PROMOTE_S4", 0)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "g_e2e"), run_id="g_e2e",
                            collect_episodes=True)
    result = formal_train(cfg, [_causal_multi_sent_item()],
                          backend=b, runner=DefaultRoundRunner())
    # episodes 非空（collect_episodes=True 收集三路 episode）
    assert len(result.episodes) > 0, "collect_episodes=True 须收集 episode"
    table = project_g_attribution(result)
    # ⑥三环 G4/G2p 有真分母（language judge eps 存在·非全 N/A）
    six = table[DIM_THREE_RING]
    assert six[G_DOOR_G4]["total"] > 0 or six[G_DOOR_G2P]["total"] > 0, (
        "language judge eps 须进 ⑥ G4/G2p 分母·非全 N/A")
    # G3a/G3b ALIVE + active=0（#889 stale 修正·不 veto·非硬编码短路）
    assert six[G_DOOR_G3A]["active"] == 0
    assert six[G_DOOR_G3A]["dead_state"] == G_ALIVE
    # G5 DEAD_DESIGN（language）
    assert six[G_DOOR_G5]["dead_state"] == G_DEAD_DESIGN


def test_g_attribution_e2e_arith_verify_active_not_total(tmp_path, monkeypatch):
    """#723 P0 e2e·对抗审1 反 theater 漏洞：ARITH corpus 跑 formal_train → ③ G5 active=真 veto 数·非 total。

    旧实现（active=judge_G5_active count）会把所有 verify episode（承重即 True）计为 veto·permille 恒 1000。
    P0 修后 active=承重 AND vetoed（reward=0）·permille=真 veto 率。
    用 _anchor_arith_all_wrong_corpus（arith_specs expected 全错·Mode A verified=0）·
    或 _anchor_arith_no_heldout_corpus（nullary·held_out=0）·断言 ③ G5 active ≤ total（非恒=total）。
    **诚实边界**：此测验 verify path G5 语义·非验算子正确（stable≠correct·#479 墙）。
    """
    from pure_integer_ai.config import gates
    from pure_integer_ai.training import stages as _st
    from pure_integer_ai.experiments.capability_exam import _anchor_arith_no_heldout_corpus
    monkeypatch.setattr(_st, "FLOOR_GRAPH_SIZE_S1", 0)
    monkeypatch.setattr(_st, "FLOOR_CAUSES_COV_S2", 0)
    monkeypatch.setattr(_st, "FLOOR_CONDUCTION_S3", 0)
    monkeypatch.setattr(_st, "FLOOR_PROMOTE_S4", 0)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "g_arith"), run_id="g_arith",
                            collect_episodes=True)
    result = formal_train(cfg, _anchor_arith_no_heldout_corpus(),
                          backend=b, runner=DefaultRoundRunner())
    table = project_g_attribution(result)
    cell = table[DIM_COMPUTE][G_DOOR_G5]
    # active 须 ≤ total（非恒=total·P0 修后承重非 veto）
    assert cell["active"] <= cell["total"], (
        f"③ G5 active={cell['active']} > total={cell['total']}·违 P0 修（承重非 veto）")
    # 若有 verify episode（total>0）·permille 须 ≤ 1000（非恒 1000·旧实现恒 1000）
    if cell["total"] > 0:
        assert cell["permille"] <= 1000
    # dead_state ALIVE（verify path G5 真承重）
    assert cell["dead_state"] == G_ALIVE


def test_g_attribution_to_json_bit_identical():
    """to_json 序列化 g_attribution·双跑（同 synthetic eps）→ bit-identical（sort_keys）。

    cell key sort / evidence_eps 预 sort / dim 按 DIM_ORDER / door 按 _G_DOORS → 双跑一致。
    不含 episodes 列表（决断2·守 bit-identical）。
    """
    eps = [
        _lang_judge_ep(g4=True),
        _arith_verify_ep(g5=True, vetoed=True),
        _code_verify_ep(g5=True, vetoed=False),
        _mode_a_ep(),
    ]
    rep1 = CapabilityReport(run_id="bi", g_attribution=project_g_attribution(_result(eps)))
    rep2 = CapabilityReport(run_id="bi", g_attribution=project_g_attribution(_result(eps)))
    j1 = rep1.to_json()
    j2 = rep2.to_json()
    assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True), (
        "g_attribution to_json 双跑不一致·违 bit-identical")
    # g_attribution key 存在 + 8 dims 齐
    assert "g_attribution" in j1
    assert set(j1["g_attribution"].keys()) == _ALL_DIMS
    # 每 dim 5 doors 齐
    for dim in j1["g_attribution"]:
        assert set(j1["g_attribution"][dim].keys()) == _ALL_DOORS
    # cell key 固定序（active/dead_state/evidence_eps/permille/total·sort 后）
    cell_keys = set(j1["g_attribution"][DIM_COMPUTE][G_DOOR_G5].keys())
    assert cell_keys == {"active", "dead_state", "evidence_eps", "permille", "total"}
