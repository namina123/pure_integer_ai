"""tests/test_w1_arith_learning_alive — W1 算术域 reward>0 真流深化（验 op_confidence 学习闭环真活）。

验（doc/重来_断奶阶段训练设计_2026-07-11.md W1）：
1. op_confidence 半环真活（_verify_generalization + _run_task_driven_generate 写 record_op_outcome·台账非空 sn>0）
2. 算术域泛化率量化（generalization.verified/total_held_out>0·学到能力·反"学不会"）
3. reward 落 edge strength 永久 no-op（COMPOSES 边 strength 全=DEFAULT_STRENGTH·架构铁律·非 dead-end·非 defer）
4. β_arith 不染算术域（experience_count 台账空·_run_verify_round 绕 episode_loop·不写 e_sn/e_tn）

**架构认知更正**（W0 memory topic dead-end 措辞误导·W1 更正）：reward 落 **edge strength** = 永久 no-op
（§4.5·COMPOSES 边 inert·HEAD_STRUCTURAL·reward_propagate 只走 EDGE_CAUSES·架构铁律·非 dead-end·非 defer·
doc/重来_算术域observe设计补充.md:151 + VM图灵完备:171）。算术域"学习"走 **op_confidence 台账**
（_verify_generalization:2656 + _run_task_driven_generate:2755 写 record_op_outcome → recognize_operators:1072
读择优）·与 edge strength 是两条独立路径。W1 非解 dead-end（让 reward 落 edge 违架构）·是验已活+锁死。

三台账诊断证 W0 生产路径跑后：op_confidence sn=22 strength=23·experience_count=0·COMPOSES strength
全=DEFAULT·generalization verified=10/10（square 骨架学到→识别 10 held-out→vm_proof 全验）。
"""
from __future__ import annotations

from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith
from pure_integer_ai.storage.op_confidence import OP_CONFIDENCE_TABLE
from pure_integer_ai.storage.experience_count import EXPERIENCE_COUNT_TABLE
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.storage.edge_store import DEFAULT_STRENGTH


def test_w1_op_confidence_half_ring_alive(tmp_path):
    """★op_confidence 半环真活：_verify_generalization + _run_task_driven_generate 写 record_op_outcome。

    台账非空（sn>0）+ generalization.verified>0（学习闭环真活·非 theater·非 dead-end）。
    算术域"学到能力"= square 骨架从发现集学到 → 识别 held-out 新输入 → vm_proof 全验复现。
    反"学不会"：op_confidence 台账真有记录 + 泛化率>0（reward 落 op_confidence 非 edge strength）。
    """
    result, backend = run_weaning_arith(
        rounds_per_stage=1, training_mode=True, flat_floors=True,
        run_dir=str(tmp_path / "opc"), return_backend=True)
    # op_confidence 台账非空（_verify_generalization:2656 + _run_task_driven_generate:2755 写 record_op_outcome）
    rows = backend.select(OP_CONFIDENCE_TABLE, where={}, limit=100)
    assert rows, (
        "op_confidence 台账须非空（_verify_generalization + _run_task_driven_generate 写 record_op_outcome·"
        "算术域学习闭环真活·非 theater·非 dead-end）")
    sn_total = sum(r["sn"] for r in rows)
    assert sn_total > 0, (
        f"op_confidence sn 须>0（verified 写 sn++·学习发生·reward 落 op_confidence 台账非 edge strength）"
        f"·got sn_total={sn_total}")
    # generalization.verified>0（泛化率量化学到能力·op_confidence 半环产物）
    g = result.generalization
    assert g.verified > 0, (
        f"generalization.verified 须>0（vm_proof 验骨架绑参复现 held-out 新输入值·学到能力·反学不会）"
        f"·got {g.verified}")


def test_w1_arith_generalization_quantified(tmp_path):
    """泛化率量化：generalization.verified/total_held_out>0 + discovered_operators 非空。

    square 骨架从发现集学到（discovered_operators 非空）→ 识别 held-out 新输入（recognized>0）→
    vm_proof 全验复现（verified/total_held_out>0·泛化率量化"学到能力"·反"学不会"）。
    op_confidence 半环在 stage loop 前（formal_train.py:1695·非 reward 驱动·gate OFF 也跑）·W1 验绝对真活。
    """
    result, _ = run_weaning_arith(
        rounds_per_stage=1, training_mode=True, flat_floors=True,
        run_dir=str(tmp_path / "gen"), return_backend=True)
    assert result.discovered_operators, (
        "discovered_operators 须非空（骨架发现学到·square 算子·auto_discover_operators 真生产 caller）")
    g = result.generalization
    assert g.total_held_out > 0, (
        f"total_held_out 须>0（per-shape 留 held-out 新输入识别·非发现集→真泛化非循环 theater）"
        f"·got {g.total_held_out}")
    assert g.recognized > 0, (
        f"recognized 须>0（识别 held-out 命中已学骨架·recognize_operators READ 消费）·got {g.recognized}")
    assert g.verified > 0, (
        f"verified 须>0（vm_proof 验复现·泛化率>0·学到能力）·got {g.verified}")
    # 泛化率 = verified/total_held_out（构造性必然对正确算子·fixture spec 正确→全验）
    rate = g.verified * 1000 // max(g.total_held_out, 1)
    assert rate > 0, f"泛化率 rate_permille 须>0（学到能力覆盖 held-out 新输入）·got {rate}"


def test_w1_edge_strength_permanent_noop(tmp_path):
    """COMPOSES edge strength 不变（架构铁律·回归防护·非 dead-end·非 defer）。

    COMPOSES 边 strength 全=DEFAULT_STRENGTH（未被任何机制改）。架构铁律：reward_propagate 只走
    EDGE_CAUSES（reward_propagate.py:133-134）·COMPOSES 边 inert（dag_path head_types 不含 COMPOSES·
    §4.5·doc/重来_算术域observe设计补充.md:151+VM图灵完备:171 明确"永久非 defer"）。算术域学习走
    op_confidence 台账非 edge strength（独立路径）。

    **诚实边界**（审1）：算术域 _run_verify_round 绕 episode_loop（formal_train.py:375）·reward_propagate
    从未对算术项调用·故本测试验"COMPOSES strength 不被任何机制改"（回归防护·若未来违架构让 reward 落
    COMPOSES·本测试 catch）·非专门验"reward 对 COMPOSES 的非影响"（那须 reward 真跑后比 edge 变化·
    arith reward 走 op_confidence 非 edge）。反"解 dead-end 让 reward 落 edge"违架构铁律。
    """
    _, backend = run_weaning_arith(
        rounds_per_stage=1, training_mode=True, flat_floors=True,
        run_dir=str(tmp_path / "edge"), return_backend=True)
    rows = backend.select("edge", where={"edge_type": EDGE_COMPOSES}, limit=200)
    assert rows, "COMPOSES 边须存在（observe 建 struct_ref COMPOSES 树 + discover 建独立根）"
    strengths = {r["strength"] for r in rows}
    # 全=DEFAULT_STRENGTH·未被 reward 强化（reward_propagate 只走 EDGE_CAUSES·COMPOSES inert·HEAD_STRUCTURAL）
    assert strengths == {DEFAULT_STRENGTH}, (
        f"COMPOSES edge strength 须全=DEFAULT_STRENGTH({DEFAULT_STRENGTH})（reward 落 edge 永久 no-op·"
        f"架构铁律·非 dead-end·非 defer·§4.5·reward_propagate 只走 EDGE_CAUSES）·got strengths={strengths}。"
        f"算术域学习走 op_confidence 台账非 edge strength·非解 dead-end")


def test_w1_beta_arith_no_contam_arith(tmp_path):
    """β_arith 不染算术域：experience_count 台账空。

    算术域 _run_verify_round 绕 episode_loop（formal_train.py:375 早返·_is_verify_modality 守 CODE/ARITH
    进 _run_verify_round 非 episode_loop）·不调 propagate_reward·不写 e_sn/e_tn。β_arith（e_sn/e_tn rate）
    不染算术域 reward（直调 vm_proof_fn·reward=执行值对错非 experience rate）。验 STEP4 β_arith 缓解对
    算术域 reward 无染（生产路径·_run_verify_round reward 腿独立于 experience_count）。
    """
    _, backend = run_weaning_arith(
        rounds_per_stage=1, training_mode=True, flat_floors=True,
        run_dir=str(tmp_path / "beta"), return_backend=True)
    rows = backend.select(EXPERIENCE_COUNT_TABLE, where={}, limit=100)
    assert len(rows) == 0, (
        f"experience_count 台账须空（算术域 _run_verify_round:375 绕 episode_loop 不调 propagate_reward·"
        f"不写 e_sn/e_tn·β_arith e_sn/e_tn rate 不染算术域 reward·直调 vm_proof_fn·reward=执行值对错）"
        f"·got {len(rows)} 行")
