"""tests/test_w5_d5_calibration — W5 D5 Mode B 预验台账（工程闸门·墙内弱）。

验（doc/重来_断奶阶段训练设计_2026-07-11.md W5）：
1. e2e：run_weaning_arith(calibrate_mode_b=True) → _run_calibration_phase 真 caller → 台账非空 +
   mode_b_prevalidated=True（D5 过）+ weaning_ready 仍 False（D1/D2/D3/D4/E2 defer）
2. bit-identical：默认 calibrate_mode_b=False → 台账空 → D5 blocker（既有行为不变·零翻）
3. floor guard：stable-low flat window（mode_b_pass rate 平但 <FLOOR_MODE_B=500）→ mode_b_prevalidated=False
   （FLOOR_MODE_B 真守·非仅 not_rising·既有测空缺 stable-low 场景·反 theater）
4. Mode A/B 真独立（防换名）：坏 corpus（expected 全错=999）→ Mode A fail（vm_proof≠999）∧ Mode B agree
   （两树同函数异 shape·cross_verify 无 expected·AGREE）→ false positive（a=0,b=1）→ false_pass_rate>0

**W5 = stage4 并行 Mode A vs B 评估 + record_calibration 真 caller**。general-purpose agent 核证 D5 断点
（record_calibration 全建零 caller / 路径 B 已接 mode_b_prevalidated 真函数非硬编 / stage4 零并行评估 /
window 判定就位 / D5 域无关同 D4·Mode A/B 零 judge/teacher）+ Plan agent 设计 stage4 并行评估子阶段
（_run_calibration_phase·WEANING_WINDOW_ROUNDS 轮·Mode A vm_proof + Mode B cross_verify_pair·读优先级纯评估）。

**关键**：D5 路径 B 已读 mode_b_prevalidated(backend) 真函数（formal_train.py·非硬编·异于 W4）·W5 不改路径 B·
只补 caller 填台账。D5 域无关（同 D4·不似 D3·无须判定接口）→ 算术域走通用 track。

**诚实边界**：D5 单闸门过非真断奶（weaning_ready 仍 False·D1/D2/D3/D4/E2 defer）·flat trend=不回升=通过
（MUTABLE_MONOTONE·学树 stage4 静态·4 轮同 rate·FLOOR_MODE_B=500 守低平台）·stable≠correct（墙内弱）。
"""
from __future__ import annotations

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith
from pure_integer_ai.experiments.collection import load_arith_bad_corpus
from pure_integer_ai.teacher.weaning_calibration import (
    register_weaning_calibration, record_calibration,
    false_pass_rate, mode_b_agreement, mode_b_prevalidated,
)


def test_w5_d5_calibration_e2e(tmp_path):
    """★e2e：_run_calibration_phase 真 caller → 台账非空 + mode_b_prevalidated=True（D5 过）+ weaning_ready False。

    run_weaning_arith(calibrate_mode_b=True) → formal_train stage4 末跑 _run_calibration_phase
    （WEANING_WINDOW_ROUNDS=4 轮并行 Mode A vs B·record_calibration 真写台账）→ :2068 mode_b_prevalidated(backend)
    读非空台账 → flat high trend（算术 fixture 全对→mode_b 4 轮全 1→rate 1000·不回升 + >=500）→ True → D5 过。
    weaning_ready 仍 False（D1/D2/D3/D4/E2 defer·只 D5 单闸门过·诚实非真断奶）。
    补 test_weaning_gates 模块测盲区（record_calibration 函数→端到端 formal_train→D5 通过）。
    """
    result, backend = run_weaning_arith(
        calibrate_mode_b=True, return_backend=True,
        run_dir=str(tmp_path / "w5e2e"))
    rows = backend.select("weaning_calibration", where={})
    # ① 台账非空（_run_calibration_phase 真 caller·非手工注入·反纸面闭合）
    assert len(rows) > 0, "calibration 台账须非空（_run_calibration_phase 真 caller）"
    # ② mode_b_prevalidated=True（台账非空 + flat high trend=不回升=通过 + floor 1000>=500）
    assert mode_b_prevalidated(backend) is True, (
        "台账非空 + flat high trend → mode_b_prevalidated=True（D5 过）")
    # ③ D5 blocker 消失（mode_b_prevalidated=True → :2068 mode_b_ok=True → D5 过）
    assert "D5_mode_b_prevalidated" not in result.weaning_blockers, (
        f"D5 须过（mode_b_prevalidated=True）·blockers={result.weaning_blockers}")
    # ④ weaning_ready 仍 False（D1/D2/D3/D4/E2 defer·只 D5 单闸门过·诚实非真断奶）
    assert not result.weaning_ready, (
        "weaning_ready 仍 False（D1/D2/D3/D4/E2 defer·W5 只过 D5）")
    # ⑤ round_id distinct ≥4（WEANING_WINDOW_ROUNDS=4 填窗·10M 偏移 namespace·mode_b_pass_series 按 rid 分组）
    distinct_rids = {int(r["round_id"]) for r in rows}
    assert len(distinct_rids) >= 4, (
        f"须 ≥4 distinct round_id（WEANING_WINDOW_ROUNDS=4）·got {len(distinct_rids)}")
    # ⑥ calibration round_id 在 10M namespace（避 training round_id 碰撞·W5 设计）
    assert all(int(r["round_id"]) >= 10_000_000 for r in rows), (
        "calibration round_id 须在 10M namespace（避 training 碰撞）")
    # ⑦ mode_a/mode_b 0/1 + 有 mode_b_pass=1（cross_verify_pair 真 caller·all_agree·算术 fixture 正确）
    assert all(int(r["mode_a_pass"]) in (0, 1) for r in rows), "mode_a_pass 须 0/1"
    assert all(int(r["mode_b_pass"]) in (0, 1) for r in rows), "mode_b_pass 须 0/1"
    assert any(int(r["mode_b_pass"]) == 1 for r in rows), (
        "须有 mode_b_pass=1（cross_verify_pair 真 caller·all_agree）")


def test_w5_d5_bit_identical_default_off(tmp_path):
    """★bit-identical：默认 calibrate_mode_b=False → 台账空 → D5 blocker（既有行为不变·零翻）。

    默认 calibrate_mode_b=False → stage4 块 `if config.calibrate_mode_b:` 不执行 → _run_calibration_phase
    不调 → 台账空 → mode_b_prevalidated(backend) 返 False（series<4）→ D5 blocker（同既有行为·bit-identical）。
    W0-W4 既有 reward 闭环不受影响（conduction_rate>0）。
    """
    result, backend = run_weaning_arith(
        return_backend=True, run_dir=str(tmp_path / "w5bit"))   # 默认 calibrate_mode_b=False
    rows = backend.select("weaning_calibration", where={})
    # 台账空（_run_calibration_phase 未调·bit-identical）
    assert len(rows) == 0, "默认 calibrate_mode_b=False → 台账须空（bit-identical·既有行为不变）"
    # mode_b_prevalidated=False（台账空→series<4→False）
    assert mode_b_prevalidated(backend) is False
    # D5 blocker 在（同既有·formal_train.py D5 标注）
    assert "D5_mode_b_prevalidated" in result.weaning_blockers, (
        f"默认 off → D5 须 blocker·blockers={result.weaning_blockers}")
    assert not result.weaning_ready
    # W0-W4 既有 reward 闭环不受 W5 影响（conduction_rate>0·vm_proof 自锚 reward 真流）
    assert result.final_metrics.conduction_rate > 0, "W0 reward 闭环须不受 W5 影响"


def test_w5_d5_floor_guard_stable_low_rejected():
    """★floor guard：stable-low flat window（rate 平但 <FLOOR_MODE_B=500）→ mode_b_prevalidated=False。

    not_rising=True（flat 平）但 window[-1]<FLOOR_MODE_B（500）→ False。验 FLOOR_MODE_B 真守
    （非仅 not_rising 判定·低通过率平台拒绝·反 theater·既有测 stable-high/rising/empty 空缺 stable-low）。
    record_calibration producer（W5 同源 caller）注入·fresh DictBackend + bootstrap + register。
    """
    b = DictBackend()
    bootstrap(b)
    register_weaning_calibration(b)
    # 4 轮·每轮 mode_b 通过率 2/10=200（flat 平·<FLOOR_MODE_B=500·低通过率平台）
    for rid in [1, 2, 3, 4]:
        for _ in range(2):
            record_calibration(b, round_id=rid, mode_a_pass=1, mode_b_pass=1)
        for _ in range(8):
            record_calibration(b, round_id=rid, mode_a_pass=1, mode_b_pass=0)
    assert mode_b_prevalidated(b) is False, (
        "stable-low flat（rate 200<500）→ FLOOR_MODE_B 守→mode_b_prevalidated=False（低平台拒绝）")


def test_w5_mode_a_b_independent_not_renamed(tmp_path):
    """★Mode A/B 真独立（防换名）：坏 corpus → Mode A fail ∧ Mode B agree → false positive → false_pass_rate>0。

    坏 corpus（load_arith_bad_corpus·2 square·expected=999 全错）：Mode A = vm_proof execute 学树（25/36）
    vs spec.expected（999）→ 25≠999 → mode_a_pass=0（fail）。Mode B = cross_verify_pair（root_a `b*b` ×
    root_b `Sigma(1,b,b)`·两树同函数异 shape·无 expected·都算 25/36 → AGREE）→ mode_b_pass=1。
    **关键反 theater**：mode_a=0 ∧ mode_b=1 同时存在（false positive）= 两路独立判据·非换名
    （若换名同机制→两者同 0 或同 1→无 false positive→false_pass_rate=(0,n)·此测失败）。
    false_pass_rate/mode_b_agreement 真采（台账非空后返真值·非默认 (0,1)）。
    """
    result, backend = run_weaning_arith(
        calibrate_mode_b=True, return_backend=True, corpus=load_arith_bad_corpus(),
        run_dir=str(tmp_path / "w5ind"))
    rows = backend.select("weaning_calibration", where={})
    assert len(rows) > 0
    # Mode A fail（expected=999 ≠ 真值 25/36）→ mode_a_pass=0
    mode_a_vals = {int(r["mode_a_pass"]) for r in rows}
    assert 0 in mode_a_vals, (
        f"坏 corpus（expected 全错）→ Mode A 须 fail（vm_proof≠999）·mode_a_vals={mode_a_vals}")
    # Mode B agree（两树同函数异 shape·cross_verify 无 expected·AGREE）→ mode_b_pass=1
    mode_b_vals = {int(r["mode_b_pass"]) for r in rows}
    assert 1 in mode_b_vals, (
        f"Mode B 须 agree（两树同函数异 shape·无 expected·AGREE）·mode_b_vals={mode_b_vals}")
    # 关键反 theater：mode_a=0 ∧ mode_b=1 同时存在（false positive·证明两路独立·非换名）
    false_positives = [r for r in rows
                       if int(r["mode_a_pass"]) == 0 and int(r["mode_b_pass"]) == 1]
    assert false_positives, (
        "须存在 mode_a=0 ∧ mode_b=1 行（false positive·Mode A fail 但 Mode B agree·"
        "两路独立判·非换名·若换名则两者同 0 或同 1）")
    # false_pass_rate 真采（台账非空后返真值·p>0 有 false positive·非默认 (0,1)）
    fp_p, fp_q = false_pass_rate(backend)
    assert fp_p > 0, f"false_pass_rate 须 p>0（有 false positive 行）·got ({fp_p},{fp_q})"
    # mode_b_agreement 真采（返有理对 int,int·坏 corpus mode_a 全 fail→q=0→(0,1) 退化诚实·非默认 crash）
    mb_p, mb_q = mode_b_agreement(backend)
    assert isinstance(mb_p, int) and isinstance(mb_q, int), (
        "mode_b_agreement 须返有理对 (int,int)")
