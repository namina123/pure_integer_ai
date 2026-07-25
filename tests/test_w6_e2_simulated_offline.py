"""tests/test_w6_e2_simulated_offline — W6 E2 模拟退场验证（解 teacher_offline 循环依赖·墙内工程）。

验（doc/重来_断奶阶段训练设计_2026-07-11.md W6）：
1. e2e：probe_holdout=2 + simulate_offline_eval=True → _run_simulated_offline_eval 真 caller → E2 算术域过
   + holdout_retention 真值 + weaning_ready 仍 False（D1/D2/D5 defer）
   **W7 后**：同 simulate_offline_eval flag 激活 D3 路径 B -> D3 亦过（W7 correctness·非 W6 验证目标）·
   此 test 仍聚焦 E2 单闸门；旧 probe_holdout 只够运行诊断，不再通过严格 D4。
2. bit-identical：默认 simulate_offline_eval=False → E2 blocker + holdout_retention=0（既有行为不变·零翻）
3. e2_execution_ready_arith 单元：三条件 and + 通用 e2_execution_ready() 仍 False（语言域 defer W8）
4. 反 theater①：教师真不参与（cross_verify_pair 签名零 teacher + 算术域 teacher=None 架构事实
   + teacher_offline=False → E2 不过）
5. holdout_retention 真度量 + trivial 诚实边界 + 无探针 no-op

**W6 = stage4 末模拟退场 eval 子阶段 + e2_execution_ready_arith 算术域判定接口 + E2 路径 B 域特化分支**。
general-purpose agent 核证 E2 断点（e2_execution_ready() 硬编 False / 路径 B 读硬编函数·六 D-check 唯一 /
ctx.holdout_retention 无写入点永 0 / ctx.probe_corpus 零 reader / 循环依赖 teacher 退场在 ready 之后）+
Plan agent 设计解读 A（算术域聚焦·teacher=None 天然退场·语言域翻 MODE_OFF defer W8）。

**关键**：E2 路径 B 域特化分支（teacher=None 读 ctx.e2_eval_passed / else 读 e2_execution_ready() defer W8）·
解 teacher_offline 循环依赖（eval 在 weaning_ready 之前·预验非后验·ready 读 eval 结果非驱动 eval）。

**诚实边界**：E2 单闸门过非真断奶（weaning_ready 仍 False·D1-D5 defer）·算术域 teacher=None 是架构事实
非"模拟退场"（语言域真翻 MODE_OFF defer W8）·fixture 同源 trivial（retention 恒 1000·真泛化 defer W8）。
"""
from __future__ import annotations

import inspect

from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith
from pure_integer_ai.teacher.weaning_e2 import (
    e2_execution_ready_arith, e2_execution_ready,
)
from pure_integer_ai.training.mode_b_cross_verify import cross_verify_pair


def test_w6_e2_eval_e2e(tmp_path):
    """★e2e：probe_holdout=2 + simulate_offline_eval=True → E2 算术域过 + holdout_retention 真值 + weaning_ready False。

    run_weaning_arith(probe_holdout=2, simulate_offline_eval=True) → formal_train stage4 末跑
    _run_simulated_offline_eval（读 ctx.probe_corpus 末 2 held-out square 样本·observe 建学树 +
    cross_verify_pair 零教师自锚）→ ctx.e2_eval_passed=True（三条件 and）→ 路径 B 读 ctx → e2_passed=True
    → E2 blocker 消失。holdout_retention 真值（cross_verify 通过率×1000·算术 fixture 正确→1000）。
    weaning_ready 仍 False（D1/D2/D5 defer·W7 后 D3 同 flag 路径 B 亦过·D4 probe 亦过·
    此 test 聚焦 E2 单闸门·诚实非真断奶）。
    """
    result, _backend = run_weaning_arith(
        probe_holdout=2, simulate_offline_eval=True,
        return_backend=True, run_dir=str(tmp_path / "w6e2e"))
    # E2 过（模拟退场 eval 三条件 and → ctx.e2_eval_passed=True → 路径 B 读 True）
    assert "E2_independent_production" not in result.weaning_blockers, (
        f"simulate_offline_eval=True 须 E2 过（ctx.e2_eval_passed=True）·blockers={result.weaning_blockers}")
    # 旧尾切足以运行诊断，但缺 provenance，V-00 严格 D4 必须继续阻塞。
    assert "D4_probe_set_disjoint" in result.weaning_blockers, (
        f"旧尾切缺 provenance，必须保留 D4 blocker·got {result.weaning_blockers}")
    # holdout_retention 真值（非 0·cross_verify 通过率×1000·W4 defer 的 track 首采真值）
    assert result.holdout_retention > 0, (
        f"eval 采的 holdout_retention 须 >0（cross_verify 通过率×1000·非硬编 0）·got {result.holdout_retention}")
    # weaning_ready 仍 False（D1/D2/D3/D5 defer·只 E2 单闸门过·诚实非真断奶）
    assert not result.weaning_ready, (
        "weaning_ready 仍 False（D1/D2/D3/D5 defer·W6 只过 E2·诚实非真断奶）")


def test_w6_bit_identical_default_off(tmp_path):
    """★bit-identical：默认 simulate_offline_eval=False → E2 blocker + holdout_retention=0（既有行为不变·零翻）。

    默认 simulate_offline_eval=False → stage4 块 `if config.simulate_offline_eval:` 不执行 →
    _run_simulated_offline_eval 不调 → ctx.e2_eval_passed 默认 False → 路径 B 读 False → E2 blocker。
    ctx.holdout_retention 默认 0（eval 未采真值）。W0-W5 既有行为 bit-identical。
    """
    result = run_weaning_arith(run_dir=str(tmp_path / "w6bit"))   # 默认 simulate_offline_eval=False
    # E2 blocker（默认 off → ctx.e2_eval_passed=False → 路径 B 读 False·同既有·零翻）
    assert "E2_independent_production" in result.weaning_blockers, (
        f"默认 simulate_offline_eval=False 须 E2 blocker（bit-identical）·blockers={result.weaning_blockers}")
    assert not result.weaning_ready
    # holdout_retention 默认 0（eval 未跑·未采真值·bit-identical）
    assert result.holdout_retention == 0, (
        "默认 off → holdout_retention 须 0（eval 未采真值·bit-identical）")
    # W0-W5 既有 reward 闭环不受 W6 影响
    assert result.final_metrics.conduction_rate > 0, "W0 reward 闭环须不受 W6 影响"


def test_w6_e2_execution_ready_arith_unit():
    """★e2_execution_ready_arith 算术域判定接口（三条件 and·镜像 W3 judge_source_independent_arith 范式）。

    全 True → True（算术域三条件就位）·任一 False → False。
    通用 e2_execution_ready()（无参）仍 False（语言域 defer W8·既有 test_weaning_gates 覆盖）。
    """
    # 全 True → True（算术域三条件就位）
    assert e2_execution_ready_arith(
        teacher_offline=True, probe_input_novel=True,
        produced_without_teacher_anchor=True) is True
    # 任一 False → False
    assert e2_execution_ready_arith(
        teacher_offline=False, probe_input_novel=True,
        produced_without_teacher_anchor=True) is False
    assert e2_execution_ready_arith(
        teacher_offline=True, probe_input_novel=False,
        produced_without_teacher_anchor=True) is False
    assert e2_execution_ready_arith(
        teacher_offline=True, probe_input_novel=True,
        produced_without_teacher_anchor=False) is False
    # 通用 e2_execution_ready()（无参）仍 False（语言域 defer W8·不受 W6 新增算术域接口影响）
    assert e2_execution_ready() is False


def test_w6_teacher_not_participating_anti_theater(tmp_path):
    """★反 theater①：教师真不参与产出（cross_verify_pair 签名零 teacher + 算术域 teacher=None 架构事实）。

    算术域 teacher=None → teacher_offline 恒 True（eval guard 守 teacher is None）。cross_verify_pair 是纯函数
    （签名零 teacher 参数·只 execute_composes_value + rational.eq·mode_b_cross_verify.py:64-97·机制保证非教师锚）。
    探针产出 = VM 执行值（execute_composes_value·非教师 GT·非录放层命中）。
    E2 过证明 teacher_offline=True（ctx.teacher is None·反 theater：若教师参与则 teacher_offline=False→E2 不过）。

    **反 theater 三层守**：
    ① 签名层：cross_verify_pair 参数零 teacher（inspect.signature·精确非脆弱全源码 grep）
    ② 架构层：算术域 run_weaning_arith 不传 teacher → ctx.teacher is None（teacher_offline 恒 True）
    ③ 语义层：teacher_offline=False → e2_execution_ready_arith 返 False（教师参与则 E2 不过·E2 守）
    """
    # ① 签名层：cross_verify_pair 参数零 teacher（机制保证·非教师锚·反 theater①）
    sig_params = list(inspect.signature(cross_verify_pair).parameters)
    assert sig_params == ["graph", "root_a", "root_b", "probes"], (
        f"cross_verify_pair 签名须零 teacher 参数（纯函数·只 graph/root_a/root_b/probes）·got {sig_params}")
    # ② e2e 验证：E2 过证明 teacher_offline=True（ctx.teacher is None·算术域天然退场·架构事实）
    result, _backend = run_weaning_arith(
        probe_holdout=2, simulate_offline_eval=True,
        return_backend=True, run_dir=str(tmp_path / "w6anti"))
    assert "E2_independent_production" not in result.weaning_blockers, (
        "E2 过证明 teacher_offline=True（ctx.teacher is None·反 theater①：若教师参与→teacher_offline=False→E2 不过）")
    # ③ 语义层：teacher_offline=False → e2_execution_ready_arith 返 False（教师参与则 E2 不过·反 theater 守）
    assert e2_execution_ready_arith(
        teacher_offline=False, probe_input_novel=True,
        produced_without_teacher_anchor=True) is False, (
        "teacher_offline=False → E2 不过（反 theater：教师参与产出则非独立·E2 守）")


def test_w6_holdout_retention_real_measurement_no_probe(tmp_path):
    """★holdout_retention 真度量 + trivial 诚实边界 + 无探针 no-op。

    ① 无探针（probe_holdout=0 + simulate_offline_eval=True）→ eval no-op → E2 blocker + holdout_retention=0
       （无 held-out 探针→eval 无候选→ctx.e2_eval_passed 默认 False·诚实·不伪造通过）。
    ② 有探针（probe_holdout=2 + simulate_offline_eval=True）→ holdout_retention 真值 >0（非硬编 0·W4 defer 首采）。
    ③ trivial 诚实边界：算术 fixture probe/training 同源（都 square n²）→ cross_verify 恒 agree →
       retention 恒 1000（fixture 局限·非机制 bug·真泛化保持 defer W8 真语料）。
    """
    # ① 无探针 → eval no-op → E2 blocker + holdout_retention=0
    r_no_probe = run_weaning_arith(
        simulate_offline_eval=True, probe_holdout=0,
        run_dir=str(tmp_path / "w6noprobe"))
    assert "E2_independent_production" in r_no_probe.weaning_blockers, (
        "probe_holdout=0 → 无 held-out 探针 → eval no-op → E2 blocker（诚实·不伪造）")
    assert r_no_probe.holdout_retention == 0, "无探针 → holdout_retention=0（eval 未采真值）"

    # ② 有探针 → holdout_retention 真值 >0（非硬编 0·W4 defer 首采）
    r_probe, _backend = run_weaning_arith(
        simulate_offline_eval=True, probe_holdout=2,
        return_backend=True, run_dir=str(tmp_path / "w6probe"))
    assert r_probe.holdout_retention > 0, (
        f"有探针 → holdout_retention 真值须 >0（cross_verify 通过率×1000）·got {r_probe.holdout_retention}")

    # ③ trivial 诚实边界：算术 fixture 同源 → retention=1000（恒 agree·fixture 局限·非机制 bug）
    assert r_probe.holdout_retention == 1000, (
        f"算术 fixture probe/training 同源（都 square n²）→ cross_verify 恒 agree → retention=1000·"
        f"trivial 诚实边界（真泛化保持 defer W8 真语料）·got {r_probe.holdout_retention}")
