"""experiments.run_weaning_train — 断奶阶段训练 W0/W2 真训练 run 入口（激活 formal_train 生产 reward 闭环）。

W0 是断奶训练首步（doc/重来_断奶阶段训练设计_2026-07-11.md W0）：激活 formal_train 生产路径的
reward 闭环·验算术域 reward>0 真流过生产路径（非 scratch 诊断脚本·非 flat_floors 放水 reward）。

**激活路径**：翻 TRAINING_MODE（算术域 vm_proof 绕 judge 不需 TEACHER_MODE·teacher=None）
→ active["reward"]=True（stages.py:180）→ eff_stage=stage（非 STAGE2 observe-only·formal_train.py:1947）
→ stage3 算术域 item 走 _run_verify_round（formal_train.py:374·_is_verify_modality 守 CODE/ARITH）
→ vm_proof_fn all-pass → reward=1 → Episode → metrics conduction_rate>0（reward>0 真流·非零 episode）。

算术域 reward 不染 β_arith（_run_verify_round 绕 episode_loop·:546 reward 不落 strength·直调 vm_proof_fn
不用 JudgeWeights·:1032）·不依赖 TEACHER_MODE（teacher=None·H2 跳过·weights 默认 (1,1,1)）。

**W2 mock POST**（weaning_phase=WEANING_POST·doc W2·§7.6 defer 的"formal_train POST 跑"）：
翻 MODE_B_CROSS_VERIFY_MODE + 复制 corpus 填 arith_source_b（Sigma(1,{p},{p}) 迭代 vs {p}*{p} 闭式·
square n² 异 shape·R6 真守）→ _run_verify_round 走 POST 分支（:578·cross_verify_pair 两路独立
execute_composes_value + rational.eq·all_agree→reward=1）→ E2 第三条件 produced_without_teacher_anchor
算术域就位（VM 执行值自锚·非教师锚·非录放层命中·weaning_e2.produced_without_teacher_anchor_arith）。
**诚实**：mock POST 非真断奶（weaning_ready 仍 False·非 rep.ready 切换·E2 整体仍 False·teacher_offline
defer W6 / probe_input_novel defer W4·W2 只验第三条件算术域就位·非 E2 过）。

**W0 诚实边界**：
- flat_floors=True 绕 stage 门控（FLOOR_GRAPH_SIZE/CAUSES/CONDUCTION/PROMOTE=0·算术域无 CAUSES 边
  跑不过 stage2 CAUSES 门控·W0 验 reward 闭环激活非门控标定·门控阈值 pre-registration defer D5·
  W0 不 claim 断奶 PASS）。
- weaning_ready=False（D3/D4/D5/E2 永 False·诚实 theatrical·E2 真墙 #493）。
- weaning_blockers 非空（诚实标注断奶阻断闸门·不静默）。
- TEACHER_MODE 不翻（算术域 vm_proof 绕 judge·TEACHER_MODE defer W8 语言域）。

**gate OFF 退化 bit-identical**：TRAINING_MODE OFF → eff_stage=STAGE2_CAUSES_ABS → stage<STAGE3
返空（formal_train.py:368）→ 零 episode → conduction_rate=0（既有 observe-only 行为·CI 默认）。
weaning_phase=WEANING_PRE（默认）→ POST 短路（arith_source_b=None + MODE_B_CROSS_VERIFY_MODE OFF）
→ W0/W1 既有行为 bit-identical。

铁律：纯整数 / bit-identical（gate OFF + weaning_phase=PRE 退化）/ gate 二分（TRAINING_MODE 入口翻·
formal_train 不翻守 CI）/ 不写死 / 只 stdlib / 几百 G 不重训（新 run_id·tmp run_dir）。
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import replace

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.training import stages
from pure_integer_ai.cognition.shared.types import WEANING_PRE, WEANING_POST
from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, FormalTrainResult
from pure_integer_ai.experiments.collection import load_arith_corpus
from pure_integer_ai.crosscut.determinism.hasher import Hasher


def _derive_probe_version(run_id: str) -> int:
    """W4 D4·run_id → 确定性探针版本号（守几百 G 不重训·新 run_id 新版本·bit-identical 可复现）。

    Hasher 纯整 hash·& 0xFFFF 钳 16 位（版本号小整数·足够区分 run_id·非语义值）。"""
    return Hasher("probe.ver").h63(run_id) & 0xFFFF


def run_weaning_arith(*, rounds_per_stage: int = 2,
                      training_mode: bool = True,
                      flat_floors: bool = True,
                      run_dir: str | None = None,
                      return_backend: bool = False,
                      weaning_phase: int = WEANING_PRE,
                      probe_holdout: int = 0,
                      probe_version: int | None = None,
                      calibrate_mode_b: bool = False,
                      simulate_offline_eval: bool = False,
                      weaning_round_series: bool = False,
                      mix_bad_corpus: bool = False,
                      corpus: list | None = None) -> FormalTrainResult | tuple:
    """W0/W2 真训练 run：算术域 fixture + TRAINING_MODE ON → reward>0 真流过生产路径。

    算术域 12 square 样本（load_arith_corpus·spec 正确·vm_proof all-pass reward=1）·stage3 走
    _run_verify_round·conduction_rate>0（reward>0 真流·非 scratch 诊断脚本·非 flat_floors 放水 reward）。

    flat_floors=True 绕 stage 门控（FLOOR_*=0·诚实标注·门控标定 defer D5）。training_mode=False
    时验 gate OFF bit-identical（observe-only·零 episode）。

    weaning_phase（W2 mock POST·默认 WEANING_PRE 守 W0/W1 bit-identical）：
      - WEANING_PRE：_run_verify_round 走 PRE 分支（spec.expected vm_proof·W0/W1 既有）。
      - WEANING_POST：走 POST cross-verify 分支（:578·Mode B 自锚·§7.6）——翻 MODE_B_CROSS_VERIFY_MODE
        + 复制 corpus 填 arith_source_b（Sigma(1,{p},{p}) 迭代 vs {p}*{p} 闭式·square n² 异 shape·R6 真守）+
        collect_episodes=True（W2 测试查 episode.reward/judge_G5_active）。
        **诚实**：mock POST 非真断奶（weaning_ready 仍 False·非 rep.ready 切换·E2 整体仍 False·
        teacher_offline defer W6 / probe_input_novel defer W4·W2 只验第三条件算术域就位）。

    probe_holdout（W4 兼容探针尾切·默认 0 不切）：
      - >0：formal_train 主入口切 corpus 末尾 N 作 held-out probe（不喂 boot/discovery/H2/stage/generate/
        base_freq 全部下游），可运行保持率诊断并证明精确签名不重复。
      - probe_version（默认 None→派生自 run_id）探针版本号·守几百 G 不重训·bit-identical 可复现。
      **V-00 纠偏**：尾切没有 dedup/provenance cluster，不能识别同源改写，不再解锁断奶 D4；
      严格 D4 必须通过 FormalTrainConfig.evaluation_plan 注入五类 split ledger。
      **probe_holdout 须 << len(corpus)=12**（切分影响 discovery held-out 池·probe_holdout=10→training 2→泛化池空）。

    calibrate_mode_b（W5 D5 Mode B 预验台账·默认 False 不跑·bit-identical）：
      - True：stage4 末跑 _run_calibration_phase（WEANING_WINDOW_ROUNDS=4 轮并行 Mode A vs B 评估·
        record_calibration 真写台账）→ mode_b_prevalidated(backend) 接非空台账真判定 → D5 过。
      - Mode A = vm_proof vs spec.expected（静态整数）·Mode B = cross_verify_pair（两路独立 build·R6 守）·
        两路真独立（异机制·非换名·测 4 用坏 corpus 验可分：expected 全错→Mode A fail ∧ Mode B agree）。
      - corpus（默认 None→load_arith_corpus·bit-identical）：自定义 corpus 供测（坏 corpus 验 Mode A/B 独立）。
        calibrate_mode_b 或 POST 时填 arith_source_b（Mode B 参树必需·_square_sigma_source_b）。
      **诚实**：D5 单闸门过非真断奶（weaning_ready 仍 False·D3/D4/E2 defer）·D5 域无关（同 D4·无须判定接口）·
      flat trend=不回升=通过（MUTABLE_MONOTONE·学树静态·FLOOR_MODE_B=500 守低平台）·stable≠correct（墙内弱）。

    simulate_offline_eval（W6 E2 模拟退场 eval·默认 False 不跑·bit-identical）：
      - True：stage4 末（weaning_check 之前·预验·解 teacher_offline 循环依赖）跑 _run_simulated_offline_eval：
        读 ctx.probe_corpus（W4 held-out 探针·须配 probe_holdout>0）→ observe 探针建学树 + cross_verify_pair
        （零教师自锚）→ 采 holdout_retention 真值 + 设 ctx.e2_eval_passed（算术域三条件 and）→ 路径 B 读 ctx → E2 算术域过。
      - 算术域 teacher=None 天然退场（架构事实·无 recording/replay/GT·无 mode 可翻）·语言域真翻 MODE_OFF defer W8。
      **诚实**：E2 单闸门过非真断奶（weaning_ready 仍 False·D1-D5 defer）·算术域 fixture 同源 trivial
      （retention 恒 1000·真泛化保持 defer W8 真语料）·teacher=None 是架构事实非"模拟退场"（语言域 defer W8）。

    返 FormalTrainResult（stages_completed / final_metrics / weaning_ready / weaning_blockers）。
    return_backend=True 时返 (result, backend)·供 W1/W2 测试查台账/episode。默认 False 守 W0 既有调用 bit-identical。
    """
    own_dir = run_dir is None
    if own_dir:
        run_dir = tempfile.mkdtemp(prefix="weaning_w0_")
    is_post = (weaning_phase == WEANING_POST)
    run_id = "w0_arith"
    config = FormalTrainConfig(
        run_dir=run_dir,
        run_id=run_id,
        rounds_per_stage=rounds_per_stage,
        weaning_phase=weaning_phase,                       # W2 mock POST 注入（默认 PRE 守 bit-identical）
        collect_episodes=is_post,                          # W2 POST 测试查 episode.reward/judge_G5_active·PRE 默认 False
        probe_holdout=probe_holdout,                       # W4 兼容尾切（只作精确内容和保持率诊断）
        probe_version=probe_version if probe_version is not None
                       else _derive_probe_version(run_id),  # W4 D4 探针版本（派生自 run_id·守几百 G 不重训）
        calibrate_mode_b=calibrate_mode_b,                 # W5 D5 Mode B 预验（默认 False 不跑·bit-identical）
        simulate_offline_eval=simulate_offline_eval,       # W6 E2 模拟退场 eval（默认 False 不跑·bit-identical）
        weaning_round_series=weaning_round_series,         # W7 断点6 per-round series（默认 False per-stage bit-identical）
    )
    if corpus is None:
        corpus = load_arith_corpus()   # 12 square 样本（spec 正确·vm_proof all-pass reward=1）
    if mix_bad_corpus:
        # W7 断点2：prepend arith_bad corpus（load_arith_bad_corpus·2 个 square + 全错 spec expected=999·反 theater
        # 既有坏 corpus）→ verify round execute(5)=25≠999→reward=0→veto=1→D2 neg_pathway_active 过。
        # 前置避免被 _split_holdout 切走作 probe。反 theater：veto=vm_proof 真执行值比对失败（非伪造）。
        # fixture 局限（手编坏 spec·非自然失败）defer W8 真语料。默认 False 不混→D2 False→bit-identical。
        from pure_integer_ai.experiments.collection import load_arith_bad_corpus
        corpus = list(load_arith_bad_corpus()) + list(corpus)
    if is_post or calibrate_mode_b or simulate_offline_eval:
        # W2 mock POST / W5 calibration / W6 eval：复制 corpus 填 arith_source_b（异 shape Sigma 迭代 vs MUL 闭式·
        # square n²·R6 真守·Mode B 参树必需）。不改 load_arith_corpus·PRE+calibration-off+eval-off 路径 bit-identical。
        corpus = [replace(item, arith_source_b=_square_sigma_source_b(item.arith_source))
                  for item in corpus]
    backend = DictBackend()

    gate_token = gates.push_gate_overrides({
        "TRAINING_MODE": training_mode,
        "MODE_B_CROSS_VERIFY_MODE": is_post,
    })
    floor_token = stages.push_stage_floor_overrides({
        "FLOOR_GRAPH_SIZE_S1": 0,
        "FLOOR_CAUSES_COV_S2": 0,
        "FLOOR_CONDUCTION_S3": 0,
        "FLOOR_PROMOTE_S4": 0,
    } if flat_floors else {})
    try:
        # W0 仅在当前上下文绕阶段门控，避免并发训练共享阈值串扰。
        result = formal_train(config, corpus, backend=backend)
    finally:
        stages.reset_stage_floor_overrides(floor_token)
        gates.reset_gate_overrides(gate_token)
    return (result, backend) if return_backend else result


def _square_sigma_source_b(arith_source: str) -> str:
    """square source `lambda {p}: {p} * {p}` → 异 shape source_b `lambda {p}: Sigma(1, {p}, {p})`。

    Sigma(1, n, n) = Σ_{i=1}^{n} n = n*n（CTRL_WHILE 迭代累加·arith_observe._build_sigma_prod :467）·
    与 {p}*{p}（MUL 闭式直线 BinOp）异 builder 代码路径·R6 真守·AGREE（同函数 n² 异 shape）。
    Sigma 机制 test_mode_b_cross_verify:112 已验活（Sigma 迭代 vs 闭式 AGREE）。
    """
    m = re.match(r"lambda\s+(\w+)\s*:\s*\1\s*\*\s*\1\s*$", arith_source)
    if not m:
        raise ValueError(f"arith_source 非 `lambda {{p}}: {{p}} * {{p}}` square 格式: {arith_source!r}")
    p = m.group(1)
    return f"lambda {p}: Sigma(1, {p}, {p})"


def _main() -> None:
    """W0 真训练 run 主入口（脚本跑·打印度量·验 reward>0 真流）。"""
    result = run_weaning_arith(rounds_per_stage=2, training_mode=True, flat_floors=True)
    m = result.final_metrics
    print(f"[W0] run_id={result.run_id}")
    print(f"[W0] stages_completed={result.stages_completed}")
    # StageMetrics 无 reward_pos/episode_count（在 RoundMetrics·metrics.py:54）·用 conduction_rate（×1000）
    print(f"[W0] conduction_rate={m.conduction_rate}（reward>0 episode 占比 ×1000）")
    print(f"[W0] weaning_ready={result.weaning_ready} "
          f"weaning_blockers={result.weaning_blockers}")
    if m.conduction_rate > 0:
        print("[W0] PASS: 算术域 reward>0 真流过 formal_train 生产路径（非 scratch）")
    else:
        print("[W0] FAIL: 零 reward 信号（生产路径未激活）")
    if not result.weaning_ready and result.weaning_blockers:
        print("[W0] 诚实: weaning_ready=False（D3/D4/D5/E2 永 False·theatrical·E2 真墙 #493）")


if __name__ == "__main__":
    _main()
