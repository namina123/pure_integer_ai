"""teacher.weaning — 断奶判据（D1-D5/E2 六闸门·2026-07-02·#358 完整实现）。

weaning_check(history, *, neg_pathway_active, judge_source_independent,
              probe_set_disjoint, mode_b_prevalidated, e2_passed) -> WeaningReport

  断奶 = 六闸门全过（D1 双规定曲线方向性 + D2 负通路活跃 + D3 裁判源独立 + D4 探针隔离
        + D5 Mode B 预验 + E2 教师下线独立产出）·非布尔阈值·非单看 4 能力指标平台。

  **D1 病根纠错（最核心·2026-07-02）**：原 `_max_recent_increment` 用 `abs()` 对称平台（涨落都算
    非平台）·违 D1 方向性·是 "weaning 10/10 全 DEF_REPLAY 伪影" 病根（布尔/对称阈值在 reward 永正下
    都能被趋平退化伪造达成）。**正解**：4 能力指标保 abs 平台化语义（能力涨落都算非平台·对能力指标
    正确）·但断奶判据**额外要求两规定曲线方向性满足**（交叉积·legacy weaning_curves.py:48-72 原型）：
    ① intervention_rate 单调降∧无回升∧降至阈值以下 ② holdout_retention 后窗不回升∧达下限。
    不能只看 4 能力指标平台。

  度量七项（同源 D2 jsonl·§十二阶段4）：
    能力① oracle 导通率 / ② REALIZES 命中率 / ③ judge 自评通过率 / ④ OOV 晋升率（abs 平台化语义）
    规定曲线① intervention_rate 教师介入率（方向性·单调降）/ ② holdout_retention 探针保持率（方向性·不回升）
    M2 第5项 dependency 依赖度（退场准备度·须低）

  与收敛判据（卷三模块5）口径不同（B3 落盘）：收敛用 N=1000 episode 窗口·断奶用 window_rounds=4 runs
    窗口·两者同源度量 jsonl（D2）但窗口/判据不同不混。

铁律：纯整数（度量×1000·交叉积有理对·增量纯整·无浮点）/ 确定性（趋势判据确定性·无墙钟·history 顺序定）/
  不写死（阈值 oracle 标·初值占位）/ 不走外挂 LLM（断奶后 LLM 退场·weaning 是退场判据）。
诚实边界：断奶是行为经验性判据非结构保证（趋平退化内禀·weaning 抓趋势不保证语义正确·D 墙）/
  双曲线是趋势模型非精确拟合 / stable≠correct / 语言域 E2 未就位（defer W8）→ can_ween(语言域) 永 False·
  算术域 W7 六闸门机制接通 → can_ween(arithmetic)=True（机制接通意义·真泛化 defer W8 真语料·不伪造）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.config import gates   # floor 端到端激活率 gate-gate conjunct（审1 严重-1·call-time getattr 读生产翻转值）

# 断奶趋势窗口（B3 落盘·D1·window_rounds=4 runs·双曲线趋势确认窗口）
WEANING_WINDOW_ROUNDS = 4
# 平台化阈值（4 能力指标最近窗口内最大增量 < 此值=平台化·×1000·oracle 标）
THETA_PLATEAU = 5       # 0.5% 增量·防数值噪声误判平台
# 4 能力指标下限（防全 0 平台假断奶·各度量须达此下限才认平台·×1000）
FLOOR_CONDUCTION = 500     # oracle 导通率 ≥ 50%
FLOOR_REALIZES = 400       # REALIZES 命中率 ≥ 40%
FLOOR_JUDGE_SELF = 600     # judge 自评通过率 ≥ 60%
FLOOR_OOV_PROMOTE = 100    # OOV 晋升率 ≥ 10%
# D1 两规定曲线 + 依赖度阈值（oracle 标占位·§十一 M2 line703 同口径·×1000）
FLOOR_INTERVENTION = 200   # 教师介入率 ≤ 20%（退场准备度·曲线① 须降至阈值以下）
FLOOR_RETENTION = 700      # 探针保持率 ≥ 70%（保持率高·曲线② 须达下限）
FLOOR_DEPENDENCY = 300     # 依赖度 ≤ 30%（M2 第5项·租借比例阈值）

METRIC_CONDUCTION = "conduction_rate"
METRIC_REALIZES = "realizes_rate"
METRIC_JUDGE_SELF = "judge_self_rate"
METRIC_OOV_PROMOTE = "oov_promote_rate"

_SCALE = 1000


@dataclass
class WeaningMetrics:
    """单 run 断奶度量（七项·纯整×1000·同源 D2 jsonl）。

    能力①-④（0..1000·abs 平台化语义）/ 规定曲线①②（0..1000·方向性语义）/ 依赖度（0..1000·须低）。
    rounds 是该 run 的训练轮次序（趋势横轴）。
    """
    rounds: int
    conduction_rate: int = 0       # 能力① oracle 导通率（reward 闭环导通）
    realizes_rate: int = 0         # 能力② REALIZES 命中率（语言域=QA sink 命中·算术域=vm_proof 执行 pass 率·=REACHED_SINK∧reward>0·无 count_g5 条件）
    judge_self_rate: int = 0       # 能力③ judge 自评通过率（语言域=reward>0 占比·算术域=vm_proof 自锚·W7 断点5 count_g5_self_assess 口径纠正：G5_active 是 verify 门 active 标志非 judge 排除条件·count_g5 只影响此度量非 realizes）
    oov_promote_rate: int = 0      # 能力④ OOV 晋升率（SHADOW→PRIMARY 达率·算术域 COMPOSES 直接 PRIMARY→恒 0·W7 floor override）
    intervention_rate: int = 0     # 规定曲线① 教师介入率（question/evaluate/define 占总推理·须单调降·W7：算术域 teacher=None→恒 0·语义非 vacuous=真无教师介入·但 [0,0,0,0] 过单调降逻辑 vacuous 无判别力·承重在 D2-D5/E2）
    holdout_retention: int = 0     # 规定曲线② 留出探针保持率（D4 探针集·须不回升∧达下限·W7：算术域 fixture 同源→cross_verify 恒 agree→恒 1000·trivial 诚实边界·真泛化 defer W8）
    dependency: int = 0            # M2 第5项 依赖度（租借比例·退场准备度·须低于阈值·W7：算术域 teacher=None→恒 0·语义非 vacuous=真无教师·逻辑 vacuous 无判别力·承重在 D2-D5/E2）

    def __post_init__(self) -> None:
        assert_int(self.rounds, self.conduction_rate, self.realizes_rate,
                   self.judge_self_rate, self.oov_promote_rate,
                   self.intervention_rate, self.holdout_retention, self.dependency,
                   _where="WeaningMetrics")


@dataclass
class WeaningReport:
    """断奶判据报告（D1-D5/E2 六闸门·非布尔阈值）。"""
    ready: bool = False                                  # 六闸门全过（D1×3+D2+D3+D4+D5+E2）
    plateaued: dict[str, bool] = field(default_factory=dict)  # 4 能力指标平台化（abs 语义）
    max_increments: dict[str, int] = field(default_factory=dict)  # 4 能力指标最近窗口最大增量
    floors_met: bool = False                            # 4 能力指标达下限（防全 0 假断奶）
    intervention_decreasing: bool = False               # D1 曲线① 方向性（单调降∧无回升∧降至阈值以下）
    retention_stable: bool = False                      # D1 曲线② 方向性（后窗不回升∧达下限）
    dependency_low: bool = False                        # D1 依赖度 ≤ 阈值（M2 第5项）
    neg_pathway_active: bool = False                    # D2 负通路活跃硬前置
    judge_source_independent: bool = False              # D3 裁判源独立硬前置
    probe_set_disjoint: bool = False                    # D4 探针集隔离硬前置
    mode_b_prevalidated: bool = False                   # D5 Mode B 预验硬前置
    e2_passed: bool = False                             # E2 教师下线独立产出硬前置（最硬·默认 False·算术域 W7 path-B 读 ctx.e2_eval_passed 可 True·语言域 defer W8）
    window_rounds: int = WEANING_WINDOW_ROUNDS


def _max_recent_increment(series: list[int]) -> int:
    """4 能力指标最近 WEANING_WINDOW_ROUNDS 窗口内最大增量（abs 对称·能力指标平台化语义）。

    能力指标 abs 对称正确：能力涨落都算非平台（已平台即稳·无论微涨微落）。
    **不可用于两规定曲线**（方向性语义不同·见 _intervention_decreasing/_retention_stable）。
    series 按 rounds 升序·窗口取最后 WEANING_WINDOW_ROUNDS 个点。增量<0（退化）按绝对值计。
    """
    if len(series) < 2:
        return 0 if not series else _SCALE   # 单点/空：单点视为已饱和大增量·空=0
    window = series[-(WEANING_WINDOW_ROUNDS):]
    max_inc = 0
    for i in range(1, len(window)):
        inc = abs(window[i] - window[i - 1])
        if inc > max_inc:
            max_inc = inc
    return max_inc


def _avg_pair(window: list[int]) -> tuple[int, int]:
    """窗口均值有理对 (sum, count)·交叉积比较用·纯整禁浮点。count=0→(0,1)。"""
    if not window:
        return (0, 1)
    return (sum(window), len(window))


def _intervention_decreasing(series: list[int]) -> bool:
    """D1 曲线①·intervention_rate 须单调降∧无回升∧降至阈值以下（交叉积方向性·替 abs 对称病根）。

    单调非增=window 内所有相邻 diff ≤ 0（无回升）·是 D1 '单调降∧无回升' 的纯整实现。
    降至阈值以下=最新值 ≤ FLOOR_INTERVENTION（趋势确认非单点过阈·防高介入率平台假断奶）。
    """
    if len(series) < WEANING_WINDOW_ROUNDS:
        return False
    window = series[-WEANING_WINDOW_ROUNDS:]
    # 有回升（diff>0）→违方向性
    for i in range(1, len(window)):
        if window[i] > window[i - 1]:
            return False
    return window[-1] <= FLOOR_INTERVENTION


def _retention_stable(series: list[int]) -> bool:
    """D1 曲线②·holdout_retention 须后窗不回升∧达下限（交叉积方向性）。

    '不回升'=后窗 late half ≤ 前窗 early half（交叉积有理对比较·late_p·early_q ≤ early_p·late_q）=
    已平台化收敛（仍在升=未稳态·不可断奶）。达下限=最新值 ≥ FLOOR_RETENTION（保持率高·防低保持率
    平台假断奶·保持率掉由能力指标+负通路+floor 三重 catch）。
    """
    if len(series) < WEANING_WINDOW_ROUNDS:
        return False
    window = series[-WEANING_WINDOW_ROUNDS:]
    half = len(window) // 2
    early_p, early_q = _avg_pair(window[:half])
    late_p, late_q = _avg_pair(window[half:])
    # late ≤ early ⇔ late_p·early_q ≤ early_p·late_q（交叉积·禁浮点）
    not_rising = late_p * early_q <= early_p * late_q
    return not_rising and window[-1] >= FLOOR_RETENTION


def _floor_ok(m: WeaningMetrics, overrides: dict[str, int] | None = None) -> bool:
    """4 能力指标达下限（防全 0 平台假断奶·D1 诚实）。

    W7 断点7：overrides 允许域特化 floor override（算术域 COMPOSES 直接 PRIMARY·oov_promote 恒 0·
    FLOOR_OOV_PROMOTE 不适用→算术域传 {METRIC_OOV_PROMOTE:0}·架构事实非 vacuous）。
    默认 None 用原 floor（bit-identical·language/code 域不受影响）。
    """
    ov = overrides or {}
    return (m.conduction_rate >= ov.get(METRIC_CONDUCTION, FLOOR_CONDUCTION)
            and m.realizes_rate >= ov.get(METRIC_REALIZES, FLOOR_REALIZES)
            and m.judge_self_rate >= ov.get(METRIC_JUDGE_SELF, FLOOR_JUDGE_SELF)
            and m.oov_promote_rate >= ov.get(METRIC_OOV_PROMOTE, FLOOR_OOV_PROMOTE))


def weaning_check(
    history: list[WeaningMetrics], *,
    neg_pathway_active: bool = False,
    judge_source_independent: bool = False,
    probe_set_disjoint: bool = False,
    mode_b_prevalidated: bool = False,
    e2_passed: bool = False,
    floor_overrides: dict[str, int] | None = None,
) -> WeaningReport:
    """断奶判据（D1-D5/E2 六闸门·非布尔阈值·window_rounds=4 runs）。

    history 按 rounds 升序·至少 WEANING_WINDOW_ROUNDS 个 run 才判平台化。
    六硬前置默认 False（闸门关·诚实·须 caller 显式验证通过才置 True）。
    返 WeaningReport·ready=True = 可断奶（LLM 退场·切 Mode B self-consistency·D 墙）。

    **E2 是最硬闸门**：语言域执行条件未就位（defer W8）→ e2_passed(语言域) 永 False→can_ween(语言域) 永 False。
    算术域 W7 path-B 读 ctx.e2_eval_passed（模拟退场 eval 三条件 and）→ e2_passed 可 True→can_ween(arithmetic)=True。
    诚实：can_ween(语言域)永False 是决策层 truth（E2 defer W8·三路堵·非'统计层不能学'）·统计层持续学习就绪判据另建（连 A/C·未建 defer）·算术域机制接通意义（真泛化 defer W8 真语料）。
    """
    rep = WeaningReport()
    rep.neg_pathway_active = neg_pathway_active
    rep.judge_source_independent = judge_source_independent
    rep.probe_set_disjoint = probe_set_disjoint
    rep.mode_b_prevalidated = mode_b_prevalidated
    rep.e2_passed = e2_passed
    if not history:
        return rep
    # 按 rounds 升序（确定性·无墙钟）
    hs = sorted(history, key=lambda m: m.rounds)
    conduction = [m.conduction_rate for m in hs]
    realizes = [m.realizes_rate for m in hs]
    judge_self = [m.judge_self_rate for m in hs]
    oov = [m.oov_promote_rate for m in hs]
    intervention = [m.intervention_rate for m in hs]
    retention = [m.holdout_retention for m in hs]
    dependency = [m.dependency for m in hs]

    inc_conduction = _max_recent_increment(conduction)
    inc_realizes = _max_recent_increment(realizes)
    inc_judge = _max_recent_increment(judge_self)
    inc_oov = _max_recent_increment(oov)

    rep.max_increments = {
        METRIC_CONDUCTION: inc_conduction,
        METRIC_REALIZES: inc_realizes,
        METRIC_JUDGE_SELF: inc_judge,
        METRIC_OOV_PROMOTE: inc_oov,
    }
    # 4 能力指标平台化（abs 语义·保）
    rep.plateaued = {
        METRIC_CONDUCTION: inc_conduction < THETA_PLATEAU,
        METRIC_REALIZES: inc_realizes < THETA_PLATEAU,
        METRIC_JUDGE_SELF: inc_judge < THETA_PLATEAU,
        METRIC_OOV_PROMOTE: inc_oov < THETA_PLATEAU,
    }
    # 4 能力指标下限（最新 run·防全 0 平台假断奶）
    latest = hs[-1]
    rep.floors_met = _floor_ok(latest, floor_overrides)
    # D1 两规定曲线方向性（替 abs 对称病根）
    rep.intervention_decreasing = _intervention_decreasing(intervention)
    rep.retention_stable = _retention_stable(retention)
    # D1 依赖度 ≤ 阈值（M2 第5项）
    rep.dependency_low = latest.dependency <= FLOOR_DEPENDENCY

    # 六闸门全过才 ready
    enough_window = len(hs) >= WEANING_WINDOW_ROUNDS
    all_plateaued = all(rep.plateaued.values())
    rep.ready = (
        enough_window
        and all_plateaued
        and rep.floors_met
        and rep.intervention_decreasing
        and rep.retention_stable
        and rep.dependency_low
        and rep.neg_pathway_active
        and rep.judge_source_independent
        and rep.probe_set_disjoint
        and rep.mode_b_prevalidated
        and rep.e2_passed
    )
    return rep


# ============================================================================
# 语言域统计层断奶判定（#1143·5判据 formal gate·非 can_ween·绝不妥协）
# ============================================================================
# 承接 weaning_check（arith D1-D5/E2 六闸门）。语言域 can_ween 永 False（E2 truth 墙·
# produced_without_teacher_anchor 三路堵·决策层 accepted·正交）。统计层断奶 = **另建**
# 判定（capability_exam FOOTNOTE_WEANING / 上文 docstring L189「统计层持续学习就绪判据
# 另建·连 A/C·未建 defer」）：5判据 + 反 theater 4 锚点·**显式排除** E2(truth)/D5(Mode B
# 两路独立 builder·语言域无等价机制)/D3(裁判源独立 GT·defer #731) 三条 truth/独立源腿。
#
# 统计层断奶 ≠ can_ween（缺 truth/独立源腿）· ≠ truth（不验语义正确·stable≠correct·#479 守）
# · ≠ 机制接通 MECHANISM_LIVE（须多轮 plateau+floor+held-out+教师退场·非 once-met）。
# 是**独立第三层 verdict**·诚实标 weaker-than-can_ween（统计 robustness·非 source-independence·
# 非 truth）。excluded_gates 强制复述（防「can_ween 减腿」式 theater·镜像 capability_exam
# dead-G门 footnote 强制复述）。
#
# 反 theater 4 锚点（复用 weaning_check D1 机器·非松判据）：
#   1. 多轮 plateau（②③④ 度量窗口 max 增量 < THETA_PLATEAU·稳定收敛非 once）
#   2. floor 达下限（防全-0 假平台）
#   3. held-out 泛化 D4（probe_set_disjoint·训练/留出不重叠·非死记）
#   4. 教师退场（intervention_rate 单调降∧达下限 + dependency 低·非依赖教师锚）
# + 2 前置（precondition·非 plateau·静态 yes/no）：
#   ①编码接地 encoding_grounded（码点 concept_correspondence + 产出真词 #1040 DISPATCH_TOKEN_CHAIN）
#   ⑤跨语言汇聚 crosslingual_seeded（PURE_ALIAS 桥 P0b + 生成侧收敛 TC5）
#
# 5判据 → WeaningMetrics 映射（domain-agnostic·语言域校准口径）：
#   conduction_rate=②reward>0真流 / realizes_rate=④泛化(QA sink) / judge_self_rate=②③
#   oov_promote_rate=④泛化(OOV→词) / intervention_rate=教师退场 / holdout_retention=④泛化(held-out)
#   / dependency=教师退场（须低）。
#
# 铁律：纯整数（WeaningMetrics×1000·复用 assert_int）/ bit-identical（新函数·默认不接 formal_train·
#   gate OFF 退化既有 weaning_check 行为）/ 反 theater（4 锚点 veto·excluded_gates 明示·独立 flag 非松动）/
#   不写死（thresholds pre-registered·复用 weaning_check 既有 FLOOR_*/THETA_*·非跑完调）。

# 语言域统计层断奶显式排除的决策层腿（truth/独立源·统计层不要求·诚实标 weaker-than-can_ween）
EXCLUDED_GATE_E2 = "E2(truth·#479 produced_without_teacher_anchor 三路堵·决策层永 False)"
EXCLUDED_GATE_D5 = "D5(Mode B 两路独立 builder·语言域无 cross_verify 等价机制)"
EXCLUDED_GATE_D3 = "D3(裁判源独立 GT·defer #731 独立 LLM GT 源)"
STATISTICAL_EXCLUDED_GATES: tuple[str, ...] = (EXCLUDED_GATE_E2, EXCLUDED_GATE_D5, EXCLUDED_GATE_D3)


# 语言域 held-out 识别率下限（pre-registered·MED-2·镜像 capability_exam THRESH_RATE_PERMILLE=500·
# 跑前定·跑完不许调·防 post-hoc tuning theater）。语言 held-out = lang 识别/recognize 率 over
# held-out slice（**非 arith-only holdout_retention field**·2审 HIGH-3）。
LANG_HOLDOUT_FLOOR = 500
# floor 端到端下游激活率阈值（pre-registered·断奶 critical path 第 2 件·doc/重来_floor_端到端下游激活率_2026-07-17）：
# held-out R-skeleton cue slot 学到的对应词激活率 ≥ LANG_FLOOR_ACTIVATION=500（50%）·distractor 误激活率
# ≤ LANG_FLOOR_FALSE_POS=200（20%）。跑前定·**不许 post-hoc tuning**（反 theater·同 LANG_HOLDOUT_FLOOR 纪律）。
# 独立常量（语义不同·非复用 LANG_HOLDOUT_FLOOR：activation=下游 cue slot 激活 / holdout=识别率）。
LANG_FLOOR_ACTIVATION = 500     # cue slot 对应词激活率 ≥ 50%
LANG_FLOOR_FALSE_POS = 200      # false-positive（distractor 误激活）≤ 20%


@dataclass
class StatisticalWeaningReport:
    """语言域统计层断奶判定报告（5判据 + 5 反 theater 锚点·非 can_ween·独立 verdict·2审纠偏后）。

    statistical_ready=True = 语言域统计独立达成·**非** can_ween（缺 E2/D5/D3 truth/独立源腿·
    诚实标 weaker-than-can_ween）。

    **5 反 theater 锚点（2审 HIGH-1/2/3 纠偏后）**：
      1. plateau（②③④ 多轮稳定·防 once-met）
      2. floor（防全-0 假平台）
      3. withhold = D2 neg_pathway（**硬 gate**·防 permissive-degenerate 全-max·2审 HIGH-2：
         D2 是统计的（veto/dead-end>0·系统会拒奖）非 truth·已为语言算·**非 optional**）
      4. teacher fadeout（intervention↓ + dependency low·**GUARDED by fadeout_measured**：
         防 stub-0 vacuous 过·2审 HIGH-1：未建 intervention 聚合前 fadeout_measured=False→不过）
      5. held-out 泛化（**语言专用 heldout_generalization_permille ≥ LANG_HOLDOUT_FLOOR**·
         非 arith-only holdout_retention field·**GUARDED by heldout_measured**·2审 HIGH-3）
         ∧ D4 probe_set_disjoint
    + 2 前置：①encoding_grounded ⑤crosslingual_seeded。
    显式排除 E2(truth)/D5(Mode B)/D3(独立 GT) 三决策层腿（excluded_gates 强制复述）。
    """
    statistical_ready: bool = False
    # 锚点 1+2：plateau + floor（REAL D1 parts for language·复用 weaning_check）
    enough_window: bool = False
    plateaued: dict[str, bool] = field(default_factory=dict)
    floors_met: bool = False
    # 锚点 3：withhold D2（硬 gate·closes permissive vector）
    neg_pathway_active: bool = False
    # 锚点 4：teacher-independence（无教师=结构性独立·有教师=须 measured fadeout）
    teacher_present: bool = False                # 教师在场？（False=teacher=None·无教师可依赖=结构性独立·平行 arith vm_proof 无教师自锚）
    fadeout_measured: bool = False               # 有教师时 fadeout data 真测（2审 HIGH-1 guard·防 stub-0 vacuous）
    intervention_decreasing: bool = False
    dependency_low: bool = False
    # 锚点 5：held-out 泛化（语言专用 heldout_generalization·非 arith field·GUARDED by heldout_measured）
    heldout_measured: bool = False
    heldout_generalization_permille: int = 0
    probe_set_disjoint: bool = False        # D4 held-out split done
    # 锚点 6：floor 端到端下游激活率（反 theater 首版机制层预验·**GUARDED by floor_measured**·审1 严重-1 gate-gated conjunct）
    floor_measured: bool = False               # floor data 真测（total>0·防 stub-0 vacuous·同 fadeout/heldout measured-guard）
    floor_activation_permille: int = 0         # held-out cue slot 学到的对应词激活率（measure_floor_activation）
    floor_false_positive_permille: int = 0     # distractor 误激活率（specificity eval 层硬闸兜底）
    # 前置
    encoding_grounded: bool = False
    crosslingual_seeded: bool = False
    window_rounds: int = WEANING_WINDOW_ROUNDS
    excluded_gates: tuple[str, ...] = STATISTICAL_EXCLUDED_GATES


def language_statistical_weaning_check(
    history: list[WeaningMetrics], *,
    encoding_grounded: bool = False,
    crosslingual_seeded: bool = False,
    probe_set_disjoint: bool = False,
    neg_pathway_active: bool = False,             # D2 硬 gate（withhold·closes permissive·2审 HIGH-2）
    teacher_present: bool = False,                # 锚4：教师在场？（False=teacher=None→结构性独立·有教师→须 measured fadeout）
    fadeout_measured: bool = False,               # 锚点4 data 真测（有教师时·2审 HIGH-1 guard·防 stub-0 vacuous）
    heldout_measured: bool = False,               # 锚点5 data 真测非 arith-stub（2审 HIGH-3 guard·P2 建 lang held-out）
    heldout_generalization_permille: int = 0,     # 语言 held-out 识别率（lang_rate·若 heldout_measured）
    floor_overrides: dict[str, int] | None = None,
    floor_measured: bool = False,                 # 锚6 floor data 真测（防 stub-0·审1 严重-1 gate-gated conjunct）
    floor_activation_permille: int = 0,           # held-out cue slot 对应词激活率（measure_floor_activation·FLOOR_ACTIVATION_MODE）
    floor_false_positive_permille: int = 0,       # distractor 误激活率（specificity 硬闸）
) -> StatisticalWeaningReport:
    """语言域统计层断奶判定（5判据 + 5 反 theater 锚点·**非 can_ween**·绝不妥协 #1143·2审纠偏后）。

    history 按 rounds 升序·至少 WEANING_WINDOW_ROUNDS 个 run 才判 plateau。
    **2 对抗审纠偏后**（HIGH-1/2/3）：D2 升硬 gate（closes permissive-degenerate）·fadeout/heldout
    加 measured-guard（防 stub-0/arith-field vacuous 过）·heldout 用语言专用 heldout_generalization_permille
    （非 arith-only holdout_retention）。

    statistical_ready = plateau ∧ floor ∧ D2(neg_pathway)
                        ∧ fadeout(fadeout_measured ∧ intervention↓ ∧ dependency_low)
                        ∧ heldout(heldout_measured ∧ probe_set_disjoint
                                   ∧ heldout_generalization ≥ LANG_HOLDOUT_FLOOR)
                        ∧ ①encoding ∧ ⑤crosslingual
    显式排除 E2(truth)/D5(Mode B)/D3(独立 GT)（excluded_gates 复述·weaker-than-can_ween·非松动）。

    **反 theater（2 审核心）**：未建测量（fadeout_measured/heldout_measured=False）→ 对应锚点不过
    → statistical_ready=False（**防 stub-0 vacuous 过**·2审 HIGH-1/3）。D2 硬（防 permissive 全-max·HIGH-2）。
    **诚实边界**：statistical_ready ≠ can_ween（缺 truth/源腿）· ≠ truth（stable≠correct·#479）
    · ≠ MECHANISM_LIVE（须多轮 plateau + withhold + held-out + fadeout·非 once）。独立第三层 verdict。
    """
    rep = StatisticalWeaningReport(
        encoding_grounded=encoding_grounded,
        crosslingual_seeded=crosslingual_seeded,
        probe_set_disjoint=probe_set_disjoint,
        neg_pathway_active=neg_pathway_active,
        teacher_present=teacher_present,
        fadeout_measured=fadeout_measured,
        heldout_measured=heldout_measured,
        heldout_generalization_permille=heldout_generalization_permille,
        floor_measured=floor_measured,
        floor_activation_permille=floor_activation_permille,
        floor_false_positive_permille=floor_false_positive_permille,
    )
    if not history:
        return rep
    # 复用 weaning_check D1 机器：读 plateaued + floors_met（REAL for language·conduction/realizes/judge/oov）。
    # intervention_decreasing/dependency_low 读 WeaningMetrics 字段——语言域须 P2 建
    # 真测量（intervention 聚合 from teacher_recording）才非 stub·故 measured-guard 守（防 vacuous 过）。
    base = weaning_check(
        history,
        neg_pathway_active=neg_pathway_active,          # D2 真值（base.neg_pathway_active=此值·硬 gate）
        judge_source_independent=False,                 # D3 排除（独立 GT·defer #731）
        probe_set_disjoint=probe_set_disjoint,
        mode_b_prevalidated=False,                      # D5 排除（Mode B·语言域无等价机制）
        e2_passed=False,                                # E2 排除（truth·#479 永 False）
        floor_overrides=floor_overrides,
    )
    rep.plateaued = dict(base.plateaued)
    rep.floors_met = base.floors_met
    rep.intervention_decreasing = base.intervention_decreasing
    rep.dependency_low = base.dependency_low
    rep.window_rounds = base.window_rounds
    enough = len(history) >= WEANING_WINDOW_ROUNDS
    rep.enough_window = enough
    all_plateaued = all(base.plateaued.values()) if base.plateaued else False
    # 锚点 1+2：plateau + floor（防 once-met / 全-0 假平台）+ 锚6 floor 端到端激活（gate-gated conjunct·审1 严重-1）。
    # **bit-identical 守**：FLOOR_ACTIVATION_MODE OFF（CI default）→ floor_conjunct=True → anchor_pf 退既有 →
    #   SW2/9/16 逐字过（floor 三参数 default False/0/0·若直接 ADD 进 anchor_pf 则 gate OFF 时恒卡 False 破
    #   SW2/9/16·tests/test_statistical_weaning.py:67/119/179·审1 严重-1）；gate ON（生产·_measure_floor_pass
    #   真跑）→ floor_conjunct 走真测量守 verdict（measured-guard + activation≥阈 + fp≤阈）。
    # ★ foot-gun（post-impl 审 MEDIUM-1·piece 3 前）：formal_train statistical 路径**未传 floor 三参数**
    #   （orchestrator `_measure_floor_pass` defer piece 3·见设计档 §9）→ gate ON 时 floor_measured=False →
    #   floor_conjunct=False → statistical_ready=False（**silent veto·非 measurement 失败**）。故 piece 3
    #   orchestrator 落地前 FLOOR_ACTIVATION_MODE 须保持 OFF（env-gated·默认 OFF 守）。FC7 验此 gate-gated 行为。
    floor_conjunct = True
    if getattr(gates, "FLOOR_ACTIVATION_MODE", False):
        floor_conjunct = (floor_measured
                          and floor_activation_permille >= LANG_FLOOR_ACTIVATION
                          and floor_false_positive_permille <= LANG_FLOOR_FALSE_POS)
    anchor_pf = enough and all_plateaued and base.floors_met and floor_conjunct
    # 锚点 4 teacher-independence：无教师（teacher_present=False）→ 结构性独立（无教师可依赖·平行 arith
    #   vm_proof teacher=None 自锚·= 用户「断奶后自主学习」目标：从语料学一切·无教师）·有教师 → 须 measured
    #   fadeout（intervention↓+dependency low·2审 HIGH-1 measured-guard 守防 stub-0 vacuous）。
    # **competence 由锚 1/2/3/5（plateau/floor/D2/held-out）守**·此锚只守 teacher-dependency（无教师即无依赖·诚实非 vacuous）。
    anchor_fadeout = ((not teacher_present) or
                      (fadeout_measured and base.intervention_decreasing and base.dependency_low))
    # 锚点 5：held-out——语言专用 heldout_generalization（非 arith field）+ measured-guard（2审 HIGH-3）
    anchor_heldout = (heldout_measured and probe_set_disjoint
                      and heldout_generalization_permille >= LANG_HOLDOUT_FLOOR)
    # 统计层断奶 = 锚点 1-5 + ①⑤ 前置（显式排除 E2/D5/D3·D2 锚点3 硬 gate）
    rep.statistical_ready = (anchor_pf and neg_pathway_active
                             and anchor_fadeout and anchor_heldout
                             and encoding_grounded and crosslingual_seeded)
    return rep
