"""cognition.result.anti_collapse — 模块4 防塌三柱闭环验收（§十四子问题3）。

anti_collapse_verify(episode) -> CollapseReport
  防塌三柱缺一即塌：
    ①结构化 judge（非自媚）：G4/G2p/G3a/G3b/G5 门否决·structure_overlap/coverage 进 live judge·deterministic
    ②真负通路：死路/结构失败→reward<0 或 reward=0→tn++ not sn++·无负通路=命门确定塌
    ③探索压力（③最小版进首版·完整版 defer）：PR bias 方差<阈（趋平）时注入 seeded 探索对抗单调只增趋平退化
  ①②是负通路两个半边非独立两柱（①judge veto reward=0 + ②步进死路 reward<0 都 failure→tn++）。
  防塌靠 failure→tn++ 非靠负值（judge 公式非负·负值只来自步进死路）。
  三柱 presuppose 层1 闭合（衔接四条件·非可选·首版必验真闭合否则三柱 moot）。

  ③是唯一在"无显式失败"时 active 的柱（①②在趋平时 dormant 挡不住·③ proactive probe）。
  seeded 探索~50-100 行纯整数确定性（seed=hash(input,run_id,ref)·非墙钟随机·bit-identical）。
  B2 落盘：注入=新种子（A3_add_seed·线性叠加零损失·非权重扰动·扰动破 H4 strength×rate 语义）。

gate EXPLORATION_MODE：闭环核心默认 ON（③最小版进首版）·OFF=方差够时不注入 bit-identical（柱③仍在）。
铁律：纯整数（方差 ×1000 缩放/seed 确定性 hash）/ 无墙钟（seed=输入 hash+run_id+ref 非墙钟随机）。
诚实边界：语义正确性是 #479 W2 truth 墙（**非 D 物理接地墙**·三柱只能推迟/检测不能消除·§十四）/ 收敛是行为经验性非结构保证。
defer：③完整版（C5 资产 creative_traverse+好奇窗/震撼窗·质变正交能力 seeded 扰动演化不出）。
"""
from __future__ import annotations

from functools import cmp_to_key
from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.integer import compare as cmp
from pure_integer_ai.crosscut.integer.rational import ZERO, add, sub, mul, make
from pure_integer_ai.crosscut.integer.valtypes import Rational
from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import Episode, CollapseReport, ConceptRef
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper

# 方差 ×1000 缩放（纯整无浮点·§十四低③）
VARIANCE_SCALE = 1000
# 趋平阈值（oracle 标·方差<此值=趋平信号·注入 seeded 探索）
THETA_VARIANCE = 1

_SEED_HASHER = Hasher("pure_integer_ai.anti_collapse.seed.v1")
# seeded 探索同值 tiebreak hash（确定性·seed 影响·非墙钟随机）
_TIE_HASHER = Hasher("pure_integer_ai.explore.tie.v1")


def integer_variance(pr_vector: dict[ConceptRef, Rational]) -> int:
    """PR bias 方差（纯整 ×1000·Σ(xᵢ-x̄)²/n·x̄ 整数均值·全 Rational 闭运算零浮点）。

    空向量 → 0（无方差·趋平判据由调用方结合柱③处理）。
    """
    if not pr_vector:
        return 0
    vals = list(pr_vector.values())
    n = len(vals)
    # x̄ = Σx / n
    total = ZERO
    for v in vals:
        total = add(total, v)
    mean = make(total.num, total.den * n)   # total / n
    # Σ(xᵢ - x̄)²
    acc = ZERO
    for v in vals:
        d = sub(v, mean)
        acc = add(acc, mul(d, d))
    var = make(acc.num, acc.den * n)   # Σ(...) / n
    # ×1000 缩放取整（纯整·截断）
    return (var.num * VARIANCE_SCALE) // var.den


def deterministic_seed(run_id: int, ref: ConceptRef | None) -> int:
    """确定性 seed = hash(run_id, ref)·非墙钟随机·bit-identical（§十四低③）。

    公开（dag_path 柱③ proactive 注入复用同一 seed 家族·anti_collapse_verify 现场 注入亦用）。
    """
    r = ref if ref is not None else (-1, -1)
    return _SEED_HASHER.h63((run_id, r[0], r[1]))


def inject_seeded_exploration(pr_wrapper: A3PRWrapper, e_set: set[ConceptRef],
                              seed: int) -> ConceptRef | None:
    """seeded 探索：加低 PR 候选进 e（A3_add_seed·B2 新种子·线性叠加零损失）。

    候选选 = 热区内 x_c 最低且不在 e 中的节点（确定性·seed 影响 tiebreak·非墙钟随机）。
    扰动权重会破 H4 strength×rate 语义→用新种子非权重扰动（B2 落盘）。
    返加入的节点 / None（无候选·热区空）。
    """
    assert_int(seed, _where="inject_seeded_exploration.seed")
    candidates = [n for n in pr_wrapper.matrix.nodes if n not in e_set]
    if not candidates:
        return None
    # 最低 x_c 选 min——**值序**（cross_compare 交叉积·非 (num,den) 元组序）。
    # #995 修：旧版 rank_key=(x.num,x.den,tie) 元组序非值序·Rational(2,5)=0.4 的元组 (2,5,...) >
    # Rational(1,2)=0.5 的元组 (1,2,...)（因 2>1）·min 误选高值节点（应选 0.4 却选 0.5）。
    # cross_compare 零误差比序（镜像 attractor.py:118-125 min-by-seed_rank 范式）。
    def rank_cmp(n_a: ConceptRef, n_b: ConceptRef) -> int:
        x_a = pr_wrapper.seed_rank(n_a)
        x_b = pr_wrapper.seed_rank(n_b)
        c = cmp.cross_compare(x_a.num, x_a.den, x_b.num, x_b.den)
        if c != 0:
            return c   # 值序：x_a 值 < x_b 值 → -1（min 取最低值 x_c）
        # 同值 tiebreak：seed-hash（确定性·非墙钟随机·异 node 异 hash·总序）
        tie_a = _TIE_HASHER.h63((seed, n_a[0], n_a[1]))
        tie_b = _TIE_HASHER.h63((seed, n_b[0], n_b[1]))
        return (tie_a > tie_b) - (tie_a < tie_b)
    target = min(candidates, key=cmp_to_key(rank_cmp))
    pr_wrapper.add_seed(target)   # B4 逐个叠加·线性性零损失
    e_set.add(target)
    return target


def linkage_four_conditions_hold(episode: Episode) -> bool:
    """层1 闭合验收（衔接四条件·三柱 presuppose·首版必验真闭合）。

    四条件（卷二模块7/8/4 已守·此处 Episode 层 sanity 验收）：
      ① reward 落点⊇{CAUSES}（卷二模块8 只走 CAUSES 头·构造守·Episode 验符号契约）
      ② PR 邻接⊇遍历边集（卷二模块1/2/7 守·Episode 验 pr_vector 非空=PR 装配活跃）
      ③ PR 用 strength×rate（卷二模块7 effective_weight 守·构造守）
      ④ 无回声兜底主导（卷二模块4 步进产真实结构 path·Episode 验 reward 符号契约）
    首版 Episode 层验：pr_vector 非空（②PR 装配）∧ reward 符号契约（①④·reward<0 iff DEAD_END）。
    """
    pr_ok = bool(episode.pr_vector)   # ② PR 装配活跃
    # ①④ reward 符号契约：reward<0 iff terminal==DEAD_END（步进死路产负·judge 产≥0）
    from pure_integer_ai.cognition.shared.types import TERMINAL_DEAD_END, REWARD_DEAD_END
    if episode.terminal == TERMINAL_DEAD_END:
        symbol_ok = episode.reward == REWARD_DEAD_END
    else:
        symbol_ok = episode.reward >= 0
    return pr_ok and symbol_ok


def anti_collapse_verify(episode: Episode, *,
                         pr_wrapper: A3PRWrapper | None = None,
                         e_set: set[ConceptRef] | None = None) -> CollapseReport:
    """防塌三柱闭环验收。返 CollapseReport。

    pr_wrapper/e_set 可选：提供则柱③方差趋平时真注入 seeded 探索（proactive probe）；
    不提供则柱③ 仅报告（方差够=柱在·方差低=需注入由 caller 接线）。
    """
    assert_no_float(episode.reward, _where="anti_collapse_verify.reward")
    report = CollapseReport()

    # —— 柱① 结构 judge 非自媚（任一 G veto active=judge 在工作·D1 扩 5 门） ——
    report.pillar1_ok = (
        episode.judge_G4_active or episode.judge_G2p_active
        or episode.judge_G3a_active or episode.judge_G3b_active
        or episode.judge_G5_active
    )

    # —— 柱② 真负通路（judge veto reward=0 + 步进死路 reward<0 都 failure→tn++） ——
    neg_pathway_active = (episode.judge_veto_count > 0
                         or episode.dead_end_count > 0)
    report.pillar2_ok = neg_pathway_active
    report.failure_count = episode.judge_veto_count + episode.dead_end_count
    report.neg_reward_count = episode.dead_end_count   # 负值只来自步进死路

    # —— 柱③ 探索压力（③最小版·方差够=自然多样性 / 方差趋平=须 proactive 注入·可证伪） ——
    # stub#1 修：旧版三分支全 pillar3_ok=True（无 False 路径=theater）·今 falsifiable：
    #   方差够 → dormant OK / 方差趋平 → 须注入缓解（注入了 OK·没注入=失守 False）。
    pr_var = integer_variance(episode.pr_vector)
    if pr_var >= THETA_VARIANCE:
        report.pillar3_ok = True   # 方差够·自然探索多样性在（dormant OK·③不必活跃）
    else:
        # 趋平·须 proactive 注入缓解（柱③ falsifiable：注入=缓解 OK·没注入=失守 False）
        injected = bool(getattr(episode, "exploration_injected", False))   # dag_path 内已注入
        if (not injected and gates.EXPLORATION_MODE
                and pr_wrapper is not None and e_set is not None):
            # 现场 注入（caller 直传 pr_wrapper 路径·dag_path 内注入的补充·B2 新种子线性叠加零损失）
            seed = deterministic_seed(episode.run_id, episode.ref)
            injected = inject_seeded_exploration(pr_wrapper, e_set, seed) is not None
        report.pillar3_ok = injected

    # —— 层1 闭合验收（衔接四条件·三柱 presuppose·首版必验真闭合） ——
    assert linkage_four_conditions_hold(episode), (
        "层1 未闭合：衔接四条件破·三柱 moot（pr_vector 空 或 reward 符号契约破）"
    )
    return report
