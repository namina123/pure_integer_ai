"""训练图规模、防塌和断奶阻塞诊断。"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.cognition.shared.types import Episode
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
)
from pure_integer_ai.experiments.train_context import TrainContext

def _graph_size(ctx: TrainContext) -> int:
    return ctx.backend.count("concept_node")


def _edge_count(ctx: TrainContext) -> int:
    """边总数（pre_flight ② 内存代理·概念点+边总数=图资源·防超线性膨胀）。"""
    return ctx.backend.count("edge")

def _anti_collapse_summary(
        eps: list[Episode | TypedLanguageEpisode],
        ) -> dict[str, Any]:
    """防塌三柱验收汇总（致命5：anti_collapse_verify 生产 caller·pre_flight/主循环调·非 theater）。
    P0-3 决断（doc/重来_P0决断集_修正分析十三.md §四）：anti_collapse 已接此处·credit_sink 弃
    COOCCURS reward 落点=防塌柱①有意断(见 reward_propagate 落点③)·均非缺口。

    对非空 pr_vector 的 episode 跑 anti_collapse_verify（柱①②③ falsifiable）·汇总各柱通过率。
    空 pr_vector episode 跳过（dag_path 未跑·无 PR 向量可验·层1 闭合前提缺）。
    返 {verified, total, pillar1_ok, pillar2_ok, pillar3_ok, low_variance}。
    """
    from pure_integer_ai.cognition.result.anti_collapse import (
        anti_collapse_verify, integer_variance, THETA_VARIANCE)
    verified = 0
    p1 = p2 = p3 = low_var = 0
    legacy = [ep for ep in eps if isinstance(ep, Episode)]
    for ep in legacy:
        if not ep.pr_vector:
            continue   # 无 PR 向量·跳过（层1 闭合前提缺·不验）
        rep = anti_collapse_verify(ep)
        verified += 1
        p1 += int(rep.pillar1_ok)
        p2 += int(rep.pillar2_ok)
        p3 += int(rep.pillar3_ok)
        if integer_variance(ep.pr_vector) < THETA_VARIANCE:
            low_var += 1
    return {"verified": verified, "total": len(legacy),
            "pillar1_ok": p1, "pillar2_ok": p2, "pillar3_ok": p3,
            "low_variance": low_var}


def _weaning_blockers(rep: Any) -> list[str]:
    """D1-D5/E2 未过闸门清单（诚实标注·不静默·§十一 #4-bis）。

    weaning_ready=False 时列出未过闸门·进训练日志（run_id 下）·驱动诊断与续训决策。
    """
    blockers: list[str] = []
    # W7 修：not rep.plateaued 是 dict truthiness（非空=True→not=False·恒漏 plateau 失败）·须 all(values)。
    # 既有 bug：D1 plateau 失败但 floors_met 过时 blockers 不标（不诚实）·W7 round_series=False 场景暴露。
    if not all(rep.plateaued.values()) or not rep.floors_met:
        blockers.append("D1_capability_plateau")       # 4 能力指标平台/下限
    if not rep.intervention_decreasing:
        blockers.append("D1_intervention_decreasing")  # 曲线① 方向性
    if not rep.retention_stable:
        blockers.append("D1_retention_stable")         # 曲线② 方向性
    if not rep.dependency_low:
        blockers.append("D1_dependency_low")           # 依赖度
    if not rep.neg_pathway_active:
        blockers.append("D2_neg_pathway_active")       # 负通路活跃
    if not rep.judge_source_independent:
        blockers.append("D3_judge_source_independent")  # 裁判源独立
    if not rep.probe_set_disjoint:
        blockers.append("D4_probe_set_disjoint")       # 探针集隔离
    if not rep.mode_b_prevalidated:
        blockers.append("D5_mode_b_prevalidated")      # Mode B 预验
    if not rep.e2_passed:
        blockers.append("E2_independent_production")   # 教师下线独立产出（最硬·当前永未过）
    return blockers


def _causes_coverage(ctx: TrainContext) -> int:
    """CAUSES 覆盖率（有 CAUSES 出边节点占比 ×1000·阶段2 门控）。"""
    from pure_integer_ai.storage.edge_types import EDGE_CAUSES
    nodes = ctx.backend.select("concept_node", where=None)
    if not nodes:
        return 0
    causes_from = {(r["space_id_from"], r["local_id_from"])
                   for r in ctx.backend.select("edge", where={"edge_type": EDGE_CAUSES})}
    covered = sum(1 for n in nodes
                  if (n["space_id"], n["local_id"]) in causes_from)
    return (covered * 1000) // len(nodes)

__all__ = [
    "_anti_collapse_summary",
    "_causes_coverage",
    "_edge_count",
    "_graph_size",
    "_weaning_blockers",
]
