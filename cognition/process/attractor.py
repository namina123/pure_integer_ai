"""cognition.process.attractor — 模块5 attractor 动态演化（松入严留·§十三A）。

maybe_expand_attractor(c, e, pr_wrapper, subgraph_edges, workmem) -> bool
  入口（任一非循环结构判据·松·三支）：
    in_degree(c, {T_STEP, PRECEDES}) ≥ θ_conv ‖ c ∈ promoted_transition_targets
    ‖ AttractorMarker.in_basin(c)（defer·首版 tier≥TIER_PRIMARY 代理）
  ★阶段3 越界②归位（2026-07-04）：删原第四支「c 入 CAUSES 边 success_rate ≥ θ_rw」——
    CAUSES 边级 success_rate 冒充"已知/稳定"判据是漂移越界②（架构漂移审计§五）。
    effective_freq 高=通识=应终止（落 dag_path word_terminated·非 attractor entry·方向相反）。
    删支止血（value=诚实/一致性·entry 进 e_set·PR 不回流步进·非性能）。
    attractor entry 换源 effective_freq 方向一致版 = 阶段8a（3a 完整化·本阶段不做）。
  保留（全部·严）：
    x_c ≥ θ_coh（相干/在轨·PR of c w.r.t 当前 e）且 |e| < K 或 coherence(c) > min coherence(e)
  cap K 硬上界防发散·PR 强点亮只作保留不作入口（防 e 退化为全高 PR 集失指向·§十三A 核心洞察）。
  e 是每次遍历状态（热区刷新重置 e=e₀·无跨遍历累积）。

B4 逐个叠加：add_seed 调 pr_wrapper.add_seed（线性性零损失·热区小重算廉价）。
gate ATTRACTOR_MODE：default OFF（stage 纪律·OFF=e 固定 e₀ bit-identical）→ 验承重翻 ON。

纯整数（入度计数/x_c Rational/coherence Rational）/ 无墙钟（e 是遍历状态非时间累积）。
诚实边界：动态演化不判语义目标（e 据结构里程碑扩张不判"是否到达正确答案"·§十三A）·
阈值待 oracle 标定·热区外 c 的 x_c 无定义→e 限热区（近视·自然发散上界）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer import compare as cmp
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_T_STEP
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper

NodeRef = ConceptRef

# ---- oracle 标定起点（§十五 B 组初值·oracle 验后调） ----
THETA_CONV = 2          # 入度阈值（结构汇聚判据）
THETA_COH_NUM = 1       # x_c ≥ 1/1000（相干阈值·oracle 标）
THETA_COH_DEN = 1000
K_CAP = 8               # cap K 硬上界（oracle 标·防发散）
K_CAP_SOFT = 4          # 软上界（溢出踢最低相干）


def _in_degree_seq(c: NodeRef, subgraph_edges: list[dict[str, Any]]) -> int:
    """c 的序头入度（{T_STEP, PRECEDES}·非循环结构判据）。"""
    sid, lid = c
    n = 0
    for e in subgraph_edges:
        if (e["edge_type"] in (EDGE_T_STEP, EDGE_PRECEDES)
                and e["space_id_to"] == sid and e["local_id_to"] == lid):
            n += 1
    return n


def _build_in_degree_seq_map(subgraph_edges: list[dict[str, Any]]) -> dict[NodeRef, int]:
    """单遍建 {node: {T_STEP,PRECEDES} 入度} map（perf round3·2026-07-13）。

    maybe_expand_attractor 每 node 调 _in_degree_seq·cProfile n=4 实测 4673 调 × 全边扫 = 23.3s（top self·12% wall）。
    dag_path_step 建一次传 maybe_expand_attractor·O(1) 查替每调用全扫·O(#nodes×#edges)->O(#edges+#nodes)。
    bit-identical：同入度值（单遍累加 == 每次过滤计数）·只组织成 map 非改语义。
    """
    m: dict[NodeRef, int] = {}
    for e in subgraph_edges:
        if e["edge_type"] in (EDGE_T_STEP, EDGE_PRECEDES):
            key = (e["space_id_to"], e["local_id_to"])
            m[key] = m.get(key, 0) + 1
    return m


def _node_tier(c: NodeRef, backend: Any) -> int:
    """c 的 concept_node tier（TIER_TEACHER 判据代理·首版用 TIER_PRIMARY）。"""
    from pure_integer_ai.storage.node_store import NodeStore
    ns = NodeStore(backend)
    row = ns.get(c[0], c[1])
    return row["tier"] if row else 0


def maybe_expand_attractor(c: NodeRef, e: set[NodeRef],
                           pr_wrapper: A3PRWrapper,
                           subgraph_edges: list[dict[str, Any]],
                           workmem: Any, *, backend: Any = None,
                           theta_conv: int = THETA_CONV,
                           theta_coh_num: int = THETA_COH_NUM,
                           theta_coh_den: int = THETA_COH_DEN,
                           k_cap: int = K_CAP,
                           k_cap_soft: int = K_CAP_SOFT,
                           in_degree_map: dict[NodeRef, int] | None = None) -> bool:
    """attractor 松入严留扩张。返 True=已扩张（c 入 e）。

    backend：可选·供 node tier 判据（TIER_TEACHER 代理·首版 TIER_PRIMARY·defer 真档）。
    """
    assert_int(c[0], c[1], _where="maybe_expand_attractor.c")
    if c in e:
        return False
    if len(e) >= k_cap:
        return False   # cap K 硬上界
    # —— 入口（任一非循环结构判据·松） ——
    promoted = list(getattr(workmem, "promoted_transition_targets", []) or [])
    _in_deg = (in_degree_map.get(c, 0) if in_degree_map is not None
               else _in_degree_seq(c, subgraph_edges))
    entry = (
        _in_deg >= theta_conv
        or c in promoted
        # AttractorMarker.in_basin defer（首版用其余·调整标注②）
        or (backend is not None and _node_tier(c, backend) >= TIER_PRIMARY)  # TIER_TEACHER 代理
    )
    if not entry:
        return False
    # —— 保留（全部·严） ——
    x_c = pr_wrapper.seed_rank(c)
    if not cmp.cross_ge(x_c.num, x_c.den, theta_coh_num, theta_coh_den):
        return False   # 不相干/不在轨·不加
    if len(e) >= k_cap_soft:
        # 溢出踢最低相干（coherence = seed_rank）·Rational 无 __lt__（valtypes.py·纯值无算术方法）
        # -> 手动 min 用 cross_compare + 节点自然序 tiebreak（bit-identical·同 seed_rank 取最小 ConceptRef）。
        # 既有潜伏 bug·F2（PRECEDES_OI_MODE·含环节点致 active 集大）触发 k_cap_soft 溢出路径暴露。
        min_node = None
        min_rnk = None
        for n in sorted(e):
            rn = pr_wrapper.seed_rank(n)
            if min_node is None or cmp.cross_compare(rn.num, rn.den,
                                                     min_rnk.num, min_rnk.den) < 0:
                min_node = n
                min_rnk = rn
        if not cmp.cross_gt(x_c.num, x_c.den,
                            pr_wrapper.seed_rank(min_node).num,
                            pr_wrapper.seed_rank(min_node).den):
            return False   # 不够好不加（c 不优于最低）
        pr_wrapper.remove_seed(min_node)
        e.discard(min_node)
    pr_wrapper.add_seed(c)   # 扩张·B4 逐个叠加·线性性零损失
    e.add(c)
    return True
