"""training.time_seq_proof — 时序验序器（刀 A·语言域第一个 LIVE form_proof_fn·构造性检查层）。

镜像 vm_proof_fn_factory 范式（vm_proof.py:89-121）·但验 **PRECEDES DAG 无环（Kahn）**·非 COMPOSES 执行值。

机制（Option A·时序 cue 边不入图·瞬态计算）：
  - caller（_run_time_seq_verify_round）resolve segments.precedes_pairs（token index）→ ConceptRef 对
    （cue_pair_edges·A 先于 B 的跨 cue 词 shortcut）+ query intra-space EDGE_PRECEDES 边集（precedes_edges·
    role_precedes 三建边器建的 token 序 / 句间序 / struct 锚）
  - factory 闭包捕获两边集 → 内层 Kahn 合并验序 → 读 is_dag → 返 1/0/None
  - 同 vm_proof_fn_factory 捕获 expected（闭包传·非持久化边）·通道分离：reward 通道边入图 / self_proof_fn 通道闭包传

返值（守 SelfProofFn 协议·judge.py:49·三参数 output/dag_path/graph 占位·内层只用闭包边集）：
  1 = DAG 无环（verified·构造性检查通过）
  0 = 有环（mismatch·PRECEDES 结构矛盾·cycle_nodes 非空）
  None = 两边集均空（vacate·无时序边可验·诚实退场·非 pass）

**诚实分层（构造性检查 ≠ 构造性验证）**：
  - 构造性检查 ✅：Kahn 算法确定性可执行·验 PRECEDES DAG 无环（图查询+拓扑·确定性）
  - 构造性验证 ❌：cue 对 + token 序 PRECEDES 均 single-source（系统从 cue 词/token 序建·非 R6 独立源）→
    非构造性验证（须 R6 独立源升验证·Layer0 下 session）
  - 与 vm_proof 对比：vm_proof 有 R6 独立源 expected（真构造性验证·arith/code）·时序无 R6（构造性检查·语言域）·
    两者通道同（self_proof_fn 独立 episode）但验证强度不同
  - stable≠correct：PRECEDES DAG 无环 ≠ 语义时序正确（#479 墙·语言命题无执行值）

依赖方向：training(L7)→algorithm(L?) + cognition(L5) 全向下·lint 允许（镜像 vm_proof.py）。
judge(cognition L5) 不 import training·time_seq_proof_fn 在 training 接线层建（守解耦）。

铁律：纯整数（NodeRef = (int,int)·Kahn 内部纯整）/ 确定性（Kahn 队列按 (sid,lid) 自然序·bit-identical）/
      fail-loud（有环→0 非 pass·无边→None 非 theater）/ stable≠correct（DAG 无环≠对·#479）/ 永不接 reward。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.algorithm.a2_topology import kahn_topo, NodeRef
from pure_integer_ai.cognition.result.judge import SelfProofFn


def time_seq_proof_fn_factory(*, cue_pair_edges: list[tuple[NodeRef, NodeRef]],
                              precedes_edges: list[tuple[NodeRef, NodeRef]]) -> SelfProofFn:
    """造时序验序器 fn（闭包捕获 cue 对 + PRECEDES 边集·镜像 vm_proof_fn_factory :89-121）。

    cue_pair_edges : 段内时序 cue shortcut 对（A 先于 B·跨 cue 词·resolve 自 segments.precedes_pairs·
                     PRECEDES_CUE_FORWARD 提取·Option A 闭包传·不入图）。
    precedes_edges : intra-space EDGE_PRECEDES 边集（role_precedes 三建边器建·token 序 / 句间序 / struct 锚·
                     caller query backend.select 过滤本 space）。

    返 SelfProofFn(output, dag_path, graph) -> int|None：
      1（DAG 无环）/ 0（有环）/ None（两边集均空 vacate）。同 vm_proof_fn 三态。
      _run_time_seq_verify_round 直调（绕 judge·reward=1 iff r==1）·self_proof_check 不经（镜像 _run_verify_round）。

    **构造性检查层**（诚实·非构造性验证）：Kahn 确定性验 DAG 无环·非 R6 独立源验证·Layer0 下 session 升验证。
    """
    # 合并边集（factory 调用时算·闭包捕获·fn 内层只读·确定性）
    combined: list[tuple[NodeRef, NodeRef]] = list(cue_pair_edges) + list(precedes_edges)

    def time_seq_proof_fn(output: Any, dag_path: Any, graph: Any) -> int | None:
        # output/dag_path/graph 占位（守 SelfProofFn 协议三参数·内层只用闭包 combined·同 vm_proof_fn 用 dag_path.sink 不用 output）
        if not combined:
            return None   # 两边集均空 → vacate（无时序边可验·诚实退场·非 pass·非 theater）
        result = kahn_topo(combined)
        # is_dag = not cycle_nodes（a2_topology.py:42·Kahn 余留 in_degree>0 者 = 环节点）
        return 1 if result.is_dag else 0

    return time_seq_proof_fn
