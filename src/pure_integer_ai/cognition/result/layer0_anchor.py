"""cognition.result.layer0_anchor — Layer0 外部锚门（停止决策守门·防 cue 自产边 theater）。

构造性检查 ≠ 构造性验证（分层墙认知更正 §七纪律核心）：
  构造性检查：检查机制构造性（Kahn DAG 无环 / execute_composes_value 确定性可执行）。
  构造性验证：检查机制构造性 **AND** 被验数据来自 R6 独立源（非系统自产）。
  两者齐才真构造性验证。single-source self-check = theater（违 vm_proof.py:9 R6 独立源原则）。

外部锚门（分层墙 §八b "找到就停纪律"·doc 提议 code 零·本模块落 code）：
  停止决策前查依据至少一路外部来源·全自产不准停。

  **反 theater 执行点（已落·非纸面闭合）**：聚合计数"构造性验证"经 count_layer0.external_verified
  （调 is_constructive_verification·reward>0 AND EXTERNAL）·SELF_PRODUCED reward>0（自产检查通过）
  **不计** external_verified → capability_exam.layer0_attribution 生产消费（project_layer0）·
  任何读 external_verified 判"构造性验证学到"的聚合器自动排除自产 theater。

  **停止决策路径接入 defer（诚实边界·2 审 P1-2）**：external_anchor_satisfied 是 per-episode 判据·
  count_layer0 消费（anchor_satisfied/violated 桶·非孤儿）·但 weaning_check / convergence 停止决策
  路径**尚未读 verify_source**（断奶现 E2 恒 False 不可达·接入须断奶路径真活·非对死路径执行·defer
  到断奶路径 live 时）。即"全自产不准停"当前是**聚合汇报**（capability_exam 报 anchor_violated）
  非断奶 veto 执行。

三函数（count_layer0 消费 external_anchor_satisfied + is_constructive_verification·生产消费非孤儿）：
  external_anchor_satisfied(ep) — per-episode 外部锚门判据（False iff SELF_PRODUCED）。
  is_constructive_verification(ep) — 单 episode 是否真构造性验证（reward>0 AND EXTERNAL·反 theater）。
  count_layer0(episodes) — 汇总 dict（capability_exam.project_layer0 消费·纯整数·key 固定序）。

诚实边界：
  - 本门只**标记 + 守门**·**不提供 R6 独立源**（R6 加固属刀G ConceptNet Causes loader 移植 / 时序升构造性验证·defer）。
  - stable≠correct：外部锚门满足（有 R6 源）≠ 语义正确（#479 墙·语言命题无执行值）。
  - 不碰 reward 通道（CAUSES-only assert·本门只读 Episode.verify_source / reward·不写）。
  - 标记在 Episode（非边）·不碰 #355 EDGE_PRECEDES epistemic_origin·刀A Option A 时序边不入图继续成立。
单向依赖：cognition.result → cognition.shared.types 向下 + crosscut.guards 向下（同 judge.py）。
"""
from __future__ import annotations

from typing import Any, Iterable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.types import (
    VERIFY_SOURCE_NONE, VERIFY_SOURCE_EXTERNAL, VERIFY_SOURCE_SELF_PRODUCED,
)


def external_anchor_satisfied(episode: Any) -> bool:
    """per-episode 外部锚门判据：verify episode 是否有外部 R6 来源（False iff SELF_PRODUCED）。

    返 True（NONE 放行 / EXTERNAL 满足）：非 SELF_PRODUCED 即过锚门判据。
    返 False（SELF_PRODUCED·全自产）：自产检查 episode（time_seq cue 对+token 序）·违外部锚门。

    **消费**：count_layer0 消费此函数填 anchor_satisfied/violated 桶（capability_exam 报）。
    **诚实边界（2 审 P1-2）**：本函数是 per-episode 判据·聚合汇报用·weaning_check 停止决策路径
    接入 defer（断奶现 E2 恒 False 不可达）·"全自产不准停"当前是汇报（anchor_violated 计数）非
    断奶 veto 执行·反 theater 真执行点在 external_verified（排除 SELF_PRODUCED·is_constructive_verification）。

    getattr 防 fake Episode（SimpleNamespace / 非 dataclass·镜像 capability_exam._door_vetoed getattr 范式）。
    """
    src = getattr(episode, "verify_source", VERIFY_SOURCE_NONE)
    if src == VERIFY_SOURCE_SELF_PRODUCED:
        return False   # 全自产·不准停（防 cue 自产边 theater）
    return True         # NONE（放行）/ EXTERNAL（满足）


def is_constructive_verification(episode: Any) -> bool:
    """是否真构造性验证（构造性检查 + R6 外部源·两齐）= reward>0 AND EXTERNAL。

    **反 theater 锚点**：统计"构造性验证数"须用此·**非 reward>0**——reward>0 含
    SELF_PRODUCED 检查通过（time_seq Kahn 无环 reward=1）·那是构造性检查非验证·计入即 theater。
    reward==0（验证失败 / 检查未过）→ False（无论来源）。
    """
    reward = getattr(episode, "reward", 0)
    src = getattr(episode, "verify_source", VERIFY_SOURCE_NONE)
    return reward > 0 and src == VERIFY_SOURCE_EXTERNAL


def count_layer0(episodes: Iterable[Any]) -> dict[str, int]:
    """汇总 episodes → Layer0 分类计数 dict（capability_exam.project_layer0 消费·纯整数）。

    返（key 固定序·bit-identical）：
      external_verified          : 真构造性验证数（调 is_constructive_verification·reward>0 AND EXTERNAL·
                                   **反 theater 执行点**·SELF_PRODUCED 不计）
      self_produced_check_passed : 构造性检查通过·非验证（reward>0 AND SELF_PRODUCED·time_seq 无环）
      self_produced_check_failed : 构造性检查未过（reward==0 AND SELF_PRODUCED·time_seq 有环）
      anchor_satisfied           : 非 SELF_PRODUCED episode 数（调 external_anchor_satisfied·NONE 放行 +
                                   EXTERNAL 满足·**停止决策实际门是 external_verified>0**·anchor_satisfied 含 NONE 非可直接判停）
      anchor_violated            : SELF_PRODUCED episode 数（违外部锚门·全自产·2 审 P1-2 defer 接 weaning_check）
      total                      : episode 总数（分母·诊断用）

    诚实：anchor_violated = SELF_PRODUCED 总数（不问 reward·全自产即违锚门·无论检查过否不准停）。
    恒等：anchor_satisfied + anchor_violated == total（每 episode 落一桶）。
    """
    external_verified = 0
    self_produced_check_passed = 0
    self_produced_check_failed = 0
    anchor_satisfied = 0
    anchor_violated = 0
    total = 0
    for ep in episodes:
        total += 1
        reward = getattr(ep, "reward", 0)
        src = getattr(ep, "verify_source", VERIFY_SOURCE_NONE)
        # 反 theater 执行点：调 is_constructive_verification（reward>0 AND EXTERNAL）·SELF_PRODUCED reward>0
        # （自产检查通过）不计 external_verified（2 审 P1-2/P2-1·聚合器消费守门函数非孤儿·DRY）。
        if is_constructive_verification(ep):
            external_verified += 1
        # SELF_PRODUCED 细分（检查通过 vs 未过·均违锚门）
        if src == VERIFY_SOURCE_SELF_PRODUCED:
            if reward > 0:
                self_produced_check_passed += 1
            else:
                self_produced_check_failed += 1
        # 外部锚门判据：调 external_anchor_satisfied（False iff SELF_PRODUCED）·NONE/EXTERNAL 满足·SELF_PRODUCED 违。
        if external_anchor_satisfied(ep):
            anchor_satisfied += 1
        else:
            anchor_violated += 1
    out = {
        "external_verified": external_verified,
        "self_produced_check_passed": self_produced_check_passed,
        "self_produced_check_failed": self_produced_check_failed,
        "anchor_satisfied": anchor_satisfied,
        "anchor_violated": anchor_violated,
        "total": total,
    }
    assert_int(*out.values(), _where="count_layer0")
    return out
