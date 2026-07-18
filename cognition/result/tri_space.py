"""cognition.result.tri_space — 模块6 三空间协同三时间尺度（§十四子问题4 + §十三）。

tri_space_coordination(episode) -> None
  三路三时间尺度协同（快/中/慢）：
    快环·核心反传：reward→核心 CAUSES strength+sn/tn（steer·卷二模块8 落点① 已处理·此处不重复）
    中环·记忆回放检索反哺（非 strength 反哺·**A+B 半活（#728）：replay 真活 / exclude_refs 半 defer·见下方"中环 defer 现状"**）：
      reward>0 → 正经验→回放候选（query_memory_ranked·衰减 w·检索消费非 strength 回写核心）
      reward<0 → 负经验→逆向剔除（query_negative_memories·exclude_refs）
    慢环·伴随晋升闸：promote 三闸归一编排器（卷一模块8 已处理·reward 确认才 promote·此处不重复）

  记忆无独立记忆边 strength（§十五决策1·反哺=检索消费非 strength 回写核心）。
  reward 环在核心层闭合（记忆主导是输出/检索层不同时间尺度·B2 核心解·两层分离）。

**中环 defer 现状（B1 诚实降级 2026-07-03·M10 第一刀 11a/11c/11d 部分解 2026-07-05·非"只差 caller"）**：
  本模块（query_memory_ranked / query_negative_memories / tri_space_coordination）是记忆层的 **READ 侧
  原语**·机制正确性已由 test_stage5 单测覆盖·但**整条中环生产 defer**（五连断·任一单接=theater）：
    1. ~~MemorySpace 生产未实例化~~ → **11a 已实例化**（formal_train make_train_context 挂 TrainContext·
       训练期 SpaceContext.memory_read 守 None 守 observe bit-identical·11d 落点② 写用 TrainContext.memory_read）
    2. ~~reward_propagate 落点② 写 memory_item defer~~ → **11d 已真写**（reward!=0 + sink 非 None 写
       memory_read·SEG_EPISODIC/NEGATIVE·info_ref 单 sink 两列·content_hash 留 0 占位）
    3. ~~memory_item schema 缺 seg_type/info_ref/context_tag/round_id~~ → **11c 已扩列**（5 列·info_ref 拆两列）
    4. ~~tri_space_coordination 零生产 caller~~ → **#728 A half 已接**（episode.py:134-139 生产调
       tri_space_coordination·gate MEMORY_REPLAY_MODE 门控·OFF early-return / ON + memory_read 非 None 时
       query memory → 写 workmem.replay/exclude·每 episode fresh）
    5. ~~replay_candidates 零下游消费~~ → **#728 B half 已接**（dag_path.py:221-235 local_seeds 扩张读
       workmem.replay_candidates 作额外种子·路径层序变·反 theater e2e test_728_memory_replay 锁）。
       **exclude_refs 仍半 defer**（写活·消费 defer：dag_path 层 sink 保护 + intent.sink 固定 caller 传 +
       粒度不匹配 三重阻断·无有效读法·诚实标半 defer·非 theater·replay 活即破双 theater）。
       （MemoryRef→ConceptRef 桥本身经 content_hash 结构可达·concept_identity 表 + ConceptIndex
       lazy 重建 + memory_item content_hash 列·**非** info_ref 前置。）
  断 1-4 已解（11a/11c/11d + #728 A half caller）·断 5 replay 活（#728 B half）/ exclude 半 defer。
  **中环 A+B 半活**（replay 反哺真活·gate MEMORY_REPLAY_MODE 门控·生产 try/finally flip 见 formal_train）·
  **非"整层 defer"**（旧 docstring 自陈五连断仍断 = stale·#728 done 后纠注·P3 #1054·零逻辑变·bit-identical）。
  本模块保留作记忆层实施期就绪 READ 原语（删=浪费已验机制·同 companion 范式）。11d 写活后 memory_item
  表生产有数据·**#732 G5-C 闸落 code**（promote_memory_consolidate·record_use 接线 count/sc 累加 + caller 侧
  sum 聚合 by info_ref·消费者部分活·主线 doc §十三 G5-C reward 回溯）。exclude_refs 消费者 defer（#728 诚实
  标半 defer·三重阻断·非本模块可解·须 dag_path/slot_dispatch 层粒度对齐·独立 session）。

铁律：append-only（记忆 item/伴随晋升闸台账）/ MUTABLE_MONOTONE（核心 strength sn 单调·晋升闸台账）/
  最少冗余（记忆无独立记忆边·反哺=检索消费非 strength 回写）。
诚实边界：记忆检索是结构/时序匹配非语义回忆（stable≠correct·§十四）。
defer：exclude_refs 消费者（写活·三重阻断·半 defer·见上断5）+ 真时序衰减。中环 replay A+B 半活（#728·gate
  MEMORY_REPLAY_MODE·**非整层 defer**·旧"五连断仍断"stale 已纠）。首版 rank 用
  success_count×count 活动度代理排序（×1000·纯整数）·真时序衰减 w*decay_k-logical_age 随 Stage 6 落。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer import compare as cmp
from pure_integer_ai.cognition.shared.types import Episode, ConceptRef
from pure_integer_ai.config import gates

# 负经验剔除阈值（success_rate < 此值=负经验→exclude_refs·oracle 标）
NEG_MEMORY_THRESHOLD_NUM = 1   # success_rate < 1/2 = 负经验
NEG_MEMORY_THRESHOLD_DEN = 2
# 回放候选上限（防膨胀·oracle 标）
REPLAY_CANDIDATE_CAP = 16
# 活动度评分 ×1000 缩放
RANK_SCALE = 1000


def _success_rate(row: dict[str, Any]) -> tuple[int, int]:
    """memory_item success_rate = success_count / (2·count)（success_count +=2 per success·§十三）。

    返 (num, den)·count=0 → (0,1)。
    """
    count = row.get("count", 0) or 0
    sc = row.get("success_count", 0) or 0
    if count == 0:
        return (0, 1)
    return (sc, 2 * count)


def query_memory_ranked(memory_space: Any, workmem: Any, *,
                        session_id: int | None = None,
                        limit: int = REPLAY_CANDIDATE_CAP
                        ) -> list[tuple[ConceptRef, int]]:
    """记忆回放检索排序（衰减 w·检索消费非 strength 回写核心·B3 落盘）。

    rank = success_rate × count 活动度（×1000·首版代理·真时序衰减 w*decay_k-logical_age 随 Stage 6）。
    返 [(info_ref, score), ...]·按 score 降序·ref 自然序 tiebreak·确定性 bit-identical。

    **#728 纠偏 A**：返 info_ref concept ref（row["info_ref_space"]/row["info_ref_id"]·reward_propagate
    落点② 传 path_result.sink）·非 memory_ref (space_id, local_id 行 id)。workmem.replay 下游消费者
    （dag_path local_seeds 扩张）用 concept ref·存 memory_ref 需二次查询转·且 memory_ref 是记忆空间
    内部坐标不应泄漏 workmem 语义层。退化行（info_ref=0/0·ms.put 不传 info_ref）返 (0,0)·caller 自行过滤。
    """
    rows = memory_space.query_by_session(session_id)
    ranked: list[tuple[ConceptRef, int]] = []
    for row in rows:
        info_ref: ConceptRef = (row["info_ref_space"], row["info_ref_id"])
        snum, sden = _success_rate(row)
        count = row.get("count", 0) or 0
        # rank = success_rate × count × 1000 = (snum/sden) × count × 1000
        rank = (snum * count * RANK_SCALE) // sden
        ranked.append((info_ref, rank))
    # 降序·ref 自然序 tiebreak（确定性）
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked[:limit]


def query_negative_memories(memory_space: Any, workmem: Any, *,
                             session_id: int | None = None
                             ) -> list[ConceptRef]:
    """负经验记忆检索（success_rate < 阈值=负经验→exclude_refs·逆向剔除）。

    返 info_ref concept ref 列表·ref 自然序（确定性）。**#728 纠偏 A**：返 info_ref 非 memory_ref
    （同 query_memory_ranked·workmem.exclude_refs 下游用 concept ref）。
    退化行（info_ref=0/0）返 (0,0)·caller 自行过滤。
    """
    rows = memory_space.query_by_session(session_id)
    neg: list[ConceptRef] = []
    for row in rows:
        snum, sden = _success_rate(row)
        # success_rate < NEG_MEMORY_THRESHOLD → 负经验
        if cmp.cross_lt(snum, sden,
                        NEG_MEMORY_THRESHOLD_NUM, NEG_MEMORY_THRESHOLD_DEN):
            neg.append((row["info_ref_space"], row["info_ref_id"]))
    return sorted(neg)


def tri_space_coordination(episode: Episode, *, workmem: Any,
                           memory_space: Any = None) -> None:
    """三空间协同三时间尺度（快/中/慢环）。

    memory_space 可选：提供则中环记忆回放检索反哺（reward>0 回放候选 / reward<0 逆向剔除）；
    None → 中环 defer（记忆未启用·M10 门控前·核心快环已闭合）。
    快环（核心反传）/ 慢环（伴随晋升）由卷二模块8 / 卷一模块8 已处理·此处不重复。

    **#728 落地（A 半 caller 接线 + gate 门控）**：
    - gate MEMORY_REPLAY_MODE OFF → early-return（第一道门·CI 回归 bit-identical·workmem.replay 永空）。
    - memory_space=None → early-return（第二道门·既有测 memory_read=None 默认零侵入）。
    - gate ON + memory_space 非 None → **每 episode 清 workmem.replay/exclude 再 query 写**（纠偏 C·fresh
      per episode·memory_space 持久层累积学习·transient 重算 top-N·防累积膨胀 + bit-identical 破坏点2）。
    - 写 workmem.replay 存 **info_ref concept ref**（纠偏 A·query_memory_ranked 返 info_ref·过滤 (0,0) 退化行）。
    - 写 workmem.exclude 存 info_ref concept ref（query_negative_memories 返·过滤 (0,0)）。
    - **exclude_refs 消费者 defer**（纠偏 B·dag_path 层三重阻断无有效读法·诚实标半 defer·replay 活即破双 theater）。

    **生产现状**：gate OFF（默认）→ early-return·中环生产不执行。gate ON + memory_read 传入（formal_train
    生产 :401）→ 真活。A 半（本函数 caller 接线）+ B 半（dag_path local_seeds 扩张）同实施 = 闭环非 theater。
    """
    assert_no_float(episode.reward, _where="tri_space_coordination.reward")
    # —— 快环·核心反传（卷二模块8 落点① 已处理·reward→核心 CAUSES strength+sn/tn·steer 下轮） ——
    # 此处不重复（propagate_reward 已落·核心层闭合）

    # —— 中环·记忆回放检索反哺（非 strength 反哺·B3 落盘·记忆未启用则 defer） ——
    # #728 第一道门：gate OFF → early-return（workmem.replay 永空·dag_path local_seeds == seeds·bit-identical）
    if not gates.MEMORY_REPLAY_MODE:
        return   # CI 回归 gate OFF·中环 defer·核心快环已闭合·两层分离 B2
    # 第二道门：memory_space=None → early-return（既有测 memory_read=None 默认零侵入·双守）
    if memory_space is None:
        return   # M10 门控前·记忆未启用·中环 defer（核心快环已闭合·两层分离 B2）

    # #728 纠偏 C：每 episode 清 workmem.replay/exclude（fresh per episode·memory_space 持久层累积学习·
    # transient 重算 top-N·防累积膨胀 + bit-identical 破坏点2·决断2 破坏点2 不触发）
    workmem.replay_candidates.clear()
    workmem.exclude_refs.clear()

    if episode.reward > 0:
        # 正经验→回放候选（卷二模块8 落点② SEG_EPISODIC 已写·此处检索消费非 strength 回写）
        ranked = query_memory_ranked(memory_space, workmem)
        for ref, _score in ranked:
            # 过滤 (0,0) 退化行（ms.put 不传 info_ref·生产 reward_propagate 落点② 传 sink 非 0/0）
            if ref[0] != 0 and ref not in workmem.replay_candidates:
                workmem.replay_candidates.append(ref)
    elif episode.reward < 0:
        # 负经验→逆向剔除（卷二模块8 落点② SEG_NEGATIVE 已写·exclude_refs）
        neg = query_negative_memories(memory_space, workmem)
        for ref in neg:
            if ref[0] != 0:   # 过滤 (0,0) 退化行
                workmem.exclude_refs.add(ref)

    # —— 慢环·伴随晋升闸（卷一模块8 route_to_space 三闸归一编排器已处理·reward 确认才 promote） ——
    # 此处不重复（G1-G7 晋升闸已落·成功率为主 count/成功数为辅·quality gate·freeze 后写真核心边）
