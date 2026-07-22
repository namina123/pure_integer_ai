"""旧 memory_item 只读迁移查询与跨 episode replay 关闭边界。

`query_memory_ranked` 和 `query_negative_memories` 仅用于核证历史表，不是正式
Memory resolver。`tri_space_coordination` 保留兼容调用面，但只清理旧 transient
状态；它不得依据当前 episode reward 为下一次查询写 seed 或 exclusion。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.integer import compare as cmp
from pure_integer_ai.cognition.shared.types import Episode, ConceptRef

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
    """关闭 reward 驱动的跨 episode 回放，并清除旧迁移状态。"""
    _ = episode, memory_space
    workmem.replay_candidates.clear()
    workmem.exclude_refs.clear()
