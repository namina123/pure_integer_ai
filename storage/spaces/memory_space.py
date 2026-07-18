"""storage.spaces.memory_space — 记忆空间（§十五决策1）。

纯整数·两层物理分开（阅读一层 session_id=NULL / 交互二层 session_id 隔离）·带衰减·自晋升。
**两层必须文件层级物理层级分开**（用户铁律·否则无法多会话/复制固定给别人）→
两 space_id 独立（paths.py per-space dump 独立文件）。

memory_item：
  (space_id, local_id, content_hash, status, session_id, count, success_count, ...)
  - status: EXPERIENCE→CONSOLIDATED 单向 flip（MUTABLE_MONOTONE·§十三决断4 四重判据 gating）
  - session_id: 交互二层会话隔离（一层阅读 NULL/文档 batch id·A4 执行点）
  - count/success_count: §十三 G5-C 记忆项延迟晋升闸素材（**#732 已激活**·record_use 接线累加·
    G5-B 边级 promote 不读 memory_space·G5-C caller 侧 sum 聚合 by info_ref 算比率门）。
    **三个 G5 同名不同物**：G5-A judge.py:167 自证机门因子（§十四:1245）/ G5-B promote.py:_reward_ok
    边级 promote（edge sn/tn）/ G5-C §十三:1108 记忆项延迟晋升（memory_item SEG_EPISODIC 比率门·本表）。

记忆→核心砍：记忆内部自晋升（带衰减→无衰减巩固·记忆空间内状态晋升·不跨 space）。
记忆直接参与边计算带时序衰减（经 e 种子+检索回放·非独立记忆边·line939 表注）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_MEMORY, SpaceRegistry

# memory_item status（MUTABLE_MONOTONE 单向 flip·§十三决断4）
STATUS_EXPERIENCE = 1     # 带衰减经验
STATUS_CONSOLIDATED = 2   # 无衰减巩固记忆（flip 后 effective_weight 用 raw strength 不衰减）

# memory_item seg_type（落点② reward 符号契约·reward_propagate.py·M10 第一刀 11c）
# 与 STATUS_* 同范式·不同列不冲突（status 是种类②记忆概念自晋升·seg_type 是种类①episode 段类型）。
SEG_EPISODIC = 1    # reward>0 正经验（episode 成功·G5 reward 回溯数据源·line1065）
SEG_NEGATIVE = 2    # reward<0 负经验（episode 死路）

_MEMORY_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("content_hash", TYPE_INT),   # sig（Hasher.h63·决策6 台账范式）
    ("status", TYPE_INT),         # EXPERIENCE/CONSOLIDATED·MUTABLE_MONOTONE flip
    ("session_id", TYPE_INT),     # 交互二层会话隔离·一层阅读 NULL
    ("count", TYPE_INT),          # §十三 G5-C 比率门素材（#732 record_use 接线激活·caller 侧 sum 聚合）
    ("success_count", TYPE_INT),  # §十三 G5-C 比率门素材（reward>0 record_use success=True → +=2·#732 激活）
    # M10 第一刀扩列（11c·落点② reward 写 memory_item·reward_propagate.py）：
    ("seg_type", TYPE_INT),        # SEG_EPISODIC(reward>0)/SEG_NEGATIVE(reward<0)
    ("info_ref_space", TYPE_INT),  # 概念 ref space（单 sink·与 experience_count 概念聚合正交）
    ("info_ref_id", TYPE_INT),     # 概念 ref local_id
    ("context_tag", TYPE_INT),     # ctx_code（pack_ctx_code·阶段6 已落·非死码）
    ("round_id", TYPE_INT),        # episode round（WorkMemory.round_id·G5 回溯时序锚）
    # G2 overlap / G3 struct_anchor_count / G4 autocorr_conf / G6 typed_purity
    # 随 promote（Stage 6·§十三防塌C4）扩列·Stage 1 不预先固化
]
_MEMORY_INDEXES = [
    ("space_id", "local_id"),
    ("space_id", "session_id"),  # 跨会话隔离检索
    ("content_hash",),
    ("seg_type",),   # M10 第一刀·G5 回溯按段过滤 SEG_EPISODIC 候选
    ("round_id",),   # M10 第一刀·G5 回溯时序锚
    # #732 G5-C 记忆项延迟晋升闸 caller 侧 by info_ref 聚合查询（sum count/sc by info_ref·守 space_id 两层物理分开）
    ("space_id", "info_ref_space", "info_ref_id"),
]


def register_memory_table(backend: StorageBackend) -> None:
    backend.register_table(
        "memory_item", _MEMORY_COLUMNS,
        disc.DISC_MUTABLE_MONOTONE, _MEMORY_INDEXES, core=True,
    )


class MemorySpace:
    """记忆空间（纯整数·两层物理分开·带衰减·自晋升）。

    两层物理分开 = 两 space_id（阅读层/交互层各自独立 dump·paths.py）。
    本类管一个 space_id 的记忆项；两层各实例化一个。
    """

    def __init__(self, registry: SpaceRegistry, backend: StorageBackend,
                 space_id: int) -> None:
        self.registry = registry
        self.backend = backend
        self.space_id = space_id

    @classmethod
    def create(cls, registry: SpaceRegistry, name: str) -> "MemorySpace":
        sid = registry.register(SPACE_TYPE_MEMORY, name)
        return cls(registry, registry.backend, sid)

    def new_local_id(self) -> int:
        return self.backend.next_id(self.space_id)

    def put(self, local_id: int, content_hash: int, *,
            session_id: int | None = None,
            seg_type: int = 0, info_ref_space: int = 0, info_ref_id: int = 0,
            context_tag: int = 0, round_id: int = 0) -> None:
        """插入记忆项（初始 status=EXPERIENCE·带衰减）。

        M10 第一刀扩参（11c）：seg_type/info_ref/context_tag/round_id 默认 0 退化
        （既有 ms.put(local_id, content_hash) 调用 bit-identical）。落点② reward 写
        传 seg_type=SEG_EPISODIC/NEGATIVE + info_ref=sink 两列 + context_tag=ctx_code
        + round_id=workmem.round_id。content_hash 留 0 占位（reward_propagate 无 surface·
        未来 lazy backfill from concept_identity·本刀不补）。
        """
        assert_int(local_id, content_hash, seg_type, info_ref_space, info_ref_id,
                   context_tag, round_id, _where="MemorySpace.put")
        self.backend.insert("memory_item", {
            "space_id": self.space_id, "local_id": local_id,
            "content_hash": content_hash, "status": STATUS_EXPERIENCE,
            "session_id": session_id, "count": 0, "success_count": 0,
            "seg_type": seg_type, "info_ref_space": info_ref_space,
            "info_ref_id": info_ref_id, "context_tag": context_tag,
            "round_id": round_id,
        })

    def record_use(self, local_id: int, *, success: bool) -> None:
        """count++（每次使用）；success 时 success_count += 2（§十三 G5-C 比率门素材）。

        **#732 接线模式**（审1 P2-3 + 审2 P1-3 诚实标注）：record_use 函数语义守 §十三:1120
        （success_count += 2 per success）·但**调用模式偏离** line 1120 同 memory_item 多事件累加意图——
        #732 每 episode 新建一行 memory_item（落点② line 1200 守）+ 同行单次 record_use·per-row count 恒 1·
        G5-C caller 侧 sum(success_count)/sum(count) by info_ref 跨 episode 聚合算比率（数学等价·语义偏离）。
        doc §十三 line 1120 已同步修注（#732 实施片）。
        """
        delta_sc = 2 if success else 0
        self.backend.update("memory_item",
                            where={"space_id": self.space_id, "local_id": local_id},
                            set_={"count": ("+=", 1),
                                  "success_count": ("+=", delta_sc)})

    def consolidate(self, local_id: int) -> None:
        """status flip EXPERIENCE→CONSOLIDATED（MUTABLE_MONOTONE 单向·§十三决断4）。

        四重判据 gating（频次+时序稳定+reward+结构锚）由 promote 层（Stage 6）判定后调此。
        flip 后 CONSOLIDATED 态 effective_weight 用 raw strength 不衰减。
        """
        rows = self.backend.select("memory_item",
                                   where={"space_id": self.space_id,
                                          "local_id": local_id}, limit=1)
        if not rows:
            raise KeyError(f"consolidate: 记忆项不存在 ({self.space_id},{local_id})")
        old = rows[0]["status"]
        if old == STATUS_CONSOLIDATED:
            return  # 已巩固·幂等
        if old != STATUS_EXPERIENCE:
            raise disc.MonotoneViolation(
                f"status flip 须 EXPERIENCE→CONSOLIDATED·当前 {old}"
            )
        self.backend.update("memory_item",
                            where={"space_id": self.space_id, "local_id": local_id},
                            set_={"status": STATUS_CONSOLIDATED})

    def query_by_session(self, session_id: int | None) -> list[dict[str, Any]]:
        """按 session_id 检索（跨会话隔离·A4 执行点）。None 查阅读层。"""
        where: dict[str, Any] = {"space_id": self.space_id}
        if session_id is not None:
            where["session_id"] = session_id
        return self.backend.select("memory_item", where=where)
