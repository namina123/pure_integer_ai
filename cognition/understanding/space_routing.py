"""cognition.understanding.space_routing — 模块8 三空间落点分流（§十三三空间）。

route_to_space：按 stage 路由概念点/边到对应空间。
  STAGE_TRAINING          → 核心（ZERO_AI_MEMORY_ACTIVE=0·只核心养洁净·记忆建但不参与）
  STAGE_POST_WEANING_READ → 伴随 staging(sign=0) → 过闸晋升记忆一层（择优·带时序·M10 门控）
  STAGE_USER_INTERACTION  → 记忆二层（全量·不经检疫·交互经验本就全量记·H4 stage 纪律）
  STAGE_EXTERNAL_DEFINE   → 伴随检疫 → 过闸晋升（教师 define 分流：元定义 PRIMARY 直落核心 /
                            知识定义 sign=0 检疫·G2 机械核查非教师自标·H3 仅断奶前）

落点 space_id 选择（observe 建图时各 build_* 按 stage 传 space·M4·非硬编码 CORE）：
  本模块提供 target_space_id(stage, ctx) 供 observe 决定概念点落哪个 space。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.types import (
    SpaceContext, STAGE_TRAINING, STAGE_POST_WEANING_READ,
    STAGE_USER_INTERACTION, STAGE_EXTERNAL_DEFINE, WEANING_PRE,
)

# 教师 define 内容类型（G2 机械核查·非教师自标）
META_DEFINITION = 1      # 元定义（出厂硬件·模态标记/IS_A 骨架/高频概念·仅断奶前直落核心）
KNOWLEDGE_DEFINITION = 2  # 知识定义（具体诠释·sign=0 检疫）


def target_space_id(stage: int, ctx: SpaceContext) -> int:
    """按 stage 决定概念点落点，缺少该阶段设施时拒绝写入。

    训练期 → 核心；训练后阅读 → 记忆一层；交互 → 记忆二层；外部 define → 伴随检疫。
    """
    if stage == STAGE_TRAINING:
        return ctx.core.space_id
    if stage == STAGE_POST_WEANING_READ:
        if ctx.memory_read is None:
            raise RuntimeError("训练后阅读缺少 MemoryRead，拒绝降级写入 Core")
        if ctx.companion is None:
            raise RuntimeError("训练后阅读缺少 Companion 检疫空间")
        return ctx.memory_read.space_id
    if stage == STAGE_USER_INTERACTION:
        if ctx.memory_interact is None:
            raise RuntimeError("用户交互缺少 MemoryInteract，拒绝降级写入其他空间")
        return ctx.memory_interact.space_id
    if stage == STAGE_EXTERNAL_DEFINE:
        if ctx.companion is None:
            raise RuntimeError("外部定义缺少 Companion，拒绝降级写入 Core")
        return ctx.companion.space_id
    raise ValueError(f"未知训练 stage: {stage}")


def route_to_space(stage: int, ctx: SpaceContext,
                   *, teacher_content_type: int | None = None) -> str:
    """返回经设施核验的路由描述，未知阶段和缺失空间均失败。

    EXTERNAL_DEFINE 教师分流：元定义 PRIMARY 直落核心（仅断奶前·H3）/ 知识定义 sign=0 检疫。
    G2：内容类型须系统侧机械核查非教师自标（白名单 = 模态标记码点/IS_A 骨架/高频概念频次阈值）。
    """
    if stage == STAGE_TRAINING:
        return "CORE"   # 训练期核心养洁净
    if stage == STAGE_POST_WEANING_READ:
        if ctx.memory_read is None:
            raise RuntimeError("训练后阅读缺少 MemoryRead，拒绝降级写入 Core")
        if ctx.companion is None:
            raise RuntimeError("训练后阅读缺少 Companion 检疫空间")
        return "COMPANION_QUARANTINE→MEMORY_READ"   # 伴随 staging → 过闸晋升记忆一层
    if stage == STAGE_USER_INTERACTION:
        if ctx.memory_interact is None:
            raise RuntimeError("用户交互缺少 MemoryInteract，拒绝降级写入其他空间")
        return "MEMORY_INTERACT"   # 记忆二层全量（不经检疫·H4 stage 纪律）
    if stage == STAGE_EXTERNAL_DEFINE:
        if ctx.companion is None:
            raise RuntimeError("外部定义缺少 Companion，拒绝降级写入 Core")
        if (teacher_content_type == META_DEFINITION
                and ctx.weaning_phase == WEANING_PRE):
            return "CORE_PRIMARY"   # 元定义直落核心（仅断奶前·冷启动硬依赖·H3）
        return "COMPANION_QUARANTINE"   # 知识定义 sign=0 检疫
    raise ValueError(f"未知训练 stage: {stage}")


def gate_check_memory_steady(ctx: SpaceContext) -> bool:
    """在真实 Memory 稳态探针接线前保持硬阻断。"""
    _ = ctx
    return False
