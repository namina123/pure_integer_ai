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
    """按 stage 决定概念点落点 space_id（M4·非硬编码 CORE）。

    训练期 → 核心；训练后阅读 → 记忆一层；交互 → 记忆二层；外部 define → 伴随检疫。
    """
    if stage == STAGE_TRAINING:
        return ctx.core.space_id
    if stage == STAGE_POST_WEANING_READ:
        if ctx.memory_read is None:
            return ctx.core.space_id
        return ctx.memory_read.space_id
    if stage == STAGE_USER_INTERACTION:
        if ctx.memory_interact is None:
            return ctx.memory_read.space_id if ctx.memory_read else ctx.core.space_id
        return ctx.memory_interact.space_id
    if stage == STAGE_EXTERNAL_DEFINE:
        # 伴随检疫（概念点先入伴随 sign=0·过闸晋升）——返回伴随 space_id
        if ctx.companion is None:
            return ctx.core.space_id
        return ctx.companion.space_id
    return ctx.core.space_id


def route_to_space(stage: int, ctx: SpaceContext,
                   *, teacher_content_type: int | None = None) -> str:
    """路由判定（返回落点描述·供 observe/审计用）。

    EXTERNAL_DEFINE 教师分流：元定义 PRIMARY 直落核心（仅断奶前·H3）/ 知识定义 sign=0 检疫。
    G2：内容类型须系统侧机械核查非教师自标（白名单 = 模态标记码点/IS_A 骨架/高频概念频次阈值）。
    """
    if stage == STAGE_TRAINING:
        return "CORE"   # 训练期核心养洁净
    if stage == STAGE_POST_WEANING_READ:
        return "COMPANION_QUARANTINE→MEMORY_READ"   # 伴随 staging → 过闸晋升记忆一层
    if stage == STAGE_USER_INTERACTION:
        return "MEMORY_INTERACT"   # 记忆二层全量（不经检疫·H4 stage 纪律）
    if stage == STAGE_EXTERNAL_DEFINE:
        if (teacher_content_type == META_DEFINITION
                and ctx.weaning_phase == WEANING_PRE):
            return "CORE_PRIMARY"   # 元定义直落核心（仅断奶前·冷启动硬依赖·H3）
        return "COMPANION_QUARANTINE"   # 知识定义 sign=0 检疫
    return "CORE"


def gate_check_memory_steady(ctx: SpaceContext) -> bool:
    """M10 门控：开记忆前重测防塌三柱（衔接四条件记忆主导守恒·§十三M10）。

    首版占位：返回 True（ Stage 5 防塌三柱/收敛判据落地后接真实度量）。
    训练后开记忆须重新验收防塌三柱（防误判"训练期验收通过=稳态安全"）。
    """
    return True   # defer：Stage 5 防塌三柱度量接线后填实
