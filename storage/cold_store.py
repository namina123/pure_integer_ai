"""storage.cold_store — 旧 Stage 1 列表接口的兼容隔离层。

**决策7 设计层债补（诚实标注）**：
  - ColdStore 旧实现是软 mark-and-skip（archive 复制不删）= 冗余快照·不释放热区内存·
    无 evict/page-in·对真分页零贡献。本骨架不复制该模式。
  - 当前模块不是 K-01 location manifest，也不是 K-02 sealed segment/page-in 实现。
  - 几百G不重训红线靠增量 checkpoint 续训（training/cursor·Stage 6）·非真分页兜底
    （决策7第2条：checkpoint 解决中断恢复/耐久性·真分页解决运行时内存墙·两者正交）。

Stage 1 旧接口只为兼容测试保留。现行实现位于 ``sealed_segment``、
``segment_repository``、``tiered_segment_store`` 和 ``segment_cache``；生产 caller
应通过 ``storage.build_tiered_segment_store`` 构造，不得把本列表计入 readiness。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PageRequest:
    """分页请求（纯整数·offset/limit）。"""

    space_id: int
    offset: int
    limit: int


@dataclass(frozen=True)
class PageResult:
    """分页结果。"""

    rows: list[dict[str, Any]]
    has_more: bool


class ColdStore:
    """旧冷存储骨架。Stage 1 占位接口，不属于现行 K 线交付。

    真分页实现时：archive 后真删释放热区 + 接 HotCache miss 回填 page-in。
    当前 archive_to_cold 是审计件（复制留档·不释放热区·诚实标注零分页贡献）。
    """

    def __init__(self) -> None:
        self._archived: list[dict[str, Any]] = []  # 审计留档（非真分页释放）

    def archive_to_cold(self, rows: list[dict[str, Any]]) -> None:
        """冷区留档（审计件·append-only·不释放热区·零真分页贡献·诚实标注）。

        现行 K-02 已由 sealed segment 和 page-in 协议替代；本接口仅留档可回溯。
        """
        self._archived.extend(dict(r) for r in rows)

    def page(self, req: PageRequest) -> PageResult:
        """按旧 offset 接口读取占位列表，不提供稳定续页保证。"""
        start = req.offset
        end = start + req.limit
        rows = self._archived[start:end]
        return PageResult(rows=rows, has_more=end < len(self._archived))

    def archived_count(self) -> int:
        return len(self._archived)
