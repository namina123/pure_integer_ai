"""storage.cold_store — ColdStore 真分页骨架（§十五决策7·真分页 defer）。

**决策7 设计层债补（诚实标注）**：
  - ColdStore 旧实现是软 mark-and-skip（archive 复制不删）= 冗余快照·不释放热区内存·
    无 evict/page-in·对真分页零贡献。本骨架不复制该模式。
  - 真分页载体是 cognition 层 HotZone（BFS 按需调页·HOTZONE_MODE 默认 OFF）·非本模块。
  - 几百G不重训红线靠增量 checkpoint 续训（training/cursor·Stage 6）·非真分页兜底
    （决策7第2条：checkpoint 解决中断恢复/耐久性·真分页解决运行时内存墙·两者正交）。

Stage 1 提供骨架接口（PageRequest/PageResult）+ 占位·真分页实现 defer（百万前 HotZone 够用·
设计目标 5-8 万节点/20-30 万边）。冷区脱离核心（append-only 原则·核心永不删·改走冷区）。
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
    """冷存储骨架（真分页 defer）。Stage 1 占位·接口定形。

    真分页实现时：archive 后真删释放热区 + 接 HotCache miss 回填 page-in。
    当前 archive_to_cold 是审计件（复制留档·不释放热区·诚实标注零分页贡献）。
    """

    def __init__(self) -> None:
        self._archived: list[dict[str, Any]] = []  # 审计留档（非真分页释放）

    def archive_to_cold(self, rows: list[dict[str, Any]]) -> None:
        """冷区留档（审计件·append-only·不释放热区·零真分页贡献·诚实标注）。

        真分页 defer：未来改"archive 后真删 + page-in 回填"。当前仅留档可回溯。
        """
        self._archived.extend(dict(r) for r in rows)

    def page(self, req: PageRequest) -> PageResult:
        """分页读冷区（骨架·defer 真实现）。"""
        start = req.offset
        end = start + req.limit
        rows = self._archived[start:end]
        return PageResult(rows=rows, has_more=end < len(self._archived))

    def archived_count(self) -> int:
        return len(self._archived)
