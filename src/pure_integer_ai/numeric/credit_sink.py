"""numeric.credit_sink — 数值来源 reward 信用汇聚（append-only·单调·依赖 crosscut）。

数值关联轴上的概念，其定点值带"信用"——reward 反传时信用汇聚到数值来源。
CreditSink 是 append-only 单调累加器：信用只增不减（MUTABLE_MONOTONE），
任何 delta<0 拒绝（信用不可凭空扣除，只能转移/衰减经他处）。

纯整数（credit 是 sn/tn 计数测度族）；append-only 守完整性（audit_event 链式可选挂接）。

【诚实标注·零生产 caller（C1 设计决断 2026-07-03·doc/重来_ConceptNumeric数值轴设计决断.md）】
本模块 **零生产 caller**（仅 test_stage0 单测）。reward_propagate.py:21/163-164 docstring 明示
**主动弃用** COOCCURS→credit_sink reward 落点（防塌柱① 决断：弃 COOCCURS reward 防塌）·
全文零 `import credit_sink`。**非占位待接·是故意断线**。保留范式 = W2-B6（designed+tested+
零 caller+阻塞于整层 defer）·待用途② reward 涌现激活（须不违防塌柱①·或新 reward 落点）·
concept_numeric.M 经本 sink 单调累加。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class CreditViolation(AssertionError):
    """信用单调违例（delta<0 或非整数）。"""


class CreditSink:
    """append-only 单调信用汇聚。credit 只增。

    每个 (space_id, local_id, axis_id) 一个信用桶；append(delta) 累加，delta<0 抛。
    """

    def __init__(self) -> None:
        self._buckets: dict[tuple[int, int, int], int] = {}

    def append(self, space_id: int, local_id: int, axis_id: int, delta: int) -> int:
        """追加信用 delta（须 ≥ 0）到桶，返回桶当前总额。append-only·单调。"""
        assert_no_float(space_id, local_id, axis_id, delta, _where="CreditSink.append")
        assert_int(space_id, local_id, axis_id, delta, _where="CreditSink.append")
        if delta < 0:
            raise CreditViolation(
                f"credit_sink: delta 须 ≥ 0（append-only 单调），got delta={delta}"
            )
        key = (space_id, local_id, axis_id)
        self._buckets[key] = self._buckets.get(key, 0) + delta
        return self._buckets[key]

    def credit_of(self, space_id: int, local_id: int, axis_id: int) -> int:
        """桶当前信用总额（未建桶 = 0）。"""
        return self._buckets.get((space_id, local_id, axis_id), 0)

    def total(self) -> int:
        """全桶信用总和（计数测度·纯整数）。"""
        return sum(self._buckets.values())

    def items(self) -> list[tuple[tuple[int, int, int], int]]:
        """全桶快照（确定性排序：按 key 升序）。"""
        return sorted(self._buckets.items())
