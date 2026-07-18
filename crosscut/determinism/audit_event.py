"""crosscut.determinism.audit_event — append-only 审计事件链（核心无墙钟的时间源）。

硬约束「核心无墙钟」的执行点：timestamp_seq 是 AUTOINCREMENT 整数序号，**唯一时间源**
（禁 time/datetime）。每个事件带 event_hash，链式引用前一个 event_hash（prev_hash），
形成 append-only 哈希链——任何中间篡改/重排/删除都会断链。

事件结构（纯整数 + 哈希·audit_event 是核心表故 op 存 op_hash 整数非 str）：
  seq          —— timestamp_seq，AUTOINCREMENT 单调递增（append-only）
  op           —— 操作名 op_hash（Hasher.h63(op_name)·纯整数·守"核心无 str"）
  payload_hash —— 负载的 Hasher.h63（负载本身不入链，只入其哈希·可控体积）
  prev_hash    —— 前一事件的 event_hash（首事件为 0）
  event_hash   —— Hasher.h63((seq, op, payload_hash, prev_hash))，本事件身份

op_name→op_hash 是确定映射（调用方传 str·链存 int·可逆经 registry 若需）。
Stage 0 提供内存版 append-only 链；SQLite 持久化落 storage/audit.py（audit_event 表）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.determinism.hasher import Hasher

_ZERO_HASH = 0  # 首事件的 prev_hash 占位


@dataclass(frozen=True)
class AuditEvent:
    """一个审计事件（append-only 链节点·全纯整数）。"""

    seq: int
    op: int           # op_hash（Hasher.h63(op_name)·纯整数）
    payload_hash: int
    prev_hash: int
    event_hash: int


class AuditLog:
    """append-only 审计事件链。永不删/改；只 append。

    timestamp_seq 是唯一时间源（核心无墙钟）。event_hash 链式 prev_hash 守完整性。
    op_name（str）经 op_hasher 转 op_hash（int）存储·守"核心无 str"。
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._hasher = Hasher("pure_integer_ai.audit_event.v1")  # 固定族 seed·跨宿主一致
        self._op_hasher = Hasher("pure_integer_ai.audit.op.v1")  # op_name→op_hash

    def _payload_hash(self, payload: Any) -> int:
        return self._hasher.h63(payload)

    def _op_hash(self, op_name: str) -> int:
        return self._op_hasher.h63(op_name)

    def append(self, op_name: str, payload: Any) -> AuditEvent:
        """追加事件。seq = len+1（AUTOINCREMENT）；prev_hash = 上一事件 event_hash。"""
        seq = len(self._events) + 1
        op = self._op_hash(op_name)
        payload_hash = self._payload_hash(payload)
        prev_hash = self._events[-1].event_hash if self._events else _ZERO_HASH
        event_hash = self._hasher.h63((seq, op, payload_hash, prev_hash))
        ev = AuditEvent(seq, op, payload_hash, prev_hash, event_hash)
        self._events.append(ev)
        return ev

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        """全事件快照（持久化/重建用）。"""
        return tuple(self._events)

    def event_hash_sequence(self) -> tuple[int, ...]:
        """全事件 event_hash 序列（assert_reproducible / 确定性回放比对用）。"""
        return tuple(e.event_hash for e in self._events)

    def seq_sequence(self) -> tuple[int, ...]:
        """全事件 seq 序列（append-only 完整性：须为 1..n 严格递增）。"""
        return tuple(e.seq for e in self._events)

    def verify_chain(self) -> bool:
        """重算并校验整链：seq 严格递增从 1 起 + prev_hash 链接 + event_hash 一致。

        任何中间篡改/重排/删除都会断链。返回 True = 完整 append-only。
        """
        prev = _ZERO_HASH
        for i, e in enumerate(self._events, start=1):
            if e.seq != i:
                return False
            if e.prev_hash != prev:
                return False
            recomputed = self._hasher.h63((e.seq, e.op, e.payload_hash, e.prev_hash))
            if recomputed != e.event_hash:
                return False
            prev = e.event_hash
        return True

    def __len__(self) -> int:
        return len(self._events)
