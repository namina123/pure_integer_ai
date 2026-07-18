"""storage.audit — audit_event 持久化表（挂 crosscut.audit_event 链）。

把 crosscut.determinism.audit_event 的内存 append-only 链持久化到核心表 audit_event
（timestamp_seq AUTOINCREMENT + event_hash 链式 prev_hash·§十五决策2/6）。

核心无墙钟执行点：timestamp_seq 是 AUTOINCREMENT 整数序号·唯一时间源（禁 time/datetime）。
event_hash 链式 prev_hash 守 append-only 完整性（任何篡改/重排/删除断链）。

audit_event 是核心表·纯整数（op 存 op_hash 整数·crosscut AuditLog 已如此）。
persist 把内存事件落表（op 已 int·event_hash 一致·自洽链）；rebuild 从表重建；
verify_persisted_chain 重算验链。append-only：DISC_APPEND_ONLY（拒 UPDATE/DELETE）。
"""
from __future__ import annotations

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.crosscut.determinism.audit_event import AuditEvent, AuditLog
from pure_integer_ai.crosscut.determinism.hasher import Hasher

_AUDIT_COLUMNS = [
    ("seq", TYPE_INT),            # timestamp_seq·AUTOINCREMENT 单调（append-only）
    ("op", TYPE_INT),             # op_hash（纯整数·守"核心无 str"）
    ("payload_hash", TYPE_INT),   # 负载 Hasher.h63（负载本身不入链·只入其哈希）
    ("prev_hash", TYPE_INT),      # 前一事件 event_hash（首事件 0）
    ("event_hash", TYPE_INT),     # 本事件身份 = Hasher.h63((seq,op,payload_hash,prev_hash))
]
_AUDIT_INDEXES = [("seq",), ("event_hash",)]

_AUDIT_HASHER = Hasher("pure_integer_ai.audit_event.v1")  # 与 crosscut AuditLog 同族·跨宿主一致


def register_audit_table(backend: StorageBackend) -> None:
    backend.register_table(
        "audit_event", _AUDIT_COLUMNS,
        disc.DISC_APPEND_ONLY, _AUDIT_INDEXES, core=True,
    )


class PersistedAuditLog:
    """把 crosscut AuditLog 事件持久化到 audit_event 表·或从表重建。"""

    def __init__(self, backend: StorageBackend) -> None:
        self._b = backend

    def persist(self, log: AuditLog) -> int:
        """把内存 AuditLog 全事件落表（append-only·返回落盘条数）。

        落表前校验内存链 verify_chain()·断链抛（守 append-only 完整性）。
        op 已是 op_hash 整数·event_hash 与表重算一致（自洽链）。
        """
        if not log.verify_chain():
            raise disc.DisciplineViolation("persist: 内存 audit 链断链·拒落盘")
        n = 0
        for ev in log.events:
            self._b.insert("audit_event", {
                "seq": ev.seq, "op": ev.op,
                "payload_hash": ev.payload_hash, "prev_hash": ev.prev_hash,
                "event_hash": ev.event_hash,
            })
            n += 1
        return n

    def rebuild_chain(self) -> AuditLog:
        """从表重建内存 AuditLog（续训/崩溃恢复·按 seq 升序·表为权威）。

        op_name 已丢失（表存 op_hash）·重建事件 op=op_hash（int·AuditEvent.op 类型一致）。
        重建后 verify_chain() 应过（event_hash 与表一致·不重算）。
        """
        rows = self._b.select("audit_event", order_by="seq")
        log = AuditLog()
        for r in rows:
            log._events.append(AuditEvent(  # type: ignore[attr-defined]
                seq=r["seq"], op=r["op"], payload_hash=r["payload_hash"],
                prev_hash=r["prev_hash"], event_hash=r["event_hash"],
            ))
        return log

    def verify_persisted_chain(self) -> bool:
        """校验持久化链：seq 严格递增 + prev_hash 链接 + event_hash 一致（重算）。"""
        rows = self._b.select("audit_event", order_by="seq")
        prev = 0
        for i, r in enumerate(rows, start=1):
            if r["seq"] != i or r["prev_hash"] != prev:
                return False
            recomputed = _AUDIT_HASHER.h63(
                (r["seq"], r["op"], r["payload_hash"], r["prev_hash"])
            )
            if recomputed != r["event_hash"]:
                return False
            prev = r["event_hash"]
        return True
