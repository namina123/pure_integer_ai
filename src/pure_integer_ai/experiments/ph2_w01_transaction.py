"""W-01 阶段 0 的 append-only SQLite 事务事件和显式提交 owner。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.v02_run_store import canonical_json_bytes
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)


W01_TRANSACTION_EVENT_TABLE = "ph2_w01_transaction_event"
W01_EVENT_BEGIN = 1
W01_EVENT_PREVIEW = 2
W01_EVENT_COMMIT = 3
W01_EVENT_PUBLISHED = 4
W01_EVENT_SEQUENCE = (
    W01_EVENT_BEGIN,
    W01_EVENT_PREVIEW,
    W01_EVENT_COMMIT,
    W01_EVENT_PUBLISHED,
)


class W01TransactionError(RuntimeError):
    """W-01 事务出现越级、身份漂移、重复内容冲突或损坏。"""


@dataclass(frozen=True)
class W01TransactionEvent:
    """一个已显式提交的 W-01 事务状态事件。"""

    run_id: int
    event_seq: int
    event_kind: int
    identity_sha256: str
    payload_sha256: str
    payload: dict[str, Any]


def register_w01_transaction_table(backend: StorageBackend) -> None:
    """注册 append-only 事务事件表，事件序号是唯一可见顺序。"""
    register_extension_table(
        backend,
        W01_TRANSACTION_EVENT_TABLE,
        [
            ("run_id", TYPE_INT),
            ("event_seq", TYPE_INT),
            ("event_kind", TYPE_INT),
            ("identity_sha256", TYPE_TEXT),
            ("payload_sha256", TYPE_TEXT),
            ("payload_json", TYPE_TEXT),
        ],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("run_id",), ("run_id", "event_seq")],
        recovery_key=("run_id", "event_seq"),
    )


def _sha256_key(value: tuple[int, ...]) -> str:
    """把严格整数身份键编码为固定 SHA-256 文本。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise W01TransactionError("transaction identity 必须是 SHA-256 byte tuple")
    return hashlib.sha256(bytes(value)).hexdigest()


def _payload_parts(payload: dict[str, Any]) -> tuple[str, str]:
    """返回规范 JSON 文本和其 SHA-256。"""
    if not isinstance(payload, dict):
        raise W01TransactionError("transaction payload 必须是 object")
    encoded = canonical_json_bytes(payload)
    return encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()


class W01TransactionStore:
    """唯一拥有 begin/preview/commit/published/recover 的事务 owner。"""

    def __init__(
            self,
            backend: StorageBackend,
            *,
            run_id: int,
            execution_identity_key: tuple[int, ...],
            ) -> None:
        """绑定显式后端、正 run id 和不可漂移执行身份。"""
        if type(run_id) is not int or run_id <= 0:
            raise W01TransactionError("transaction run_id 必须为正严格整数")
        self.backend = backend
        self.run_id = run_id
        self.identity_sha256 = _sha256_key(execution_identity_key)
        register_w01_transaction_table(backend)
        self.backend.commit()
        self._validate_history()

    def events(self) -> tuple[W01TransactionEvent, ...]:
        """按严格事件序回读并核验当前 run 的全部提交事件。"""
        rows = self.backend.select(
            W01_TRANSACTION_EVENT_TABLE,
            where={"run_id": self.run_id},
            order_by="event_seq",
        )
        events = tuple(self._decode(row) for row in rows)
        kinds = tuple(item.event_kind for item in events)
        if kinds != W01_EVENT_SEQUENCE[:len(kinds)]:
            raise W01TransactionError("W-01 transaction event 发生缺失、重复或越级")
        if any(item.identity_sha256 != self.identity_sha256 for item in events):
            raise W01TransactionError("同 run execution identity 漂移")
        return events

    def begin(self, payload: dict[str, Any]) -> W01TransactionEvent:
        """显式提交 staged/begin 事件，精确重放幂等。"""
        return self._append(W01_EVENT_BEGIN, payload)

    def preview(self, payload: dict[str, Any]) -> W01TransactionEvent:
        """在全部 shard/barrier 完整后显式提交 preview 事件。"""
        return self._append(W01_EVENT_PREVIEW, payload)

    def commit(self, payload: dict[str, Any]) -> W01TransactionEvent:
        """显式提交逻辑结果；该事件仍不等于 stage adopted 可见。"""
        return self._append(W01_EVENT_COMMIT, payload)

    def published(self, payload: dict[str, Any]) -> W01TransactionEvent:
        """仅在 manifest+seal 已原子可见后追加最终 published 事件。"""
        return self._append(W01_EVENT_PUBLISHED, payload)

    def event_count(self) -> int:
        """返回当前 run 已提交的规范事件数。"""
        return len(self.events())

    def close(self) -> None:
        """只关闭后端连接；禁止把 close 偷换成隐式事务提交。"""
        self.backend.close()

    def _append(
            self,
            event_kind: int,
            payload: dict[str, Any],
            ) -> W01TransactionEvent:
        """按固定状态机追加一个事件，并在失败时恢复提交前快照。"""
        if event_kind not in W01_EVENT_SEQUENCE:
            raise W01TransactionError("未知 W-01 transaction event")
        existing = self.events()
        expected_position = W01_EVENT_SEQUENCE.index(event_kind)
        payload_json, payload_sha256 = _payload_parts(payload)
        if len(existing) > expected_position:
            previous = existing[expected_position]
            if (previous.event_kind != event_kind
                    or previous.payload_sha256 != payload_sha256
                    or previous.payload != payload):
                raise W01TransactionError("同 transaction event identity 内容漂移")
            return previous
        if len(existing) != expected_position:
            raise W01TransactionError("W-01 transaction event 不得跳级")
        state = self.backend.recovery_state_snapshot()
        try:
            self.backend.insert(W01_TRANSACTION_EVENT_TABLE, {
                "run_id": self.run_id,
                "event_seq": expected_position + 1,
                "event_kind": event_kind,
                "identity_sha256": self.identity_sha256,
                "payload_sha256": payload_sha256,
                "payload_json": payload_json,
            })
            self.backend.commit()
        except BaseException:
            self.backend.restore_recovery_state(state)
            self.backend.commit()
            raise
        return self.events()[-1]

    def _validate_history(self) -> None:
        """构造 owner 时立即检查历史，避免恢复到损坏或其他身份。"""
        self.events()

    @staticmethod
    def _decode(row: dict[str, Any]) -> W01TransactionEvent:
        """严格解析物理行并核验 payload hash 和字段集合。"""
        required = {
            "run_id", "event_seq", "event_kind", "identity_sha256",
            "payload_sha256", "payload_json",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise W01TransactionError("transaction event row 字段漂移")
        if any(type(row[name]) is not int for name in (
                "run_id", "event_seq", "event_kind")):
            raise W01TransactionError("transaction event 整数字段非法")
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise W01TransactionError("transaction payload JSON 损坏") from exc
        if not isinstance(payload, dict):
            raise W01TransactionError("transaction payload 根不是 object")
        payload_json, payload_sha256 = _payload_parts(payload)
        if (payload_json != row["payload_json"]
                or payload_sha256 != row["payload_sha256"]):
            raise W01TransactionError("transaction payload 非规范或摘要漂移")
        return W01TransactionEvent(
            row["run_id"],
            row["event_seq"],
            row["event_kind"],
            row["identity_sha256"],
            row["payload_sha256"],
            payload,
        )


__all__ = [
    "W01_EVENT_BEGIN",
    "W01_EVENT_COMMIT",
    "W01_EVENT_PREVIEW",
    "W01_EVENT_PUBLISHED",
    "W01_TRANSACTION_EVENT_TABLE",
    "W01TransactionError",
    "W01TransactionEvent",
    "W01TransactionStore",
    "register_w01_transaction_table",
]
