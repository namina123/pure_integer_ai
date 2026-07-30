"""W-03 新 owner 的 append-only 四事件事务状态机。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)


W03_TRANSACTION_EVENT_TABLE = "ph2_w03_transaction_event"
W03_EVENT_BEGIN = 1
W03_EVENT_PREVIEW = 2
W03_EVENT_COMMIT = 3
W03_EVENT_PUBLISHED = 4
W03_EVENT_SEQUENCE = (
    W03_EVENT_BEGIN,
    W03_EVENT_PREVIEW,
    W03_EVENT_COMMIT,
    W03_EVENT_PUBLISHED,
)


class W03TransactionError(RuntimeError):
    """W-03 事务越级、identity 漂移、覆盖或物理行损坏。"""


@dataclass(frozen=True)
class W03TransactionEvent:
    """一个已提交的 W-03 transaction event。"""

    run_id: int
    event_seq: int
    event_kind: int
    identity_sha256: str
    payload_sha256: str
    payload: dict[str, Any]


def register_w03_transaction_table(backend: StorageBackend) -> None:
    """注册 W-03 独有事务表，禁止复用 W-02 owner/table。"""
    register_extension_table(
        backend,
        W03_TRANSACTION_EVENT_TABLE,
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


def _identity_sha256(value: tuple[int, ...]) -> str:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise W03TransactionError("W-03 execution identity 不是 byte tuple")
    return hashlib.sha256(bytes(value)).hexdigest()


def _payload_parts(payload: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise W03TransactionError("W-03 transaction payload 必须是 object")
    encoded = canonical_json_bytes(payload)
    return encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()


class W03TransactionStore:
    """唯一拥有 W-03 begin/preview/commit/published 的事务 owner。"""

    def __init__(
            self,
            backend: StorageBackend,
            *,
            run_id: int,
            execution_identity_key: tuple[int, ...],
            ) -> None:
        if type(run_id) is not int or run_id <= 0:
            raise W03TransactionError("W-03 transaction run_id 非法")
        self.backend = backend
        self.run_id = run_id
        self.identity_sha256 = _identity_sha256(execution_identity_key)
        register_w03_transaction_table(backend)
        self.backend.commit()
        self.events()

    def events(self) -> tuple[W03TransactionEvent, ...]:
        """严格回读连续事件和统一 execution identity。"""
        rows = self.backend.select(
            W03_TRANSACTION_EVENT_TABLE,
            where={"run_id": self.run_id},
            order_by="event_seq",
        )
        events = tuple(self._decode(row) for row in rows)
        if tuple(item.event_kind for item in events) != W03_EVENT_SEQUENCE[
                :len(events)]:
            raise W03TransactionError("W-03 transaction event 缺失、重复或越级")
        if any(item.identity_sha256 != self.identity_sha256 for item in events):
            raise W03TransactionError("同 run W-03 execution identity 漂移")
        return events

    def begin(self, payload: dict[str, Any]) -> W03TransactionEvent:
        return self._append(W03_EVENT_BEGIN, payload)

    def preview(self, payload: dict[str, Any]) -> W03TransactionEvent:
        return self._append(W03_EVENT_PREVIEW, payload)

    def commit(
            self,
            payload: dict[str, Any],
            *,
            rollback_state: dict[str, Any],
            ) -> W03TransactionEvent:
        if not isinstance(rollback_state, dict):
            raise W03TransactionError("W-03 rollback state 类型非法")
        return self._append(
            W03_EVENT_COMMIT,
            payload,
            rollback_state=rollback_state,
        )

    def published(self, payload: dict[str, Any]) -> W03TransactionEvent:
        return self._append(W03_EVENT_PUBLISHED, payload)

    def close(self) -> None:
        self.backend.close()

    def _append(
            self,
            event_kind: int,
            payload: dict[str, Any],
            *,
            rollback_state: dict[str, Any] | None = None,
            ) -> W03TransactionEvent:
        if event_kind not in W03_EVENT_SEQUENCE:
            raise W03TransactionError("未知 W-03 transaction event")
        existing = self.events()
        position = W03_EVENT_SEQUENCE.index(event_kind)
        payload_json, payload_sha256 = _payload_parts(payload)
        if len(existing) > position:
            prior = existing[position]
            if (prior.event_kind != event_kind
                    or prior.payload_sha256 != payload_sha256
                    or prior.payload != payload):
                raise W03TransactionError(
                    "同 W-03 transaction event identity 内容漂移")
            return prior
        if len(existing) != position:
            raise W03TransactionError("W-03 transaction event 不得跳级")
        safe_state = (
            self.backend.recovery_state_snapshot()
            if rollback_state is None else rollback_state
        )
        try:
            self.backend.insert(W03_TRANSACTION_EVENT_TABLE, {
                "run_id": self.run_id,
                "event_seq": position + 1,
                "event_kind": event_kind,
                "identity_sha256": self.identity_sha256,
                "payload_sha256": payload_sha256,
                "payload_json": payload_json,
            })
            self.backend.commit()
        except BaseException:
            self.backend.restore_recovery_state(safe_state)
            self.backend.commit()
            raise
        return self.events()[-1]

    @staticmethod
    def _decode(row: dict[str, Any]) -> W03TransactionEvent:
        fields = {
            "run_id", "event_seq", "event_kind", "identity_sha256",
            "payload_sha256", "payload_json",
        }
        if not isinstance(row, dict) or set(row) != fields:
            raise W03TransactionError("W-03 transaction row 字段漂移")
        if any(type(row[name]) is not int for name in (
                "run_id", "event_seq", "event_kind")):
            raise W03TransactionError("W-03 transaction 整数字段非法")
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise W03TransactionError("W-03 transaction payload JSON 损坏") from exc
        if not isinstance(payload, dict):
            raise W03TransactionError("W-03 transaction payload 根非法")
        payload_json, payload_sha256 = _payload_parts(payload)
        if (payload_json != row["payload_json"]
                or payload_sha256 != row["payload_sha256"]):
            raise W03TransactionError("W-03 transaction payload 非规范或漂移")
        return W03TransactionEvent(
            row["run_id"],
            row["event_seq"],
            row["event_kind"],
            row["identity_sha256"],
            row["payload_sha256"],
            payload,
        )


__all__ = [
    "W03_EVENT_BEGIN",
    "W03_EVENT_COMMIT",
    "W03_EVENT_PREVIEW",
    "W03_EVENT_PUBLISHED",
    "W03_TRANSACTION_EVENT_TABLE",
    "W03TransactionError",
    "W03TransactionEvent",
    "W03TransactionStore",
]
