"""W-06 独立 owner 的 append-only 五事件事务状态机。"""
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


W06_TRANSACTION_EVENT_TABLE = "ph2_w06_transaction_event"
W06_EVENT_BEGIN = 1
W06_EVENT_PREVIEW = 2
W06_EVENT_COMMIT = 3
W06_EVENT_CURSOR = 4
W06_EVENT_PUBLISHED = 5
W06_EVENT_SEQUENCE = (
    W06_EVENT_BEGIN,
    W06_EVENT_PREVIEW,
    W06_EVENT_COMMIT,
    W06_EVENT_CURSOR,
    W06_EVENT_PUBLISHED,
)


class W06TransactionError(RuntimeError):
    """W-06 事务越级、身份漂移、重复异内容提交或物理行损坏。"""


@dataclass(frozen=True)
class W06TransactionEvent:
    """一个已提交的 W-06 transaction event。"""

    run_id: int
    event_seq: int
    event_kind: int
    identity_sha256: str
    payload_sha256: str
    payload: dict[str, Any]


def register_w06_transaction_table(backend: StorageBackend) -> None:
    """注册 W-06 独有事务表，禁止复用历史 stage owner/table。"""
    register_extension_table(
        backend,
        W06_TRANSACTION_EVENT_TABLE,
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
        raise W06TransactionError("W-06 execution identity 不是 byte tuple")
    return hashlib.sha256(bytes(value)).hexdigest()


def _payload_parts(payload: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise W06TransactionError("W-06 transaction payload 必须是 object")
    encoded = canonical_json_bytes(payload)
    return encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()


class W06TransactionStore:
    """唯一拥有 W-06 begin/preview/commit/cursor/published 的 owner。"""

    def __init__(
            self,
            backend: StorageBackend,
            *,
            run_id: int,
            execution_identity_key: tuple[int, ...],
            ) -> None:
        if type(run_id) is not int or run_id <= 0:
            raise W06TransactionError("W-06 transaction run_id 非法")
        self.backend = backend
        self.run_id = run_id
        self.identity_sha256 = _identity_sha256(execution_identity_key)
        register_w06_transaction_table(backend)
        self.backend.commit()
        self.events()

    def events(self) -> tuple[W06TransactionEvent, ...]:
        rows = self.backend.select(
            W06_TRANSACTION_EVENT_TABLE,
            where={"run_id": self.run_id},
            order_by="event_seq",
        )
        events = tuple(self._decode(row) for row in rows)
        if tuple(item.event_kind for item in events) != W06_EVENT_SEQUENCE[
                :len(events)]:
            raise W06TransactionError("W-06 transaction event 缺失、重复或越级")
        if any(item.identity_sha256 != self.identity_sha256 for item in events):
            raise W06TransactionError("同 run W-06 execution identity 漂移")
        return events

    def begin(self, payload: dict[str, Any]) -> W06TransactionEvent:
        return self._append(W06_EVENT_BEGIN, payload)

    def preview(self, payload: dict[str, Any]) -> W06TransactionEvent:
        return self._append(W06_EVENT_PREVIEW, payload)

    def commit(self, payload: dict[str, Any]) -> W06TransactionEvent:
        return self._append(W06_EVENT_COMMIT, payload)

    def cursor(self, payload: dict[str, Any]) -> W06TransactionEvent:
        return self._append(W06_EVENT_CURSOR, payload)

    def published(self, payload: dict[str, Any]) -> W06TransactionEvent:
        return self._append(W06_EVENT_PUBLISHED, payload)

    def _append(
            self,
            event_kind: int,
            payload: dict[str, Any],
            ) -> W06TransactionEvent:
        if event_kind not in W06_EVENT_SEQUENCE:
            raise W06TransactionError("未知 W-06 transaction event")
        existing = self.events()
        position = W06_EVENT_SEQUENCE.index(event_kind)
        payload_json, payload_sha256 = _payload_parts(payload)
        if len(existing) > position:
            prior = existing[position]
            if (prior.event_kind != event_kind
                    or prior.payload_sha256 != payload_sha256
                    or prior.payload != payload):
                raise W06TransactionError(
                    "同 W-06 transaction event identity 内容漂移")
            return prior
        if len(existing) != position:
            raise W06TransactionError("W-06 transaction event 不得跳级")
        self.backend.insert(W06_TRANSACTION_EVENT_TABLE, {
            "run_id": self.run_id,
            "event_seq": position + 1,
            "event_kind": event_kind,
            "identity_sha256": self.identity_sha256,
            "payload_sha256": payload_sha256,
            "payload_json": payload_json,
        })
        self.backend.commit()
        return self.events()[-1]

    @staticmethod
    def _decode(row: dict[str, Any]) -> W06TransactionEvent:
        fields = {
            "run_id", "event_seq", "event_kind", "identity_sha256",
            "payload_sha256", "payload_json",
        }
        if not isinstance(row, dict) or set(row) != fields:
            raise W06TransactionError("W-06 transaction row 字段漂移")
        if any(type(row[name]) is not int for name in (
                "run_id", "event_seq", "event_kind")):
            raise W06TransactionError("W-06 transaction 整数字段非法")
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise W06TransactionError(
                "W-06 transaction payload JSON 损坏") from error
        if not isinstance(payload, dict):
            raise W06TransactionError("W-06 transaction payload 根非法")
        payload_json, payload_sha256 = _payload_parts(payload)
        if (payload_json != row["payload_json"]
                or payload_sha256 != row["payload_sha256"]):
            raise W06TransactionError("W-06 transaction payload 非规范或漂移")
        return W06TransactionEvent(
            row["run_id"],
            row["event_seq"],
            row["event_kind"],
            row["identity_sha256"],
            row["payload_sha256"],
            payload,
        )


__all__ = [
    "W06_EVENT_BEGIN",
    "W06_EVENT_COMMIT",
    "W06_EVENT_CURSOR",
    "W06_EVENT_PREVIEW",
    "W06_EVENT_PUBLISHED",
    "W06_TRANSACTION_EVENT_TABLE",
    "W06TransactionError",
    "W06TransactionEvent",
    "W06TransactionStore",
]
