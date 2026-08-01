"""W-04 新 owner 的 append-only 四事件事务状态机。"""
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


W04_TRANSACTION_EVENT_TABLE = "ph2_w04_transaction_event"
W04_EVENT_BEGIN = 1
W04_EVENT_PREVIEW = 2
W04_EVENT_COMMIT = 3
W04_EVENT_PUBLISHED = 4
W04_EVENT_SEQUENCE = (
    W04_EVENT_BEGIN,
    W04_EVENT_PREVIEW,
    W04_EVENT_COMMIT,
    W04_EVENT_PUBLISHED,
)


class W04TransactionError(RuntimeError):
    """W-04 事务越级、identity 漂移、覆盖或物理行损坏。"""


@dataclass(frozen=True)
class W04TransactionEvent:
    """一个已提交的 W-04 transaction event。"""

    run_id: int
    event_seq: int
    event_kind: int
    identity_sha256: str
    payload_sha256: str
    payload: dict[str, Any]


def register_w04_transaction_table(backend: StorageBackend) -> None:
    """注册 W-04 独有事务表，禁止复用 W-03 owner/table。"""
    register_extension_table(
        backend,
        W04_TRANSACTION_EVENT_TABLE,
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
    """把 execution identity 的整数 tuple 编码为 SHA-256。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise W04TransactionError("W-04 execution identity 不是 byte tuple")
    return hashlib.sha256(bytes(value)).hexdigest()


def _payload_parts(payload: dict[str, Any]) -> tuple[str, str]:
    """返回 canonical JSON 文本和 SHA-256。"""
    if not isinstance(payload, dict):
        raise W04TransactionError("W-04 transaction payload 必须是 object")
    encoded = canonical_json_bytes(payload)
    return encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()


class W04TransactionStore:
    """唯一拥有 W-04 begin/preview/commit/published 的事务 owner。"""

    def __init__(
            self,
            backend: StorageBackend,
            *,
            run_id: int,
            execution_identity_key: tuple[int, ...],
            ) -> None:
        if type(run_id) is not int or run_id <= 0:
            raise W04TransactionError("W-04 transaction run_id 非法")
        self.backend = backend
        self.run_id = run_id
        self.identity_sha256 = _identity_sha256(execution_identity_key)
        register_w04_transaction_table(backend)
        self.backend.commit()
        self.events()

    def events(self) -> tuple[W04TransactionEvent, ...]:
        """严格回读连续事件和统一 execution identity。"""
        rows = self.backend.select(
            W04_TRANSACTION_EVENT_TABLE,
            where={"run_id": self.run_id},
            order_by="event_seq",
        )
        events = tuple(self._decode(row) for row in rows)
        if tuple(item.event_kind for item in events) != W04_EVENT_SEQUENCE[
                :len(events)]:
            raise W04TransactionError("W-04 transaction event 缺失、重复或越级")
        if any(item.identity_sha256 != self.identity_sha256 for item in events):
            raise W04TransactionError("同 run W-04 execution identity 漂移")
        return events

    def begin(self, payload: dict[str, Any]) -> W04TransactionEvent:
        """追加或回读 begin event。"""
        return self._append(W04_EVENT_BEGIN, payload)

    def preview(self, payload: dict[str, Any]) -> W04TransactionEvent:
        """追加或回读 merge preview event。"""
        return self._append(W04_EVENT_PREVIEW, payload)

    def commit(self, payload: dict[str, Any]) -> W04TransactionEvent:
        """追加或回读 coordinator commit event。"""
        return self._append(W04_EVENT_COMMIT, payload)

    def published(self, payload: dict[str, Any]) -> W04TransactionEvent:
        """追加或回读 manifest published event。"""
        return self._append(W04_EVENT_PUBLISHED, payload)

    def _append(
            self,
            event_kind: int,
            payload: dict[str, Any],
            ) -> W04TransactionEvent:
        if event_kind not in W04_EVENT_SEQUENCE:
            raise W04TransactionError("未知 W-04 transaction event")
        existing = self.events()
        position = W04_EVENT_SEQUENCE.index(event_kind)
        payload_json, payload_sha256 = _payload_parts(payload)
        if len(existing) > position:
            prior = existing[position]
            if (prior.event_kind != event_kind
                    or prior.payload_sha256 != payload_sha256
                    or prior.payload != payload):
                raise W04TransactionError(
                    "同 W-04 transaction event identity 内容漂移")
            return prior
        if len(existing) != position:
            raise W04TransactionError("W-04 transaction event 不得跳级")
        self.backend.insert(W04_TRANSACTION_EVENT_TABLE, {
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
    def _decode(row: dict[str, Any]) -> W04TransactionEvent:
        """从物理行恢复事务事件并校验 canonical payload。"""
        fields = {
            "run_id", "event_seq", "event_kind", "identity_sha256",
            "payload_sha256", "payload_json",
        }
        if not isinstance(row, dict) or set(row) != fields:
            raise W04TransactionError("W-04 transaction row 字段漂移")
        if any(type(row[name]) is not int for name in (
                "run_id", "event_seq", "event_kind")):
            raise W04TransactionError("W-04 transaction 整数字段非法")
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise W04TransactionError("W-04 transaction payload JSON 损坏") from exc
        if not isinstance(payload, dict):
            raise W04TransactionError("W-04 transaction payload 根非法")
        payload_json, payload_sha256 = _payload_parts(payload)
        if (payload_json != row["payload_json"]
                or payload_sha256 != row["payload_sha256"]):
            raise W04TransactionError("W-04 transaction payload 非规范或漂移")
        return W04TransactionEvent(
            row["run_id"],
            row["event_seq"],
            row["event_kind"],
            row["identity_sha256"],
            row["payload_sha256"],
            payload,
        )


__all__ = [
    "W04_EVENT_BEGIN",
    "W04_EVENT_COMMIT",
    "W04_EVENT_PREVIEW",
    "W04_EVENT_PUBLISHED",
    "W04_TRANSACTION_EVENT_TABLE",
    "W04TransactionError",
    "W04TransactionEvent",
    "W04TransactionStore",
]
