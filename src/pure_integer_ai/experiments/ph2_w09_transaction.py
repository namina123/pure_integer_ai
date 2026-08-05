"""W-09 单一 owner 的 append-only 五事件事务状态机。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w09_contract import W09_OWNER_KEY
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)


W09_TRANSACTION_EVENT_TABLE = "ph2_w09_transaction_event"
W09_EVENT_BEGIN = 1
W09_EVENT_PREVIEW = 2
W09_EVENT_COMMIT = 3
W09_EVENT_CURSOR = 4
W09_EVENT_PUBLISHED = 5
W09_EVENT_SEQUENCE = (
    W09_EVENT_BEGIN,
    W09_EVENT_PREVIEW,
    W09_EVENT_COMMIT,
    W09_EVENT_CURSOR,
    W09_EVENT_PUBLISHED,
)


class W09TransactionError(RuntimeError):
    """W-09 事务越级、身份漂移、重复写或物理行损坏。"""


@dataclass(frozen=True)
class W09TransactionEvent:
    """一个不可覆盖的 W-09 事务事件。"""

    run_id: int
    event_seq: int
    event_kind: int
    owner_key: str
    identity_sha256: str
    payload_sha256: str
    payload: dict[str, Any]


def register_w09_transaction_table(backend: StorageBackend) -> None:
    """注册 W-09 独有表，禁止借用历史 stage 的 owner 或事件表。"""
    register_extension_table(
        backend,
        W09_TRANSACTION_EVENT_TABLE,
        [
            ("run_id", TYPE_INT),
            ("event_seq", TYPE_INT),
            ("event_kind", TYPE_INT),
            ("owner_key", TYPE_TEXT),
            ("identity_sha256", TYPE_TEXT),
            ("payload_sha256", TYPE_TEXT),
            ("payload_json", TYPE_TEXT),
        ],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("run_id",), ("run_id", "event_seq")],
        recovery_key=("run_id", "event_seq"),
    )


def _identity_sha256(value: tuple[int, ...]) -> str:
    """把严格 byte tuple execution identity 转为 SHA-256。"""
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(item) is not int or not 0 <= item <= 255 for item in value)
    ):
        raise W09TransactionError("W-09 execution identity 不是 byte tuple")
    return hashlib.sha256(bytes(value)).hexdigest()


def _payload_parts(payload: dict[str, Any]) -> tuple[str, str]:
    """生成规范 JSON 字符串及其内容摘要。"""
    if not isinstance(payload, dict):
        raise W09TransactionError("W-09 transaction payload 必须是 object")
    encoded = canonical_json_bytes(payload)
    return encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()


class W09TransactionStore:
    """唯一拥有 W-09 begin/preview/commit/cursor/published 的 owner。"""

    def __init__(
        self,
        backend: StorageBackend,
        *,
        run_id: int,
        owner_key: str,
        execution_identity_key: tuple[int, ...],
    ) -> None:
        """绑定正整数 run、唯一 owner 和不可漂移的 execution identity。"""
        if type(run_id) is not int or run_id <= 0:
            raise W09TransactionError("W-09 transaction run_id 非法")
        if owner_key != W09_OWNER_KEY:
            raise W09TransactionError("W-09 transaction owner 漂移")
        self.backend = backend
        self.run_id = run_id
        self.owner_key = owner_key
        self.identity_sha256 = _identity_sha256(execution_identity_key)
        register_w09_transaction_table(backend)
        self.backend.commit()
        self.events()

    def events(self) -> tuple[W09TransactionEvent, ...]:
        """规范回读当前 run 的完整事件前缀并核对 owner/identity。"""
        rows = self.backend.select(
            W09_TRANSACTION_EVENT_TABLE,
            where={"run_id": self.run_id},
            order_by="event_seq",
        )
        events = tuple(self._decode(row) for row in rows)
        if tuple(item.event_kind for item in events) != W09_EVENT_SEQUENCE[: len(events)]:
            raise W09TransactionError("W-09 transaction event 缺失、重复或越级")
        if any(
            item.identity_sha256 != self.identity_sha256
            or item.owner_key != self.owner_key
            for item in events
        ):
            raise W09TransactionError("同 run W-09 execution identity 或 owner 漂移")
        return events

    def begin(self, payload: dict[str, Any]) -> W09TransactionEvent:
        """幂等追加 begin 事件。"""
        return self._append(W09_EVENT_BEGIN, payload)

    def preview(self, payload: dict[str, Any]) -> W09TransactionEvent:
        """幂等追加 merge preview 事件。"""
        return self._append(W09_EVENT_PREVIEW, payload)

    def commit(self, payload: dict[str, Any]) -> W09TransactionEvent:
        """幂等追加唯一语义 commit 事件。"""
        return self._append(W09_EVENT_COMMIT, payload)

    def cursor(self, payload: dict[str, Any]) -> W09TransactionEvent:
        """幂等追加完整分片 cursor 事件。"""
        return self._append(W09_EVENT_CURSOR, payload)

    def published(self, payload: dict[str, Any]) -> W09TransactionEvent:
        """幂等追加 manifest 已发布事件。"""
        return self._append(W09_EVENT_PUBLISHED, payload)

    def _append(
        self,
        event_kind: int,
        payload: dict[str, Any],
    ) -> W09TransactionEvent:
        """只允许顺序追加；同位置重放必须逐字等价。"""
        if event_kind not in W09_EVENT_SEQUENCE:
            raise W09TransactionError("未知 W-09 transaction event")
        existing = self.events()
        position = W09_EVENT_SEQUENCE.index(event_kind)
        payload_json, payload_sha256 = _payload_parts(payload)
        if len(existing) > position:
            prior = existing[position]
            if (
                prior.event_kind != event_kind
                or prior.payload_sha256 != payload_sha256
                or prior.payload != payload
            ):
                raise W09TransactionError("同 W-09 transaction event 内容漂移")
            return prior
        if len(existing) != position:
            raise W09TransactionError("W-09 transaction event 不得跳级")
        self.backend.insert(
            W09_TRANSACTION_EVENT_TABLE,
            {
                "run_id": self.run_id,
                "event_seq": position + 1,
                "event_kind": event_kind,
                "owner_key": self.owner_key,
                "identity_sha256": self.identity_sha256,
                "payload_sha256": payload_sha256,
                "payload_json": payload_json,
            },
        )
        self.backend.commit()
        return self.events()[-1]

    @staticmethod
    def _decode(row: dict[str, Any]) -> W09TransactionEvent:
        """严格解码并复算一行事务事件。"""
        fields = {
            "run_id",
            "event_seq",
            "event_kind",
            "owner_key",
            "identity_sha256",
            "payload_sha256",
            "payload_json",
        }
        if not isinstance(row, dict) or set(row) != fields:
            raise W09TransactionError("W-09 transaction row 字段漂移")
        if any(
            type(row[name]) is not int
            for name in ("run_id", "event_seq", "event_kind")
        ):
            raise W09TransactionError("W-09 transaction 整数字段非法")
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise W09TransactionError("W-09 transaction payload JSON 损坏") from error
        if not isinstance(payload, dict):
            raise W09TransactionError("W-09 transaction payload 根非法")
        payload_json, payload_sha256 = _payload_parts(payload)
        if payload_json != row["payload_json"] or payload_sha256 != row["payload_sha256"]:
            raise W09TransactionError("W-09 transaction payload 非规范或漂移")
        return W09TransactionEvent(
            row["run_id"],
            row["event_seq"],
            row["event_kind"],
            row["owner_key"],
            row["identity_sha256"],
            row["payload_sha256"],
            payload,
        )


__all__ = [
    "W09_EVENT_BEGIN",
    "W09_EVENT_COMMIT",
    "W09_EVENT_CURSOR",
    "W09_EVENT_PREVIEW",
    "W09_EVENT_PUBLISHED",
    "W09_EVENT_SEQUENCE",
    "W09_TRANSACTION_EVENT_TABLE",
    "W09TransactionError",
    "W09TransactionEvent",
    "W09TransactionStore",
    "register_w09_transaction_table",
]
