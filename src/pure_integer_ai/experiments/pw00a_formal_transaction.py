"""PW-00A PREPARED/PUBLISHED/ABORTED append-only 正式发布事务。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)


PW00A_FORMAL_LOAD_EVENT_TABLE = "pw00a_formal_load_event"
PW00A_EVENT_PREPARED = 1
PW00A_EVENT_PUBLISHED = 2
PW00A_EVENT_ABORTED = 3
_TERMINAL_EVENTS = {PW00A_EVENT_PUBLISHED, PW00A_EVENT_ABORTED}


# object-model: exception
class PW00AFormalTransactionError(RuntimeError):
    """PW-00A 发布事务缺失、越级、漂移或出现复数正式状态。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PW00AFormalEvent:
    """一条不可覆盖的 PW-00A 正式装载事件。"""

    run_id: int
    publish_epoch: int
    event_seq: int
    event_kind: int
    manifest_sha256: str
    payload_sha256: str
    payload: dict[str, Any]


def register_pw00a_formal_event_table(backend: StorageBackend) -> None:
    """注册正式发布专用扩展表，不把事务行归入 Core owner。"""
    register_extension_table(
        backend,
        PW00A_FORMAL_LOAD_EVENT_TABLE,
        [
            ("run_id", TYPE_INT),
            ("publish_epoch", TYPE_INT),
            ("event_seq", TYPE_INT),
            ("event_kind", TYPE_INT),
            ("manifest_sha256", TYPE_TEXT),
            ("payload_sha256", TYPE_TEXT),
            ("payload_json", TYPE_TEXT),
        ],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[
            ("run_id",),
            ("run_id", "event_seq"),
            ("publish_epoch",),
            ("event_kind",),
        ],
        recovery_key=("run_id", "event_seq"),
    )


def _sha256_text(payload: bytes) -> str:
    """返回规范载荷的十六进制 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    """要求摘要是小写十六进制 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise PW00AFormalTransactionError(f"{label} SHA-256 非法")
    return value


def _payload_parts(payload: dict[str, Any]) -> tuple[str, str]:
    """形成规范 JSON 与其稳定摘要。"""
    if not isinstance(payload, dict):
        raise PW00AFormalTransactionError("PW00A event payload 必须是 object")
    encoded = canonical_json_bytes(payload)
    return encoded.decode("utf-8"), _sha256_text(encoded)


# object-model: lifecycle; owner=formal-publication; cleanup=backend-close
class PW00AFormalEventStore:
    """唯一拥有 PW-00A 双阶段正式发布事件的持久 owner。"""

    def __init__(self, backend: StorageBackend) -> None:
        """绑定 backend，注册表并立即审计现有全局事件。"""
        self.backend = backend
        register_pw00a_formal_event_table(backend)
        self.backend.commit()
        self.all_events()

    def all_events(self) -> tuple[PW00AFormalEvent, ...]:
        """严格回读全局事件并拒绝复数发布或同 epoch 漂移。"""
        rows = self.backend.select(
            PW00A_FORMAL_LOAD_EVENT_TABLE,
            where=None,
            order_by="publish_epoch",
        )
        decoded = tuple(self._decode(row) for row in rows)
        grouped: dict[int, list[PW00AFormalEvent]] = {}
        for event in decoded:
            grouped.setdefault(event.run_id, []).append(event)
        for run_id, events in grouped.items():
            ordered = sorted(events, key=lambda item: item.event_seq)
            if events != ordered:
                raise PW00AFormalTransactionError(
                    f"PW00A run {run_id} 事件物理顺序漂移")
            kinds = tuple(item.event_kind for item in ordered)
            if kinds not in {
                    (PW00A_EVENT_PREPARED,),
                    (PW00A_EVENT_PREPARED, PW00A_EVENT_PUBLISHED),
                    (PW00A_EVENT_PREPARED, PW00A_EVENT_ABORTED)}:
                raise PW00AFormalTransactionError(
                    f"PW00A run {run_id} 事件缺失、重复或越级")
            if (len({item.publish_epoch for item in ordered}) != 1
                    or len({item.manifest_sha256 for item in ordered}) != 1):
                raise PW00AFormalTransactionError(
                    f"PW00A run {run_id} epoch 或 manifest 漂移")
        published = tuple(
            item for item in decoded
            if item.event_kind == PW00A_EVENT_PUBLISHED)
        if len(published) > 1:
            raise PW00AFormalTransactionError("PW00A 存在复数正式发布")
        epochs: dict[int, int] = {}
        for event in decoded:
            prior = epochs.setdefault(event.publish_epoch, event.run_id)
            if prior != event.run_id:
                raise PW00AFormalTransactionError("PW00A publish_epoch 被复用")
        return decoded

    def events(self, run_id: int) -> tuple[PW00AFormalEvent, ...]:
        """返回一个正整数 run 的规范事件前缀。"""
        if type(run_id) is not int or run_id <= 0:
            raise PW00AFormalTransactionError("PW00A run_id 非法")
        return tuple(
            item for item in self.all_events() if item.run_id == run_id)

    def published_event(self) -> PW00AFormalEvent | None:
        """返回唯一正式可见事件；PREPARED 不构成正式状态。"""
        published = tuple(
            item for item in self.all_events()
            if item.event_kind == PW00A_EVENT_PUBLISHED)
        return published[0] if published else None

    def prepared(
            self,
            *,
            run_id: int,
            publish_epoch: int,
            manifest_sha256: str,
            payload: dict[str, Any],
            ) -> PW00AFormalEvent:
        """排他追加 PREPARED，已存在时只允许逐字幂等重放。"""
        return self._append(
            run_id,
            publish_epoch,
            PW00A_EVENT_PREPARED,
            manifest_sha256,
            payload,
        )

    def published(
            self,
            *,
            run_id: int,
            publish_epoch: int,
            manifest_sha256: str,
            payload: dict[str, Any],
            ) -> PW00AFormalEvent:
        """为匹配 PREPARED 追加唯一 PUBLISHED。"""
        existing = self.published_event()
        if existing is not None and existing.run_id != run_id:
            raise PW00AFormalTransactionError("PW00A 已由其他 run 正式发布")
        return self._append(
            run_id,
            publish_epoch,
            PW00A_EVENT_PUBLISHED,
            manifest_sha256,
            payload,
        )

    def aborted(
            self,
            *,
            run_id: int,
            publish_epoch: int,
            manifest_sha256: str,
            payload: dict[str, Any],
            ) -> PW00AFormalEvent:
        """为失败 PREPARED 追加 ABORTED，历史 run 此后不可复用。"""
        return self._append(
            run_id,
            publish_epoch,
            PW00A_EVENT_ABORTED,
            manifest_sha256,
            payload,
        )

    def _append(
            self,
            run_id: int,
            publish_epoch: int,
            event_kind: int,
            manifest_sha256: str,
            payload: dict[str, Any],
            ) -> PW00AFormalEvent:
        """执行事件顺序、run/epoch 唯一性和幂等内容检查。"""
        if (type(run_id) is not int or run_id <= 0
                or type(publish_epoch) is not int or publish_epoch <= 0):
            raise PW00AFormalTransactionError("PW00A run/epoch 非法")
        _strict_sha256(manifest_sha256, label="PW00A manifest")
        if event_kind not in {
                PW00A_EVENT_PREPARED,
                PW00A_EVENT_PUBLISHED,
                PW00A_EVENT_ABORTED}:
            raise PW00AFormalTransactionError("未知 PW00A event kind")
        payload_json, payload_sha256 = _payload_parts(payload)
        existing = self.events(run_id)
        if existing:
            if (existing[0].publish_epoch != publish_epoch
                    or existing[0].manifest_sha256 != manifest_sha256):
                raise PW00AFormalTransactionError(
                    "同 PW00A run 的 epoch 或 manifest 漂移")
            if existing[-1].event_kind in _TERMINAL_EVENTS:
                expected = existing[-1]
                if (expected.event_kind == event_kind
                        and expected.payload_sha256 == payload_sha256
                        and expected.payload == payload):
                    return expected
                raise PW00AFormalTransactionError("PW00A terminal run 禁止追加")
        expected_kind = (
            PW00A_EVENT_PREPARED if not existing
            else PW00A_EVENT_PUBLISHED
        )
        if event_kind == PW00A_EVENT_ABORTED and existing:
            expected_kind = PW00A_EVENT_ABORTED
        if event_kind != expected_kind:
            raise PW00AFormalTransactionError("PW00A event 不得越级")
        for event in self.all_events():
            if (event.publish_epoch == publish_epoch
                    and event.run_id != run_id):
                raise PW00AFormalTransactionError("PW00A epoch 已被其他 run 使用")
        self.backend.insert(
            PW00A_FORMAL_LOAD_EVENT_TABLE,
            {
                "run_id": run_id,
                "publish_epoch": publish_epoch,
                "event_seq": len(existing) + 1,
                "event_kind": event_kind,
                "manifest_sha256": manifest_sha256,
                "payload_sha256": payload_sha256,
                "payload_json": payload_json,
            },
        )
        self.backend.commit()
        return self.events(run_id)[-1]

    @staticmethod
    def _decode(row: dict[str, Any]) -> PW00AFormalEvent:
        """严格解码一行并复算 canonical payload。"""
        fields = {
            "run_id", "publish_epoch", "event_seq", "event_kind",
            "manifest_sha256", "payload_sha256", "payload_json",
        }
        if not isinstance(row, dict) or set(row) != fields:
            raise PW00AFormalTransactionError("PW00A event row 字段漂移")
        if any(type(row[key]) is not int for key in (
                "run_id", "publish_epoch", "event_seq", "event_kind")):
            raise PW00AFormalTransactionError("PW00A event 整数字段非法")
        _strict_sha256(row["manifest_sha256"], label="PW00A manifest")
        _strict_sha256(row["payload_sha256"], label="PW00A payload")
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise PW00AFormalTransactionError(
                "PW00A payload JSON 损坏") from error
        payload_json, payload_sha256 = _payload_parts(payload)
        if (payload_json != row["payload_json"]
                or payload_sha256 != row["payload_sha256"]):
            raise PW00AFormalTransactionError("PW00A payload 非规范或漂移")
        return PW00AFormalEvent(
            row["run_id"],
            row["publish_epoch"],
            row["event_seq"],
            row["event_kind"],
            row["manifest_sha256"],
            row["payload_sha256"],
            payload,
        )


__all__ = [
    "PW00A_EVENT_ABORTED",
    "PW00A_EVENT_PREPARED",
    "PW00A_EVENT_PUBLISHED",
    "PW00A_FORMAL_LOAD_EVENT_TABLE",
    "PW00AFormalEvent",
    "PW00AFormalEventStore",
    "PW00AFormalTransactionError",
    "register_pw00a_formal_event_table",
]
