"""W-03 train envelope、projection、Use/outcome 与 retention 持久账。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_bytes,
    record_from_dict,
)
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)


W03_ARTIFACT_TABLE = "ph2_w03_runtime_artifact"
ARTIFACT_TRAIN_ENVELOPE = "TRAIN_ENVELOPE"
ARTIFACT_EVIDENCE_ACCOUNT = "EVIDENCE_ACCOUNT"
ARTIFACT_PROJECTION = "PROJECTION"
ARTIFACT_GENERATION_CHOICE = "GENERATION_CHOICE"
ARTIFACT_GENERATION_DECISION = "GENERATION_DECISION"
ARTIFACT_GENERATION_USE = "GENERATION_USE"
ARTIFACT_GENERATION_OUTCOME = "GENERATION_OUTCOME"
ARTIFACT_W02_RETENTION = "W02_RETENTION"
W03_ARTIFACT_KINDS = (
    ARTIFACT_TRAIN_ENVELOPE,
    ARTIFACT_EVIDENCE_ACCOUNT,
    ARTIFACT_PROJECTION,
    ARTIFACT_GENERATION_CHOICE,
    ARTIFACT_GENERATION_DECISION,
    ARTIFACT_GENERATION_USE,
    ARTIFACT_GENERATION_OUTCOME,
    ARTIFACT_W02_RETENTION,
)


class W03ArtifactError(RuntimeError):
    """W-03 artifact owner 的 kind、ordinal、SHA 或 canonical JSON 漂移。"""


@dataclass(frozen=True)
class W03ArtifactRecord:
    """一条完成 size/SHA 回验的 W-03 append-only artifact。"""

    artifact_kind: str
    ordinal: int
    payload_sha256: str
    payload: dict[str, Any]


class W03ArtifactStore:
    """只拥有 W-03 runtime artifacts，不写 W-02、Memory 或 evaluator。"""

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend
        register_extension_table(
            backend,
            W03_ARTIFACT_TABLE,
            [
                ("artifact_kind", TYPE_TEXT),
                ("ordinal", TYPE_INT),
                ("size_bytes", TYPE_INT),
                ("payload_sha256", TYPE_TEXT),
                ("payload_json", TYPE_TEXT),
            ],
            discipline=disc.DISC_APPEND_ONLY,
            indexes=[("artifact_kind",), ("artifact_kind", "ordinal")],
            recovery_key=("artifact_kind", "ordinal"),
        )

    def put(
            self,
            artifact_kind: str,
            ordinal: int,
            payload: dict[str, Any],
            ) -> W03ArtifactRecord:
        """按 kind/ordinal 幂等追加，异内容重放 fail closed。"""
        if artifact_kind not in W03_ARTIFACT_KINDS:
            raise W03ArtifactError("W-03 artifact kind 未注册")
        if type(ordinal) is not int or ordinal <= 0:
            raise W03ArtifactError("W-03 artifact ordinal 必须是严格正整数")
        if not isinstance(payload, dict):
            raise W03ArtifactError("W-03 artifact payload 必须是 object")
        encoded = canonical_json_bytes(payload)
        row = {
            "artifact_kind": artifact_kind,
            "ordinal": ordinal,
            "size_bytes": len(encoded),
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "payload_json": encoded.decode("utf-8"),
        }
        existing = self.backend.select(W03_ARTIFACT_TABLE, {
            "artifact_kind": artifact_kind,
            "ordinal": ordinal,
        })
        if existing:
            if len(existing) != 1 or existing[0] != row:
                raise W03ArtifactError("W-03 artifact identity 异内容覆盖")
            return self._decode(existing[0])
        self.backend.insert(W03_ARTIFACT_TABLE, row)
        return self._decode(row)

    def records(
            self,
            artifact_kind: str | None = None,
            ) -> tuple[W03ArtifactRecord, ...]:
        """逐行回验 canonical JSON、size、SHA 和连续 ordinal。"""
        if (artifact_kind is not None
                and artifact_kind not in W03_ARTIFACT_KINDS):
            raise W03ArtifactError("W-03 artifact kind 未注册")
        where = (
            None if artifact_kind is None
            else {"artifact_kind": artifact_kind}
        )
        values = tuple(sorted(
            (
                self._decode(row)
                for row in self.backend.select(
                    W03_ARTIFACT_TABLE,
                    where=where,
                )
            ),
            key=lambda item: (item.artifact_kind, item.ordinal),
        ))
        by_kind: dict[str, list[int]] = {}
        for item in values:
            by_kind.setdefault(item.artifact_kind, []).append(item.ordinal)
        if any(ordinals != list(range(1, len(ordinals) + 1))
               for ordinals in by_kind.values()):
            raise W03ArtifactError("W-03 artifact ordinal 缺失、重复或越级")
        return values

    def counts(self) -> tuple[tuple[str, int], ...]:
        """返回非空 kind 的稳定计数。"""
        return tuple(
            (kind, len(self.records(kind)))
            for kind in W03_ARTIFACT_KINDS
            if self.records(kind)
        )

    def payloads(self, artifact_kind: str) -> tuple[dict[str, Any], ...]:
        return tuple(item.payload for item in self.records(artifact_kind))

    @staticmethod
    def _decode(row: dict[str, Any]) -> W03ArtifactRecord:
        fields = {
            "artifact_kind", "ordinal", "size_bytes",
            "payload_sha256", "payload_json",
        }
        if not isinstance(row, dict) or set(row) != fields:
            raise W03ArtifactError("W-03 artifact row 字段漂移")
        if (row["artifact_kind"] not in W03_ARTIFACT_KINDS
                or type(row["ordinal"]) is not int
                or row["ordinal"] <= 0
                or type(row["size_bytes"]) is not int
                or row["size_bytes"] <= 0
                or not isinstance(row["payload_sha256"], str)
                or not isinstance(row["payload_json"], str)):
            raise W03ArtifactError("W-03 artifact row 类型非法")
        encoded = row["payload_json"].encode("utf-8")
        if (len(encoded) != row["size_bytes"]
                or hashlib.sha256(encoded).hexdigest()
                != row["payload_sha256"]):
            raise W03ArtifactError("W-03 artifact size/SHA 漂移")
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError as exc:
            raise W03ArtifactError("W-03 artifact JSON 损坏") from exc
        if (not isinstance(payload, dict)
                or canonical_json_bytes(payload) != encoded):
            raise W03ArtifactError("W-03 artifact JSON 非 canonical object")
        return W03ArtifactRecord(
            row["artifact_kind"],
            row["ordinal"],
            row["payload_sha256"],
            payload,
        )


def persist_training_payload(
        store: W03ArtifactStore,
        payload: W03TrainingPayload,
        ) -> None:
    """按 owner/stable key 保存全部 163 条 train envelope。"""
    records = (
        *sorted(payload.source_refs, key=lambda item: item.stable_key),
        *sorted(payload.observations, key=lambda item: item.stable_key),
        *sorted(payload.teacher_evidence, key=lambda item: item.stable_key),
    )
    if len(records) != 163:
        raise W03ArtifactError("W-03 train envelope count 必须为 163")
    for ordinal, record in enumerate(records, start=1):
        store.put(ARTIFACT_TRAIN_ENVELOPE, ordinal, record.to_dict())


def restore_training_payload(store: W03ArtifactStore) -> W03TrainingPayload:
    """从 envelope 行恢复 typed train payload，并拒绝 evaluator 类型。"""
    values = []
    for payload in store.payloads(ARTIFACT_TRAIN_ENVELOPE):
        try:
            values.append(record_from_dict(payload))
        except (TypeError, ValueError, KeyError) as exc:
            raise W03ArtifactError("W-03 train envelope 无法恢复") from exc
    if len(values) != 163:
        raise W03ArtifactError("W-03 train envelope 不完整")
    if any(not isinstance(item, (
            SourceRefRecord,
            ObservationRecord,
            TeacherEvidenceRecord,
            )) for item in values):
        raise W03ArtifactError("W-03 train envelope 混入非 train owner")
    return W03TrainingPayload(
        tuple(item for item in values if isinstance(item, SourceRefRecord)),
        tuple(item for item in values if isinstance(item, ObservationRecord)),
        tuple(item for item in values if isinstance(item, TeacherEvidenceRecord)),
    )


__all__ = [
    "ARTIFACT_EVIDENCE_ACCOUNT",
    "ARTIFACT_GENERATION_CHOICE",
    "ARTIFACT_GENERATION_DECISION",
    "ARTIFACT_GENERATION_OUTCOME",
    "ARTIFACT_GENERATION_USE",
    "ARTIFACT_PROJECTION",
    "ARTIFACT_TRAIN_ENVELOPE",
    "ARTIFACT_W02_RETENTION",
    "W03ArtifactError",
    "W03ArtifactRecord",
    "W03ArtifactStore",
    "persist_training_payload",
    "restore_training_payload",
]
