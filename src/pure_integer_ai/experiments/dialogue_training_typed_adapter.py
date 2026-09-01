"""公开对话 pack 的 typed 课程适配器。

该模块只消费数据源已经声明的 typed payload。普通文本、HTML、Markdown、
代码和表格不会因为出现某些词面而被猜成语义课程。适配结果是稳定整数键，
可作为后续 semantic/generation owner 的唯一入口。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.experiments.conversation_training_pack import (
    DialogueTrainingCase,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


class TypedDialogueAdapterError(ValueError):
    """typed 课程附件不完整或与声明的 kind 不一致。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


@dataclass(frozen=True, slots=True)
class TypedDialogueCourseRequest:
    """一条可交给后续 owner 的 typed 请求，不包含猜测出的答案。"""

    case_id: str
    payload_kind: str
    payload: CanonicalJsonObject
    source_ref: SourceRef | None
    split: str

    def __post_init__(self) -> None:
        if not self.case_id or not isinstance(self.case_id, str):
            raise TypedDialogueAdapterError("typed case_id 非法")
        if not self.payload_kind or self.payload_kind.strip() != self.payload_kind:
            raise TypedDialogueAdapterError("typed payload_kind 非法")
        if not isinstance(self.payload, CanonicalJsonObject):
            raise TypedDialogueAdapterError("typed payload 类型非法")
        if self.source_ref is not None and not isinstance(self.source_ref, SourceRef):
            raise TypedDialogueAdapterError("typed source_ref 类型非法")
        if self.split not in {"train", "heldout"}:
            raise TypedDialogueAdapterError("typed request 不得来自 negative split")

    def stable_key(self) -> tuple[int, ...]:
        encoded = canonical_json_bytes(self.payload.to_value())
        return (
            2,
            *_packed(tuple(ord(item) for item in self.case_id)),
            *_packed(tuple(ord(item) for item in self.payload_kind)),
            *_packed(tuple(encoded)),
            *_packed(() if self.source_ref is None else self.source_ref.stable_key()),
            *_packed(tuple(ord(item) for item in self.split)),
        )


@dataclass(frozen=True, slots=True)
class TypedDialogueAdapterReport:
    """训练入口可审计的 typed 覆盖计数。"""

    total_items: int
    typed_items: int
    by_kind: tuple[tuple[str, int], ...]
    request_keys: tuple[tuple[int, ...], ...]

    @property
    def complete(self) -> bool:
        return self.total_items >= self.typed_items >= 0

    def to_dict(self) -> dict[str, Any]:
        key_bytes = json.dumps(
            self.request_keys,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        return {
            "total_items": self.total_items,
            "typed_items": self.typed_items,
            "by_kind": [list(item) for item in self.by_kind],
            "request_count": len(self.request_keys),
            "request_keys_sha256": hashlib.sha256(key_bytes).hexdigest(),
        }


class TypedDialogueCourseAdapter:
    """把 pack 的显式 typed 附件转为稳定 request；无附件即明确无 request。"""

    def adapt(self, case: DialogueTrainingCase) -> TypedDialogueCourseRequest | None:
        if not isinstance(case, DialogueTrainingCase):
            raise TypeError("typed adapter 需要 DialogueTrainingCase")
        if case.typed_payload is None:
            return None
        if case.split == "negative":
            # negative typed attachments remain auditable data, but cannot be
            # submitted to the training owner as a positive request.
            return None
        if case.payload_kind == "TypedRelationQuery":
            # Authored W-06 relation observations are consumed by the shared
            # relation bridge during formal training.  They are not dialogue
            # generation requests and must not be misclassified by this
            # generation-only adapter.
            return None
        if case.payload_kind is None:
            raise TypedDialogueAdapterError("typed payload 缺少 payload_kind")
        value = case.typed_payload.to_value()
        if not isinstance(value, dict):
            raise TypedDialogueAdapterError("typed payload 必须是 object")
        if case.payload_kind == "GenerationAdoptionPostcheckQuery":
            if not isinstance(value.get("task_kind"), str):
                raise TypedDialogueAdapterError("generation typed payload 缺少 task_kind")
            if not isinstance(value.get("adoption_request"), dict):
                raise TypedDialogueAdapterError("generation typed payload 缺少 adoption_request")
        elif case.payload_kind == "GenerationGeneralizationCandidateV1":
            if not isinstance(value.get("candidate_case"), str):
                raise TypedDialogueAdapterError(
                    "generation generalization typed payload 缺少 candidate_case")
            if not isinstance(value.get("surface_candidates"), list):
                raise TypedDialogueAdapterError(
                    "generation generalization typed payload 缺少 surface_candidates")
        else:
            raise TypedDialogueAdapterError(
                f"未注册 typed payload kind: {case.payload_kind}")
        return TypedDialogueCourseRequest(
            case.case_id,
            case.payload_kind,
            case.typed_payload,
            case.source_ref,
            case.split,
        )

    def adapt_cases(
            self, cases: Iterable[DialogueTrainingCase],
            ) -> tuple[TypedDialogueCourseRequest, ...]:
        requests = [request for case in cases
                    if (request := self.adapt(case)) is not None]
        return tuple(sorted(requests, key=lambda item: item.case_id))

    def report(self, cases: Iterable[DialogueTrainingCase]) -> TypedDialogueAdapterReport:
        values = tuple(cases)
        requests = self.adapt_cases(values)
        counts: dict[str, int] = {}
        typed_values = [case.payload_kind for case in values
                        if case.typed_payload is not None]
        for kind in typed_values:
            assert kind is not None
            counts[kind] = counts.get(kind, 0) + 1
        return TypedDialogueAdapterReport(
            len(values),
            len(typed_values),
            tuple(sorted(counts.items())),
            tuple(request.stable_key() for request in requests),
        )


__all__ = [
    "TypedDialogueAdapterError",
    "TypedDialogueAdapterReport",
    "TypedDialogueCourseAdapter",
    "TypedDialogueCourseRequest",
]
