"""W-05 原创 occurrence/角色/原子命题 seed 的纯合同与校验。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_EVENT,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--atomic-v1"
STAGE = "W-05"
SUBSTAGE = "OCCURRENCE_ROLE_ATOMIC_PROPOSITION"

PREDICATE_REGISTRY = "AUTHORED_CONCEPT_V1"
ROLE_REGISTRY = "AUTHORED_ROLE_V1"
ROLE_ACTOR = 1
ROLE_PATIENT = 2
ROLE_RECIPIENT = 3
ROLE_LOCATION = 4
ALLOWED_ROLE_KINDS = frozenset({
    ROLE_ACTOR,
    ROLE_PATIENT,
    ROLE_RECIPIENT,
    ROLE_LOCATION,
})
REQUIRED_SAMPLE_ROLES = frozenset({
    "support", "refute", "conflict", "supersede"})
ALLOWED_PERTURBATIONS = frozenset({
    "NONE",
    "ROLE_SWAP",
    "ORDER_REVERSAL",
    "SCOPE_SHIFT",
    "OCCURRENCE_OMISSION",
    "OCCURRENCE_RESTORE",
})
REQUIRED_PERTURBATIONS = frozenset({
    "ROLE_SWAP",
    "ORDER_REVERSAL",
    "SCOPE_SHIFT",
    "OCCURRENCE_OMISSION",
})

_SEED_FIELDS = frozenset({
    "bindings",
    "context_local_id",
    "expected_payload",
    "expected_state",
    "family",
    "label_owner",
    "license_id",
    "logical_order",
    "occurrence_order",
    "occurrences",
    "perturbation_kind",
    "predicate_kind",
    "predicate_occurrence_id",
    "predicate_registry",
    "sample_role",
    "seed_id",
    "split",
    "supersedes_seed_id",
    "surface",
    "template_family",
})
_OCCURRENCE_FIELDS = frozenset({
    "end",
    "occurrence_id",
    "ordinal",
    "semantic_kind",
    "semantic_local_id",
    "start",
    "surface_fragment",
})
_BINDING_FIELDS = frozenset({
    "filler_occurrence_id",
    "ordinal",
    "role_kind",
    "role_registry",
})


class AuthoredAtomicCourseError(RuntimeError):
    """原创 W-05 seed 的 occurrence、角色、scope 或命题合同非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求 seed 文本为无首尾空白字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredAtomicCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredAtomicCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求课程坐标为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredAtomicCourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 span、ordinal 等位置为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredAtomicCourseError(f"{where} 必须是非负严格整数")
    return value


@dataclass(frozen=True)
class AtomicOccurrenceSeed:
    """一个来源 span、occurrence 身份和语义对象候选的 typed seed。"""

    occurrence_id: str
    surface_fragment: str
    start: int
    end: int
    ordinal: int
    semantic_kind: int
    semantic_local_id: int

    def __post_init__(self) -> None:
        _text(self.occurrence_id, where="AtomicOccurrenceSeed.occurrence_id")
        _text(self.surface_fragment, where="AtomicOccurrenceSeed.surface_fragment")
        _nonnegative_int(self.start, where="AtomicOccurrenceSeed.start")
        _nonnegative_int(self.end, where="AtomicOccurrenceSeed.end")
        _nonnegative_int(self.ordinal, where="AtomicOccurrenceSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredAtomicCourseError("atomic occurrence span 必须有正宽度")
        if self.semantic_kind not in {OBJECT_ENTITY, OBJECT_EVENT}:
            raise AuthoredAtomicCourseError("atomic semantic_kind 必须是 Entity/Event")
        _positive_int(
            self.semantic_local_id,
            where="AtomicOccurrenceSeed.semantic_local_id",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "AtomicOccurrenceSeed":
        """从严格字段集合构造 occurrence seed。"""
        if not isinstance(value, dict) or set(value) != _OCCURRENCE_FIELDS:
            raise AuthoredAtomicCourseError("atomic occurrence 字段集合漂移")
        return cls(
            _text(value["occurrence_id"], where="occurrence_id"),
            _text(value["surface_fragment"], where="surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
            value["semantic_kind"],
            value["semantic_local_id"],
        )


@dataclass(frozen=True)
class AtomicBindingSeed:
    """一个课程注入 Role 到已声明 occurrence 语义候选的绑定。"""

    role_registry: str
    role_kind: int
    filler_occurrence_id: str
    ordinal: int

    def __post_init__(self) -> None:
        if self.role_registry != ROLE_REGISTRY:
            raise AuthoredAtomicCourseError("atomic role registry 非冻结课程坐标")
        if self.role_kind not in ALLOWED_ROLE_KINDS:
            raise AuthoredAtomicCourseError("atomic role kind 非冻结课程坐标")
        _text(
            self.filler_occurrence_id,
            where="AtomicBindingSeed.filler_occurrence_id",
        )
        _nonnegative_int(self.ordinal, where="AtomicBindingSeed.ordinal")

    @classmethod
    def from_dict(cls, value: Any) -> "AtomicBindingSeed":
        """从严格字段集合构造 RoleBinding seed。"""
        if not isinstance(value, dict) or set(value) != _BINDING_FIELDS:
            raise AuthoredAtomicCourseError("atomic binding 字段集合漂移")
        return cls(
            _text(value["role_registry"], where="role_registry"),
            value["role_kind"],
            _text(
                value["filler_occurrence_id"],
                where="filler_occurrence_id",
            ),
            value["ordinal"],
        )


@dataclass(frozen=True)
class AuthoredAtomicSeed:
    """一个可编译为现役一等语义对象键的原子命题候选。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    surface: str
    occurrences: tuple[AtomicOccurrenceSeed, ...]
    occurrence_order: tuple[str, ...]
    predicate_registry: str
    predicate_kind: int
    predicate_occurrence_id: str
    context_local_id: int
    bindings: tuple[AtomicBindingSeed, ...]
    expected_state: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id),
                ("family", self.family),
                ("template_family", self.template_family),
                ("surface", self.surface),
                ("predicate_occurrence_id", self.predicate_occurrence_id),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredAtomicSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredAtomicSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredAtomicCourseError("label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredAtomicCourseError("label_owner 与 split 不一致")
        if self.sample_role not in REQUIRED_SAMPLE_ROLES:
            raise AuthoredAtomicCourseError("sample_role 不属于 atomic 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredAtomicCourseError("supersede seed 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredAtomicCourseError("非 supersede seed 不得声明替代目标")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredAtomicCourseError("expected_state 非四态")
        if self.perturbation_kind not in ALLOWED_PERTURBATIONS:
            raise AuthoredAtomicCourseError("atomic perturbation 未注册")
        if self.predicate_registry != PREDICATE_REGISTRY:
            raise AuthoredAtomicCourseError("atomic predicate registry 非冻结课程坐标")
        _positive_int(self.predicate_kind, where="AuthoredAtomicSeed.predicate_kind")
        _positive_int(
            self.context_local_id,
            where="AuthoredAtomicSeed.context_local_id",
        )
        _positive_int(self.logical_order, where="AuthoredAtomicSeed.logical_order")
        if not self.occurrences or not self.bindings:
            raise AuthoredAtomicCourseError("atomic occurrences/bindings 不能为空")

        occurrence_ids = [item.occurrence_id for item in self.occurrences]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise AuthoredAtomicCourseError("atomic occurrence_id 重复")
        spans = [(item.start, item.end, item.ordinal) for item in self.occurrences]
        if spans != sorted(spans) or len(set(spans)) != len(spans):
            raise AuthoredAtomicCourseError("atomic occurrence span 必须唯一且来源序递增")
        previous_end = -1
        for occurrence in self.occurrences:
            if occurrence.end > len(self.surface):
                raise AuthoredAtomicCourseError("atomic occurrence span 超出 surface")
            if self.surface[occurrence.start:occurrence.end] != (
                    occurrence.surface_fragment):
                raise AuthoredAtomicCourseError("atomic occurrence span 与 surface 不一致")
            if occurrence.start < previous_end:
                raise AuthoredAtomicCourseError("atomic occurrence span 不得重叠")
            previous_end = occurrence.end
        if (len(self.occurrence_order) != len(occurrence_ids)
                or set(self.occurrence_order) != set(occurrence_ids)):
            raise AuthoredAtomicCourseError("occurrence_order 必须恰好覆盖全部 occurrence")
        occurrence_index = {item.occurrence_id: item for item in self.occurrences}
        predicate_occurrence = occurrence_index.get(self.predicate_occurrence_id)
        if (predicate_occurrence is None
                or predicate_occurrence.semantic_kind != OBJECT_EVENT):
            raise AuthoredAtomicCourseError("predicate anchor 必须引用 Event occurrence")
        slots = [(item.role_kind, item.ordinal) for item in self.bindings]
        if len(set(slots)) != len(slots):
            raise AuthoredAtomicCourseError("atomic Role/ordinal slot 重复")
        if any(item.filler_occurrence_id not in occurrence_index
               for item in self.bindings):
            raise AuthoredAtomicCourseError("atomic binding 引用未知 occurrence")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredAtomicSeed":
        """从严格字段集合恢复一条 atomic seed。"""
        if set(value) != _SEED_FIELDS:
            raise AuthoredAtomicCourseError("atomic seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredAtomicCourseError("atomic seed 必须是 CC0-1.0")
        occurrences_value = value["occurrences"]
        bindings_value = value["bindings"]
        order_value = value["occurrence_order"]
        if not isinstance(occurrences_value, list):
            raise AuthoredAtomicCourseError("occurrences 必须是列表")
        if not isinstance(bindings_value, list):
            raise AuthoredAtomicCourseError("bindings 必须是列表")
        if (not isinstance(order_value, list)
                or any(not isinstance(item, str) for item in order_value)):
            raise AuthoredAtomicCourseError("occurrence_order 必须是字符串列表")
        return cls(
            _text(value["seed_id"], where="seed_id"),
            _text(value["family"], where="family"),
            _text(value["template_family"], where="template_family"),
            _text(value["label_owner"], where="label_owner"),
            _text(value["split"], where="split"),
            _text(value["sample_role"], where="sample_role"),
            _text(value["surface"], where="surface"),
            tuple(AtomicOccurrenceSeed.from_dict(item)
                  for item in occurrences_value),
            tuple(order_value),
            _text(value["predicate_registry"], where="predicate_registry"),
            value["predicate_kind"],
            _text(
                value["predicate_occurrence_id"],
                where="predicate_occurrence_id",
            ),
            value["context_local_id"],
            tuple(AtomicBindingSeed.from_dict(item) for item in bindings_value),
            _text(value["expected_state"], where="expected_state"),
            CanonicalJsonObject.from_value(value["expected_payload"]),
            _text(value["perturbation_kind"], where="perturbation_kind"),
            _text(
                value["supersedes_seed_id"],
                where="supersedes_seed_id",
                allow_empty=True,
            ),
            value["logical_order"],
        )


__all__ = [
    "ALLOWED_PERTURBATIONS",
    "ALLOWED_ROLE_KINDS",
    "AuthoredAtomicCourseError",
    "AuthoredAtomicSeed",
    "AtomicBindingSeed",
    "AtomicOccurrenceSeed",
    "LICENSE_ID",
    "PACK_NAME",
    "PREDICATE_REGISTRY",
    "REQUIRED_PERTURBATIONS",
    "REQUIRED_SAMPLE_ROLES",
    "ROLE_ACTOR",
    "ROLE_LOCATION",
    "ROLE_PATIENT",
    "ROLE_RECIPIENT",
    "ROLE_REGISTRY",
    "SOURCE_KEY",
    "STAGE",
    "SUBSTAGE",
]
