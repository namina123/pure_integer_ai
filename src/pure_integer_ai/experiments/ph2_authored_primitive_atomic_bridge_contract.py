"""显式 W-04 到 W-05 CC0 bridge 的严格映射与双 seed 合同。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_authored_atomic_course import (
    AuthoredAtomicCourseError,
    AuthoredAtomicSeed,
    read_authored_atomic_seeds,
)
from pure_integer_ai.experiments.ph2_authored_primitive_course import (
    AuthoredPrimitiveCourseError,
    PrimitiveSurfaceSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
    DatasetContractError,
    parse_canonical_json_bytes,
)


PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY = (
    "AUTHORED_CC0_PRIMITIVE_ATOMIC_BRIDGE_V1")
PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID = "CC0-1.0"
PRIMITIVE_ATOMIC_BRIDGE_PACK_NAME = (
    "AUTHORED_CC0_PRIMITIVE_ATOMIC_BRIDGE_V1--CC0-1.0--bridge-v1")
PRIMITIVE_ATOMIC_BRIDGE_STAGES = ("W-04", "W-05")
PRIMITIVE_ATOMIC_BRIDGE_SUBSTAGES = (
    "PRIMITIVE_SURFACE_MAPPING",
    "OCCURRENCE_ROLE_ATOMIC_PROPOSITION",
)
_REQUIRED_PRIMITIVE_ROLES = frozenset({
    "support", "refute", "conflict", "supersede"})
_REQUIRED_PRIMITIVE_TRAIN_PERTURBATIONS = frozenset({
    "NONE",
    "PRIMITIVE_MISMATCH",
    "SAME_SURFACE_AMBIGUITY",
    "CUE_REPLACEMENT",
})
_MAP_FIELDS = frozenset({
    "atomic_seed_id",
    "expected_payload",
    "expected_state",
    "family",
    "label_owner",
    "license_id",
    "logical_order",
    "perturbation_kind",
    "primitive_kind",
    "primitive_registry",
    "sample_role",
    "split",
    "supersedes_atomic_seed_id",
    "surface_form",
    "template_family",
})


# object-model: exception
class AuthoredPrimitiveAtomicBridgeError(RuntimeError):
    """W-04/W-05 bridge 行、owner、来源或前置依赖非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if (not isinstance(value, str) or value.strip() != value
            or (not allow_empty and not value)):
        raise AuthoredPrimitiveAtomicBridgeError(
            f"{where} 必须是规范文本")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise AuthoredPrimitiveAtomicBridgeError(
            f"{where} 必须是正严格整数")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PrimitiveAtomicBridgeMapSeed:
    """一条把 W-04 primitive 侧映射到既有 W-05 atomic seed 的记录。"""

    atomic_seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    surface_form: str
    primitive_registry: str
    primitive_kind: int
    expected_state: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_atomic_seed_id: str
    logical_order: int

    def __post_init__(self) -> None:
        for name in (
                "atomic_seed_id", "family", "template_family",
                "surface_form", "primitive_registry", "perturbation_kind"):
            _text(getattr(self, name), where=f"primitive map {name}")
        _text(
            self.supersedes_atomic_seed_id,
            where="primitive map supersedes_atomic_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map label_owner 必须是 teacher/evaluator")
        expected_split = (
            "train" if self.label_owner == "teacher" else "held_out")
        if self.split != expected_split:
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map owner/split 漂移")
        if self.sample_role not in _REQUIRED_PRIMITIVE_ROLES:
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map sample_role 未注册")
        if ((self.sample_role == "supersede")
                != bool(self.supersedes_atomic_seed_id)):
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map supersede role/link 漂移")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map expected_payload 类型非法")
        _positive_int(self.primitive_kind, where="primitive map primitive_kind")
        _positive_int(self.logical_order, where="primitive map logical_order")

    @classmethod
    def from_dict(
            cls,
            value: dict[str, Any],
            ) -> "PrimitiveAtomicBridgeMapSeed":
        if set(value) != _MAP_FIELDS:
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map 字段集合漂移")
        if value["license_id"] != PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID:
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map 必须是 CC0-1.0")
        return cls(
            str(value["atomic_seed_id"]),
            str(value["family"]),
            str(value["template_family"]),
            str(value["label_owner"]),
            str(value["split"]),
            str(value["sample_role"]),
            str(value["surface_form"]),
            str(value["primitive_registry"]),
            value["primitive_kind"],
            str(value["expected_state"]),
            CanonicalJsonObject.from_value(value["expected_payload"]),
            str(value["perturbation_kind"]),
            str(value["supersedes_atomic_seed_id"]),
            value["logical_order"],
        )

    def bind_atomic(
            self,
            atomic: AuthoredAtomicSeed,
            ) -> PrimitiveSurfaceSeed:
        """以 atomic 完整 surface 作为 W-04 context，形成现役 primitive seed。"""
        if not isinstance(atomic, AuthoredAtomicSeed):
            raise TypeError("atomic 必须是 AuthoredAtomicSeed")
        if (atomic.seed_id != self.atomic_seed_id
                or atomic.logical_order != self.logical_order):
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map 与 atomic identity/order 漂移")
        occurrence = next(
            item for item in atomic.occurrences
            if item.occurrence_id == atomic.predicate_occurrence_id)
        if self.surface_form != occurrence.surface_fragment:
            raise AuthoredPrimitiveAtomicBridgeError(
                "W-04 primitive surface 未绑定 W-05 predicate occurrence")
        try:
            return PrimitiveSurfaceSeed(
                self.atomic_seed_id,
                self.family,
                self.template_family,
                self.label_owner,
                self.split,
                self.sample_role,
                self.surface_form,
                atomic.surface,
                self.primitive_registry,
                self.primitive_kind,
                self.expected_state,
                self.expected_payload,
                self.perturbation_kind,
                self.supersedes_atomic_seed_id,
                self.logical_order,
            )
        except AuthoredPrimitiveCourseError as exc:
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive map 无法形成冻结 W-04 seed") from exc


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PrimitiveAtomicBridgeSeed:
    """共享一个公开 SourceRef 的 W-04 primitive 与 W-05 atomic seed。"""

    primitive: PrimitiveSurfaceSeed
    atomic: AuthoredAtomicSeed

    def __post_init__(self) -> None:
        if (not isinstance(self.primitive, PrimitiveSurfaceSeed)
                or not isinstance(self.atomic, AuthoredAtomicSeed)
                or self.primitive.seed_id != self.atomic.seed_id
                or self.primitive.logical_order != self.atomic.logical_order
                or self.primitive.context != self.atomic.surface):
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive/atomic bridge seed identity 漂移")


def _read_map_seeds(
        path: str | Path,
        ) -> tuple[PrimitiveAtomicBridgeMapSeed, ...]:
    sample = Path(path)
    try:
        payload = sample.read_bytes()
    except OSError as exc:
        raise AuthoredPrimitiveAtomicBridgeError(
            "primitive map sample 无法读取") from exc
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredPrimitiveAtomicBridgeError(
            "primitive map sample 必须非空并以换行结束")
    values = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredPrimitiveAtomicBridgeError(
                f"primitive map 第 {line_number} 行为空或缺换行")
        try:
            decoded = parse_canonical_json_bytes(
                line[:-1], require_object=True)
        except DatasetContractError as exc:
            raise AuthoredPrimitiveAtomicBridgeError(
                f"primitive map 第 {line_number} 行不是规范 JSON") from exc
        assert isinstance(decoded, dict)
        values.append(PrimitiveAtomicBridgeMapSeed.from_dict(decoded))
    if len({item.atomic_seed_id for item in values}) != len(values):
        raise AuthoredPrimitiveAtomicBridgeError(
            "primitive map atomic_seed_id 重复")
    orders = [item.logical_order for item in values]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredPrimitiveAtomicBridgeError(
            "primitive map logical_order 必须严格递增")
    return tuple(values)


def read_authored_primitive_atomic_bridge_seeds(
        map_path: str | Path,
        atomic_path: str | Path,
        ) -> tuple[PrimitiveAtomicBridgeSeed, ...]:
    """读取两份 CC0 sample，并闭合覆盖、owner、supersede 与 predicate link。"""
    try:
        atomic = read_authored_atomic_seeds(atomic_path)
    except AuthoredAtomicCourseError as exc:
        raise AuthoredPrimitiveAtomicBridgeError(
            "bridge atomic sample 非法") from exc
    mapping = _read_map_seeds(map_path)
    atomic_by_id = {item.seed_id: item for item in atomic}
    if {item.atomic_seed_id for item in mapping} != set(atomic_by_id):
        raise AuthoredPrimitiveAtomicBridgeError(
            "primitive map 未精确覆盖 atomic sample")
    bound = tuple(
        PrimitiveAtomicBridgeSeed(
            item.bind_atomic(atomic_by_id[item.atomic_seed_id]),
            atomic_by_id[item.atomic_seed_id],
        )
        for item in mapping
    )
    primitive = tuple(item.primitive for item in bound)
    index = {item.seed_id: item for item in primitive}
    for item in primitive:
        if not item.supersedes_seed_id:
            continue
        target = index.get(item.supersedes_seed_id)
        if (target is None or target.logical_order >= item.logical_order
                or target.family != item.family
                or target.split != item.split):
            raise AuthoredPrimitiveAtomicBridgeError(
                "primitive supersede target 缺失、更晚或跨 owner")
    teacher_families = {
        item.family for item in primitive if item.label_owner == "teacher"}
    evaluator_families = {
        item.family for item in primitive if item.label_owner == "evaluator"}
    teacher_templates = {
        item.template_family for item in primitive
        if item.label_owner == "teacher"}
    evaluator_templates = {
        item.template_family for item in primitive
        if item.label_owner == "evaluator"}
    atomic_families = {item.atomic.family for item in bound}
    atomic_templates = {item.atomic.template_family for item in bound}
    primitive_teacher = tuple(
        item for item in primitive if item.label_owner == "teacher")
    atomic_teacher = tuple(
        item.atomic for item in bound if item.atomic.label_owner == "teacher")
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates
            or (teacher_families | evaluator_families) & atomic_families
            or (teacher_templates | evaluator_templates) & atomic_templates
            or {item.sample_role for item in primitive}
            != _REQUIRED_PRIMITIVE_ROLES
            or not _REQUIRED_PRIMITIVE_TRAIN_PERTURBATIONS.issubset({
                item.perturbation_kind for item in primitive_teacher})
            or len(primitive_teacher) < 6
            or len(atomic_teacher) < 6
            or len(primitive_teacher) != len(atomic_teacher)):
        raise AuthoredPrimitiveAtomicBridgeError(
            "bridge owner/family/role/train coverage 未闭合")
    return bound


__all__ = [
    "PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID",
    "PRIMITIVE_ATOMIC_BRIDGE_PACK_NAME",
    "PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY",
    "PRIMITIVE_ATOMIC_BRIDGE_STAGES",
    "PRIMITIVE_ATOMIC_BRIDGE_SUBSTAGES",
    "AuthoredPrimitiveAtomicBridgeError",
    "PrimitiveAtomicBridgeMapSeed",
    "PrimitiveAtomicBridgeSeed",
    "read_authored_primitive_atomic_bridge_seeds",
]
